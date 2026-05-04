import os
import io
import base64
import uuid
import re
from pathlib import Path
from typing import Optional

import pdfplumber
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── 환경변수 ──────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── ChromaDB 초기화 ───────────────────────────────────────────
CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="workbooks",
    metadata={"hnsw:space": "cosine"},
)

# ── FastAPI 앱 ────────────────────────────────────────────────
app = FastAPI(title="문제집 출처 탐색 RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 유틸 함수 ────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    """PDF에서 페이지별 텍스트 추출"""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """텍스트를 청크로 분할"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
    """Gemini embedding API 호출"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def get_query_embedding(text: str) -> list[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]


def extract_text_from_image_base64(image_base64: str, mime_type: str = "image/jpeg") -> str:
    """Gemini Vision으로 시험지 이미지에서 문제 텍스트 추출"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    image_data = {"mime_type": mime_type, "data": image_base64}
    prompt = (
        "이 시험지 이미지에서 문제 텍스트를 최대한 정확하게 추출해줘. "
        "문제 번호, 선택지, 지문을 모두 포함해서 텍스트만 출력해. "
        "설명이나 부가 설명 없이 문제 내용만 출력해."
    )
    response = model.generate_content([prompt, image_data])
    return response.text.strip()


# ── API 엔드포인트 ────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "문제집 RAG 서버 동작 중"}


@app.get("/health")
def health():
    count = collection.count()
    return {"status": "ok", "indexed_chunks": count}


@app.post("/upload-workbook")
async def upload_workbook(
    file: UploadFile = File(...),
    workbook_name: str = Form(...),
    subject: Optional[str] = Form(None),
):
    """
    문제집 PDF 업로드 → 텍스트 추출 → ChromaDB 저장
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    file_bytes = await file.read()
    pages = extract_text_from_pdf(file_bytes)

    if not pages:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

    added_chunks = 0
    documents, embeddings, metadatas, ids = [], [], [], []

    for page_info in pages:
        page_num = page_info["page"]
        text = page_info["text"]
        chunks = chunk_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:
                continue
            try:
                emb = get_embedding(chunk)
            except Exception as e:
                continue

            chunk_id = f"{workbook_name}__p{page_num}__c{chunk_idx}__{uuid.uuid4().hex[:6]}"
            documents.append(chunk)
            embeddings.append(emb)
            metadatas.append({
                "workbook_name": workbook_name,
                "subject": subject or "",
                "page": page_num,
                "chunk_index": chunk_idx,
            })
            ids.append(chunk_id)
            added_chunks += 1

    if documents:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    return {
        "success": True,
        "workbook_name": workbook_name,
        "pages_processed": len(pages),
        "chunks_indexed": added_chunks,
    }


class SearchByTextRequest(BaseModel):
    query_text: str
    top_k: int = 5


@app.post("/search-by-text")
async def search_by_text(req: SearchByTextRequest):
    """
    텍스트로 ChromaDB 검색 → 출처 반환
    """
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="인덱싱된 문제집이 없습니다.")

    query_emb = get_query_embedding(req.query_text)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(req.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)
        hits.append({
            "workbook_name": meta.get("workbook_name"),
            "subject": meta.get("subject"),
            "page": meta.get("page"),
            "similarity": similarity,
            "matched_text": doc[:200] + ("..." if len(doc) > 200 else ""),
        })

    return {"query": req.query_text, "results": hits}


@app.post("/search-by-image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    시험지 사진 업로드 → Gemini OCR → ChromaDB 검색 → 출처 반환
    """
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="인덱싱된 문제집이 없습니다.")

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    content_type = file.content_type or "image/jpeg"
    if content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="이미지 파일(jpg/png/webp)만 업로드 가능합니다.")

    file_bytes = await file.read()
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    # Gemini로 텍스트 추출
    try:
        extracted_text = extract_text_from_image_base64(image_base64, mime_type=content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 OCR 실패: {str(e)}")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="이미지에서 텍스트를 추출할 수 없습니다.")

    # ChromaDB 검색
    query_emb = get_query_embedding(extracted_text)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)
        hits.append({
            "workbook_name": meta.get("workbook_name"),
            "subject": meta.get("subject"),
            "page": meta.get("page"),
            "similarity": similarity,
            "matched_text": doc[:200] + ("..." if len(doc) > 200 else ""),
        })

    return {
        "extracted_text": extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""),
        "results": hits,
    }


@app.get("/workbooks")
def list_workbooks():
    """등록된 문제집 목록 조회"""
    if collection.count() == 0:
        return {"workbooks": []}

    results = collection.get(include=["metadatas"])
    seen = {}
    for meta in results["metadatas"]:
        name = meta.get("workbook_name", "")
        if name not in seen:
            seen[name] = {
                "workbook_name": name,
                "subject": meta.get("subject", ""),
                "chunk_count": 0,
            }
        seen[name]["chunk_count"] += 1

    return {"workbooks": list(seen.values())}


@app.delete("/workbook/{workbook_name}")
def delete_workbook(workbook_name: str):
    """문제집 삭제"""
    results = collection.get(
        where={"workbook_name": workbook_name},
        include=["metadatas"],
    )
    ids = results.get("ids", [])
    if not ids:
        raise HTTPException(status_code=404, detail="해당 문제집을 찾을 수 없습니다.")

    collection.delete(ids=ids)
    return {"success": True, "deleted_chunks": len(ids), "workbook_name": workbook_name}

import os
import io
import base64
import uuid
import logging
from typing import Optional

import pdfplumber
import chromadb
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="workbooks",
    metadata={"hnsw:space": "cosine"},
)

app = FastAPI(title="문제집 출처 탐색 RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    pages = []

    # 1단계: pdfplumber로 텍스트 추출 시도
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    words = page.extract_words()
                    text = " ".join([w["text"] for w in words])
                text = text.strip()
                if text:
                    pages.append({"page": i, "text": text})
        logger.info(f"pdfplumber 추출 결과: {len(pages)}페이지")
    except Exception as e:
        logger.error(f"pdfplumber 오류: {e}")

    # 2단계: 텍스트 없으면 Gemini Vision OCR
    if not pages:
        logger.info("텍스트 없음 → Gemini OCR 시도")
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            total = len(doc)
            logger.info(f"PDF 페이지 수: {total}")
            model = genai.GenerativeModel("gemini-1.5-flash")

            for i in range(min(total, 20)):  # 최대 20페이지
                page = doc[i]
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("jpeg")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                prompt = "이 교재 페이지에서 문제, 선택지, 지문 텍스트를 전부 추출해줘. 텍스트만 출력해."
                try:
                    response = model.generate_content([
                        prompt,
                       {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
                    ])
                    text = response.text.strip()
                    logger.info(f"페이지 {i+1} OCR 완료: {len(text)}자")
                    if text:
                        pages.append({"page": i + 1, "text": text})
                except Exception as e:
                    logger.error(f"페이지 {i+1} OCR 오류: {e}")
                    continue
            doc.close()
        except Exception as e:
            logger.error(f"fitz/OCR 전체 오류: {e}")

    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
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
    model = genai.GenerativeModel("gemini-1.5-flash")
    image_data = {"mime_type": mime_type, "data": image_base64}
    prompt = (
        "이 시험지 이미지에서 문제 텍스트를 최대한 정확하게 추출해줘. "
        "문제 번호, 선택지, 지문을 모두 포함해서 텍스트만 출력해. "
        "설명이나 부가 설명 없이 문제 내용만 출력해."
    )
    response = model.generate_content([prompt, image_data])
    return response.text.strip()


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
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    logger.info(f"업로드 시작: {file.filename}, 문제집명: {workbook_name}")
    file_bytes = await file.read()
    logger.info(f"파일 크기: {len(file_bytes)} bytes")

    pages = extract_text_from_pdf(file_bytes)
    logger.info(f"추출된 페이지 수: {len(pages)}")

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
                logger.error(f"임베딩 오류: {e}")
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
    if collection.count() == 0:
        raise HTTPException(status_code=404, detail="인덱싱된 문제집이 없습니다.")

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    content_type = file.content_type or "image/jpeg"
    if content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="이미지 파일(jpg/png/webp)만 업로드 가능합니다.")

    file_bytes = await file.read()
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    try:
        extracted_text = extract_text_from_image_base64(image_base64, mime_type=content_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 OCR 실패: {str(e)}")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="이미지에서 텍스트를 추출할 수 없습니다.")

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
    results = collection.get(
        where={"workbook_name": workbook_name},
        include=["metadatas"],
    )
    ids = results.get("ids", [])
    if not ids:
        raise HTTPException(status_code=404, detail="해당 문제집을 찾을 수 없습니다.")

    collection.delete(ids=ids)
    return {"success": True, "deleted_chunks": len(ids), "workbook_name": workbook_name}

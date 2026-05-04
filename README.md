# 📚 문제집 출처 탐색 RAG 서버

시험지 사진을 찍으면 어떤 문제집 몇 페이지에서 나왔는지 알려주는 RAG(검색 증강 생성) API 서버입니다.

## 🏗️ 기술 스택

| 역할 | 기술 |
|---|---|
| API 서버 | FastAPI + Uvicorn |
| 벡터 DB | ChromaDB (로컬 영속 저장) |
| PDF 파싱 | pdfplumber |
| AI (OCR + 임베딩) | Google Gemini API |
| 배포 | Railway |

---

## 🚀 Railway 배포 방법

### 1단계 — GitHub 레포 준비

```bash
git init
git add .
git commit -m "init: RAG 서버 초기 세팅"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2단계 — Railway 프로젝트 생성

1. [railway.app](https://railway.app) 접속 후 로그인
2. **New Project** → **Deploy from GitHub repo** 선택
3. 방금 push한 레포 선택
4. Railway가 자동으로 `Procfile`을 감지해 배포 시작

### 3단계 — 환경변수 설정

Railway 대시보드 → 프로젝트 선택 → **Variables** 탭

| 변수명 | 값 | 필수 |
|---|---|---|
| `GEMINI_API_KEY` | Gemini API 키 | ✅ |
| `CHROMA_PATH` | `/data/chroma_db` | 권장 |

> **GEMINI_API_KEY 발급**: [Google AI Studio](https://aistudio.google.com/app/apikey) → Create API Key

### 4단계 — 볼륨 마운트 (ChromaDB 영속화)

Railway의 기본 파일시스템은 배포 시 초기화됩니다. 데이터를 유지하려면:

1. Railway 대시보드 → **Volumes** → **New Volume**
2. Mount Path: `/data`
3. Variables에 `CHROMA_PATH=/data/chroma_db` 추가

> 볼륨 없이 사용하면 재배포 시 인덱싱된 문제집이 초기화됩니다.

### 5단계 — 배포 확인

```bash
# 서버 상태 확인
curl https://YOUR-APP.railway.app/health

# 응답 예시
# {"status":"ok","indexed_chunks":0}
```

---

## 📡 API 명세

### 기본 URL
```
https://YOUR-APP.railway.app
```

### 엔드포인트 목록

#### `GET /health`
서버 상태 및 인덱싱된 청크 수 확인

#### `POST /upload-workbook`
문제집 PDF 업로드 및 인덱싱

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `file` | File (PDF) | 문제집 PDF |
| `workbook_name` | string | 문제집 이름 |
| `subject` | string (선택) | 과목명 |

#### `POST /search-by-image`
시험지 이미지로 출처 검색

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `file` | File (이미지) | 시험지 사진 (jpg/png/webp) |
| `top_k` | int (선택, 기본 5) | 반환할 결과 수 |

#### `POST /search-by-text`
텍스트로 출처 검색 (JSON body)

```json
{
  "query_text": "다음 중 이차방정식의 근의 공식으로 올바른 것은?",
  "top_k": 5
}
```

#### `GET /workbooks`
등록된 문제집 목록 조회

#### `DELETE /workbook/{workbook_name}`
문제집 삭제

---

## 🖥️ 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
export GEMINI_API_KEY="your-api-key-here"

# 서버 실행
uvicorn main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

---

## 🔧 프론트 연동

`frontend-example.html` 파일 또는 아래 코드 예시 참고

```javascript
const RAG_SERVER = "https://YOUR-APP.railway.app";

// 문제집 업로드
async function uploadWorkbook(pdfFile, workbookName) {
  const formData = new FormData();
  formData.append("file", pdfFile);
  formData.append("workbook_name", workbookName);
  const res = await fetch(`${RAG_SERVER}/upload-workbook`, {
    method: "POST",
    body: formData,
  });
  return res.json();
}

// 시험지 이미지 검색
async function searchByImage(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);
  formData.append("top_k", "5");
  const res = await fetch(`${RAG_SERVER}/search-by-image`, {
    method: "POST",
    body: formData,
  });
  return res.json();
}
```

# RAG Agent — Document Intelligence

A production-grade Retrieval-Augmented Generation (RAG) chat application. Upload PDF, DOCX, PPTX, or TXT documents and ask questions grounded in their content, with accurate source citations, hybrid search, and real-time streaming answers.

---

## Features

- **Hybrid Search** — combines semantic (vector) search with keyword (BM25/FTS5) search using Reciprocal Rank Fusion (RRF) for more accurate retrieval.
- **Source Citations** — every answer links claims back to the exact filename and page.
- **Streaming Responses** — answers stream token-by-token in real time.
- **Multi-format Ingestion** — PDF, DOCX, PPTX, and TXT supported via Docling parsing, with semantic chunking and sliding-window overlap.
- **Per-Document Chat Sessions** — click a document to open or resume a conversation scoped to it, alongside an unscoped "all documents" mode.
- **Re-ranking & Hallucination Guard** — retrieved chunks are re-ranked for relevance before answering, and the assistant declines to answer when context isn't sufficiently relevant.
- **Usage Tracking** — per-answer token counts and timing breakdown, plus a usage dashboard with charts.
- **Light/Dark Theme** — full app-wide theme toggle.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite, axios |
| Backend | FastAPI (Python, async) |
| Vector Storage | `sqlite-vec` (embedded, single-file SQLite) |
| Keyword Search | SQLite FTS5 |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim, CPU-friendly) |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Re-ranking | Groq-based LLM relevance scoring |
| Document Parsing | Docling |

This stack was deliberately chosen to be lightweight and storage-conscious: `sqlite-vec` avoids running a separate vector database server, `all-MiniLM-L6-v2` is a small embedding model, and Groq is used for both generation and re-ranking so no local GPU/large model is required.

---

## Project Structure

```
backend/
├── .env                    # Config: Groq key, model, chunk size, paths
├── requirements.txt
├── Dockerfile
├── app/
│   ├── main.py              # FastAPI app, CORS, lifespan, routers
│   ├── config.py            # Pydantic Settings
│   ├── models/schemas.py    # Request/response models
│   ├── routers/
│   │   ├── documents.py     # Upload, list, delete documents
│   │   ├── search.py        # Hybrid search endpoints
│   │   ├── chat.py          # Sessions, messages, SSE streaming
│   │   └── system.py        # Health, models, config
│   └── services/
│       ├── ingestion.py     # Full pipeline: parse → chunk → embed → store
│       ├── chunking.py      # Semantic chunking + sliding window
│       ├── embedding.py     # Embedding model (lazy-loaded)
│       ├── vectorstore.py   # sqlite-vec + FTS5 + RRF hybrid search
│       ├── llm.py           # Groq client: generation, reranking, guards
│       └── chat.py          # Session management, RAG pipeline orchestration

frontend/
├── .env                     # VITE_API_BASE_URL
├── src/
│   ├── App.jsx               # Session init, theme, health polling
│   ├── services/api.js       # API client + SSE streaming
│   ├── components/
│   │   ├── Sidebar.jsx       # Upload, document list, session picker
│   │   ├── ChatArea.jsx      # Messages, citations, streaming
│   │   └── Toast.jsx
│   └── styles/index.css      # Design system (light/dark theme variables)
```

---

## Getting Started

### Prerequisites

- Python **3.11** (not newer — some ML dependencies lack prebuilt wheels for very recent Python versions)
- Node.js (for the frontend)
- A [Groq API key](https://console.groq.com/keys)

### Backend Setup

```bash
cd backend
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

Create `backend/.env`:

```dotenv
# Groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Embedding
EMBEDDING_MODEL=all-MiniLM-L6-v2

# SQLite-Vec
SQLITE_VEC_PATH=./data/rag_store.db

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Server
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173

# Upload
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE_MB=50
```

Run the backend:

```bash
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

Run the frontend:

```bash
npm run dev
```

Visit the printed local URL (typically `http://localhost:5173`).

---

## Deployment

- **Backend** → deploy via Docker to [Hugging Face Spaces](https://huggingface.co/spaces) (free tier offers significantly more RAM than most alternatives, which matters for the embedding model).
- **Frontend** → deploy to Vercel, Netlify, or Cloudflare Pages as a static Vite build.

After deploying, update:
- Frontend's `VITE_API_BASE_URL` to point to the deployed backend URL.
- Backend's `CORS_ORIGINS` to include the deployed frontend URL.

---

## Security Notes

- Never commit `.env` — it's excluded via `.gitignore`.
- Rotate any API key immediately if it's ever exposed (pasted in chat, committed to git, etc.).
- Secrets are read from environment variables only, never hardcoded or logged.

---

## Known Considerations

- The embedding model is lazy-loaded on first request to reduce memory usage at startup.
- PPTX/PDF citation precision depends on what Docling's extraction preserves — page-level citation is reliable; line-level precision may vary by file type.
- Groq's free tier has rate limits (requests-per-minute); each chat query makes multiple API calls (query rephrasing, re-ranking, generation), so heavy testing can hit these limits faster than expected.

# ── RAG Agent Backend — Production Dockerfile ──────────────────
# Optimised for CPU-only inference on free hosting tiers
# (Hugging Face Spaces, Render, Railway)

FROM python:3.11-slim AS base

# System dependencies required by:
#   - sqlite-vec: loads as a native SQLite extension
#   - PyMuPDF (fitz): PDF parsing fallback
#   - sentence-transformers: model downloads
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
        libffi-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies (cached layer) ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Application code ──────────────────────────────────────────
COPY app/ ./app/

# ── Runtime directories ──────────────────────────────────────
RUN mkdir -p data/uploads logs

# ── Port & command ────────────────────────────────────────────
# Reads $PORT from environment (set by HF Spaces / Render / Railway)
# Falls back to 7860 (HF Spaces default) if unset
ENV PORT=7860
EXPOSE ${PORT}

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}

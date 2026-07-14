"""FastAPI application entry point for the RAG Agent backend."""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Monkey-patch huggingface_hub to avoid WinError 1314 symlink failures on Windows.
# Windows without Developer Mode cannot create symlinks; the HF library's
# are_symlinks_supported() incorrectly returns True, then os.symlink() crashes.
# We force it to return False so HF falls back to file copies instead.
try:
    import huggingface_hub.file_download as _hf_fd
    _hf_fd.are_symlinks_supported = lambda *args, **kwargs: False
except ImportError:
    pass

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.routers import documents, search, chat, system, tools, feedback, usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rate limiter (Phase 7)
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("RAG Agent Backend starting up")
    logger.info(f"  LLM: Groq")
    logger.info(f"  Groq model: {settings.groq_model}")
    logger.info(f"  Embedding model: {settings.embedding_model}")
    logger.info(f"  SQLite-vec path: {settings.sqlite_vec_path}")
    logger.info(f"  Chunk size: {settings.chunk_size}, overlap: {settings.chunk_overlap}")
    logger.info(f"  Rate limit: {settings.rate_limit}")
    logger.info(f"  API key: {'set' if settings.groq_api_key else 'NOT SET'}")
    logger.info("=" * 60)

    # Pre-initialize vector store
    from app.services.vectorstore import get_db
    get_db()
    logger.info("SQLite-vec initialized")

    # Pre-initialize trace tables
    from app.services.tracing import _ensure_trace_tables
    _ensure_trace_tables()
    logger.info("Trace/feedback tables initialized")

    from app.services.usage import _ensure_usage_table
    _ensure_usage_table()
    logger.info("Usage table initialized")

    yield

    logger.info("RAG Agent Backend shutting down")


app = FastAPI(
    title="RAG Agent API",
    description="Production-grade Retrieval-Augmented Generation agent backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(system.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(tools.router)
app.include_router(feedback.router)
app.include_router(usage.router)


@app.get("/")
async def root():
    return {
        "name": "RAG Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }

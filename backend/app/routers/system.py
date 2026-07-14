"""System health, config, and model listing endpoints."""

import logging
from fastapi import APIRouter

from app.config import settings
from app.services.llm import check_groq_connection, list_groq_models
from app.services.vectorstore import is_connected as vectorstore_connected
from app.models.schemas import HealthResponse, ConfigResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check — reports Groq API and SQLite-vec connectivity."""
    llm_ok = await check_groq_connection()
    vec_ok = vectorstore_connected()
    
    return HealthResponse(
        status="healthy" if (llm_ok and vec_ok) else "degraded",
        llm_connected=llm_ok,
        vectorstore_connected=vec_ok,
        model=settings.groq_model,
    )


@router.get("/models")
async def get_models():
    """List available Groq models."""
    models = await list_groq_models()
    return {"models": models, "active_model": settings.groq_model}


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Expose non-secret runtime configuration. Never exposes API key."""
    return ConfigResponse(
        groq_model=settings.groq_model,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_size_mb=settings.max_upload_size_mb,
        sqlite_vec_path=settings.sqlite_vec_path,
    )

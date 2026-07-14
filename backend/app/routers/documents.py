"""Document ingestion and management API endpoints."""

import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.services.ingestion import ingest_document
from app.services.tools import summarize_document
from app.services.vectorstore import (
    get_all_documents,
    get_document,
    delete_document_chunks,
    delete_document_metadata,
)
from app.services.chat import get_sessions_for_document
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
    ChatSession,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for processing.
    Returns immediately with status 'processing' (or 'duplicate').
    The ingestion pipeline (parse → chunk → embed → store) runs in the background.
    Poll GET /documents/{doc_id} to check when status becomes 'ready' or 'failed'.
    """
    # Validate file type
    allowed_types = {".pdf", ".docx", ".doc", ".txt", ".ppt", ".pptx"}
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(allowed_types))}"
        )
    
    # Read file
    file_bytes = await file.read()
    
    # Check file size
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB"
        )
    
    # Ingest — returns immediately with 'processing' status;
    # background task handles the rest.
    result = await ingest_document(file_bytes, filename, file.content_type or "")
    
    return DocumentUploadResponse(**result)


@router.get("", response_model=DocumentListResponse)
def list_documents():
    """List all uploaded documents with metadata."""
    docs = get_all_documents()
    return DocumentListResponse(
        documents=[DocumentMetadata(**d) for d in docs],
        total=len(docs)
    )


@router.get("/{doc_id}")
def get_document_detail(doc_id: str):
    """Get details of a specific document."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """Remove a document and all its chunks from the vector store."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    delete_document_chunks(doc_id)
    delete_document_metadata(doc_id)
    
    return {"status": "deleted", "doc_id": doc_id}


@router.post("/{doc_id}/summarize")
async def summarize_doc(doc_id: str):
    """Generate a full document summary (separate from chat Q&A)."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    result = await summarize_document(doc_id)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@router.get("/{doc_id}/sessions", response_model=list[ChatSession])
def list_document_sessions(doc_id: str):
    """List all chat sessions scoped to a specific document."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    sessions = get_sessions_for_document(doc_id)
    return [
        ChatSession(
            session_id=s["session_id"],
            created_at=s["created_at"],
            message_count=s["message_count"],
            document_id=s.get("document_id"),
            document_ids=s.get("document_ids", []),
            document_filename=s.get("document_filename"),
            preview=s.get("preview", "Empty conversation"),
        )
        for s in sessions
    ]


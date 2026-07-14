"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# === Documents ===

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    file_hash: str
    document_type: str
    author: Optional[str] = None
    date_uploaded: datetime
    total_pages: int = 0
    total_chunks: int = 0
    status: str = "processing"  # processing, ready, error


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    message: str
    total_chunks: int = 0


# === Search / Retrieval ===

class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    content: str
    score: float
    metadata: dict = {}


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=50)


class SearchFilteredRequest(SearchRequest):
    filters: dict = Field(default_factory=dict)
    # e.g. {"document_type": "pdf", "filename": "report.pdf", "date_from": "2025-01-01"}


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total: int


# === Chat ===

class ChatSessionCreate(BaseModel):
    document_id: Optional[str] = None
    document_ids: Optional[list[str]] = None


class ChatSession(BaseModel):
    session_id: str
    created_at: datetime
    message_count: int = 0
    document_id: Optional[str] = None
    document_ids: list[str] = []
    document_filename: Optional[str] = None
    preview: Optional[str] = None


class Citation(BaseModel):
    filename: str
    page_number: int
    chunk_id: str
    relevance_score: float
    snippet: str


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    citations: list[Citation] = []
    trace_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatMessageRequest(BaseModel):
    message: str
    metadata_filters: dict = Field(default_factory=dict)


class ChatMessageResponse(BaseModel):
    session_id: str
    message: ChatMessage
    trace_id: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]
    document_id: Optional[str] = None
    document_filename: Optional[str] = None


# === Feedback ===

class FeedbackRequest(BaseModel):
    trace_id: str
    session_id: str
    rating: str  # "up" or "down"
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str


# === System ===

class HealthResponse(BaseModel):
    status: str
    llm_connected: bool
    vectorstore_connected: bool
    model: str


class ConfigResponse(BaseModel):
    groq_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    max_upload_size_mb: int
    sqlite_vec_path: str = ""

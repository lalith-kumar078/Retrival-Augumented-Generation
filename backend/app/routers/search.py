"""Search/retrieval API endpoints."""

import logging
from fastapi import APIRouter

from app.services.vectorstore import hybrid_search
from app.models.schemas import SearchRequest, SearchFilteredRequest, SearchResponse, ChunkResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Hybrid search endpoint combining vector (semantic) search and
    keyword (BM25) search with Reciprocal Rank Fusion.
    """
    results = await hybrid_search(request.query, top_k=request.top_k)
    
    chunk_results = []
    for r in results:
        chunk_results.append(ChunkResult(
            chunk_id=r.get("chunk_id", ""),
            doc_id=r.get("doc_id", ""),
            filename=r.get("filename", ""),
            page_number=r.get("page_number", 1),
            content=r.get("content", ""),
            score=r.get("score", 0.0),
            metadata={
                k: v for k, v in r.items()
                if k not in ("chunk_id", "doc_id", "filename", "page_number", "content", "score", "vector")
            },
        ))
    
    return SearchResponse(
        query=request.query,
        results=chunk_results,
        total=len(chunk_results)
    )


@router.post("/filtered", response_model=SearchResponse)
async def search_filtered(request: SearchFilteredRequest):
    """
    Hybrid search with metadata filtering.
    Supports filters like: document_type, filename, date_from, date_to, author.
    """
    results = await hybrid_search(
        request.query,
        top_k=request.top_k,
        filters=request.filters
    )
    
    chunk_results = []
    for r in results:
        chunk_results.append(ChunkResult(
            chunk_id=r.get("chunk_id", ""),
            doc_id=r.get("doc_id", ""),
            filename=r.get("filename", ""),
            page_number=r.get("page_number", 1),
            content=r.get("content", ""),
            score=r.get("score", 0.0),
            metadata={
                k: v for k, v in r.items()
                if k not in ("chunk_id", "doc_id", "filename", "page_number", "content", "score", "vector")
            },
        ))
    
    return SearchResponse(
        query=request.query,
        results=chunk_results,
        total=len(chunk_results)
    )

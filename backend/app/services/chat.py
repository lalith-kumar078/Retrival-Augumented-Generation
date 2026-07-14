"""Chat session management service — integrates full RAG pipeline.

Pipeline per message:
1. Multi-query retrieval (generate variations → hybrid search for each → dedup)
2. Re-ranking (cross-encoder or embedding fallback)
3. Hallucination guard (relevance check)
4. LLM generation with context + conversation history
5. Citation extraction
6. Pipeline tracing (log every stage with trace_id)
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional, AsyncGenerator

from app.services.vectorstore import hybrid_search, get_document
from app.services.llm import (
    generate_response,
    generate_response_stream,
    generate_query_variations,
    build_rag_prompt,
)
from app.services.usage import log_token_usage
from app.services.retrieval import (
    multi_query_retrieve,
    rerank_chunks,
    check_relevance,
    INSUFFICIENT_INFO_RESPONSE,
)
from app.services.tracing import log_trace
from app.config import settings

logger = logging.getLogger(__name__)

# In-memory session store (strict isolation — each session is a separate dict)
_sessions: dict[str, dict] = {}


# ── Session CRUD ──────────────────────────────────────────────────

def create_session(document_id: str = None, document_ids: list[str] = None) -> dict:
    """Create a new chat session, optionally scoped to document(s)."""
    session_id = str(uuid.uuid4())

    # Resolve document filename for display
    doc_filename = None
    effective_doc_ids = []

    if document_id:
        effective_doc_ids = [document_id]
        doc = get_document(document_id)
        if doc:
            doc_filename = doc.get("filename", "")
    elif document_ids:
        effective_doc_ids = document_ids
        if len(document_ids) == 1:
            doc = get_document(document_ids[0])
            if doc:
                doc_filename = doc.get("filename", "")

    _sessions[session_id] = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "messages": [],
        "message_count": 0,
        "document_id": document_id,
        "document_ids": effective_doc_ids,
        "document_filename": doc_filename,
    }
    scope_label = f" (scoped to {doc_filename or document_id})" if document_id else " (unscoped)"
    logger.info(f"Created chat session: {session_id}{scope_label}")
    return _sessions[session_id]


def get_session(session_id: str) -> Optional[dict]:
    """Get a session by ID. Returns None if not found (strict isolation)."""
    return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    """Delete a session and all its history."""
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Deleted chat session: {session_id}")
        return True
    return False


def get_session_history(session_id: str) -> list[dict]:
    """Get conversation history for a specific session only."""
    session = _sessions.get(session_id)
    if session:
        return session["messages"]
    return []


def _get_session_preview(session: dict) -> str:
    messages = session.get("messages", [])
    if not messages:
        return "Empty conversation"
    
    # Try to find the first user message
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    if user_msg:
        content = user_msg.get("content", "")
    else:
        content = messages[0].get("content", "")
        
    content_stripped = content.strip().replace("\n", " ")
    if len(content_stripped) > 60:
        return content_stripped[:57] + "..."
    return content_stripped


def list_all_sessions() -> list[dict]:
    """List all active sessions (metadata only, no message bodies)."""
    return [
        {
            "session_id": s["session_id"],
            "created_at": s["created_at"],
            "message_count": s["message_count"],
            "document_id": s.get("document_id"),
            "document_ids": s.get("document_ids", []),
            "document_filename": s.get("document_filename"),
            "preview": _get_session_preview(s),
        }
        for s in _sessions.values()
    ]


def get_sessions_for_document(doc_id: str) -> list[dict]:
    """Return all sessions scoped to a specific document."""
    results = []
    for s in _sessions.values():
        if s.get("document_id") == doc_id or doc_id in s.get("document_ids", []):
            results.append({
                "session_id": s["session_id"],
                "created_at": s["created_at"],
                "message_count": s["message_count"],
                "document_id": s.get("document_id"),
                "document_ids": s.get("document_ids", []),
                "document_filename": s.get("document_filename"),
                "preview": _get_session_preview(s),
            })
    return results


def add_message(session_id: str, role: str, content: str, citations: list = None, trace_id: str = None):
    """Add a message to a session's history."""
    session = _sessions.get(session_id)
    if session is None:
        return

    message = {
        "role": role,
        "content": content,
        "citations": citations or [],
        "trace_id": trace_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
    session["messages"].append(message)
    session["message_count"] += 1


def _get_doc_filters(session_id: str, extra_filters: dict = None) -> dict:
    """Build metadata filters, auto-adding document scope from session."""
    filters = dict(extra_filters or {})
    session = _sessions.get(session_id)
    if session:
        doc_id = session.get("document_id")
        doc_ids = session.get("document_ids", [])
        # If session is scoped to a single document, add doc_id filter
        if doc_id and "doc_id" not in filters:
            filters["doc_id"] = doc_id
        elif len(doc_ids) == 1 and "doc_id" not in filters:
            filters["doc_id"] = doc_ids[0]
        # Multi-doc scope handled by doc_ids filter
        elif len(doc_ids) > 1 and "doc_ids" not in filters:
            filters["doc_ids"] = doc_ids
    return filters


# ── Full RAG Pipeline (non-streaming) ─────────────────────────────

async def process_message(
    session_id: str,
    user_message: str,
    trace_id: str,
    metadata_filters: dict = None,
) -> dict:
    """
    Process a user message through the full enhanced RAG pipeline:
    1. Generate query variations (multi-query)
    2. Multi-query hybrid search with dedup
    3. Re-rank results
    4. Hallucination guard
    5. LLM generation with context + conversation history
    6. Extract citations
    7. Log full pipeline trace
    """
    start_time = time.time()

    # Add user message to history
    add_message(session_id, "user", user_message, trace_id=trace_id)

    # Get conversation history for context
    history = get_session_history(session_id)
    conversation_context = [
        {"role": m["role"], "content": m["content"]}
        for m in history[:-1]  # Exclude the just-added message
    ]

    # Build filters with document scope
    filters = _get_doc_filters(session_id, metadata_filters)

    # Step 1: Generate query variations
    try:
        variations = await generate_query_variations(user_message, num_variations=3)
    except Exception:
        variations = []

    # Step 2: Multi-query retrieval
    chunks = await multi_query_retrieve(
        query=user_message,
        query_variations=variations,
        top_k=20,
        filters=filters,
    )
    retrieved_ids = [c.get("chunk_id", "") for c in chunks]

    # Step 3: Re-rank
    chunks = await rerank_chunks(user_message, chunks, top_k=10)
    reranked_ids = [c.get("chunk_id", "") for c in chunks]

    # Step 4: Hallucination guard
    relevance_pass = True
    if not chunks:
        answer = "I don't have enough information to answer that. Please upload some documents first."
        citations = []
        relevance_pass = False
    elif not settings.disable_hallucination_guard and not await check_relevance(user_message, chunks):
        answer = INSUFFICIENT_INFO_RESPONSE
        citations = _build_citations(chunks[:3])  # Still show what was found
        relevance_pass = False
    else:
        # Step 5: Generate response
        answer = await generate_response(
            query=user_message,
            context_chunks=chunks,
            conversation_history=conversation_context,
        )
        citations = _build_citations(chunks[:5])

    # Add assistant message to history
    add_message(session_id, "assistant", answer, citations=citations, trace_id=trace_id)

    # Step 6: Log trace
    duration_ms = (time.time() - start_time) * 1000
    prompt_messages = build_rag_prompt(user_message, chunks, conversation_context)
    final_prompt = prompt_messages[-1]["content"] if prompt_messages else ""

    log_trace(
        trace_id=trace_id,
        session_id=session_id,
        query=user_message,
        retrieved_ids=retrieved_ids,
        reranked_ids=reranked_ids,
        final_prompt=final_prompt,
        llm_response=answer,
        citations=citations,
        relevance_pass=relevance_pass,
        duration_ms=duration_ms,
    )

    groq_calls = 2 + (1 if relevance_pass else 0)
    logger.info(f"[QUOTA] Trace {trace_id}: Query finished using {groq_calls} Groq API calls.")


    return {
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "trace_id": trace_id,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Streaming RAG Pipeline ────────────────────────────────────────

async def process_message_stream(
    session_id: str,
    user_message: str,
    trace_id: str,
    metadata_filters: dict = None,
) -> AsyncGenerator[dict, None]:
    """
    Process a user message with streaming response.
    Yields dicts with type: 'token', 'citations', 'error', or 'done'.
    Same pipeline as process_message but streams the LLM output.
    """
    start_time = time.time()

    # Add user message to history
    add_message(session_id, "user", user_message, trace_id=trace_id)

    # Get conversation history
    history = get_session_history(session_id)
    conversation_context = [
        {"role": m["role"], "content": m["content"]}
        for m in history[:-1]
    ]

    # Build filters with document scope
    filters = _get_doc_filters(session_id, metadata_filters)

    try:
        # Step 1: Generate query variations
        try:
            variations = await generate_query_variations(user_message, num_variations=3)
        except Exception:
            variations = []

        # Step 2: Multi-query retrieval
        t_retrieval_start = time.time()
        chunks = await multi_query_retrieve(
            query=user_message,
            query_variations=variations,
            top_k=20,
            filters=filters,
        )
        retrieved_ids = [c.get("chunk_id", "") for c in chunks]
        t_retrieval_ms = (time.time() - t_retrieval_start) * 1000

        # Step 3: Re-rank
        t_rerank_start = time.time()
        chunks = await rerank_chunks(user_message, chunks, top_k=10)
        reranked_ids = [c.get("chunk_id", "") for c in chunks]
        t_rerank_ms = (time.time() - t_rerank_start) * 1000

        # Step 4: Hallucination guard
        relevance_pass = True
        if not chunks:
            msg = "I don't have enough information to answer that. Please upload some documents first."
            total_duration_ms = (time.time() - start_time) * 1000
            yield {"type": "token", "content": msg}
            yield {"type": "citations", "citations": []}
            yield {"type": "stats", "stats": {
                "duration_ms": total_duration_ms,
                "retrieval_ms": t_retrieval_ms,
                "rerank_ms": t_rerank_ms,
                "generation_ms": 0,
                "chunks_retrieved": 0,
                "chunks_reranked": 0,
                "usage": {}
            }}
            yield {"type": "done"}
            add_message(session_id, "assistant", msg, trace_id=trace_id)

            log_trace(
                trace_id=trace_id, session_id=session_id, query=user_message,
                retrieved_ids=[], reranked_ids=[], relevance_pass=False,
                duration_ms=total_duration_ms,
            )
            return

        if not settings.disable_hallucination_guard and not await check_relevance(user_message, chunks):
            total_duration_ms = (time.time() - start_time) * 1000
            citations = _build_citations(chunks[:3])
            yield {"type": "citations", "citations": citations}
            yield {"type": "token", "content": INSUFFICIENT_INFO_RESPONSE}
            yield {"type": "stats", "stats": {
                "duration_ms": total_duration_ms,
                "retrieval_ms": t_retrieval_ms,
                "rerank_ms": t_rerank_ms,
                "generation_ms": 0,
                "chunks_retrieved": len(retrieved_ids),
                "chunks_reranked": len(reranked_ids),
                "usage": {}
            }}
            yield {"type": "done"}
            add_message(session_id, "assistant", INSUFFICIENT_INFO_RESPONSE, citations=citations, trace_id=trace_id)

            log_trace(
                trace_id=trace_id, session_id=session_id, query=user_message,
                retrieved_ids=retrieved_ids, reranked_ids=reranked_ids,
                relevance_pass=False, duration_ms=total_duration_ms,
            )
            return

        # Build citations
        citations = _build_citations(chunks[:5])
        yield {"type": "citations", "citations": citations}

        # Step 5: Stream LLM response
        full_response = ""
        t_llm_start = time.time()
        final_usage = None
        async for chunk in generate_response_stream(
            query=user_message,
            context_chunks=chunks,
            conversation_history=conversation_context,
        ):
            if chunk["type"] == "content":
                token = chunk["content"]
                full_response += token
                yield {"type": "token", "content": token}
            elif chunk["type"] == "usage":
                final_usage = chunk["usage"]
                
        t_llm_ms = (time.time() - t_llm_start) * 1000

        # Store full response in history
        add_message(session_id, "assistant", full_response, citations=citations, trace_id=trace_id)

        # Log trace and usage
        total_duration_ms = (time.time() - start_time) * 1000
        prompt_messages = build_rag_prompt(user_message, chunks, conversation_context)
        final_prompt = prompt_messages[-1]["content"] if prompt_messages else ""

        log_trace(
            trace_id=trace_id, session_id=session_id, query=user_message,
            retrieved_ids=retrieved_ids, reranked_ids=reranked_ids,
            final_prompt=final_prompt, llm_response=full_response,
            citations=citations, relevance_pass=True, duration_ms=total_duration_ms,
        )

        if final_usage:
            log_token_usage(
                trace_id=trace_id,
                session_id=session_id,
                prompt_tokens=final_usage.get("prompt_tokens", 0),
                completion_tokens=final_usage.get("completion_tokens", 0),
                total_tokens=final_usage.get("total_tokens", 0),
            )

        yield {
            "type": "stats",
            "stats": {
                "duration_ms": total_duration_ms,
                "retrieval_ms": t_retrieval_ms,
                "rerank_ms": t_rerank_ms,
                "generation_ms": t_llm_ms,
                "chunks_retrieved": len(retrieved_ids),
                "chunks_reranked": len(reranked_ids),
                "usage": final_usage or {}
            }
        }
        
        yield {"type": "done"}

    except Exception as e:
        logger.error(f"Stream pipeline error: {e}", exc_info=True)
        error_msg = f"An error occurred: {str(e)}"
        yield {"type": "error", "content": error_msg}
        yield {"type": "done"}
        add_message(session_id, "assistant", f"[Error] {error_msg}", trace_id=trace_id)


# ── Helpers ───────────────────────────────────────────────────────

def _build_citations(chunks: list[dict]) -> list[dict]:
    """Build citation list from chunks, deduplicating by filename+page."""
    citations = []
    seen = set()
    for chunk in chunks:
        key = (chunk.get("filename", ""), chunk.get("page_number", 0))
        if key not in seen:
            seen.add(key)
            citations.append({
                "filename": chunk.get("filename", "unknown"),
                "page_number": chunk.get("page_number", 1),
                "line_number": chunk.get("line_number"),
                "chunk_id": chunk.get("chunk_id", ""),
                "relevance_score": round(chunk.get("rerank_score", chunk.get("score", 0)), 4),
                "snippet": chunk.get("content", "")[:200] + "...",
            })
    return citations

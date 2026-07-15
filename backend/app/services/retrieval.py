"""Retrieval quality layer — multi-query, re-ranking, hallucination guard."""

import asyncio
import logging
import time
import numpy as np
from typing import Optional

from app.services.vectorstore import hybrid_search

logger = logging.getLogger(__name__)

# Relevance threshold for hallucination guard (Groq 0-10 scale)
# Set to 1.0: the guard only blocks truly irrelevant context (all scores 0).
# The LLM generation step itself naturally says "not enough info" when context
# is insufficient, so the guard is a last-resort safety net, not a strict filter.
# Previous values of 5.0 and 3.0 both caused false blocks on legitimate queries.
RELEVANCE_THRESHOLD = 1.0


async def multi_query_retrieve(
    query: str,
    query_variations: list[str],
    top_k: int = 10,
    filters: dict = None,
) -> list[dict]:
    """
    Multi-query retrieval: run hybrid search for original query + all variations concurrently,
    deduplicate results by chunk_id, keep highest score per chunk.
    """
    start_time = time.time()
    all_queries = [query] + query_variations
    seen_chunks = {}

    # Run searches concurrently
    results_list = await asyncio.gather(
        *(hybrid_search(q, top_k=top_k, filters=filters) for q in all_queries)
    )

    for results in results_list:
        for chunk in results:
            cid = chunk.get("chunk_id", "")
            if cid not in seen_chunks or chunk.get("score", 0) > seen_chunks[cid].get("score", 0):
                seen_chunks[cid] = chunk

    # Sort by score descending and return top_k
    deduped = sorted(seen_chunks.values(), key=lambda x: x.get("score", 0), reverse=True)
    
    logger.info(f"Multi-query retrieval total took {(time.time() - start_time)*1000:.1f}ms ({len(all_queries)} queries, {len(deduped)} unique chunks)")
    return deduped[:top_k]


async def rerank_chunks(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """
    Re-rank chunks using Groq API.
    Falls back to hybrid search scores if Groq fails.
    """
    if not chunks:
        return []

    start_time = time.time()
    from app.services.llm import score_chunks_with_groq
    
    # We only want to re-rank the top candidates to save API cost and latency
    chunks_to_score = chunks[:20] 
    
    try:
        scores = await score_chunks_with_groq(query, chunks_to_score)
    except Exception as e:
        logger.warning(f"Groq re-ranking threw an exception: {e}")
        scores = {}
    
    if scores:
        for chunk in chunks_to_score:
            cid = chunk.get("chunk_id", "")
            chunk["rerank_score"] = float(scores.get(cid, 0.0))
            
        reranked = sorted(chunks_to_score, key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info(f"Groq re-ranking took {(time.time() - start_time)*1000:.1f}ms, top score: {reranked[0].get('rerank_score', 0)}")
        return reranked[:top_k]
    else:
        logger.warning("Groq re-ranking failed, falling back to hybrid search scores")
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("score", 0.0)
            chunk["is_fallback"] = True
        return chunks[:top_k]

from collections import deque
_recent_guard_results = deque(maxlen=10)

async def check_relevance(query: str, chunks: list[dict], threshold: float = None) -> bool:
    """
    Hallucination guard: check if retrieved chunks are relevant enough
    to the query. Returns False if context doesn't sufficiently match.
    """
    if not chunks:
        _recent_guard_results.append(False)
        return False

    # Use the top chunk's rerank_score
    top_chunk = chunks[0]
    top_score = top_chunk.get("rerank_score", 0.0)

    # If we fell back to hybrid search scores, the scale is 0.0 - 1.0
    if top_chunk.get("is_fallback"):
        actual_threshold = 0.01  # very permissive threshold for RRF scores
    else:
        actual_threshold = threshold or RELEVANCE_THRESHOLD

    is_relevant = top_score >= actual_threshold
    
    score_summary = ", ".join(f"{c.get('chunk_id','?')}={c.get('rerank_score',0)}" for c in chunks[:5])
    logger.debug(f"RELEVANCE CHECK: top_score={top_score}, threshold={actual_threshold}, relevant={is_relevant}, scores=[{score_summary}]")

    if not is_relevant:
        logger.info(f"Hallucination guard triggered: top_score={top_score} < threshold={actual_threshold}")

    _recent_guard_results.append(is_relevant)
    if len(_recent_guard_results) == 10 and _recent_guard_results.count(False) >= 8:
        logger.warning(f"HIGH HALLUCINATION GUARD BLOCK RATE: {_recent_guard_results.count(False)} of the last 10 queries blocked! Check if reranker JSON parsing is failing or scales changed.")

    return is_relevant


INSUFFICIENT_INFO_RESPONSE = (
    "I don't have enough information to answer that based on the uploaded documents. "
    "The retrieved context doesn't appear sufficiently relevant to your question. "
    "Try rephrasing your question or uploading more relevant documents."
)

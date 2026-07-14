"""Pipeline tracing and feedback logging service (Phase 6).

Logs every pipeline stage with a unique trace_id:
  query → retrieved chunks → reranked chunks → final prompt → LLM response

Also handles thumbs-up/down feedback with full context logging on downvote.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

from app.services.vectorstore import get_db

logger = logging.getLogger(__name__)

_TRACE_TABLE_INITIALIZED = False


def _ensure_trace_tables():
    """Create tracing/feedback tables if they don't exist."""
    global _TRACE_TABLE_INITIALIZED
    if _TRACE_TABLE_INITIALIZED:
        return

    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id       TEXT PRIMARY KEY,
            session_id     TEXT DEFAULT '',
            query          TEXT NOT NULL,
            retrieved_ids  TEXT DEFAULT '[]',
            reranked_ids   TEXT DEFAULT '[]',
            final_prompt   TEXT DEFAULT '',
            llm_response   TEXT DEFAULT '',
            citations      TEXT DEFAULT '[]',
            relevance_pass INTEGER DEFAULT 1,
            duration_ms    REAL DEFAULT 0,
            created_at     TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id       TEXT NOT NULL,
            session_id     TEXT DEFAULT '',
            rating         TEXT NOT NULL,
            comment        TEXT DEFAULT '',
            full_context   TEXT DEFAULT '',
            created_at     TEXT NOT NULL,
            FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
        )
    """)
    conn.commit()
    _TRACE_TABLE_INITIALIZED = True
    logger.info("Trace/feedback tables initialized")


def log_trace(
    trace_id: str,
    session_id: str,
    query: str,
    retrieved_ids: list[str] = None,
    reranked_ids: list[str] = None,
    final_prompt: str = "",
    llm_response: str = "",
    citations: list[dict] = None,
    relevance_pass: bool = True,
    duration_ms: float = 0,
):
    """Log a full pipeline trace."""
    _ensure_trace_tables()
    conn = get_db()

    conn.execute(
        """INSERT OR REPLACE INTO traces
           (trace_id, session_id, query, retrieved_ids, reranked_ids,
            final_prompt, llm_response, citations, relevance_pass,
            duration_ms, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trace_id,
            session_id,
            query,
            json.dumps(retrieved_ids or []),
            json.dumps(reranked_ids or []),
            final_prompt[:5000],  # Cap at 5KB
            llm_response[:5000],
            json.dumps(citations or []),
            1 if relevance_pass else 0,
            duration_ms,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    logger.info(f"Logged trace {trace_id}: query='{query[:50]}...', duration={duration_ms:.0f}ms")


def get_trace(trace_id: str) -> Optional[dict]:
    """Fetch a full pipeline trace by trace_id."""
    _ensure_trace_tables()
    conn = get_db()

    row = conn.execute(
        "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
    ).fetchone()

    if row is None:
        return None

    trace = dict(row)
    # Parse JSON fields
    for field in ("retrieved_ids", "reranked_ids", "citations"):
        try:
            trace[field] = json.loads(trace[field])
        except (json.JSONDecodeError, TypeError):
            pass
    trace["relevance_pass"] = bool(trace.get("relevance_pass", 1))
    return trace


def log_feedback(
    trace_id: str,
    session_id: str,
    rating: str,
    comment: str = "",
):
    """
    Log user feedback. On downvote, stores the full prompt + context
    from the trace for later review.
    """
    _ensure_trace_tables()
    conn = get_db()

    full_context = ""
    if rating == "down":
        # Retrieve the full trace for logging
        trace = get_trace(trace_id)
        if trace:
            full_context = json.dumps({
                "query": trace.get("query", ""),
                "final_prompt": trace.get("final_prompt", ""),
                "llm_response": trace.get("llm_response", ""),
                "retrieved_ids": trace.get("retrieved_ids", []),
                "citations": trace.get("citations", []),
            })
        logger.warning(f"Downvote received for trace {trace_id}")

    conn.execute(
        """INSERT INTO feedback
           (trace_id, session_id, rating, comment, full_context, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            trace_id,
            session_id,
            rating,
            comment,
            full_context,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    logger.info(f"Feedback logged: trace={trace_id}, rating={rating}")

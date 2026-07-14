import logging
from datetime import datetime, timedelta
from typing import Optional

from app.services.vectorstore import get_db

logger = logging.getLogger(__name__)

_USAGE_TABLE_INITIALIZED = False

def _ensure_usage_table():
    global _USAGE_TABLE_INITIALIZED
    if _USAGE_TABLE_INITIALIZED:
        return

    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT,
            session_id TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    _USAGE_TABLE_INITIALIZED = True
    logger.info("Usage table initialized")

def log_token_usage(trace_id: str, session_id: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
    _ensure_usage_table()
    conn = get_db()
    conn.execute(
        """INSERT INTO token_usage 
           (trace_id, session_id, prompt_tokens, completion_tokens, total_tokens, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (trace_id, session_id, prompt_tokens, completion_tokens, total_tokens, datetime.utcnow().isoformat())
    )
    conn.commit()

def get_usage_stats():
    _ensure_usage_table()
    conn = get_db()
    
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    
    # Aggregates
    row_all = conn.execute("SELECT SUM(total_tokens) as t, COUNT(*) as c FROM token_usage").fetchone()
    row_today = conn.execute("SELECT SUM(total_tokens) as t, COUNT(*) as c FROM token_usage WHERE created_at >= ?", (today_start,)).fetchone()
    
    total_all = row_all['t'] or 0
    requests_all = row_all['c'] or 0
    total_today = row_today['t'] or 0
    requests_today = row_today['c'] or 0
    
    # Time series (last 50 requests)
    rows = conn.execute(
        "SELECT total_tokens, created_at FROM token_usage ORDER BY id DESC LIMIT 50"
    ).fetchall()
    
    timeseries = []
    for r in reversed(rows):
        timeseries.append({
            "tokens": r["total_tokens"],
            "timestamp": r["created_at"]
        })
        
    return {
        "total_tokens_all_time": total_all,
        "total_requests_all_time": requests_all,
        "total_tokens_today": total_today,
        "total_requests_today": requests_today,
        "timeseries": timeseries
    }

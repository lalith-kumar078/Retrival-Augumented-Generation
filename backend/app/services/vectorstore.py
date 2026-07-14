"""SQLite-vec vector store service for storing and searching document chunks.

Uses:
  - sqlite-vec's vec0 virtual table for vector similarity search
  - SQLite FTS5 for full-text keyword search
  - Standard SQLite tables for metadata
  - Reciprocal Rank Fusion (RRF) to combine results
"""

import asyncio
import json
import logging
import re
import sqlite3
import struct
import threading
import time
import numpy as np
from typing import Optional
from pathlib import Path

import sqlite_vec

from app.config import settings
from app.services.embedding import embed_query, get_embedding_dimension, EMBEDDING_DIM

logger = logging.getLogger(__name__)

_local_db = threading.local()


# ── helpers ────────────────────────────────────────────────────────

def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary blob (little-endian f32)."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _deserialize_f32(blob: bytes) -> list[float]:
    """Deserialize a binary blob back into a list of floats."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


# ── connection & schema ───────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Get or create the thread-local SQLite database connection with sqlite-vec loaded."""
    if hasattr(_local_db, "conn"):
        try:
            # Test if the connection is still alive/valid
            _local_db.conn.execute("SELECT 1")
            return _local_db.conn
        except sqlite3.ProgrammingError:
            pass

    db_path = settings.sqlite_vec_abs_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Connecting to SQLite-vec at {db_path} in thread {threading.current_thread().name}")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    _init_schema(conn)
    conn.execute("PRAGMA journal_mode=WAL;")
    _local_db.conn = conn
    return conn


def _init_schema(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    cur = conn.cursor()

    # ── documents metadata table ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id        TEXT PRIMARY KEY,
            filename      TEXT NOT NULL,
            file_hash     TEXT NOT NULL,
            document_type TEXT NOT NULL,
            author        TEXT DEFAULT '',
            date_uploaded TEXT NOT NULL,
            total_pages   INTEGER DEFAULT 0,
            total_chunks  INTEGER DEFAULT 0,
            status        TEXT DEFAULT 'processing'
        )
    """)

    # ── chunks metadata table ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id      TEXT PRIMARY KEY,
            doc_id        TEXT NOT NULL,
            filename      TEXT NOT NULL,
            page_number   INTEGER DEFAULT 1,
            content       TEXT NOT NULL,
            document_type TEXT DEFAULT '',
            date_uploaded TEXT DEFAULT '',
            vec_min       REAL DEFAULT 0,
            vec_max       REAL DEFAULT 0,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    # ── vec0 virtual table for vector search ──
    cur.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)

    # ── FTS5 virtual table for keyword search ──
    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id,
            content,
            tokenize='porter unicode61'
        )
    """)

    conn.commit()
    logger.info("SQLite-vec schema initialized")


# ── document CRUD ─────────────────────────────────────────────────

def store_document_metadata(doc_meta: dict):
    """Store document-level metadata."""
    conn = get_db()
    conn.execute(
        """INSERT OR REPLACE INTO documents
           (doc_id, filename, file_hash, document_type, author,
            date_uploaded, total_pages, total_chunks, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_meta["doc_id"],
            doc_meta["filename"],
            doc_meta["file_hash"],
            doc_meta["document_type"],
            doc_meta.get("author", ""),
            doc_meta["date_uploaded"],
            doc_meta.get("total_pages", 0),
            doc_meta.get("total_chunks", 0),
            doc_meta.get("status", "processing"),
        ),
    )
    conn.commit()
    logger.info(f"Stored metadata for document: {doc_meta.get('filename', 'unknown')}")


def update_document_status(doc_id: str, status: str, total_chunks: int = 0):
    """Update the status of a document."""
    conn = get_db()
    if total_chunks > 0:
        conn.execute(
            "UPDATE documents SET status = ?, total_chunks = ? WHERE doc_id = ?",
            (status, total_chunks, doc_id),
        )
    else:
        conn.execute(
            "UPDATE documents SET status = ? WHERE doc_id = ?",
            (status, doc_id),
        )
    conn.commit()


def get_all_documents() -> list[dict]:
    """Get all document metadata."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM documents ORDER BY date_uploaded DESC").fetchall()
    return [dict(r) for r in rows]


def get_document(doc_id: str) -> Optional[dict]:
    """Get a specific document's metadata."""
    conn = get_db()
    row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return dict(row) if row else None


def document_hash_exists(file_hash: str) -> Optional[str]:
    """Check if a document with this hash already exists. Returns doc_id if found."""
    conn = get_db()
    row = conn.execute(
        "SELECT doc_id, status FROM documents WHERE file_hash = ?", (file_hash,)
    ).fetchone()
    if row:
        if not row["status"].startswith("failed"):
            return row["doc_id"]
    return None


def delete_document_metadata(doc_id: str):
    """Delete document metadata."""
    conn = get_db()
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    conn.commit()
    logger.info(f"Deleted metadata for document: {doc_id}")


# ── chunk storage ─────────────────────────────────────────────────

def store_chunks(chunks: list[dict]):
    """
    Store embedded chunks in SQLite.
    Each chunk dict should have:
      chunk_id, doc_id, filename, page_number, content, document_type,
      date_uploaded, vector_f32, vec_min, vec_max
    """
    if not chunks:
        return

    conn = get_db()

    for chunk in chunks:
        # 1. Insert metadata row
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, doc_id, filename, page_number, content,
                document_type, date_uploaded, vec_min, vec_max)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk["filename"],
                chunk.get("page_number", 1),
                chunk["content"],
                chunk.get("document_type", ""),
                chunk.get("date_uploaded", ""),
                chunk.get("vec_min", 0.0),
                chunk.get("vec_max", 0.0),
            ),
        )

        # 2. Insert vector into vec0
        vec_blob = _serialize_f32(chunk["vector_f32"])
        conn.execute(
            "INSERT OR REPLACE INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk["chunk_id"], vec_blob),
        )

        # 3. Insert into FTS5 for keyword search
        conn.execute(
            "INSERT OR REPLACE INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
            (chunk["chunk_id"], chunk["content"]),
        )

    conn.commit()
    logger.info(f"Stored {len(chunks)} chunks in SQLite-vec")


def delete_document_chunks(doc_id: str):
    """Delete all chunks for a document from all tables."""
    conn = get_db()

    # Get chunk_ids first
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
    ).fetchall()
    chunk_ids = [r["chunk_id"] for r in rows]

    if chunk_ids:
        placeholders = ",".join("?" for _ in chunk_ids)
        conn.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})", chunk_ids)
        conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)

    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.commit()
    logger.info(f"Deleted {len(chunk_ids)} chunks for document: {doc_id}")


# ── search ────────────────────────────────────────────────────────

async def vector_search(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    """Perform vector similarity search using sqlite-vec's vec0 MATCH in a thread."""
    start_time = time.time()
    
    query_vector = await embed_query(query)
    query_blob = _serialize_f32(query_vector)

    # If there is a document scope filter, we want to retrieve more candidates
    # from the vector index before filtering, so we don't end up with 0 results.
    has_doc_filter = filters and ("doc_id" in filters or "doc_ids" in filters)
    k_val = 500 if has_doc_filter else top_k * 2

    def _sync_search():
        conn = get_db()
        rows = conn.execute(
            """
            SELECT v.chunk_id, v.distance
            FROM chunks_vec v
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (query_blob, k_val),
        ).fetchall()

        results = []
        for row in rows:
            chunk_id = row["chunk_id"]
            distance = row["distance"]

            meta = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if meta is None:
                continue

            meta_dict = dict(meta)

            # Apply filters here
            if filters and not _matches_filters(meta_dict, filters):
                continue

            meta_dict["score"] = 1.0 / (1.0 + distance)
            results.append(meta_dict)

            if len(results) >= top_k:
                break
        return results

    results = await asyncio.to_thread(_sync_search)
    logger.info(f"Vector search took {(time.time() - start_time)*1000:.1f}ms (found {len(results)})")
    return results


async def keyword_search(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    """Perform keyword search using SQLite FTS5 in a thread."""
    start_time = time.time()
    
    # Clean query to avoid FTS5 syntax errors (e.g. from commas, question marks)
    cleaned_query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query).strip()
    if not cleaned_query:
        cleaned_query = f'"{query}"'
    
    def _sync_search():
        conn = get_db()
        try:
            sql = """
                SELECT f.chunk_id, f.rank
                FROM chunks_fts f
                JOIN chunks c ON f.chunk_id = c.chunk_id
                WHERE f.content MATCH ?
            """
            params = [cleaned_query]

            if filters:
                if "doc_id" in filters:
                    sql += " AND c.doc_id = ?"
                    params.append(filters["doc_id"])
                elif "doc_ids" in filters and isinstance(filters["doc_ids"], list):
                    placeholders = ",".join("?" for _ in filters["doc_ids"])
                    sql += f" AND c.doc_id IN ({placeholders})"
                    params.extend(filters["doc_ids"])

            sql += " ORDER BY f.rank LIMIT ?"
            params.append(top_k * 5 if filters else top_k)

            rows = conn.execute(sql, params).fetchall()

            results = []
            for i, row in enumerate(rows):
                chunk_id = row["chunk_id"]

                meta = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
                if meta is None:
                    continue

                meta_dict = dict(meta)
                
                if filters and not _matches_filters(meta_dict, filters):
                    continue

                meta_dict["score"] = 1.0 / (i + 1)
                results.append(meta_dict)
                
                if len(results) >= top_k:
                    break
            return results

        except Exception as e:
            logger.warning(f"FTS5 search failed: {e}")
            return _manual_bm25_search(query, top_k, filters)

    results = await asyncio.to_thread(_sync_search)
    logger.info(f"Keyword search took {(time.time() - start_time)*1000:.1f}ms (found {len(results)})")
    return results


def _manual_bm25_search(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    """Fallback BM25-like keyword search when FTS5 fails."""
    conn = get_db()
    try:
        from rank_bm25 import BM25Okapi
        
        sql = "SELECT * FROM chunks"
        params = []
        if filters:
            if "doc_id" in filters:
                sql += " WHERE doc_id = ?"
                params.append(filters["doc_id"])
            elif "doc_ids" in filters and isinstance(filters["doc_ids"], list):
                placeholders = ",".join("?" for _ in filters["doc_ids"])
                sql += f" WHERE doc_id IN ({placeholders})"
                params.extend(filters["doc_ids"])
                
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return []

        all_data = [dict(r) for r in rows]
        
        if filters:
            all_data = [item for item in all_data if _matches_filters(item, filters)]
            if not all_data:
                return []

        corpus = [doc["content"].lower().split() for doc in all_data]
        bm25 = BM25Okapi(corpus)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                item = all_data[idx].copy()
                item["score"] = float(scores[idx])
                results.append(item)
        return results
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        return []


async def hybrid_search(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    """
    Hybrid search combining vector search and keyword search using
    Reciprocal Rank Fusion (RRF), running concurrently.
    """
    start_time = time.time()
    
    vector_results, keyword_results = await asyncio.gather(
        vector_search(query, top_k=top_k * 2, filters=filters),
        keyword_search(query, top_k=top_k * 2, filters=filters)
    )

    print("\n--- HYBRID SEARCH DIAGNOSTIC LOGS ---")
    print(f"Query: '{query}'")
    print(f"Filters: {filters}")
    print("\n(a) Raw results from vector search alone:")
    for idx, r in enumerate(vector_results):
        print(f"  [{idx}] chunk_id={r.get('chunk_id')}, filename={r.get('filename')}, score={r.get('score'):.4f}")
    
    print("\n(b) Raw results from keyword/FTS5 search alone:")
    for idx, r in enumerate(keyword_results):
        print(f"  [{idx}] chunk_id={r.get('chunk_id')}, filename={r.get('filename')}, score={r.get('score'):.4f}")

    k = 60
    scores = {}
    chunk_data = {}

    for rank, result in enumerate(vector_results):
        cid = result.get("chunk_id", str(rank))
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_data[cid] = result

    for rank, result in enumerate(keyword_results):
        cid = result.get("chunk_id", str(rank))
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for cid, score in ranked:
        if cid in chunk_data:
            item = chunk_data[cid]
            item["score"] = score

            if filters:
                if not _matches_filters(item, filters):
                    continue

            results.append(item)

    print("\n(c) Fused RRF result:")
    for idx, r in enumerate(results):
        print(f"  [{idx}] chunk_id={r.get('chunk_id')}, filename={r.get('filename')}, fused_rrf_score={r.get('score'):.4f}")
    print("-------------------------------------\n")

    logger.info(f"Hybrid search total took {(time.time() - start_time)*1000:.1f}ms")
    return results


def _matches_filters(item: dict, filters: dict) -> bool:
    """Check if a chunk matches the given metadata filters."""
    for key, value in filters.items():
        if key == "date_from":
            item_date = item.get("date_uploaded", "")
            if item_date and item_date < value:
                return False
        elif key == "date_to":
            item_date = item.get("date_uploaded", "")
            if item_date and item_date > value:
                return False
        elif key == "doc_ids" and isinstance(value, list):
            # Multi-document scope: chunk must belong to one of the listed docs
            if item.get("doc_id") not in value:
                return False
        else:
            if item.get(key) and item.get(key) != value:
                return False
    return True


# ── connectivity ──────────────────────────────────────────────────

def is_connected() -> bool:
    """Check if SQLite-vec is accessible."""
    try:
        get_db()
        return True
    except Exception:
        return False

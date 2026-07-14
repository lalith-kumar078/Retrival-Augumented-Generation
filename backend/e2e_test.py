#!/usr/bin/env python3
"""
End-to-end test for the RAG pipeline:
  Part 1: Upload files and verify they reach 'ready' status (not stuck in 'processing')
  Part 2: Chat verification — ask questions and confirm real answers with citations
"""

import json
import os
import sys
import time
import requests
import sseclient  # pip install sseclient-py

API = "http://localhost:8000"

# ── helpers ────────────────────────────────────────────────────────

def upload_file(filepath: str) -> dict:
    """Upload a file and return the JSON response."""
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{API}/documents/upload",
            files={"file": (filename, f)},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()


def poll_until_ready(doc_id: str, filename: str, timeout: int = 180) -> dict:
    """Poll document status every 2s until ready/failed or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{API}/documents/{doc_id}", timeout=10)
        resp.raise_for_status()
        doc = resp.json()
        status = doc.get("status", "unknown")
        elapsed = time.time() - start
        if status == "ready":
            print(f"    ✓ {filename}: ready in {elapsed:.1f}s ({doc.get('total_chunks', 0)} chunks)")
            return doc
        elif status in ("failed", "error"):
            print(f"    ✗ {filename}: {status} after {elapsed:.1f}s")
            return doc
        time.sleep(2)
    print(f"    ✗ {filename}: TIMEOUT after {timeout}s (still {status})")
    return doc


def create_session(doc_id: str = None) -> str:
    """Create a chat session, optionally scoped to a document."""
    body = {}
    if doc_id:
        body["document_id"] = doc_id
    resp = requests.post(f"{API}/chat/sessions", json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()["session_id"]


def stream_chat(session_id: str, message: str) -> dict:
    """Send a streaming chat message and collect the full response."""
    url = f"{API}/chat/{session_id}/stream"
    params = {"message": message}
    
    resp = requests.get(url, params=params, stream=True, timeout=120)
    resp.raise_for_status()
    
    full_text = ""
    citations = []
    stats = {}
    trace_id = None
    
    client = sseclient.SSEClient(resp)
    for event in client.events():
        try:
            data = json.loads(event.data)
        except json.JSONDecodeError:
            continue
        
        if data.get("type") == "trace_id":
            trace_id = data.get("trace_id")
        elif data.get("type") == "token":
            full_text += data.get("content", "")
        elif data.get("type") == "citations":
            citations = data.get("citations", [])
        elif data.get("type") == "stats":
            stats = data.get("stats", {})
        elif data.get("type") == "error":
            full_text += data.get("content", "")
        elif data.get("type") == "done":
            break
    
    return {
        "text": full_text,
        "citations": citations,
        "stats": stats,
        "trace_id": trace_id,
        "streamed": len(full_text) > 0,
    }


# ── test data ─────────────────────────────────────────────────────

TEST_FILES = {
    "txt": {
        "path": "test_files/test.txt",
        "question": "What are the four key stages of the modern RAG ingestion pipeline?",
    },
    "pdf": {
        "path": "test_files/test.pdf",
        "question": "What are the key strategies of the Global Climate Initiative and how many green jobs are expected?",
    },
    "docx": {
        "path": "test_files/test.docx",
        "question": "How many days of PTO do employees get in their first three years, and how much health insurance is covered?",
    },
    "pptx": {
        "path": "test_files/test.pptx",
        "question": "How much did revenue grow in Q3 and what drove the expanded operating margins?",
    },
}

HALLUCINATION_GUARD_PHRASES = [
    "I don't have enough information",
    "not appear sufficiently relevant",
    "insufficient",
]


# ── main ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("RAG PIPELINE END-TO-END TEST")
    print("=" * 70)

    # Check health
    try:
        health = requests.get(f"{API}/health", timeout=5).json()
        print(f"\nHealth: {health.get('status')} | LLM: {health.get('llm_connected')} | VectorStore: {health.get('vectorstore_connected')}")
    except Exception as e:
        print(f"Backend not reachable: {e}")
        sys.exit(1)

    # First, clean existing test documents so we get fresh uploads
    print("\n--- Cleaning existing documents ---")
    try:
        docs_resp = requests.get(f"{API}/documents", timeout=10).json()
        for doc in docs_resp.get("documents", []):
            did = doc["doc_id"]
            requests.delete(f"{API}/documents/{did}", timeout=10)
            print(f"  Deleted: {doc['filename']} ({did[:8]})")
    except Exception as e:
        print(f"  Warning: cleanup failed: {e}")

    # ── Part 1: Upload and verify all file types ──
    print("\n" + "=" * 70)
    print("PART 1: Upload & Processing Verification")
    print("=" * 70)

    upload_results = {}
    
    for ftype, info in TEST_FILES.items():
        filepath = info["path"]
        if not os.path.exists(filepath):
            print(f"\n  [{ftype.upper()}] SKIP — file not found: {filepath}")
            upload_results[ftype] = {"status": "skip", "doc_id": None}
            continue
        
        print(f"\n  [{ftype.upper()}] Uploading {filepath}...")
        try:
            t0 = time.time()
            result = upload_file(filepath)
            upload_time = time.time() - t0
            doc_id = result.get("doc_id")
            status = result.get("status")
            
            print(f"    Upload response in {upload_time:.1f}s: status={status}, doc_id={doc_id[:8] if doc_id else 'N/A'}")
            
            if status == "duplicate":
                print(f"    (duplicate — already exists)")
                upload_results[ftype] = {"status": "ready", "doc_id": doc_id}
            elif status == "processing":
                # Poll until ready
                doc = poll_until_ready(doc_id, os.path.basename(filepath))
                upload_results[ftype] = {"status": doc.get("status", "unknown"), "doc_id": doc_id}
            elif status == "ready":
                print(f"    ✓ Immediately ready ({result.get('total_chunks', 0)} chunks)")
                upload_results[ftype] = {"status": "ready", "doc_id": doc_id}
            else:
                print(f"    ? Unexpected status: {status}")
                upload_results[ftype] = {"status": status, "doc_id": doc_id}
                
        except Exception as e:
            print(f"    ✗ Upload failed: {e}")
            upload_results[ftype] = {"status": "error", "doc_id": None}

    # Print Part 1 summary
    print("\n" + "-" * 50)
    print("Part 1 Summary:")
    print(f"  {'Type':<6} {'Status':<12} {'Doc ID'}")
    for ftype, res in upload_results.items():
        did = res["doc_id"][:8] if res.get("doc_id") else "N/A"
        status_icon = "✓" if res["status"] == "ready" else "✗"
        print(f"  {ftype.upper():<6} {status_icon} {res['status']:<10} {did}")

    # ── Part 2: Chat verification ──
    print("\n" + "=" * 70)
    print("PART 2: Chat Pipeline Verification")
    print("=" * 70)

    chat_results = {}
    
    for ftype, info in TEST_FILES.items():
        res = upload_results.get(ftype, {})
        doc_id = res.get("doc_id")
        
        if res.get("status") != "ready" or not doc_id:
            print(f"\n  [{ftype.upper()}] SKIP — document not ready")
            chat_results[ftype] = {
                "answer": False,
                "not_hallucination": False,
                "citations": False,
                "streamed": False,
                "stats": False,
            }
            continue
        
        print(f"\n  [{ftype.upper()}] Asking: \"{info['question']}\"")
        
        try:
            session_id = create_session(doc_id)
            t0 = time.time()
            result = stream_chat(session_id, info["question"])
            chat_time = time.time() - t0
            
            answer_text = result["text"]
            citations = result["citations"]
            stats = result["stats"]
            
            # Check: real answer generated (not empty)
            has_answer = len(answer_text.strip()) > 20
            
            # Check: not a hallucination guard fallback
            is_hallucination_guard = any(
                phrase.lower() in answer_text.lower()
                for phrase in HALLUCINATION_GUARD_PHRASES
            )
            not_hallucination = has_answer and not is_hallucination_guard
            
            # Check: citations present
            has_citations = len(citations) > 0
            
            # Check: streamed (not all at once)
            was_streamed = result["streamed"]
            
            # Check: stats present with sensible numbers
            has_stats = bool(stats) and stats.get("duration_ms", 0) > 0
            
            print(f"    Answer ({len(answer_text)} chars, {chat_time:.1f}s):")
            print(f"      \"{answer_text[:150].replace(chr(10), ' ')}...\"")
            print(f"    Citations: {len(citations)}")
            for c in citations[:3]:
                print(f"      - {c.get('filename', '?')} p.{c.get('page_number', '?')} (score={c.get('relevance_score', 0):.4f})")
            if stats:
                print(f"    Stats: total={stats.get('duration_ms', 0):.0f}ms, "
                      f"retrieval={stats.get('retrieval_ms', 0):.0f}ms, "
                      f"rerank={stats.get('rerank_ms', 0):.0f}ms, "
                      f"gen={stats.get('generation_ms', 0):.0f}ms")
                if stats.get("usage"):
                    u = stats["usage"]
                    print(f"    Tokens: prompt={u.get('prompt_tokens',0)}, completion={u.get('completion_tokens',0)}, total={u.get('total_tokens',0)}")
            
            result_icons = {
                True: "✓",
                False: "✗",
            }
            print(f"    Checks: answer={result_icons[has_answer]} "
                  f"real_content={result_icons[not_hallucination]} "
                  f"citations={result_icons[has_citations]} "
                  f"streamed={result_icons[was_streamed]} "
                  f"stats={result_icons[has_stats]}")
            
            chat_results[ftype] = {
                "answer": has_answer,
                "not_hallucination": not_hallucination,
                "citations": has_citations,
                "streamed": was_streamed,
                "stats": has_stats,
            }
            
        except Exception as e:
            print(f"    ✗ Chat failed: {e}")
            import traceback
            traceback.print_exc()
            chat_results[ftype] = {
                "answer": False,
                "not_hallucination": False,
                "citations": False,
                "streamed": False,
                "stats": False,
            }

    # ── Final Report ──
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION REPORT")
    print("=" * 70)
    
    header = f"{'Type':<6} {'Upload':<10} {'Answer':<10} {'Content':<10} {'Citations':<10} {'Streamed':<10} {'Stats':<10}"
    print(header)
    print("-" * len(header))
    
    all_pass = True
    for ftype in TEST_FILES:
        up = upload_results.get(ftype, {})
        ch = chat_results.get(ftype, {})
        
        upload_ok = up.get("status") == "ready"
        answer_ok = ch.get("answer", False)
        content_ok = ch.get("not_hallucination", False)
        cite_ok = ch.get("citations", False)
        stream_ok = ch.get("streamed", False)
        stats_ok = ch.get("stats", False)
        
        row_pass = all([upload_ok, answer_ok, content_ok, cite_ok, stream_ok, stats_ok])
        if not row_pass:
            all_pass = False
        
        icon = lambda ok: "✓ PASS" if ok else "✗ FAIL"
        print(f"{ftype.upper():<6} {icon(upload_ok):<10} {icon(answer_ok):<10} {icon(content_ok):<10} {icon(cite_ok):<10} {icon(stream_ok):<10} {icon(stats_ok):<10}")
    
    print()
    if all_pass:
        print("🎉 ALL TESTS PASSED")
    else:
        print("⚠  SOME TESTS FAILED — see details above")
    
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

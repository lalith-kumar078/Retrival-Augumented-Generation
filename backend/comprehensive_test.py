"""Comprehensive end-to-end test for the RAG application."""

import requests
import json
import time
import sys
import os

BASE_URL = "http://localhost:8000"
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
INFO = "\033[94mℹ INFO\033[0m"

results = {"passed": 0, "failed": 0, "errors": []}


def test(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
        results["passed"] += 1
    else:
        print(f"  {FAIL} {name} — {detail}")
        results["failed"] += 1
        results["errors"].append(f"{name}: {detail}")


# ──────────────────────────────────────────────────────────────────
# 1. HEALTH & SYSTEM CHECKS
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. HEALTH & SYSTEM CHECKS")
print("=" * 60)

try:
    r = requests.get(f"{BASE_URL}/")
    test("Root endpoint returns 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    test("Root returns app name", data.get("name") == "RAG Agent API", f"got {data}")
except Exception as e:
    test("Root endpoint reachable", False, str(e))

try:
    r = requests.get(f"{BASE_URL}/health")
    test("Health endpoint returns 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    test("Health status is 'healthy' or 'degraded'", data.get("status") in ("healthy", "degraded"), f"got {data.get('status')}")
    test("Vectorstore connected", data.get("vectorstore_connected") is True, f"got {data.get('vectorstore_connected')}")
except Exception as e:
    test("Health endpoint reachable", False, str(e))

try:
    r = requests.get(f"{BASE_URL}/config")
    test("Config endpoint returns 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    test("Config has groq_model", "groq_model" in data, f"keys: {list(data.keys())}")
    test("Config has embedding_model", "embedding_model" in data)
except Exception as e:
    test("Config endpoint reachable", False, str(e))

# ──────────────────────────────────────────────────────────────────
# 2. DOCUMENT UPLOAD — ALL FILE TYPES
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DOCUMENT UPLOAD TESTS")
print("=" * 60)

# First, clear any previously uploaded test documents to avoid hash duplicates
try:
    r = requests.get(f"{BASE_URL}/documents")
    if r.status_code == 200:
        existing_docs = r.json().get("documents", [])
        for doc in existing_docs:
            did = doc["doc_id"]
            requests.delete(f"{BASE_URL}/documents/{did}")
        print(f"  {INFO} Cleaned {len(existing_docs)} existing documents")
except Exception:
    pass

uploaded_doc_ids = {}

test_files = {
    "txt": ("test_files/test.txt", "text/plain"),
    "pdf": ("test_files/test.pdf", "application/pdf"),
    "docx": ("test_files/test.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "pptx": ("test_files/test.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
}

for ftype, (fpath, content_type) in test_files.items():
    print(f"\n  --- Upload: {ftype.upper()} ---")
    if not os.path.exists(fpath):
        test(f"Upload {ftype}", False, f"File not found: {fpath}")
        continue

    try:
        with open(fpath, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/documents/upload",
                files={"file": (os.path.basename(fpath), f, content_type)},
                timeout=120,
            )

        test(f"Upload {ftype} returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")

        if r.status_code == 200:
            data = r.json()
            doc_id = data.get("doc_id")
            status = data.get("status")
            chunks = data.get("total_chunks", 0)

            test(f"Upload {ftype} has doc_id", bool(doc_id), f"got {doc_id}")
            test(f"Upload {ftype} status is 'ready' or 'duplicate'",
                 status in ("ready", "duplicate"),
                 f"got status='{status}', message='{data.get('message')}'")
            test(f"Upload {ftype} has chunks (or is duplicate)",
                 chunks > 0 or status == "duplicate",
                 f"got {chunks} chunks, status={status}")

            if doc_id:
                uploaded_doc_ids[ftype] = doc_id
                print(f"    {INFO} doc_id={doc_id}, status={status}, chunks={chunks}")
        else:
            try:
                err = r.json()
                print(f"    {INFO} Error detail: {err.get('detail', r.text[:200])}")
            except Exception:
                print(f"    {INFO} Raw error: {r.text[:200]}")

    except requests.exceptions.Timeout:
        test(f"Upload {ftype}", False, "Request timed out (>120s)")
    except Exception as e:
        test(f"Upload {ftype}", False, str(e))

# ──────────────────────────────────────────────────────────────────
# 3. DOCUMENT LISTING & RETRIEVAL
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. DOCUMENT LISTING & RETRIEVAL")
print("=" * 60)

try:
    r = requests.get(f"{BASE_URL}/documents")
    test("List documents returns 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    total = data.get("total", 0)
    docs = data.get("documents", [])
    test(f"Documents exist in store", total > 0, f"total={total}")
    print(f"  {INFO} Total documents: {total}")
    for doc in docs:
        st = doc.get("status")
        ch = doc.get("total_chunks", 0)
        print(f"    → {doc.get('filename')} | status={st} | chunks={ch}")
        test(f"Document '{doc.get('filename')}' has status 'ready'",
             st == "ready", f"got status='{st}'")
except Exception as e:
    test("List documents", False, str(e))

# Get individual document detail
for ftype, doc_id in uploaded_doc_ids.items():
    try:
        r = requests.get(f"{BASE_URL}/documents/{doc_id}")
        test(f"GET document detail for {ftype}", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            test(f"Document {ftype} detail has filename", bool(data.get("filename")))
    except Exception as e:
        test(f"GET document detail for {ftype}", False, str(e))

# ──────────────────────────────────────────────────────────────────
# 4. SEARCH / RETRIEVAL
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. SEARCH & RETRIEVAL")
print("=" * 60)

try:
    r = requests.post(
        f"{BASE_URL}/search",
        json={"query": "test document content", "top_k": 5},
        timeout=60,
    )
    test("Search returns 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        search_results = data.get("results", [])
        test("Search returns results", len(search_results) > 0, f"got {len(search_results)} results")
        print(f"  {INFO} Search returned {len(search_results)} results")
        for sr in search_results[:3]:
            print(f"    → chunk_id={sr.get('chunk_id', 'N/A')[:30]}... | score={sr.get('score', 0):.4f} | file={sr.get('filename')}")
except Exception as e:
    test("Search endpoint", False, str(e))

# Filtered search (scoped to a single document)
if uploaded_doc_ids:
    first_ftype = list(uploaded_doc_ids.keys())[0]
    first_doc_id = uploaded_doc_ids[first_ftype]
    try:
        r = requests.post(
            f"{BASE_URL}/search/filtered",
            json={"query": "test", "top_k": 5, "filters": {"doc_id": first_doc_id}},
            timeout=60,
        )
        test("Filtered search returns 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            results_list = data.get("results", [])
            test("Filtered search returns results", len(results_list) > 0, f"got {len(results_list)} results")
            # Verify all results belong to the filtered document
            all_match = all(sr.get("doc_id") == first_doc_id for sr in results_list)
            test("Filtered results scoped correctly", all_match, "some results from wrong doc")
    except Exception as e:
        test("Filtered search", False, str(e))

# ──────────────────────────────────────────────────────────────────
# 5. CHAT SESSION & Q&A
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. CHAT SESSION & Q&A")
print("=" * 60)

session_id = None
if uploaded_doc_ids:
    first_doc_id = list(uploaded_doc_ids.values())[0]
    # Create a chat session
    try:
        r = requests.post(
            f"{BASE_URL}/chat/sessions",
            json={"document_id": first_doc_id},
            timeout=30,
        )
        test("Create chat session returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
        if r.status_code == 200:
            data = r.json()
            session_id = data.get("session_id")
            test("Chat session has session_id", bool(session_id))
            print(f"  {INFO} session_id={session_id}")
    except Exception as e:
        test("Create chat session", False, str(e))

    # Send a message
    if session_id:
        try:
            r = requests.post(
                f"{BASE_URL}/chat/{session_id}/message",
                json={"message": "What is this document about?"},
                timeout=120,
            )
            test("Chat message returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:300]}")
            if r.status_code == 200:
                data = r.json()
                msg = data.get("message", {})
                test("Chat response has content", bool(msg.get("content")), "empty response")
                test("Chat response has role 'assistant'", msg.get("role") == "assistant", f"got {msg.get('role')}")
                test("Chat response has trace_id", bool(data.get("trace_id")))
                print(f"  {INFO} Response preview: {msg.get('content', '')[:150]}...")
                citations = msg.get("citations", [])
                print(f"  {INFO} Citations: {len(citations)}")
        except requests.exceptions.Timeout:
            test("Chat message", False, "Request timed out (>120s)")
        except Exception as e:
            test("Chat message", False, str(e))

    # List chat sessions
    try:
        r = requests.get(f"{BASE_URL}/chat/sessions", timeout=10)
        test("List sessions returns 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            sessions = r.json()
            test("Sessions list is not empty", len(sessions) > 0, f"got {len(sessions)} sessions")
    except Exception as e:
        test("List sessions", False, str(e))

# ──────────────────────────────────────────────────────────────────
# 6. UNSUPPORTED FILE TYPE (NEGATIVE TEST)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. NEGATIVE TESTS")
print("=" * 60)

try:
    import io
    fake_file = io.BytesIO(b"fake content")
    r = requests.post(
        f"{BASE_URL}/documents/upload",
        files={"file": ("bad.xyz", fake_file, "application/octet-stream")},
        timeout=30,
    )
    test("Unsupported file type returns 400", r.status_code == 400, f"got {r.status_code}")
except Exception as e:
    test("Unsupported file type rejection", False, str(e))

# Non-existent document
try:
    r = requests.get(f"{BASE_URL}/documents/nonexistent-id-12345")
    test("Non-existent doc returns 404", r.status_code == 404, f"got {r.status_code}")
except Exception as e:
    test("Non-existent doc 404", False, str(e))

# ──────────────────────────────────────────────────────────────────
# 7. DOCUMENT DELETE
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. DOCUMENT DELETE")
print("=" * 60)

if uploaded_doc_ids:
    # Delete one document and verify
    del_ftype = list(uploaded_doc_ids.keys())[0]
    del_doc_id = uploaded_doc_ids[del_ftype]
    try:
        r = requests.delete(f"{BASE_URL}/documents/{del_doc_id}", timeout=30)
        test(f"Delete document ({del_ftype}) returns 200", r.status_code == 200, f"got {r.status_code}")
        # Verify it's gone
        r2 = requests.get(f"{BASE_URL}/documents/{del_doc_id}")
        test(f"Deleted document returns 404", r2.status_code == 404, f"got {r2.status_code}")
    except Exception as e:
        test("Document delete", False, str(e))

# ──────────────────────────────────────────────────────────────────
# 8. USAGE ENDPOINT
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. USAGE & AUXILIARY ENDPOINTS")
print("=" * 60)

try:
    r = requests.get(f"{BASE_URL}/usage/stats", timeout=10)
    test("Usage /stats returns 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        test("Usage stats has daily_limit", "daily_limit" in data, f"keys: {list(data.keys())}")
        print(f"  {INFO} Usage stats: {json.dumps(data, indent=2)[:200]}")
except Exception as e:
    test("Usage endpoint", False, str(e))

# ──────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
total = results["passed"] + results["failed"]
print(f"  Total:  {total}")
print(f"  Passed: \033[92m{results['passed']}\033[0m")
print(f"  Failed: \033[91m{results['failed']}\033[0m")

if results["errors"]:
    print(f"\n  Failures:")
    for err in results["errors"]:
        print(f"    \033[91m✗\033[0m {err}")

print()
sys.exit(0 if results["failed"] == 0 else 1)

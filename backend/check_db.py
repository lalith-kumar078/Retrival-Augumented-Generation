import sqlite3

conn = sqlite3.connect("data/rag_store.db")
cursor = conn.cursor()

cursor.execute("SELECT doc_id, document_type, filename, status FROM documents")
print("Documents:")
for row in cursor.fetchall():
    print(row)

cursor.execute("SELECT doc_id, chunk_id, content FROM chunks")
print("\nChunks:")
for row in cursor.fetchall():
    print(f"Doc: {row[0][:8]}, Chunk ID: {row[1]}, Content: {repr(row[2])}")

cursor.execute("SELECT rowid, doc_id FROM chunks_fts")
print("\nFTS Chunks:")
for row in cursor.fetchall():
    print(row)

conn.close()

import requests
import json
import time

FILES = {
    "txt": "test_files/test.txt",
    "docx": "test_files/test.docx",
    "pdf": "test_files/test.pdf",
    "pptx": "test_files/test.pptx"
}

for ftype, path in FILES.items():
    print(f"\n--- Uploading {ftype} ---")
    try:
        with open(path, "rb") as f:
            res = requests.post("http://localhost:8000/documents/upload", files={"file": f})
        
        print(f"Status Code: {res.status_code}")
        print("Response:", json.dumps(res.json(), indent=2))
        
        if res.status_code == 200:
            doc_id = res.json().get("doc_id")
            if doc_id:
                time.sleep(1)
                doc_res = requests.get(f"http://localhost:8000/documents/{doc_id}")
                print("Document Status from GET:", json.dumps(doc_res.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

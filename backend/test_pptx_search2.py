import requests
import time
import json
from pprint import pprint

print("Uploading test2.pptx...")
with open("test_files/test2.pptx", "rb") as f:
    res = requests.post("http://localhost:8000/documents/upload", files={"file": f})

print("Upload Response:", res.status_code)
try:
    data = res.json()
    pprint(data)
except:
    print(res.text)

if res.status_code == 200:
    doc_id = data.get("doc_id")
    time.sleep(2)
    
    print("\nQuerying for the content...")
    search_res = requests.post("http://localhost:8000/search", json={"query": "new content"})
    print("Search Result:", search_res.status_code)
    try:
        pprint(search_res.json())
    except:
        print(search_res.text)


# Rag
# ragg
# ragg

# finallrag

## Backend Setup

This backend requires a Python 3.11 virtual environment due to dependency compatibility (e.g., numpy, pydantic-core).

**Important:** You MUST activate the virtual environment before running the server, installing packages, or running tests.

To activate the virtual environment:
```powershell
# From the backend directory
venv\Scripts\activate
```

After activation, you can run the server:
```powershell
uvicorn app.main:app --reload --port 8000
```

"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Groq API
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # SQLite-Vec
    sqlite_vec_path: str = "./data/rag_store.db"

    # Hallucination Guard
    disable_hallucination_guard: bool = False

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173"

    # Upload
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 50


    # Rate limiting
    rate_limit: str = "30/minute"

    # Limits
    groq_daily_token_limit: int = 500000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def sqlite_vec_abs_path(self) -> str:
        path = Path(self.sqlite_vec_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def upload_abs_path(self) -> str:
        path = Path(self.upload_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


settings = Settings()

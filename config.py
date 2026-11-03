"""
Configuration module for the AI Commercial Court Research Engine.
Loads application parameters from environment variables with sensible defaults.
Supports both local hosting and Vercel serverless deployment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")


class Config:
    """Application configuration values."""

    BASE_DIR = BASE_DIR

    # Vercel serverless environment detection
    IS_VERCEL = bool(os.getenv("VERCEL"))

    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "court-research-ai-default-secure-key")
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    PORT = int(os.getenv("PORT", 5000))

    # PostgreSQL Database URL
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/court_research_db"
    )

    # File Uploads (Uses /tmp on Vercel because root filesystem is read-only)
    UPLOAD_FOLDER = Path("/tmp/uploads") if IS_VERCEL else (BASE_DIR / os.getenv("UPLOAD_FOLDER", "uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB default
    ALLOWED_EXTENSIONS = {"pdf"}

    # AI / Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    FAISS_INDEX_PATH = (Path("/tmp/uploads/vector_index.faiss") if IS_VERCEL else (BASE_DIR / os.getenv("FAISS_INDEX_PATH", "uploads/vector_index.faiss")))
    CHUNK_MAP_PATH = BASE_DIR / "uploads/chunk_mapping.json"
    VECTORS_NPY_PATH = BASE_DIR / "uploads/vectors.npy"

    # Optional Hugging Face Cloud Inference (Free cloud embeddings for Vercel)
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY", "")

    # Text Chunking Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))  # approximate word/character length
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

    # Optional LLM integration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")

    @classmethod
    def ensure_upload_dir(cls):
        """Ensures the upload folder exists on disk."""
        try:
            cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

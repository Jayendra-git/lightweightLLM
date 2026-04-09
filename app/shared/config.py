from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "rag" / "data" / "docs"
CHROMA_DIR = ROOT_DIR / os.getenv("CHROMA_DIR", "rag/chroma")
DEFAULT_COLLECTION_NAME = "mini_rag_docs"
DEFAULT_BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_EMBED_MODEL_ID = os.getenv(
    "EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_ADAPTER_DIR = ROOT_DIR / os.getenv("LORA_ADAPTER_DIR", "artifacts/lora-persona")
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))


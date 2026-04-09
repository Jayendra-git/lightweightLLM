from __future__ import annotations

from pathlib import Path

import chromadb

from app.shared.config import (
    CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL_ID,
    DOCS_DIR,
)
from app.shared.hf_utils import load_sentence_transformer


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    cleaned = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(end - overlap, 0)
    return chunks


def load_documents(docs_dir: Path) -> list[dict]:
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for chunk_index, chunk in enumerate(chunk_text(text)):
            documents.append(
                {
                    "id": f"{path.stem}-{chunk_index}",
                    "source": path.name,
                    "text": chunk,
                }
            )
    return documents


def main() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_documents(DOCS_DIR)
    if not documents:
        raise RuntimeError(f"No markdown documents found in {DOCS_DIR}")

    embedder = load_sentence_transformer(DEFAULT_EMBED_MODEL_ID)
    embeddings = embedder.encode(
        [doc["text"] for doc in documents],
        normalize_embeddings=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(DEFAULT_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=DEFAULT_COLLECTION_NAME)
    collection.add(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[{"source": doc["source"]} for doc in documents],
        embeddings=embeddings,
    )
    print(
        f"Ingested {len(documents)} chunks from {len(list(DOCS_DIR.glob('*.md')))} markdown files into {CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()

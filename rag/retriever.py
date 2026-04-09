from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from app.shared.config import (
    CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL_ID,
    DOCS_DIR,
)
from app.shared.hf_utils import load_sentence_transformer
from rag.ingest import load_documents


@dataclass
class RetrievedChunk:
    source: str
    text: str
    distance: float

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "text": self.text,
            "distance": self.distance,
        }


class MiniRAGRetriever:
    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_id: str = DEFAULT_EMBED_MODEL_ID,
    ) -> None:
        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = load_sentence_transformer(embedding_model_id)
        self.docs_dir = DOCS_DIR

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        try:
            if self.collection.count() == 0:
                return self._retrieve_fallback(query, top_k)

            query_embedding = self.embedding_model.encode(
                [query], normalize_embeddings=True
            ).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            chunks = []
            for document, metadata, distance in zip(documents, metadatas, distances):
                chunks.append(
                    RetrievedChunk(
                        source=metadata.get("source", "unknown"),
                        text=document,
                        distance=distance,
                    )
                )
            return chunks
        except Exception as exc:
            print(f"Chroma retrieval failed, using in-memory fallback: {exc}")
            return self._retrieve_fallback(query, top_k)

    def _retrieve_fallback(self, query: str, top_k: int) -> list[RetrievedChunk]:
        documents = load_documents(self.docs_dir)
        if not documents:
            return []

        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )[0]
        doc_embeddings = self.embedding_model.encode(
            [doc["text"] for doc in documents],
            normalize_embeddings=True,
        )

        scored = []
        for doc, embedding in zip(documents, doc_embeddings):
            similarity = float((query_embedding * embedding).sum())
            scored.append(
                RetrievedChunk(
                    source=doc["source"],
                    text=doc["text"],
                    distance=1 - similarity,
                )
            )

        scored.sort(key=lambda chunk: chunk.distance)
        return scored[:top_k]

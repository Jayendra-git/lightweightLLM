from __future__ import annotations

from app.shared.config import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_TOP_K,
)
from app.shared.modeling import PersonaGenerator
from app.shared.prompting import build_messages
from rag.retriever import MiniRAGRetriever


class ChatService:
    def __init__(self) -> None:
        self.retriever = MiniRAGRetriever()
        self.generator = PersonaGenerator(
            model_id=DEFAULT_BASE_MODEL_ID,
            adapter_dir=DEFAULT_ADAPTER_DIR,
        )

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> dict:
        chunks = [chunk.as_dict() for chunk in self.retriever.retrieve(question, top_k)]
        messages = build_messages(question, chunks)
        result = self.generator.generate(messages)
        return {
            "answer": result.text,
            "sources": chunks,
            "adapter_loaded": result.adapter_loaded,
        }


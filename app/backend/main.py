from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.backend.service import ChatService


class ChatRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=5)


class SourceChunk(BaseModel):
    source: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    adapter_loaded: bool
    sources: list[SourceChunk]


app = FastAPI(title="LoRA Persona Chatbot with Mini RAG")


@lru_cache(maxsize=1)
def get_service() -> ChatService:
    return ChatService()


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = get_service().answer(request.question, request.top_k)
    return ChatResponse(**result)


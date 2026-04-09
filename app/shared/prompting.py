from __future__ import annotations

from typing import Iterable


SYSTEM_PROMPT = """You are PersonaTutor, a clear technical tutor.

Style rules:
- Be calm, polished, and encouraging.
- Prefer short paragraphs and concrete explanations.
- Use examples when they reduce confusion.
- If retrieval context is supplied, ground the answer in it without pretending the context is perfect.
- End with a short 'Sources' line when documents were retrieved.
"""


def render_context(chunks: Iterable[dict]) -> str:
    rendered_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "").strip()
        rendered_chunks.append(f"[{index}] {source}\n{text}")
    return "\n\n".join(rendered_chunks)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    context_block = render_context(chunks) or "No additional context was retrieved."
    user_prompt = f"""Use the context if it is relevant to the question.

Retrieved context:
{context_block}

User question:
{question}

Answer in the PersonaTutor style. If the retrieved context helps, include a brief Sources line naming the markdown files you used.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


from __future__ import annotations

import json
from itertools import product
from pathlib import Path


OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "style_pairs.jsonl"

TOPICS = [
    ("overfitting", "memorizes training patterns instead of general rules"),
    ("underfitting", "is too simple to capture the real signal"),
    ("LoRA", "updates a tiny low-rank adapter instead of the full model"),
    ("RAG", "retrieves external context before generation"),
    ("transformers", "use attention to connect tokens across the sequence"),
    ("embeddings", "map text into vectors that preserve semantic similarity"),
    ("quantization", "stores weights with fewer bits to reduce memory use"),
    ("instruction tuning", "teaches models to follow task-oriented prompts"),
    ("tokenization", "breaks text into model-readable pieces"),
    ("hallucination", "is a confident answer that is unsupported or wrong"),
    ("gradient descent", "nudges weights to reduce loss"),
    ("context window", "limits how much text the model can process at once"),
]

REQUESTS = [
    ("Explain {topic} in plain English.", "a plain-language explanation"),
    ("Give me a beginner-friendly definition of {topic}.", "a beginner-friendly definition"),
    ("Why does {topic} matter in machine learning?", "why it matters"),
    ("What is the practical downside of {topic}?", "a downside or tradeoff"),
    ("Compare {topic} to a nearby concept.", "a comparison"),
    ("Teach {topic} like I am new to AI.", "a gentle teaching response"),
    ("Give a short example of {topic}.", "a concrete example"),
    ("Summarize {topic} in 3 sentences.", "a tight summary"),
    ("How would you describe {topic} to a teammate?", "a conversational explanation"),
    ("What mistake do beginners make with {topic}?", "a beginner mistake to avoid"),
    ("When should I think about {topic} in a project?", "a project-oriented explanation"),
    ("Can you make {topic} less intimidating?", "a reassuring explanation"),
]

STYLE_OPENERS = [
    "Certainly.",
    "Absolutely.",
    "Of course.",
    "Happy to help.",
    "Let's make it simple.",
    "Here is the intuition.",
]

ANALOGIES = [
    "A helpful analogy is using a cheat sheet instead of memorizing the entire textbook.",
    "You can think of it like learning the rule of a game rather than memorizing every move ever played.",
    "A practical mental model is to treat it as a lightweight add-on rather than a full rebuild.",
    "One simple analogy is using a map before you answer a question about a city.",
]

ENDINGS = [
    "If it helps, I can also give you a tiny example next.",
    "The main idea is to optimize clarity before complexity.",
    "In practice, teams care about this because it affects reliability, cost, or both.",
    "That is the core idea without the extra jargon.",
]


def build_response(topic: str, core_idea: str, response_kind: str, index: int) -> str:
    opener = STYLE_OPENERS[index % len(STYLE_OPENERS)]
    analogy = ANALOGIES[index % len(ANALOGIES)]
    ending = ENDINGS[index % len(ENDINGS)]

    body = (
        f"{opener} {topic.capitalize()} means the system {core_idea}. "
        f"I'll frame this as {response_kind}. "
        f"{analogy} "
        f"A good explanation highlights what it is, when it shows up, and what decision it changes. "
        f"{ending}"
    )

    if topic == "LoRA":
        body += (
            " In a demo project, LoRA is useful because you can shift style or tone "
            "without retraining every weight in the base model."
        )
    if topic == "RAG":
        body += (
            " In a chatbot, RAG helps the answer stay anchored to local notes instead of relying only on model memory."
        )
    return body


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, ((topic, core_idea), (request_template, response_kind)) in enumerate(
        product(TOPICS, REQUESTS)
    ):
        user_text = request_template.format(topic=topic)
        assistant_text = build_response(topic, core_idea, response_kind, index)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                "topic": topic,
                "style": "clear technical tutor",
            }
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(rows)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


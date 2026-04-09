from __future__ import annotations

import argparse
from pathlib import Path

from app.shared.config import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL_ID
from app.shared.modeling import PersonaGenerator
from app.shared.prompting import build_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick local test for the tuned style.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    generator = PersonaGenerator(args.model_id, adapter_dir)
    result = generator.generate(build_messages(args.question, []))
    print(result.text)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from app.shared.hf_utils import load_causal_lm, load_tokenizer


def _preferred_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _render_messages_fallback(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = message["role"].capitalize()
        lines.append(f"{role}: {message['content']}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


@dataclass
class GenerationResult:
    text: str
    adapter_loaded: bool


class PersonaGenerator:
    def __init__(self, model_id: str, adapter_dir: Path | None = None) -> None:
        self.model_id = model_id
        self.adapter_dir = adapter_dir
        self.tokenizer = load_tokenizer(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"torch_dtype": _preferred_dtype()}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

        model = load_causal_lm(model_id, **model_kwargs)
        self.adapter_loaded = False

        if adapter_dir and adapter_dir.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
            self.adapter_loaded = True

        if not torch.cuda.is_available():
            model.to("cpu")

        self.model = model.eval()

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 220,
        temperature: float = 0.7,
    ) -> GenerationResult:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = _render_messages_fallback(messages)

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}

        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated[0][encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(text=text, adapter_loaded=self.adapter_loaded)

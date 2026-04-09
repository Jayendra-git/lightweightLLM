from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_with_local_fallback(loader, *args: Any, **kwargs: Any):
    if args and isinstance(args[0], str):
        local_path = _resolve_snapshot_path(args[0])
        if local_path is not None:
            return loader(local_path, *args[1:], **kwargs)

    local_kwargs = dict(kwargs)
    local_kwargs["local_files_only"] = True
    try:
        return loader(*args, **local_kwargs)
    except Exception:
        return loader(*args, **kwargs)


def _resolve_snapshot_path(model_id: str) -> str | None:
    try:
        return snapshot_download(repo_id=model_id, local_files_only=True)
    except Exception:
        return None


def load_sentence_transformer(model_id: str) -> SentenceTransformer:
    local_path = _resolve_snapshot_path(model_id)
    if local_path is not None:
        return SentenceTransformer(local_path)
    return SentenceTransformer(model_id)


def load_tokenizer(model_id: str):
    return _load_with_local_fallback(AutoTokenizer.from_pretrained, model_id)


def load_causal_lm(model_id: str, **kwargs: Any):
    return _load_with_local_fallback(AutoModelForCausalLM.from_pretrained, model_id, **kwargs)

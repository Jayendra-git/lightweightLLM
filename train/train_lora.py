from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from app.shared.config import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL_ID
from app.shared.hf_utils import load_causal_lm, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a style-only LoRA adapter.")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_BASE_MODEL_ID,
        help="Base instruct model from Hugging Face.",
    )
    parser.add_argument(
        "--data-path",
        default="train/data/style_pairs.jsonl",
        help="Path to the synthetic chat-format JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ADAPTER_DIR),
        help="Where to save the adapter weights.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def pick_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def format_example(messages: list[dict], tokenizer: AutoTokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    turns = []
    for message in messages:
        turns.append(f"{message['role'].capitalize()}: {message['content']}")
    return "\n\n".join(turns)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm(
        args.model_id,
        torch_dtype=pick_dtype(),
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=args.data_path, split="train")

    def tokenize_batch(batch: dict) -> dict:
        texts = [format_example(messages, tokenizer) for messages in batch["messages"]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    model.print_trainable_parameters()
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()

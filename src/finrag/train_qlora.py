from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from finrag.config import DATA_DIR


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TRAIN_FILE = DATA_DIR / "fine_tuning" / "finqa_train.jsonl"
DEFAULT_OUTPUT_DIR = DATA_DIR.parent / "outputs" / "qwen2_5_7b_finqa_lora"


def print_runtime_versions() -> None:
    import importlib.metadata as metadata

    packages = [
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "huggingface-hub",
    ]
    versions = []
    for package in packages:
        try:
            versions.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            versions.append(f"{package}=NOT INSTALLED")
    print("Runtime package versions:")
    for version in versions:
        print(f"  {version}")


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. In Colab, choose Runtime > Change runtime type > GPU, "
            "then rerun this script."
        )
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def load_messages_dataset(path: Path, eval_fraction: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if not path.exists():
        raise FileNotFoundError(
            f"Training file not found: {path}. Run `PYTHONPATH=src python -m finrag.fine_tuning` first."
        )
    dataset = load_dataset("json", data_files=str(path), split="train")
    if eval_fraction <= 0 or len(dataset) < 20:
        return dataset, None
    split = dataset.train_test_split(test_size=eval_fraction, seed=seed)
    return split["train"], split["test"]


def render_messages(example: dict[str, Any], tokenizer: Any) -> str:
    messages = example["messages"]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    rendered = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        rendered.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(rendered)


def tokenize_dataset(dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = render_messages(example, tokenizer)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def training_arguments_kwargs(args: argparse.Namespace, bf16: bool, has_eval: bool) -> dict[str, Any]:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "max_grad_norm": 0.3,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "fp16": not bf16,
        "bf16": bf16,
    }
    if has_eval:
        signature = inspect.signature(TrainingArguments)
        eval_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
        kwargs[eval_key] = "steps"
        kwargs["eval_steps"] = args.eval_steps
    return kwargs


def write_run_config(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "finrag_training_config.json"
    payload = vars(args).copy()
    payload["train_file"] = str(args.train_file)
    payload["output_dir"] = str(args.output_dir)
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    require_cuda()
    print_runtime_versions()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"Using compute dtype: {compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_messages_dataset(
        args.train_file,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    train_dataset = tokenize_dataset(train_dataset, tokenizer, args.max_length)
    if eval_dataset is not None:
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer, args.max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        **training_arguments_kwargs(args, bf16=bf16, has_eval=eval_dataset is not None)
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    write_run_config(args)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune a 7B instruct model for FinRAG.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

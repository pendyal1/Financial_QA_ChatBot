from __future__ import annotations

import argparse
import os
from pathlib import Path

from finrag.adapter_config import resolve_base_model
from finrag.answer import SYSTEM_PROMPT, build_context, print_response
from finrag.hallucination_detection import extract_citations, verify_answer
from finrag.models import RAGResponse
from finrag.sec_live import retrieve_live_sec


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Run this on a Colab GPU runtime.")


def generate_adapter_answer(
    question: str,
    adapter_path: Path | None,
    model_name: str | None,
    top_k: int,
    max_new_tokens: int,
    trust_remote_code: bool,
) -> RAGResponse:
    require_cuda()
    model_name = resolve_base_model(adapter_path, model_name)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    company, retrieved = retrieve_live_sec(question, top_k=top_k)
    context = build_context(retrieved, question=question)

    tokenizer_path = adapter_path if adapter_path and adapter_path.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if adapter_path and adapter_path.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}\n\nAnswer:"},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nEvidence:\n{context}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    citations = extract_citations(answer)
    verification = verify_answer(answer, retrieved, expected_tickers=[company.ticker])
    return RAGResponse(
        question=question,
        answer=answer,
        citations=citations,
        retrieved=retrieved,
        verification=verification,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer with a local Hugging Face LoRA adapter.")
    parser.add_argument("question")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=os.getenv("FINRAG_LORA_ADAPTER_PATH") or None,
        help="Optional path to the saved LoRA adapter directory. If omitted, use the base Qwen model.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Base model. If omitted, infer from adapter_config.json, then HF_BASE_MODEL.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_response(
        generate_adapter_answer(
            question=args.question,
            adapter_path=args.adapter_path,
            model_name=args.model_name,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
        )
    )


if __name__ == "__main__":
    main()

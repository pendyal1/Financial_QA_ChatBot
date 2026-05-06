from __future__ import annotations

import argparse
import os
import re

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from finrag.answer import SYSTEM_PROMPT
from finrag.config import DEFAULT_GENERATOR_MODEL


DEFAULT_MAX_INPUT_TOKENS = int(os.getenv("FINRAG_MAX_INPUT_TOKENS", "8192"))


class GenerateRequest(BaseModel):
    question: str
    context: str
    allowed_citations: list[str] = Field(default_factory=list)
    max_new_tokens: int = 350


class GenerateResponse(BaseModel):
    answer: str


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Run this server in a Colab GPU runtime.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def load_model(model_name: str, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require_cuda()
    token = hf_token()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    print(f"Loaded generator model: {model_name}")
    return tokenizer, model


def build_prompt(tokenizer, question: str, context: str, allowed_citations: list[str]) -> str:
    citation_rule = ""
    if allowed_citations:
        citation_rule = (
            "\nOnly use these citation IDs exactly: "
            + ", ".join(f"[{citation}]" for citation in allowed_citations)
        )
    user_content = (
        f"Question: {question}\n\n"
        f"Evidence:\n{context}\n\n"
        "Write 2-4 concise bullets that directly answer the question. "
        "Each bullet must contain a substantive claim in words plus one bracketed citation. "
        "Do not output only citation IDs. "
        "Do not include citations unless they support the words in the same bullet. "
        "Do not mention evidence outside the supplied context. "
        f"{citation_rule}\n\nAnswer:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{user_content}"


def clean_generation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"^(assistant|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def create_app(model_name: str, trust_remote_code: bool, max_input_tokens: int) -> FastAPI:
    tokenizer, model = load_model(model_name, trust_remote_code)
    app = FastAPI(title="FinRAG Open Finance GPU Server")

    @app.get("/health")
    def health() -> dict[str, str | int]:
        return {"status": "ok", "model": model_name, "max_input_tokens": max_input_tokens}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        prompt = build_prompt(
            tokenizer=tokenizer,
            question=request.question,
            context=request.context,
            allowed_citations=request.allowed_citations,
        )
        input_device = next(model.parameters()).device
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(input_device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        answer = clean_generation(tokenizer.decode(generated, skip_special_tokens=True))
        return GenerateResponse(answer=answer)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve DragonLLM/Qwen-Open-Finance-R-8B-FP8 for FinRAG generation on a GPU."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_GENERATOR_MODEL,
        help="Hugging Face CausalLM model to serve.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    app = create_app(
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
        max_input_tokens=args.max_input_tokens,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

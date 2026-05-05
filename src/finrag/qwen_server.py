from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from finrag.answer import SYSTEM_PROMPT


DEFAULT_MODEL = os.getenv("HF_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_ADAPTER = os.getenv("FINRAG_LORA_ADAPTER_PATH", "")


class GenerateRequest(BaseModel):
    question: str
    context: str
    allowed_citations: list[str] = []
    max_new_tokens: int = 350


class GenerateResponse(BaseModel):
    answer: str


class PassageItem(BaseModel):
    chunk_id: str
    score: float
    ticker: str
    company: str
    source: str
    source_url: str
    text: str


class HallucinateRequest(BaseModel):
    answer: str
    passages: list[PassageItem]


class ClaimItem(BaseModel):
    claim_text: str
    sentence_index: int
    is_numerical: bool
    label: str
    confidence: float
    supporting_chunk_ids: list[str]
    evidence_snippet: str


class HallucinateResponse(BaseModel):
    overall_risk: str
    confidence_score: float
    grounded_count: int
    partial_count: int
    unsupported_count: int
    claims: list[ClaimItem]


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found. Run this server in a Colab GPU runtime.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def load_model(model_name: str, adapter_path: str | None, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    require_cuda()
    adapter = Path(adapter_path).expanduser() if adapter_path else None
    tokenizer_path = adapter if adapter and adapter.exists() else model_name
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
    if adapter and adapter.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, adapter)
        print(f"Loaded LoRA adapter: {adapter}")
    else:
        model = base_model
        print("No LoRA adapter loaded; using base model.")

    model.eval()
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
        "Synthesize a direct answer in your own words — do not copy evidence verbatim. "
        "Write 2-4 concise sentences or bullets. "
        "Each sentence must make a substantive claim and end with one bracketed citation. "
        "Do not output only citation IDs. "
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
    text = re.sub(r"^(assistant|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def create_app(model_name: str, adapter_path: str | None, trust_remote_code: bool) -> FastAPI:
    tokenizer, model = load_model(model_name, adapter_path, trust_remote_code)
    app = FastAPI(title="FinRAG Qwen GPU Server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        prompt = build_prompt(
            tokenizer=tokenizer,
            question=request.question,
            context=request.context,
            allowed_citations=request.allowed_citations,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        answer = clean_generation(tokenizer.decode(generated, skip_special_tokens=True))
        return GenerateResponse(answer=answer)

    @app.post("/hallucinate", response_model=HallucinateResponse)
    def hallucinate(request: HallucinateRequest) -> HallucinateResponse:
        from finrag.hallucination import detect_hallucinations
        from finrag.models import RetrievalResult

        retrieved = [
            RetrievalResult(
                chunk_id=p.chunk_id,
                score=p.score,
                ticker=p.ticker,
                company=p.company,
                source=p.source,
                source_url=p.source_url,
                text=p.text,
            )
            for p in request.passages
        ]
        report = detect_hallucinations(request.answer, retrieved)
        return HallucinateResponse(
            overall_risk=report.overall_risk,
            confidence_score=report.confidence_score,
            grounded_count=report.grounded_count,
            partial_count=report.partial_count,
            unsupported_count=report.unsupported_count,
            claims=[
                ClaimItem(
                    claim_text=cv.claim.text,
                    sentence_index=cv.claim.sentence_index,
                    is_numerical=cv.claim.is_numerical,
                    label=cv.label,
                    confidence=cv.confidence,
                    supporting_chunk_ids=cv.supporting_chunk_ids,
                    evidence_snippet=cv.evidence_snippet,
                )
                for cv in report.claims
            ],
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Qwen 2.5 7B for FinRAG generation on a GPU.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    app = create_app(
        model_name=args.model_name,
        adapter_path=args.adapter_path or None,
        trust_remote_code=args.trust_remote_code,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

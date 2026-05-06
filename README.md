# FinRAG

Financial question answering over live SEC filings with retrieval-augmented generation.

This branch uses `DragonLLM/Qwen-Open-Finance-R-8B-FP8` as the generator. There is no fine-tuning and no LoRA adapter. The Streamlit app runs locally, retrieves SEC evidence for the company in the question, and sends the retrieved context to a Qwen server running on a Colab GPU.

## Architecture

- Front end: Streamlit (`demo/app.py`)
- Retrieval corpus: live SEC EDGAR APIs at question time
- Company resolution: public company name or ticker from SEC `company_tickers.json`
- Filing text: latest relevant `10-K`, `10-Q`, and `8-K` documents
- Numeric facts: SEC `companyfacts` API for common financial statement questions
- Retrieval: sentence-transformer embeddings over fresh filing chunks
- Reranking: `BAAI/bge-reranker-v2-m3`
- Generator: `DragonLLM/Qwen-Open-Finance-R-8B-FP8` served by `finrag.qwen_server`
- Verification: citation checks and hallucination risk scoring

The app no longer accepts uploaded filings. The user must mention a public company name or ticker in the question.

## Key Files

- `demo/app.py`: Streamlit UI
- `src/finrag/sec_live.py`: SEC company resolution, filing fetch, companyfacts fetch, chunking, retrieval
- `src/finrag/rerank.py`: cross-encoder reranking
- `src/finrag/qwen_server.py`: FastAPI GPU server for `DragonLLM/Qwen-Open-Finance-R-8B-FP8`
- `src/finrag/remote_qwen.py`: local client for the Colab endpoint
- `src/finrag/answer.py`: extractive debug fallback and citation-aware response assembly
- `scripts/colab_setup.sh`: Colab dependency setup
- `scripts/colab_start_qwen_server.sh`: starts the GPU model server
- `scripts/colab_start_ngrok.py`: exposes the Colab server through ngrok

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

Set a SEC User-Agent with your contact email:

```bash
export SEC_USER_AGENT="FinRAG academic research your_email@example.com"
```

Start the local UI:

```bash
streamlit run demo/app.py
```

In the UI, paste the public Colab endpoint URL into `Open Finance Qwen endpoint`.

## Colab GPU Setup

Use a CUDA-backed Colab runtime.

```bash
%cd /content/gdrive/MyDrive/project_folder/Financial_QA_ChatBot
!bash scripts/colab_setup.sh
```

The model is gated on Hugging Face, so your HF account must have access to `DragonLLM/Qwen-Open-Finance-R-8B-FP8`. Log in before starting the server:

```python
from huggingface_hub import login
login()
```

Start the model server:

```bash
!PYTHONPATH=src python -m finrag.qwen_server --port 8000
```

Or use the launcher:

```bash
!bash scripts/colab_start_qwen_server.sh
```

Verify it:

```bash
!curl -s http://127.0.0.1:8000/health
```

Expose the server with ngrok:

```python
import os
os.environ["NGROK_AUTHTOKEN"] = "PASTE_YOUR_NGROK_AUTH_TOKEN"
%run scripts/colab_start_ngrok.py
```

Paste the printed `https://...ngrok-free.app` URL into the Streamlit app.

## Environment Variables

- `SEC_USER_AGENT`: SEC-compliant app/contact header
- `COLAB_QWEN_ENDPOINT`: optional default endpoint for the Streamlit UI
- `FINRAG_GENERATOR_MODEL`: override generator model, defaults to `DragonLLM/Qwen-Open-Finance-R-8B-FP8`
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`: optional non-interactive Hugging Face token for the gated model
- `FINRAG_MAX_INPUT_TOKENS`: prompt truncation limit for the model server, defaults to `8192`
- `EMBEDDING_MODEL`: override retrieval embedding model
- `RERANKER_MODEL`: override reranker model

## Example Questions

```text
What risks did Apple report related to supply chains?
What does Microsoft report about AI-related risks?
What revenue did Amazon report?
What cybersecurity risks does Microsoft describe?
What capital expenditure trends did NVIDIA discuss?
```

## RAG Flow

1. Resolve the requested company from the question.
2. Fetch recent SEC filing metadata through the SEC submissions API.
3. Fetch filing text and cache it under `data/sec_cache`.
4. Fetch SEC companyfacts when the question asks for numeric financial-statement facts.
5. Chunk filing text.
6. Retrieve relevant chunks with embeddings and lexical/risk boosts.
7. Rerank the candidates.
8. Send the question, evidence, and allowed citation IDs to the Colab Qwen endpoint.
9. Verify generated citations against retrieved evidence.

## Debug Fallback

The Streamlit backend selector includes `Debug extractive fallback`. This does not call a model. It selects high-signal sentences from the retrieved SEC evidence and is useful for testing retrieval when the Colab GPU endpoint is unavailable.

## Benchmark Preparation

Prepare benchmark CSV files:

```bash
PYTHONPATH=src python -m finrag.benchmarks financebench
PYTHONPATH=src python -m finrag.benchmarks tatqa --split dev
```

Run FinanceBench evaluation with the extractive fallback:

```bash
PYTHONPATH=src python -m finrag.evaluate_benchmark \
  --input-csv data/evaluation/financebench_eval.csv \
  --backend extractive
```

Run it through the Colab Qwen endpoint:

```bash
PYTHONPATH=src python -m finrag.evaluate_benchmark \
  --input-csv data/evaluation/financebench_eval.csv \
  --backend remote-qwen \
  --endpoint https://YOUR-NGROK-URL.ngrok-free.app
```

## Current Limits

- Best on single-company questions.
- Not for real-time stock prices.
- Not for forecasting or investment advice.
- Long multi-company comparisons are not optimized yet.

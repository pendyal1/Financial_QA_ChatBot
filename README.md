# FinRAG

Financial QA over SEC filings, with live EDGAR retrieval, citation-grounded answers, and hallucination checks.

## Current Architecture

The repo now uses a different architecture than the original static-index prototype.

- Retrieval corpus: live SEC EDGAR APIs at question time
- Numeric facts: SEC `companyfacts` API when the question is financial-statement oriented
- Dense retrieval: sentence-transformer embeddings over freshly pulled filing chunks
- Reranking: `BAAI/bge-reranker-v2-m3`
- Generator: `Qwen/Qwen2.5-14B-Instruct`
- Fine-tuning: QLoRA on a curated training mix
- Local app: Streamlit on CPU
- Remote generation: Colab GPU endpoint

The interactive app no longer accepts uploaded files. It expects the user to mention a public company name or ticker in the question, then it pulls the relevant filing material directly from the SEC.

## Project Direction

FinRAG is meant to act like a research assistant for financial analysts:

- pull relevant SEC filings
- answer questions with evidence-backed citations
- expose the retrieved evidence
- flag weakly supported answers

Primary evaluation target:

- FinanceBench

Secondary reasoning datasets:

- FinQA
- ConvFinQA
- TAT-QA

## Data Sources

Training and evaluation data:

- FinQA: https://huggingface.co/datasets/ibm-research/finqa
- ConvFinQA: https://huggingface.co/datasets/AdaptLLM/ConvFinQA
- FinanceBench: https://huggingface.co/datasets/PatronusAI/financebench
- TAT-QA: https://github.com/NExTplusplus/TAT-QA

Live retrieval APIs:

- SEC EDGAR API overview: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- Company submissions: `https://data.sec.gov/submissions/CIK##########.json`
- Company facts: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`

## Repo Structure

Key modules:

- `src/finrag/sec_live.py`: live SEC company resolution, submissions fetch, companyfacts fetch, chunking, and retrieval
- `src/finrag/rerank.py`: cross-encoder reranking with `BAAI/bge-reranker-v2-m3`
- `src/finrag/answer.py`: extractive fallback, answer assembly, hallucination verification
- `src/finrag/remote_qwen.py`: local client for the Colab Qwen endpoint
- `src/finrag/qwen_server.py`: FastAPI GPU generation server
- `src/finrag/fine_tuning.py`: curated fine-tuning mixture builder
- `src/finrag/train_qlora.py`: QLoRA fine-tuning entrypoint
- `src/finrag/benchmarks.py`: benchmark preparation helpers
- `src/finrag/evaluate_benchmark.py`: benchmark evaluation with gold evidence
- `demo/app.py`: local Streamlit UI

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run everything from the repo root:

```bash
export PYTHONPATH=src
```

## Streamlit Demo

Start the local app:

```bash
streamlit run demo/app.py
```

The app asks for:

- a financial question
- retrieved chunk count
- either:
  - `Colab GPU Qwen endpoint`, or
  - `Debug extractive fallback`

Example questions:

```text
What risks did Apple report related to supply chains?
What does Microsoft report about AI-related risks?
What revenue did Amazon report?
What cybersecurity risks does Microsoft describe?
```

Important constraint:

- mention a public company or ticker in the question

## Live Retrieval Flow

For each question, the app:

1. resolves the company from the question
2. fetches the latest relevant SEC filings from EDGAR
3. fetches SEC `companyfacts` for numeric questions
4. chunks filing text
5. runs dense retrieval
6. reranks with `BAAI/bge-reranker-v2-m3`
7. sends retrieved evidence to either:
   - the local extractive fallback, or
   - the Colab Qwen endpoint
8. verifies citations and assigns hallucination risk

## Build The Curated Training Mix

Prepare the multi-dataset fine-tuning file:

```bash
PYTHONPATH=src python -m finrag.fine_tuning
```

This writes:

- `data/fine_tuning/financial_qa_mix_train.jsonl`
- `data/fine_tuning/financial_qa_mix_manifest.json`

Optional dataset limits:

```bash
PYTHONPATH=src python -m finrag.fine_tuning \
  --finqa-limit 3000 \
  --convfinqa-limit 6000 \
  --tatqa-limit 4000
```

## QLoRA Training On Colab

Use a CUDA-backed Google Colab runtime.

Install the Colab stack:

```bash
%cd /content/gdrive/MyDrive/project_folder/Financial_QA_ChatBot
!bash scripts/colab_setup.sh
```

Build the curated training file in Colab:

```bash
!PYTHONPATH=src python -m finrag.fine_tuning
```

Train the LoRA adapter:

```bash
!PYTHONPATH=src python -m finrag.train_qlora \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --train-file data/fine_tuning/financial_qa_mix_train.jsonl \
  --output-dir /content/gdrive/MyDrive/finrag-adapters/qwen2_5_14b_financial_qa_lora \
  --epochs 1 \
  --max-length 1536 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32
```

The default training script now expects:

- base model: `Qwen/Qwen2.5-14B-Instruct`
- train file: `data/fine_tuning/financial_qa_mix_train.jsonl`

## Serve Qwen From Colab

Start the GPU server:

```bash
!bash scripts/colab_start_qwen_server.sh
```

The server launcher now defaults to:

- model: `Qwen/Qwen2.5-14B-Instruct`
- adapter: `/content/gdrive/MyDrive/finrag-adapters/qwen2_5_14b_financial_qa_lora`

Verify the server:

```bash
!curl -s http://127.0.0.1:8000/health
```

Expose it publicly with ngrok:

```python
import os
os.environ["NGROK_AUTHTOKEN"] = "PASTE_YOUR_NGROK_AUTH_TOKEN"
%run scripts/colab_start_ngrok.py
```

Paste the printed `https://...ngrok-free.app` URL into the Streamlit app.

## Benchmark Preparation

Prepare benchmark CSV files:

```bash
PYTHONPATH=src python -m finrag.benchmarks financebench
PYTHONPATH=src python -m finrag.benchmarks tatqa --split dev
```

Run FinanceBench evaluation:

```bash
PYTHONPATH=src python -m finrag.evaluate_benchmark \
  --input-csv data/evaluation/financebench_eval.csv \
  --backend extractive
```

Or evaluate through the remote Qwen endpoint:

```bash
PYTHONPATH=src python -m finrag.evaluate_benchmark \
  --input-csv data/evaluation/financebench_eval.csv \
  --backend remote-qwen \
  --endpoint https://YOUR-NGROK-URL.ngrok-free.app
```

## What Changed

Compared with the earlier prototype, this repo now:

- removes the upload workflow from the UI
- removes reliance on the local FAISS index for the interactive app
- pulls SEC filings live from EDGAR
- consults `companyfacts` for numeric questions
- upgrades the generator default from Qwen 2.5 7B to Qwen 2.5 14B
- replaces FinQA-only fine-tuning prep with a curated FinQA + ConvFinQA + TAT-QA mixture
- adds a stronger reranker

## Current Limits

- best on single-company questions
- not for real-time stock prices
- not for forecasting or investment advice
- not yet optimized for long multi-company comparisons in a single query

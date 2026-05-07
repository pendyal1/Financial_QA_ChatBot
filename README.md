# FinRAG

FinRAG is a retrieval-augmented question-answering system for public-company SEC filings. It answers natural language financial questions by pulling filing content from SEC EDGAR at query time, retrieving and reranking relevant evidence, generating cited answers, and assigning a hallucination-risk label.

Course project for STAT GR5293, Generative AI Using LLMs, Columbia University, Spring 2026.

## Deliverables

- `FinRAG_Report.pdf`: final report with method, architecture, case studies, ablations, limitations, and future work.
- `FinRAG_Slides.pdf`: final presentation slides.
- `FinRAG_Demo.mp4`: demo recording.
- `notebooks/finetune_qwen7b_colab.ipynb`: Colab workflow used for QLoRA training and GPU endpoint serving.
- Fine-tuned LoRA adapter: https://drive.google.com/drive/folders/1wryt5WIhQ32U-AUfkCa0mG8u-FuCgBDL?usp=sharing

The fine-tuned LoRA adapter is too large to include in GitHub. Download it from the Google Drive link above when reproducing the Qwen + LoRA system.

## System Summary

The current implementation is a live SEC retrieval system, not a static uploaded-document prototype.

- Retrieval corpus: SEC EDGAR filings fetched at question time.
- Numeric facts: SEC `companyfacts` API for financial-statement questions.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Reranker: `BAAI/bge-reranker-v2-m3`.
- Generator: `Qwen/Qwen2.5-7B-Instruct`, optionally with the fine-tuned LoRA adapter.
- Fine-tuning: QLoRA on a curated FinQA + ConvFinQA + TAT-QA mixture.
- Local UI: Streamlit on CPU.
- Remote generation: FastAPI server on a Google Colab GPU runtime, exposed with Cloudflare Quick Tunnel or ngrok.
- Fallback backend: extractive answer generation that runs fully locally without a GPU.

The app expects each question to mention a public company name or ticker. It then resolves the company to an SEC CIK and fetches relevant SEC material directly.

## Repository Structure

```text
.
|-- demo/app.py                         # Streamlit app
|-- notebooks/finetune_qwen7b_colab.ipynb
|-- scripts/
|   |-- colab_setup.sh                  # Colab dependency setup
|   |-- colab_start_qwen_server.py      # Starts FastAPI Qwen server
|   |-- colab_start_qwen_server.sh
|   |-- colab_start_ngrok.py
|   |-- run_ablation.py
|   `-- compare_results.py
|-- src/finrag/
|   |-- sec_live.py                     # SEC lookup, filing fetch, chunking, retrieval
|   |-- rerank.py                       # BGE cross-encoder reranking
|   |-- answer.py                       # Extractive fallback and response assembly
|   |-- remote_qwen.py                  # Client for Colab Qwen endpoint
|   |-- qwen_server.py                  # FastAPI GPU generation server
|   |-- fine_tuning.py                  # Curated fine-tuning data builder
|   |-- train_qlora.py                  # QLoRA training entrypoint
|   |-- evaluate_benchmark.py           # Benchmark evaluation
|   `-- hallucination_detection.py      # Citation and support checks
|-- data/evaluation/                    # Sample questions and saved ablation outputs
|-- tests/                              # Unit tests
|-- requirements.txt                    # Local CPU/demo dependencies
`-- requirements-colab.txt              # Colab GPU dependencies
```

## Data Sources

Training and evaluation datasets:

- FinQA: https://huggingface.co/datasets/ibm-research/finqa
- ConvFinQA: https://huggingface.co/datasets/AdaptLLM/ConvFinQA
- FinanceBench: https://huggingface.co/datasets/PatronusAI/financebench
- TAT-QA: https://github.com/NExTplusplus/TAT-QA

Live retrieval APIs:

- SEC EDGAR API overview: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- Company submissions: `https://data.sec.gov/submissions/CIK##########.json`
- Company facts: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`

## Quick Reproduction Path

Use this path to reproduce the demo with the least setup.

1. Set up the local Python environment.
2. Run the Streamlit app locally with the extractive fallback to confirm retrieval works.
3. Open the Colab notebook with a GPU runtime.
4. Install Colab dependencies and download the LoRA adapter from Google Drive.
5. Start the Qwen FastAPI server in Colab.
6. Expose the Colab server with Cloudflare Quick Tunnel or ngrok.
7. Paste the public endpoint into the local Streamlit app and ask one of the sample questions.

## Local Setup

Use Python 3.10 or newer.

```bash
git clone https://github.com/pendyal1/Financial_QA_ChatBot.git
cd Financial_QA_ChatBot

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

Create a local environment file:

```bash
cp .env.example .env
```

Edit `.env` and set a real SEC user agent:

```text
SEC_USER_AGENT="FinRAG academic research your_email@example.com"
COLAB_QWEN_ENDPOINT=""
```

The SEC asks automated clients to identify themselves. The default placeholder works for code inspection, but reproducible SEC requests should use your own contact email.

Run the unit tests:

```bash
python -m unittest
```

Run the local app:

```bash
streamlit run demo/app.py
```

In the app, choose `Debug extractive fallback` to run without a GPU endpoint. This reproduces the live SEC retrieval, reranking, citation display, and hallucination-risk checks on CPU.

## Sample Questions

More examples are in `example_questions.txt` and `data/evaluation/sample_questions.csv`.

```text
What risks did Apple report related to supply chains?
What cybersecurity risks does Microsoft describe?
What revenue did Amazon report?
What does NVIDIA report about demand or supply risks?
What risks does Tesla describe around manufacturing or production?
```

Avoid real-time stock-price, forecasting, or investment-advice questions. FinRAG is designed for filing-grounded QA, not market-data lookup.

## Reproduce The Colab GPU Endpoint

The Streamlit app should run locally on your laptop or desktop. Colab is used only to host Qwen 2.5 7B on a GPU.

Open `notebooks/finetune_qwen7b_colab.ipynb` in Google Colab and choose:

```text
Runtime > Change runtime type > GPU
```

The notebook begins with a CUDA check:

```python
import torch
assert torch.cuda.is_available(), "Choose Runtime > Change runtime type > GPU before running."
print(torch.cuda.get_device_name(0))
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Set the project path to wherever this repository is stored in Drive:

```python
PROJECT_DIR = "/content/drive/MyDrive/Financial_QA_ChatBot"
%cd $PROJECT_DIR
```

Install the Colab dependency stack:

```bash
!bash scripts/colab_setup.sh
```

This script keeps Colab's preinstalled CUDA-enabled `torch`, removes optional packages that caused version conflicts during our run, and installs the text-generation stack used by the project.

Download the LoRA adapter from Google Drive:

```text
https://drive.google.com/drive/folders/1wryt5WIhQ32U-AUfkCa0mG8u-FuCgBDL?usp=sharing
```

Place the adapter folder at the Colab Drive path used by the notebook:

```text
/content/drive/MyDrive/finrag-adapters/qwen2_5_7b_financial_qa_lora
```

Set `ADAPTER_PATH` before starting the server. This keeps the adapter path explicit and avoids confusion with older Colab mounts that used `/content/gdrive`.

```python
import os
os.environ["ADAPTER_PATH"] = "/content/drive/MyDrive/finrag-adapters/qwen2_5_7b_financial_qa_lora"
```

Start the GPU generation server:

```bash
!ADAPTER_PATH=/content/drive/MyDrive/finrag-adapters/qwen2_5_7b_financial_qa_lora bash scripts/colab_start_qwen_server.sh
```

The launcher starts `finrag.qwen_server` on port `8000`, loads `Qwen/Qwen2.5-7B-Instruct`, and loads the LoRA adapter if the adapter directory exists. Verify the server:

```bash
!curl -s http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok","model":"Qwen/Qwen2.5-7B-Instruct"}
```

## Expose The Colab Server With Cloudflare

Use Cloudflare Quick Tunnel to create a public HTTPS URL for the Colab Qwen server:

```bash
%%bash
pkill -f cloudflared || true
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
nohup ./cloudflared tunnel --url http://127.0.0.1:8000 > cloudflared.log 2>&1 &
sleep 8
grep -o 'https://[-a-zA-Z0-9.]*trycloudflare.com' cloudflared.log | tail -1
```

Copy the printed `https://...trycloudflare.com` URL. Keep the Colab runtime alive while using the endpoint.

Run the local Streamlit app and paste that URL into the `Colab Qwen endpoint` field:

```bash
export PYTHONPATH=src
COLAB_QWEN_ENDPOINT="https://YOUR-TUNNEL.trycloudflare.com" streamlit run demo/app.py
```

You can also leave `COLAB_QWEN_ENDPOINT` empty and paste the endpoint directly in the app.

## ngrok Fallback

If Cloudflare Quick Tunnel returns a 500 error or does not provide a URL, use ngrok instead. Get a free auth token from the ngrok dashboard, then run this in Colab:

```python
import os
os.environ["NGROK_AUTHTOKEN"] = "PASTE_YOUR_NGROK_AUTH_TOKEN"
%run scripts/colab_start_ngrok.py
```

Copy the printed `https://...ngrok-free.app` URL into the Streamlit app.

## Optional: Share The Streamlit App With Cloudflare

For the class demo, the app itself can also be shared with a Cloudflare tunnel. This is separate from the Colab Qwen endpoint tunnel.

Start Streamlit locally:

```bash
export PYTHONPATH=src
COLAB_QWEN_ENDPOINT="https://YOUR-QWEN-ENDPOINT" \
  streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```

In another terminal, expose the Streamlit port:

```bash
cloudflared tunnel --url http://127.0.0.1:8501
```

Use the printed `https://...trycloudflare.com` URL to open the Streamlit UI from another browser or machine. This tunnel exposes your local Streamlit session while the command is running.

On macOS, `cloudflared` can be installed with Homebrew:

```bash
brew install cloudflared
```

## Reproduce Fine-Tuning

If you want to regenerate the adapter instead of using the shared Google Drive artifact, run these commands in the Colab GPU notebook.

Build the curated training file:

```bash
!PYTHONPATH=src python -m finrag.fine_tuning
```

This writes:

- `data/fine_tuning/financial_qa_mix_train.jsonl`
- `data/fine_tuning/financial_qa_mix_manifest.json`

Optional smaller run:

```bash
!PYTHONPATH=src python -m finrag.fine_tuning \
  --finqa-limit 3000 \
  --convfinqa-limit 6000 \
  --tatqa-limit 4000
```

Train the QLoRA adapter:

```bash
!PYTHONPATH=src python -m finrag.train_qlora \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --train-file data/fine_tuning/financial_qa_mix_train.jsonl \
  --output-dir /content/drive/MyDrive/finrag-adapters/qwen2_5_7b_financial_qa_lora \
  --epochs 1 \
  --max-length 1536 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32
```

The report used this configuration:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Method: 4-bit QLoRA
- LoRA rank: `16`
- LoRA alpha: `32`
- Max sequence length: `1536`
- Learning rate: `2e-4`
- Batch size: `1`
- Gradient accumulation: `16`
- Epochs: `1`
- Hardware: Google Colab GPU runtime, T4 or A100

Test the saved adapter in Colab:

```bash
!PYTHONPATH=src python -m finrag.hf_adapter_answer \
  --adapter-path /content/drive/MyDrive/finrag-adapters/qwen2_5_7b_financial_qa_lora \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  "What risks did Apple report related to supply chains?"
```

## Reproduce Benchmark And Ablation Results

Saved ablation outputs are included in `data/evaluation/`:

- `ablation_extractive.json`
- `ablation_qwen_base.json`
- `ablation_qwen_lora.json`
- `ablation_retrieval_no_reranker.json`
- `ablation_retrieval_reranker.json`
- `ablation_summary.csv`

Build benchmark CSV files:

```bash
PYTHONPATH=src python -m finrag.benchmarks financebench
PYTHONPATH=src python -m finrag.benchmarks tatqa --split dev
```

Run FinanceBench evaluation with the local extractive backend:

```bash
PYTHONPATH=src python -m finrag.evaluate_benchmark \
  --input-csv data/evaluation/financebench_eval.csv \
  --backend extractive
```

Run evaluation through the Colab Qwen endpoint:

```bash
PYTHONPATH=src python -m finrag.evaluate_benchmark \
  --input-csv data/evaluation/financebench_eval.csv \
  --backend remote-qwen \
  --endpoint https://YOUR-QWEN-ENDPOINT
```

Regenerate the ablation files:

```bash
# System A: extractive fallback, no GPU endpoint needed
PYTHONPATH=src python scripts/run_ablation.py --system extractive

# System B: base Qwen, start the Colab server without an adapter
PYTHONPATH=src python scripts/run_ablation.py \
  --system qwen_base \
  --endpoint https://YOUR-QWEN-ENDPOINT

# System C: Qwen + LoRA, start the Colab server with the adapter
PYTHONPATH=src python scripts/run_ablation.py \
  --system qwen_lora \
  --endpoint https://YOUR-QWEN-ENDPOINT

# Systems D/E: live retrieval with and without reranking
PYTHONPATH=src python scripts/run_ablation.py --system retrieval_no_reranker
PYTHONPATH=src python scripts/run_ablation.py --system retrieval_reranker
```

Build the summary table:

```bash
PYTHONPATH=src python scripts/compare_results.py
```

The included `ablation_summary.csv` reports:

- Extractive baseline: low hallucination risk but weak generation quality.
- Base Qwen: better lexical answer quality, with a small high-hallucination rate.
- Qwen + LoRA: higher confidence than base Qwen, but lower lexical-overlap metrics.
- Reranking: slightly improves gold-token overlap over dense-only retrieval in the saved run.

## Verification Checklist

Use this checklist before grading or demoing:

```bash
python -m unittest
streamlit run demo/app.py
```

Then verify the Colab endpoint:

```bash
curl -s https://YOUR-QWEN-ENDPOINT/health
```

Ask a sample question in Streamlit and confirm:

- Retrieved sources are shown with SEC citation IDs.
- The answer includes bracketed citations.
- Confidence score and hallucination risk are displayed.
- The answer backend is either `Colab GPU Qwen endpoint` or `Debug extractive fallback`.

## Current Limits

- Best on single-company questions.
- Not intended for real-time stock prices, forecasting, or investment advice.
- Multi-company comparison is limited by the current single-company resolution path.
- Broad conceptual questions can retrieve relevant but not always ideal chunks.
- Low hallucination risk means the answer is grounded; it does not guarantee ideal answer style or synthesis.

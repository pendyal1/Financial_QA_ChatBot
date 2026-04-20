# FinRAG
### Financial Retrieval-Augmented Generation with Hallucination Detection

FinRAG is a retrieval-augmented question answering system designed for financial documents.  
The system allows users to ask natural language questions about financial reports and receive answers that are grounded in verifiable source documents with citations.

Large language models are powerful for natural language understanding, but they often generate responses that are not supported by reliable sources. In financial contexts, unsupported claims can lead to incorrect interpretations of financial data.

FinRAG addresses this challenge by combining:

- Retrieval-Augmented Generation (RAG)
- Domain-specific fine-tuning on financial QA datasets
- Hallucination detection and citation verification

The result is a system that produces answers grounded in financial evidence while identifying when the model may be generating unsupported claims.

---

# Project Objectives

The main goals of FinRAG are:

- Build a retrieval-augmented generation pipeline for financial question answering
- Generate answers grounded in financial documents with source citations
- Detect hallucinated or unsupported claims in model outputs
- Evaluate whether retrieval and domain-specific training improve answer reliability

The project explores the following research questions:

1. Does retrieval-augmented generation improve answer accuracy compared to a baseline language model?
2. Can hallucination detection methods identify unsupported claims in generated responses?
3. Does fine-tuning on financial question answering datasets improve financial reasoning performance?

---

# System Architecture

FinRAG consists of three major components.

## 1. Retrieval-Augmented Generation (RAG)

Financial documents are segmented into smaller passages and stored in a vector database.  
When a user submits a question, the system retrieves the most relevant passages and provides them as context to the language model.

Pipeline:

```

Financial Documents
↓
Document Chunking
↓
Embedding Model
↓
Vector Database
↓
Retriever
↓
Language Model
↓
Answer + Citations

```

---

## 2. Hallucination Detection Layer

The hallucination detection module evaluates whether the generated answer is supported by the retrieved evidence.

This component performs:

- citation verification  
- evidence comparison  
- confidence scoring  
- unsupported claim detection  

Example Output:

```

Answer: Tesla reported supply chain risks in its 2023 10-K filing.

Source: Tesla 10-K 2023 – Risk Factors Section

Confidence Score: 0.91
Hallucination Risk: Low

```

---

## 3. Domain-Specific Fine-Tuning

To improve financial reasoning capabilities, a base language model is fine-tuned on financial question answering datasets.

Fine-tuning helps the model better understand:

- financial terminology
- numerical reasoning
- financial document structures

---

# Data Sources

The project relies on publicly available financial datasets.

### Financial Question Answering Datasets

FinQA  
https://huggingface.co/datasets/ibm-research/finqa

TAT-QA  
https://github.com/NExTplusplus/TAT-QA

FinanceBench  
https://github.com/patronus-ai/financebench

Financial QA Benchmark  
https://www.kaggle.com/datasets/yousefsaeedian/a-new-benchmark-for-financial-question-answering

### Financial Document Corpora

SEC EDGAR Company Facts Dataset  
https://www.kaggle.com/datasets/jamesglang/sec-edgar-company-facts-september2023

SEC Financial Statement Data Sets  
https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets

These datasets provide financial reports, financial tables, and question-answer pairs that support both training and evaluation of financial question answering systems.

---

# Evaluation

FinRAG will be evaluated using both quantitative and qualitative metrics.

## Quantitative Metrics

- Answer accuracy
- Citation correctness
- Retrieval relevance
- Hallucination detection accuracy

## Experimental Comparisons

The following systems will be compared:

1. Baseline language model (no retrieval)
2. Retrieval-Augmented Generation model
3. RAG + fine-tuned financial QA model
4. RAG + hallucination detection system

---

# Expected Outcomes

This project aims to produce:

- A working financial RAG system
- A hallucination detection framework
- Experimental evaluation results
- An interactive demo for financial question answering
- A reproducible research repository

---

# Project Structure

Example repository structure:

```

FinRAG/
│
├── data/
│   ├── raw_documents/
│   ├── processed_chunks/
│
├── src/
│   ├── retrieval/
│   ├── rag_pipeline/
│   ├── hallucination_detection/
│   ├── fine_tuning/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_evaluation.ipynb
│
├── demo/
│   └── app.py
│
├── requirements.txt
└── README.md

````

---

# Setup

Install dependencies:

```bash
pip install -r requirements.txt
````

Example dependencies:

```
transformers
langchain
faiss-cpu
sentence-transformers
datasets
pandas
numpy
```

## Current MVP Quickstart

The repository now includes a working baseline RAG pipeline:

1. Download latest SEC 10-K filings for the starter company set.

```bash
PYTHONPATH=src python -m finrag.download_sec_filings --tickers AAPL MSFT TSLA NVDA AMZN
```

2. Split the filings into retrievable chunks.

```bash
PYTHONPATH=src python -m finrag.chunk_documents
```

3. Build a FAISS vector index.

```bash
PYTHONPATH=src python -m finrag.build_index
```

4. Ask a question with the local debug extractor.

```bash
PYTHONPATH=src python -m finrag.answer "What risks did Apple report related to supply chains?"
```

For the project model path, run Qwen 2.5 7B on a Colab GPU server and call it from the local app. Do not load Qwen 7B directly from local Streamlit on a CPU-only Mac.

5. Run the starter evaluation set.

```bash
PYTHONPATH=src python -m finrag.evaluate
```

6. Launch the interactive demo locally on CPU.

```bash
PYTHONPATH=src streamlit run demo/app.py
```

Paste the Colab Qwen endpoint URL into the app. The local app performs retrieval and citation verification; the Colab GPU endpoint performs Qwen generation. The local extractor is only a debug path for checking retrieval and citations without the Colab endpoint.

### Downloaded Starter Data

The current local data pull uses SEC 10-K filings for:

- Apple: filed 2025-10-31
- Microsoft: filed 2025-07-30
- Tesla: filed 2026-01-29
- NVIDIA: filed 2026-02-25
- Amazon: filed 2026-02-06

Generated data artifacts live under `data/raw_documents/`, `data/processed_chunks/`, `data/index/`, and `data/fine_tuning/`. These are ignored by git because they are reproducible.

### Implementation Modules

- `src/finrag/download_sec_filings.py`: downloads SEC filings and extracts text
- `src/finrag/chunk_documents.py`: creates chunk-level JSONL
- `src/finrag/build_index.py`: embeds chunks and writes the FAISS index
- `src/finrag/retrieve.py`: retrieves relevant chunks for a query
- `src/finrag/answer.py`: generates cited answers using retrieval evidence
- `src/finrag/hallucination_detection.py`: checks citation validity and support
- `src/finrag/query.py`: detects company intent and risk questions so retrieval stays on the requested filing
- `src/finrag/evaluate.py`: runs the starter evaluation CSV
- `src/finrag/fine_tuning.py`: prepares a small FinQA JSONL file for later fine-tuning experiments
- `src/finrag/train_qlora.py`: QLoRA fine-tunes a 7B Hugging Face model on a CUDA GPU
- `src/finrag/hf_adapter_answer.py`: answers RAG questions with a saved Hugging Face LoRA adapter
- `src/finrag/qwen_server.py`: serves Qwen 2.5 7B from Colab GPU over HTTP
- `src/finrag/remote_qwen.py`: local CPU client that sends retrieved evidence to the Colab Qwen server
- `demo/app.py`: Streamlit demo

## 7B QLoRA Fine-Tuning On Google Colab

The recommended training route is QLoRA, not full fine-tuning. QLoRA loads the 7B model in 4-bit precision and trains only LoRA adapter weights, which is appropriate for Colab Pro GPUs.

Default base model:

```text
Qwen/Qwen2.5-7B-Instruct
```

This is a public 7B instruct model on Hugging Face. No Hugging Face Inference API is required. A Hugging Face token is only needed if your environment needs authenticated downloads, if you choose a gated model, or if you want to push the adapter back to the Hub.

### Colab Setup

Use a CUDA-backed Colab runtime, then run:

```bash
pip install -r requirements-colab.txt
```

Do not use `--force-reinstall` on Colab. It can replace Colab's CUDA/PyTorch packages and produce CUDA toolkit conflicts. If you already ran a force reinstall, restart the Colab runtime before continuing.

Verify the runtime:

```bash
python - <<'PY'
import torch, transformers, peft, accelerate, bitsandbytes
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("accelerate", accelerate.__version__)
print("bitsandbytes", bitsandbytes.__version__)
from transformers import PreTrainedModel
print("PreTrainedModel import OK")
PY
```

Prepare more FinQA examples:

```bash
PYTHONPATH=src python -m finrag.fine_tuning --limit 2000
```

Train the LoRA adapter:

```bash
PYTHONPATH=src python -m finrag.train_qlora \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --train-file data/fine_tuning/finqa_train.jsonl \
  --output-dir /content/drive/MyDrive/finrag-adapters/qwen2_5_7b_finqa_lora \
  --epochs 1 \
  --max-length 1536 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32
```

Start a Qwen generation server on the Colab GPU. It will use the saved LoRA adapter if that adapter directory exists; otherwise omit `--adapter-path` to serve the base model.

```bash
PYTHONPATH=src python -m finrag.qwen_server \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --adapter-path /content/drive/MyDrive/finrag-adapters/qwen2_5_7b_finqa_lora \
  --port 8000
```

Expose the Colab server with Cloudflare Tunnel:

```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
./cloudflared tunnel --url http://127.0.0.1:8000
```

Copy the printed `https://...trycloudflare.com` URL.

Then run Streamlit locally on your Mac:

```bash
PYTHONPATH=src streamlit run demo/app.py
```

Choose `Colab GPU Qwen endpoint` in the app and paste the Cloudflare URL.

There is also a Colab notebook wrapper at `notebooks/finetune_qwen7b_colab.ipynb` with cells for training, starting the Qwen server, and creating the tunnel.

### GPU Notes

The training script and Qwen server intentionally exit if CUDA is unavailable. If you run them in a local macOS terminal, they will fail with a GPU error. Run those commands from an actual Colab GPU runtime or a VS Code terminal attached to that runtime. Run only Streamlit locally on CPU.

---

# Example Usage

Example query:

```
What risks did Apple report in its most recent 10-K filing?
```

Example output:

```
Answer:
Apple identified supply chain disruptions and foreign exchange fluctuations as potential risks.

Source:
Apple 10-K 2023 – Risk Factors Section

Confidence Score: 0.88
Hallucination Risk: Low
```

---

# Disclaimer

This project relies on publicly available financial datasets. Data accessibility, quality, or preprocessing challenges may affect dataset usability. If limitations arise, the project scope may shift toward alternative financial text corpora or focus more heavily on evaluating retrieval and hallucination detection methods using available datasets.

---

# License

This project is intended for academic and research purposes.

#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch

print("Before setup:")
print("  torch", torch.__version__)
print("  cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

# This RAG app serves DragonLLM/Qwen-Open-Finance-R-8B-FP8 directly. It does
# not train or load LoRA adapters.
pip uninstall -y torchvision torchaudio torchtext >/dev/null 2>&1 || true

# Keep Colab's preinstalled torch/CUDA stack. Install only serving, retrieval,
# and Streamlit dependencies.
pip install -q --upgrade \
  "accelerate>=1.1.0" \
  "beautifulsoup4>=4.12.3" \
  "faiss-cpu>=1.8.0" \
  "fastapi>=0.115.0" \
  "huggingface-hub>=0.34.0,<1.0.0" \
  "pandas>=2.2.0,<3.0.0" \
  "python-dotenv>=1.0.1" \
  "pyngrok>=7.2.0" \
  "requests==2.32.4" \
  "scikit-learn>=1.5.0" \
  "sentence-transformers>=3.3.1,<4.0.0" \
  "sentencepiece>=0.2.0" \
  "streamlit>=1.37.0" \
  "transformers>=4.51.0,<5.0.0" \
  "tqdm>=4.66.0" \
  "uvicorn>=0.30.0"

python - <<'PY'
import importlib.metadata as metadata
import torch
import transformers
import accelerate

print("After setup:")
print("  torch", torch.__version__)
print("  cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("  transformers", transformers.__version__)
print("  accelerate", accelerate.__version__)
try:
    print("  triton", metadata.version("triton"))
except metadata.PackageNotFoundError:
    print("  triton", "NOT INSTALLED")
from transformers import PreTrainedModel
print("  PreTrainedModel import OK")
PY

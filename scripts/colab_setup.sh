#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch

print("Before setup:")
print("  torch", torch.__version__)
print("  cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

# Qwen text fine-tuning does not need torchvision/torchaudio. If these are
# mismatched with Colab's torch build, transformers can fail while importing
# optional vision modules.
pip uninstall -y torchvision torchaudio torchtext >/dev/null 2>&1 || true

# Keep Colab's preinstalled torch/CUDA stack. Install only the text-training
# dependencies this project needs.
pip install -q --upgrade \
  "accelerate==1.1.1" \
  "datasets==2.21.0" \
  "faiss-cpu>=1.8.0" \
  "fastapi>=0.115.0" \
  "fsspec==2024.6.1" \
  "huggingface-hub==0.26.5" \
  "pandas==2.2.2" \
  "peft==0.13.2" \
  "python-dotenv>=1.0.1" \
  "requests==2.32.4" \
  "scikit-learn>=1.5.0" \
  "sentence-transformers==3.3.1" \
  "sentencepiece>=0.2.0" \
  "streamlit>=1.37.0" \
  "transformers==4.46.3" \
  "uvicorn>=0.30.0"

python - <<'PY'
import torch
import transformers
import peft
import accelerate
import bitsandbytes

print("After setup:")
print("  torch", torch.__version__)
print("  cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("  transformers", transformers.__version__)
print("  peft", peft.__version__)
print("  accelerate", accelerate.__version__)
print("  bitsandbytes", bitsandbytes.__version__)
from transformers import PreTrainedModel
print("  PreTrainedModel import OK")
PY

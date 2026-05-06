from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_ADAPTER_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def adapter_base_model(adapter_path: str | Path | None) -> str | None:
    if not adapter_path:
        return None
    config_path = Path(adapter_path).expanduser() / "adapter_config.json"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = payload.get("base_model_name_or_path")
    return str(model_name) if model_name else None


def resolve_base_model(adapter_path: str | Path | None, explicit_model: str | None = None) -> str:
    if explicit_model:
        return explicit_model
    adapter_model = adapter_base_model(adapter_path)
    if adapter_model:
        return adapter_model
    env_model = os.getenv("HF_BASE_MODEL") or os.getenv("MODEL_NAME")
    if env_model:
        return env_model
    return DEFAULT_ADAPTER_BASE_MODEL

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("HF_BASE_MODEL")
PORT = int(os.getenv("PORT", "8000"))
ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    os.getenv(
        "FINRAG_LORA_ADAPTER_PATH",
        "/content/drive/MyDrive/Generative AI Project FinRAG/finrag_lora_adapter",
    ),
)
LOG_FILE = Path(os.getenv("LOG_FILE", "qwen_server.log"))
PID_FILE = Path(os.getenv("PID_FILE", "qwen_server.pid"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_existing_server() -> None:
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            old_pid = None
        if old_pid and process_alive(old_pid):
            os.kill(old_pid, signal.SIGTERM)
            for _ in range(20):
                if not process_alive(old_pid):
                    break
                time.sleep(1)
    PID_FILE.unlink(missing_ok=True)


def build_command() -> list[str]:
    command = [
        sys.executable,
        "-m",
        "finrag.qwen_server",
        "--port",
        str(PORT),
    ]
    if MODEL_NAME:
        command.extend(["--model-name", MODEL_NAME])
    if Path(ADAPTER_PATH).is_dir():
        print(f"Using LoRA adapter: {ADAPTER_PATH}")
        command.extend(["--adapter-path", ADAPTER_PATH])
    else:
        print(f"Adapter not found at {ADAPTER_PATH}; serving base model.")
    return command


def healthcheck() -> None:
    url = f"http://127.0.0.1:{PORT}/health"
    last_error: Exception | None = None
    for _ in range(90):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                print("Qwen server is healthy:")
                print(response.read().decode("utf-8"))
                return
        except Exception as exc:  # pragma: no cover - operational path
            last_error = exc
            time.sleep(5)
    raise RuntimeError(f"Qwen server is not reachable at {url}: {last_error}")


def main() -> None:
    stop_existing_server()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = LOG_FILE.open("w", encoding="utf-8")
    process = subprocess.Popen(
        build_command(),
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    print(f"Started qwen_server pid={process.pid} on port {PORT}")
    print(f"Logs: {LOG_FILE}")
    healthcheck()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if LOG_FILE.exists():
            print("Last 80 log lines:", file=sys.stderr)
            try:
                lines = LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]
                for line in lines:
                    print(line, file=sys.stderr)
            except Exception:
                pass
        raise

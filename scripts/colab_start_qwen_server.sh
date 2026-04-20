#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
ADAPTER_PATH="${ADAPTER_PATH:-/content/gdrive/MyDrive/finrag-adapters/qwen2_5_7b_finqa_lora}"
LOG_FILE="${LOG_FILE:-qwen_server.log}"

pkill -f "finrag.qwen_server" >/dev/null 2>&1 || true

ADAPTER_ARG=()
if [[ -d "$ADAPTER_PATH" ]]; then
  ADAPTER_ARG=(--adapter-path "$ADAPTER_PATH")
  echo "Using LoRA adapter: $ADAPTER_PATH"
else
  echo "Adapter not found at $ADAPTER_PATH; serving base model."
fi

nohup env PYTHONPATH=src python -m finrag.qwen_server \
  --model-name "$MODEL_NAME" \
  "${ADAPTER_ARG[@]}" \
  --port "$PORT" \
  > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Started qwen_server pid=$SERVER_PID on port $PORT"
echo "Logs: $LOG_FILE"

for attempt in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/qwen_health.json 2>/dev/null; then
    echo "Qwen server is healthy:"
    cat /tmp/qwen_health.json
    echo
    exit 0
  fi

  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Qwen server exited before becoming healthy. Last 80 log lines:"
    tail -80 "$LOG_FILE" || true
    exit 1
  fi

  if [[ "$attempt" == "1" || "$attempt" == "15" || "$attempt" == "30" || "$attempt" == "60" ]]; then
    echo "Waiting for Qwen server to load... attempt $attempt/90"
    tail -20 "$LOG_FILE" || true
  fi
  sleep 5
done

echo "Timed out waiting for Qwen server health. Last 80 log lines:"
tail -80 "$LOG_FILE" || true
exit 1

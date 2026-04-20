from __future__ import annotations

import os
import sys
import time
import urllib.request

from pyngrok import ngrok


PORT = int(os.getenv("QWEN_PORT", "8000"))
TOKEN = os.getenv("NGROK_AUTHTOKEN", "")


def healthcheck() -> None:
    url = f"http://127.0.0.1:{PORT}/health"
    for _ in range(12):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                print(response.read().decode("utf-8"))
                return
        except Exception as exc:
            last_error = exc
            time.sleep(5)
    raise RuntimeError(f"Qwen server is not reachable at {url}: {last_error}")


def main() -> None:
    if not TOKEN:
        raise ValueError("Set NGROK_AUTHTOKEN before running this script.")
    healthcheck()
    ngrok.set_auth_token(TOKEN)
    public_url = ngrok.connect(PORT, "http").public_url
    print(public_url)
    print("Keep this cell/process running while local Streamlit uses the endpoint.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--text-file", default="fixtures/sample.txt")
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    text = Path(args.text_file).read_text(encoding="utf-8")
    submit = requests.post(f"{base_url}/predict/text", json={"text": text}, timeout=30)
    submit.raise_for_status()
    job = submit.json()
    print(f"submitted {job['job_id']}")

    deadline = time.time() + args.timeout_seconds
    status = None
    while time.time() < deadline:
        response = requests.get(f"{base_url}{job['status_url']}", timeout=30)
        response.raise_for_status()
        status = response.json()
        print(f"{status['job_id']} {status['status']}")
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(args.poll_seconds)

    if not status or status["status"] != "completed":
        raise SystemExit(f"job did not complete: {status}")

    result = requests.get(f"{base_url}/jobs/{job['job_id']}/result.json", timeout=30)
    result.raise_for_status()
    preds = requests.get(f"{base_url}/jobs/{job['job_id']}/preds.norm.f16.bin", timeout=30)
    preds.raise_for_status()
    print(f"result bytes={len(result.content)} preds bytes={len(preds.content)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--text")
    parser.add_argument("--text-file")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Provide --text or --text-file")

    base_url = args.url.rstrip("/")
    response = requests.post(f"{base_url}/predict/text", json={"text": text}, timeout=30)
    response.raise_for_status()
    job = response.json()
    print(json.dumps(job, indent=2))

    if not args.poll:
        return 0

    while True:
        status = requests.get(f"{base_url}{job['status_url']}", timeout=30)
        status.raise_for_status()
        data = status.json()
        print(json.dumps(data, indent=2))
        if data["status"] in {"completed", "failed"}:
            return 0 if data["status"] == "completed" else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

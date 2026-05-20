from __future__ import annotations

import argparse
import json
from pathlib import Path

from tribev2_api.config import get_settings
from tribev2_api.inference.outputs import write_result_json
from tribev2_api.inference.text import run_text_prediction
from tribev2_api.jobs import create_job, job_dir, utc_now
from tribev2_api.metadata import build_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default="fixtures/sample.txt")
    args = parser.parse_args()

    settings = get_settings()
    text = Path(args.text_file).read_text(encoding="utf-8")
    record = create_job(settings, "text")
    record.status = "running"
    record.started_at = utc_now()

    path = job_dir(settings, record.job_id)
    prediction = run_text_prediction(text, path, settings)
    completed_at = utc_now()
    result = write_result_json(
        path,
        job_id=record.job_id,
        status="completed",
        input_type="text",
        text_chars=len(text),
        preds=prediction.preds,
        segments=prediction.segments,
        events_count=prediction.events_count,
        model_metadata=build_metadata(settings),
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=completed_at,
    )
    print(json.dumps({"job_id": record.job_id, "result": result["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from tribev2_api.config import Settings

JobStatus = Literal["queued", "running", "completed", "failed"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


@dataclass
class JobRecord:
    job_id: str
    input_type: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    artifacts: dict = field(default_factory=dict)

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data["status_url"] = f"/jobs/{self.job_id}"
        if self.status == "completed":
            data["result_url"] = f"/jobs/{self.job_id}/result.json"
            data["predictions_url"] = f"/jobs/{self.job_id}/preds.norm.f16.bin"
        return data


def job_dir(settings: Settings, job_id: str) -> Path:
    return settings.jobs_dir / job_id


def metadata_path(settings: Settings, job_id: str) -> Path:
    return job_dir(settings, job_id) / "metadata.json"


def logs_path(settings: Settings, job_id: str) -> Path:
    return job_dir(settings, job_id) / "logs.txt"


def result_path(settings: Settings, job_id: str) -> Path:
    return job_dir(settings, job_id) / "result.json"


def preds_bin_path(settings: Settings, job_id: str) -> Path:
    return job_dir(settings, job_id) / "preds.norm.f16.bin"


def create_job(settings: Settings, input_type: str) -> JobRecord:
    prefix = "txt" if input_type == "text" else input_type[:3]
    record = JobRecord(
        job_id=new_job_id(prefix),
        input_type=input_type,
        status="queued",
        created_at=utc_now(),
    )
    path = job_dir(settings, record.job_id)
    (path / "input").mkdir(parents=True, exist_ok=True)
    save_job(settings, record)
    append_log(settings, record.job_id, f"created {record.job_id}")
    return record


def save_job(settings: Settings, record: JobRecord) -> None:
    path = job_dir(settings, record.job_id)
    path.mkdir(parents=True, exist_ok=True)
    metadata_path(settings, record.job_id).write_text(
        json.dumps(asdict(record), indent=2),
        encoding="utf-8",
    )


def load_job(settings: Settings, job_id: str) -> JobRecord | None:
    path = metadata_path(settings, job_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return JobRecord(**data)


def append_log(settings: Settings, job_id: str, message: str) -> None:
    path = logs_path(settings, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")

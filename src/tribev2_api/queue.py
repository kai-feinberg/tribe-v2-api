from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass

from tribev2_api.config import Settings
from tribev2_api.inference.outputs import write_result_json
from tribev2_api.inference.text import run_text_prediction
from tribev2_api.jobs import (
    JobRecord,
    append_log,
    job_dir,
    load_job,
    save_job,
    utc_now,
)
from tribev2_api.metadata import build_metadata


@dataclass
class TextJobPayload:
    job_id: str
    text: str


class JobQueue:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._queue: queue.Queue[TextJobPayload | None] = queue.Queue(
            maxsize=settings.max_pending_jobs
        )
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._worker, name="tribev2-job-worker", daemon=True)
        self._thread.start()

    def enqueue_text(self, record: JobRecord, text: str) -> None:
        try:
            self._queue.put_nowait(TextJobPayload(job_id=record.job_id, text=text))
        except queue.Full as exc:
            raise RuntimeError("Job queue is full.") from exc

    def pending_count(self) -> int:
        return self._queue.qsize()

    def _worker(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                return
            self._run_text_job(payload)
            self._queue.task_done()

    def _run_text_job(self, payload: TextJobPayload) -> None:
        record = load_job(self.settings, payload.job_id)
        if record is None:
            return

        record.status = "running"
        record.started_at = utc_now()
        save_job(self.settings, record)
        append_log(self.settings, record.job_id, "started text inference")

        path = job_dir(self.settings, record.job_id)
        try:
            prediction = run_text_prediction(payload.text, path, self.settings)
            result = write_result_json(
                path,
                job_id=record.job_id,
                status="completed",
                input_type="text",
                text_chars=len(payload.text),
                preds=prediction.preds,
                segments=prediction.segments,
                events_count=prediction.events_count,
                model_metadata=build_metadata(self.settings),
                created_at=record.created_at,
                started_at=record.started_at,
                completed_at=utc_now(),
            )
            record.status = "completed"
            record.completed_at = result["completed_at"]
            record.artifacts = result["artifacts"]
            save_job(self.settings, record)
            append_log(
                self.settings,
                record.job_id,
                f"completed text inference in {prediction.elapsed_seconds:.2f}s",
            )
        except Exception as exc:
            record.status = "failed"
            record.completed_at = utc_now()
            record.error = str(exc)
            save_job(self.settings, record)
            append_log(self.settings, record.job_id, f"failed: {exc}")
            append_log(self.settings, record.job_id, traceback.format_exc())

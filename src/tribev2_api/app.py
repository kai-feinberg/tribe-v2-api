from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tribev2_api.config import get_settings
from tribev2_api.inference.mesh import fsaverage5_mesh_path
from tribev2_api.jobs import (
    create_job,
    load_job,
    read_logs_tail,
    preds_bin_path,
    result_path,
    save_job,
)
from tribev2_api.metadata import build_metadata
from tribev2_api.queue import JobQueue

settings = get_settings()
job_queue = JobQueue(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    job_queue.start()
    yield


app = FastAPI(
    title="TRIBE V2 API Template",
    version="0.1.0",
    lifespan=lifespan,
)


class TextPredictionRequest(BaseModel):
    text: str = Field(min_length=1)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "queue_pending": job_queue.pending_count(),
        "fake_inference": settings.fake_inference,
    }


@app.get("/metadata")
def metadata() -> dict:
    return build_metadata(settings)


@app.get("/mesh/fsaverage5.bin")
def get_mesh():
    path = fsaverage5_mesh_path(settings)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename="fsaverage5.bin",
    )


@app.post("/predict/text", status_code=202)
def predict_text(request: TextPredictionRequest) -> dict:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input is empty.")
    if len(text) > settings.max_text_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds max length of {settings.max_text_chars} characters.",
        )

    record = create_job(settings, "text")
    try:
        job_queue.enqueue_text(record, text)
    except RuntimeError as exc:
        record.status = "failed"
        record.error = str(exc)
        save_job(settings, record)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return record.to_public_dict()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    record = load_job(settings, job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    data = record.to_public_dict()
    data["logs_tail"] = read_logs_tail(settings, job_id)
    return data


@app.get("/jobs/{job_id}/result.json")
def get_result(job_id: str):
    path = result_path(settings, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found.")
    return FileResponse(path, media_type="application/json", filename="result.json")


@app.get("/jobs/{job_id}/preds.norm.f16.bin")
def get_predictions(job_id: str):
    path = preds_bin_path(settings, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Prediction blob not found.")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename="preds.norm.f16.bin",
    )

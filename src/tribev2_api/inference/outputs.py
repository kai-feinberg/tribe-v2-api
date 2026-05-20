from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_VERTEX_COUNT = 20484


def segment_times(segments: list[Any]) -> dict:
    indices = list(range(len(segments)))
    starts = []
    durations = []
    for segment in segments:
        start = float(getattr(segment, "start", 0.0) or 0.0)
        offset = float(getattr(segment, "offset", 0.0) or 0.0)
        duration = float(getattr(segment, "duration", 0.0) or 0.0)
        starts.append(start + offset)
        durations.append(duration)
    return {
        "indices": indices,
        "seconds": starts,
        "durations_seconds": durations,
    }


def normalize_predictions(preds: np.ndarray) -> np.ndarray:
    try:
        from tribev2.plotting.utils import robust_normalize

        return robust_normalize(preds, percentile=99).astype(np.float16)
    except Exception:
        values = preds.astype(np.float32)
        low, high = np.percentile(values, [1, 99])
        if high <= low:
            return np.zeros_like(values, dtype=np.float16)
        values = np.clip((values - low) / (high - low), 0.0, 1.0)
        return values.astype(np.float16)


def save_prediction_artifacts(job_path: Path, preds: np.ndarray) -> dict:
    if preds.ndim != 2:
        raise ValueError(f"Expected 2D predictions, got shape {list(preds.shape)}")
    if preds.shape[1] != EXPECTED_VERTEX_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VERTEX_COUNT} vertices, got shape {list(preds.shape)}"
        )

    raw_path = job_path / "preds.raw.npy"
    norm_path = job_path / "preds.norm.f16.bin"
    np.save(raw_path, preds)
    normalize_predictions(preds).tofile(norm_path)
    return {
        "raw_npy": "preds.raw.npy",
        "normalized_f16_bin": "preds.norm.f16.bin",
        "dtype": "float16",
        "shape": [int(preds.shape[0]), int(preds.shape[1])],
        "normalized": True,
    }


def basic_reductions(preds: np.ndarray) -> dict:
    mean_abs_by_time = np.mean(np.abs(preds), axis=1).astype(float)
    return {
        "global_mean_abs_by_timestep": mean_abs_by_time.tolist(),
        "note": (
            "Yeo7 and heuristic axis reductions are planned for the text MVP "
            "output pass once the runtime produces real predictions."
        ),
    }


def write_result_json(
    job_path: Path,
    *,
    job_id: str,
    status: str,
    input_type: str,
    text_chars: int,
    preds: np.ndarray,
    segments: list[Any],
    events_count: int,
    model_metadata: dict,
    created_at: str | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> dict:
    artifacts = save_prediction_artifacts(job_path, preds)
    result = {
        "job_id": job_id,
        "status": status,
        "input_type": input_type,
        "created_at": created_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "input": {
            "text_chars": text_chars,
        },
        "model": model_metadata.get("model", {}),
        "license": model_metadata.get("model", {}).get("license"),
        "mesh": model_metadata.get("brain_space", {}),
        "prediction": artifacts,
        "time": segment_times(segments),
        "network_traces": {},
        "heuristic_axes": {},
        "summary": basic_reductions(preds),
        "events": {
            "count": events_count,
        },
        "artifacts": {
            "result_json": "result.json",
            "predictions_bin": "preds.norm.f16.bin",
        },
    }
    (job_path / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

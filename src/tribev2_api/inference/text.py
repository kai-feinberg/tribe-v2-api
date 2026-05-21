from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tribev2_api.config import Settings
from tribev2_api.inference.cpu import (
    configure_cpu_runtime,
    prepare_runtime_model_dir,
    runtime_config_update,
)
from tribev2_api.inference.patches import apply_text_timing_patch, set_current_text


@dataclass
class TextPrediction:
    preds: np.ndarray
    segments: list[Any]
    events_count: int
    elapsed_seconds: float
    total_segments: int
    kept_segments: int


class FakeSegment:
    def __init__(self, start: float, duration: float) -> None:
        self.start = start
        self.offset = 0.0
        self.duration = duration


_MODEL = None


def _model_source(settings: Settings) -> str | Path:
    if settings.runtime_model_dir.exists():
        return settings.runtime_model_dir
    if settings.model_snapshot_dir.exists():
        return prepare_runtime_model_dir(settings.model_snapshot_dir, "cpu")
    return settings.model_repo


def load_model(settings: Settings, progress: ProgressCallback | None = None):
    global _MODEL
    if _MODEL is not None:
        if progress:
            progress("model_ready", "Model already loaded in this worker.", 18)
        return _MODEL
    configure_cpu_runtime(settings.force_cpu)
    apply_text_timing_patch()
    from tribev2 import TribeModel

    if progress:
        progress("model_loading", "Loading TRIBE V2 model into CPU memory.", 10)
    source = _model_source(settings)
    _MODEL = TribeModel.from_pretrained(
        source,
        cache_folder=settings.cache_dir,
        device="cpu",
        config_update=runtime_config_update("cpu"),
    )
    return _MODEL


ProgressCallback = Any


def run_text_prediction(
    text: str,
    job_path: Path,
    settings: Settings,
    progress: ProgressCallback | None = None,
) -> TextPrediction:
    if settings.fake_inference:
        return fake_text_prediction()

    text = text.strip()
    if not text:
        raise ValueError("Text input is empty.")

    input_dir = job_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    text_path = input_dir / "input.txt"
    text_path.write_text(text, encoding="utf-8")

    model = load_model(settings, progress)
    set_current_text(text)
    started = time.time()
    try:
        if progress:
            progress("events", "Creating audio/text events from input text.", 24)
        events = model.get_events_dataframe(text_path=str(text_path))
        if progress:
            progress("events", f"Extracted {len(events)} raw events.", 42)
        events = drop_text_events_unless_enabled(events, settings.enable_text_events)
        if progress:
            progress("predicting", "Running TRIBE V2 prediction on CPU.", 58)
        preds, segments = model.predict(events=events, verbose=False)
    finally:
        set_current_text(None)

    if not isinstance(preds, np.ndarray):
        preds = np.asarray(preds)

    total_segments = estimate_total_segments(segments)
    kept_segments = int(preds.shape[0])
    if progress:
        progress(
            "artifact_writing",
            f"Predicted {kept_segments} / {total_segments} segments.",
            88,
            processed_segments=kept_segments,
            total_segments=total_segments,
            kept_segments=kept_segments,
        )

    return TextPrediction(
        preds=preds,
        segments=list(segments),
        events_count=int(len(events)),
        elapsed_seconds=time.time() - started,
        total_segments=total_segments,
        kept_segments=kept_segments,
    )


def estimate_total_segments(segments: list[Any]) -> int:
    if not segments:
        return 0
    starts = [
        float(getattr(segment, "start", 0.0) or 0.0)
        + float(getattr(segment, "offset", 0.0) or 0.0)
        for segment in segments
    ]
    durations = [float(getattr(segment, "duration", 1.0) or 1.0) for segment in segments]
    stop = max(start + duration for start, duration in zip(starts, durations, strict=False))
    tr = max(min(durations), 1e-6)
    return max(len(segments), int(round(stop / tr)))


def drop_text_events_unless_enabled(events: Any, enabled: bool) -> Any:
    if enabled or "type" not in events.columns:
        return events
    event_type = events["type"].astype(str).str.lower()
    return events[event_type != "word"].reset_index(drop=True)


def fake_text_prediction() -> TextPrediction:
    rng = np.random.default_rng(1234)
    preds = rng.normal(0, 1, size=(3, 20484)).astype(np.float32)
    segments = [FakeSegment(float(i), 1.0) for i in range(preds.shape[0])]
    return TextPrediction(
        preds=preds,
        segments=segments,
        events_count=7,
        elapsed_seconds=0.01,
        total_segments=3,
        kept_segments=3,
    )

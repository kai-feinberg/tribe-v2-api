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


def load_model(settings: Settings):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    configure_cpu_runtime(settings.force_cpu)
    apply_text_timing_patch()
    from tribev2 import TribeModel

    source = _model_source(settings)
    _MODEL = TribeModel.from_pretrained(
        source,
        cache_folder=settings.cache_dir,
        device="cpu",
        config_update=runtime_config_update("cpu"),
    )
    return _MODEL


def run_text_prediction(text: str, job_path: Path, settings: Settings) -> TextPrediction:
    if settings.fake_inference:
        return fake_text_prediction()

    text = text.strip()
    if not text:
        raise ValueError("Text input is empty.")

    input_dir = job_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    text_path = input_dir / "input.txt"
    text_path.write_text(text, encoding="utf-8")

    model = load_model(settings)
    set_current_text(text)
    started = time.time()
    try:
        events = model.get_events_dataframe(text_path=str(text_path))
        preds, segments = model.predict(events=events, verbose=False)
    finally:
        set_current_text(None)

    if not isinstance(preds, np.ndarray):
        preds = np.asarray(preds)

    return TextPrediction(
        preds=preds,
        segments=list(segments),
        events_count=int(len(events)),
        elapsed_seconds=time.time() - started,
    )


def fake_text_prediction() -> TextPrediction:
    rng = np.random.default_rng(1234)
    preds = rng.normal(0, 1, size=(3, 20484)).astype(np.float32)
    segments = [FakeSegment(float(i), 1.0) for i in range(preds.shape[0])]
    return TextPrediction(
        preds=preds,
        segments=segments,
        events_count=7,
        elapsed_seconds=0.01,
    )

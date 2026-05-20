from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    force_cpu: bool = True
    cache_dir: Path = Path("./cache")
    jobs_dir: Path = Path("./jobs")
    max_text_chars: int = 5000
    max_pending_jobs: int = 3
    result_ttl_hours: int = 24
    delete_inputs_after_job: bool = True
    keep_failed_inputs: bool = False
    debug_artifacts: bool = False
    fake_inference: bool = False
    model_repo: str = "facebook/tribev2"
    model_checkpoint: str = "best.ckpt"

    @property
    def model_snapshot_dir(self) -> Path:
        return self.cache_dir / "official_model_repo"

    @property
    def runtime_model_dir(self) -> Path:
        return self.model_snapshot_dir / "runtime-cpu"


def get_settings() -> Settings:
    settings = Settings(
        host=os.environ.get("TRIBE_HOST", "0.0.0.0"),
        port=_int_env("TRIBE_PORT", 8000),
        force_cpu=_bool_env("TRIBE_FORCE_CPU", True),
        cache_dir=Path(os.environ.get("TRIBE_CACHE_DIR", "./cache")),
        jobs_dir=Path(os.environ.get("TRIBE_JOBS_DIR", "./jobs")),
        max_text_chars=_int_env("TRIBE_MAX_TEXT_CHARS", 5000),
        max_pending_jobs=_int_env("TRIBE_MAX_PENDING_JOBS", 3),
        result_ttl_hours=_int_env("TRIBE_RESULT_TTL_HOURS", 24),
        delete_inputs_after_job=_bool_env("TRIBE_DELETE_INPUTS_AFTER_JOB", True),
        keep_failed_inputs=_bool_env("TRIBE_KEEP_FAILED_INPUTS", False),
        debug_artifacts=_bool_env("TRIBE_DEBUG_ARTIFACTS", False),
        fake_inference=_bool_env("TRIBE_FAKE_INFERENCE", False),
        model_repo=os.environ.get("TRIBE_MODEL_REPO", "facebook/tribev2"),
        model_checkpoint=os.environ.get("TRIBE_MODEL_CHECKPOINT", "best.ckpt"),
    )
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    return settings

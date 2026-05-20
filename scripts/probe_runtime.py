from __future__ import annotations

import json
import resource
import time

from tribev2_api.config import get_settings
from tribev2_api.inference.cpu import (
    configure_cpu_runtime,
    prepare_runtime_model_dir,
    runtime_config_update,
)


def _rss_mb() -> float:
    # Linux reports KB; macOS reports bytes. This script is intended for Linux
    # Docker/VPS, but this keeps accidental local runs readable.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if value > 10_000_000:
        return value / (1024 * 1024)
    return value / 1024


def main() -> int:
    settings = get_settings()
    configure_cpu_runtime(settings.force_cpu)
    started = time.time()

    import torch
    from tribev2 import TribeModel

    source = settings.runtime_model_dir
    if not source.exists() and settings.model_snapshot_dir.exists():
        source = prepare_runtime_model_dir(settings.model_snapshot_dir, "cpu")
    if not source.exists():
        source = settings.model_repo

    model = TribeModel.from_pretrained(
        source,
        cache_folder=settings.cache_dir,
        device="cpu",
        config_update=runtime_config_update("cpu"),
    )
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "status": "ok",
                "repo": settings.model_repo,
                "source": str(source),
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "model_class": model.__class__.__name__,
                "elapsed_seconds": round(elapsed, 2),
                "max_rss_mb": round(_rss_mb(), 1),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

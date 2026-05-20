from __future__ import annotations

import json
import time

from huggingface_hub import snapshot_download

from tribev2_api.config import get_settings
from tribev2_api.inference.cpu import configure_cpu_runtime, prepare_runtime_model_dir


def main() -> int:
    settings = get_settings()
    configure_cpu_runtime(settings.force_cpu)
    started = time.time()

    settings.model_snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=settings.model_repo,
        allow_patterns=["config.yaml", settings.model_checkpoint],
        local_dir=settings.model_snapshot_dir,
        local_dir_use_symlinks=False,
    )
    runtime_path = prepare_runtime_model_dir(settings.model_snapshot_dir, "cpu")
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "status": "ok",
                "repo": settings.model_repo,
                "snapshot_path": str(snapshot_path),
                "runtime_path": str(runtime_path),
                "elapsed_seconds": round(elapsed, 2),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

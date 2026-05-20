from pathlib import Path

import numpy as np

from tribev2_api.inference.outputs import save_prediction_artifacts


def test_save_prediction_artifacts(tmp_path: Path):
    preds = np.zeros((2, 20484), dtype=np.float32)

    artifacts = save_prediction_artifacts(tmp_path, preds)

    assert artifacts["shape"] == [2, 20484]
    assert artifacts["dtype"] == "float16"
    assert (tmp_path / "preds.raw.npy").exists()
    assert (tmp_path / "preds.norm.f16.bin").stat().st_size == 2 * 20484 * 2

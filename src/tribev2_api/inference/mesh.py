from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from tribev2_api.config import Settings


def fsaverage5_mesh_path(settings: Settings) -> Path:
    out = settings.cache_dir / "mesh" / "fsaverage5.bin"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    import nibabel
    from nilearn import datasets

    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")

    def load_gifti_mesh(path):
        gifti = nibabel.load(path)
        return (
            gifti.darrays[0].data.astype(np.float32),
            gifti.darrays[1].data.astype(np.uint32),
        )

    def load_scalar(path):
        return nibabel.load(path).darrays[0].data.astype(np.float32)

    pial_l_v, _ = load_gifti_mesh(fsavg.pial_left)
    infl_l_v, infl_l_f = load_gifti_mesh(fsavg.infl_left)
    pial_r_v, _ = load_gifti_mesh(fsavg.pial_right)
    infl_r_v, infl_r_f = load_gifti_mesh(fsavg.infl_right)
    half_l = ((pial_l_v + infl_l_v) * 0.5).astype(np.float32)
    half_r = ((pial_r_v + infl_r_v) * 0.5).astype(np.float32)
    half_l = half_l.copy()
    half_r = half_r.copy()
    half_l[:, 0] -= half_l[:, 0].max() + 6
    half_r[:, 0] -= half_r[:, 0].min() - 6
    sulc_l = load_scalar(fsavg.sulc_left)
    sulc_r = load_scalar(fsavg.sulc_right)

    with out.open("wb") as handle:
        handle.write(
            struct.pack(
                "<IIII",
                half_l.shape[0],
                infl_l_f.shape[0],
                half_r.shape[0],
                infl_r_f.shape[0],
            )
        )
        handle.write(half_l.tobytes())
        handle.write(infl_l_f.tobytes())
        handle.write(sulc_l.tobytes())
        handle.write(half_r.tobytes())
        handle.write(infl_r_f.tobytes())
        handle.write(sulc_r.tobytes())
    return out

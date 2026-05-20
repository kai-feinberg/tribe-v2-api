from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any


def force_cpu_environment() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def apply_torch_cpu_patches() -> None:
    import torch

    if getattr(torch, "_tribev2_api_cpu_patched", False):
        return

    def redirect(args, kwargs):
        next_args = []
        for arg in args:
            if isinstance(arg, torch.device) and arg.type == "cuda":
                next_args.append(torch.device("cpu"))
            elif isinstance(arg, str) and "cuda" in arg:
                next_args.append("cpu")
            else:
                next_args.append(arg)
        if "device" in kwargs:
            device = kwargs["device"]
            if isinstance(device, torch.device) and device.type == "cuda":
                kwargs["device"] = torch.device("cpu")
            elif isinstance(device, str) and "cuda" in device:
                kwargs["device"] = "cpu"
        return tuple(next_args), kwargs

    original_module_to = torch.nn.Module.to
    original_tensor_to = torch.Tensor.to

    def module_to(self, *args, **kwargs):
        args, kwargs = redirect(args, kwargs)
        return original_module_to(self, *args, **kwargs)

    def tensor_to(self, *args, **kwargs):
        args, kwargs = redirect(args, kwargs)
        return original_tensor_to(self, *args, **kwargs)

    torch.nn.Module.to = module_to
    torch.Tensor.to = tensor_to
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.cuda.is_available = lambda: False
    torch._tribev2_api_cpu_patched = True


def runtime_config_update(device: str = "cpu") -> dict[str, Any]:
    return {
        "data.num_workers": 0,
        "data.batch_size": 1,
        "data.text_feature.device": device,
        "data.audio_feature.device": device,
        "data.image_feature.image.device": device,
        "data.video_feature.image.device": device,
        "data.image_feature.image.batch_size": 1,
        "data.video_feature.image.batch_size": 1,
    }


def prepare_runtime_model_dir(snapshot_dir: Path, device: str = "cpu") -> Path:
    source_config = snapshot_dir / "config.yaml"
    source_checkpoint = snapshot_dir / "best.ckpt"
    if not source_config.exists() or not source_checkpoint.exists():
        return snapshot_dir

    runtime_dir = snapshot_dir / f"runtime-{device}"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    config_text = source_config.read_text(encoding="utf-8")
    config_text = re.sub(
        r"(?m)^(\s*device:\s*)cuda\s*$",
        lambda match: f"{match.group(1)}{device}",
        config_text,
    )
    (runtime_dir / "config.yaml").write_text(config_text, encoding="utf-8")

    runtime_checkpoint = runtime_dir / "best.ckpt"
    if not runtime_checkpoint.exists():
        try:
            os.link(source_checkpoint, runtime_checkpoint)
        except OSError:
            shutil.copy2(source_checkpoint, runtime_checkpoint)

    return runtime_dir


def configure_cpu_runtime(force: bool = True) -> None:
    if not force:
        return
    force_cpu_environment()
    apply_torch_cpu_patches()

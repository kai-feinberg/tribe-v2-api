from __future__ import annotations

from tribev2_api import __version__
from tribev2_api.config import Settings


def build_metadata(settings: Settings) -> dict:
    return {
        "wrapper": {
            "name": "tribev2-api-template",
            "version": __version__,
        },
        "model": {
            "repo": settings.model_repo,
            "checkpoint": settings.model_checkpoint,
            "upstream_project_url": "https://github.com/facebookresearch/tribev2",
            "huggingface_url": "https://huggingface.co/facebook/tribev2",
            "license": "CC-BY-NC-4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        },
        "brain_space": {
            "mesh": "fsaverage5",
            "vertex_count": 20484,
            "subject_model": "average_subject",
            "hemodynamic_lag_seconds": 5.0,
        },
        "features": {
            "force_cpu": settings.force_cpu,
            "text_word_timing_patch": True,
            "text_events_enabled": settings.enable_text_events,
            "image_as_video_adapter": False,
            "video_input": False,
            "audio_input": False,
            "fake_inference": settings.fake_inference,
        },
        "limits": {
            "max_text_chars": settings.max_text_chars,
            "max_pending_jobs": settings.max_pending_jobs,
            "result_ttl_hours": settings.result_ttl_hours,
        },
        "disclaimer": (
            "Outputs are predicted fMRI-like responses from a computational "
            "model, not direct measurements of individual brain activity. "
            "Use is intended for non-commercial research/internal evaluation."
        ),
    }

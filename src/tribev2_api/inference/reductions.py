from __future__ import annotations

from functools import lru_cache

import numpy as np


REGION_GROUPS = {
    "Emotional Engagement": [
        "G_Ins_lg_and_S_cent_ins",
        "G_insular_short",
        "S_circular_insula_ant",
        "S_circular_insula_inf",
        "S_circular_insula_sup",
        "G_orbital",
        "S_orbital_lateral",
        "S_orbital_med-olfact",
        "S_orbital-H_Shaped",
        "G_and_S_cingul-Ant",
        "G_and_S_cingul-Mid-Ant",
        "G_subcallosal",
    ],
    "Visual Attention": [
        "S_calcarine",
        "G_cuneus",
        "G_occipital_sup",
        "G_oc-temp_med-Lingual",
        "S_oc-temp_med_and_Lingual",
        "G_occipital_middle",
        "S_oc_middle_and_Lunatus",
        "S_oc_sup_and_transversal",
        "Pole_occipital",
        "G_and_S_occipital_inf",
        "S_occipital_ant",
    ],
    "Auditory Processing": [
        "G_temp_sup-Lateral",
        "G_temp_sup-G_T_transv",
        "G_temp_sup-Plan_tempo",
        "G_temp_sup-Plan_polar",
        "S_temporal_sup",
        "S_temporal_transverse",
    ],
    "Memory Encoding": [
        "G_oc-temp_med-Parahip",
        "G_oc-temp_lat-fusifor",
        "S_oc-temp_lat",
        "S_collat_transv_ant",
        "S_collat_transv_post",
        "Pole_temporal",
        "G_temporal_inf",
        "S_temporal_inf",
    ],
    "Reward & Motivation": [
        "G_rectus",
        "S_suborbital",
        "G_and_S_frontomargin",
        "G_and_S_transv_frontopol",
        "G_subcallosal",
        "G_and_S_cingul-Mid-Post",
    ],
    "Language Comprehension": [
        "G_front_inf-Opercular",
        "G_front_inf-Triangul",
        "G_front_inf-Orbital",
        "S_front_inf",
        "G_pariet_inf-Angular",
        "G_pariet_inf-Supramar",
        "G_temporal_middle",
    ],
    "Social Cognition": [
        "G_front_sup",
        "S_front_sup",
        "G_precuneus",
        "S_subparietal",
        "G_cingul-Post-dorsal",
        "G_cingul-Post-ventral",
        "S_intrapariet_and_P_trans",
        "S_interm_prim-Jensen",
    ],
}


DOMAIN_DESCRIPTIONS = {
    "Emotional Engagement": "Insula, orbitofrontal, anterior cingulate, and subcallosal regions.",
    "Visual Attention": "Occipital and visual word-form regions.",
    "Auditory Processing": "Superior temporal and transverse temporal regions.",
    "Memory Encoding": "Parahippocampal, fusiform, temporal pole, and inferior temporal regions.",
    "Reward & Motivation": "Medial orbitofrontal, frontal pole, subcallosal, and cingulate regions.",
    "Language Comprehension": "Inferior frontal, angular, supramarginal, and middle temporal regions.",
    "Social Cognition": "Medial prefrontal, precuneus, posterior cingulate, and TPJ proxy regions.",
}


@lru_cache(maxsize=1)
def load_destrieux_parcellation() -> tuple[list[str], np.ndarray]:
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in atlas["labels"]
    ]
    label_map = np.concatenate([np.asarray(atlas["map_left"]), np.asarray(atlas["map_right"])])
    return labels, label_map


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - float(values.mean())) / std


def compute_cognitive_domains(preds: np.ndarray) -> dict:
    try:
        labels, label_map = load_destrieux_parcellation()
    except Exception as exc:
        return {
            "scores": {},
            "time_series": {},
            "metadata": {"available": False, "error": str(exc)},
        }

    label_name_to_idx = {name: i for i, name in enumerate(labels)}
    global_mean = float(np.mean(preds))
    global_std = float(np.std(preds)) or 1.0
    scores: dict[str, float] = {}
    time_series: dict[str, list[float]] = {}
    masks: dict[str, np.ndarray] = {}

    for category, region_names in REGION_GROUPS.items():
        mask = np.zeros(preds.shape[1], dtype=bool)
        for region_name in region_names:
            if region_name in label_name_to_idx:
                mask |= label_map == label_name_to_idx[region_name]
        masks[category] = mask
        if not mask.any():
            series = np.zeros(preds.shape[0], dtype=np.float32)
        else:
            raw = np.mean(preds[:, mask], axis=1)
            series = _sigmoid((raw - global_mean) / global_std).astype(np.float32)
        time_series[category] = series.astype(float).tolist()
        scores[category] = float(series.mean())

    scores["Overall Impact"] = float(np.mean(list(scores.values()))) if scores else 0.0
    return {
        "scores": scores,
        "time_series": time_series,
        "descriptions": DOMAIN_DESCRIPTIONS,
        "metadata": {
            "available": True,
            "atlas": "Destrieux fsaverage5",
            "score_range": [0, 1],
            "normalization": "per-category mean activation z-scored against global prediction distribution, then sigmoid-scaled",
        },
    }


def compute_interpretive_axes(cognitive_domains: dict) -> dict:
    series = cognitive_domains.get("time_series", {})
    if not series:
        return {"scores": {}, "time_series": {}, "metadata": {"available": False}}

    emotional = np.asarray(series.get("Emotional Engagement", []), dtype=np.float32)
    visual = np.asarray(series.get("Visual Attention", []), dtype=np.float32)
    auditory = np.asarray(series.get("Auditory Processing", []), dtype=np.float32)
    memory = np.asarray(series.get("Memory Encoding", []), dtype=np.float32)
    reward = np.asarray(series.get("Reward & Motivation", []), dtype=np.float32)
    language = np.asarray(series.get("Language Comprehension", []), dtype=np.float32)
    social = np.asarray(series.get("Social Cognition", []), dtype=np.float32)

    arousal = _sigmoid((_zscore(emotional) + _zscore(visual) + _zscore(auditory)) / 3)
    curiosity = _sigmoid((_zscore(memory) + _zscore(language) + _zscore(reward)) / 3)
    tension = _sigmoid((_zscore(emotional) + _zscore(social) - _zscore(reward)) / 3)
    rage_bait = _sigmoid((_zscore(emotional) + _zscore(tension) + _zscore(social)) / 3)
    reward_drive = _sigmoid((_zscore(reward) + _zscore(visual)) / 2)
    clarity = _sigmoid((_zscore(language) + _zscore(memory)) / 2)

    axes = {
        "Arousal / Intensity": arousal,
        "Curiosity Proxy": curiosity,
        "Tension / Conflict Proxy": tension,
        "Rage-Bait / Anger Proxy": rage_bait,
        "Reward Drive Proxy": reward_drive,
        "Message Clarity Proxy": clarity,
    }
    return {
        "scores": {name: float(values.mean()) for name, values in axes.items()},
        "time_series": {name: values.astype(float).tolist() for name, values in axes.items()},
        "metadata": {
            "available": True,
            "interpretive": True,
            "warning": (
                "These axes are heuristic interpretations derived from predicted cortical "
                "activation categories. They are not direct measurements of emotions."
            ),
            "score_range": [0, 1],
        },
    }

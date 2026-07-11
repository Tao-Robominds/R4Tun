"""T3 parameter hints: merge T1/T2 exemplar JSONs with per-tunnel T3 SAM geometry."""
from __future__ import annotations

import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = "opus4.6"
PARAM_SUFFIX = "_m_s_k_"

EXEMPLAR_ROOT = REPO_ROOT / "logs" / "{tunnel}" / "regular_hint" / MODEL / "parameters"
V3_ROOT = REPO_ROOT / "logs" / "{tunnel}" / "regular_hint_v3" / MODEL / "parameters"

CONTINUOUS_OVERRIDES = {
    "hough_threshold_horizontal": 50,
    "hough_threshold_oblique": 50,
    "maxLineGap_horizontal": 12,
    "k_consensus_version": "v3",
    "k_pattern_correction": "on",
    "k_pattern_outlier_tol_px": 150,
    "hint_mode": "off",
}

VARIANT_GRID: dict[str, dict] = {
    "base_v3": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {},
        "sam_flip": False,
    },
    "t2_detect": {
        "detecting_tunnel": "2-5",
        "detecting_source": "regular_hint",
        "sam_source": "v3",
        "detecting_overrides": {},
        "sam_flip": False,
    },
    "t2_best": {
        "detecting_tunnel": "2-2",
        "detecting_source": "regular_hint",
        "sam_source": "v3",
        "detecting_overrides": {},
        "sam_flip": False,
    },
    "t1_detect": {
        "detecting_tunnel": "1-5",
        "detecting_source": "regular_hint",
        "sam_source": "v3",
        "detecting_overrides": {},
        "sam_flip": False,
    },
    "hough_low": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
        },
        "sam_flip": False,
    },
    "gap_wide": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": False,
    },
    "consensus_tight": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {"k_pattern_outlier_tol_px": 120},
        "sam_flip": False,
    },
    "flip_on": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {},
        "sam_flip": True,
    },
    "hough_low_flip": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
        },
        "sam_flip": True,
    },
    "center_walk_312": {
        "detecting_tunnel": "3-1-2",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
        },
        "sam_flip": True,
        "flip_mode": "gt_handedness",
    },
    "center_walk_313": {
        "detecting_tunnel": "3-1-3",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": True,
        "flip_mode": "gt_ring_flip",
        "center_snap_after_pass1": True,
        "flip_preset_source": "pred_gt",
    },
    "center_walk_313_nosnap": {
        "detecting_tunnel": "3-1-3",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": True,
        "flip_mode": "gt_ring_flip",
    },
    "center_walk_313_mirror": {
        "detecting_tunnel": "3-1-3",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": True,
        "flip_mode": "gt_ring_flip",
        "flip_preset_source": "per_ring_mirror",
    },
    "geo_313": {
        "detecting_tunnel": "3-1-3",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": False,
        "sam_hint_override": "geometric_gt_k_flip",
    },
    "geo_313_flip": {
        "detecting_tunnel": "3-1-3",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": True,
        "flip_mode": "gt_ring_flip",
        "sam_hint_override": "geometric_gt_k",
    },
    "oracle_313_solo": {
        "detecting_tunnel": "3-1-3",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": False,
        "sam_oracle": "oracle_swap",
    },
    "cross_312_313": {
        "detecting_tunnel": "3-1-2",
        "detecting_source": "v3",
        "cross_tunnel_detect": True,
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
        },
        "sam_flip": True,
        "flip_mode": "gt_handedness",
    },
    "cross_311_313": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "cross_tunnel_detect": True,
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
        },
        "sam_flip": True,
        "flip_mode": "gt_ring_flip",
    },
    "cross_311_313_snap": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "cross_tunnel_detect": True,
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": True,
        "flip_mode": "gt_ring_flip",
        "center_snap_after_pass1": True,
        "flip_preset_source": "pred_gt",
    },
    "per_tunnel_313": {
        "detecting_tunnel": "3-1-3",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {
            "hough_threshold_horizontal": 40,
            "hough_threshold_oblique": 40,
            "maxLineGap_horizontal": 15,
            "maxLineGap_oblique": 50,
        },
        "sam_flip": True,
        "flip_mode": "gt_handedness",
        "center_snap_after_pass1": True,
        "flip_preset_source": "handedness",
        "per_tunnel": True,
    },
    "per_tunnel_v3": {
        "detecting_tunnel": "3-1-1",
        "detecting_source": "v3",
        "sam_source": "v3",
        "detecting_overrides": {},
        "sam_flip": True,
        "per_tunnel": True,
    },
}


def _param_path(tunnel: str, stage: str, source: str) -> Path:
    if source == "v3":
        root = Path(str(V3_ROOT).replace("{tunnel}", tunnel))
    else:
        root = Path(str(EXEMPLAR_ROOT).replace("{tunnel}", tunnel))
    return root / f"parameters_{stage}{PARAM_SUFFIX}{MODEL}.json"


def load_hint_json(tunnel: str, stage: str, source: str = "regular_hint") -> dict:
    path = _param_path(tunnel, stage, "v3" if source == "v3" else "regular_hint")
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def merge_t3_hints(
    target_tunnel: str,
    *,
    detecting_tunnel: str = "2-5",
    detecting_source: str = "regular_hint",
    sam_source: str = "v3",
    detecting_overrides: dict | None = None,
) -> tuple[dict, dict]:
    """Return (detecting_params, sam_params) adapted for a continuous T3 tunnel."""
    det_src = detecting_source
    det_tun = detecting_tunnel if det_src != "v3" else target_tunnel
    try:
        detecting = load_hint_json(det_tun, "detecting", det_src)
    except FileNotFoundError:
        detecting = load_hint_json("2-5", "detecting", "regular_hint")

    sam_tun = target_tunnel if sam_source == "v3" else detecting_tunnel
    try:
        sam = load_hint_json(sam_tun, "sam", sam_source)
    except FileNotFoundError:
        sam = load_hint_json(target_tunnel, "sam", "v3")

    detecting = copy.deepcopy(detecting)
    sam = copy.deepcopy(sam)
    detecting.update(CONTINUOUS_OVERRIDES)
    detecting["K_height"] = sam.get("K_height", 1174.95)
    detecting["AB_height"] = sam.get("AB_height", 3524.87)
    if detecting_overrides:
        detecting.update(detecting_overrides)
    return detecting, sam


def variant_spec(variant_id: str, target_tunnel: str) -> tuple[dict, dict]:
    spec = VARIANT_GRID.get(variant_id, VARIANT_GRID["base_v3"])
    if spec.get("per_tunnel"):
        detecting, sam = merge_t3_hints(
            target_tunnel,
            detecting_tunnel=target_tunnel,
            detecting_source="v3",
            sam_source="v3",
            detecting_overrides=spec.get("detecting_overrides"),
        )
        if target_tunnel == "3-1-1":
            detecting["hough_threshold_horizontal"] = 40
            detecting["hough_threshold_oblique"] = 40
        return detecting, sam
    det_tun = spec["detecting_tunnel"]
    if spec["detecting_source"] == "v3" and not spec.get("cross_tunnel_detect"):
        det_tun = target_tunnel
    sam_tun = target_tunnel
    return merge_t3_hints(
        sam_tun,
        detecting_tunnel=det_tun,
        detecting_source=spec["detecting_source"],
        sam_source=spec["sam_source"],
        detecting_overrides=spec.get("detecting_overrides"),
    )


def variant_ids() -> list[str]:
    return list(VARIANT_GRID.keys())

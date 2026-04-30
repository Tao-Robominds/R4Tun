"""Schema + prompt helpers for per-regime LLM warm start."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

# Canonical raster constraints from the ring-based BO-ready pipeline.
CANONICAL_TUNNEL_DIAMETER = 7.5
CANONICAL_DEPTH_MAP_RESOLUTION = 0.005
CANONICAL_NUM_SLICING_PLANES = 1

# (key, default, lo, hi, dtype, description)
PRE_SCHEMA: List[Tuple[str, Any, Any, Any, type, str]] = [
    ("ring_spacing", 1.816, 1.2, 2.4, float, "ring spacing in metres"),
    ("tunnel_diameter", CANONICAL_TUNNEL_DIAMETER, 6.0, 9.0, float, "physical diameter (forced canonical post-parse)"),
    ("depth_map_resolution", CANONICAL_DEPTH_MAP_RESOLUTION, 0.003, 0.01, float, "depth-map pixel resolution (forced canonical post-parse)"),
    ("radius_min", 3.0, 2.6, 3.8, float, "inner radial cutoff"),
    ("radius_max", 4.2, 3.5, 4.8, float, "outer radial cutoff"),
    ("gradient_threshold", 10.0, 0.05, 50.0, float, "denoise surface gradient threshold"),
    ("double_zero_cutoff", False, None, None, bool, "whether double-empty bins trigger denoise cutoff"),
    ("smoothing_offset", 0.0, -0.02, 0.02, float, "additive denoise smoothing offset"),
    ("target_distances", [0.08079722923035461, 0.040398614615177304, 0.02], None, None, list, "progressive upsample target distances in metres"),
    ("curvature_neighbors", 6, 4, 32, int, "PCA neighbors for curvature"),
    ("interpolation_window", 5, 1, 11, int, "main depth-map interpolation window"),
    ("num_slicing_planes", CANONICAL_NUM_SLICING_PLANES, 1, 7, int, "unfold slicing planes (forced to 1 post-parse)"),
    ("samples_per_ring", 1210, 600, 3000, int, "unfold samples per ring"),
    ("outlier_depth_threshold_low", 0.003, 0.0005, 0.02, float, "low-region outlier threshold"),
    ("outlier_depth_threshold_high", 0.008, 0.001, 0.03, float, "high-density outlier threshold"),
    ("outlier_high_density_ring_start", 0, -1, 10, int, "start ring index for high-density outlier mode"),
    ("outlier_high_density_ring_end", 5, -1, 15, int, "end ring index for high-density outlier mode"),
    ("outlier_neighbors", 20, 5, 80, int, "outlier neighborhood size"),
    ("max_outlier_points", 5000, 500, 25000, int, "max outlier points for interpolation"),
    ("outlier_interpolation_radius", 0.06, 0.01, 0.2, float, "radius for outlier interpolation"),
    ("outlier_num_interpolations", 2, 1, 10, int, "number of interpolation points"),
    ("outlier_duplicate_threshold", 0.02, 0.001, 0.08, float, "duplicate pruning threshold"),
    ("outlier_bidirectional", False, None, None, bool, "bidirectional outlier mode"),
    ("outlier_depth_map_window", 1, 1, 7, int, "outlier depth-map interpolation window"),
]

DET_SCHEMA: List[Tuple[str, Any, Any, Any, type, str]] = [
    ("binary_threshold", 139, 50, 220, int, "depth-map threshold for edge extraction"),
    ("hough_threshold", 37, 10, 250, int, "Hough vote threshold"),
    ("hough_min_length", 31, 5, 500, int, "min line length in pixels"),
    ("hough_max_gap", 133, 1, 600, int, "max line gap in pixels"),
    ("angle_pos_min", 4.84, -45.0, 45.0, float, "positive groove angle min"),
    ("angle_pos_max", 13.55, -45.0, 45.0, float, "positive groove angle max"),
    ("angle_neg_min", -14.67, -45.0, 45.0, float, "negative groove angle min"),
    ("angle_neg_max", -5.82, -45.0, 45.0, float, "negative groove angle max"),
    ("eps", 0.07, 0.01, 1.0, float, "DBSCAN epsilon"),
    ("k_expected_height_px", 300, 100, 1500, int, "expected K stripe height"),
    ("k_gap_tolerance_px", 150, 10, 1200, int, "K groove-pair gap tolerance"),
    ("groove_snap_px", 60, 1, 300, int, "groove snap radius"),
    ("ring_offset", 193.3, -1500.0, 1500.0, float, "ring offset in Y"),
    ("ring_spacing_px", -360.0, -2500.0, 2500.0, float, "ring spacing in pixels"),
    ("reverse_ring_order", True, None, None, bool, "reverse detected ring order"),
]

DET_PASS_THROUGH_KEYS = ("stagger_groups", "group_offsets", "per_ring_offsets")


def _coerce(value: Any, dtype: type) -> Any:
    if dtype is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"true", "1", "yes", "y"}:
                return True
            if s in {"false", "0", "no", "n"}:
                return False
        raise ValueError(f"cannot coerce {value!r} to bool")
    if dtype is int:
        return int(round(float(value)))
    if dtype is float:
        return float(value)
    if dtype is list:
        if not isinstance(value, list):
            raise ValueError(f"expected list, got {type(value).__name__}")
        return value
    if dtype is dict:
        if not isinstance(value, dict):
            raise ValueError(f"expected dict, got {type(value).__name__}")
        return value
    return value


def validate_and_clamp(parsed: Dict[str, Any], schema: List[Tuple[str, Any, Any, Any, type, str]]) -> tuple[Dict[str, Any], List[str]]:
    out: Dict[str, Any] = {}
    clamp_log: List[str] = []
    for key, default, lo, hi, dtype, _desc in schema:
        raw = parsed.get(key, default)
        try:
            val = _coerce(raw, dtype)
        except Exception:
            val = default
            clamp_log.append(f"{key}: invalid {raw!r}, fallback={default!r}")

        if dtype in (int, float) and lo is not None and hi is not None:
            orig = val
            if val < lo:
                val = lo
            if val > hi:
                val = hi
            if val != orig:
                clamp_log.append(f"{key}: clamped {orig!r} -> {val!r} [{lo}, {hi}]")

        if key == "target_distances":
            if not isinstance(val, list) or len(val) == 0:
                val = default
                clamp_log.append(f"{key}: invalid list, fallback={default!r}")
            else:
                vv = []
                for i, x in enumerate(val):
                    try:
                        xx = float(x)
                    except Exception:
                        xx = float(default[min(i, len(default) - 1)])
                    xx = max(0.005, min(0.2, xx))
                    vv.append(xx)
                val = sorted(vv, reverse=True)
                if len(val) > 6:
                    val = val[:6]
                    clamp_log.append(f"{key}: truncated to 6 entries")

        out[key] = val
    return out, clamp_log


def force_canonical_constraints(pre_params: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    if pre_params.get("depth_map_resolution") != CANONICAL_DEPTH_MAP_RESOLUTION:
        notes.append(
            f"depth_map_resolution forced {pre_params.get('depth_map_resolution')} -> {CANONICAL_DEPTH_MAP_RESOLUTION}"
        )
    if pre_params.get("tunnel_diameter") != CANONICAL_TUNNEL_DIAMETER:
        notes.append(
            f"tunnel_diameter forced {pre_params.get('tunnel_diameter')} -> {CANONICAL_TUNNEL_DIAMETER}"
        )
    if pre_params.get("num_slicing_planes") != CANONICAL_NUM_SLICING_PLANES:
        notes.append(
            f"num_slicing_planes forced {pre_params.get('num_slicing_planes')} -> {CANONICAL_NUM_SLICING_PLANES}"
        )
    pre_params["depth_map_resolution"] = CANONICAL_DEPTH_MAP_RESOLUTION
    pre_params["tunnel_diameter"] = CANONICAL_TUNNEL_DIAMETER
    pre_params["num_slicing_planes"] = CANONICAL_NUM_SLICING_PLANES
    return notes


def _schema_table(schema: List[Tuple[str, Any, Any, Any, type, str]]) -> str:
    rows = ["| key | type | default | range | description |", "|---|---|---|---|---|"]
    for key, default, lo, hi, dtype, desc in schema:
        rr = "n/a" if lo is None or hi is None else f"[{lo}, {hi}]"
        rows.append(
            f"| {key} | {dtype.__name__} | {json.dumps(default)} | {rr} | {desc} |"
        )
    return "\n".join(rows)


def build_prompt(regime_label: str, regime_stats: Dict[str, Any], pre_default: Dict[str, Any], det_default: Dict[str, Any]) -> str:
    return f"""You are generating a warm-start parameter guess for ring-based tunnel segmentation detection.

Task:
- Produce ONE JSON object with two top-level keys:
  - "preprocessing": dict
  - "detection": dict
- This is zero-shot by regime descriptors (no trial-and-error, no BO, no GT fitting).
- Prefer conservative but plausible values near defaults unless regime stats justify a shift.

Regime label:
- {regime_label}

Regime summary statistics (medians from historical rings in this regime):
{json.dumps(regime_stats, indent=2)}

Current defaults (reference):
preprocessing_default = {json.dumps(pre_default, indent=2)}
detection_default = {json.dumps(det_default, indent=2)}

Allowed preprocessing keys:
{_schema_table(PRE_SCHEMA)}

Allowed detection keys:
{_schema_table(DET_SCHEMA)}

Additional detection pass-through keys (optional):
- stagger_groups (dict)
- group_offsets (dict)
- per_ring_offsets (dict)

Output requirements:
1) Return STRICT JSON only (no markdown, no explanation).
2) Include every key in PRE_SCHEMA and DET_SCHEMA.
3) You may include pass-through detection keys if needed; otherwise omit them.
4) Keep values physically meaningful.
"""

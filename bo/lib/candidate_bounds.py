"""Structural validation and clipping for candidate layouts."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from lib.layout_bo import RingContext, decode_x, offsets_to_arc_widths, _coerce_search_x
from lib.search_space import LAYOUT_RECOVERY_PARAMS, encode_r_surface_min


MIN_ARC_WIDTH_FRAC = 0.02
DEFAULT_MAX_K_SHIFT = 0.25


def max_k_shift_from_rules(v3_rules: dict[str, Any]) -> float:
    p90 = v3_rules.get("reject_k_shift_norm_abs_p90")
    if p90 is not None and np.isfinite(float(p90)):
        return float(min(DEFAULT_MAX_K_SHIFT, float(p90)))
    return DEFAULT_MAX_K_SHIFT


def validate_candidate(
    ctx: RingContext,
    x: np.ndarray,
    *,
    sam_k_center_norm: float,
    v3_rules: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    try:
        x = _coerce_search_x(ctx, np.asarray(x, dtype=float))
    except Exception as exc:
        return False, f"invalid search_x: {exc}"

    k_y, offsets, layout, _r = decode_x(ctx, x)
    H = ctx.H
    k_norm = float(k_y / max(H, 1))

    if abs(float(x[1])) > 1e-6 and ctx.blocks[0] == "K":
        pass  # K offset slot should be 0 — soft check only

    widths = offsets_to_arc_widths(ctx.blocks, offsets, H)
    if float(np.min(widths)) < MIN_ARC_WIDTH_FRAC * H:
        return False, "arc width below minimum"

    max_shift = max_k_shift_from_rules(v3_rules or {})
    if abs(k_norm - sam_k_center_norm) > max_shift:
        return False, f"k shift {abs(k_norm - sam_k_center_norm):.4f} > {max_shift}"

    lo = v3_rules.get("reject_k_center_norm_lo") if v3_rules else None
    hi = v3_rules.get("reject_k_center_norm_hi") if v3_rules else None
    if lo is not None and hi is not None and float(lo) <= k_norm <= float(hi):
        if abs(k_norm - sam_k_center_norm) > max_shift * 0.5:
            return False, "k in failure band with large shift from SAM"

    for spec in LAYOUT_RECOVERY_PARAMS:
        val = layout.get(spec.name)
        if val is None:
            alt = spec.name.replace("hough_threshold", "hough_oblique_threshold")
            val = layout.get(alt)
        if val is not None and not (spec.lo <= float(val) <= spec.hi):
            return False, f"form param {spec.name} out of range"

    return True, "ok"


def clip_form_to_ranges(form: dict[str, float], ranges: dict[str, Any]) -> dict[str, float]:
    out = dict(form)
    mapping = {
        "hough_threshold": "hough_oblique_threshold",
        "hough_horizontal_threshold": "hough_horizontal_threshold",
        "merge_distance_threshold": "line_merge_distance",
        "single_ring_visual_slot_snap_px": "line_snap_tolerance_px",
        "slot_inset_y": "segmentation_slot_inset_y",
    }
    for form_key, range_key in mapping.items():
        band = ranges.get(range_key)
        if not band or not isinstance(band, dict):
            continue
        if form_key in out:
            out[form_key] = float(np.clip(out[form_key], band.get("lo", out[form_key]), band.get("hi", out[form_key])))
    r_band = ranges.get("r_surface_min_frac")
    if r_band and "r_surface_min_frac" in out:
        out["r_surface_min_frac"] = float(np.clip(out["r_surface_min_frac"], r_band["lo"], r_band["hi"]))
    return out


def should_penalise(
    evidence_coverage: float,
    k_confidence: float,
    rho_K: float,
    v3_rules: dict[str, Any],
) -> bool:
    cov_min = float(v3_rules.get("penalise_coverage_min", 90.0))
    k_max = float(v3_rules.get("penalise_k_confidence_max", 1.0))
    return evidence_coverage >= cov_min and k_confidence <= k_max and rho_K < 0.5

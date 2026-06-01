"""Layout-recovery BO search space (v2: includes searchable r_surface_min).

Encoded vector:
  x = [k_y_frac, off_frac[K], off_frac[B1], …, layout_param_frac…, r_surface_min_frac]

Layout tail (5 normalized dims → physical values):
  - hough_threshold, hough_horizontal_threshold, merge_distance_threshold
  - single_ring_visual_slot_snap_px, slot_inset_y

Final dim: r_surface_min_frac → ring-adaptive [r_lo, r_hi] (intrinsic radial band).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

LOG_COLUMN_ALIASES = {
    "hough_threshold": "hough_oblique_threshold",
    "hough_horizontal_threshold": "hough_horizontal_threshold",
    "merge_distance_threshold": "line_merge_distance",
    "single_ring_visual_slot_snap_px": "line_snap_tolerance_px",
    "slot_inset_y": "segmentation_slot_inset_y",
}


@dataclass(frozen=True)
class ParamSpec:
    name: str
    stage: str  # "detection" | "segmentation"
    default: float
    lo: float
    hi: float

    def encode(self, value: float) -> float:
        span = max(self.hi - self.lo, 1e-9)
        return float(np.clip((value - self.lo) / span, 0.0, 1.0))

    def decode(self, frac: float) -> float:
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(self.lo + frac * (self.hi - self.lo))


LAYOUT_RECOVERY_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("hough_threshold", "detection", 37.0, 15.0, 90.0),
    ParamSpec("hough_horizontal_threshold", "detection", 50.0, 10.0, 120.0),
    ParamSpec("merge_distance_threshold", "detection", 3.0, 1.0, 12.0),
    ParamSpec("single_ring_visual_slot_snap_px", "detection", 20.0, 5.0, 80.0),
    ParamSpec("slot_inset_y", "segmentation", 0.0, 0.0, 25.0),
)

N_LAYOUT_TAIL = len(LAYOUT_RECOVERY_PARAMS)
N_R_SURFACE = 1


def search_dim(segment_count: int) -> int:
    """k_y + per-block offsets + layout tail + r_surface_min."""
    return 1 + int(segment_count) + N_LAYOUT_TAIL + N_R_SURFACE


def v1_search_dim(segment_count: int) -> int:
    """Step-2 vector without searchable r_surface_min."""
    return 1 + int(segment_count) + N_LAYOUT_TAIL


def encode_r_surface_min(value: float, r_lo: float, r_hi: float) -> float:
    span = max(float(r_hi) - float(r_lo), 1e-9)
    return float(np.clip((float(value) - float(r_lo)) / span, 0.0, 1.0))


def decode_r_surface_min(frac: float, r_lo: float, r_hi: float) -> float:
    frac = float(np.clip(frac, 0.0, 1.0))
    return round(float(r_lo) + frac * (float(r_hi) - float(r_lo)), 4)


def default_layout_fracs() -> np.ndarray:
    return np.array([p.encode(p.default) for p in LAYOUT_RECOVERY_PARAMS], dtype=float)


def r_surface_frac_index(segment_count: int) -> int:
    return 1 + int(segment_count) + N_LAYOUT_TAIL


def decode_layout_params(x: np.ndarray, segment_count: int) -> dict[str, float]:
    start = 1 + int(segment_count)
    tail = np.asarray(x[start : start + N_LAYOUT_TAIL], dtype=float)
    if tail.size != N_LAYOUT_TAIL:
        raise ValueError(f"Expected {N_LAYOUT_TAIL} layout dims, got {tail.size}")
    out: dict[str, float] = {}
    for spec, frac in zip(LAYOUT_RECOVERY_PARAMS, tail):
        val = spec.decode(float(frac))
        if spec.name in ("hough_threshold", "hough_horizontal_threshold"):
            val = float(int(round(val)))
        out[spec.name] = round(val, 4) if spec.name not in ("hough_threshold", "hough_horizontal_threshold") else val
    return out


def decode_r_surface_frac(x: np.ndarray, segment_count: int) -> float:
    idx = r_surface_frac_index(segment_count)
    x = np.asarray(x, dtype=float).ravel()
    if x.size <= idx:
        raise ValueError(f"search_x missing r_surface_min_frac at index {idx}")
    return float(x[idx])


def layout_params_for_log(layout: dict[str, float]) -> dict[str, Any]:
    logged: dict[str, Any] = {}
    for spec in LAYOUT_RECOVERY_PARAMS:
        val = layout[spec.name]
        logged[spec.name] = val
        logged[LOG_COLUMN_ALIASES[spec.name]] = val
    return logged


def search_space_summary(
    segment_count: int,
    *,
    r_lo: float | None = None,
    r_hi: float | None = None,
    r_otsu_ref: float | None = None,
    r_ceiling_ref: float | None = None,
) -> dict[str, Any]:
    r_bounds = None
    if r_lo is not None and r_hi is not None:
        r_bounds = {
            "lo": round(float(r_lo), 4),
            "hi": round(float(r_hi), 4),
            "otsu_ref": round(float(r_otsu_ref), 4) if r_otsu_ref is not None else None,
            "ceiling_ref": round(float(r_ceiling_ref), 4) if r_ceiling_ref is not None else None,
        }
    return {
        "segment_count": segment_count,
        "search_dim": search_dim(segment_count),
        "layout_variables": [
            "k_y (K position)",
            "per_ring_offsets (A/B offsets)",
            *[f"{p.name} ({p.lo}–{p.hi}, default={p.default})" for p in LAYOUT_RECOVERY_PARAMS],
            f"r_surface_min (ring-adaptive r_lo–r_hi)",
        ],
        "excluded": "preprocessing, binary_threshold, full SAM4Tun space",
        "r_surface_min_bounds": r_bounds,
        "params": [
            {
                "name": p.name,
                "log_column": LOG_COLUMN_ALIASES[p.name],
                "stage": p.stage,
                "lo": p.lo,
                "hi": p.hi,
                "default": p.default,
            }
            for p in LAYOUT_RECOVERY_PARAMS
        ],
    }

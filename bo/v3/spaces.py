"""Ax search-space definitions for v3 BO calibration.

The two stages are:

* :func:`preprocessing_space` — 13 parameters covering radius gating,
  gradient/smoothing, target distances for the up-sampler, outlier interp.
* :func:`detection_space` — Hough/Canny/dilation/eps/k_expected_height_px
  plus per-block offsets when the trial runs in single-ring local mode.

Bounds are ported from the v1/v2 skopt declarations in
``stages/v2/bo/run_preprocessing_iou_bo.py`` and
``stages/v2/bo/run_detection_boundary_bo.py`` and widened (where
needed) to encompass both the R4Tun seed values and the irregular
regimes BO has to discover.
"""

from __future__ import annotations

from typing import Any

# --- Preprocessing ----------------------------------------------------------

PREPROCESSING_PARAMETERS: list[dict[str, Any]] = [
    {"name": "radius_min", "type": "range", "bounds": [1.8, 3.8], "value_type": "float"},
    {"name": "radius_max", "type": "range", "bounds": [2.0, 4.4], "value_type": "float"},
    {"name": "gradient_threshold", "type": "range", "bounds": [0.03, 0.40], "value_type": "float"},
    {"name": "smoothing_offset", "type": "range", "bounds": [-0.02, 0.02], "value_type": "float"},
    {"name": "curvature_neighbors", "type": "range", "bounds": [8, 40], "value_type": "int"},
    {"name": "interpolation_window", "type": "range", "bounds": [1, 15], "value_type": "int"},
    {"name": "target_distance_1", "type": "range", "bounds": [0.03, 0.12], "value_type": "float"},
    {"name": "target_distance_2", "type": "range", "bounds": [0.015, 0.06], "value_type": "float"},
    {"name": "target_distance_3", "type": "range", "bounds": [0.008, 0.04], "value_type": "float"},
    {"name": "outlier_interpolation_radius", "type": "range", "bounds": [0.01, 0.08], "value_type": "float"},
    {"name": "outlier_num_interpolations", "type": "range", "bounds": [1, 5], "value_type": "int"},
    {"name": "outlier_depth_map_window", "type": "range", "bounds": [1, 9], "value_type": "int"},
    {"name": "outlier_neighbors", "type": "range", "bounds": [8, 40], "value_type": "int"},
]


def preprocessing_space() -> list[dict[str, Any]]:
    """Return a fresh copy of the preprocessing parameter spec."""
    return [dict(p) for p in PREPROCESSING_PARAMETERS]


def preprocessing_constraints() -> list[str]:
    """Linear / inequality constraints for the preprocessing space.

    ``radius_max`` must be strictly above ``radius_min`` (the agents
    pipeline collapses the radius gate when this fails). Ax's
    parameter_constraints accept linear forms ``"a * x + b * y <= c"``;
    we encode ``radius_min - radius_max <= -0.05``.
    """
    return ["radius_min - radius_max <= -0.05"]


# --- Detection --------------------------------------------------------------

# Core detection parameters (always active).
DETECTION_CORE_PARAMETERS: list[dict[str, Any]] = [
    {"name": "binary_threshold", "type": "range", "bounds": [60, 220], "value_type": "int"},
    {"name": "hough_threshold", "type": "range", "bounds": [10, 120], "value_type": "int"},
    {"name": "hough_min_length", "type": "range", "bounds": [10, 200], "value_type": "int"},
    {"name": "hough_max_gap", "type": "range", "bounds": [5, 180], "value_type": "int"},
    {"name": "angle_pos_min", "type": "range", "bounds": [3.0, 14.0], "value_type": "float"},
    {"name": "angle_pos_max", "type": "range", "bounds": [5.0, 18.0], "value_type": "float"},
    {"name": "angle_neg_min", "type": "range", "bounds": [-18.0, -5.0], "value_type": "float"},
    {"name": "angle_neg_max", "type": "range", "bounds": [-14.0, -3.0], "value_type": "float"},
    {"name": "eps", "type": "range", "bounds": [0.03, 0.20], "value_type": "float"},
    {"name": "k_expected_height_px", "type": "range", "bounds": [120.0, 700.0], "value_type": "float"},
    {"name": "canny_low", "type": "range", "bounds": [20, 180], "value_type": "int"},
    {"name": "canny_high", "type": "range", "bounds": [80, 255], "value_type": "int"},
    {"name": "dilation_kernel_size", "type": "range", "bounds": [2, 5], "value_type": "int"},
    {"name": "dilation_iterations", "type": "range", "bounds": [1, 3], "value_type": "int"},
]


# Per-block offset parameters, only active when single_ring_local detector
# mode is selected (which it always is for our v3 single-ring calibration
# rings). Offsets are signed pixel distances from the detected K row to
# the corresponding block's start row in the unwrapped depth map.
DETECTION_OFFSET_PARAMETERS: list[dict[str, Any]] = [
    {"name": "offset_K", "type": "range", "bounds": [-50.0, 50.0], "value_type": "float"},
    {"name": "offset_B1", "type": "range", "bounds": [-1500.0, 1500.0], "value_type": "float"},
    {"name": "offset_B2", "type": "range", "bounds": [-2500.0, 2500.0], "value_type": "float"},
    {"name": "offset_A1", "type": "range", "bounds": [-3500.0, 3500.0], "value_type": "float"},
    {"name": "offset_A2", "type": "range", "bounds": [-3500.0, 3500.0], "value_type": "float"},
    {"name": "offset_A3", "type": "range", "bounds": [-3500.0, 3500.0], "value_type": "float"},
    {"name": "offset_A4", "type": "range", "bounds": [-3500.0, 3500.0], "value_type": "float"},
]


def detection_space(*, include_offsets: bool = True, include_a4: bool = True) -> list[dict[str, Any]]:
    """Return a fresh copy of the detection parameter spec.

    Parameters
    ----------
    include_offsets:
        If True, add the per-block ``offset_*`` parameters to the search
        space (use when the trial runs single-ring local detection mode,
        which is what the v3 calibration rings always do).
    include_a4:
        If True, include ``offset_A4`` for 7-segment K-bearing rings.
        All v3 calibration rings are 7-segment, so the default keeps it on.
    """
    out = [dict(p) for p in DETECTION_CORE_PARAMETERS]
    if include_offsets:
        for p in DETECTION_OFFSET_PARAMETERS:
            if p["name"] == "offset_A4" and not include_a4:
                continue
            out.append(dict(p))
    return out


def detection_constraints() -> list[str]:
    """Linear constraints for the detection space.

    Enforce ``angle_pos_min < angle_pos_max`` and
    ``angle_neg_min < angle_neg_max`` (the line-angle filters need a
    non-empty acceptance band) and ``canny_low < canny_high``.
    """
    return [
        "angle_pos_min - angle_pos_max <= -0.5",
        "angle_neg_min - angle_neg_max <= -0.5",
        "canny_low - canny_high <= -1",
    ]

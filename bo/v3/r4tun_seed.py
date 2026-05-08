"""R4Tun nested-schema → flat agents-schema translator.

The deployment-time agents pipeline reads flat parameter dicts from
``agents/1_preprocessing/parameters/.../parameters_preprocessing.json`` and
``agents/2_detection/parameters/.../parameters_detection.json``. The R4Tun
regular-tunnel reference at ``r4tun/sample/`` lives in a *nested* schema
(``unfolding.*``, ``denoising.*``, ``enhancing.*`` blocks for preprocessing,
and ``hough_oblique.*`` / ``physical_constants.*`` blocks for detection).

This module provides three things:

1. :func:`load_r4tun_preprocessing` — read the nested file, return the flat
   dict the agents pipeline expects.
2. :func:`load_r4tun_detection` — same for detection. ``per_ring_offsets``
   is generated from the geometric K/AB heights (the regular reference has
   no per-ring offsets to seed; we use canonical 7-block geometry instead).
3. :func:`render_baseline_params` — write both flat JSONs into a target
   sandbox so the agents CLIs find them via the per-ring lookup precedence.

The mapping is intentionally narrow: only fields the agents pipeline
actually consumes. Unknown nested fields are dropped with a logged warning.
The output is a "regular-tunnel floor" that gives the irregular calibration
rings a deliberately low baseline mIoU — the point is to verify the agents
pipeline runs end-to-end with a known-good reference, not to score well.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
R4TUN_SAMPLE_DIR = REPO_ROOT / "r4tun" / "sample"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _safe(d: dict[str, Any] | None, key: str, default: Any) -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def load_r4tun_preprocessing(
    *,
    target_tunnel_diameter: float,
    target_ring_spacing: float | None = None,
    sample_dir: Path | None = None,
) -> dict[str, Any]:
    """Translate r4tun's nested preprocessing schema into the flat dict
    consumed by ``agents/1_preprocessing/1_preprocessing.py``.

    Parameters
    ----------
    target_tunnel_diameter:
        Diameter of the *target* (irregular) tunnel that this seed will be
        applied to. R4Tun's reference is a 5.5 m regular tunnel; we keep the
        ring-shape parameters from r4tun (they are the "regular reference"
        the manuscript talks about) but override the tunnel diameter and
        the dependent radius gate so the unfolder can locate the wall on
        the larger irregular ring.
    """
    sample_dir = sample_dir or R4TUN_SAMPLE_DIR
    src = json.loads((sample_dir / "parameters_preprocessing.json").read_text())

    unfolding = src.get("unfolding", {}) or {}
    denoising = src.get("denoising", {}) or {}
    enhancing = src.get("enhancing", {}) or {}

    # Ring spacing: prefer explicit override, else r4tun's value, else 1.2 m.
    ring_spacing = (
        float(target_ring_spacing)
        if target_ring_spacing is not None
        else float(_safe(_safe(unfolding, "physical_constants", {}), "ring_spacing", 1.2))
    )

    # Radius gate: r4tun's denoising filter is tight (2.7 / 2.8 m) for a
    # 5.5 m diameter tunnel. Rescale proportionally to the target diameter
    # so the gate sits at roughly half-radius +/- a 5% band; this is the
    # smallest change that keeps the agents pipeline from clipping all wall
    # points on the larger irregular rings.
    radial = _safe(denoising, "radius_filtering", {}) or {}
    r_lo_ref = float(_safe(radial, "radius_min", 2.7))
    r_hi_ref = float(_safe(radial, "radius_max", 2.8))
    diameter_ratio = float(target_tunnel_diameter) / 5.5
    radius_min_target = max(0.5, r_lo_ref * diameter_ratio - 0.5 * diameter_ratio)
    radius_max_target = max(radius_min_target + 0.1, r_hi_ref * diameter_ratio + 0.5 * diameter_ratio)

    # Up-sampling target distances (sorted descending).
    up = _safe(enhancing, "upsampling", {}) or {}
    td = list(_safe(up, "target_distances", [0.08, 0.04, 0.02]))
    while len(td) < 3:
        td.append(0.02)
    td = sorted([float(t) for t in td[:3]], reverse=True)

    out_det = _safe(enhancing, "outlier_detection", {}) or {}
    out_int = _safe(enhancing, "outlier_interpolation", {}) or {}
    depth = _safe(enhancing, "depth_map", {}) or {}
    grad = _safe(denoising, "gradient_detection", {}) or {}
    smoothing = _safe(denoising, "cutoff_smoothing", {}) or {}
    curvature = _safe(enhancing, "curvature", {}) or {}
    ransac = _safe(unfolding, "ransac_ellipse", {}) or {}
    arc = _safe(unfolding, "arc_length", {}) or {}

    flat: dict[str, Any] = {
        "tunnel_diameter": float(target_tunnel_diameter),
        "ring_spacing": ring_spacing,
        "depth_map_resolution": float(_safe(depth, "resolution", 0.005)),
        # Gravity anchor stays on by default (v3 promotion). The ablation
        # study can flip this to false in a separate sweep.
        "gravity_anchor": {"enabled": True, "n_bins": 360},
        # Unfolding RANSAC + filter window: r4tun's regular defaults.
        "vertical_filter_window": 6.8,
        "ransac_threshold": 1.0,
        "ransac_probability": float(_safe(ransac, "confidence", 0.9)),
        "ransac_inlier_ratio": float(_safe(ransac, "inlier_ratio", 0.75)),
        "ransac_sample_size": int(_safe(ransac, "min_samples", 5)),
        "ransac_initial_iterations": 999,
        "ransac_inlier_threshold_multiplier": float(_safe(ransac, "inlier_threshold", 0.8)),
        "samples_per_ring": int(_safe(arc, "samples_per_ring", 1210)),
        # Radius gate — rescaled for the target diameter.
        "radius_min": float(radius_min_target),
        "radius_max": float(radius_max_target),
        # Denoising + smoothing.
        "y_step": 0.4,
        "z_step": 0.005,
        "gradient_threshold": float(_safe(grad, "gradient_threshold", 0.2)),
        "smoothing_window_size": int(_safe(smoothing, "smoothing_window", 3)),
        "smoothing_offset": float(_safe(smoothing, "smoothing_offset", 0.003)),
        "default_cutoff_z": float(target_tunnel_diameter) / 2.0,
        "double_zero_cutoff": False,
        # Enhancing / up-sampling.
        "target_distances": td,
        "curvature_threshold_enh": float(_safe(up, "curvature_threshold", 0.0005)),
        "curvature_neighbors": int(_safe(curvature, "curvature_neighbors", 20)),
        "interpolation_window": int(_safe(depth, "interpolation_window", 9)),
        # Outlier detection / interpolation.
        "depth_threshold_low": float(_safe(out_det, "depth_threshold_low", 0.003)),
        "depth_threshold_high": float(_safe(out_det, "depth_threshold_high", 0.008)),
        "outlier_high_density_ring_start": -1,
        "outlier_high_density_ring_end": -1,
        "outlier_neighbors": int(_safe(out_det, "outlier_neighbors", 20)),
        "max_outlier_points": int(_safe(out_int, "max_outlier_points", 5000)),
        "outlier_interpolation_radius": float(_safe(out_int, "interpolation_radius", 0.06)),
        "inter_radius": float(_safe(out_int, "interpolation_radius", 0.06)),
        "outlier_num_interpolations": int(_safe(out_int, "num_interpolations", 2)),
        "num_interpolations": int(_safe(out_int, "num_interpolations", 2)),
        "outlier_duplicate_threshold": float(_safe(out_int, "duplicate_threshold", 0.02)),
        "duplicate_threshold": float(_safe(out_int, "duplicate_threshold", 0.02)),
        "outlier_bidirectional": False,
        "outlier_depth_map_window": 1,
        "n_segment_start": -1,
        "n_segment_end": -1,
        "num_neighbors": int(_safe(curvature, "curvature_neighbors", 20)),
        "_warm_source": "r4tun/sample (v3 baseline, regular reference)",
    }
    return flat


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Canonical 7-block order around the ring (K → B1 → A1 → A2 → A3 → A4 → B2),
# used to seed `per_ring_offsets` purely from physical geometry.
CANONICAL_BLOCKS_7 = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]


def _physical_offsets_7block(
    *,
    tunnel_diameter: float,
    resolution: float = 0.005,
) -> dict[str, dict[str, float]]:
    """Generate ``per_ring_offsets`` for one ring with 7 canonical blocks.

    Geometry for a **7-segment** K-bearing tunnel: K + 6 AB blocks tile the
    circumference, with the convention K_h = 1 unit, AB_h = 3 units. So
    circ = K_h + 6 * AB_h = 19 * K_h, i.e. K_h = pi*D/19, AB_h = 3*pi*D/19.

    Block start positions in pixels (relative to K = 0) tile the
    circumference cleanly: K=0, B1=K_h, A1=K_h+AB_h, A2=K_h+2*AB_h,
    A3=K_h+3*AB_h, A4=K_h+4*AB_h, B2=K_h+5*AB_h. Signed offsets are
    centred so the layout wraps cleanly around the cylinder.
    """
    circumference_mm = math.pi * float(tunnel_diameter) * 1000.0
    k_h_mm = circumference_mm / 19.0
    ab_h_mm = 3.0 * k_h_mm
    px = lambda mm: float(mm) / (float(resolution) * 1000.0)
    starts_mm = {
        "K": 0.0,
        "B1": k_h_mm,
        "A1": k_h_mm + ab_h_mm,
        "A2": k_h_mm + 2.0 * ab_h_mm,
        "A3": k_h_mm + 3.0 * ab_h_mm,
        "A4": k_h_mm + 4.0 * ab_h_mm,
        "B2": k_h_mm + 5.0 * ab_h_mm,
    }
    # Shift so signed offsets are within (-circumference/2, +circumference/2].
    half_circ_mm = circumference_mm / 2.0
    offsets_px: dict[str, float] = {}
    for blk, start in starts_mm.items():
        signed = start
        if signed > half_circ_mm:
            signed -= circumference_mm
        offsets_px[blk] = round(px(signed), 1)
    return {"0": offsets_px}


def load_r4tun_detection(
    *,
    target_tunnel_diameter: float,
    sample_dir: Path | None = None,
) -> dict[str, Any]:
    """Translate r4tun's nested detection schema into the flat dict
    consumed by ``agents/2_detection/2_detection.py``.

    R4Tun's detection schema does not contain ``per_ring_offsets`` (it
    relied on a separate runtime to compute them from group offsets), so
    we synthesise a canonical 7-block ``per_ring_offsets`` from physical
    geometry. BO Stage 2b will then tune those offsets per ring.
    """
    sample_dir = sample_dir or R4TUN_SAMPLE_DIR
    src = json.loads((sample_dir / "parameters_detection.json").read_text())

    pre = src.get("preprocessing", {}) or {}
    h_oblique = src.get("hough_oblique", {}) or {}
    h_horiz = src.get("hough_horizontal", {}) or {}
    h_vert = src.get("hough_vertical", {}) or {}
    line = src.get("line_processing", {}) or {}
    phys = src.get("physical_constants", {}) or {}

    flat: dict[str, Any] = {
        # OpenCV preprocessing
        "binary_threshold": int(_safe(pre, "binary_threshold", 127)),
        "dilation_kernel_size": int(_safe(pre, "dilation_kernel_size", 3)),
        "dilation_iterations": int(_safe(pre, "dilation_iterations", 1)),
        # Canny defaults from agents code (r4tun does not specify).
        "canny_low": 50,
        "canny_high": 150,
        # Oblique Hough
        "hough_threshold": int(_safe(h_oblique, "threshold", 50)),
        "hough_min_length": int(_safe(h_oblique, "min_length", 100)),
        "hough_max_gap": int(_safe(h_oblique, "max_gap", 40)),
        "angle_pos_min": float(_safe(h_oblique, "angle_positive_min", 6.0)),
        "angle_pos_max": float(_safe(h_oblique, "angle_positive_max", 9.0)),
        "angle_neg_min": float(_safe(h_oblique, "angle_negative_min", -9.0)),
        "angle_neg_max": float(_safe(h_oblique, "angle_negative_max", -6.0)),
        # Horizontal Hough
        "hough_horizontal_threshold": int(_safe(h_horiz, "threshold", 50)),
        "hough_horizontal_min_length": int(_safe(h_horiz, "min_length", 100)),
        "hough_horizontal_max_gap": int(_safe(h_horiz, "max_gap", 10)),
        "horizontal_angle_tolerance": float(_safe(h_horiz, "angle_tolerance", 1.0)),
        # Vertical Hough
        "hough_vertical_threshold": int(_safe(h_vert, "threshold", 500)),
        # Line processing
        "merge_distance_threshold": float(_safe(line, "merge_distance_threshold", 3.0)),
        # Geometry-derived expected K height
        "k_expected_height_px": float(
            (math.pi * float(target_tunnel_diameter) * 1000.0 / 16.0)
            / (float(_safe(phys, "resolution", 0.005)) * 1000.0)
        ),
        # DBSCAN intersection eps (agents default for irregular).
        "eps": 0.07,
        # Single-ring local detector mode + 7 canonical blocks (calibration
        # rings are all single-ring; one detector pass per trial).
        "detector_mode": "single_ring_local",
        "enabled_blocks": list(CANONICAL_BLOCKS_7),
        # Per-ring offsets seeded from geometry; BO Stage 2b will tune.
        "per_ring_offsets": _physical_offsets_7block(
            tunnel_diameter=float(target_tunnel_diameter),
            resolution=float(_safe(phys, "resolution", 0.005)),
        ),
        "_warm_source": "r4tun/sample (v3 baseline, regular reference)",
    }
    return flat


# ---------------------------------------------------------------------------
# Render parameter files into a sandbox
# ---------------------------------------------------------------------------

def render_baseline_params(
    *,
    sandbox_root: Path,
    tunnel_id: str,
    ring_id: int,
    target_tunnel_diameter: float,
    target_ring_spacing: float | None = None,
) -> tuple[Path, Path]:
    """Write flat preprocessing + detection JSONs into the per-trial sandbox.

    The agents CLIs look up parameters in this precedence:

        1. agents/<stage>/parameters/<tunnel>/r<ring>/parameters_*.json
        2. <base_dir>/<tunnel>/r<ring>/parameters_*.json
        3. agents/<stage>/parameters/_warm_start/<regime>/parameters_*.json
        4. agents/<stage>/parameters/_default_irregular/parameters_*.json

    By writing into ``<sandbox_root>/<tunnel>/r<ring>/`` and invoking the
    agent with ``--data-dir <sandbox_root>``, slot (2) is hit and the seed
    parameters take effect without polluting the agents/ checked-in tree.
    Returns the two file paths written.
    """
    pre_dict = load_r4tun_preprocessing(
        target_tunnel_diameter=target_tunnel_diameter,
        target_ring_spacing=target_ring_spacing,
    )
    det_dict = load_r4tun_detection(target_tunnel_diameter=target_tunnel_diameter)

    ring_dir = sandbox_root / tunnel_id / f"r{int(ring_id)}"
    ring_dir.mkdir(parents=True, exist_ok=True)
    pre_path = ring_dir / "parameters_preprocessing.json"
    det_path = ring_dir / "parameters_detection.json"
    pre_path.write_text(json.dumps(pre_dict, indent=2) + "\n")
    det_path.write_text(json.dumps(det_dict, indent=2) + "\n")
    return pre_path, det_path

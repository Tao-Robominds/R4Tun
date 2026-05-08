#!/usr/bin/env python3
"""
Ring-native preprocessing: ellipse unfolding + r4tun-style denoising/enhancing.

Outputs per ring (same contract as before):
  unwrapped.csv, denoised.csv, enhanced.csv, ring_count.txt (=1),
  depth_map.png, depth_map_outlier.npy, pixel_to_point.pkl

Run::

    ./venv/bin/python agents/1_preprocessing/1_preprocessing.py 4-1 110

Warm-start JSON from r4tun references::

    ./venv/bin/python agents/1_preprocessing/scripts/warm_from_r4tun.py 4-1 110
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ring_denoising import denoise_ring  # noqa: E402
from _ring_enhancing import run_ring_enhancing  # noqa: E402
from _ring_unfolding import unfold_single_ring  # noqa: E402

RESOLUTION_M = 0.005


def load_parameters(
    tunnel_id: Optional[str] = None,
    ring_id: Optional[int] = None,
    regime_label: Optional[str] = None,
    base_dir: str = "data",
) -> Tuple[Dict[str, Any], bool]:
    """Load parameters_preprocessing.json (per-ring, warm-start, or default)."""
    param_file = "parameters_preprocessing.json"
    script_dir = str(SCRIPT_DIR)

    # When INTRINSIC_PARAMS_BASE_DIR_ONLY=1 (set by the v3 BO driver) we
    # skip the agents/.../parameters/<tunnel>/r<ring>/ and warm-start
    # lookups so per-trial sandbox params are not shadowed by checked-in
    # v1/v2-tuned per-ring overrides.
    base_only = os.environ.get("INTRINSIC_PARAMS_BASE_DIR_ONLY") == "1"

    if tunnel_id is not None and ring_id is not None:
        ring_key = f"r{int(ring_id)}"
        candidates = []
        if not base_only:
            candidates.append(os.path.join(script_dir, "parameters", tunnel_id, ring_key, param_file))
        candidates.append(os.path.join(base_dir, tunnel_id, ring_key, param_file))
        for p in candidates:
            if os.path.exists(p):
                print(f"Loading parameters from {p}")
                with open(p, encoding="utf-8") as f:
                    return json.load(f), True

    if regime_label and not base_only:
        warm_path = os.path.join(
            script_dir, "parameters", "_warm_start", str(regime_label), param_file
        )
        if os.path.exists(warm_path):
            print(f"Loading warm-start parameters from {warm_path}")
            with open(warm_path, encoding="utf-8") as f:
                return json.load(f), True

    default_path = os.path.join(script_dir, "parameters", "_default_irregular", param_file)
    if os.path.exists(default_path):
        print(f"Loading default-template parameters from {default_path}")
        with open(default_path, encoding="utf-8") as f:
            return json.load(f), True

    print("Warning: No parameter file found, using hardcoded defaults")
    return {}, False


def get_param(params: Dict, *keys, default=None, allow_default: bool = True):
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if allow_default:
                return default
            raise KeyError(f"Parameter not found: {' -> '.join(keys)}")
    return value


def load_point_cloud(filepath: str) -> pd.DataFrame:
    data = np.loadtxt(filepath)
    ncols = data.shape[1] if data.ndim == 2 else 0
    if ncols < 3:
        raise ValueError(f"Point cloud must have at least 3 columns, got {ncols}")
    out = {
        "x": data[:, 0],
        "y": data[:, 1],
        "z": data[:, 2],
    }
    if ncols >= 4:
        out["intensity"] = data[:, 3]
    else:
        out["intensity"] = np.full(len(data), -1200.0)
    if ncols >= 6:
        out["segment"] = data[:, 4].astype(int)
        out["ring"] = data[:, 5].astype(int)
    return pd.DataFrame(out)


def ensure_ring_pointcloud(tunnel_id: str, ring_id: int, base_dir: str) -> Path:
    ring_key = f"r{int(ring_id)}"
    root = Path(base_dir)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    tunnel_dir = root / tunnel_id / ring_key
    tunnel_dir.mkdir(parents=True, exist_ok=True)
    dst = (tunnel_dir / f"{tunnel_id}_r{int(ring_id)}.txt").resolve()
    if dst.is_file():
        return dst
    src_name = f"{tunnel_id.replace('-', '_')}_ring{int(ring_id)}.txt"
    src = (REPO_ROOT / "data" / "rings" / src_name).resolve()
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing per-ring input: expected {dst} or fallback {src}"
        )
    shutil.copy2(src, dst)
    print(f"Copied {src.relative_to(REPO_ROOT)} → {dst.relative_to(REPO_ROOT)}")
    return dst


def build_enhancing_params(p: Dict[str, Any]) -> Dict[str, Any]:
    td = p["target_distances"]
    return {
        "upsampling_stage1_target_distance": float(td[0]),
        "upsampling_stage2_target_distance": float(td[1]),
        "upsampling_stage3_target_distance": float(td[2]),
        "curvature_threshold": float(
            p.get("curvature_threshold_enh", p.get("curvature_threshold", 0.005))
        ),
        "depth_threshold_low": float(p.get("depth_threshold_low", 0.005)),
        "depth_threshold_high": float(p.get("depth_threshold_high", 0.015)),
        "inter_radius": float(p.get("outlier_interpolation_radius", p.get("inter_radius", 0.04))),
        "duplicate_threshold": float(
            p.get("outlier_duplicate_threshold", p.get("duplicate_threshold", 0.02))
        ),
        "n_segment_start": int(p.get("n_segment_start", -1)),
        "n_segment_end": int(p.get("n_segment_end", -1)),
        "num_neighbors": int(p.get("num_neighbors", p.get("curvature_neighbors", 20))),
        "num_interpolations": int(p.get("outlier_num_interpolations", p.get("num_interpolations", 2))),
        "resolution": float(p.get("depth_map_resolution", RESOLUTION_M)),
        "window_size": int(p.get("interpolation_window", 9)),
    }


def run_preprocessing(
    tunnel_id: str,
    ring_id: int,
    base_dir: str = "data",
    regime_label: Optional[str] = None,
) -> None:
    ring_key = f"r{int(ring_id)}"
    print("=" * 60)
    print(f"Ring preprocessing: {tunnel_id}/{ring_key}")
    print("=" * 60)

    params, _loaded = load_parameters(
        tunnel_id=tunnel_id, ring_id=ring_id, regime_label=regime_label, base_dir=base_dir
    )
    # Always fill missing keys from explicit defaults (template JSONs are partial).
    allow_defaults = True

    root = Path(base_dir)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    tunnel_dir = str(root / tunnel_id / ring_key)
    os.makedirs(tunnel_dir, exist_ok=True)

    tunnel_diameter = float(get_param(params, "tunnel_diameter", default=7.5, allow_default=allow_defaults))
    pc_path = ensure_ring_pointcloud(tunnel_id, ring_id, base_dir)
    df_raw = load_point_cloud(str(pc_path))

    vu = float(get_param(params, "vertical_filter_window", default=6.8, allow_default=allow_defaults))
    grav = params.get("gravity_anchor", {}) if isinstance(params.get("gravity_anchor"), dict) else {}
    grav_enabled = bool(grav.get("enabled", True))
    grav_n_bins = int(grav.get("n_bins", 360))
    grav_meta: Dict[str, Any] = {}
    df_u, ring_count = unfold_single_ring(
        df_raw,
        tunnel_diameter=tunnel_diameter,
        vertical_filter_window=vu,
        ransac_threshold=float(get_param(params, "ransac_threshold", default=1.0, allow_default=allow_defaults)),
        ransac_probability=float(get_param(params, "ransac_probability", default=0.9, allow_default=allow_defaults)),
        ransac_inlier_ratio=float(get_param(params, "ransac_inlier_ratio", default=0.75, allow_default=allow_defaults)),
        ransac_sample_size=int(get_param(params, "ransac_sample_size", default=5, allow_default=allow_defaults)),
        ransac_initial_iterations=int(
            get_param(params, "ransac_initial_iterations", default=999, allow_default=allow_defaults)
        ),
        ransac_inlier_threshold_multiplier=float(
            get_param(params, "ransac_inlier_threshold_multiplier", default=0.8, allow_default=allow_defaults)
        ),
        gravity_anchor_enabled=grav_enabled,
        gravity_anchor_n_bins=grav_n_bins,
        gravity_meta_out=grav_meta,
    )

    out_u = os.path.join(tunnel_dir, "unwrapped.csv")
    df_u.to_csv(out_u, index=False)
    with open(os.path.join(tunnel_dir, "ring_count.txt"), "w", encoding="utf-8") as f:
        f.write(str(ring_count))
    if grav_meta:
        with open(os.path.join(tunnel_dir, "gravity_anchor_meta.json"), "w", encoding="utf-8") as f:
            json.dump(grav_meta, f, indent=2)
    print(
        f"  Wrote unwrapped.csv ({len(df_u)} pts), ring_count={ring_count}"
        + (
            f", gravity_anchor enabled (theta_shift={grav_meta.get('theta_shift', float('nan')):.3f})"
            if grav_enabled and grav_meta.get('enabled') else ", gravity_anchor disabled"
        )
    )

    mask_lo = float(get_param(params, "radius_min", default=2.37, allow_default=allow_defaults))
    mask_hi = float(get_param(params, "radius_max", default=3.8, allow_default=allow_defaults))
    y_step = float(get_param(params, "y_step", default=0.4, allow_default=allow_defaults))
    z_step = float(get_param(params, "z_step", default=0.005, allow_default=allow_defaults))
    grad_thr = float(get_param(params, "gradient_threshold", default=0.15, allow_default=allow_defaults))
    sm_win = int(get_param(params, "smoothing_window_size", default=5, allow_default=allow_defaults))
    sm_off = float(get_param(params, "smoothing_offset", default=-0.002, allow_default=allow_defaults))
    def_cut = float(get_param(params, "default_cutoff_z", default=tunnel_diameter / 2.0, allow_default=True))

    df_d = denoise_ring(
        df_u,
        ring_count=ring_count,
        mask_r_low=mask_lo,
        mask_r_high=mask_hi,
        y_step=y_step,
        z_step=z_step,
        grad_threshold=grad_thr,
        smoothing_window_size=sm_win,
        smoothing_offset=sm_off,
        default_cutoff_z=def_cut,
    )
    df_d.to_csv(os.path.join(tunnel_dir, "denoised.csv"), index=False)
    valid_n = int((df_d["pred"] != 0).sum())
    print(f"  denoised.csv valid points: {valid_n}/{len(df_d)}")

    enh = build_enhancing_params(params)
    if not np.isclose(enh["resolution"], RESOLUTION_M):
        print(f"  Forcing depth_map_resolution {enh['resolution']} → canonical {RESOLUTION_M}")
        enh["resolution"] = RESOLUTION_M

    hd_start = int(params.get("outlier_high_density_ring_start", -1))
    hd_end = int(params.get("outlier_high_density_ring_end", -1))
    outlier_hd_disabled = hd_start < 0 or hd_end < 0

    run_ring_enhancing(
        df_d,
        tunnel_dir,
        tunnel_diameter=tunnel_diameter,
        enhancing_params=enh,
        outlier_hd_disabled=outlier_hd_disabled,
    )

    print("=" * 60)
    print(f"Done: {tunnel_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-ring preprocessing (r4tun-derived)")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("ring_id", type=int, help="Ring id (e.g. 110)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--regime-label", default=None, help="Warm-start regime label")
    args = parser.parse_args()

    run_preprocessing(
        args.tunnel_id,
        args.ring_id,
        base_dir=args.data_dir,
        regime_label=args.regime_label,
    )

#!/usr/bin/env python3
"""Gravity-align an existing unwrapped.csv + regenerate depth maps.

Why
---
The current unfolder computes ``theta`` from an **per-ring** PCA basis
with a RANSAC ellipse center. The PCA eigenvector sign and the RANSAC
center drift per ring, so ``theta = 0`` points to different physical
directions on each ring. Consequently, each held-out ring sits at a
different cyclic phase relative to the calibration template, and our
candidate search has to guess a rotation per ring.

This script collapses the rotation degree of freedom by shifting each
ring's ``theta`` so that ``theta = 0`` always lands on the **world-
frame physical bottom** of the tunnel (lowest ``z``). Direction is
also fixed so that ``+theta`` runs counterclockwise when viewed from
``+x`` (deterministic across rings).

It operates on the already-computed ``context_unwrapped.csv`` (keeping
the per-ring ``r`` and ``h`` untouched), regenerates ``depth_map.npy``
and ``depth_map_outlier.npy``, and writes everything into a sandbox
directory under ``logs/gravity_unwrap_v1/<tunnel>/r<ring>/``.

We then copy the rest of the preprocessing outputs (denoised, enhanced,
final, etc.) from the source A0 ring and regenerate depth maps with
the shifted coordinate. Detection/segmentation can then be run from
this sandbox unchanged.

Usage
-----
    ./venv/bin/python methods/plans/scripts/gravity_align_unwrap.py \
        --rings 4-3/r170,4-3/r171,4-8/r330,5-2/r144 \
        --source-root logs/iterative_reflection_proof_v4/heldout_iterative_reflection \
        --source-subdir A2_iterative_intrinsic_reflection
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = REPO_ROOT / "logs" / "gravity_unwrap_v1"


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        if not np.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _compute_z_profile(
    df: pd.DataFrame,
    n_bins: int = 360,
    normalize: bool = True,
) -> np.ndarray:
    """Compute z-median by theta-bin over [0, t_range). NaN-fill returns."""
    theta = df["theta"].to_numpy(dtype=np.float64)
    z = df["z"].to_numpy(dtype=np.float64)
    t_min = float(theta.min())
    t_max = float(theta.max())
    t_range = max(1e-9, t_max - t_min)
    idx = np.clip(((theta - t_min) / t_range * n_bins).astype(int), 0, n_bins - 1)
    prof = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            prof[b] = float(np.median(z[m]))
    # fill NaN by linear interp along axis (cyclic)
    finite = np.isfinite(prof)
    if finite.any() and not finite.all():
        good_idx = np.where(finite)[0]
        good_val = prof[finite]
        all_idx = np.arange(n_bins)
        prof = np.interp(all_idx, good_idx, good_val, period=n_bins)
    if normalize and finite.any():
        prof = prof - float(np.nanmedian(prof))
        s = float(np.nanstd(prof))
        if s > 1e-9:
            prof = prof / s
    return prof


def _gravity_align_theta(
    df: pd.DataFrame,
    n_bins: int = 360,
    ref_profile: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Shift ``theta`` so that physical-bottom is at ``theta=0``.

    We bin points by ``theta`` (360 bins), compute the median ``z`` per
    bin, then locate the bin whose median ``z`` is minimum (physical
    bottom). That bin's left edge becomes the new zero.

    We also fix direction: after shift, check whether ``z`` rises
    monotonically as theta increases through the first half; if not,
    flip the theta sign (reverse direction).
    """

    theta = df["theta"].to_numpy(dtype=np.float64)
    z = df["z"].to_numpy(dtype=np.float64)
    t_min = float(theta.min())
    t_max = float(theta.max())
    t_range = max(1e-9, t_max - t_min)

    # bin by theta
    bins = np.linspace(t_min, t_max, n_bins + 1)
    idx = np.clip(((theta - t_min) / t_range * n_bins).astype(int), 0, n_bins - 1)
    z_by_bin = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            z_by_bin[b] = float(np.median(z[m]))
    finite = np.isfinite(z_by_bin)
    if not finite.any():
        return df.copy(), {"theta_shift": 0.0, "reversed": 0.0}
    bot_bin = int(np.argmin(np.where(finite, z_by_bin, np.inf)))
    top_bin = int(np.argmax(np.where(finite, z_by_bin, -np.inf)))
    t_shift = float(bins[bot_bin])

    # Shift: theta' = (theta - t_shift) mod t_range
    theta_shifted = ((theta - t_shift) % t_range)

    # Direction fix: after the shift, physical bottom is at theta' = 0.
    # A tunnel cross-section is approximately left-right symmetric in
    # z, so z(theta') alone can't tell direction. But a per-tunnel
    # CALIBRATION z-profile IS asymmetric (K block sits on one side
    # of top). We cross-correlate the held-out z-profile against the
    # calibration reference in both directions and pick the better.
    # Without a reference, we fall back to a consistent convention
    # (do not flip).

    # Shifted z profile
    df_shift = df.copy()
    df_shift["theta"] = theta_shifted.astype(np.float64)
    # Ensure range is [0, t_range] for binning
    df_shift["theta"] = df_shift["theta"] - float(df_shift["theta"].min())
    z_prof = _compute_z_profile(df_shift, n_bins=n_bins, normalize=True)

    reversed_flag = False
    corr_fwd = float("nan")
    corr_rev = float("nan")
    if ref_profile is not None and len(ref_profile) == n_bins:
        # Normalized cross-correlation of profiles (both 0-mean, unit-var)
        corr_fwd = float(np.dot(z_prof, ref_profile) / n_bins)
        z_prof_rev = z_prof[::-1]
        corr_rev = float(np.dot(z_prof_rev, ref_profile) / n_bins)
        if corr_rev > corr_fwd + 0.02:  # small epsilon; must strictly exceed
            theta_shifted = (t_range - theta_shifted) % t_range
            reversed_flag = True

    out = df.copy()
    out["theta"] = theta_shifted.astype(np.float64)
    meta = {
        "theta_shift": t_shift,
        "theta_range": t_range,
        "reversed": 1.0 if reversed_flag else 0.0,
        "bottom_bin_z": float(z_by_bin[bot_bin]),
        "top_bin_z": float(z_by_bin[top_bin]),
        "corr_fwd": corr_fwd,
        "corr_rev": corr_rev,
        "n_bins": int(n_bins),
    }
    return out, meta


def _shift_depth_map(
    depth_map: np.ndarray,
    src_theta_min: float,
    theta_shift: float,
    theta_range: float,
    resolution: float,
    reversed_flag: bool,
) -> tuple[np.ndarray, int]:
    """Apply gravity-alignment to an existing depth map.

    The original ``depth_map`` is indexed by ``(y=theta_pixel, x=h_pixel)``.
    We shift rows so that the physical-bottom row (corresponding to
    ``src_theta_min + theta_shift``) lands at ``y=0``, then optionally
    reverse rows for direction flip.

    Returns (shifted_depth_map, row_shift).
    """
    H = depth_map.shape[0]
    row_shift = int(round((theta_shift) / resolution)) % H
    out = np.roll(depth_map, -row_shift, axis=0)
    if reversed_flag:
        out = out[::-1, :]
    return out, row_shift


def _shift_pixel_to_point(
    ptp: list[dict[str, int]],
    row_shift: int,
    H: int,
    reversed_flag: bool,
) -> list[dict[str, int]]:
    """Apply the same y-shift + optional reversal to a pixel_to_point list."""
    out = []
    for rec in ptp:
        py = int(rec.get("pixel_y", 0))
        y_new = (py - row_shift) % H
        if reversed_flag:
            y_new = (H - 1) - y_new
        out.append({"pixel_x": int(rec.get("pixel_x", 0)), "pixel_y": y_new, "index": int(rec.get("index", 0))})
    return out


def _build_reference_profile(
    tunnel: str,
    calib_root: Path,
    n_bins: int = 360,
) -> np.ndarray | None:
    """Build a per-tunnel reference z-profile from GT-calibrated unwrap.

    Uses the BO-best calibration unwrapped.csv for the matching tunnel,
    gravity-shifted (bottom → 0) but not direction-flipped. Returns a
    0-mean unit-var profile in [0, t_range] binned into ``n_bins``.
    """

    # Find calibration unwrapped.csv for the same tunnel
    calib_dir = calib_root / tunnel
    if not calib_dir.exists():
        return None
    candidates = sorted(calib_dir.glob("r*/best/*/r*/unwrapped.csv"))
    if not candidates:
        candidates = sorted(calib_dir.glob("r*/best/unwrapped.csv"))
    if not candidates:
        return None
    # Take the first one as canonical reference
    df = pd.read_csv(candidates[0])
    # Gravity-shift only (no direction flip, no ref comparison)
    df_shift, _ = _gravity_align_theta(df, n_bins=n_bins, ref_profile=None)
    df_shift = df_shift.copy()
    df_shift["theta"] = df_shift["theta"] - float(df_shift["theta"].min())
    return _compute_z_profile(df_shift, n_bins=n_bins, normalize=True)


def _process_ring(
    ring_key: str,
    source_root: Path,
    source_subdir: str,
    calib_root: Path | None = None,
    resolution_default: float = 0.005,
) -> dict[str, Any]:
    """Gravity-align one ring and regenerate depth maps."""

    tunnel, ring_dirname = ring_key.split("/", 1)
    src = source_root / tunnel / ring_dirname / source_subdir
    dst = SANDBOX_ROOT / tunnel / ring_dirname / "gravity"
    if not src.exists():
        raise FileNotFoundError(f"source not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Load the context_unwrapped.csv (full, with r/theta/h) and the
    # target-only unwrapped.csv (used for detection metrics).
    ctx_path = src / "context_unwrapped.csv"
    if not ctx_path.exists():
        raise FileNotFoundError(f"context_unwrapped.csv missing: {ctx_path}")
    df_ctx = pd.read_csv(ctx_path)

    # Preprocessing parameters (resolution)
    pp_path = src / "parameters_preprocessing.json"
    res = resolution_default
    pp: dict[str, Any] = {}
    if pp_path.exists():
        pp = json.loads(pp_path.read_text())
        res = float(pp.get("depth_map_resolution", resolution_default))

    # Reference profile for this tunnel (for direction disambiguation)
    ref_profile = None
    if calib_root is not None:
        ref_profile = _build_reference_profile(tunnel, calib_root)

    # Gravity-align (theta shift + direction flag)
    df_ctx_g, meta = _gravity_align_theta(df_ctx, ref_profile=ref_profile)

    # Save new unwrapped CSVs (target only vs context). We KEEP the
    # relative theta within the CSV consistent with the shifted image
    # so that detection/segmentation continues to work.
    df_ctx_g.to_csv(dst / "context_unwrapped.csv", index=False)
    target_ring = int(df_ctx_g["ring"].mode()[0])
    df_target = df_ctx_g[df_ctx_g["ring"] == target_ring].copy().reset_index(drop=True)
    df_target.to_csv(dst / "unwrapped.csv", index=False)

    ring_count_src = src / "ring_count.txt"
    if ring_count_src.exists():
        shutil.copy2(ring_count_src, dst / "ring_count.txt")
    else:
        (dst / "ring_count.txt").write_text("1\n")

    # Load existing depth map and shift it (cheap, keeps density).
    dm_src = src / "depth_map.npy"
    if not dm_src.exists():
        raise FileNotFoundError(f"depth_map.npy missing: {dm_src}")
    dm = np.load(dm_src)
    dm_out_src = src / "depth_map_outlier.npy"
    dm_out = np.load(dm_out_src) if dm_out_src.exists() else dm
    ptp_src = src / "pixel_to_point.pkl"
    import pickle
    with open(ptp_src, "rb") as f:
        ptp = pickle.load(f)

    # Original theta extent (from the source unwrapped.csv, not shifted)
    src_ctx = pd.read_csv(ctx_path)
    src_theta_min = float(src_ctx["theta"].min())
    theta_range = float(meta["theta_range"])
    theta_shift = float(meta["theta_shift"])
    reversed_flag = bool(meta["reversed"] > 0.5)

    dm_g, row_shift = _shift_depth_map(dm, src_theta_min, theta_shift, theta_range, res, reversed_flag)
    dm_out_g, _ = _shift_depth_map(dm_out, src_theta_min, theta_shift, theta_range, res, reversed_flag)
    ptp_g = _shift_pixel_to_point(ptp, row_shift=row_shift, H=dm.shape[0], reversed_flag=reversed_flag)

    np.save(dst / "depth_map.npy", dm_g)
    np.save(dst / "depth_map_outlier.npy", dm_out_g)

    # Render a PNG for visual inspection + pipelines that expect depth_map.png
    try:
        from PIL import Image
        valid = np.isfinite(dm_g) & (dm_g > 0)
        png = np.zeros_like(dm_g, dtype=np.uint8)
        if valid.any():
            lo, hi = np.percentile(dm_g[valid], [2, 98])
            if hi - lo > 1e-9:
                png[valid] = np.clip((dm_g[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(png).save(dst / "depth_map.png")
    except Exception:  # noqa: BLE001
        pass
    with open(dst / "pixel_to_point.pkl", "wb") as f:
        pickle.dump(ptp_g, f)

    # Shift any theta-containing CSV in the source (denoised/enhanced/final)
    for name in ("denoised.csv", "enhanced.csv", "context_denoised.csv", "context_enhanced.csv", "final.csv"):
        sp = src / name
        if not sp.exists():
            continue
        df_other = pd.read_csv(sp)
        if "theta" in df_other.columns:
            th = df_other["theta"].to_numpy(dtype=np.float64)
            th2 = (th - theta_shift) % theta_range
            if reversed_flag:
                th2 = (theta_range - th2) % theta_range
            df_other["theta"] = th2
        df_other.to_csv(dst / name, index=False)

    if pp:
        (dst / "parameters_preprocessing.json").write_text(json.dumps(pp, indent=2, sort_keys=True) + "\n")
    det_path = src / "parameters_detection.json"
    if det_path.exists():
        shutil.copy2(det_path, dst / "parameters_detection.json")

    (dst / "gravity_meta.json").write_text(
        json.dumps({"ring_key": ring_key, "source": str(src), "row_shift": row_shift, **meta, "resolution": res}, indent=2, sort_keys=True) + "\n"
    )

    return {"ring_key": ring_key, "dst": str(dst), "row_shift": row_shift, **meta}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rings", type=str, required=True, help="csv of ring_keys e.g. '4-3/r171,4-8/r330'")
    p.add_argument(
        "--source-root",
        type=str,
        default="logs/iterative_reflection_proof_v4/heldout_iterative_reflection",
    )
    p.add_argument("--source-subdir", type=str, default="A2_iterative_intrinsic_reflection")
    p.add_argument(
        "--calib-root",
        type=str,
        default="logs/detection_boundary_structural_panel_v3/artifacts",
        help="Root dir containing calibration artifacts per tunnel (for z-profile reference).",
    )
    args = p.parse_args()

    source_root = REPO_ROOT / args.source_root
    calib_root = REPO_ROOT / args.calib_root
    rings = [s.strip() for s in args.rings.split(",") if s.strip()]
    for rk in rings:
        try:
            meta = _process_ring(rk, source_root, args.source_subdir, calib_root=calib_root)
            print(json.dumps(meta, indent=2, sort_keys=True))
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {rk}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Depth-discontinuity anchor intrinsic metric.

Given a ring directory with:
  - depth_map.npy (L x W, interpolated, ~70% dense)
  - detection/labelmap.npy (L x W, class IDs per pixel)

We compute a physical-evidence score for the PREDICTED block layout:

  1) ``row_grad[y] = |∂depth/∂y|`` averaged over valid x in row ``y``.
  2) For each CLASS transition in the labelmap (y where class[y] != class[y-1]),
     accumulate ``row_grad[y]``.
  3) Normalize by the sum of the top-N row_grad values in the image
     (N = number of transitions). This gives a score in [0, 1] where 1
     means "every predicted boundary sits exactly on the strongest depth
     discontinuities available."

Higher scores mean the predicted layout is physically well-anchored.

Usage:
    ./venv/bin/python methods/plans/scripts/depth_disc_anchor.py \\
        --rings 4-3/r170,4-3/r171 \\
        --base logs/gravity_unwrap_v1/pipeline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_depth_discontinuity_score(
    depth_map: np.ndarray,
    labelmap: np.ndarray,
    *,
    valid_threshold: float = 0.0,
    smooth_sigma_y: float = 3.0,
) -> dict[str, Any]:
    """Compute depth-discontinuity anchor score for a labelmap.

    Returns a dict with: ``anchor_score`` (in [0, 1]), ``boundary_rows``
    (list of y where class transitions), ``row_grad`` (array), and
    ``topk_sum`` / ``boundary_sum`` for traceability.
    """
    if depth_map.shape != labelmap.shape:
        raise ValueError(f"shape mismatch: depth={depth_map.shape} label={labelmap.shape}")
    L, W = depth_map.shape
    valid = np.isfinite(depth_map) & (depth_map > valid_threshold)

    # Per-row mean depth (over valid pixels only)
    row_sum = np.where(valid, depth_map, 0.0).sum(axis=1)
    row_cnt = valid.sum(axis=1)
    row_mean = np.where(row_cnt > 0, row_sum / np.maximum(row_cnt, 1), np.nan)
    # fill NaN with cyclic interp for gradient stability
    finite = np.isfinite(row_mean)
    if finite.any() and not finite.all():
        idx_good = np.where(finite)[0]
        vals = row_mean[finite]
        all_idx = np.arange(L)
        row_mean = np.interp(all_idx, idx_good, vals, period=L)

    # Smooth along y to suppress jitter (replication at ends)
    if smooth_sigma_y and smooth_sigma_y > 0:
        from scipy.ndimage import gaussian_filter1d
        row_mean_s = gaussian_filter1d(row_mean, sigma=float(smooth_sigma_y), mode="wrap")
    else:
        row_mean_s = row_mean

    # Cyclic gradient: diff along y with wrap
    row_next = np.roll(row_mean_s, -1)
    row_grad = np.abs(row_next - row_mean_s)

    # Find boundary rows in labelmap (transitions in the most-common class per row)
    # Compute per-row mode class. A fast proxy: mode via argmax of bincount per row.
    H = L
    # Restrict to NON-BG (class > 0) for mode computation
    mode_cls = np.zeros(H, dtype=np.int32)
    for y in range(H):
        row = labelmap[y]
        nonbg = row[row > 0]
        if nonbg.size == 0:
            mode_cls[y] = 0
            continue
        u, c = np.unique(nonbg, return_counts=True)
        mode_cls[y] = int(u[np.argmax(c)])
    # Cyclic transitions
    next_cls = np.roll(mode_cls, -1)
    trans_mask = (mode_cls != next_cls) & (mode_cls > 0) & (next_cls > 0)
    boundary_rows = np.where(trans_mask)[0]
    n_trans = len(boundary_rows)

    if n_trans == 0:
        return {
            "anchor_score": 0.0,
            "boundary_rows": [],
            "boundary_sum": 0.0,
            "topk_sum": 0.0,
            "n_transitions": 0,
        }
    boundary_sum = float(np.sum(row_grad[boundary_rows]))
    # Top-k scores in row_grad with exclusion of nearby neighbors (so
    # top-k is realistically attainable by any layout with n_trans
    # transitions separated by at least min_sep rows).
    min_sep = max(1, int(0.01 * H))
    topk = _topk_with_min_separation(row_grad, n_trans, min_sep)
    topk_sum = float(np.sum(row_grad[topk])) if len(topk) else 0.0
    anchor = float(boundary_sum / max(topk_sum, 1e-12))
    return {
        "anchor_score": min(1.0, anchor),
        "boundary_rows": boundary_rows.tolist(),
        "boundary_sum": boundary_sum,
        "topk_sum": topk_sum,
        "n_transitions": n_trans,
        "row_grad_mean": float(row_grad.mean()),
        "row_grad_max": float(row_grad.max()),
    }


def _topk_with_min_separation(values: np.ndarray, k: int, min_sep: int) -> list[int]:
    """Return indices of k largest values with at least ``min_sep`` gap."""
    order = np.argsort(-values)
    chosen = []
    for idx in order:
        if len(chosen) >= k:
            break
        if all(min(abs(idx - c), len(values) - abs(idx - c)) >= min_sep for c in chosen):
            chosen.append(int(idx))
    return chosen


def _score_ring(base: Path) -> dict[str, Any] | None:
    dm_path = base / "depth_map.npy"
    lm_path = base / "detection" / "labelmap.npy"
    if not dm_path.exists() or not lm_path.exists():
        return None
    dm = np.load(dm_path)
    lm = np.load(lm_path)
    return compute_depth_discontinuity_score(dm, lm)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rings", type=str, required=True)
    p.add_argument("--base", type=str, default="logs/gravity_unwrap_v1/pipeline")
    args = p.parse_args()

    base_root = Path(args.base)
    rings = [s.strip() for s in args.rings.split(",") if s.strip()]
    rows = []
    for rk in rings:
        t, r = rk.split("/", 1)
        ring_base = base_root / t / r
        result = _score_ring(ring_base)
        if result is None:
            rows.append({"ring": rk, "anchor_score": float("nan"), "n_trans": 0})
            continue
        rows.append({"ring": rk, "anchor_score": result["anchor_score"], "n_trans": result["n_transitions"]})
        print(f'{rk:<15s} anchor_score={result["anchor_score"]:.3f} n_trans={result["n_transitions"]} boundary_sum={result["boundary_sum"]:.3f} topk_sum={result["topk_sum"]:.3f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

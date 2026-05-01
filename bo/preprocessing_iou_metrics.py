"""Preprocessing BO metrics for target-ring depth maps.

Foreground "GT" for preprocessing is derived from point-cloud labels
(`segment > 0`) aligned to depth-map pixels via `pixel_to_point.pkl`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _load_depth_map(ring_dir: Path) -> np.ndarray:
    """Load depth map preferring outlier-enhanced output."""
    outlier = ring_dir / "depth_map_outlier.npy"
    plain = ring_dir / "depth_map.npy"
    if outlier.exists():
        return np.load(outlier)
    if plain.exists():
        return np.load(plain)
    raise FileNotFoundError(f"Missing depth map in {ring_dir} (expected depth_map_outlier.npy or depth_map.npy)")


def build_gt_foreground_mask_from_segment_mapping(ring_dir: Path) -> np.ndarray:
    """Build a pixel foreground mask from denoised segment labels + pixel mapping.

    Uses:
    - `denoised.csv` (`segment` column; foreground is segment > 0)
    - `pixel_to_point.pkl` (pixel -> denoised row index mapping)
    - depth map shape for mask dimensions
    """
    depth_map = _load_depth_map(ring_dir)
    h, w = depth_map.shape
    fg_mask = np.zeros((h, w), dtype=bool)

    denoised_path = ring_dir / "denoised.csv"
    mapping_path = ring_dir / "pixel_to_point.pkl"
    if not denoised_path.exists():
        raise FileNotFoundError(f"Missing denoised.csv: {denoised_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing pixel_to_point.pkl: {mapping_path}")

    seg = pd.read_csv(denoised_path, usecols=["segment"]).to_numpy().reshape(-1)
    with mapping_path.open("rb") as f:
        mapping = pickle.load(f)

    n = len(seg)
    for row in mapping:
        idx = int(row.get("index", -1))
        px = int(row.get("pixel_x", -1))
        py = int(row.get("pixel_y", -1))
        if 0 <= idx < n and 0 <= px < w and 0 <= py < h and int(seg[idx]) > 0:
            fg_mask[py, px] = True
    return fg_mask


def compute_foreground_mask_iou_metrics(ring_dir: Path) -> Dict[str, Any]:
    """Compute IoU and diagnostics for preprocessing BO."""
    depth_map = _load_depth_map(ring_dir)
    valid = np.isfinite(depth_map) & (depth_map > 0.0)
    gt_fg = build_gt_foreground_mask_from_segment_mapping(ring_dir)

    tp = int(np.count_nonzero(valid & gt_fg))
    fp = int(np.count_nonzero(valid & (~gt_fg)))
    fn = int(np.count_nonzero((~valid) & gt_fg))
    denom = tp + fp + fn
    iou = float(tp / denom) if denom > 0 else 0.0

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    valid_ratio = float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0
    gt_fg_ratio = float(np.count_nonzero(gt_fg) / gt_fg.size) if gt_fg.size else 0.0

    return {
        "foreground_mask_iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "valid_ratio": valid_ratio,
        "gt_foreground_ratio": gt_fg_ratio,
        "depth_shape_h": int(depth_map.shape[0]),
        "depth_shape_w": int(depth_map.shape[1]),
    }


def _largest_empty_row_band(valid_mask: np.ndarray) -> int:
    row_valid = valid_mask.sum(axis=1)
    largest = 0
    cur = 0
    for x in row_valid == 0:
        if x:
            cur += 1
            largest = max(largest, cur)
        else:
            cur = 0
    return int(largest)


def compute_target_guarded_metrics(
    ring_dir: Path,
    *,
    baseline_valid_ratio: float | None = None,
    min_coverage_ratio: float = 0.70,
    max_empty_row_band_ratio: float = 0.45,
) -> Dict[str, Any]:
    """Guarded reward for preprocessing BO.

    Primary signal:
      target_foreground_recall = TP / (TP + FN)

    Guardrails:
      - valid coverage must not collapse vs baseline
      - largest empty row band ratio must stay below threshold

    Diagnostic:
      - foreground_mask_iou (old objective) is reported but not optimized directly.
    """
    depth_map = _load_depth_map(ring_dir)
    valid = np.isfinite(depth_map) & (depth_map > 0.0)
    gt_fg = build_gt_foreground_mask_from_segment_mapping(ring_dir)

    tp = int(np.count_nonzero(valid & gt_fg))
    fp = int(np.count_nonzero(valid & (~gt_fg)))
    fn = int(np.count_nonzero((~valid) & gt_fg))
    denom = tp + fp + fn
    iou = float(tp / denom) if denom > 0 else 0.0

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    valid_ratio = float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0
    gt_fg_ratio = float(np.count_nonzero(gt_fg) / gt_fg.size) if gt_fg.size else 0.0

    largest_empty = _largest_empty_row_band(valid)
    empty_ratio = float(largest_empty / depth_map.shape[0]) if depth_map.shape[0] > 0 else 1.0

    # Guardrail 1: coverage floor relative to baseline.
    coverage_ok = True
    coverage_factor = 1.0
    if baseline_valid_ratio is not None and baseline_valid_ratio > 0:
        floor = float(min_coverage_ratio) * float(baseline_valid_ratio)
        coverage_ok = valid_ratio >= floor
        if floor > 0:
            coverage_factor = max(0.0, min(1.0, valid_ratio / floor))

    # Guardrail 2: empty-band ceiling.
    empty_ok = empty_ratio <= float(max_empty_row_band_ratio)
    if empty_ok:
        empty_factor = 1.0
    else:
        span = max(1e-9, 1.0 - float(max_empty_row_band_ratio))
        empty_factor = max(0.0, 1.0 - ((empty_ratio - float(max_empty_row_band_ratio)) / span))

    guarded_score = float(recall * coverage_factor * empty_factor)

    return {
        "guarded_score": guarded_score,
        "target_foreground_recall": recall,
        "precision": precision,
        "foreground_mask_iou": iou,  # diagnostic only
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "valid_ratio": valid_ratio,
        "gt_foreground_ratio": gt_fg_ratio,
        "largest_empty_row_band": largest_empty,
        "empty_row_band_ratio": empty_ratio,
        "coverage_ok": bool(coverage_ok),
        "empty_band_ok": bool(empty_ok),
        "coverage_factor": float(coverage_factor),
        "empty_factor": float(empty_factor),
        "baseline_valid_ratio": None if baseline_valid_ratio is None else float(baseline_valid_ratio),
        "min_coverage_ratio": float(min_coverage_ratio),
        "max_empty_row_band_ratio": float(max_empty_row_band_ratio),
        "depth_shape_h": int(depth_map.shape[0]),
        "depth_shape_w": int(depth_map.shape[1]),
    }

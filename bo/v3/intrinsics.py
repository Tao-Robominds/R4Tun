"""Per-trial intrinsic-vector + mIoU collection for v3 BO.

Aggregates the curated intrinsic fields the manuscript and the v3 plan
need from a finished pipeline trial:

* **Preprocessing intrinsics** — read from the per-ring sandbox dir
  (``valid_ratio``, ``largest_empty_row_band_ratio``, depth-shape, etc.)
  computed from ``depth_map_outlier.npy``.
* **Detection intrinsics** — delegated to
  :func:`agents.2_detection.scripts.extract_intrinsics.extract_detection_metrics`.
* **Segmentation intrinsics** — delegated to
  :func:`agents.3_segmentation.scripts.extract_intrinsics.extract_segmentation_metrics`.
* **Primary mIoU** — fixed-class ``sklearn.jaccard_score`` from
  ``agents/evaluation.py`` (the deployment metric).
* **Secondary mIoU** — permutation-invariant Hungarian assignment over
  per-ring class IDs (logged-only; flags residual anchoring failures).

This module never imports the agent modules at top level so a single
import does not trigger heavy matplotlib/cv2 setup.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_ROOT = REPO_ROOT / "agents"


# ---------------------------------------------------------------------------
# Dynamic import helpers (the agent dirs use leading-digit names that are
# not legal Python identifiers, so we have to load by file path).
# ---------------------------------------------------------------------------

_LOADED: dict[str, Any] = {}


def _load_module(name: str, path: Path) -> Any:
    if name in _LOADED:
        return _LOADED[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _LOADED[name] = mod
    return mod


def _load_detection_intrinsics() -> Any:
    return _load_module(
        "_v3_detection_intrinsics",
        AGENTS_ROOT / "2_detection" / "scripts" / "extract_intrinsics.py",
    )


def _load_segmentation_intrinsics() -> Any:
    return _load_module(
        "_v3_segmentation_intrinsics",
        AGENTS_ROOT / "3_segmentation" / "scripts" / "extract_intrinsics.py",
    )


# ---------------------------------------------------------------------------
# Preprocessing intrinsics (computed locally from depth_map_outlier.npy)
# ---------------------------------------------------------------------------

def extract_preprocessing_intrinsics(ring_dir: Path) -> dict[str, Any]:
    """Compute the preprocessing-stage intrinsic fields used by Step 3.

    The fields mirror what v1/v2's BO ``trial_meta.json`` exposed
    (``valid_ratio``, ``largest_empty_row_band``, ``empty_row_band_ratio``,
    ``depth_shape_h``, ``depth_shape_w``) so the cross-stage correlation
    profile in :mod:`bo.v3.aggregate_calibration` can use a uniform schema.
    """
    out: dict[str, Any] = {
        "pre_valid_ratio": None,
        "pre_largest_empty_row_band": None,
        "pre_empty_row_band_ratio": None,
        "pre_depth_shape_h": None,
        "pre_depth_shape_w": None,
        "pre_gravity_anchor_enabled": None,
        "pre_gravity_theta_shift": None,
        "pre_bottom_bin_z": None,
    }
    dm_path = ring_dir / "depth_map_outlier.npy"
    if not dm_path.exists():
        # Fall back to depth_map.npy if outlier is missing.
        dm_path = ring_dir / "depth_map.npy"
    if dm_path.exists():
        try:
            dm = np.load(dm_path)
            valid = np.isfinite(dm)
            total = int(dm.size)
            valid_count = int(valid.sum())
            out["pre_valid_ratio"] = float(valid_count / total) if total else 0.0
            out["pre_depth_shape_h"] = int(dm.shape[0])
            out["pre_depth_shape_w"] = int(dm.shape[1])
            row_valid = valid.any(axis=1)
            empty = ~row_valid
            if empty.size:
                # Longest run of empty rows (cyclic-aware, so a single huge
                # gap that wraps around the cylinder still counts once).
                max_run, run = 0, 0
                # Run twice over the array to capture wrap.
                for v in np.concatenate([empty, empty]):
                    if v:
                        run += 1
                        if run > max_run:
                            max_run = run
                    else:
                        run = 0
                max_run = min(max_run, int(empty.size))
                out["pre_largest_empty_row_band"] = int(max_run)
                out["pre_empty_row_band_ratio"] = float(max_run / int(empty.size))
        except Exception:  # noqa: BLE001
            pass
    grav_path = ring_dir / "gravity_anchor_meta.json"
    if grav_path.exists():
        try:
            grav = json.loads(grav_path.read_text())
            out["pre_gravity_anchor_enabled"] = bool(grav.get("enabled", False))
            out["pre_gravity_theta_shift"] = grav.get("theta_shift")
            out["pre_bottom_bin_z"] = grav.get("bottom_bin_z")
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# Detection / segmentation intrinsic wrappers
# ---------------------------------------------------------------------------

def extract_detection_intrinsics(ring_dir: Path) -> dict[str, Any]:
    """Run the detection extractor on a ring sandbox dir."""
    mod = _load_detection_intrinsics()
    return mod.extract_detection_metrics(str(ring_dir))


def extract_segmentation_intrinsics(ring_dir: Path) -> dict[str, Any]:
    """Run the segmentation extractor on a ring sandbox dir."""
    mod = _load_segmentation_intrinsics()
    return mod.extract_segmentation_metrics(str(ring_dir))


# ---------------------------------------------------------------------------
# mIoU helpers
# ---------------------------------------------------------------------------

def _load_pred_gt(ring_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    final = ring_dir / "final.csv"
    if not final.exists():
        return None
    try:
        df = pd.read_csv(final, usecols=lambda c: c in {"pred", "segment"})
    except Exception:  # noqa: BLE001
        return None
    if "pred" not in df.columns or "segment" not in df.columns:
        return None
    df = df[df["segment"].notna()].copy()
    if df.empty:
        return None
    pred = df["pred"].astype(int).to_numpy()
    gt = df["segment"].astype(int).to_numpy()
    return gt, pred


def fixed_class_miou(ring_dir: Path, *, max_class: int = 7) -> float | None:
    """Fixed-class canonical mIoU on ``final.csv`` (deployment metric).

    Class IDs are interpreted at face value (the gravity-anchored detector
    emits canonical 1..max_class), background = 0, anything outside
    [0, max_class] is dropped.
    """
    pair = _load_pred_gt(ring_dir)
    if pair is None:
        return None
    gt, pred = pair
    valid = (gt >= 0) & (gt <= max_class) & (pred >= 0) & (pred <= max_class)
    if not valid.any():
        return None
    gt_v = gt[valid]
    pred_v = pred[valid]
    classes = np.sort(np.unique(np.concatenate([gt_v, pred_v])))
    if classes.size == 0:
        return None
    iou = jaccard_score(gt_v, pred_v, average=None, labels=classes, zero_division=0)
    return float(np.mean(iou))


def permutation_invariant_miou(ring_dir: Path, *, max_class: int = 7) -> float | None:
    """Hungarian-assigned permutation-invariant mIoU (secondary diagnostic).

    Uses the IoU matrix between unique GT segments and predicted classes,
    then takes the assignment-mean. Background (class 0) is treated like
    any other class so the comparison is fair to the fixed-class metric.
    """
    pair = _load_pred_gt(ring_dir)
    if pair is None:
        return None
    gt, pred = pair
    valid = (gt >= 0) & (gt <= max_class) & (pred >= 0) & (pred <= max_class)
    if not valid.any():
        return None
    gt_v = gt[valid]
    pred_v = pred[valid]
    gt_ids = np.unique(gt_v)
    pred_ids = np.unique(pred_v)
    if gt_ids.size == 0 or pred_ids.size == 0:
        return None
    iou_mat = np.zeros((gt_ids.size, pred_ids.size), dtype=np.float64)
    for i, g in enumerate(gt_ids):
        gmask = gt_v == g
        for j, p in enumerate(pred_ids):
            pmask = pred_v == p
            inter = int((gmask & pmask).sum())
            union = int((gmask | pmask).sum())
            iou_mat[i, j] = inter / union if union else 0.0
    # Hungarian on -IoU to maximise.
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matched = iou_mat[row_ind, col_ind]
    if matched.size == 0:
        return None
    # Average over matched pairs (unmatched GT segments contribute 0).
    n_classes_for_avg = max(int(gt_ids.size), int(pred_ids.size))
    return float(matched.sum() / n_classes_for_avg)


# ---------------------------------------------------------------------------
# Public collector
# ---------------------------------------------------------------------------

def collect_trial_intrinsics(
    ring_dir: Path,
    *,
    max_class: int = 7,
    include_segmentation: bool = True,
) -> dict[str, Any]:
    """Compute the curated intrinsic vector + mIoU pair for one trial.

    Missing fields are reported as ``None`` rather than raising; callers
    can decide whether absence counts as a trial failure.
    """
    rec: dict[str, Any] = {
        "miou_fixed_class": fixed_class_miou(ring_dir, max_class=max_class),
        "miou_permutation": permutation_invariant_miou(ring_dir, max_class=max_class),
    }
    rec.update(extract_preprocessing_intrinsics(ring_dir))
    try:
        rec.update(extract_detection_intrinsics(ring_dir))
    except Exception as exc:  # noqa: BLE001
        rec["det_extract_error"] = repr(exc)
    if include_segmentation:
        try:
            rec.update(extract_segmentation_intrinsics(ring_dir))
        except Exception as exc:  # noqa: BLE001
            rec["seg_extract_error"] = repr(exc)
    return rec

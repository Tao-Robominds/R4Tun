"""Offline bottom-baseline scorer.

The held-out runner outputs (`logs/v3/heldout/<arm>/<tunnel>/r<ring>/final.csv`)
carry K-anchored canonical pred IDs (`K=1, B1=2, A1=3, ..., B2=7`) emitted by
``agents/2_detection/2_detection.py``. Those pred IDs are silently canonical
regardless of whether gravity anchoring was on, which neutralises the
gravity-anchor toggle for the fixed-class mIoU comparison.

This script computes a SECOND mIoU per ring per arm with the K-anchored
canonical labels removed. For each ring it:

1. Loads ``final.csv`` (point cloud rows with ``segment``, ``pred``).
2. Loads ``pixel_to_point.pkl`` to map each point index to its
   (pixel_row, pixel_col) in the unfolded depth map.
3. Groups points by pred ID and computes the mean pixel-row per
   predicted segment.
4. Re-ranks the pred IDs strictly by mean pixel-row (smallest row -> 1,
   largest -> n). Background (pred=0) stays 0.
5. Re-computes ``jaccard_score`` of the y-rank-relabelled pred against
   the raw GT ``segment`` column (which is itself y-ranked at annotation
   time in the unwrap frame seg2tunnel was authored in).

The result is the "no canonical, no anchor" bottom baseline. Without
the K-anchored canonical relabel, the gravity-anchor toggle becomes a
genuine experimental variable: pred y-rank now reflects whichever
unwrap rotation the pipeline actually produced.

Usage::

    ./venv/bin/python -m bo.v3.score_unrotated

Writes:

* ``logs/v3/heldout/scoreboard_yrank.csv`` — one row per (arm, ring),
  with both the canonical and y-rank fixed-class mIoU side by side.
* ``logs/v3/heldout/yrank_summary.json`` — per-arm means and lift.
"""
from __future__ import annotations

import csv
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("bo.v3.yrank")

ARMS = ("a_unanchored", "b_anchored")
HELDOUT_ROOT = REPO_ROOT / "logs" / "v3" / "heldout"
PANEL_PATH = REPO_ROOT / "data" / "v3" / "panels" / "heldout" / "heldout_panel_v3.json"


# ---------------------------------------------------------------------------
# Per-ring relabel + score
# ---------------------------------------------------------------------------

def _pixel_row_per_point(p2p_path: Path) -> Optional[np.ndarray]:
    """Return ``rows[idx] = pixel_row`` (or NaN where missing).

    The ``pixel_to_point.pkl`` schema varies; this matches ``bo.v3.ontology``
    (``index|point_index|idx`` and ``pixel_y|row|pixel_row``).
    """
    if not p2p_path.exists():
        return None
    try:
        with open(p2p_path, "rb") as f:
            p2p = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read %s: %r", p2p_path, exc)
        return None
    if not p2p:
        return None
    max_idx = -1
    for entry in p2p:
        idx = entry.get("index", entry.get("point_index", entry.get("idx")))
        if idx is None:
            continue
        if int(idx) > max_idx:
            max_idx = int(idx)
    if max_idx < 0:
        return None
    rows = np.full(max_idx + 1, np.nan, dtype=np.float64)
    for entry in p2p:
        idx = entry.get("index", entry.get("point_index", entry.get("idx")))
        row = entry.get("pixel_y", entry.get("row", entry.get("pixel_row")))
        if idx is None or row is None:
            continue
        i = int(idx)
        if 0 <= i <= max_idx:
            rows[i] = float(row)
    return rows


def _y_rank_remap(pred: np.ndarray, mean_y_per_id: dict[int, float]) -> tuple[np.ndarray, dict[int, int]]:
    """Map pred IDs to y-rank IDs (1..n by ascending mean y).

    Background pred=0 stays 0. Pred IDs absent from ``mean_y_per_id`` are
    treated as background as well. Returns the remapped pred array and
    the ``original_id -> new_id`` mapping.
    """
    nonzero = [(pid, y) for pid, y in mean_y_per_id.items() if pid != 0 and not np.isnan(y)]
    nonzero.sort(key=lambda t: t[1])
    remap: dict[int, int] = {0: 0}
    for new_id, (orig_id, _y) in enumerate(nonzero, start=1):
        remap[int(orig_id)] = new_id
    new_pred = np.zeros_like(pred, dtype=np.int64)
    for orig_id, new_id in remap.items():
        new_pred[pred == orig_id] = new_id
    return new_pred, remap


def _score_ring(ring_dir: Path, *, max_class: int = 7) -> dict[str, Any]:
    """Return the canonical and y-rank mIoU pair for one ring.

    Missing inputs return ``None`` for the affected metric.
    """
    final_path = ring_dir / "final.csv"
    p2p_path = ring_dir / "pixel_to_point.pkl"
    if not final_path.exists():
        return {
            "miou_fixed_canonical": None,
            "miou_fixed_yrank": None,
            "n_pred_segments": 0,
            "remap": None,
            "note": "final.csv missing",
        }
    try:
        df = pd.read_csv(final_path, usecols=lambda c: c in {"segment", "pred"})
    except Exception as exc:  # noqa: BLE001
        return {
            "miou_fixed_canonical": None,
            "miou_fixed_yrank": None,
            "n_pred_segments": 0,
            "remap": None,
            "note": f"final.csv read failed: {exc!r}",
        }
    if "pred" not in df.columns or "segment" not in df.columns:
        return {
            "miou_fixed_canonical": None,
            "miou_fixed_yrank": None,
            "n_pred_segments": 0,
            "remap": None,
            "note": "pred or segment column missing",
        }
    df = df[df["segment"].notna()].copy()
    if df.empty:
        return {
            "miou_fixed_canonical": None,
            "miou_fixed_yrank": None,
            "n_pred_segments": 0,
            "remap": None,
            "note": "no rows with segment",
        }
    gt = df["segment"].astype(int).to_numpy()
    pred = df["pred"].astype(int).to_numpy()
    valid = (gt >= 0) & (gt <= max_class) & (pred >= 0) & (pred <= max_class)
    if not valid.any():
        return {
            "miou_fixed_canonical": None,
            "miou_fixed_yrank": None,
            "n_pred_segments": 0,
            "remap": None,
            "note": "no in-range labels",
        }
    gt_v = gt[valid]
    pred_v = pred[valid]
    classes = np.sort(np.unique(np.concatenate([gt_v, pred_v])))
    miou_canonical = float(
        np.mean(jaccard_score(gt_v, pred_v, average=None, labels=classes, zero_division=0))
    )
    # Build y-rank remap from pixel_to_point.
    rows_per_idx = _pixel_row_per_point(p2p_path)
    if rows_per_idx is None:
        return {
            "miou_fixed_canonical": miou_canonical,
            "miou_fixed_yrank": None,
            "n_pred_segments": int(len({int(p) for p in pred_v if int(p) != 0})),
            "remap": None,
            "note": "pixel_to_point.pkl missing or empty",
        }
    # Per pred ID, mean pixel-row over the rows corresponding to its points.
    # We use df.index (not yet reset) as point indices, matching how
    # final.csv is written: row N is point index N.
    df_indices = df.index.to_numpy()  # original point indices (pre-segment-filter)
    df_indices_valid = df_indices[valid]
    mean_y_per_id: dict[int, float] = {}
    pred_unique = sorted({int(p) for p in pred_v if int(p) != 0})
    for pid in pred_unique:
        mask = pred_v == pid
        idxs = df_indices_valid[mask]
        idxs = idxs[idxs < rows_per_idx.size]
        if idxs.size == 0:
            continue
        ys = rows_per_idx[idxs]
        ys = ys[~np.isnan(ys)]
        if ys.size == 0:
            continue
        mean_y_per_id[pid] = float(ys.mean())
    if not mean_y_per_id:
        return {
            "miou_fixed_canonical": miou_canonical,
            "miou_fixed_yrank": None,
            "n_pred_segments": len(pred_unique),
            "remap": None,
            "note": "no pred segment had any pixel rows resolved",
        }
    pred_yrank, remap = _y_rank_remap(pred_v, mean_y_per_id)
    classes_y = np.sort(np.unique(np.concatenate([gt_v, pred_yrank])))
    miou_yrank = float(
        np.mean(jaccard_score(gt_v, pred_yrank, average=None, labels=classes_y, zero_division=0))
    )
    return {
        "miou_fixed_canonical": miou_canonical,
        "miou_fixed_yrank": miou_yrank,
        "n_pred_segments": len(pred_unique),
        "remap": remap,
        "note": "ok",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    panel = json.loads(PANEL_PATH.read_text())
    rings = list(panel["rings"])
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_root = HELDOUT_ROOT / arm
        for rinfo in rings:
            tid = rinfo["tunnel_id"]
            rid = int(rinfo["ring_id"])
            rk = rinfo["ring_key"]
            ring_dir = arm_root / tid / f"r{rid}"
            res = _score_ring(ring_dir)
            rows.append({
                "arm": arm,
                "ring_key": rk,
                "split": rinfo.get("split"),
                "regime_label": rinfo.get("regime_label"),
                **res,
            })
            logger.info(
                "[%s] %s canonical=%s yrank=%s n_pred=%d %s",
                arm, rk,
                res.get("miou_fixed_canonical"),
                res.get("miou_fixed_yrank"),
                res.get("n_pred_segments") or 0,
                res.get("note"),
            )
    out_csv = HELDOUT_ROOT / "scoreboard_yrank.csv"
    keys = list(rows[0].keys())
    # remap is a dict; serialise as JSON string for CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            if r2.get("remap") is not None:
                r2["remap"] = json.dumps({str(k): int(v) for k, v in r2["remap"].items()})
            w.writerow(r2)
    logger.info("wrote %s (%d rows)", out_csv.relative_to(REPO_ROOT), len(rows))

    # Per-arm means.
    summary: dict[str, Any] = {}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        c = [r["miou_fixed_canonical"] for r in sub if r["miou_fixed_canonical"] is not None]
        y = [r["miou_fixed_yrank"] for r in sub if r["miou_fixed_yrank"] is not None]
        summary[arm] = {
            "n_rings": len(sub),
            "n_canonical_ok": len(c),
            "n_yrank_ok": len(y),
            "mean_miou_canonical": float(np.mean(c)) if c else None,
            "mean_miou_yrank": float(np.mean(y)) if y else None,
        }
    if summary["a_unanchored"]["mean_miou_yrank"] is not None and summary["b_anchored"]["mean_miou_yrank"] is not None:
        summary["yrank_lift_b_minus_a"] = (
            summary["b_anchored"]["mean_miou_yrank"]
            - summary["a_unanchored"]["mean_miou_yrank"]
        )
    out_json = HELDOUT_ROOT / "yrank_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

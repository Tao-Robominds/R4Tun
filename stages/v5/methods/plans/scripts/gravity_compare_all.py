#!/usr/bin/env python3
"""Compute naive + permutation-invariant mIoU for gravity-aligned and A0 baseline.

Reports a side-by-side table: A0 naive, A0 permuted, Gravity naive,
Gravity permuted, delta, and whether each meets the 0.4 target.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score


def permutation_invariant_miou(gt: np.ndarray, pred: np.ndarray, n_classes: int = 8) -> tuple[float, dict[int, int]]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for g, p in zip(gt, pred):
        if 0 <= g < n_classes and 0 <= p < n_classes:
            cm[g, p] += 1
    row_sum = cm.sum(1)
    col_sum = cm.sum(0)
    iou = np.zeros_like(cm, dtype=np.float64)
    for g in range(n_classes):
        for p in range(n_classes):
            u = row_sum[g] + col_sum[p] - cm[g, p]
            if u > 0:
                iou[g, p] = cm[g, p] / u
    row_ind, col_ind = linear_sum_assignment(-iou)
    gt_present = set(np.unique(gt))
    ious = [iou[g, col_ind[g]] for g in range(n_classes) if g in gt_present]
    return (float(np.mean(ious)) if ious else 0.0), {int(g): int(col_ind[g]) for g in range(n_classes) if g in gt_present}


def _load(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    if "segment" not in df.columns or "pred" not in df.columns:
        return None
    gt = df["segment"].fillna(0).astype(int).to_numpy()
    pred = df["pred"].astype(int).to_numpy()
    return gt, pred


def _score(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    classes = sorted(set(np.unique(gt)) | set(np.unique(pred)))
    naive = float(jaccard_score(gt, pred, average="macro", labels=classes, zero_division=0))
    perm, _ = permutation_invariant_miou(gt, pred)
    return naive, perm


def main() -> int:
    rings = [
        "4-3/r170", "4-3/r171",
        "4-4/r212", "4-4/r217",
        "4-5/r244",
        "4-6/r275", "4-6/r276",
        "5-1/r110", "5-1/r111",
        "5-6/r284",
        "5-7/r316", "5-7/r322",
    ]
    threshold = 0.40

    rows = []
    for rk in rings:
        t, r = rk.split("/", 1)
        a0_path = Path(f"logs/proxy_validation_v1/heldout_reflection_test/{rk}/A0_no_reflection/final.csv")
        grav_path = Path(f"logs/gravity_unwrap_v1/pipeline/{rk}/final.csv")

        a0 = _load(a0_path)
        grav = _load(grav_path)

        a0_n, a0_p = _score(*a0) if a0 is not None else (float("nan"), float("nan"))
        g_n, g_p = _score(*grav) if grav is not None else (float("nan"), float("nan"))

        delta_perm = g_p - a0_p if np.isfinite(a0_p) and np.isfinite(g_p) else float("nan")
        meets_target = bool(np.isfinite(g_p) and g_p >= threshold)
        rows.append({
            "ring": rk,
            "A0_naive": a0_n,
            "A0_perm": a0_p,
            "Gravity_naive": g_n,
            "Gravity_perm": g_p,
            "delta_perm": delta_perm,
            "meets_0.4": meets_target,
        })

    df = pd.DataFrame(rows)
    # Report
    print(f"\n{'='*90}")
    print("Gravity-align pipeline vs A0 baseline (permutation-invariant mIoU is the fair metric)")
    print(f"{'='*90}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}" if np.isfinite(v) else "—"))
    # Summary
    valid = df.dropna(subset=["A0_perm", "Gravity_perm"])
    print(f"\nValid rings: {len(valid)}/{len(df)}")
    print(f"Mean A0 (perm):      {valid['A0_perm'].mean():.3f}")
    print(f"Mean Gravity (perm): {valid['Gravity_perm'].mean():.3f}")
    print(f"Mean delta:          {valid['delta_perm'].mean():+.3f}")
    print(f"Hit-rate >= 0.4 (perm): A0={int((valid['A0_perm']>=threshold).sum())}/{len(valid)},  Gravity={int((valid['Gravity_perm']>=threshold).sum())}/{len(valid)}")

    out = Path("logs/gravity_unwrap_v1/gravity_vs_a0_compare.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

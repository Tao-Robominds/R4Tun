#!/usr/bin/env python3
"""
Reproduce Table 9 (error composition as a fraction of ground-truth points)
for Opus-4.6 m+s+k vs the SAM4Tun baseline, using the current 5-sample data.

Per-point ground-truth vs prediction is read from only_label.csv
(gt_labels / pred_labels) -- the same arrays used to compute the reported
mIoU -- so the four categories partition every ground-truth point:

  Correct : pred == gt        (background-correct + segment-correct)
  FN      : gt  > 0, pred == 0 (segment predicted as background)
  FP      : gt == 0, pred  > 0 (background predicted as segment)
  Swap    : gt  > 0, pred  > 0, gt != pred (segment predicted as wrong class)

Families follow the paper convention:
  Regular = reg u con (1-1, 2-1, 3-1-1)
  Complex = com        (4-1, 5-1)

Points are pooled across the rings in each family (fraction of GT points).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

FAMILIES = {
    "Regular": ["1-1", "2-1", "3-1-1"],
    "Complex": ["4-1", "5-1"],
}

METHODS = {
    "SAM4Tun": lambda ring: REPO_ROOT / "data" / "static" / ring / "only_label.csv",
    "m+s+k": lambda ring: REPO_ROOT
    / "data"
    / "ablation"
    / "memory+state+knowledge"
    / f"{ring}_opus4.6"
    / "only_label.csv",
}


def load_gt_pred(path: Path):
    df = pd.read_csv(path, usecols=["gt_labels", "pred_labels"])
    return df["gt_labels"].to_numpy(int), df["pred_labels"].to_numpy(int)


def categorise(gt: np.ndarray, pred: np.ndarray):
    correct = gt == pred
    fn = (gt > 0) & (pred == 0)
    fp = (gt == 0) & (pred > 0)
    swap = (gt > 0) & (pred > 0) & (gt != pred)
    n = len(gt)
    return {
        "Correct": correct.sum() / n * 100,
        "FN": fn.sum() / n * 100,
        "FP": fp.sum() / n * 100,
        "Swap": swap.sum() / n * 100,
        "n": n,
    }


def main():
    results = {}
    for fam, rings in FAMILIES.items():
        for method, path_fn in METHODS.items():
            gts, preds = [], []
            for ring in rings:
                gt, pred = load_gt_pred(path_fn(ring))
                gts.append(gt)
                preds.append(pred)
            gt = np.concatenate(gts)
            pred = np.concatenate(preds)
            results[(fam, method)] = categorise(gt, pred)

    hdr = f"{'Category':<9} {'Method':<9} {'Correct':>8} {'FN':>6} {'FP':>6} {'Swap':>6}"
    print(hdr)
    print("-" * len(hdr))
    for fam in FAMILIES:
        for method in METHODS:
            r = results[(fam, method)]
            print(
                f"{fam:<9} {method:<9} "
                f"{r['Correct']:>7.0f}% {r['FN']:>5.0f}% {r['FP']:>5.0f}% {r['Swap']:>5.0f}%"
            )

    print("\n(unrounded, with pooled point counts)")
    for fam in FAMILIES:
        for method in METHODS:
            r = results[(fam, method)]
            print(
                f"  {fam:<8} {method:<7} "
                f"Correct={r['Correct']:.2f} FN={r['FN']:.2f} "
                f"FP={r['FP']:.2f} Swap={r['Swap']:.2f}  n={r['n']:,}"
            )


if __name__ == "__main__":
    main()

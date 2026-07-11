#!/usr/bin/env python3
"""Table 9: error composition (Opus m+s+k vs SAM4Tun static), pooled over GT points."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

REGULAR = [
    "1-1", "1-2", "1-3", "1-4", "1-5",
    "2-1", "2-2", "2-3", "2-4", "2-5",
    "3-1-1", "3-1-2", "3-1-3",
]
COMPLEX = [
    "4-1", "4-2", "4-3", "4-4", "4-5", "4-6", "4-7", "4-8", "4-9", "4-10",
    "5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7",
]


def path_static(tid: str) -> Path:
    return REPO / "data" / "static" / tid / "only_label.csv"


def path_msk(tid: str) -> Path:
    return (
        REPO / "data" / "ablation" / "memory+state+knowledge" / f"{tid}_opus4.6" / "only_label.csv"
    )


def categorise(gt: np.ndarray, pred: np.ndarray) -> dict:
    n = len(gt)
    correct = gt == pred
    fn = (gt > 0) & (pred == 0)
    fp = (gt == 0) & (pred > 0)
    swap = (gt > 0) & (pred > 0) & (gt != pred)
    return {
        "Correct": correct.sum() / n * 100,
        "FN": fn.sum() / n * 100,
        "FP": fp.sum() / n * 100,
        "Swap": swap.sum() / n * 100,
        "n": n,
    }


def pool(tunnels: list[str], path_fn) -> tuple[dict | None, list[str]]:
    missing = []
    gts, preds = [], []
    for tid in tunnels:
        p = path_fn(tid)
        if not p.is_file():
            missing.append(tid)
            continue
        df = pd.read_csv(p, usecols=["gt_labels", "pred_labels"])
        gts.append(df["gt_labels"].to_numpy(int))
        preds.append(df["pred_labels"].to_numpy(int))
    if not gts:
        return None, missing
    return categorise(np.concatenate(gts), np.concatenate(preds)), missing


def main() -> None:
    families = {"Regular": REGULAR, "Complex": COMPLEX}
    methods = {"SAM4Tun": path_static, "m+s+k": path_msk}
    results: dict = {}
    gaps: dict = {}

    for fam, tunnels in families.items():
        for method, pfn in methods.items():
            stats, missing = pool(tunnels, pfn)
            results[(fam, method)] = stats
            if missing:
                gaps[(fam, method)] = missing

    print("Table 9 — error composition (Opus-4.6 m+s+k vs SAM4Tun static)")
    print("Pooled fraction of ground-truth points from only_label.csv\n")
    hdr = f"{'Category':<9} {'Method':<9} {'Correct':>8} {'FN':>6} {'FP':>6} {'Swap':>6}"
    print(hdr)
    print("-" * len(hdr))
    for fam in families:
        for method in methods:
            r = results.get((fam, method))
            if r is None:
                print(f"{fam:<9} {method:<9}  (no data)")
            else:
                print(
                    f"{fam:<9} {method:<9} "
                    f"{r['Correct']:>7.0f}% {r['FN']:>5.0f}% {r['FP']:>5.0f}% {r['Swap']:>5.0f}%"
                )

    print("\nDetail (unrounded):")
    for fam in families:
        for method in methods:
            r = results.get((fam, method))
            if r:
                print(
                    f"  {fam} {method}: Correct={r['Correct']:.2f}% FN={r['FN']:.2f}% "
                    f"FP={r['FP']:.2f}% Swap={r['Swap']:.2f}% n={r['n']:,}"
                )

    if gaps:
        print("\nMissing only_label.csv:")
        for k, v in sorted(gaps.items()):
            print(f"  {k[0]} / {k[1]}: {len(v)} tunnels — {', '.join(v)}")


if __name__ == "__main__":
    main()

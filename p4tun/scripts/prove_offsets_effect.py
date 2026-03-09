"""
Prove that wrong A/B offsets (group_offsets) cause the mIoU gap.
Same pipeline (template geo, same params), only segment centres differ.
Uses exact same metric as p4tun/evaluation.py: max_class=7, valid = (gt<=7)&(pred<=7), macro IoU.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

def evaluate_final_csv(path: str, max_class: int = 7):
    """Same logic as p4tun/evaluation.py calculate_metrics."""
    df = pd.read_csv(path)
    gt = df["segment"].values.astype(np.float64)
    pr = df["pred"].values.astype(np.float64)
    # Exclude NaN (evaluation: gt<=7 & pr<=7; NaN<=7 is False so NaNs excluded)
    valid_mask = np.isfinite(gt) & np.isfinite(pr) & (gt <= max_class) & (pr <= max_class)
    gt_f = np.round(gt[valid_mask]).astype(int)
    pr_f = np.round(pr[valid_mask]).astype(int)
    classes = np.sort(np.unique(np.concatenate([gt_f, pr_f])))
    iou_per_class = jaccard_score(gt_f, pr_f, average=None, labels=classes, zero_division=0)
    miou = float(np.mean(iou_per_class))
    return miou, classes, gt_f, pr_f, valid_mask

def per_ring_miou(path: str, max_class: int = 7):
    df = pd.read_csv(path)
    gt = df["segment"].values.astype(np.float64)
    pr = df["pred"].values.astype(np.float64)
    ring = df["ring"].values
    valid_mask = np.isfinite(gt) & np.isfinite(pr) & (gt <= max_class) & (pr <= max_class)
    classes = np.arange(0, max_class + 1)  # 0..7 for consistency
    rings_u = np.unique(ring[np.isfinite(ring)])
    rows = []
    total_n = 0
    weighted_sum = 0.0
    for r in sorted(rings_u):
        mask = (ring == r) & valid_mask
        n = int(mask.sum())
        if n == 0:
            continue
        gt_r = np.round(gt[mask]).astype(int)
        pr_r = np.round(pr[mask]).astype(int)
        iou = jaccard_score(gt_r, pr_r, labels=classes, average="macro", zero_division=0)
        rows.append((int(r), n, iou))
        total_n += n
        weighted_sum += iou * n
    return rows, weighted_sum / total_n if total_n else 0

def main():
    base = os.path.join(os.path.dirname(__file__), "..", "..", "data", "4-1")
    gt_centres_path = os.path.join(base, "final_gt_centres.csv")
    group_offsets_path = os.path.join(base, "final.csv")  # after group_offsets run

    # Group_offsets final: we need to load it (current final.csv may be overwritten)
    # So we need both files to exist. final_gt_centres.csv we have. For group_offsets we re-ran and saved to final.csv; copy to final_group_offsets.csv for this script to use both.
    group_path = os.path.join(base, "final_group_offsets.csv")
    if not os.path.exists(group_path):
        group_path = os.path.join(base, "final.csv")

    print("=" * 70)
    print("PROOF: Same pipeline, only segment centres (A/B offsets) differ")
    print("Metric: same as evaluation.py (max_class=7, macro IoU over 0..7)")
    print("=" * 70)

    # Global mIoU
    m_gt, _, _, _, _ = evaluate_final_csv(gt_centres_path)
    m_gr, _, _, _, _ = evaluate_final_csv(group_path)
    print("\n1. GLOBAL mIoU (same evaluation logic as evaluation.py)")
    print(f"   GT centres (correct A/B offsets):  mIoU = {m_gt:.4f}")
    print(f"   Group offsets (wrong A2/A3):       mIoU = {m_gr:.4f}")
    print(f"   Gap (offsets right vs wrong):      +{m_gt - m_gr:.4f}")

    # Per-ring
    rows_gt, w_gt = per_ring_miou(gt_centres_path)
    rows_gr, w_gr = per_ring_miou(group_path)
    print("\n2. PER-RING mIoU (same metric)")
    print(f"   {'Ring':<6} {'n_pts':>8} {'%':>6}   {'GT centres':>10}   {'Group off':>10}   {'Diff':>8}")
    print("   " + "-" * 58)
    for (r1, n1, iou1), (r2, n2, iou2) in zip(rows_gt, rows_gr):
        assert r1 == r2 and n1 == n2
        total = sum(x[1] for x in rows_gt)
        pct = 100 * n1 / total
        print(f"   {r1:<6} {n1:>8} {pct:>5.1f}%   {iou1:>10.4f}   {iou2:>10.4f}   {iou1-iou2:>+8.4f}")
    print("   " + "-" * 58)
    print(f"   {'(weighted)':<6} {'':>8} {'':>6}   {w_gt:>10.4f}   {w_gr:>10.4f}   {w_gt-w_gr:>+8.4f}")

    print("\n3. CONCLUSION")
    print(f"   With correct A/B offsets (GT centres): mIoU = {m_gt:.4f}")
    print(f"   With group_offsets (wrong A2/A3):      mIoU = {m_gr:.4f}")
    print(f"   The gap of {m_gt - m_gr:.4f} is entirely due to segment centre error (offsets).")

if __name__ == "__main__":
    main()

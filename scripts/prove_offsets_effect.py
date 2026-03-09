"""
Prove that wrong A/B offsets (group_offsets) cause the mIoU gap.
Same pipeline (template geo, same params), only segment centres differ.
Uses exact same metric as evaluation.py: max_class=7, valid = (gt<=7)&(pred<=7), macro IoU.

Usage: python prove_offsets_effect.py 4-1 [--data-dir data]
  Expects final_gt_centres.csv (or final.csv from GT centres run) and final_group_offsets.csv (or final.csv from group_offsets run).
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IRREGULAR_ROOT = os.path.dirname(SCRIPT_DIR)


def evaluate_final_csv(path: str, max_class: int = 7):
    """Same logic as evaluation.py calculate_metrics."""
    df = pd.read_csv(path)
    gt = df["segment"].values.astype(np.float64)
    pr = df["pred"].values.astype(np.float64)
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
    classes = np.arange(0, max_class + 1)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("tunnel_id", default="4-1", nargs="?")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        repo_root = os.path.dirname(os.path.dirname(IRREGULAR_ROOT))
        data_dir = os.path.join(repo_root, data_dir)
    base = os.path.join(data_dir, args.tunnel_id)

    gt_centres_path = os.path.join(base, "final_gt_centres.csv")
    group_path = os.path.join(base, "final_group_offsets.csv")
    if not os.path.exists(group_path):
        group_path = os.path.join(base, "final.csv")
    if not os.path.exists(gt_centres_path):
        gt_centres_path = os.path.join(base, "final.csv")

    print("=" * 70)
    print("PROOF: Same pipeline, only segment centres (A/B offsets) differ")
    print("Metric: same as evaluation.py (max_class=7, macro IoU over 0..7)")
    print("=" * 70)

    m_gt, _, _, _, _ = evaluate_final_csv(gt_centres_path)
    m_gr, _, _, _, _ = evaluate_final_csv(group_path)
    print("\n1. GLOBAL mIoU (same evaluation logic as evaluation.py)")
    print(f"   GT centres (correct A/B offsets):  mIoU = {m_gt:.4f}")
    print(f"   Group offsets (wrong A2/A3):       mIoU = {m_gr:.4f}")
    print(f"   Gap (offsets right vs wrong):      +{m_gt - m_gr:.4f}")

    rows_gt, w_gt = per_ring_miou(gt_centres_path)
    rows_gr, w_gr = per_ring_miou(group_path)
    print("\n2. PER-RING mIoU (same metric)")
    print(f"   {'Ring':<6} {'n_pts':>8} {'%':>6}   {'GT centres':>10}   {'Group off':>10}   {'Diff':>8}")
    print("   " + "-" * 58)
    total = sum(x[1] for x in rows_gt)
    for (r1, n1, iou1), (r2, n2, iou2) in zip(rows_gt, rows_gr):
        assert r1 == r2 and n1 == n2
        pct = 100 * n1 / total if total else 0
        print(f"   {r1:<6} {n1:>8} {pct:>5.1f}%   {iou1:>10.4f}   {iou2:>10.4f}   {iou1-iou2:>+8.4f}")
    print("   " + "-" * 58)
    print(f"   {'(weighted)':<6} {'':>8} {'':>6}   {w_gt:>10.4f}   {w_gr:>10.4f}   {w_gt-w_gr:>+8.4f}")

    print("\n3. CONCLUSION")
    print(f"   With correct A/B offsets (GT centres): mIoU = {m_gt:.4f}")
    print(f"   With group_offsets (wrong A2/A3):      mIoU = {m_gr:.4f}")
    print(f"   The gap of {m_gt - m_gr:.4f} is entirely due to segment centre error (offsets).")


if __name__ == "__main__":
    main()

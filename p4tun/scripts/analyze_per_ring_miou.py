"""
Per-ring mIoU and segment-centre error analysis.
Usage: python p4tun/scripts/analyze_per_ring_miou.py 4-1 [--data-dir data]
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tunnel_id", default="4-1", nargs="?")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    tunnel_dir = os.path.join(args.data_dir, args.tunnel_id)
    final_path = os.path.join(tunnel_dir, "final.csv")
    if not os.path.exists(final_path):
        print(f"Not found: {final_path}")
        return
    df = pd.read_csv(final_path)
    gt = np.nan_to_num(df["segment"].values, nan=-1).astype(int)
    pr = np.nan_to_num(df["pred"].values, nan=-1).astype(int)
    ring = df["ring"].values
    valid = (gt >= 1) & (gt <= 7) & (pr >= 0) & (pr <= 7)
    classes = np.arange(1, 8)
    # Per physical ring
    rings = np.unique(ring)
    print("Per-ring point count and mIoU (blocks only, 1-7):")
    print("-" * 60)
    total_pts = 0
    weighted_miou = 0
    for r in sorted(rings):
        if np.isnan(r):
            continue
        mask = (ring == r) & valid
        n = mask.sum()
        if n == 0:
            print(f"  Ring {int(r)}: 0 points")
            continue
        total_pts += n
        iou_r = jaccard_score(gt[mask], pr[mask], labels=classes, average="macro", zero_division=0)
        weighted_miou += iou_r * n
        pct = 100 * n / valid.sum()
        print(f"  Ring {int(r)}: n={n:>7} ({pct:5.1f}%)  mIoU(ring)={iou_r:.4f}")
    if total_pts > 0:
        overall = weighted_miou / total_pts
        print("-" * 60)
        print(f"  Weighted mIoU (by ring): {overall:.4f}")
    # Global mIoU for comparison
    miou_global = jaccard_score(gt[valid], pr[valid], labels=classes, average="macro", zero_division=0)
    print(f"  Global mIoU (all points):  {miou_global:.4f}")

if __name__ == "__main__":
    main()

"""
Per-ring mIoU analysis.
Usage: python analyze_per_ring_miou.py 4-1 [--data-dir data]
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IRREGULAR_ROOT = os.path.dirname(SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tunnel_id", default="4-1", nargs="?")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        repo_root = os.path.dirname(os.path.dirname(IRREGULAR_ROOT))
        data_dir = os.path.join(repo_root, data_dir)
    tunnel_dir = os.path.join(data_dir, args.tunnel_id)

    final_path = os.path.join(tunnel_dir, "final.csv")
    if not os.path.exists(final_path):
        print(f"Not found: {final_path}")
        return
    df = pd.read_csv(final_path)
    gt = df["segment"].values.astype(np.float64)
    pr = df["pred"].values.astype(np.float64)
    ring = df["ring"].values
    valid = np.isfinite(gt) & np.isfinite(pr) & (gt >= 1) & (gt <= 7) & (pr >= 0) & (pr <= 7)
    classes = np.arange(1, 8)
    rings = np.unique(ring[np.isfinite(ring)])
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
        iou_r = jaccard_score(gt[mask].astype(int), pr[mask].astype(int), labels=classes, average="macro", zero_division=0)
        weighted_miou += iou_r * n
        pct = 100 * n / valid.sum()
        print(f"  Ring {int(r)}: n={n:>7} ({pct:5.1f}%)  mIoU(ring)={iou_r:.4f}")
    if total_pts > 0:
        overall = weighted_miou / total_pts
        print("-" * 60)
        print(f"  Weighted mIoU (by ring): {overall:.4f}")
    miou_global = jaccard_score(gt[valid].astype(int), pr[valid].astype(int), labels=classes, average="macro", zero_division=0)
    print(f"  Global mIoU (all points):  {miou_global:.4f}")


if __name__ == "__main__":
    main()

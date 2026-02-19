#!/usr/bin/env python3
"""Compare detected all_segments.csv vs all_segments_gt.csv; report per-block pixel drift."""
import os
import sys
import pandas as pd
import numpy as np


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    tunnel_id = sys.argv[2] if len(sys.argv) > 2 else "5-1"
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    det_path = os.path.join(tunnel_dir, "all_segments.csv")

    if not os.path.exists(gt_path):
        print(f"GT not found: {gt_path}")
        sys.exit(1)
    if not os.path.exists(det_path):
        print(f"Detected not found: {det_path}")
        sys.exit(1)

    gt = pd.read_csv(gt_path)
    det = pd.read_csv(det_path)

    merged = gt.merge(det, on=["Ring", "Block"], suffixes=("_gt", "_det"))
    merged["drift_px"] = np.sqrt(
        (merged["X_gt"] - merged["X_det"]) ** 2 + (merged["Y_gt"] - merged["Y_det"]) ** 2
    )

    print(f"Drift comparison: {tunnel_id} ({len(merged)} blocks matched)")
    print(f"  Mean drift (px):  {merged['drift_px'].mean():.2f}")
    print(f"  Max drift (px):   {merged['drift_px'].max():.2f}")
    print(f"  Median drift (px): {merged['drift_px'].median():.2f}")
    print("\nPer-ring mean drift (px):")
    per_ring = merged.groupby("Ring")["drift_px"].agg(["mean", "max", "count"])
    per_ring.columns = ["mean_px", "max_px", "n"]
    print(per_ring.to_string())
    print("\nPer-block drift (px) - sample (first 15):")
    print(merged[["Ring", "Block", "drift_px"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

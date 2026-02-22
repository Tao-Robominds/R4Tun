#!/usr/bin/env python3
"""
Evaluate 5-1 on the "easy" subset only: exclude ring 4 and the 8 wrap blocks.
Reports mIoU, OA, F1 for 5-1 without the problem regions.

Usage: python evaluate_5_1_easy_subset.py [tunnel_id] [final.csv path]
  Default tunnel_id: 5-1
  If second arg given, use that as path to final CSV (e.g. data/5-1/final_wrap_a.csv).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

# 5-1: enhanced ring 107..113 = ring index 0..6
RING_OFFSET_5_1 = 107
RING_4_ENHANCED = 111  # ring index 4

WRAP_BLOCKS = {
    "0_A3", "1_B2", "2_A1", "4_A1", "4_A2", "4_A3", "4_A4", "4_B1", "5_A4", "6_A3",
}
SEG_TO_BLOCK = {1: "K", 2: "B1", 3: "A1", 4: "A2", 5: "A3", 6: "A4", 7: "B2"}

CLASS_NAMES_7 = {
    0: "Background", 1: "K-block", 2: "B1-block", 3: "A1-block",
    4: "A2-block", 5: "A3-block", 6: "A4-block", 7: "B2-block",
}


def build_easy_mask(ring: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """True for points that are NOT in ring 4 and NOT in a wrap block."""
    # Exclude points with invalid ring
    valid_ring = np.isfinite(ring)
    # Ring 4 (ring index 4) in 5-1 has enhanced ring value 111
    not_ring4 = ring != RING_4_ENHANCED

    # Wrap blocks: (ring_index, block_name); only block points (segment 1-7) can be wrap
    ring_idx = np.where(valid_ring, (ring - RING_OFFSET_5_1).astype(int), -1)
    seg_int = np.where(np.isnan(segment) | (segment < 1), 0, np.minimum(segment.astype(int), 7))
    block_names = np.array([SEG_TO_BLOCK.get(int(s), "") for s in seg_int])
    keys = np.array([f"{r}_{b}" for r, b in zip(ring_idx, block_names)])
    not_wrap = np.array([k not in WRAP_BLOCKS for k in keys])

    return valid_ring & not_ring4 & not_wrap  # only points with valid ring, not ring 4, not wrap block


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    tunnel_id = sys.argv[1] if len(sys.argv) > 1 else "5-1"
    tunnel_dir = os.path.join(base_dir, "data", tunnel_id)
    if len(sys.argv) > 2:
        final_path = sys.argv[2]
        if not os.path.isabs(final_path):
            final_path = os.path.join(base_dir, final_path)
    else:
        final_path = os.path.join(tunnel_dir, "final.csv")

    if not os.path.exists(final_path):
        print(f"Not found: {final_path}")
        sys.exit(1)

    print(f"Loading {final_path} (segment, ring, pred only)...")
    df = pd.read_csv(final_path, usecols=["segment", "ring", "pred"])
    n_total = len(df)

    # Same filter as main evaluation: only points with both labels in 0..7
    seg = np.where(np.isnan(df["segment"].values) | (df["segment"].values < 0), 0, df["segment"].values)
    pr = df["pred"].values
    valid_class = (seg <= 7) & (pr <= 7)
    df = df.loc[valid_class].copy()
    n_after_class_filter = len(df)
    if n_after_class_filter < n_total:
        print(f"Filtered {n_total - n_after_class_filter:,} points with class > 7 (match main evaluation)")

    # Build easy mask (exclude ring 4 and wrap blocks)
    ring_vals = df["ring"].values
    seg_vals = df["segment"].values
    easy = build_easy_mask(ring_vals, seg_vals)
    n_easy = easy.sum()
    n_excluded = n_after_class_filter - n_easy
    print(f"Points (class 0-7): {n_after_class_filter:,}")
    print(f"Easy subset (exclude ring 4 + 8 wrap blocks): {n_easy:,} ({100 * n_easy / n_after_class_filter:.1f}%)")
    print(f"Excluded: {n_excluded:,} ({100 * n_excluded / n_after_class_filter:.1f}%)")

    gt = df.loc[easy, "segment"].values
    pred = df.loc[easy, "pred"].values
    gt = np.where(np.isnan(gt) | (gt < 0), 0, np.minimum(gt.astype(int), 7))
    pred = np.where(np.isnan(pred) | (pred < 0), 0, np.minimum(pred.astype(int), 7))
    classes = np.sort(np.unique(np.concatenate([gt, pred])))

    oa = accuracy_score(gt, pred)
    f1 = f1_score(gt, pred, average="macro", labels=classes, zero_division=0)
    iou_per_class = jaccard_score(gt, pred, average=None, labels=classes, zero_division=0)
    miou = float(np.mean(iou_per_class))

    print()
    print("=" * 50)
    print(f"{tunnel_id} EASY SUBSET (no ring 4, no wrap blocks)")
    print("=" * 50)
    print(f"  OA:   {oa:.3f}")
    print(f"  F1:   {f1:.3f}")
    print(f"  mIoU: {miou:.3f}")
    print("  (Full 5-1 mIoU ~0.762 is higher because ring 4 uses GT override.)")
    print()
    print("Per-class IoU (easy subset):")
    for c, iou in zip(classes, iou_per_class):
        name = CLASS_NAMES_7.get(int(c), f"Class_{c}")
        print(f"  {name}: {iou:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()

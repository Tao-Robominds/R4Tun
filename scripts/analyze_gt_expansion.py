#!/usr/bin/env python3
"""Compute per-ring k_to_b and ab_step from all_segments_gt.csv for BO bounds."""
import os
import sys
import pandas as pd
import numpy as np

WALK_ORDER = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]


def wrap_step(y_from: float, y_to: float, height: int) -> float:
    """Forward step (increasing Y with wrap) from y_from to y_to; always in [0, height)."""
    d = (y_to - y_from) % height
    if d < 0:
        d += height
    return d


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    tunnel_id = sys.argv[2] if len(sys.argv) > 2 else "5-1"
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    if not os.path.exists(gt_path):
        print(f"Not found: {gt_path}")
        sys.exit(1)

    gt = pd.read_csv(gt_path)
    # Image height from depth map (circumference in px)
    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if os.path.exists(depth_path):
        depth = np.load(depth_path)
        img_height = depth.shape[0]
    else:
        img_height = 4712
    print(f"Image height (Y): {img_height}px")

    k_to_b_list = []
    ab_step_list = []
    for ring in sorted(gt["Ring"].unique()):
        df = gt[gt["Ring"] == ring].set_index("Block")
        ys = {b: df.loc[b, "Y"] for b in WALK_ORDER if b in df.index}
        if len(ys) != 7:
            print(f"Ring {ring}: missing blocks, skip")
            continue
        steps = []
        for i, b in enumerate(WALK_ORDER):
            next_b = WALK_ORDER[(i + 1) % 7]
            step = wrap_step(ys[b], ys[next_b], img_height)
            steps.append(step)
        # k_to_b = K -> B1 (first step)
        k_to_b_list.append(steps[0])
        # ab_step = mean of B1->A1, A1->A2, A2->A3, A3->A4, A4->B2 (steps 1..5)
        ab_step_list.append(np.mean(steps[1:6]))

    k_to_b_arr = np.array(k_to_b_list)
    ab_arr = np.array(ab_step_list)
    print("\nPer-ring from GT (px):")
    print("Ring  k_to_b   ab_step")
    for r in range(len(k_to_b_list)):
        print(f"  {r}   {k_to_b_list[r]:7.1f}   {ab_step_list[r]:7.1f}")
    print(f"\nMin   {k_to_b_arr.min():7.1f}   {ab_arr.min():7.1f}")
    print(f"Max   {k_to_b_arr.max():7.1f}   {ab_arr.max():7.1f}")
    print(f"Mean  {k_to_b_arr.mean():7.1f}   {ab_arr.mean():7.1f}")
    # Suggest bounds: allow ±40% around min/max
    k_lo = max(50, k_to_b_arr.min() * 0.6)
    k_hi = min(img_height, k_to_b_arr.max() * 1.4)
    ab_lo = max(100, ab_arr.min() * 0.6)
    ab_hi = min(img_height, ab_arr.max() * 1.4)
    print(f"\nSuggested bounds: k_to_b_r* [{k_lo:.0f}, {k_hi:.0f}], ab_step_r* [{ab_lo:.0f}, {ab_hi:.0f}]")


if __name__ == "__main__":
    main()

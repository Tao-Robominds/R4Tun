"""
Reproduce the 0.805 mIoU baseline for tunnel 5-1.

Uses GT per-instance XY bounding boxes with shrink(sx=4, sy=2)
and distance-based conflict resolution. Requires GT labels in enhanced.csv.

Usage:
    python baselines/5-1/reproduce_0805.py
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

TUNNEL_DIR = "data/5-1"
SX, SY = 4, 2
BLOCK_TO_LABEL = {"K": 1, "B1": 2, "A1": 3, "A2": 4, "A3": 5, "A4": 6, "B2": 7}


def main():
    df = pd.read_csv(f"{TUNNEL_DIR}/enhanced.csv")
    ptp = pd.DataFrame(pickle.load(open(f"{TUNNEL_DIR}/pixel_to_point.pkl", "rb")))
    all_segments = pd.read_csv(f"{TUNNEL_DIR}/all_segments.csv")
    image = cv2.imread(f"{TUNNEL_DIR}/depth_map.png")
    h, w = image.shape[:2]

    gt = df["segment"].values
    gt_ring = df["ring"].values

    py_arr, px_arr, pidx = ptp["pixel_y"].values, ptp["pixel_x"].values, ptp["index"].values
    valid = np.isin(pidx, df.index.values)
    py_pts = np.full(len(df), -1, dtype=int)
    px_pts = np.full(len(df), -1, dtype=int)
    py_pts[pidx[valid]] = py_arr[valid]
    px_pts[pidx[valid]] = px_arr[valid]

    rings = sorted(all_segments["Ring"].unique())

    # Compute GT per-block XY bounding boxes
    gt_block_bounds = {}
    for r in rings:
        gt_r = 107 + r
        ring_data = all_segments[all_segments["Ring"] == r]
        for _, row in ring_data.iterrows():
            block = row["Block"]
            label = BLOCK_TO_LABEL[block]
            mask = (gt == label) & (gt_ring == gt_r) & (py_pts >= 0) & (px_pts >= 0)
            if mask.sum() == 0:
                continue
            by, bx = py_pts[mask], px_pts[mask]
            gt_block_bounds[(r, block)] = (int(bx.min()), int(bx.max()), int(by.min()), int(by.max()))

    # Build label map with shrink and distance-based conflict resolution
    label_map = np.zeros((h, w), dtype=np.int32)
    dist_map = np.full((h, w), np.inf, dtype=np.float64)
    ring_map = np.zeros((h, w), dtype=np.int32)

    for r in rings:
        ring_data = all_segments[all_segments["Ring"] == r].sort_values("Y")
        for _, row in ring_data.iterrows():
            block = row["Block"]
            label = BLOCK_TO_LABEL[block]
            key = (r, block)
            if key not in gt_block_bounds:
                continue

            x_min, x_max, y_min, y_max = gt_block_bounds[key]
            cx, cy = row["X"], row["Y"]

            x_min_s, x_max_s = x_min + SX, x_max - SX
            y_min_s, y_max_s = y_min + SY, y_max - SY
            if x_min_s > x_max_s or y_min_s > y_max_s:
                continue

            x_mask = np.zeros(w, dtype=bool)
            x_mask[x_min_s : x_max_s + 1] = True
            y_mask = np.zeros(h, dtype=bool)
            y_mask[y_min_s : y_max_s + 1] = True
            bm = y_mask[:, None] & x_mask[None, :]

            yy, xx = np.arange(h), np.arange(w)
            dy = np.minimum(np.abs(yy - cy), h - np.abs(yy - cy))
            dx = np.minimum(np.abs(xx - cx), w - np.abs(xx - cx))
            dist = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)

            update = bm & (dist < dist_map)
            label_map[update] = label
            dist_map[update] = dist[update]
            ring_map[update] = r

    # Project to point cloud
    pred_init = df["pred"].values.copy()
    updatable = np.isin(pred_init[pidx[valid]], [0, 7])
    yv, xv = py_arr[valid][updatable], px_arr[valid][updatable]
    bm_pts = (yv >= 0) & (yv < h) & (xv >= 0) & (xv < w)
    fi = pidx[valid][updatable][bm_pts]
    fy, fx = yv[bm_pts], xv[bm_pts]

    pred = pred_init.copy()
    pred[fi] = label_map[fy, fx]

    ring_count = int(open(f"{TUNNEL_DIR}/ring_count.txt").read())
    pred_ring = np.full(len(df), -1, dtype=int)
    fix_rm = np.where(
        (ring_map >= 1) & (ring_map <= ring_count - 1),
        ring_count - ring_map,
        ring_map,
    )
    pred_ring[fi] = fix_rm[fy, fx]

    df["pred"] = pred
    df["pred_ring"] = pred_ring
    df.to_csv(f"{TUNNEL_DIR}/final.csv", index=False)

    # Quick mIoU check
    eval_mask = (gt <= 7) & (pred <= 7)
    gt_e, pred_e = gt[eval_mask].astype(int), pred[eval_mask].astype(int)
    ious = []
    names = ["BG", "K", "B1", "A1", "A2", "A3", "A4", "B2"]
    for c in range(8):
        tp = ((gt_e == c) & (pred_e == c)).sum()
        fn = ((gt_e == c) & (pred_e != c)).sum()
        fp = ((gt_e != c) & (pred_e == c)).sum()
        iou = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        ious.append(iou)
    miou = np.mean(ious)
    print(f"mIoU = {miou:.4f}")
    for n, v in zip(names, ious):
        print(f"  {n}: {v:.4f}")


if __name__ == "__main__":
    main()

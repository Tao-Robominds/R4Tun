"""
Compute theoretical upper bound: Direct GT label projection.

For each tunnel: paint each 3D point's GT segment onto its projected pixel (last-write-wins),
then assign each point the label at its pixel; evaluate mIoU at point level.
Ceiling < 1.0 due to multiple points mapping to the same pixel (conflicts).
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "data")
TUNNELS = ["1-4", "2-2", "3-1", "4-1", "5-1"]


def segment_count_from_df(df: pd.DataFrame) -> int:
    segs = df["segment"].dropna()
    segs = segs[segs > 0].astype(int).unique()
    return len(segs)


def run_tunnel(tunnel_id: str) -> dict:
    tunnel_dir = os.path.join(DATA_DIR, tunnel_id)
    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    pkl_path = os.path.join(tunnel_dir, "pixel_to_point.pkl")
    depth_path = os.path.join(tunnel_dir, "depth_map.png")

    if not os.path.exists(enhanced_path):
        return {"tunnel": tunnel_id, "mIoU": None, "error": "missing enhanced.csv"}
    if not os.path.exists(pkl_path):
        return {"tunnel": tunnel_id, "mIoU": None, "error": "missing pkl"}

    df = pd.read_csv(enhanced_path)
    try:
        with open(pkl_path, "rb") as f:
            ptp = pickle.load(f)
    except (EOFError, Exception):
        return {"tunnel": tunnel_id, "mIoU": None, "error": "pkl load failed"}
    if not ptp or (hasattr(ptp, "__len__") and len(ptp) == 0):
        return {"tunnel": tunnel_id, "mIoU": None, "error": "empty pkl"}

    gt = df["segment"].values
    n_pts = len(gt)
    if hasattr(ptp, "__iter__") and ptp:
        first = ptp[0]
        h, w = first.get("pixel_y", 0) + 1, first.get("pixel_x", 0) + 1
    else:
        h = w = 0
    if os.path.exists(depth_path):
        import cv2
        img = cv2.imread(depth_path)
        h, w = img.shape[:2]

    label_map = np.zeros((h, w), dtype=np.int32)
    ptp_df = pd.DataFrame(ptp)
    idx_arr = ptp_df["index"].values
    py_arr = ptp_df["pixel_y"].values
    px_arr = ptp_df["pixel_x"].values

    # Paint GT to pixels (last-write-wins)
    for i in range(len(ptp)):
        idx = idx_arr[i]
        if idx >= n_pts:
            continue
        seg = gt[idx]
        if np.isnan(seg) or seg < 0:
            continue
        py, px = int(py_arr[i]), int(px_arr[i])
        if 0 <= py < h and 0 <= px < w:
            label_map[py, px] = int(seg)

    # Pred for each point in pixel_to_point
    pred = np.zeros(n_pts, dtype=np.int32)
    pred[:] = 0
    for i in range(len(ptp)):
        idx = idx_arr[i]
        if idx >= n_pts:
            continue
        py, px = int(py_arr[i]), int(px_arr[i])
        if 0 <= py < h and 0 <= px < w:
            pred[idx] = label_map[py, px]

    # Restrict to points that are in pixel_to_point and have valid GT (0-7)
    max_class = 7
    in_ptp = np.zeros(n_pts, dtype=bool)
    in_ptp[idx_arr[idx_arr < n_pts]] = True
    valid_gt = (gt >= 0) & (gt <= max_class) & ~np.isnan(gt)
    valid_pred = (pred >= 0) & (pred <= max_class)
    eval_mask = in_ptp & valid_gt & valid_pred
    gt_eval = gt[eval_mask].astype(int)
    pred_eval = pred[eval_mask]

    if len(gt_eval) == 0:
        return {"tunnel": tunnel_id, "mIoU": None, "error": "no valid points"}

    n_gt_total = (valid_gt & (gt > 0)).sum()
    n_mapped = eval_mask.sum()
    pct_mapped = 100.0 * n_mapped / n_gt_total if n_gt_total else 0
    n_pixels_filled = np.sum(label_map > 0)
    n_ptp = len(ptp)
    conflicts = n_ptp - n_pixels_filled  # multiple points per pixel

    classes = np.sort(np.unique(np.concatenate([gt_eval, pred_eval])))
    iou_per_class = jaccard_score(gt_eval, pred_eval, average=None, labels=classes, zero_division=0)
    miou = float(np.mean(iou_per_class))

    return {
        "tunnel": tunnel_id,
        "mIoU": round(miou, 4),
        "n_pts": n_pts,
        "n_mapped": int(n_mapped),
        "pct_mapped": round(pct_mapped, 1),
        "n_ptp": n_ptp,
        "n_pixels_filled": int(n_pixels_filled),
        "conflicts": int(conflicts),
        "h": h,
        "w": w,
        "pct_pixel_coverage": round(100.0 * n_pixels_filled / (h * w), 2) if h * w else 0,
    }


def main():
    results = []
    for tid in TUNNELS:
        r = run_tunnel(tid)
        results.append(r)
    for r in results:
        if r.get("mIoU") is not None:
            print(r["tunnel"], r["mIoU"], r["pct_mapped"], r["conflicts"], r["pct_pixel_coverage"])
        else:
            print(r["tunnel"], r.get("error", "?"))
    return results


if __name__ == "__main__":
    main()

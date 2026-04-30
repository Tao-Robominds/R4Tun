"""Extract GT detection artifacts for the ceiling experiment.

Reads `{data_dir}/{tunnel_id}/r{ring_id}/enhanced.csv` (with the GT
`segment` column) and `pixel_to_point.pkl`, then writes:

    all_segments_gt.csv         Ring,Block,X,Y,quality (one row per block)
    boundaries_per_ring_gt.json {"0": [{y, block}, ...]} sorted by y
    detected_gt.csv             Type,X,Y,Confidence (one row for K)

Block centroids are computed in pixel coordinates (pixel_x, pixel_y) of
the surface points that survived preprocessing — i.e. exactly the points
that get back-projected from the depth map during segmentation.

`boundaries_per_ring_gt.json` puts the slot start at the **midpoint
between adjacent block centroids on the depth-map y-axis** (cyclic).
This is what `agents/3_segmentation/segmentation.py::build_boundary_label_map`
expects: `bounds[i].y` is the *start* of slot `i`, with label
`bounds[i].block`. Using centroids directly would label only half of each
block's region; midpoints give the exact GT partition.

Run with the project venv only:

    ./venv/bin/python methods/ablation/scripts/extract_gt_detection.py \\
        --tunnel-id 4-1 --ring-id 116 --data-dir data/ablation
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict

import cv2
import numpy as np
import pandas as pd


SEG_TO_BLOCK_IRREGULAR: Dict[int, str] = {
    1: "K", 2: "B1", 3: "A1", 4: "A2", 5: "A3", 6: "A4", 7: "B2",
}


def _cyclic_midpoint(y_a: float, y_b: float, height: float) -> float:
    """Midpoint along the shorter arc between y_a and y_b on a [0,H) circle."""
    if y_b > y_a:
        return float((y_a + y_b) / 2.0)
    # y_b <= y_a: wrap-around case (going forward through the top edge).
    mid = ((y_a + y_b + height) / 2.0) % height
    return float(mid)


def _circular_mean_y(pixel_y: np.ndarray, height: float) -> float:
    """Mean pixel_y on a [0,H) cyclic axis (handles wrap around the seam)."""
    if len(pixel_y) == 0:
        return float("nan")
    angles = pixel_y.astype(float) * (2.0 * np.pi / height)
    s = float(np.sin(angles).sum())
    c = float(np.cos(angles).sum())
    mean_rad = np.arctan2(s, c)
    if mean_rad < 0:
        mean_rad += 2.0 * np.pi
    return float(mean_rad * height / (2.0 * np.pi))


def extract(tunnel_id: str, ring_id: int, data_dir: str) -> Dict:
    unit = os.path.join(data_dir, tunnel_id, f"r{int(ring_id)}")
    enhanced_path = os.path.join(unit, "enhanced.csv")
    p2p_path = os.path.join(unit, "pixel_to_point.pkl")
    depth_path = os.path.join(unit, "depth_map.png")
    if not os.path.exists(enhanced_path):
        raise FileNotFoundError(enhanced_path)
    if not os.path.exists(p2p_path):
        raise FileNotFoundError(p2p_path)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(depth_path)
    depth_img = cv2.imread(depth_path)
    if depth_img is None:
        raise RuntimeError(f"failed to read {depth_path}")
    height = int(depth_img.shape[0])

    df = pd.read_csv(enhanced_path, usecols=["segment"])
    with open(p2p_path, "rb") as f:
        p2p = pickle.load(f)
    p2p_df = pd.DataFrame(p2p)
    if p2p_df.empty:
        raise RuntimeError(f"empty pixel_to_point.pkl for {tunnel_id}/r{ring_id}")

    p2p_df = p2p_df.set_index("index")
    df_joined = p2p_df.join(df, how="inner")
    df_joined = df_joined.dropna(subset=["segment"])
    if df_joined.empty:
        raise RuntimeError(f"no GT-labeled surface points for {tunnel_id}/r{ring_id}")

    rows = []
    for seg_id, block in SEG_TO_BLOCK_IRREGULAR.items():
        sub = df_joined[df_joined["segment"].astype(int) == seg_id]
        if sub.empty:
            continue
        rows.append({
            "Ring": 0,
            "Block": block,
            "X": float(sub["pixel_x"].mean()),
            "Y": _circular_mean_y(sub["pixel_y"].to_numpy(), float(height)),
            "quality": 1.0,
            "n_points": int(len(sub)),
        })
    if not rows:
        raise RuntimeError(f"no blocks recovered for {tunnel_id}/r{ring_id}")

    seg_df = pd.DataFrame(rows).sort_values("Y").reset_index(drop=True)
    seg_df.to_csv(os.path.join(unit, "all_segments_gt.csv"), index=False)

    # Build the most accurate GT boundaries directly from the depth-map
    # raster: for each pixel-y row, count the GT segments of surface points
    # mapped to that row, smooth the counts cyclically, then take the
    # arg-max. Slot boundaries fall where the smoothed dominant segment
    # changes. Smoothing kills edge oscillations where two adjacent blocks
    # overlap by a few pixels; the only remaining ceiling loss is
    # intra-pixel mixing (multiple GT segments at the same pixel).
    surface_segments = sorted(SEG_TO_BLOCK_IRREGULAR.keys())
    counts = np.zeros((height, len(surface_segments)), dtype=np.int64)
    seg_to_col = {s: i for i, s in enumerate(surface_segments)}
    for s in surface_segments:
        sub = df_joined[df_joined["segment"].astype(int) == s]
        if sub.empty:
            continue
        py = sub["pixel_y"].to_numpy(dtype=np.int64)
        py = np.clip(py, 0, height - 1)
        np.add.at(counts[:, seg_to_col[s]], py, 1)

    smooth_window = max(50, height // 200)
    if smooth_window % 2 == 0:
        smooth_window += 1
    pad = smooth_window // 2
    cyc = np.concatenate([counts[-pad:], counts, counts[:pad]], axis=0)
    cumsum = np.cumsum(cyc, axis=0)
    smoothed = (cumsum[smooth_window:] - cumsum[:-smooth_window]).astype(np.float64)
    if smoothed.shape[0] != height:
        smoothed = smoothed[:height]
    if smoothed.sum(axis=1).min() == 0:
        empty_rows = smoothed.sum(axis=1) == 0
        if empty_rows.any():
            valid_y = np.flatnonzero(~empty_rows)
            ys = np.arange(height)
            d_fwd = (valid_y[None, :] - ys[:, None]) % height
            d_bwd = (ys[:, None] - valid_y[None, :]) % height
            d = np.minimum(d_fwd, d_bwd)
            chosen = valid_y[np.argmin(d, axis=1)]
            smoothed = smoothed[chosen]

    dom_col = smoothed.argmax(axis=1)
    y_to_seg = np.array([surface_segments[c] for c in dom_col], dtype=np.int32)

    bounds = []
    prev_seg = int(y_to_seg[(0 - 1) % height])
    for y in range(height):
        s = int(y_to_seg[y])
        if s != prev_seg:
            bounds.append({"y": int(y), "block": SEG_TO_BLOCK_IRREGULAR[s]})
            prev_seg = s
    if not bounds:
        only_seg = int(y_to_seg[0])
        bounds = [{"y": 0, "block": SEG_TO_BLOCK_IRREGULAR[only_seg]}]
    bounds_sorted = sorted(bounds, key=lambda d: d["y"])
    with open(os.path.join(unit, "boundaries_per_ring_gt.json"), "w") as f:
        json.dump({"0": bounds_sorted}, f, indent=2)

    k_rows = seg_df[seg_df["Block"] == "K"]
    if k_rows.empty:
        detected = pd.DataFrame(columns=["Type", "X", "Y", "Confidence"])
    else:
        detected = pd.DataFrame({
            "Type": ["k_gt"],
            "X": [float(k_rows["X"].iloc[0])],
            "Y": [float(k_rows["Y"].iloc[0])],
            "Confidence": [1.0],
        })
    detected.to_csv(os.path.join(unit, "detected_gt.csv"), index=False)

    return {
        "unit_dir": unit,
        "n_blocks": int(len(seg_df)),
        "blocks_present": [str(b) for b in seg_df["Block"].tolist()],
        "boundaries": bounds,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tunnel-id", required=True)
    p.add_argument("--ring-id", required=True, type=int)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    info = extract(args.tunnel_id, args.ring_id, args.data_dir)
    print(
        f"[gt-extract] {args.tunnel_id}/r{args.ring_id}: "
        f"{info['n_blocks']} blocks present={info['blocks_present']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

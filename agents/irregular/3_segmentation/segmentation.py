"""
Irregular Tunnel Geometric Segmentation — Boundary-Based

Each ring is divided into N slots by boundary Y positions on the circular axis.
Boundaries: from detection only — tunnel_dir/boundaries_per_ring.json (written by 2_detection.py).
If that file is missing: adaptive-cap Voronoi (k_cap, ab_cap).

Pipeline:
    1_preprocessing.py → depth_map.png, enhanced.csv, pixel_to_point.pkl
    2_detection.py     → all_segments.csv, boundaries_per_ring.json
    segmentation.py    → final.csv (segmented point cloud)

Parameters (from parameters_segmentation.json):
    ring_half_width     — X extent of ring band (default: image_width / ring_count / 2)
    k_cap, ab_cap       — used only when boundaries_per_ring.json is missing (Voronoi fallback)
    r_surface_min       — radial cutoff (m): points with r < r_surface_min keep pred=0
                          to drop groove false positives; None = disabled
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2

# Preprocessing assigns pred=7 to retained surface points (before block labeling).
# Segmentation overwrites these with block labels 1..N.
PRED_SURFACE = 7

DEFAULTS = {
    "ring_half_width": None,
    "k_cap": 130,
    "ab_cap": 390,
    "r_surface_min": None,
    "slot_inset_y": 0,
}


def load_parameters(tunnel_id: str, base_dir: str = "data") -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "parameters", tunnel_id, "parameters_segmentation.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def compute_block_to_label_map(segment_per_ring: int) -> dict:
    if segment_per_ring == 7:
        return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
    return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}


# =============================================================================
# Boundary-Based Label Map (primary)
# =============================================================================

def build_boundary_label_map(
    segments_df: pd.DataFrame,
    height: int,
    width: int,
    block_to_label: dict,
    ring_half_width: float,
    boundaries_per_ring: dict,
    slot_inset_y: float = 0,
) -> tuple:
    """Build label map from explicit boundary positions per ring.

    Each ring has N entries [{y, block}, ...] sorted by Y.
    Entry i: from y_i to y_{i+1} (circular), pixels belong to block_i.
    If slot_inset_y > 0, slots are inset by that many pixels at each boundary
    (reduces groove false positives; journal 2026-02-18: sy=2 gave +0.012 mIoU).
    """
    label_map = np.zeros((height, width), dtype=np.int32)
    ring_map = np.full((height, width), -1, dtype=np.int32)
    inset = max(0, float(slot_inset_y))

    for ring_idx in sorted(segments_df['Ring'].unique()):
        ring_key = str(ring_idx)
        if ring_key not in boundaries_per_ring:
            continue

        ring_segs = segments_df[segments_df['Ring'] == ring_idx]
        if ring_segs.empty:
            continue

        band_center = float(np.mean(ring_segs['X'].values))
        x_lo = max(0, int(np.floor(band_center - ring_half_width)))
        x_hi = min(width - 1, int(np.ceil(band_center + ring_half_width)))
        if x_lo > x_hi:
            continue

        bounds = boundaries_per_ring[ring_key]
        n = len(bounds)
        if n == 0:
            continue

        col_labels = np.zeros(height, dtype=np.int32)
        ys = np.arange(height, dtype=np.float64)

        for i in range(n):
            start_y = float(bounds[i]['y'])
            end_y = float(bounds[(i + 1) % n]['y'])
            label = block_to_label.get(bounds[i]['block'], 0)

            if inset > 0:
                s_ins = start_y + inset
                e_ins = end_y - inset
                if end_y > start_y:
                    slot_len = end_y - start_y
                    if slot_len <= 2 * inset:
                        continue
                    mask = (ys >= s_ins) & (ys < e_ins)
                else:
                    slot_len = (height - start_y) + end_y
                    if slot_len <= 2 * inset:
                        continue
                    mask = (ys >= s_ins) | (ys < e_ins)
            else:
                if end_y > start_y:
                    mask = (ys >= start_y) & (ys < end_y)
                else:
                    mask = (ys >= start_y) | (ys < end_y)
            col_labels[mask] = label

        xs = np.arange(x_lo, x_hi + 1)
        label_map[np.ix_(np.arange(height), xs)] = col_labels[:, None]
        ring_map[np.ix_(np.arange(height), xs)] = int(ring_idx)

    return label_map, ring_map


# =============================================================================
# Adaptive-Cap Voronoi Label Map (fallback)
# =============================================================================

def build_voronoi_label_map(
    segments_df: pd.DataFrame,
    height: int,
    width: int,
    block_to_label: dict,
    ring_half_width: float,
    k_cap: float,
    ab_cap: float,
) -> tuple:
    """Nearest-centroid assignment with per-type distance caps.

    Pixels beyond the cap for all centroids become background (label 0).
    """
    label_map = np.zeros((height, width), dtype=np.int32)
    ring_map = np.full((height, width), -1, dtype=np.int32)
    all_ys = np.arange(height, dtype=np.float64)

    for ring_idx in sorted(segments_df['Ring'].unique()):
        ring_segs = segments_df[segments_df['Ring'] == ring_idx]
        if ring_segs.empty:
            continue

        band_center = float(np.mean(ring_segs['X'].values))
        x_lo = max(0, int(np.floor(band_center - ring_half_width)))
        x_hi = min(width - 1, int(np.ceil(band_center + ring_half_width)))
        if x_lo > x_hi:
            continue

        centroids_y, labels, caps = [], [], []
        for _, seg in ring_segs.iterrows():
            lid = block_to_label.get(seg['Block'], 0)
            if lid == 0:
                continue
            centroids_y.append(float(seg['Y']))
            labels.append(lid)
            caps.append(k_cap if seg['Block'] == 'K' else ab_cap)

        if not centroids_y:
            continue

        cy = np.array(centroids_y, dtype=np.float64)
        lb = np.array(labels, dtype=np.int32)
        cap_arr = np.array(caps, dtype=np.float64)

        dy = all_ys[:, None] - cy[None, :]
        dy = np.where(dy > height / 2, dy - height, dy)
        dy = np.where(dy < -height / 2, dy + height, dy)
        dist_abs = np.abs(dy)

        dist_masked = np.where(dist_abs > cap_arr[None, :], np.inf, dist_abs)
        nearest = np.argmin(dist_masked, axis=1)
        min_dist = dist_masked[np.arange(height), nearest]

        valid = np.isfinite(min_dist)
        col_labels = np.where(valid, lb[nearest], 0)
        col_rings = np.where(valid, int(ring_idx), -1)

        xs = np.arange(x_lo, x_hi + 1)
        label_map[np.ix_(np.arange(height), xs)] = col_labels[:, None]
        ring_map[np.ix_(np.arange(height), xs)] = col_rings[:, None]

    return label_map, ring_map


# =============================================================================
# Point Cloud Projection
# =============================================================================

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    """Project 2D label/ring maps back to 3D point cloud."""
    df_out = df.copy()
    pred = df_out['pred'].values
    pred_ring = np.full(len(df_out), -1, dtype=int)

    p2p_df = pd.DataFrame(pixel_to_point)
    py = p2p_df['pixel_y'].values
    px = p2p_df['pixel_x'].values
    indices = p2p_df['index'].values

    h, w = segmented_map.shape

    valid_idx = np.isin(indices, df_out.index.values)
    updatable = np.isin(pred[indices[valid_idx]], [0, PRED_SURFACE])

    y_sel = py[valid_idx][updatable]
    x_sel = px[valid_idx][updatable]
    in_bounds = (y_sel >= 0) & (y_sel < h) & (x_sel >= 0) & (x_sel < w)

    final_indices = indices[valid_idx][updatable][in_bounds]
    final_y = y_sel[in_bounds]
    final_x = x_sel[in_bounds]

    pred[final_indices] = segmented_map[final_y, final_x]
    pred_ring[final_indices] = instance_map[final_y, final_x]

    df_out['pred'] = pred
    df_out['pred_ring'] = pred_ring
    return df_out


# =============================================================================
# Main Pipeline
# =============================================================================

def run_segmentation(
    tunnel_id: str,
    base_dir: str = "data",
    segments_file: str = None,
    override_params: dict = None,
) -> dict:
    """Run segmentation and return {'df': DataFrame, 'label_map': ndarray}."""
    tunnel_dir = os.path.join(base_dir, tunnel_id)

    params = load_parameters(tunnel_id, base_dir)
    if override_params:
        params.update(override_params)

    if segments_file is None:
        segments_file = os.path.join(tunnel_dir, "all_segments.csv")
    elif not os.path.isabs(segments_file):
        segments_file = os.path.join(tunnel_dir, segments_file)
    if not os.path.exists(segments_file):
        raise FileNotFoundError(f"Segments file not found: {segments_file}")

    segments_df = pd.read_csv(segments_file)
    if "ring" in segments_df.columns and "Ring" not in segments_df.columns:
        segments_df = segments_df.rename(columns={"ring": "Ring"})
    if "segment_name" in segments_df.columns and "Block" not in segments_df.columns:
        segments_df = segments_df.rename(columns={"segment_name": "Block"})

    depth_path = os.path.join(tunnel_dir, "depth_map.png")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    img = cv2.imread(depth_path)
    height, width = img.shape[:2]

    with open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), "rb") as f:
        pixel_to_point = pickle.load(f)

    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    denoised_path = os.path.join(tunnel_dir, "denoised.csv")
    df = pd.read_csv(enhanced_path if os.path.exists(enhanced_path) else denoised_path)
    if "pred" not in df.columns:
        df["pred"] = 0
    else:
        df["pred"] = np.where(
            np.isin(df["pred"].values, [0, PRED_SURFACE]),
            df["pred"].values, 0,
        )

    unique_blocks = set(segments_df["Block"].unique()) - {"K"}
    segment_count = 1 + len(unique_blocks)
    block_to_label = compute_block_to_label_map(segment_count)

    ring_count = int(open(os.path.join(tunnel_dir, "ring_count.txt"), "r").read())
    ring_half_width = params.get("ring_half_width", DEFAULTS["ring_half_width"])
    if ring_half_width is None:
        ring_half_width = width / ring_count / 2.0

    boundaries_path = os.path.join(tunnel_dir, "boundaries_per_ring.json")
    boundaries_per_ring = None
    if os.path.exists(boundaries_path):
        with open(boundaries_path, "r") as f:
            boundaries_per_ring = json.load(f)
    slot_inset_y = params.get("slot_inset_y", DEFAULTS["slot_inset_y"])

    if boundaries_per_ring is not None:
        method = "boundary"
        label_map, ring_map = build_boundary_label_map(
            segments_df, height, width, block_to_label,
            ring_half_width, boundaries_per_ring,
            slot_inset_y=slot_inset_y,
        )
    else:
        method = "adaptive_cap"
        k_cap = params.get("k_cap", DEFAULTS["k_cap"])
        ab_cap = params.get("ab_cap", DEFAULTS["ab_cap"])
        label_map, ring_map = build_voronoi_label_map(
            segments_df, height, width, block_to_label,
            ring_half_width, k_cap, ab_cap,
        )

    fix_ring = np.where(
        (ring_map >= 1) & (ring_map <= (ring_count - 1)),
        ring_count - ring_map,
        ring_map,
    )

    updated_df = project_back_to_point_cloud(label_map, fix_ring, pixel_to_point, df)

    r_surface_min = params.get("r_surface_min", DEFAULTS["r_surface_min"])
    r_surface_min_per_ring = params.get("r_surface_min_per_ring", None)
    if ("r" in updated_df.columns and
            (r_surface_min is not None or (r_surface_min_per_ring is not None and isinstance(r_surface_min_per_ring, dict)))):
        pred_vals = updated_df["pred"].values
        r_vals = updated_df["r"].values
        pred_ring_vals = updated_df["pred_ring"].values
        block_mask = pred_vals > 0
        if r_surface_min_per_ring:
            fallback = r_surface_min if r_surface_min is not None else 0.0
            thresh = np.full(len(updated_df), fallback, dtype=np.float64)
            for ring_key, t in r_surface_min_per_ring.items():
                thresh[pred_ring_vals == int(ring_key)] = float(t)
            reclass = block_mask & (r_vals < thresh)
        else:
            reclass = block_mask & (r_vals < r_surface_min)
        n_reclass = int(reclass.sum())
        updated_df.loc[reclass, "pred"] = 0
        updated_df.loc[reclass, "pred_ring"] = -1
        if n_reclass > 0:
            label = "per_ring" if r_surface_min_per_ring else str(r_surface_min)
            print(f"  Radial filter (r_surface_min={label}): {n_reclass:,} points → background")

    out_csv = os.path.join(tunnel_dir, "final.csv")
    updated_df.to_csv(out_csv, index=False)

    print(f"Segmentation complete: {tunnel_id}")
    print(f"  Segments: {len(segments_df)}, Points: {len(updated_df)}")
    print(f"  Method: {method}, ring_half_width={ring_half_width:.1f}")
    if slot_inset_y != 0:
        print(f"  slot_inset_y: {slot_inset_y}")
    if r_surface_min is not None:
        print(f"  r_surface_min: {r_surface_min}")
    print(f"  Output: {out_csv}")

    return {"df": updated_df, "label_map": label_map}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geometric segmentation for irregular tunnels"
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument(
        "--segments-file", default=None,
        help="Segments CSV (default: all_segments.csv)",
    )
    args = parser.parse_args()
    run_segmentation(args.tunnel_id, base_dir=args.data_dir, segments_file=args.segments_file)

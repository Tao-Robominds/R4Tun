"""
Complex Staggered Geometric Segmentation (Stage 3 — no SAM)

Pure geometric pixel assignment from segment centres (all_segments.csv).
Uses per-block-type bounding boxes with tunable half-heights and half-width.
Y-axis wraps (cylindrical projection). Overlaps resolved by nearest centre.
No GPU, no SAM — fast evaluation for BO.

Pipeline:
    1_preprocessing.py → depth_map.png, enhanced.csv, pixel_to_point.pkl
    2_detection.py or GT → all_segments.csv (Ring, Block, X, Y in pixels)
    3_geometric.py → final.csv (segmented point cloud)

Tunable parameters (17):
    K_half_height, B1_half_height, B2_half_height,
    A1_half_height, A2_half_height, A3_half_height, A4_half_height,
    K_centre_offset, B1_centre_offset, B2_centre_offset,
    A1_centre_offset, A2_centre_offset, A3_centre_offset, A4_centre_offset,
    segment_half_width, shrink_x, shrink_y.
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd

from importlib.util import spec_from_file_location, module_from_spec
_sam_path = os.path.join(os.path.dirname(__file__), "3_sam.py")
_spec = spec_from_file_location("_sam", _sam_path)
_sam = module_from_spec(_spec)
_spec.loader.exec_module(_sam)
project_back_to_point_cloud = _sam.project_back_to_point_cloud
compute_block_to_label_map = _sam.compute_block_to_label_map


SEGMENT_COUNT = 7

DEFAULTS = {
    "K_half_height": 153,
    "B1_half_height": 418,
    "B2_half_height": 377,
    "A1_half_height": 351,
    "A2_half_height": 371,
    "A3_half_height": 380,
    "A4_half_height": 377,
    "K_centre_offset": -3,
    "B1_centre_offset": -31,
    "B2_centre_offset": 14,
    "A1_centre_offset": 1,
    "A2_centre_offset": -26,
    "A3_centre_offset": -22,
    "A4_centre_offset": -17,
    "segment_half_width": 197,
    "shrink_x": 4,
    "shrink_y": 2,
}


def load_parameters(tunnel_id: str, base_dir: str = "data") -> dict:
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, "parameters", tunnel_id, "parameters_geometric.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    path_sample = os.path.join(script_dir, "parameters", "sample", "parameters_geometric.json")
    if os.path.exists(path_sample):
        with open(path_sample, "r") as f:
            return json.load(f)
    return {}


def get_param(params: dict, key: str):
    return params[key] if key in params else DEFAULTS[key]


def build_geometric_label_map(
    segments_df: pd.DataFrame,
    height: int,
    width: int,
    params: dict,
    block_to_label: dict,
) -> tuple:
    """
    Build label_map and ring_map from segment centres + per-block-type half-heights.
    Y-axis wraps (cylindrical projection). Overlaps resolved by nearest centre.
    """
    half_heights = {
        "K": get_param(params, "K_half_height"),
        "B1": get_param(params, "B1_half_height"),
        "B2": get_param(params, "B2_half_height"),
        "A1": get_param(params, "A1_half_height"),
        "A2": get_param(params, "A2_half_height"),
        "A3": get_param(params, "A3_half_height"),
        "A4": get_param(params, "A4_half_height"),
    }
    centre_offsets = {
        "K": get_param(params, "K_centre_offset"),
        "B1": get_param(params, "B1_centre_offset"),
        "B2": get_param(params, "B2_centre_offset"),
        "A1": get_param(params, "A1_centre_offset"),
        "A2": get_param(params, "A2_centre_offset"),
        "A3": get_param(params, "A3_centre_offset"),
        "A4": get_param(params, "A4_centre_offset"),
    }
    half_w = max(0.0, float(get_param(params, "segment_half_width")) - get_param(params, "shrink_x"))
    shrink_y = get_param(params, "shrink_y")

    label_map = np.zeros((height, width), dtype=np.int32)
    ring_map = np.full((height, width), -1, dtype=np.int32)
    best_dist_sq = np.full((height, width), np.inf, dtype=np.float64)

    for _, row in segments_df.iterrows():
        ring = int(row["Ring"])
        block = row["Block"]
        cx = float(row["X"])
        cy = float(row["Y"])
        label_id = block_to_label.get(block, 0)
        if label_id == 0:
            continue

        half_y = max(0.0, float(half_heights.get(block, 380)) - shrink_y)
        cy_shifted = cy + float(centre_offsets.get(block, 0))

        x_lo = max(0, int(np.round(cx - half_w)))
        x_hi = min(width - 1, int(np.round(cx + half_w)))
        if x_lo > x_hi:
            continue

        y_lo_raw = int(np.round(cy_shifted - half_y))
        y_hi_raw = int(np.round(cy_shifted + half_y))

        px_int = np.arange(x_lo, x_hi + 1, dtype=np.intp)
        dx_arr = px_int.astype(np.float64) - cx

        if y_lo_raw >= 0 and y_hi_raw < height:
            py_int = np.arange(y_lo_raw, y_hi_raw + 1, dtype=np.intp)
        else:
            y_indices = np.arange(y_lo_raw, y_hi_raw + 1) % height
            py_int = y_indices.astype(np.intp)

        dy_raw = py_int.astype(np.float64) - cy
        dy_arr = np.where(dy_raw > height / 2, dy_raw - height, dy_raw)
        dy_arr = np.where(dy_arr < -height / 2, dy_arr + height, dy_arr)

        dy_sq = dy_arr ** 2
        dx_sq = dx_arr ** 2

        py_flat = np.repeat(py_int, len(px_int))
        px_flat = np.tile(px_int, len(py_int))
        dist_sq = np.repeat(dy_sq, len(px_int)) + np.tile(dx_sq, len(py_int))

        better = dist_sq < best_dist_sq[py_flat, px_flat]
        if np.any(better):
            idx_b = np.where(better)[0]
            best_dist_sq[py_flat[idx_b], px_flat[idx_b]] = dist_sq[idx_b]
            label_map[py_flat[idx_b], px_flat[idx_b]] = label_id
            ring_map[py_flat[idx_b], px_flat[idx_b]] = ring

    return label_map, ring_map


def run_geometric(
    tunnel_id: str,
    base_dir: str = "data",
    segments_file: str = None,
    override_params: dict = None,
) -> pd.DataFrame:
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
    import cv2
    img = cv2.imread(depth_path)
    height, width = img.shape[:2]

    pixel_to_point_path = os.path.join(tunnel_dir, "pixel_to_point.pkl")
    with open(pixel_to_point_path, "rb") as f:
        pixel_to_point = pickle.load(f)

    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    denoised_path = os.path.join(tunnel_dir, "denoised.csv")
    if os.path.exists(enhanced_path):
        df = pd.read_csv(enhanced_path)
    else:
        df = pd.read_csv(denoised_path)

    if "pred" not in df.columns:
        df["pred"] = 0
    else:
        df["pred"] = np.where(np.isin(df["pred"].values, [0, 7]), df["pred"].values, 0)

    block_to_label = compute_block_to_label_map(SEGMENT_COUNT)
    label_map, ring_map = build_geometric_label_map(
        segments_df, height, width, params, block_to_label,
    )

    ring_count = int(open(os.path.join(tunnel_dir, "ring_count.txt"), "r").read())
    fix_ring = np.where(
        (ring_map >= 1) & (ring_map <= (ring_count - 1)),
        ring_count - ring_map,
        ring_map,
    )

    updated_df = project_back_to_point_cloud(label_map, fix_ring, pixel_to_point, df)
    out_csv = os.path.join(tunnel_dir, "final.csv")
    updated_df.to_csv(out_csv, index=False)

    if "segment" in updated_df.columns:
        only_path = os.path.join(tunnel_dir, "only_label.csv")
        pd.DataFrame({
            "gt_labels": updated_df["segment"],
            "gt_rings": updated_df["ring"],
            "pred_labels": updated_df["pred"],
            "pred_rings": updated_df["pred_ring"],
        }).to_csv(only_path, index=False)

    return updated_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geometric segmentation (no SAM)")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument(
        "--segments-file",
        default=None,
        help="Segments CSV (default: <data_dir>/<tunnel_id>/all_segments.csv)",
    )
    args = parser.parse_args()
    run_geometric(args.tunnel_id, base_dir=args.data_dir, segments_file=args.segments_file)

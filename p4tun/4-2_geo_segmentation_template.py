"""
Template-shape geometric segmentation for complex staggered tunnels (4-1, 5-1).

Uses 4-vertex polygon templates (trapezoids for K, B1, B2; rectangle for A-blocks)
from the same logic as p4tun/4-2_sam.py generate_template_mask(). No SAM, no GPU.
Segment centres from CSV (e.g. all_segments_gt.csv to prove theory, then all_segments.csv).
Overlaps resolved by nearest centre. Y-axis wraps (cylindrical).

Usage:
  # Prove theory with GT centres
  python 4-2_geo_segmentation_template.py 4-1 --segments-file all_segments_gt.csv
  # Production with detected centres
  python 4-2_geo_segmentation_template.py 4-1 --segments-file all_segments.csv
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2
from matplotlib.path import Path

# Load project_back_to_point_cloud and compute_block_to_label_map from agents
_agents_sam = os.path.join(
    os.path.dirname(__file__), "..", "agents", "irregular", "3_segmentation", "3_sam.py"
)
import importlib.util
_spec = importlib.util.spec_from_file_location("_sam", _agents_sam)
_sam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sam)
project_back_to_point_cloud = _sam.project_back_to_point_cloud
compute_block_to_label_map = _sam.compute_block_to_label_map

SEGMENT_COUNT = 7

# Default template dimensions in pixels (from 4-2_sam.py at resolution 0.005: 1 px = 5 mm)
# K: trapezoid; B1/B2: trapezoid; A: rectangle
DEFAULTS = {
    "K_half_width": 125,
    "K_half_height_pos": 124,
    "K_half_height_neg": 92,
    "K_centre_offset": 0,
    "B1_half_width": 125,
    "B1_half_height_top": 324,
    "B1_half_height_bottom_pos": 308,
    "B1_half_height_bottom_neg": 340,
    "B1_centre_offset": 0,
    "B2_half_width": 125,
    "B2_half_height_top_pos": 308,
    "B2_half_height_top_neg": 340,
    "B2_half_height_bottom": 324,
    "B2_centre_offset": 0,
    "segment_half_width": 125,
    "A1_half_height": 324,
    "A2_half_height": 324,
    "A3_half_height": 324,
    "A4_half_height": 324,
    "A1_centre_offset": 0,
    "A2_centre_offset": 0,
    "A3_centre_offset": 0,
    "A4_centre_offset": 0,
    "shrink_x": 0,
    "shrink_y": 0,
}


def get_param(params: dict, key: str):
    return params.get(key, DEFAULTS.get(key))


def load_parameters(tunnel_id: str, base_dir: str, script_dir: str) -> dict:
    """Load from p4tun/parameters/<tunnel_id>/parameters_geometric_template.json or parameters_geometric.json."""
    for name in ("parameters_geometric_template.json", "parameters_geometric.json"):
        path = os.path.join(script_dir, "parameters", tunnel_id, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    return {}


def get_trapezoid_vertices_relative(block: str, params: dict) -> np.ndarray:
    """
    Return 4 vertices (dx, dy) relative to segment centre for the block's template.
    Order: counterclockwise so Path.contains_points works (same as 4-2_sam: left-top, left-bottom, right-bottom, right-top).
    """
    if block == "K":
        w = max(0, get_param(params, "K_half_width") - get_param(params, "shrink_x"))
        hp = get_param(params, "K_half_height_pos") - get_param(params, "shrink_y")
        hn = get_param(params, "K_half_height_neg") - get_param(params, "shrink_y")
        # left side y in [-hp, +hp], right side y in [-hn, +hn]
        return np.array([[-w, -hp], [-w, hp], [w, hn], [w, -hn]], dtype=np.float64)
    elif block == "B1":
        w = max(0, get_param(params, "B1_half_width") - get_param(params, "shrink_x"))
        ht = get_param(params, "B1_half_height_top") - get_param(params, "shrink_y")
        hbp = get_param(params, "B1_half_height_bottom_pos") - get_param(params, "shrink_y")
        hbn = get_param(params, "B1_half_height_bottom_neg") - get_param(params, "shrink_y")
        # top horizontal at -ht, bottom slanted: left +hbp, right +hbn
        return np.array([[-w, -ht], [-w, hbp], [w, hbn], [w, -ht]], dtype=np.float64)
    elif block == "B2":
        w = max(0, get_param(params, "B2_half_width") - get_param(params, "shrink_x"))
        htp = get_param(params, "B2_half_height_top_pos") - get_param(params, "shrink_y")
        htn = get_param(params, "B2_half_height_top_neg") - get_param(params, "shrink_y")
        hb = get_param(params, "B2_half_height_bottom") - get_param(params, "shrink_y")
        return np.array([[-w, -htp], [-w, hb], [w, hb], [w, -htn]], dtype=np.float64)
    else:
        # A1-A4: rectangle
        w = max(0, get_param(params, "segment_half_width") - get_param(params, "shrink_x"))
        h = get_param(params, f"{block}_half_height") - get_param(params, "shrink_y")
        return np.array([[-w, -h], [-w, h], [w, h], [w, -h]], dtype=np.float64)


def build_template_label_map(
    segments_df: pd.DataFrame,
    height: int,
    width: int,
    params: dict,
    block_to_label: dict,
) -> tuple:
    """
    Build label_map and ring_map from segment centres and template polygons.
    Y-axis wraps. Overlaps resolved by nearest centre.
    """
    label_map = np.zeros((height, width), dtype=np.int32)
    ring_map = np.full((height, width), -1, dtype=np.int32)
    best_dist_sq = np.full((height, width), np.inf, dtype=np.float64)

    centre_offsets = {
        "K": get_param(params, "K_centre_offset"),
        "B1": get_param(params, "B1_centre_offset"),
        "B2": get_param(params, "B2_centre_offset"),
        "A1": get_param(params, "A1_centre_offset"),
        "A2": get_param(params, "A2_centre_offset"),
        "A3": get_param(params, "A3_centre_offset"),
        "A4": get_param(params, "A4_centre_offset"),
    }

    for _, row in segments_df.iterrows():
        ring = int(row["Ring"])
        block = row["Block"]
        cx = float(row["X"])
        cy = float(row["Y"])
        cy_shifted = cy + centre_offsets.get(block, 0)
        label_id = block_to_label.get(block, 0)
        if label_id == 0:
            continue

        vertices_rel = get_trapezoid_vertices_relative(block, params)
        path = Path(vertices_rel)
        y_extent = float(np.max(np.abs(vertices_rel[:, 1])))
        x_extent = float(np.max(np.abs(vertices_rel[:, 0])))
        x_lo = max(0, int(np.floor(cx - x_extent)))
        x_hi = min(width - 1, int(np.ceil(cx + x_extent)))
        if x_lo > x_hi:
            continue

        y_centre = cy_shifted
        y_lo = int(np.floor(y_centre - y_extent))
        y_hi = int(np.ceil(y_centre + y_extent))
        px_arr = np.arange(x_lo, x_hi + 1)
        py_raw_arr = np.arange(y_lo, y_hi + 1)
        px_grid, py_raw_grid = np.meshgrid(px_arr, py_raw_arr)
        py_wrapped = py_raw_grid % height
        py_wrapped = np.where(py_wrapped < 0, py_wrapped + height, py_wrapped)
        dy_raw = py_raw_grid.astype(np.float64) - y_centre
        dy = np.where(dy_raw > height / 2, dy_raw - height, dy_raw)
        dy = np.where(dy < -height / 2, dy + height, dy)
        dx = px_grid.astype(np.float64) - cx
        in_bounds = (py_wrapped >= 0) & (py_wrapped < height) & (px_grid >= 0) & (px_grid < width)
        points_rel = np.column_stack([dx[in_bounds], dy[in_bounds]])
        inside = path.contains_points(points_rel)
        py_valid = py_wrapped[in_bounds][inside]
        px_valid = px_grid[in_bounds][inside]
        dist_sq = points_rel[inside, 0] ** 2 + points_rel[inside, 1] ** 2
        better = dist_sq < best_dist_sq[py_valid, px_valid]
        if np.any(better):
            idx = np.where(better)[0]
            best_dist_sq[py_valid[idx], px_valid[idx]] = dist_sq[idx]
            label_map[py_valid[idx], px_valid[idx]] = label_id
            ring_map[py_valid[idx], px_valid[idx]] = ring

    return label_map, ring_map


def run_template_geometric(
    tunnel_id: str,
    base_dir: str = "data",
    segments_file: str = None,
    override_params: dict = None,
) -> dict:
    """
    Run template-shape geometric segmentation (no SAM).
    Returns dict with 'df', 'label_map'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tunnel_dir = os.path.join(base_dir, tunnel_id)

    params = load_parameters(tunnel_id, base_dir, script_dir)
    if override_params:
        params.update(override_params)
    for k, v in DEFAULTS.items():
        if k not in params:
            params[k] = v

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
    label_map, ring_map = build_template_label_map(
        segments_df, height, width, params, block_to_label
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

    return {"df": updated_df, "label_map": label_map}


def main():
    parser = argparse.ArgumentParser(
        description="Template-shape geometric segmentation (trapezoid K/B1/B2, rectangle A-blocks)"
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument(
        "--segments-file",
        default=None,
        help="Segments CSV: all_segments_gt.csv (prove theory) or all_segments.csv (default)",
    )
    args = parser.parse_args()
    result = run_template_geometric(
        args.tunnel_id,
        base_dir=args.data_dir,
        segments_file=args.segments_file,
    )
    print(f"Saved final.csv to {os.path.join(args.data_dir, args.tunnel_id)}")


if __name__ == "__main__":
    main()

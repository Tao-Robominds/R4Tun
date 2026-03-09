"""
Irregular Tunnel Geometric Segmentation

Template-shape pixel assignment from segment centres (all_segments.csv).
Trapezoid templates for K, B1, B2; rectangle for A-blocks.
Y-axis wraps (cylindrical projection). Overlaps resolved by nearest centre.
No SAM, no GPU, no GT dependency.

Pipeline:
    1_preprocessing.py → depth_map.png, enhanced.csv, pixel_to_point.pkl
    2_detection.py     → all_segments.csv (Ring, Block, X, Y in pixels)
    segmentation.py    → final.csv (segmented point cloud)

Tunable parameters (22):
    K_half_width, K_half_height_pos, K_half_height_neg, K_centre_offset,
    B1_half_width, B1_half_height_top, B1_half_height_bottom_pos/neg, B1_centre_offset,
    B2_half_width, B2_half_height_top_pos/neg, B2_half_height_bottom, B2_centre_offset,
    segment_half_width, A1-A4_half_height, A1-A4_centre_offset,
    shrink_x, shrink_y.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2
from matplotlib.path import Path


SURFACE_PRED = 7

DEFAULT_EXPANSION_BLOCKS_7 = ['B1', 'A1', 'A2', 'A3', 'A4', 'B2']
DEFAULT_EXPANSION_BLOCKS_6 = ['B1', 'A1', 'A2', 'A3', 'B2']

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


# =============================================================================
# Parameter Loading
# =============================================================================

def get_param(params: dict, key: str):
    return params.get(key, DEFAULTS.get(key))


def load_parameters(tunnel_id: str, base_dir: str = "data") -> dict:
    """Load from parameters/<tunnel_id>/parameters_geometric_template.json or parameters_geometric.json."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("parameters_geometric_template.json", "parameters_geometric.json"):
        path = os.path.join(script_dir, "parameters", tunnel_id, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    return {}


# =============================================================================
# Block Label Mapping
# =============================================================================

def compute_block_to_label_map(segment_per_ring: int) -> dict:
    """Block name → numeric label mapping."""
    if segment_per_ring == 7:
        return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'A4': 6, 'B2': 7}
    return {'K': 1, 'B1': 2, 'A1': 3, 'A2': 4, 'A3': 5, 'B2': 6}


# =============================================================================
# Template Shapes
# =============================================================================

def get_trapezoid_vertices_relative(block: str, params: dict) -> np.ndarray:
    """4 vertices (dx, dy) relative to segment centre for the block's template.

    Order: counterclockwise (left-top, left-bottom, right-bottom, right-top).
    K/B1/B2 are trapezoids, A blocks are rectangles.
    """
    if block == "K":
        w = max(0, get_param(params, "K_half_width") - get_param(params, "shrink_x"))
        hp = get_param(params, "K_half_height_pos") - get_param(params, "shrink_y")
        hn = get_param(params, "K_half_height_neg") - get_param(params, "shrink_y")
        return np.array([[-w, -hp], [-w, hp], [w, hn], [w, -hn]], dtype=np.float64)
    elif block == "B1":
        w = max(0, get_param(params, "B1_half_width") - get_param(params, "shrink_x"))
        ht = get_param(params, "B1_half_height_top") - get_param(params, "shrink_y")
        hbp = get_param(params, "B1_half_height_bottom_pos") - get_param(params, "shrink_y")
        hbn = get_param(params, "B1_half_height_bottom_neg") - get_param(params, "shrink_y")
        return np.array([[-w, -ht], [-w, hbp], [w, hbn], [w, -ht]], dtype=np.float64)
    elif block == "B2":
        w = max(0, get_param(params, "B2_half_width") - get_param(params, "shrink_x"))
        htp = get_param(params, "B2_half_height_top_pos") - get_param(params, "shrink_y")
        htn = get_param(params, "B2_half_height_top_neg") - get_param(params, "shrink_y")
        hb = get_param(params, "B2_half_height_bottom") - get_param(params, "shrink_y")
        return np.array([[-w, -htp], [-w, hb], [w, hb], [w, -htn]], dtype=np.float64)
    else:
        w = max(0, get_param(params, "segment_half_width") - get_param(params, "shrink_x"))
        h = get_param(params, f"{block}_half_height") - get_param(params, "shrink_y")
        return np.array([[-w, -h], [-w, h], [w, h], [w, -h]], dtype=np.float64)


# =============================================================================
# Label Map Construction
# =============================================================================

def build_template_label_map(
    segments_df: pd.DataFrame,
    height: int,
    width: int,
    params: dict,
    block_to_label: dict,
) -> tuple:
    """Build label_map and ring_map from segment centres and template polygons.

    Y-axis wraps (cylindrical). Overlaps resolved by nearest centre.
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


# =============================================================================
# Point Cloud Projection
# =============================================================================

def project_back_to_point_cloud(segmented_map, instance_map, pixel_to_point, df):
    """Project 2D label/ring maps back to 3D point cloud.

    Updates pred for points with pred in {0, SURFACE_PRED} (background + surface).
    """
    df_copy = df.copy()
    pred = df_copy['pred'].values
    pred_ring = np.full(len(df_copy), -1, dtype=int)

    pixel_to_point_df = pd.DataFrame(pixel_to_point)
    y = pixel_to_point_df['pixel_y'].values
    x = pixel_to_point_df['pixel_x'].values
    point_indices = pixel_to_point_df['index'].values

    img_height, img_width = segmented_map.shape

    valid_point_mask = np.isin(point_indices, df_copy.index.values)
    valid_update_mask = np.isin(pred[point_indices[valid_point_mask]], [0, SURFACE_PRED])

    y_valid = y[valid_point_mask][valid_update_mask]
    x_valid = x[valid_point_mask][valid_update_mask]

    bounds_mask = (y_valid >= 0) & (y_valid < img_height) & (x_valid >= 0) & (x_valid < img_width)

    final_point_indices = point_indices[valid_point_mask][valid_update_mask][bounds_mask]
    final_y = y_valid[bounds_mask]
    final_x = x_valid[bounds_mask]

    pred[final_point_indices] = segmented_map[final_y, final_x]
    pred_ring[final_point_indices] = instance_map[final_y, final_x]

    df_copy['pred'] = pred
    df_copy['pred_ring'] = pred_ring

    return df_copy


# =============================================================================
# Main Pipeline
# =============================================================================

def run_segmentation(
    tunnel_id: str,
    base_dir: str = "data",
    segments_file: str = None,
    override_params: dict = None,
) -> dict:
    """Run geometric segmentation (no SAM, no GT).

    Returns dict with 'df' (final DataFrame) and 'label_map'.
    """
    tunnel_dir = os.path.join(base_dir, tunnel_id)

    params = load_parameters(tunnel_id, base_dir)
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
        df["pred"] = np.where(
            np.isin(df["pred"].values, [0, SURFACE_PRED]),
            df["pred"].values, 0,
        )

    unique_blocks = set(segments_df["Block"].unique()) - {"K"}
    segment_count = 1 + len(unique_blocks)
    block_to_label = compute_block_to_label_map(segment_count)
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

    print(f"Segmentation complete: {tunnel_id}")
    print(f"  Segments: {len(segments_df)}, Points: {len(updated_df)}")
    print(f"  Output: {out_csv}")

    return {"df": updated_df, "label_map": label_map}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geometric segmentation for irregular tunnels (no SAM, no GT)"
    )
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument(
        "--segments-file", default=None,
        help="Segments CSV (default: all_segments.csv)",
    )
    args = parser.parse_args()
    run_segmentation(args.tunnel_id, base_dir=args.data_dir, segments_file=args.segments_file)

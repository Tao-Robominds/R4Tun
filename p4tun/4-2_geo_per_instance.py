"""
Per-instance geometric segmentation for complex staggered tunnels (4-1, 5-1).

Uses GT-derived per-instance bounding boxes (one asymmetric rectangle per block instance)
instead of per-type templates. Proves the mIoU ceiling (~0.80) vs per-type sizing (~0.60).

Pipeline:
  1. Load enhanced.csv + pixel_to_point.pkl; build per (ring, segment) pixel sets from GT.
  2. Compute per-instance extents: half_w, dy_neg, dy_pos from pixel extents; centres from all_segments_gt.csv.
  3. Build label map: asymmetric rectangles [cx - half_w + sx, cx + half_w - sx] x [cy - dy_neg + sy, cy + dy_pos - sy],
     overlap resolved by nearest centre (wrap-aware).
  4. Project back to point cloud, save final.csv.

Usage:
  python 4-2_geo_per_instance.py 4-1 --segments-file all_segments_gt.csv [--shrink-x 4] [--shrink-y 2]
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2

# Reuse from agents/irregular/3_segmentation/3_sam.py
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
# GT segment number -> Block name (7-seg complex staggered)
SEGMENT_TO_BLOCK = {1: "K", 2: "B1", 3: "B2", 4: "A1", 5: "A2", 6: "A3", 7: "A4"}


def compute_per_instance_bboxes(tunnel_dir: str, height: int, width: int) -> dict:
    """
    From enhanced.csv (segment, ring) and pixel_to_point.pkl, compute per (ring, block) instance:
    cx, cy (from all_segments_gt.csv), half_w, dy_neg, dy_pos (from GT pixel extents).

    Returns dict keyed by (ring, block) with values dict(cx, cy, half_w, dy_neg, dy_pos).
    Ring and block use the same types as in all_segments_gt (e.g. ring 119-125, block 'K', 'B1', ...).
    """
    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    p2p_path = os.path.join(tunnel_dir, "pixel_to_point.pkl")
    gt_segments_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    if not os.path.exists(enhanced_path) or not os.path.exists(p2p_path) or not os.path.exists(gt_segments_path):
        raise FileNotFoundError(f"Need enhanced.csv, pixel_to_point.pkl, all_segments_gt.csv in {tunnel_dir}")

    df = pd.read_csv(enhanced_path)
    with open(p2p_path, "rb") as f:
        pixel_to_point = pickle.load(f)
    gt_segments = pd.read_csv(gt_segments_path)
    if "ring" in gt_segments.columns and "Ring" not in gt_segments.columns:
        gt_segments = gt_segments.rename(columns={"ring": "Ring"})
    if "segment_name" in gt_segments.columns and "Block" not in gt_segments.columns:
        gt_segments = gt_segments.rename(columns={"segment_name": "Block"})

    # Build (ring, segment) -> list of (pixel_x, pixel_y)
    # pixel_to_point: list of {pixel_x, pixel_y, index}; index into df
    segment_col = df["segment"].values
    ring_col = df["ring"].values
    instance_pixels = {}  # (ring, segment) -> list of (px, py)
    for rec in pixel_to_point:
        idx = rec["index"]
        if idx >= len(df):
            continue
        seg = segment_col[idx]
        r = ring_col[idx]
        if np.isnan(seg) or np.isnan(r) or seg < 1 or seg > 7:
            continue
        seg_int = int(seg)
        r_int = int(r)
        key = (r_int, seg_int)
        if key not in instance_pixels:
            instance_pixels[key] = []
        instance_pixels[key].append((int(rec["pixel_x"]), int(rec["pixel_y"])))

    # Centres from GT segments (Ring, Block) -> (X, Y)
    gt_centres = {}
    for _, row in gt_segments.iterrows():
        r = int(row["Ring"])
        block = row["Block"]
        gt_centres[(r, block)] = (float(row["X"]), float(row["Y"]))

    # Per-instance bbox params: (ring, block) -> {cx, cy, half_w, dy_neg, dy_pos}
    result = {}
    for (ring, seg_num), pixels in instance_pixels.items():
        block = SEGMENT_TO_BLOCK.get(seg_num)
        if block is None:
            continue
        centre = gt_centres.get((ring, block))
        if centre is None:
            continue
        cx, cy = centre
        px_arr = np.array([p[0] for p in pixels])
        py_arr = np.array([p[1] for p in pixels])
        # X: half width from centre
        half_w = float(np.max(np.abs(px_arr - cx)))
        # Y: wrap-aware signed offset from centre
        dy_raw = py_arr.astype(np.float64) - cy
        dy = np.where(dy_raw > height / 2, dy_raw - height, dy_raw)
        dy = np.where(dy < -height / 2, dy + height, dy)
        dy_neg = float(max(0.0, -np.min(dy)))
        dy_pos = float(max(0.0, np.max(dy)))
        result[(ring, block)] = {"cx": cx, "cy": cy, "half_w": half_w, "dy_neg": dy_neg, "dy_pos": dy_pos}
    return result


def build_per_instance_label_map(
    segments_df: pd.DataFrame,
    instance_bboxes: dict,
    height: int,
    width: int,
    shrink_x: float,
    shrink_y: float,
    block_to_label: dict,
    gt_ring_order: list = None,
) -> tuple:
    """
    Build label_map and ring_map from per-instance boxes.
    Rectangle for each instance: [cx - half_w + sx, cx + half_w - sx] x [cy - dy_neg + sy, cy + dy_pos - sy].
    Overlaps resolved by nearest centre (wrap-aware). Y-axis wraps.
    If gt_ring_order is given (sorted list of GT ring IDs), segment ring 0..n-1 is mapped to gt_ring_order[0]..gt_ring_order[n-1] for bbox lookup and for ring_map output.
    """
    label_map = np.zeros((height, width), dtype=np.int32)
    ring_map = np.full((height, width), -1, dtype=np.int32)
    best_dist_sq = np.full((height, width), np.inf, dtype=np.float64)

    rings_in_df = segments_df["Ring"].astype(int)
    use_0based = rings_in_df.max() < 100 and gt_ring_order is not None and len(gt_ring_order) > 0

    for _, row in segments_df.iterrows():
        ring = int(row["Ring"])
        block = row["Block"]
        # Map 0-based ring (0..6) to GT ring (e.g. 119..125) for bbox lookup and output
        ring_for_key = gt_ring_order[ring] if use_0based and ring < len(gt_ring_order) else ring
        key = (ring_for_key, block)
        bbox = instance_bboxes.get(key)
        if bbox is None:
            continue
        label_id = block_to_label.get(block, 0)
        if label_id == 0:
            continue

        # Use centre from current segments file (GT or detected)
        cx = float(row["X"])
        cy = float(row["Y"])
        half_w = max(0.0, bbox["half_w"] - shrink_x)
        dy_neg = max(0.0, bbox["dy_neg"] - shrink_y)
        dy_pos = max(0.0, bbox["dy_pos"] - shrink_y)

        x_lo = max(0, int(np.round(cx - half_w)))
        x_hi = min(width - 1, int(np.round(cx + half_w)))
        if x_lo > x_hi:
            continue

        y_lo_raw = int(np.round(cy - dy_neg))
        y_hi_raw = int(np.round(cy + dy_pos))

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
            ring_map[py_flat[idx_b], px_flat[idx_b]] = ring_for_key

    return label_map, ring_map


def run_per_instance_geometric(
    tunnel_id: str,
    base_dir: str = "data",
    segments_file: str = None,
    shrink_x: float = 4.0,
    shrink_y: float = 2.0,
) -> dict:
    """
    Run per-instance geometric segmentation (no SAM).
    Returns dict with 'df', 'label_map', 'ring_map'.
    """
    tunnel_dir = os.path.join(base_dir, tunnel_id)

    if segments_file is None:
        segments_file = os.path.join(tunnel_dir, "all_segments_gt.csv")
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

    instance_bboxes = compute_per_instance_bboxes(tunnel_dir, height, width)
    if len(instance_bboxes) == 0:
        raise RuntimeError("No per-instance bboxes computed; check enhanced.csv and pixel_to_point.pkl")

    gt_ring_order = sorted(set(r for r, _ in instance_bboxes.keys()))
    block_to_label = compute_block_to_label_map(SEGMENT_COUNT)
    label_map, ring_map = build_per_instance_label_map(
        segments_df, instance_bboxes, height, width,
        shrink_x, shrink_y, block_to_label,
        gt_ring_order=gt_ring_order,
    )

    with open(os.path.join(tunnel_dir, "pixel_to_point.pkl"), "rb") as f:
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

    ring_count = int(open(os.path.join(tunnel_dir, "ring_count.txt"), "r").read())
    fix_ring = np.where(
        (ring_map >= 1) & (ring_map <= (ring_count - 1)),
        ring_count - ring_map,
        ring_map,
    )
    # fix_ring is applied to ring_map when projecting; we need to pass the same ring_map shape
    # project_back_to_point_cloud uses instance_map for pred_ring; we pass ring_map (raw ring ids).
    # The pipeline in 3_geometric uses fix_ring for the ring mapping when writing. So we should apply
    # fix_ring when building the output: pred_ring in df = fix_ring[py, px] at each pixel.
    # project_back_to_point_cloud(segmented_map, instance_map, ...) writes pred_ring = instance_map[py, px].
    # Pass ring indices with same convention as geometric pipeline (flip middle rings if 0-based).
    instance_map = np.where(
        (ring_map >= 1) & (ring_map <= (ring_count - 1)),
        ring_count - ring_map,
        ring_map,
    ).astype(np.int32)
    updated_df = project_back_to_point_cloud(label_map, instance_map, pixel_to_point, df)

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

    return {"df": updated_df, "label_map": label_map, "ring_map": ring_map}


def evaluate_miou(
    tunnel_id: str,
    base_dir: str = "data",
    segments_file: str = None,
    shrink_x: float = 4.0,
    shrink_y: float = 2.0,
    segment_count: int = 7,
) -> float:
    """
    Run per-instance geometric segmentation then evaluate mIoU from final.csv.
    For BO: run segmentation with given segments_file and shrink, return mIoU.
    """
    run_per_instance_geometric(
        tunnel_id,
        base_dir=base_dir,
        segments_file=segments_file,
        shrink_x=shrink_x,
        shrink_y=shrink_y,
    )
    _here = os.path.dirname(os.path.abspath(__file__))
    import sys
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from evaluation import get_miou
    return get_miou(tunnel_id, base_dir=base_dir, segment_count=segment_count)


def main():
    parser = argparse.ArgumentParser(description="Per-instance geometric segmentation (GT-derived bboxes, no SAM)")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument(
        "--segments-file",
        default="all_segments_gt.csv",
        help="Segments CSV (default: all_segments_gt.csv)",
    )
    parser.add_argument("--shrink-x", type=float, default=4.0, help="X boundary shrink (px)")
    parser.add_argument("--shrink-y", type=float, default=2.0, help="Y boundary shrink (px)")
    args = parser.parse_args()
    result = run_per_instance_geometric(
        args.tunnel_id,
        base_dir=args.data_dir,
        segments_file=args.segments_file,
        shrink_x=args.shrink_x,
        shrink_y=args.shrink_y,
    )
    print(f"Per-instance geometric segmentation done. final.csv written to {os.path.join(args.data_dir, args.tunnel_id, 'final.csv')}")


if __name__ == "__main__":
    main()

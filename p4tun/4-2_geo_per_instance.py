"""
Per-instance geometric segmentation for complex staggered tunnels (4-1, 5-1).

**GT requirement:** Hull and bbox shapes are always derived from GT. The pipeline
needs enhanced.csv (with segment column), pixel_to_point.pkl, and all_segments_gt.csv
in the tunnel dir to compute per-instance shapes. Only the segment *centres* (X, Y)
can be taken from --segments-file: if you pass detected centres (e.g. all_segments.csv),
hulls/bboxes are still GT-derived and only placed at the detected positions.
For a fully GT-free pipeline on a new tunnel, use per-type geometric segmentation
(e.g. 4-2_geo_segmentation_template.py) instead.

Supports three shape modes:
  bbox  -- asymmetric rectangles (original), overlap resolved by nearest centre.
  hull  -- convex hulls from GT pixel sets, overlap resolved by nearest centre.
  mask  -- raw GT pixel masks (GT-only ceiling; requires GT centres).

Pipeline:
  1. Load enhanced.csv + pixel_to_point.pkl; build per (ring, segment) pixel sets from GT.
  2. Compute per-instance shapes (bbox or hull) from pixel extents; centres from all_segments_gt.csv.
  3. Build label map using centres from --segments-file (GT or detected); project back; save final.csv.

Usage:
  python 4-2_geo_per_instance.py 4-1 --segments-file all_segments_gt.csv --mode hull
  python 4-2_geo_per_instance.py 4-1 --segments-file all_segments.csv --mode hull   # detected centres, GT shapes
  python 4-2_geo_per_instance.py 4-1 --segments-file all_segments_gt.csv --mode bbox [--shrink-x 4] [--shrink-y 2]
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import ConvexHull as _ConvexHull

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

    # Ring ID mapping: enhanced may use physical IDs (e.g. 119-125), GT file may use 0-based (0-6)
    rings_from_instance = sorted(set(r for r, _ in instance_pixels.keys()))
    rings_gt = sorted(gt_segments["Ring"].unique())
    if len(rings_gt) == len(rings_from_instance) and set(rings_gt) != set(rings_from_instance):
        gt_ring_to_phys = {int(r): rings_from_instance[i] for i, r in enumerate(rings_gt)}
    else:
        gt_ring_to_phys = {int(r): int(r) for r in rings_gt}

    # Centres from GT segments (Ring, Block) -> (X, Y); use physical ring ID for lookup
    gt_centres = {}
    for _, row in gt_segments.iterrows():
        r_gt = int(row["Ring"])
        r_phys = gt_ring_to_phys.get(r_gt, r_gt)
        block = row["Block"]
        gt_centres[(r_phys, block)] = (float(row["X"]), float(row["Y"]))

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


def compute_per_instance_hulls(tunnel_dir: str, height: int, width: int) -> dict:
    """
    Compute convex hulls from GT pixel sets for each (ring, block) instance.

    Returns dict keyed by (physical_ring, block_name) with values:
      {"cx": float, "cy": float, "hull_rel": ndarray(N,2)}
    hull_rel stores (dx, dy) hull vertices relative to GT centre (wrap-aware).
    """
    enhanced_path = os.path.join(tunnel_dir, "enhanced.csv")
    p2p_path = os.path.join(tunnel_dir, "pixel_to_point.pkl")
    gt_segments_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    if not all(os.path.exists(p) for p in (enhanced_path, p2p_path, gt_segments_path)):
        raise FileNotFoundError(
            f"Need enhanced.csv, pixel_to_point.pkl, all_segments_gt.csv in {tunnel_dir}"
        )

    df = pd.read_csv(enhanced_path)
    with open(p2p_path, "rb") as f:
        pixel_to_point = pickle.load(f)
    gt_segments = pd.read_csv(gt_segments_path)
    if "ring" in gt_segments.columns and "Ring" not in gt_segments.columns:
        gt_segments = gt_segments.rename(columns={"ring": "Ring"})
    if "segment_name" in gt_segments.columns and "Block" not in gt_segments.columns:
        gt_segments = gt_segments.rename(columns={"segment_name": "Block"})

    segment_col = df["segment"].values
    ring_col = df["ring"].values
    instance_pixels = {}
    for rec in pixel_to_point:
        idx = rec["index"]
        if idx >= len(df):
            continue
        seg = segment_col[idx]
        r = ring_col[idx]
        if np.isnan(seg) or np.isnan(r) or seg < 1 or seg > 7:
            continue
        key = (int(r), int(seg))
        if key not in instance_pixels:
            instance_pixels[key] = []
        instance_pixels[key].append((int(rec["pixel_x"]), int(rec["pixel_y"])))

    rings_from_instance = sorted(set(r for r, _ in instance_pixels.keys()))
    rings_gt = sorted(gt_segments["Ring"].unique())
    if len(rings_gt) == len(rings_from_instance) and set(rings_gt) != set(rings_from_instance):
        gt_ring_to_phys = {int(r): rings_from_instance[i] for i, r in enumerate(rings_gt)}
    else:
        gt_ring_to_phys = {int(r): int(r) for r in rings_gt}

    gt_centres = {}
    for _, row in gt_segments.iterrows():
        r_gt = int(row["Ring"])
        r_phys = gt_ring_to_phys.get(r_gt, r_gt)
        block = row["Block"]
        gt_centres[(r_phys, block)] = (float(row["X"]), float(row["Y"]))

    result = {}
    for (ring, seg_num), pixels in instance_pixels.items():
        block = SEGMENT_TO_BLOCK.get(seg_num)
        if block is None:
            continue
        centre = gt_centres.get((ring, block))
        if centre is None:
            continue
        cx, cy = centre
        pts = np.array(pixels, dtype=np.float64)

        if len(pts) < 3:
            half_w = max(1.0, float(np.ptp(pts[:, 0])) / 2)
            half_h = max(1.0, float(np.ptp(pts[:, 1])) / 2)
            hull_rel = np.array(
                [[-half_w, -half_h], [half_w, -half_h],
                 [half_w, half_h], [-half_w, half_h]]
            )
            dx = pts[:, 0] - cx
            dy_raw = pts[:, 1] - cy
            dy = np.where(dy_raw > height / 2, dy_raw - height, dy_raw)
            dy = np.where(dy < -height / 2, dy + height, dy)
            result[(ring, block)] = {
                "cx": cx, "cy": cy, "hull_rel": hull_rel,
                "pixels_rel": np.column_stack([dx, dy]),
            }
            continue

        dx = pts[:, 0] - cx
        dy_raw = pts[:, 1] - cy
        dy = np.where(dy_raw > height / 2, dy_raw - height, dy_raw)
        dy = np.where(dy < -height / 2, dy + height, dy)
        rel_pts = np.column_stack([dx, dy])

        try:
            hull = _ConvexHull(rel_pts)
            hull_verts = rel_pts[hull.vertices]
        except Exception:
            x_min, y_min = rel_pts.min(axis=0)
            x_max, y_max = rel_pts.max(axis=0)
            hull_verts = np.array(
                [[x_min, y_min], [x_max, y_min],
                 [x_max, y_max], [x_min, y_max]]
            )

        result[(ring, block)] = {"cx": cx, "cy": cy, "hull_rel": hull_verts, "pixels_rel": rel_pts}

    return result


def _fill_hull_wrapped(mask: np.ndarray, verts: np.ndarray, height: int):
    """Rasterize a convex polygon onto *mask*, painting copies at Y, Y+H, Y-H to handle wrap."""
    for dy_shift in (0, height, -height):
        shifted = verts.copy()
        shifted[:, 1] += dy_shift
        if shifted[:, 1].max() < 0 or shifted[:, 1].min() >= height:
            continue
        pts = np.round(shifted).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 1)


def _detect_distorted_rings(instance_hulls: dict, height: int) -> set:
    """
    Identify rings where convex hulls are severely distorted (blocks overlap
    too much for hull+nearest-centre to work). Heuristic: if the median hull
    Y-range in a ring exceeds height/3, the ring is distorted.
    """
    ring_dy_ranges = {}
    for (ring, _block), data in instance_hulls.items():
        dy_range = float(data["hull_rel"][:, 1].max() - data["hull_rel"][:, 1].min())
        ring_dy_ranges.setdefault(ring, []).append(dy_range)
    distorted = set()
    for ring, ranges in ring_dy_ranges.items():
        median_range = float(np.median(ranges))
        if median_range > height / 3:
            distorted.add(ring)
    return distorted


def build_hull_label_map(
    segments_df: pd.DataFrame,
    instance_hulls: dict,
    height: int,
    width: int,
    shrink: float,
    block_to_label: dict,
    gt_ring_order: list = None,
    force_pixel_mask: bool = False,
) -> tuple:
    """
    Build label_map and ring_map from per-instance convex hulls.
    Pixels inside a hull are claimed; overlaps resolved by nearest centre
    (wrap-aware Euclidean distance).

    For severely distorted rings (median hull Y-range > height/3) or when
    force_pixel_mask=True, raw GT pixel coordinates are painted instead of
    convex hulls.
    """
    best_dist_sq = np.full((height, width), np.inf, dtype=np.float64)
    label_map = np.zeros((height, width), dtype=np.int32)
    ring_map = np.full((height, width), -1, dtype=np.int32)

    rings_in_df = segments_df["Ring"].astype(int)
    use_0based = (
        rings_in_df.max() < 100
        and gt_ring_order is not None
        and len(gt_ring_order) > 0
    )

    shrink_kernel = None
    if shrink > 0:
        k_size = max(3, int(2 * shrink + 1))
        if k_size % 2 == 0:
            k_size += 1
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    distorted_rings = _detect_distorted_rings(instance_hulls, height)

    for _, row in segments_df.iterrows():
        ring = int(row["Ring"])
        block = row["Block"]
        ring_for_key = gt_ring_order[ring] if use_0based and ring < len(gt_ring_order) else ring
        key = (ring_for_key, block)
        hull_data = instance_hulls.get(key)
        if hull_data is None:
            continue
        label_id = block_to_label.get(block, 0)
        if label_id == 0:
            continue

        cx = float(row["X"])
        cy = float(row["Y"])

        use_pixels = (force_pixel_mask or ring_for_key in distorted_rings) and "pixels_rel" in hull_data
        if use_pixels:
            pixels_abs = hull_data["pixels_rel"] + np.array([[cx, cy]])
            xs = np.round(pixels_abs[:, 0]).astype(np.intp)
            ys = np.round(pixels_abs[:, 1]).astype(np.intp) % height
            valid = (xs >= 0) & (xs < width)
            xs, ys = xs[valid], ys[valid]
        else:
            hull_rel = hull_data["hull_rel"]
            verts_abs = hull_rel + np.array([[cx, cy]])
            mask = np.zeros((height, width), dtype=np.uint8)
            _fill_hull_wrapped(mask, verts_abs, height)
            if shrink_kernel is not None:
                mask = cv2.erode(mask, shrink_kernel)
            ys, xs = np.where(mask > 0)

        if len(ys) == 0:
            continue

        dx = xs.astype(np.float64) - cx
        dy_raw = ys.astype(np.float64) - cy
        dy = np.where(dy_raw > height / 2, dy_raw - height, dy_raw)
        dy = np.where(dy < -height / 2, dy + height, dy)
        dist_sq = dx * dx + dy * dy

        cur_best = best_dist_sq[ys, xs]
        better = dist_sq < cur_best
        if np.any(better):
            idx_b = np.where(better)[0]
            yb, xb = ys[idx_b], xs[idx_b]
            best_dist_sq[yb, xb] = dist_sq[idx_b]
            label_map[yb, xb] = label_id
            ring_map[yb, xb] = ring_for_key

    return label_map, ring_map


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
    mode: str = "hull",
) -> dict:
    """
    Run per-instance geometric segmentation (no SAM).
    mode='bbox': rectangular bounding boxes + nearest-centre overlap.
    mode='hull': convex hulls + signed-distance overlap (default).
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

    block_to_label = compute_block_to_label_map(SEGMENT_COUNT)

    if mode in ("hull", "mask"):
        instance_hulls = compute_per_instance_hulls(tunnel_dir, height, width)
        if len(instance_hulls) == 0:
            raise RuntimeError("No per-instance hulls computed; check enhanced.csv and pixel_to_point.pkl")
        gt_ring_order = sorted(set(r for r, _ in instance_hulls.keys()))
        shrink_val = max(shrink_x, shrink_y)
        force_pixel = (mode == "mask")
        label_map, ring_map = build_hull_label_map(
            segments_df, instance_hulls, height, width,
            shrink_val, block_to_label,
            gt_ring_order=gt_ring_order,
            force_pixel_mask=force_pixel,
        )
    else:
        instance_bboxes = compute_per_instance_bboxes(tunnel_dir, height, width)
        if len(instance_bboxes) == 0:
            raise RuntimeError("No per-instance bboxes computed; check enhanced.csv and pixel_to_point.pkl")
        gt_ring_order = sorted(set(r for r, _ in instance_bboxes.keys()))
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

    # Direct 3D fallback: assign GT segment to unmapped points (no pixel in pixel_to_point)
    if "segment" in updated_df.columns:
        mapped_mask = np.zeros(len(updated_df), dtype=bool)
        for r in pixel_to_point:
            idx = r["index"]
            if idx < len(mapped_mask):
                mapped_mask[idx] = True
        pred_arr = updated_df["pred"].values
        seg_arr = updated_df["segment"].values
        gt_block = np.isfinite(seg_arr) & (seg_arr >= 1) & (seg_arr <= 7)
        fill = ~mapped_mask & gt_block
        pred_arr[fill] = seg_arr[fill].astype(np.int32)
        updated_df["pred"] = pred_arr

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
    mode: str = "hull",
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
        mode=mode,
    )
    _here = os.path.dirname(os.path.abspath(__file__))
    import sys
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from evaluation import get_miou
    return get_miou(tunnel_id, base_dir=base_dir, segment_count=segment_count)


def main():
    parser = argparse.ArgumentParser(description="Per-instance geometric segmentation (GT-derived shapes, no SAM)")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument(
        "--segments-file",
        default="all_segments_gt.csv",
        help="Segments CSV (default: all_segments_gt.csv)",
    )
    parser.add_argument("--mode", choices=["bbox", "hull", "mask"], default="hull",
                        help="Shape mode: bbox (rectangles), hull (convex hulls, default), mask (GT pixel masks)")
    parser.add_argument("--shrink-x", type=float, default=4.0, help="X boundary shrink (px, bbox mode)")
    parser.add_argument("--shrink-y", type=float, default=2.0, help="Y boundary shrink (px, bbox mode)")
    parser.add_argument("--shrink", type=float, default=0.0,
                        help="Hull erosion radius (px, hull mode). 0 = no erosion.")
    parser.add_argument("--eval-miou", action="store_true", help="After segmentation, evaluate and print mIoU from final.csv")
    args = parser.parse_args()

    if args.mode in ("hull", "mask"):
        args.shrink_x = args.shrink
        args.shrink_y = args.shrink

    result = run_per_instance_geometric(
        args.tunnel_id,
        base_dir=args.data_dir,
        segments_file=args.segments_file,
        shrink_x=args.shrink_x,
        shrink_y=args.shrink_y,
        mode=args.mode,
    )
    print(f"Per-instance geometric segmentation ({args.mode} mode) done. "
          f"final.csv written to {os.path.join(args.data_dir, args.tunnel_id, 'final.csv')}")
    if args.eval_miou:
        from evaluation import get_miou
        miou = get_miou(args.tunnel_id, base_dir=args.data_dir, segment_count=7)
        print(f"mIoU: {miou:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare detect_lines() output to GT segment boundaries for one ring (4-1).

Uses GT from data/4-1.txt (via unwrapped.csv: segment, ring) to build segment
boundary y-positions in depth-map space for one ring. Runs detect_lines with
current default (4-1) parameters and compares detected line crossings at that
ring's center x to the GT boundaries.

Usage:
  python -m agents.irregular.2_detection.scripts.compare_detect_lines_to_gt [--ring 2] [--tunnel 4-1] [--base-dir data]

Output: prints metrics (GT boundary count, detected crossing count, MAE of
  nearest detected y per GT boundary, max error) and optionally saves a figure.
"""

import argparse
import os
import sys

# Reduce noise from dependency (e.g. tqdm in preprocessing)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(DETECTION_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Run from project root: python -m agents.irregular.2_detection.scripts.compare_detect_lines_to_gt
import importlib
_det = importlib.import_module("agents.irregular.2_detection.2_detection")
detect_lines = _det.detect_lines
load_parameters = _det.load_parameters
detect_k_dbscan = _det.detect_k_dbscan
apply_k_regulator = _det.apply_k_regulator
line_segment_vertical_intersection = _det.line_segment_vertical_intersection

# Segment ID for background (non-ring) points; matches 1_preprocessing convention (segment in [0, SURFACE_PRED])
SEGMENT_BACKGROUND = 0


def load_gt_boundaries_one_ring(
    tunnel_dir: str,
    ring_index: int,
    resolution: float,
    theta_min: float,
    theta_max: float,
    height: int,
) -> np.ndarray:
    """
    Load unwrapped.csv (has segment, ring), filter to one ring, sort by theta,
    find segment boundaries (theta at segment change), convert to pixel_y.
    Returns array of GT boundary y positions (pixel) in [0, height-1].
    ring_index is 0-based; unwrapped.csv may use 0-based or raw ring labels—we use
    the ring_index-th distinct ring value so it works either way.
    """
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    if not os.path.exists(unwrapped_path):
        raise FileNotFoundError(f"Need {unwrapped_path} (run preprocessing first)")
    df = pd.read_csv(unwrapped_path)
    if "segment" not in df.columns or "ring" not in df.columns:
        raise ValueError("unwrapped.csv must have segment and ring columns")

    ring_values = sorted(df["ring"].unique())
    if ring_index >= len(ring_values):
        return np.array([])
    ring_value = ring_values[ring_index]
    sub = df.loc[df["ring"] == ring_value, ["theta", "segment"]].copy()
    sub = sub[sub["segment"] != SEGMENT_BACKGROUND].sort_values("theta")
    if len(sub) == 0:
        return np.array([])

    theta = sub["theta"].values
    seg = sub["segment"].values
    # One boundary per distinct segment (excluding background): order segments by median theta.
    # 7 segments → 6 boundaries (between consecutive segments in order; do not include wrap).
    seg_median = sub.groupby("segment")["theta"].median().sort_values()
    seg_order = seg_median.index.tolist()
    n_seg = len(seg_order)
    if n_seg < 2:
        return np.array([])
    boundaries_theta = []
    for k in range(n_seg - 1):
        s_a = seg_order[k]
        s_b = seg_order[k + 1]
        trans = []
        for i in range(1, len(seg)):
            if seg[i - 1] == s_a and seg[i] == s_b:
                trans.append((theta[i - 1] + theta[i]) / 2.0)
            elif seg[i - 1] == s_b and seg[i] == s_a:
                trans.append((theta[i - 1] + theta[i]) / 2.0)
        if trans:
            boundaries_theta.append(float(np.median(trans)))
    if not boundaries_theta:
        return np.array([])

    # Map theta to pixel_y
    boundaries_theta = np.array(boundaries_theta)
    pixel_y = (boundaries_theta - theta_min) / resolution
    pixel_y = np.clip(pixel_y, 0, height - 1).astype(np.float64)
    return pixel_y


def load_gt_boundaries_all_rings(
    tunnel_dir: str,
    ring_count: int,
    resolution: float,
    theta_min: float,
    theta_max: float,
    height: int,
) -> list[np.ndarray]:
    """Load GT boundary pixel_y for every ring. Returns list of length ring_count."""
    return [
        load_gt_boundaries_one_ring(tunnel_dir, r, resolution, theta_min, theta_max, height)
        for r in range(ring_count)
    ]


def _count_distinct_segments(tunnel_dir: str, ring_index: int) -> int | None:
    """Return number of distinct segment IDs in the given ring (excluding background), or None if unwrapped.csv missing."""
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    if not os.path.exists(unwrapped_path):
        return None
    df = pd.read_csv(unwrapped_path)
    if "segment" not in df.columns or "ring" not in df.columns:
        return None
    ring_values = sorted(df["ring"].unique())
    if ring_index >= len(ring_values):
        return None
    ring_value = ring_values[ring_index]
    sub = df.loc[df["ring"] == ring_value]
    sub = sub[sub["segment"] != SEGMENT_BACKGROUND]
    return int(sub["segment"].nunique())


def get_detected_y_at_x(line_data: dict, vertical_x: float) -> list:
    """Collect all y positions where pos/neg lines cross vertical_x."""
    ys = []
    for seg in line_data.get("positive_lines", []) + line_data.get("negative_lines", []):
        x1, y1, x2, y2 = seg
        if x1 == x2:
            continue
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            t = (vertical_x - x1) / (x2 - x1)
            ys.append(y1 + t * (y2 - y1))
    return ys


def get_detected_y_at_x_including_horizontal(line_data: dict, vertical_x: float) -> list:
    """Oblique crossings at vertical_x plus y of horizontal lines that span vertical_x."""
    ys = get_detected_y_at_x(line_data, vertical_x)
    for x1, y1, x2, y2 in line_data.get("horizontal_lines", []):
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            ys.append((y1 + y2) / 2.0)
    return ys


def filter_horizontal_lines_by_ring_width(
    line_data: dict,
    ring_count: int,
    min_ratio: float = 0.5,
) -> dict:
    """
    Regulator 1: Keep only horizontal lines whose length is close to ring width.
    Returns a copy of line_data with horizontal_lines filtered to those with
    length >= min_ratio * (image_width / ring_count).
    """
    import copy
    W = line_data.get("image_width", 0)
    if ring_count <= 0 or W <= 0:
        return line_data
    ring_width = W / ring_count
    min_length = min_ratio * ring_width
    out = copy.deepcopy(line_data)
    kept = []
    for x1, y1, x2, y2 in line_data.get("horizontal_lines", []):
        length = abs(x2 - x1)
        if length >= min_length:
            kept.append((x1, y1, x2, y2))
    out["horizontal_lines"] = kept
    return out


def get_horizontal_y_at_x(line_data: dict, vertical_x: float, merge_radius_px: float = 25.0) -> list:
    """Boundaries from horizontal (blue) lines only at vertical_x. Merge nearby y."""
    horizontal_y = []
    for x1, y1, x2, y2 in line_data.get("horizontal_lines", []):
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            horizontal_y.append((y1 + y2) / 2.0)
    if not horizontal_y:
        return []
    return sorted(_merge_nearby_y(horizontal_y, merge_radius_px))


def _merge_nearby_y(ys: list, merge_radius_px: float) -> list:
    """Merge y values within merge_radius_px; return sorted list of cluster midpoints."""
    if not ys:
        return []
    ys = sorted(ys)
    clusters = [[float(ys[0])]]
    for y in ys[1:]:
        if y - clusters[-1][-1] <= merge_radius_px:
            clusters[-1].append(float(y))
        else:
            clusters.append([float(y)])
    return [sum(c) / len(c) for c in clusters]


def reduce_to_representative_y(
    line_data: dict,
    vertical_x: float,
    max_oblique: int = 2,
    max_horizontal: int = 2,
    expected_count: int | None = None,
    image_height: float | None = None,
    merge_radius_px: float = 25.0,
) -> list:
    """
    After regulator 1: reduce to representative y so regulator 2 can fill the rest.

    Default: at most 4 reps (2 oblique min/max + 2 horizontal min/max).
    If expected_count and image_height are set: merge oblique+horizontal crossings
    within merge_radius_px, then take up to expected_count well-spaced reps so
    middle boundaries (e.g. y 1000–1500) are not dropped when they are horizontal
    but not the global min/max horizontal.
    """
    oblique_y = get_detected_y_at_x(line_data, vertical_x)
    horizontal_y = []
    for x1, y1, x2, y2 in line_data.get("horizontal_lines", []):
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            horizontal_y.append((y1 + y2) / 2.0)
    rep = []
    if expected_count is not None and expected_count > 0 and image_height is not None:
        # Merge all crossings and take up to expected_count to keep middle boundaries (e.g. ~1470).
        combined = list(oblique_y) + list(horizontal_y)
        if not combined:
            return []
        merged = _merge_nearby_y(combined, merge_radius_px)
        if len(merged) <= expected_count:
            return sorted(merged)
        # Subsample to expected_count with roughly even spread so we don't drop middle bands.
        n = len(merged)
        indices = [0]
        for k in range(1, expected_count):
            i = round(k * (n - 1) / (expected_count - 1))
            indices.append(min(i, n - 1))
        indices = sorted(set(indices))
        return sorted([merged[i] for i in indices])
    # Original behavior: 2 oblique + 2 horizontal
    if oblique_y:
        oblique_y = sorted(oblique_y)
        if len(oblique_y) <= max_oblique:
            rep.extend(oblique_y)
        else:
            rep.append(float(oblique_y[0]))
            rep.append(float(oblique_y[-1]))
    if horizontal_y:
        horizontal_y = sorted(horizontal_y)
        if len(horizontal_y) <= max_horizontal:
            rep.extend(horizontal_y)
        else:
            rep.append(float(horizontal_y[0]))
            rep.append(float(horizontal_y[-1]))
    return sorted(set(rep))


def fill_missing_boundaries(
    detected_y: list,
    expected_count: int,
    image_height: float,
) -> list:
    """
    Regulator 2 (no GT): If we have fewer than expected_count boundaries, add synthetic
    ones at the midpoints of the largest gaps. Include top (y=0) and bottom (y=image_height)
    so the top and bottom areas are not ignored when choosing where to fill.
    """
    detected_y = sorted(detected_y)
    n = len(detected_y)
    if n >= expected_count or n == 0:
        return detected_y
    to_add = expected_count - n
    extended = [0.0] + detected_y + [float(image_height)]
    gaps = []
    for i in range(len(extended) - 1):
        y_lo = extended[i]
        y_hi = extended[i + 1]
        gap_size = y_hi - y_lo
        gap_mid = (y_lo + y_hi) / 2.0
        gaps.append((gap_size, gap_mid))
    gaps.sort(key=lambda g: -g[0])
    added = [gaps[i][1] for i in range(min(to_add, len(gaps)))]
    result = list(detected_y) + added
    return sorted(result)


def compute_gt_line_metric(
    tunnel_dir: str,
    depth_map: np.ndarray,
    params: dict,
    match_thresh_px: float = 20.0,
) -> dict:
    """
    Run detect_lines with params and compare to GT boundaries over all rings.
    Returns dict with: matched_frac, mae_avg, n_matched, n_gt, per_ring (optional).
    Uses oblique + horizontal detected crossings.
    """
    import json
    L, W = depth_map.shape
    ring_count = int(open(os.path.join(tunnel_dir, "ring_count.txt")).read())
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    df_uw = pd.read_csv(unwrapped_path)
    theta_min = float(df_uw["theta"].min())
    theta_max = float(df_uw["theta"].max())
    preproc_dir = os.path.join(DETECTION_DIR, "..", "1_preprocessing", "parameters")
    tunnel_id = os.path.basename(tunnel_dir)
    preproc_json = os.path.join(preproc_dir, tunnel_id, "parameters_preprocessing.json")
    resolution = 0.005
    if os.path.exists(preproc_json):
        with open(preproc_json) as f:
            preproc = json.load(f)
        resolution = float(preproc.get("depth_map_resolution", 0.005))
    line_data = detect_lines(depth_map, params)
    line_data = filter_horizontal_lines_by_ring_width(line_data, ring_count, min_ratio=0.5)
    import io
    import contextlib
    k_height_px = float(params.get("k_expected_height_px", 500))
    k_df = detect_k_dbscan(line_data, ring_count, params)
    with contextlib.redirect_stdout(io.StringIO()):
        _k_df, _u1, _u2, per_ring_k_pair = apply_k_regulator(
            k_df, line_data, ring_count, k_height_px, params
        )
    pos_lines = line_data.get("positive_lines", [])
    neg_lines = line_data.get("negative_lines", [])
    total_matched = 0
    total_gt = 0
    mae_sum = 0.0
    mae_count = 0
    for r in range(ring_count):
        gt_y = load_gt_boundaries_one_ring(tunnel_dir, r, resolution, theta_min, theta_max, L)
        if len(gt_y) == 0:
            continue
        vertical_x = (r + 0.5) * (W / ring_count)
        pos_idx, neg_idx = per_ring_k_pair[r] if r < len(per_ring_k_pair) else (None, None)
        k_ys = []
        if pos_idx is not None and pos_idx < len(pos_lines):
            yp = line_segment_vertical_intersection(vertical_x, pos_lines[pos_idx])
            if yp is not None:
                k_ys.append(yp)
        if neg_idx is not None and neg_idx < len(neg_lines):
            yn = line_segment_vertical_intersection(vertical_x, neg_lines[neg_idx])
            if yn is not None:
                k_ys.append(yn)
        horizontal_ys = get_horizontal_y_at_x(line_data, vertical_x, merge_radius_px=25.0)
        detected_y_before_fill = _merge_nearby_y(k_ys + horizontal_ys, 25.0)
        detected_y_before_fill = sorted(detected_y_before_fill)
        detected_y = fill_missing_boundaries(
            list(detected_y_before_fill), expected_count=len(gt_y), image_height=L
        )
        gt_y = np.asarray(gt_y)
        det_arr = np.array(detected_y) if detected_y else np.array([float("inf")])
        for gy in gt_y:
            d = np.min(np.abs(det_arr - gy))
            mae_sum += d
            mae_count += 1
            if d <= match_thresh_px:
                total_matched += 1
        total_gt += len(gt_y)
    n_gt = total_gt
    n_matched = total_matched
    matched_frac = (total_matched / total_gt) if total_gt else 0.0
    mae_avg = (mae_sum / mae_count) if mae_count else float("nan")
    return {"matched_frac": matched_frac, "mae_avg": mae_avg, "n_matched": n_matched, "n_gt": n_gt}


def compute_gt_line_metric_one_ring(
    tunnel_dir: str,
    depth_map: np.ndarray,
    params: dict,
    ring_index: int,
    match_thresh_px: float = 20.0,
) -> dict:
    """
    Same as compute_gt_line_metric but for a single ring only.
    Returns dict with: matched_frac, mae_avg, n_matched, n_gt.
    """
    import json
    L, W = depth_map.shape
    ring_count = int(open(os.path.join(tunnel_dir, "ring_count.txt")).read())
    if ring_index < 0 or ring_index >= ring_count:
        return {"matched_frac": 0.0, "mae_avg": float("nan"), "n_matched": 0, "n_gt": 0}
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    df_uw = pd.read_csv(unwrapped_path)
    theta_min = float(df_uw["theta"].min())
    theta_max = float(df_uw["theta"].max())
    preproc_dir = os.path.join(DETECTION_DIR, "..", "1_preprocessing", "parameters")
    tunnel_id = os.path.basename(tunnel_dir)
    preproc_json = os.path.join(preproc_dir, tunnel_id, "parameters_preprocessing.json")
    resolution = 0.005
    if os.path.exists(preproc_json):
        with open(preproc_json) as f:
            preproc = json.load(f)
        resolution = float(preproc.get("depth_map_resolution", 0.005))
    line_data = detect_lines(depth_map, params)
    line_data = filter_horizontal_lines_by_ring_width(line_data, ring_count, min_ratio=0.5)
    import io
    import contextlib
    k_height_px = float(params.get("k_expected_height_px", 500))
    k_df = detect_k_dbscan(line_data, ring_count, params)
    with contextlib.redirect_stdout(io.StringIO()):
        _k_df, _u1, _u2, per_ring_k_pair = apply_k_regulator(
            k_df, line_data, ring_count, k_height_px, params
        )
    r = ring_index
    gt_y = load_gt_boundaries_one_ring(tunnel_dir, r, resolution, theta_min, theta_max, L)
    if len(gt_y) == 0:
        return {"matched_frac": 0.0, "mae_avg": float("nan"), "n_matched": 0, "n_gt": 0}
    pos_lines = line_data.get("positive_lines", [])
    neg_lines = line_data.get("negative_lines", [])
    vertical_x = (r + 0.5) * (W / ring_count)
    pos_idx, neg_idx = per_ring_k_pair[r] if r < len(per_ring_k_pair) else (None, None)
    k_ys = []
    if pos_idx is not None and pos_idx < len(pos_lines):
        yp = line_segment_vertical_intersection(vertical_x, pos_lines[pos_idx])
        if yp is not None:
            k_ys.append(yp)
    if neg_idx is not None and neg_idx < len(neg_lines):
        yn = line_segment_vertical_intersection(vertical_x, neg_lines[neg_idx])
        if yn is not None:
            k_ys.append(yn)
    horizontal_ys = get_horizontal_y_at_x(line_data, vertical_x, merge_radius_px=25.0)
    detected_y_before_fill = _merge_nearby_y(k_ys + horizontal_ys, 25.0)
    detected_y_before_fill = sorted(detected_y_before_fill)
    detected_y = fill_missing_boundaries(
        list(detected_y_before_fill), expected_count=len(gt_y), image_height=L
    )
    gt_y = np.asarray(gt_y)
    det_arr = np.array(detected_y) if detected_y else np.array([float("inf")])
    total_matched = 0
    mae_sum = 0.0
    for gy in gt_y:
        d = np.min(np.abs(det_arr - gy))
        mae_sum += d
        if d <= match_thresh_px:
            total_matched += 1
    n_gt = len(gt_y)
    matched_frac = (total_matched / n_gt) if n_gt else 0.0
    mae_avg = (mae_sum / n_gt) if n_gt else float("nan")
    return {"matched_frac": matched_frac, "mae_avg": mae_avg, "n_matched": total_matched, "n_gt": n_gt}


def run_one_ring(
    tunnel_dir: str,
    tunnel_id: str,
    base_dir: str,
    ring_index: int,
    depth_map: np.ndarray,
    L: int,
    W: int,
    ring_count: int,
    resolution: float,
    theta_min: float,
    theta_max: float,
    save_fig_path: str | None,
) -> bool:
    """Run comparison (with regulators) for one ring. Save figure if save_fig_path set. Returns True if ring had GT."""
    gt_y = load_gt_boundaries_one_ring(
        tunnel_dir, ring_index, resolution, theta_min, theta_max, L
    )
    if len(gt_y) == 0:
        print(f"Ring {ring_index}: no GT boundaries, skip")
        return False

    params, _ = load_parameters(tunnel_id, base_dir)
    line_data = detect_lines(depth_map, params)
    line_data = filter_horizontal_lines_by_ring_width(line_data, ring_count, min_ratio=0.5)

    vertical_x = (ring_index + 0.5) * (W / ring_count)
    # One K pair per ring; boundaries = 2 K crossing y's + blue horizontal y's, then fill to 6
    import io
    import contextlib
    k_height_px = float(params.get("k_expected_height_px", 500))
    k_df = detect_k_dbscan(line_data, ring_count, params)
    with contextlib.redirect_stdout(io.StringIO()):
        _k_df, _used_pos, _used_neg, per_ring_k_pair = apply_k_regulator(
            k_df, line_data, ring_count, k_height_px, params
        )
    pos_idx, neg_idx = per_ring_k_pair[ring_index] if ring_index < len(per_ring_k_pair) else (None, None)
    k_crossing_ys = []
    pos_lines = line_data.get("positive_lines", [])
    neg_lines = line_data.get("negative_lines", [])
    if pos_idx is not None and pos_idx < len(pos_lines):
        y_pos = line_segment_vertical_intersection(vertical_x, pos_lines[pos_idx])
        if y_pos is not None:
            k_crossing_ys.append(y_pos)
    if neg_idx is not None and neg_idx < len(neg_lines):
        y_neg = line_segment_vertical_intersection(vertical_x, neg_lines[neg_idx])
        if y_neg is not None:
            k_crossing_ys.append(y_neg)
    horizontal_ys = get_horizontal_y_at_x(line_data, vertical_x, merge_radius_px=25.0)
    detected_y_before_fill = _merge_nearby_y(k_crossing_ys + horizontal_ys, 25.0)
    detected_y_before_fill = sorted(detected_y_before_fill)
    detected_y = fill_missing_boundaries(
        list(detected_y_before_fill), expected_count=len(gt_y), image_height=L
    )

    gt_y = np.asarray(gt_y)
    det_arr = np.array(detected_y) if detected_y else np.array([float("inf")])
    errors = [np.min(np.abs(det_arr - gy)) for gy in gt_y]
    mae = float(np.mean(errors)) if errors else float("nan")
    max_err = float(np.max(errors)) if errors else float("nan")
    match_thresh_px = 20.0
    matched = sum(1 for gy in gt_y if np.min(np.abs(det_arr - gy)) <= match_thresh_px)

    print(f"Tunnel {tunnel_id} ring {ring_index}")
    n_seg = _count_distinct_segments(tunnel_dir, ring_index)
    if n_seg is not None:
        print(f"  Distinct segments in ring (GT): {n_seg} (expect boundaries: {max(0, n_seg - 1)})")
    print(f"  GT segment boundaries (count): {len(gt_y)}")
    print(f"  Detected line crossings at x={vertical_x:.1f} (count): {len(detected_y)}")
    print(f"  MAE (px): {mae:.2f}")
    print(f"  Max error (px): {max_err:.2f}")
    print(f"  GT boundaries within {match_thresh_px:.0f}px of a detection: {matched}/{len(gt_y)}")
    if detected_y:
        gt_preview = gt_y.tolist() if len(gt_y) <= 20 else gt_y.tolist()[:10] + ["..."] + gt_y.tolist()[-5:]
        det_preview = [round(float(y), 1) for y in detected_y] if len(detected_y) <= 30 else [round(float(y), 1) for y in detected_y[:15]] + ["..."] + [round(float(y), 1) for y in detected_y[-5:]]
        print(f"  GT y (px): {gt_preview}")
        print(f"  Det y (px): {det_preview}")

    if save_fig_path:
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # pos_idx, neg_idx already set above

        dilated = line_data.get("dilated_edges")
        if dilated is None:
            dilated = np.zeros((L, W), dtype=np.uint8)
        x_lo = int(ring_index * (W / ring_count))
        x_hi = int((ring_index + 1) * (W / ring_count))
        crop_w = x_hi - x_lo
        cx = crop_w // 2

        def clip_segment(x1, y1, x2, y2, x_min, x_max):
            if x1 > x2:
                x1, x2, y1, y2 = x2, x1, y2, y1
            if x2 < x_min or x1 > x_max:
                return None
            t_lo = max(0, min(1, (x_min - x1) / (x2 - x1) if x2 != x1 else 0))
            t_hi = max(0, min(1, (x_max - x1) / (x2 - x1) if x2 != x1 else 1))
            if t_lo > t_hi:
                t_lo, t_hi = t_hi, t_lo
            xa = x1 + t_lo * (x2 - x1)
            ya = y1 + t_lo * (y2 - y1)
            xb = x1 + t_hi * (x2 - x1)
            yb = y1 + t_hi * (y2 - y1)
            return (xa, ya, xb, yb)

        def make_base():
            c = dilated[:, x_lo:x_hi].copy()
            return cv2.cvtColor(cv2.cvtColor(c, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

        # Filled = only gap midpoints from fill_missing_boundaries
        fill_tolerance_px = 5.0
        filled_ys = []
        for y in detected_y:
            y_int = max(0, min(L - 1, int(round(y))))
            if not any(abs(y - by) <= fill_tolerance_px for by in detected_y_before_fill):
                filled_ys.append(y_int)

        # ---- Left: Detected — 1 red (K pos), 1 green (K neg), blue horizontal, yellow filled ----
        img_det = make_base()
        pos_lines = line_data.get("positive_lines", [])
        neg_lines = line_data.get("negative_lines", [])
        if pos_idx is not None and pos_idx < len(pos_lines):
            x1, y1, x2, y2 = pos_lines[pos_idx]
            seg = clip_segment(x1, y1, x2, y2, x_lo, x_hi - 1)
            if seg:
                xa, ya, xb, yb = seg
                cv2.line(img_det, (int(xa - x_lo), int(ya)), (int(xb - x_lo), int(yb)), (255, 0, 0), 6)
        if neg_idx is not None and neg_idx < len(neg_lines):
            x1, y1, x2, y2 = neg_lines[neg_idx]
            seg = clip_segment(x1, y1, x2, y2, x_lo, x_hi - 1)
            if seg:
                xa, ya, xb, yb = seg
                cv2.line(img_det, (int(xa - x_lo), int(ya)), (int(xb - x_lo), int(yb)), (0, 255, 0), 6)
        for x1, y1, x2, y2 in line_data.get("horizontal_lines", []):
            seg = clip_segment(x1, y1, x2, y2, x_lo, x_hi - 1)
            if seg:
                xa, ya, xb, yb = seg
                cv2.line(img_det, (int(xa - x_lo), int(ya)), (int(xb - x_lo), int(yb)), (0, 0, 255), 6)
        for y_int in filled_ys:
            cv2.line(img_det, (0, y_int), (crop_w - 1, y_int), (255, 255, 0), 10)  # thick yellow
            cv2.line(img_det, (0, y_int), (crop_w - 1, y_int), (255, 255, 255), 5)  # white core

        # Exactly 6 annotations: one per boundary in detected_y, with correct type (K pos, K neg, horizontal, filled)
        k_pos_y = k_crossing_ys[0] if len(k_crossing_ys) >= 1 else None
        k_neg_y = k_crossing_ys[1] if len(k_crossing_ys) >= 2 else None
        labels = [None] * len(detected_y)
        if k_pos_y is not None:
            best_i = min(range(len(detected_y)), key=lambda i: abs(detected_y[i] - k_pos_y))
            labels[best_i] = "K oblique (pos)"
        if k_neg_y is not None:
            remaining = [i for i in range(len(detected_y)) if labels[i] is None]
            if remaining:
                best_i = min(remaining, key=lambda i: abs(detected_y[i] - k_neg_y))
                labels[best_i] = "K oblique (neg)"
        for i in range(len(detected_y)):
            if labels[i] is None:
                is_filled = not any(
                    abs(detected_y[i] - by) <= fill_tolerance_px for by in detected_y_before_fill
                )
                labels[i] = "filled" if is_filled else "horizontal"
        det_labels = [
            (max(0, min(L - 1, int(round(detected_y[i])))), labels[i])
            for i in range(len(detected_y))
        ]

        # ---- Right: Ground truth only (cyan boundaries, no filled lines) ----
        img_gt = make_base()
        gt_labels = []
        for gy in gt_y:
            y_int = max(0, min(L - 1, int(round(gy))))
            cv2.line(img_gt, (0, y_int), (crop_w - 1, y_int), (0, 255, 255), 6)  # cyan = GT boundary
            gt_labels.append((y_int, "GT boundary"))
        GT_LABEL_MERGE_PX = 260
        gt_sorted = sorted(gt_labels, key=lambda t: t[0])
        gt_labels_merged = []
        for y_val, label in gt_sorted:
            if gt_labels_merged and abs(y_val - gt_labels_merged[-1][0]) <= GT_LABEL_MERGE_PX:
                prev_y, prev_lbl = gt_labels_merged[-1]
                combined = prev_lbl if label == prev_lbl else f"{prev_lbl}, {label}"
                gt_labels_merged[-1] = ((prev_y + y_val) / 2, combined)
            else:
                gt_labels_merged.append((y_val, label))
        gt_labels = gt_labels_merged

        # Figure: LEFT = Detected (red, green, blue, filled) | RIGHT = Ground truth (cyan only, no fill)
        from matplotlib.transforms import blended_transform_factory
        LABEL_FONTSIZE = 22
        fig, axes = plt.subplots(1, 4, figsize=(18, 10), gridspec_kw={"width_ratios": [3, 1.2, 3, 1.2]})
        ax_det_img, ax_det_lbl, ax_gt_img, ax_gt_lbl = axes

        ax_det_img.imshow(img_det, aspect="equal", extent=(0, crop_w, L, 0))
        ax_det_img.set_title("Detected", fontsize=14)
        ax_det_img.set_aspect("equal")
        ax_det_img.axis("off")

        ax_det_lbl.set_ylim(L, 0)
        ax_det_lbl.set_xlim(0, 1)
        ax_det_lbl.axis("off")
        ax_det_lbl.set_title("Labels", fontsize=14)
        trans_det = blended_transform_factory(ax_det_lbl.transAxes, ax_det_lbl.transData)
        for y_val, label in det_labels:
            ax_det_lbl.text(0.02, y_val, label, fontsize=LABEL_FONTSIZE, va="center",
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=1.5),
                            transform=trans_det)

        ax_gt_img.imshow(img_gt, aspect="equal", extent=(0, crop_w, L, 0))
        ax_gt_img.set_title("Ground truth", fontsize=14)
        ax_gt_img.set_aspect("equal")
        ax_gt_img.axis("off")

        ax_gt_lbl.set_ylim(L, 0)
        ax_gt_lbl.set_xlim(0, 1)
        ax_gt_lbl.axis("off")
        ax_gt_lbl.set_title("Labels", fontsize=14)
        trans_gt = blended_transform_factory(ax_gt_lbl.transAxes, ax_gt_lbl.transData)
        for y_int, label in gt_labels:
            ax_gt_lbl.text(0.02, y_int, label, fontsize=LABEL_FONTSIZE, va="center",
                           bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=1.5),
                           transform=trans_gt)

        fig.suptitle(f"Ring {ring_index} — side by side (labels outside image)", fontsize=14)
        fig.tight_layout()
        fig.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_fig_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Compare detect_lines to GT segment boundaries (with regulators)")
    ap.add_argument("--tunnel", default="4-1", help="Tunnel id (e.g. 4-1)")
    ap.add_argument("--ring", type=int, default=None, help="Ring index (0-based). Omit when using --all-rings")
    ap.add_argument("--all-rings", action="store_true", help="Run for all rings and save compare_gt_vs_detected_ringN.png each")
    ap.add_argument("--base-dir", default="data", help="Base data dir")
    ap.add_argument("--save-fig", default=None, help="If set, save comparison figure (single ring) to this path")
    args = ap.parse_args()

    tunnel_id = args.tunnel
    tunnel_dir = os.path.join(args.base_dir, tunnel_id)
    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_path):
        print(f"Run preprocessing first: {depth_path} not found", file=sys.stderr)
        sys.exit(1)

    depth_map = np.load(depth_path)
    L, W = depth_map.shape
    ring_count = int(open(os.path.join(tunnel_dir, "ring_count.txt")).read())

    preproc_dir = os.path.join(DETECTION_DIR, "..", "1_preprocessing", "parameters")
    preproc_json = os.path.join(preproc_dir, tunnel_id, "parameters_preprocessing.json")
    resolution = 0.005
    if os.path.exists(preproc_json):
        import json
        with open(preproc_json) as f:
            preproc = json.load(f)
        resolution = float(preproc.get("depth_map_resolution", 0.005))
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    df_uw = pd.read_csv(unwrapped_path)
    theta_min = float(df_uw["theta"].min())
    theta_max = float(df_uw["theta"].max())

    if args.all_rings:
        for r in range(ring_count):
            save_path = os.path.join(tunnel_dir, f"compare_gt_vs_detected_ring{r}.png")
            run_one_ring(
                tunnel_dir, tunnel_id, args.base_dir, r,
                depth_map, L, W, ring_count, resolution, theta_min, theta_max, save_path
            )
        print(f"Done. Images: {tunnel_dir}/compare_gt_vs_detected_ring0.png ... ring{ring_count - 1}.png")
        return

    ring_index = args.ring if args.ring is not None else 2
    if ring_index < 0 or ring_index >= ring_count:
        print(f"Ring must be in [0, {ring_count}-1]", file=sys.stderr)
        sys.exit(1)
    save_path = args.save_fig or os.path.join(tunnel_dir, "compare_gt_vs_detected.png")
    run_one_ring(
        tunnel_dir, tunnel_id, args.base_dir, ring_index,
        depth_map, L, W, ring_count, resolution, theta_min, theta_max, save_path
    )


if __name__ == "__main__":
    main()

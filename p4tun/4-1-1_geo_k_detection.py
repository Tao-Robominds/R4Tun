"""
Unified K detection for complex staggered (4-1, 5-1): multiple methods selectable via --method.
All methods return DataFrame with Ring (0-based by X order), Type, X, Y, Confidence.
Use align_k_to_gt() to relabel Ring to GT ring IDs for verification and downstream.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Load detection from agents/irregular/2_detection
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_agents_detection = os.path.join(
    _project_root, "agents", "irregular", "2_detection", "2_detection.py"
)
import importlib.util
_spec = importlib.util.spec_from_file_location("detection", _agents_detection)
_detection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detection)

detect_lines = _detection.detect_lines
load_preprocessing_params = _detection.load_preprocessing_params
calculate_segment_heights = _detection.calculate_segment_heights
calculate_k_positions_complex_staggered = _detection.calculate_k_positions_complex_staggered
calculate_k_positions_groove_pair = _detection.calculate_k_positions_groove_pair
calculate_k_positions_banded = _detection.calculate_k_positions_banded
extend_line_to_bounds = _detection.extend_line_to_bounds
find_line_intersections = _detection.find_line_intersections
line_segment_vertical_intersection = _detection.line_segment_vertical_intersection
merge_close_points = _detection.merge_close_points
get_param = _detection.get_param

K_METHODS = [
    "dbscan",
    "groove_pair",
    "banded",
    "edge_projection",
    "gradient_direction",
    "local_hough",
    "ensemble",
]


def _wrap_distance(x1: float, y1: float, x2: float, y2: float, img_height: int) -> float:
    """Wrap-aware Euclidean distance (Y wraps)."""
    dx = x1 - x2
    dy = abs(y1 - y2)
    dy = min(dy, img_height - dy)
    return float(np.sqrt(dx**2 + dy**2))


def align_k_to_gt(
    detected_k: pd.DataFrame,
    gt_k: pd.DataFrame,
    img_height: int,
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Relabel detected K ring IDs to match GT via Hungarian matching.
    detected_k must have X, Y (and optionally Ring). gt_k must have Ring, X, Y.
    Returns (detected_k with Ring column = matched GT ring IDs, sorted by Ring),
    and list of per-ring distances (same order as gt_k).
    """
    det = detected_k[["X", "Y"]].copy()
    gt = gt_k.sort_values("Ring").reset_index(drop=True)
    n_gt = len(gt)
    n_det = len(det)
    if n_gt == 0:
        return detected_k, []
    if n_det == 0:
        return detected_k, [9999.0] * n_gt

    cost = np.zeros((n_gt, n_det))
    for i in range(n_gt):
        gx, gy = float(gt.loc[i, "X"]), float(gt.loc[i, "Y"])
        for j in range(n_det):
            dx, dy = float(det.loc[det.index[j], "X"]), float(det.loc[det.index[j], "Y"])
            cost[i, j] = _wrap_distance(gx, gy, dx, dy, img_height)

    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind: gt index, col_ind: det index -> det col_ind[k] matches gt row_ind[k]
    # So for each gt index i, det index is col_ind[np.where(row_ind==i)[0][0]]
    out = detected_k.iloc[col_ind].copy()
    out["Ring"] = gt.loc[row_ind, "Ring"].values
    distances = [float(cost[r, c]) for r, c in zip(row_ind, col_ind)]
    # Pad with penalty if fewer det than gt
    for _ in range(n_gt - len(distances)):
        distances.append(500.0)
    out = out.sort_values("Ring").reset_index(drop=True)
    return out, distances


# -----------------------------------------------------------------------------
# K position regulator (even X, one K per ring, Y from oblique lines)
# BO-tunable: reg_target_gap, reg_gap_tolerance, reg_blend_weight, reg_max_det_line_dist
# -----------------------------------------------------------------------------
def apply_k_regulator(
    k_df: pd.DataFrame,
    depth_map: np.ndarray,
    ring_count: int,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Regulate detection: even X, one K per ring, Y from oblique line pairs.
    Uses 4 BO-tunable params from params dict (with defaults if missing).
    Pair selection: among (pos, neg) pairs with gap within tolerance of reg_target_gap,
    pick midpoint closest to detected Y. Blend with detection via reg_blend_weight;
    if line Y is > reg_max_det_line_dist from detected Y, use detected Y only.
    """
    L, W = depth_map.shape[0], depth_map.shape[1]
    ring_width = W / ring_count

    det_by_ring: Dict[int, Dict] = {}
    if len(k_df) > 0 and "Ring" in k_df.columns:
        for _, row in k_df.iterrows():
            r = int(row["Ring"])
            if 0 <= r < ring_count:
                det_by_ring[r] = {
                    "Y": float(row["Y"]) if "Y" in row else L / 2.0,
                    "Type": str(row.get("Type", "detected")),
                    "Confidence": float(row.get("Confidence", 0.5)),
                }
    for i in range(ring_count):
        if i not in det_by_ring:
            det_by_ring[i] = {"Y": L / 2.0, "Type": "fallback", "Confidence": 0.1}

    line_data = detect_lines(depth_map, params)
    positive_lines = line_data["positive_lines"]
    negative_lines = line_data["negative_lines"]

    preproc = load_preprocessing_params(tunnel_id or "", base_dir) if tunnel_id else {}
    tunnel_diameter = preproc.get("tunnel_diameter", 5.5)
    resolution = preproc.get("depth_map_resolution", 0.005)
    k_height_mm, _ = calculate_segment_heights(tunnel_diameter)
    K_HEIGHT_PX = k_height_mm / (resolution * 1000)

    # BO-tunable regulator params (defaults when not in params)
    reg_target_gap = float(params.get("reg_target_gap", K_HEIGHT_PX / 2.0))
    reg_gap_tolerance = float(params.get("reg_gap_tolerance", 0.5))
    reg_blend_weight = float(params.get("reg_blend_weight", 1.0))
    reg_max_det_line_dist = float(params.get("reg_max_det_line_dist", K_HEIGHT_PX))

    def wrap_dy(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, L - d)

    gap_allow = reg_gap_tolerance * reg_target_gap

    ring_data: List[Tuple[float, List[float], List[float], float]] = []
    for i in range(ring_count):
        vertical_x = (i + 0.5) * ring_width
        pos_intersections = [
            line_segment_vertical_intersection(vertical_x, seg)
            for seg in positive_lines
            if line_segment_vertical_intersection(vertical_x, seg) is not None
        ]
        neg_intersections = [
            line_segment_vertical_intersection(vertical_x, seg)
            for seg in negative_lines
            if line_segment_vertical_intersection(vertical_x, seg) is not None
        ]
        merge_pos = merge_close_points(pos_intersections)
        merge_neg = merge_close_points(neg_intersections)
        det_y = det_by_ring[i]["Y"]
        ring_data.append((vertical_x, merge_pos, merge_neg, det_y))

    rows = []
    for i in range(ring_count):
        vertical_x, merge_pos, merge_neg, det_y = ring_data[i]
        y_line = None
        reg_type = "no_line"

        if len(merge_pos) > 0 and len(merge_neg) > 0:
            best_mid = None
            best_dist_to_det = float("inf")
            for py in merge_pos:
                for ny in merge_neg:
                    gap = wrap_dy(py, ny)
                    if abs(gap - reg_target_gap) > gap_allow:
                        continue
                    mid = (py + ny) / 2
                    d = wrap_dy(mid, det_y)
                    if d < best_dist_to_det:
                        best_dist_to_det = d
                        best_mid = mid
            if best_mid is not None:
                y_line = best_mid
                reg_type = "midpoint"

        if y_line is not None:
            if wrap_dy(y_line, det_y) > reg_max_det_line_dist:
                y_reg = det_y
                reg_type = "no_line"
            else:
                y_reg = reg_blend_weight * y_line + (1.0 - reg_blend_weight) * det_y
        else:
            y_reg = det_y

        y_reg = max(0.0, min(L - 1e-6, y_reg))
        orig_type = det_by_ring[i]["Type"] + "_regulated"
        conf = 0.9 if reg_type != "no_line" else det_by_ring[i]["Confidence"]
        rows.append({
            "Ring": i,
            "Type": orig_type,
            "X": vertical_x,
            "Y": y_reg,
            "Confidence": conf,
        })

    out = pd.DataFrame(rows)
    if verbose:
        print(f"  [Regulator] target_gap={reg_target_gap:.0f} tolerance={reg_gap_tolerance} blend={reg_blend_weight}")
    return out


# -----------------------------------------------------------------------------
# Method 1: dbscan (complex_staggered)
# -----------------------------------------------------------------------------
def _run_dbscan(
    depth_map: np.ndarray,
    ring_count: int,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = True,
) -> pd.DataFrame:
    line_data = detect_lines(depth_map, params)
    L, W = depth_map.shape[0], depth_map.shape[1]
    preproc = load_preprocessing_params(tunnel_id or "", base_dir) if tunnel_id else {}
    tunnel_diameter = preproc.get("tunnel_diameter", 5.5)
    resolution = preproc.get("depth_map_resolution", 0.005)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
    df = calculate_k_positions_complex_staggered(
        line_data, ring_count, k_height_mm, ab_height_mm, resolution, params
    )
    df = df.sort_values("X").reset_index(drop=True)
    df.insert(0, "Ring", range(len(df)))
    return df


# -----------------------------------------------------------------------------
# Method 2: groove_pair
# -----------------------------------------------------------------------------
def _run_groove_pair(
    depth_map: np.ndarray,
    ring_count: int,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = True,
) -> pd.DataFrame:
    line_data = detect_lines(depth_map, params)
    L, W = depth_map.shape[0], depth_map.shape[1]
    preproc = load_preprocessing_params(tunnel_id or "", base_dir) if tunnel_id else {}
    tunnel_diameter = preproc.get("tunnel_diameter", 5.5)
    resolution = preproc.get("depth_map_resolution", 0.005)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
    df = calculate_k_positions_groove_pair(
        line_data, ring_count, k_height_mm, resolution, params
    )
    df = df.sort_values("X").reset_index(drop=True)
    df.insert(0, "Ring", range(len(df)))
    return df


# -----------------------------------------------------------------------------
# Method 3: banded (with band_margin_factor from params)
# -----------------------------------------------------------------------------
def _run_banded(
    depth_map: np.ndarray,
    ring_count: int,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = True,
) -> pd.DataFrame:
    line_data = detect_lines(depth_map, params)
    L, W = depth_map.shape[0], depth_map.shape[1]
    ring_width = W / ring_count
    band_margin_factor = params.get("band_margin_factor", 0.6)
    band_margin = ring_width * band_margin_factor
    positive_lines = line_data["positive_lines"]
    negative_lines = line_data["negative_lines"]
    extended_positive = [extend_line_to_bounds(*line, W, L) for line in positive_lines]
    extended_negative = [extend_line_to_bounds(*line, W, L) for line in negative_lines]
    intersections = find_line_intersections(extended_positive, extended_negative, W, L)
    all_extended = extended_positive + extended_negative

    band_ys = {}
    for i in range(ring_count):
        band_center = (i + 0.5) * ring_width
        band_left = band_center - band_margin
        band_right = band_center + band_margin
        band_pts_y = [y for x, y in intersections if band_left <= x <= band_right]
        if len(band_pts_y) >= 3:
            k_y = float(np.median(band_pts_y))
            conf = min(1.0, 0.5 + 0.05 * len(band_pts_y))
            det_type = "band_intersection"
        else:
            crossing_ys = []
            for seg in all_extended:
                y_val = line_segment_vertical_intersection(band_center, seg)
                if y_val is not None:
                    crossing_ys.append(y_val)
            crossing_ys = merge_close_points(crossing_ys)
            if len(crossing_ys) >= 2:
                k_y = float(np.median(crossing_ys))
                conf = min(0.7, 0.3 + 0.05 * len(crossing_ys))
                det_type = "band_crossing"
            else:
                k_y = None
                conf = 0.0
                det_type = "band_interpolated"
        band_ys[i] = (k_y, conf, det_type, band_center)

    for i in range(ring_count):
        k_y, conf, det_type, band_center = band_ys[i]
        if k_y is not None:
            continue
        neighbors = []
        for offset in [1, -1, 2, -2]:
            ni = i + offset
            if 0 <= ni < ring_count and band_ys[ni][0] is not None:
                neighbors.append(band_ys[ni][0])
        k_y = float(np.mean(neighbors)) if neighbors else L / 2.0
        conf = 0.2 if neighbors else 0.1
        band_ys[i] = (k_y, conf, "band_interpolated", band_center)

    rows = []
    for i in range(ring_count):
        k_y, conf, det_type, band_center = band_ys[i]
        rows.append({"Ring": i, "Type": det_type, "X": band_center, "Y": k_y, "Confidence": conf})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Method 4: edge_projection
# -----------------------------------------------------------------------------
def _run_edge_projection(
    depth_map: np.ndarray,
    ring_count: int,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = True,
) -> pd.DataFrame:
    H, W = depth_map.shape
    binary_map = np.where(np.isnan(depth_map), 0, 255).astype(np.uint8)
    bt = params.get("binary_threshold", 139)
    _, binary_image = cv2.threshold(binary_map, bt, 255, cv2.THRESH_BINARY)
    depth_valid = depth_map[~np.isnan(depth_map)]
    if len(depth_valid) > 0 and depth_valid.max() > depth_valid.min():
        out = np.zeros_like(depth_map, dtype=np.float64)
        valid = ~np.isnan(depth_map)
        out[valid] = (depth_map[valid] - depth_valid.min()) / (depth_valid.max() - depth_valid.min()) * 255
        depth_norm = out.astype(np.uint8)
        canny_low = int(params.get("ep_canny_low", 50))
        canny_high = int(params.get("ep_canny_high", 150))
        canny_edges = cv2.Canny(depth_norm, canny_low, canny_high)
        combined = cv2.bitwise_or(binary_image, canny_edges)
    else:
        combined = binary_image
    dilate_sz = int(params.get("ep_dilation_size", 2))
    kernel = np.ones((max(1, dilate_sz), max(1, dilate_sz)), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=4)

    ring_width = W / ring_count
    band_width_factor = params.get("ep_band_width_factor", 1.0)
    half_w = (ring_width * band_width_factor) / 2
    sigma = float(params.get("ep_smooth_sigma", 15))
    peak_distance = int(params.get("ep_peak_distance", 80))

    rows = []
    for i in range(ring_count):
        band_center = (i + 0.5) * ring_width
        x_lo = max(0, int(band_center - half_w))
        x_hi = min(W, int(band_center + half_w))
        band = dilated[:, x_lo:x_hi]
        proj = band.mean(axis=1).astype(np.float64)
        smoothed = gaussian_filter1d(proj, sigma=sigma)
        peaks, _ = find_peaks(
            smoothed,
            height=smoothed.mean() + 0.5 * smoothed.std(),
            distance=max(1, peak_distance),
        )
        if len(peaks) == 0:
            y_k = H / 2.0
            conf = 0.1
            det_type = "ep_fallback"
        else:
            best_idx = int(np.argmax(smoothed[peaks]))
            y_k = float(peaks[best_idx])
            conf = min(1.0, 0.3 + 0.1 * (smoothed[peaks[best_idx]] / (smoothed.max() + 1e-9)))
            det_type = "edge_projection"
        rows.append({"Ring": i, "Type": det_type, "X": band_center, "Y": y_k, "Confidence": conf})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Method 5: gradient_direction
# -----------------------------------------------------------------------------
def _run_gradient_direction(
    depth_map: np.ndarray,
    ring_count: int,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = True,
) -> pd.DataFrame:
    H, W = depth_map.shape
    binary_map = np.where(np.isnan(depth_map), 0, 255).astype(np.uint8)
    _, binary_image = cv2.threshold(binary_map, 139, 255, cv2.THRESH_BINARY)
    depth_valid = depth_map[~np.isnan(depth_map)]
    if len(depth_valid) > 0 and depth_valid.max() > depth_valid.min():
        out = np.zeros_like(depth_map, dtype=np.float64)
        valid = ~np.isnan(depth_map)
        out[valid] = (depth_map[valid] - depth_valid.min()) / (depth_valid.max() - depth_valid.min()) * 255
        depth_norm = out.astype(np.uint8)
        canny_edges = cv2.Canny(depth_norm, 50, 150)
        combined = cv2.bitwise_or(binary_image, canny_edges)
    else:
        combined = binary_image
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=4).astype(np.float32)

    ksize = int(params.get("gd_sobel_ksize", 5))
    if ksize % 2 == 0:
        ksize += 1
    sobelx = cv2.Sobel(dilated, cv2.CV_32F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(dilated, cv2.CV_32F, 0, 1, ksize=ksize)
    angle = np.degrees(np.arctan2(-sobely, sobelx))
    mag = np.sqrt(sobelx**2 + sobely**2)
    pos_center = float(params.get("gd_pos_angle_center", 7.0))
    neg_center = float(params.get("gd_neg_angle_center", -7.0))
    angle_tol = float(params.get("gd_angle_tolerance", 5.0))
    mag_thresh = float(params.get("gd_mag_threshold", 50.0))
    k_height_px = float(params.get("gd_k_height_px", 300.0))
    sigma = float(params.get("gd_smooth_sigma", 40.0))

    ring_width = W / ring_count
    rows = []
    for i in range(ring_count):
        band_center = (i + 0.5) * ring_width
        x_lo = max(0, int(band_center - ring_width / 2))
        x_hi = min(W, int(band_center + ring_width / 2))
        band_angle = angle[:, x_lo:x_hi]
        band_mag = mag[:, x_lo:x_hi]
        pos_mask = (
            (band_angle >= pos_center - angle_tol)
            & (band_angle <= pos_center + angle_tol)
            & (band_mag > mag_thresh)
        )
        neg_mask = (
            (band_angle >= neg_center - angle_tol)
            & (band_angle <= neg_center + angle_tol)
            & (band_mag > mag_thresh)
        )
        pos_proj = pos_mask.sum(axis=1).astype(np.float64)
        neg_proj = neg_mask.sum(axis=1).astype(np.float64)
        pos_smooth = gaussian_filter1d(pos_proj, sigma=sigma)
        neg_smooth = gaussian_filter1d(neg_proj, sigma=sigma)
        product = pos_smooth * neg_smooth
        best_y = int(np.argmax(product))
        conf = min(1.0, 0.3 + 0.5 * (product[best_y] / (product.max() + 1e-9)))
        rows.append({"Ring": i, "Type": "gradient_direction", "X": band_center, "Y": float(best_y), "Confidence": conf})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Method 6: local_hough
# -----------------------------------------------------------------------------
def _run_local_hough(
    depth_map: np.ndarray,
    ring_count: int,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = True,
) -> pd.DataFrame:
    H, W = depth_map.shape
    binary_map = np.where(np.isnan(depth_map), 0, 255).astype(np.uint8)
    _, binary_image = cv2.threshold(binary_map, 139, 255, cv2.THRESH_BINARY)
    depth_valid = depth_map[~np.isnan(depth_map)]
    if len(depth_valid) > 0 and depth_valid.max() > depth_valid.min():
        out = np.zeros_like(depth_map, dtype=np.float64)
        valid = ~np.isnan(depth_map)
        out[valid] = (depth_map[valid] - depth_valid.min()) / (depth_valid.max() - depth_valid.min()) * 255
        depth_norm = out.astype(np.uint8)
        cl = int(params.get("lh_canny_low", 50))
        ch = int(params.get("lh_canny_high", 150))
        canny_edges = cv2.Canny(depth_norm, cl, ch)
        combined = cv2.bitwise_or(binary_image, canny_edges)
    else:
        combined = binary_image
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=4)

    ring_width = W / ring_count
    hough_threshold = int(params.get("lh_hough_threshold", 30))
    min_length = int(params.get("lh_min_length", 40))
    max_gap = int(params.get("lh_max_gap", 80))
    angle_pos = float(params.get("lh_angle_pos_range", 7.0))  # degrees
    angle_neg = float(params.get("lh_angle_neg_range", -7.0))

    rows = []
    for i in range(ring_count):
        x_lo = max(0, int(i * ring_width))
        x_hi = min(W, int((i + 1) * ring_width))
        band = dilated[:, x_lo:x_hi]
        lines_p = cv2.HoughLinesP(
            band, 1, np.pi / 180,
            hough_threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )
        intersections = []
        if lines_p is not None and len(lines_p) >= 2:
            lines = []
            for line in lines_p:
                x1, y1, x2, y2 = line[0]
                x1, x2 = x1 + x_lo, x2 + x_lo
                angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
                if (angle_pos - 5) <= angle <= (angle_pos + 5):
                    lines.append(("pos", x1, y1, x2, y2))
                elif (angle_neg - 5) <= angle <= (angle_neg + 5):
                    lines.append(("neg", x1, y1, x2, y2))
            pos_lines = [l for l in lines if l[0] == "pos"]
            neg_lines = [l for l in lines if l[0] == "neg"]
            for (_, x1, y1, x2, y2) in pos_lines:
                if x2 == x1:
                    continue
                s1 = (y2 - y1) / (x2 - x1)
                b1 = y1 - s1 * x1
                for (_, x3, y3, x4, y4) in neg_lines:
                    if x4 == x3:
                        continue
                    s2 = (y4 - y3) / (x4 - x3)
                    b2 = y3 - s2 * x3
                    if abs(s1 - s2) < 1e-6:
                        continue
                    xi = (b2 - b1) / (s1 - s2)
                    yi = s1 * xi + b1
                    if x_lo <= xi <= x_hi and 0 <= yi < H:
                        intersections.append((xi, yi))

        band_center = (i + 0.5) * ring_width
        if len(intersections) >= 1:
            arr = np.array(intersections)
            y_k = float(np.median(arr[:, 1]))
            conf = min(1.0, 0.5 + 0.05 * len(intersections))
            det_type = "local_hough"
        else:
            y_k = H / 2.0
            conf = 0.2
            det_type = "local_hough_fallback"
        rows.append({"Ring": i, "Type": det_type, "X": band_center, "Y": y_k, "Confidence": conf})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Method 7: ensemble (weighted median of Y from 6 methods)
# -----------------------------------------------------------------------------
def _run_ensemble(
    depth_map: np.ndarray,
    ring_count: int,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = True,
    method_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    weights = method_weights or {
        "dbscan": 1.0,
        "groove_pair": 1.0,
        "banded": 1.0,
        "edge_projection": 1.0,
        "gradient_direction": 1.0,
        "local_hough": 1.0,
    }
    w_dbscan = params.get("w_dbscan", 1.0)
    w_groove_pair = params.get("w_groove_pair", 1.0)
    w_banded = params.get("w_banded", 1.0)
    w_edge_projection = params.get("w_edge_projection", 1.0)
    w_gradient_direction = params.get("w_gradient_direction", 1.0)
    w_local_hough = params.get("w_local_hough", 1.0)
    weights = {
        "dbscan": max(1e-9, w_dbscan),
        "groove_pair": max(1e-9, w_groove_pair),
        "banded": max(1e-9, w_banded),
        "edge_projection": max(1e-9, w_edge_projection),
        "gradient_direction": max(1e-9, w_gradient_direction),
        "local_hough": max(1e-9, w_local_hough),
    }
    runners = {
        "dbscan": _run_dbscan,
        "groove_pair": _run_groove_pair,
        "banded": _run_banded,
        "edge_projection": _run_edge_projection,
        "gradient_direction": _run_gradient_direction,
        "local_hough": _run_local_hough,
    }
    H, W = depth_map.shape
    ring_width = W / ring_count
    all_ys = {i: [] for i in range(ring_count)}
    all_ws = {i: [] for i in range(ring_count)}
    for name, run in runners.items():
        try:
            df = run(depth_map, ring_count, params, tunnel_id, base_dir, verbose=False)
            if len(df) != ring_count:
                continue
            w = weights[name]
            for _, row in df.iterrows():
                r = int(row["Ring"])
                if r in all_ys:
                    all_ys[r].append(float(row["Y"]))
                    all_ws[r].append(w)
        except Exception:
            continue

    rows = []
    for i in range(ring_count):
        ys = all_ys.get(i, [])
        ws = all_ws.get(i, [])
        if not ys:
            y_k = H / 2.0
            conf = 0.1
        else:
            ys = np.array(ys)
            ws = np.array(ws)
            order = np.argsort(ys)
            ys = ys[order]
            ws = ws[order]
            cumw = np.cumsum(ws)
            half = cumw[-1] / 2.0
            idx = np.searchsorted(cumw, half)
            if idx >= len(ys):
                idx = len(ys) - 1
            y_k = float(ys[idx])
            conf = min(1.0, 0.5 + 0.1 * len(ys))
        band_center = (i + 0.5) * ring_width
        rows.append({"Ring": i, "Type": "ensemble", "X": band_center, "Y": y_k, "Confidence": conf})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------
def run_k_detection(
    depth_map: np.ndarray,
    ring_count: int,
    method: str,
    params: Dict,
    tunnel_id: Optional[str] = None,
    base_dir: str = "data",
    verbose: bool = True,
    use_regulator: bool = True,
) -> pd.DataFrame:
    """Run the selected K detection method. Returns DataFrame with Ring, Type, X, Y, Confidence.
    If use_regulator is True (default), post-process with oblique-line geometry: even X, one K per ring, Y from lines."""
    if method == "dbscan":
        k_df = _run_dbscan(depth_map, ring_count, params, tunnel_id, base_dir, verbose)
    elif method == "groove_pair":
        k_df = _run_groove_pair(depth_map, ring_count, params, tunnel_id, base_dir, verbose)
    elif method == "banded":
        k_df = _run_banded(depth_map, ring_count, params, tunnel_id, base_dir, verbose)
    elif method == "edge_projection":
        k_df = _run_edge_projection(depth_map, ring_count, params, tunnel_id, base_dir, verbose)
    elif method == "gradient_direction":
        k_df = _run_gradient_direction(depth_map, ring_count, params, tunnel_id, base_dir, verbose)
    elif method == "local_hough":
        k_df = _run_local_hough(depth_map, ring_count, params, tunnel_id, base_dir, verbose)
    elif method == "ensemble":
        k_df = _run_ensemble(depth_map, ring_count, params, tunnel_id, base_dir, verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from {K_METHODS}")

    if use_regulator and len(k_df) >= ring_count:
        k_df = apply_k_regulator(
            k_df, depth_map, ring_count, params,
            tunnel_id=tunnel_id, base_dir=base_dir, verbose=verbose,
        )
    return k_df


def main():
    parser = argparse.ArgumentParser(description="Unified K detection (multi-method)")
    parser.add_argument("tunnel_id", help="e.g. 4-1")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--method", default="groove_pair", choices=K_METHODS)
    parser.add_argument("--output", default=None, help="Output CSV path (default: data/<tunnel>/detected_k_<method>.csv)")
    parser.add_argument("--align-gt", action="store_true", help="Align ring IDs to GT and save")
    args = parser.parse_args()

    tunnel_dir = os.path.join(args.data_dir, args.tunnel_id)
    depth_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    ring_path = os.path.join(tunnel_dir, "ring_count.txt")
    if not os.path.exists(depth_path):
        print(f"Missing {depth_path}")
        sys.exit(1)
    if not os.path.exists(ring_path):
        print(f"Missing {ring_path}")
        sys.exit(1)

    depth_map = np.load(depth_path)
    ring_count = int(open(ring_path).read())
    params, _ = _detection.load_parameters(args.tunnel_id, args.data_dir)
    if not params:
        params = {}

    k_df = run_k_detection(
        depth_map, ring_count, args.method, params,
        tunnel_id=args.tunnel_id, base_dir=args.data_dir, verbose=True,
    )
    print(f"Detected {len(k_df)} K positions")

    if args.align_gt:
        gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path)
            gt_k = gt[gt["Block"] == "K"][["Ring", "X", "Y"]].copy()
            H = depth_map.shape[0]
            aligned, dists = align_k_to_gt(k_df, gt_k, H)
            print(f"Aligned to GT. Per-ring distances (px): {[f'{d:.0f}' for d in dists]}")
            print(f"Mean distance: {np.mean(dists):.1f} px")
            k_df = aligned
        else:
            print("all_segments_gt.csv not found; skipping alignment")

    out_path = args.output or os.path.join(tunnel_dir, f"detected_k_{args.method}.csv")
    k_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

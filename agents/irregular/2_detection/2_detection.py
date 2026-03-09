"""
Irregular Tunnel Detection Pipeline

Detects oblique lines, finds K-block positions via combined DBSCAN + groove-pair
fusion, and expands K to all segment positions via grouped offsets.

Produces all_segments.csv (Ring, Block, X, Y, quality) for downstream segmentation.

Tunable parameters:
  Line detection (8): binary_threshold, hough_threshold, hough_min_length,
    hough_max_gap, angle_pos_min/max, angle_neg_min/max
  K detection (5): eps, k_expected_height_px, k_gap_tolerance_px,
    k_candidates_per_ring, groove_snap_px
  Expansion (14): stagger_groups, group_offsets (2 groups x 6 blocks = 12D),
    ring_offset, ring_spacing_px

Physical constants (tunnel_diameter, resolution) inherited from preprocessing.
"""

import os
import sys
import json
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import DBSCAN


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Tuple[Dict, bool]:
    """Load parameters from parameters_detection.json.

    Priority:
        1. agents/.../parameters/<tunnel_id>/parameters_detection.json
        2. data/<tunnel_id>/parameters_detection.json
        3. agents/.../parameters/sample/parameters_detection.json
        4. Empty dict (hardcoded defaults)
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_detection.json"

    if tunnel_id:
        params_path = os.path.join(script_dir, "parameters", tunnel_id, param_file)
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                return json.load(f), True

        tunnel_path = os.path.join(base_dir, tunnel_id, param_file)
        if os.path.exists(tunnel_path):
            with open(tunnel_path, 'r') as f:
                return json.load(f), True

    sample_path = os.path.join(script_dir, "parameters", "sample", param_file)
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            return json.load(f), True

    return {}, False


def get_param(params: Dict, key: str, default=None):
    """Get parameter value with default fallback."""
    return params.get(key, default)


# =============================================================================
# Default Parameters
# =============================================================================

DEFAULT_BINARY_THRESHOLD = 127
DEFAULT_HOUGH_OBLIQUE_THRESHOLD = 50
DEFAULT_ANGLE_POSITIVE_MIN = 6.0
DEFAULT_ANGLE_POSITIVE_MAX = 9.0
DEFAULT_ANGLE_NEGATIVE_MIN = -9.0
DEFAULT_ANGLE_NEGATIVE_MAX = -6.0
DEFAULT_HOUGH_VERTICAL_THRESHOLD = 500
DEFAULT_HOUGH_HORIZONTAL_THRESHOLD = 50
DEFAULT_HOUGH_HORIZONTAL_MIN_LENGTH = 100
DEFAULT_HOUGH_HORIZONTAL_MAX_GAP = 10
DEFAULT_HORIZONTAL_ANGLE_TOLERANCE = 1.0
DEFAULT_MERGE_DISTANCE_THRESHOLD = 3.0
DEFAULT_DILATION_KERNEL_SIZE = 3
DEFAULT_DILATION_ITERATIONS = 1
DEFAULT_HOUGH_OBLIQUE_MIN_LENGTH = 100
DEFAULT_HOUGH_OBLIQUE_MAX_GAP = 40
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150

FIXED_MERGE_CLOSE_THRESHOLD = 6.0

PREPROCESSING_PARAMS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "1_preprocessing", "parameters"
)


# =============================================================================
# Physical Constants (from preprocessing)
# =============================================================================

def load_preprocessing_params(tunnel_id: str, base_dir: str = "data") -> Dict:
    """Load preprocessing parameters for inherited physical constants."""
    for subdir in [tunnel_id, "sample"]:
        params_path = os.path.join(PREPROCESSING_PARAMS_DIR, subdir, "parameters_preprocessing.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                return json.load(f)

    tunnel_path = os.path.join(base_dir, tunnel_id, "parameters_preprocessing.json")
    if os.path.exists(tunnel_path):
        with open(tunnel_path, 'r') as f:
            return json.load(f)

    return {}


def calculate_segment_heights(tunnel_diameter: float) -> Tuple[float, float]:
    """K and AB block heights (mm) from tunnel diameter.

    K = circumference / 16, AB = 3 * K.
    """
    circumference_mm = np.pi * tunnel_diameter * 1000
    k_height_mm = circumference_mm / 16
    ab_height_mm = 3 * k_height_mm
    return k_height_mm, ab_height_mm


# =============================================================================
# Line Detection
# =============================================================================

def detect_lines(depth_map_outlier: np.ndarray, params: Dict) -> Dict:
    """Unified line detection from depth map.

    BO-tunable (8): binary_threshold, hough_threshold, hough_min_length,
    hough_max_gap, angle_pos_min/max, angle_neg_min/max.

    Remaining params (dilation, canny, horizontal/vertical Hough) read from
    params with DEFAULT_* fallbacks.
    """
    L, W = depth_map_outlier.shape

    binary_threshold = params.get('binary_threshold', 139)
    dilation_kernel_size = params.get('dilation_kernel_size', DEFAULT_DILATION_KERNEL_SIZE)
    dilation_iterations = params.get('dilation_iterations', DEFAULT_DILATION_ITERATIONS)
    canny_low = params.get('canny_low', DEFAULT_CANNY_LOW)
    canny_high = params.get('canny_high', DEFAULT_CANNY_HIGH)

    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, binary_threshold, 255, cv2.THRESH_BINARY)

    depth_valid = depth_map_outlier[~np.isnan(depth_map_outlier)]
    if len(depth_valid) > 0:
        depth_min, depth_max = depth_valid.min(), depth_valid.max()
        if depth_max > depth_min:
            out = np.zeros_like(depth_map_outlier, dtype=np.float64)
            valid = ~np.isnan(depth_map_outlier)
            out[valid] = (depth_map_outlier[valid] - depth_min) / (depth_max - depth_min) * 255
            depth_normalized = out.astype(np.uint8)

            canny_edges = cv2.Canny(depth_normalized, canny_low, canny_high)
            combined_edges = cv2.bitwise_or(binary_image, canny_edges)
        else:
            combined_edges = binary_image
    else:
        combined_edges = binary_image

    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=dilation_iterations)

    hough_threshold = params.get('hough_threshold', 37)
    hough_min_length = params.get('hough_min_length', 31)
    hough_max_gap = params.get('hough_max_gap', 133)

    lines_oblique = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_threshold,
        minLineLength=hough_min_length,
        maxLineGap=hough_max_gap
    )

    hough_horizontal_threshold = params.get('hough_horizontal_threshold', DEFAULT_HOUGH_HORIZONTAL_THRESHOLD)
    hough_horizontal_min_length = params.get('hough_horizontal_min_length', DEFAULT_HOUGH_HORIZONTAL_MIN_LENGTH)
    hough_horizontal_max_gap = params.get('hough_horizontal_max_gap', DEFAULT_HOUGH_HORIZONTAL_MAX_GAP)
    horizontal_angle_tolerance = params.get('horizontal_angle_tolerance', DEFAULT_HORIZONTAL_ANGLE_TOLERANCE)

    lines_horizontal = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_horizontal_threshold,
        minLineLength=hough_horizontal_min_length,
        maxLineGap=hough_horizontal_max_gap
    )

    hough_vertical_threshold = params.get('hough_vertical_threshold', DEFAULT_HOUGH_VERTICAL_THRESHOLD)
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, hough_vertical_threshold)
    if lines_vertical is not None:
        lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= W]

    angle_pos_min = params.get('angle_pos_min', 4.84)
    angle_pos_max = params.get('angle_pos_max', 13.55)
    angle_neg_min = params.get('angle_neg_min', -14.67)
    angle_neg_max = params.get('angle_neg_max', -5.82)

    positive_lines = []
    negative_lines = []
    horizontal_lines = []

    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))

            if angle_pos_min <= angle <= angle_pos_max:
                positive_lines.append(line[0])
            elif angle_neg_min <= angle <= angle_neg_max:
                negative_lines.append(line[0])

    if lines_horizontal is not None:
        for line in lines_horizontal:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -horizontal_angle_tolerance <= angle <= horizontal_angle_tolerance:
                horizontal_lines.append(line[0])

    merge_distance_threshold = params.get('merge_distance_threshold', DEFAULT_MERGE_DISTANCE_THRESHOLD)
    merged_vertical = []
    if lines_vertical is not None:
        lines_vert_2d = lines_vertical[:, 0]
        for rho, theta in lines_vert_2d:
            if abs(theta) <= 0.5 * np.pi / 180:
                x_pos = rho * np.cos(theta)
                merged = False
                for i, (mrho, mtheta) in enumerate(merged_vertical):
                    mx = mrho * np.cos(mtheta)
                    if abs(x_pos - mx) < merge_distance_threshold:
                        merged_vertical[i] = ((rho + mrho) / 2, (theta + mtheta) / 2)
                        merged = True
                        break
                if not merged:
                    merged_vertical.append((rho, theta))
        merged_vertical.sort(key=lambda l: l[0])

    return {
        'positive_lines': positive_lines,
        'negative_lines': negative_lines,
        'horizontal_lines': horizontal_lines,
        'vertical_lines': merged_vertical,
        'dilated_edges': dilated_edges,
        'image_height': L,
        'image_width': W
    }


# =============================================================================
# Geometric Helpers
# =============================================================================

def extend_line_to_bounds(x1, y1, x2, y2, W, L):
    """Extend a line segment to image boundaries."""
    if x2 == x1:
        return x1, 0, x2, L

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    points = []
    y_at_0 = intercept
    y_at_W = slope * W + intercept

    if 0 <= y_at_0 <= L:
        points.append((0, y_at_0))
    if 0 <= y_at_W <= L:
        points.append((W, y_at_W))

    if slope != 0:
        x_at_0 = -intercept / slope
        if 0 <= x_at_0 <= W:
            points.append((x_at_0, 0))
        x_at_L = (L - intercept) / slope
        if 0 <= x_at_L <= W:
            points.append((x_at_L, L))

    if len(points) >= 2:
        points.sort(key=lambda p: p[0])
        return points[0][0], points[0][1], points[-1][0], points[-1][1]

    return x1, y1, x2, y2


def find_line_intersections(positive_lines, negative_lines, W, L):
    """Find all intersections between positive and negative slope lines."""
    intersections = []

    for pos_line in positive_lines:
        x1, y1, x2, y2 = pos_line
        if x2 == x1:
            continue
        slope1 = (y2 - y1) / (x2 - x1)
        intercept1 = y1 - slope1 * x1

        for neg_line in negative_lines:
            x3, y3, x4, y4 = neg_line
            if x4 == x3:
                continue
            slope2 = (y4 - y3) / (x4 - x3)
            intercept2 = y3 - slope2 * x3

            if abs(slope1 - slope2) < 1e-6:
                continue

            x_int = (intercept2 - intercept1) / (slope1 - slope2)
            y_int = slope1 * x_int + intercept1

            if 0 <= x_int <= W and 0 <= y_int <= L:
                intersections.append((x_int, y_int))

    return intersections


def line_segment_vertical_intersection(vertical_x: float, segment: Tuple) -> Optional[float]:
    """Y coordinate where a line segment crosses vertical x."""
    x1, y1, x2, y2 = segment
    if x1 == x2:
        return None
    if min(x1, x2) <= vertical_x <= max(x1, x2):
        t = (vertical_x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)
    return None


def merge_close_points(points: List[float]) -> List[float]:
    """Merge Y-values within FIXED_MERGE_CLOSE_THRESHOLD."""
    if len(points) == 0:
        return []
    pts = np.array(points, dtype=np.float64)
    if len(pts) == 1:
        return [float(pts[0])]

    merged = []
    while len(pts) > 0:
        p = pts[0]
        close_mask = np.abs(pts - p) < FIXED_MERGE_CLOSE_THRESHOLD
        merged.append(float(np.mean(pts[close_mask])))
        pts = pts[~close_mask]
    return merged


# =============================================================================
# K-Position Detection: Banded Fallback
# =============================================================================

def calculate_k_positions_banded(
    line_data: Dict,
    ring_count: int,
    params: Dict
) -> pd.DataFrame:
    """K positions via evenly-spaced ring bands (fallback for combined method).

    X = band center, Y = median of oblique intersections in band.
    """
    L = line_data['image_height']
    W = line_data['image_width']

    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']

    extended_positive = [extend_line_to_bounds(*line, W, L) for line in positive_lines]
    extended_negative = [extend_line_to_bounds(*line, W, L) for line in negative_lines]
    intersections = find_line_intersections(extended_positive, extended_negative, W, L)

    ring_width = W / ring_count
    band_margin = ring_width * 0.6
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
            det_type = 'band_intersection'
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
                det_type = 'band_crossing'
            else:
                k_y = None
                conf = 0.0
                det_type = 'band_interpolated'

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
        if neighbors:
            k_y = float(np.mean(neighbors))
            conf = 0.2
        else:
            k_y = L / 2.0
            conf = 0.1
        band_ys[i] = (k_y, conf, 'band_interpolated', band_center)

    k_positions = []
    for i in range(ring_count):
        k_y, conf, det_type, band_center = band_ys[i]
        k_positions.append((det_type, band_center, k_y, conf))

    k_positions.sort(key=lambda p: p[1])
    df = pd.DataFrame(k_positions, columns=['Type', 'X', 'Y', 'Confidence'])
    return df.sort_values(by='X').reset_index(drop=True)


# =============================================================================
# K-Position Detection: Combined (DBSCAN + Groove-Pair Fusion)
# =============================================================================

def calculate_k_positions_combined(
    line_data: Dict,
    ring_count: int,
    k_height_mm: float,
    resolution: float,
    params: Dict,
) -> pd.DataFrame:
    """Combined K detection: fuses DBSCAN (coverage) + groove-pair (precision).

    1. DBSCAN: cluster pos-neg line intersections, assign to ring bands.
    2. Groove-pair: for each ring band, find pos-neg crossing pairs with
       gap ~ k_expected_height_px, keep top candidates.
    3. Per-ring fusion: score all candidates via groove alignment
       (expand with offsets, count groove crossings), select best.

    Returns DataFrame with columns Type, X, Y, Confidence.
    """
    L = line_data['image_height']
    W = line_data['image_width']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']

    ring_offset = params.get('ring_offset', W / (2 * ring_count))
    ring_spacing_px = params.get('ring_spacing_px', W / ring_count)
    half_width = abs(ring_spacing_px) * 0.5

    eps = params.get('eps', 0.07)
    k_expected_height_px = params.get(
        'k_expected_height_px',
        (k_height_mm / 1000.0) / resolution / 2.0
    )
    k_gap_tolerance_px = params.get('k_gap_tolerance_px', 150.0)
    k_candidates_per_ring = int(params.get('k_candidates_per_ring', 8))
    groove_snap_px = params.get('groove_snap_px', 60.0)

    stagger_groups = params.get('stagger_groups', {})
    group_offsets = params.get('group_offsets', {})
    ring_to_group: Dict[int, str] = {}
    default_group = list(stagger_groups.keys())[0] if stagger_groups else "A"
    for grp, ring_list in stagger_groups.items():
        for r in ring_list:
            ring_to_group[r] = grp
    expansion_blocks = _derive_expansion_blocks(group_offsets)

    def _line_crossing_y(x1, y1, x2, y2, x):
        if x2 == x1:
            return None
        t = (x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    def _wrap_dist_y(a, b):
        d = abs(a - b)
        return min(d, L - d)

    def _wrap_midpoint(a, b):
        if abs(a - b) <= L / 2:
            return (a + b) / 2.0
        return ((a + b) / 2.0 + L / 2.0) % L

    def _groove_crossings_at_x(x_center):
        crossings = []
        for x1, y1, x2, y2 in positive_lines + negative_lines:
            y_c = _line_crossing_y(x1, y1, x2, y2, x_center)
            if y_c is not None and 0 <= y_c <= L:
                crossings.append(y_c)
        return sorted(crossings)

    def _groove_alignment_score(k_y, ring_idx, groove_ys):
        group = ring_to_group.get(ring_idx, default_group)
        total = 0.0
        for block in expansion_blocks:
            key = f"{group}_{block}"
            offset = group_offsets.get(key, 0.0)
            block_y = (k_y + offset) % L
            min_dist = min((_wrap_dist_y(block_y, gy) for gy in groove_ys),
                           default=groove_snap_px + 1)
            if min_dist <= groove_snap_px:
                total += 1.0 + (groove_snap_px - min_dist) / groove_snap_px
        return total

    # --- DBSCAN path ---
    extended_positive = [extend_line_to_bounds(*line, W, L) for line in positive_lines]
    extended_negative = [extend_line_to_bounds(*line, W, L) for line in negative_lines]
    intersections = find_line_intersections(extended_positive, extended_negative, W, L)

    dbscan_candidates = {}
    if len(intersections) > 0:
        intersection_array = np.array(intersections)
        x_normalized = intersection_array[:, 0] / W
        y_normalized = intersection_array[:, 1] / L
        features = np.column_stack([x_normalized, y_normalized])

        clustering = DBSCAN(eps=eps, min_samples=1).fit(features)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}

        for label in unique_labels:
            cluster_points = intersection_array[labels == label]
            cluster_x = np.mean(cluster_points[:, 0])
            cluster_y = np.mean(cluster_points[:, 1])
            conf = min(1.0, 0.5 + 0.05 * len(cluster_points))

            ring_idx = int(round((cluster_x - ring_offset) / ring_spacing_px))
            ring_idx = max(0, min(ring_count - 1, ring_idx))

            if ring_idx not in dbscan_candidates:
                dbscan_candidates[ring_idx] = []
            dbscan_candidates[ring_idx].append((cluster_x, cluster_y, conf))

    # --- Groove-pair path ---
    groove_pair_candidates = {}
    for i in range(ring_count):
        band_center = ring_offset + i * ring_spacing_px

        pos_crossings = []
        for x1, y1, x2, y2 in positive_lines:
            mid_x = (x1 + x2) / 2.0
            if band_center - half_width <= mid_x <= band_center + half_width:
                y_c = _line_crossing_y(x1, y1, x2, y2, band_center)
                if y_c is not None and 0 <= y_c <= L:
                    pos_crossings.append(y_c)

        neg_crossings = []
        for x1, y1, x2, y2 in negative_lines:
            mid_x = (x1 + x2) / 2.0
            if band_center - half_width <= mid_x <= band_center + half_width:
                y_c = _line_crossing_y(x1, y1, x2, y2, band_center)
                if y_c is not None and 0 <= y_c <= L:
                    neg_crossings.append(y_c)

        candidates = []
        for py in pos_crossings:
            for ny in neg_crossings:
                gap = _wrap_dist_y(py, ny)
                gap_err = abs(gap - k_expected_height_px)
                if gap_err > k_gap_tolerance_px:
                    continue
                mid = _wrap_midpoint(py, ny)
                candidates.append((gap_err, gap, mid))

        candidates.sort(key=lambda c: c[0])
        deduped = []
        for gap_err, gap, mid in candidates:
            if any(_wrap_dist_y(mid, em) < 30 for _, _, em in deduped):
                continue
            deduped.append((gap_err, gap, mid))
            if len(deduped) >= k_candidates_per_ring:
                break
        groove_pair_candidates[i] = deduped

    # --- Per-ring fusion ---
    k_positions = []
    groove_scores = []
    banded_df = calculate_k_positions_banded(line_data, ring_count, params)

    for i in range(ring_count):
        band_center = ring_offset + i * ring_spacing_px
        groove_ys = _groove_crossings_at_x(band_center)

        all_candidates = []
        if i in dbscan_candidates:
            for x, y, conf in dbscan_candidates[i]:
                all_candidates.append(('dbscan', x, y, conf))
        if i in groove_pair_candidates:
            for gap_err, gap, mid in groove_pair_candidates[i]:
                all_candidates.append(('groove_pair', band_center, mid, 0.8))

        if not all_candidates:
            row = banded_df.iloc[i]
            y_k = row['Y']
            conf = float(row['Confidence']) * 0.3
            det_type = 'combined_fallback'
            groove_scores.append(0.0)
        else:
            best_candidate = None
            best_score = -float('inf')
            best_groove = 0.0

            for det_type, x, y, base_conf in all_candidates:
                groove = _groove_alignment_score(y, i, groove_ys)
                if det_type == 'groove_pair':
                    gap_err = next((c[0] for c in groove_pair_candidates[i] if abs(c[2] - y) < 1), 0)
                    gap_penalty = gap_err / k_gap_tolerance_px
                    combined = groove - gap_penalty
                else:
                    combined = groove

                if combined > best_score:
                    best_score = combined
                    best_candidate = (det_type, x, y, base_conf)
                    best_groove = groove

            det_type, x, y, base_conf = best_candidate
            y_k = y
            conf = min(1.0, base_conf + 0.04 * best_groove)
            groove_scores.append(best_groove)

        k_positions.append((det_type, band_center, y_k, conf))

    groove_total = sum(groove_scores)
    groove_max = 2.0 * len(expansion_blocks) * ring_count
    print(f"  Groove alignment: {groove_total:.1f}/{groove_max:.0f} "
          f"({groove_total / groove_max * 100:.1f}%)")

    df = pd.DataFrame(k_positions, columns=['Type', 'X', 'Y', 'Confidence'])
    df.attrs['groove_alignment_total'] = float(groove_total)
    df.attrs['groove_alignment_max'] = float(groove_max)
    df.attrs['groove_alignment_pct'] = (
        groove_total / groove_max * 100 if groove_max > 0 else 0.0
    )
    df.attrs['groove_scores_per_ring'] = [float(s) for s in groove_scores]
    return df.sort_values(by='X').reset_index(drop=True)


# =============================================================================
# Segment Expansion: K → All Segments
# =============================================================================

DEFAULT_EXPANSION_BLOCKS_7 = ['B1', 'B2', 'A1', 'A2', 'A3', 'A4']
DEFAULT_EXPANSION_BLOCKS_6 = ['B1', 'B2', 'A1', 'A2', 'A3']


def _derive_expansion_blocks(group_offsets: Dict[str, float], segment_count: int = None) -> List[str]:
    """Derive expansion block names from group_offsets keys or segment_count.

    group_offsets keys have the form 'GroupName_BlockName'.
    Falls back to defaults based on segment_count.
    """
    if group_offsets:
        blocks = sorted({k.split('_', 1)[1] for k in group_offsets if '_' in k})
        if blocks:
            return blocks
    if segment_count is not None and segment_count <= 6:
        return list(DEFAULT_EXPANSION_BLOCKS_6)
    return list(DEFAULT_EXPANSION_BLOCKS_7)


def expand_k_with_grouped_offsets(
    k_positions: pd.DataFrame,
    img_height: int,
    stagger_groups: Dict[str, list],
    group_offsets: Dict[str, float],
    expansion_blocks: List[str] = None,
) -> pd.DataFrame:
    """Expand K positions to all segments via grouped offsets.

    Each ring belongs to a stagger group. All rings in the same group share
    Y-offsets from K. BO-tunable.
    """
    if expansion_blocks is None:
        expansion_blocks = _derive_expansion_blocks(group_offsets)

    ring_to_group = {}
    default_group = list(stagger_groups.keys())[0] if stagger_groups else "A"
    for group_name, ring_list in stagger_groups.items():
        for r in ring_list:
            ring_to_group[r] = group_name

    rows = []
    for ring_idx, (_, k_row) in enumerate(k_positions.iterrows()):
        k_x = float(k_row['X'])
        k_y = float(k_row['Y'])
        quality = float(k_row.get('Confidence', 1.0))

        rows.append({
            'Ring': ring_idx, 'Block': 'K',
            'X': k_x, 'Y': k_y % img_height, 'quality': quality,
        })

        group = ring_to_group.get(ring_idx, default_group)
        for block in expansion_blocks:
            key = f"{group}_{block}"
            offset = group_offsets.get(key, 0.0)
            y = (k_y + offset) % img_height
            if y < 0:
                y += img_height
            rows.append({
                'Ring': ring_idx, 'Block': block,
                'X': k_x, 'Y': round(y, 1), 'quality': quality,
            })

    return pd.DataFrame(rows, columns=['Ring', 'Block', 'X', 'Y', 'quality'])


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(
    line_data: Dict,
    k_positions: pd.DataFrame,
    tunnel_dir: str,
    all_segments: pd.DataFrame = None,
) -> None:
    """Save visualization of detected lines and segment positions."""
    dilated_edges = line_data['dilated_edges']
    L, W = line_data['image_height'], line_data['image_width']

    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

    for x1, y1, x2, y2 in line_data['positive_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
    for x1, y1, x2, y2 in line_data['negative_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    for x1, y1, x2, y2 in line_data['horizontal_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

    for _, row in k_positions.iterrows():
        cv2.circle(output_image, (int(row['X']), int(row['Y'])), 8, (0, 255, 255), -1)
        cv2.line(output_image, (int(row['X']), 0), (int(row['X']), L), (255, 0, 255), 1)

    block_colors = {
        'B1': (255, 165, 0), 'A1': (0, 200, 200), 'A2': (200, 0, 200),
        'A3': (100, 255, 100), 'A4': (100, 100, 255), 'B2': (255, 100, 100),
    }
    if all_segments is not None:
        for _, row in all_segments.iterrows():
            if row['Block'] == 'K':
                continue
            color = block_colors.get(row['Block'], (200, 200, 200))
            cv2.circle(output_image, (int(row['X']), int(row['Y'])), 5, color, -1)

    plt.figure(figsize=(16, 8))
    plt.imshow(output_image)
    plt.title('Detection Results')
    plt.savefig(os.path.join(tunnel_dir, 'detected_lines.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def run_detection(tunnel_id: str, base_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the detection pipeline: lines → K positions (combined) → all segments.

    Returns (k_positions, all_segments) DataFrames.
    """
    print(f"Detection Pipeline: {tunnel_id}")

    params, params_loaded = load_parameters(tunnel_id, base_dir)

    preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
    tunnel_diameter = preprocessing_params.get('tunnel_diameter', 5.5)
    resolution = preprocessing_params.get('depth_map_resolution', 0.005)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)

    tunnel_dir = os.path.join(base_dir, tunnel_id)

    depth_map_outlier_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_map_outlier_path):
        raise FileNotFoundError(f"depth_map_outlier.npy not found. Run preprocessing first.")

    depth_map_outlier = np.load(depth_map_outlier_path)
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    L, W = depth_map_outlier.shape

    print(f"  Lines: detecting (image {L}x{W})...")
    line_data = detect_lines(depth_map_outlier, params)
    print(f"  Lines: +{len(line_data['positive_lines'])} -{len(line_data['negative_lines'])} "
          f"H{len(line_data['horizontal_lines'])} V{len(line_data['vertical_lines'])}")

    print(f"  K positions: combined detection (ring_count={ring_count})...")
    k_positions = calculate_k_positions_combined(
        line_data, ring_count, k_height_mm, resolution, params
    )
    print(f"  K positions: {len(k_positions)} found, "
          f"types={k_positions['Type'].value_counts().to_dict()}")

    reverse_ring_order = params.get('reverse_ring_order', False)
    if reverse_ring_order:
        k_positions = k_positions.iloc[::-1].reset_index(drop=True)

    k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)

    groove_meta = {
        'groove_alignment_total': k_positions.attrs.get('groove_alignment_total', None),
        'groove_alignment_max': k_positions.attrs.get('groove_alignment_max', None),
        'groove_alignment_pct': k_positions.attrs.get('groove_alignment_pct', None),
    }
    with open(os.path.join(tunnel_dir, 'groove_alignment.json'), 'w') as f:
        json.dump(groove_meta, f, indent=2)

    print(f"  Expanding K → all segments...")
    n_rings = len(k_positions)
    stagger_groups = params.get('stagger_groups', {"A": list(range(n_rings))})
    group_offsets = params.get('group_offsets', {})
    all_segments = expand_k_with_grouped_offsets(
        k_positions, img_height=L,
        stagger_groups=stagger_groups, group_offsets=group_offsets,
    )

    output_filename = params.get('output_filename', 'all_segments.csv')
    all_segments.to_csv(os.path.join(tunnel_dir, output_filename), index=False)
    print(f"  Segments: {len(k_positions)} K → {len(all_segments)} total")

    visualize_detection(line_data, k_positions, tunnel_dir, all_segments=all_segments)
    print(f"  Saved: detected.csv, {output_filename}, detected_lines.png")

    return k_positions, all_segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Irregular tunnel detection pipeline")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()

    run_detection(args.tunnel_id, base_dir=args.data_dir)

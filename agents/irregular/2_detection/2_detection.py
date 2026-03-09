"""
Irregular Tunnel Detection Pipeline

Detects oblique groove lines in the depth map, finds K-block positions,
and expands to all segment positions via per-ring offsets.

Outputs:
  - all_segments.csv (Ring, Block, X, Y, quality) — segment centroids
  - boundaries_per_ring.json — boundary positions for downstream segmentation
  - detected_lines.png — visualization

Detection modes (set via parameters_detection.json):
  - k_and_offsets (default): detect K via groove pairs, expand with per-ring offsets
  - groove_slots: direct groove-based slot detection
  - combined (legacy): DBSCAN + groove-pair fusion
"""

import os
import json
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
        3. Empty dict (hardcoded defaults)
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

    return {}, False


# =============================================================================
# Defaults (safe-fixed values, overridable via params JSON)
# =============================================================================

DEFAULT_K_EXPECTED_HEIGHT_PX = 300
DEFAULT_DILATION_KERNEL_SIZE = 3
DEFAULT_DILATION_ITERATIONS = 1
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150
DEFAULT_HOUGH_HORIZONTAL_THRESHOLD = 50
DEFAULT_HOUGH_HORIZONTAL_MIN_LENGTH = 100
DEFAULT_HOUGH_HORIZONTAL_MAX_GAP = 10
DEFAULT_HORIZONTAL_ANGLE_TOLERANCE = 1.0
DEFAULT_HOUGH_VERTICAL_THRESHOLD = 500
DEFAULT_MERGE_DISTANCE_THRESHOLD = 3.0
FIXED_MERGE_CLOSE_THRESHOLD = 6.0

PREPROCESSING_PARAMS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "1_preprocessing", "parameters"
)


# =============================================================================
# Physical Constants (from preprocessing)
# =============================================================================

def load_preprocessing_params(tunnel_id: str, base_dir: str = "data") -> Dict:
    """Load preprocessing parameters for inherited physical constants."""
    params_path = os.path.join(PREPROCESSING_PARAMS_DIR, tunnel_id, "parameters_preprocessing.json")
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
# Groove-Based Slot Detection
# =============================================================================

def _grooves_to_slots(
    groove_ys: List[float], circumference: float
) -> List[Dict]:
    """Convert sorted groove positions to circular slots.

    Returns list of dicts with keys: start, end, height, centroid.
    Heights sum to *circumference* by construction.
    """
    n = len(groove_ys)
    slots = []
    for i in range(n):
        start = groove_ys[i]
        end = groove_ys[(i + 1) % n]
        if end > start:
            height = end - start
            centroid = (start + end) / 2.0
        else:
            height = (end + circumference) - start
            centroid = (start + height / 2.0) % circumference
        slots.append({
            'start': start, 'end': end,
            'height': height, 'centroid': centroid,
        })
    return slots


def _identify_k_slot(
    slots: List[Dict], k_expected_height_px: float
) -> int:
    """Identify the K slot using engineering constraints.

    K = the slot whose height is closest to k_expected_height_px AND whose
    both circular neighbors are larger than itself.  When multiple candidates
    tie, prefer the one forming the best [large, SMALL, large] triplet.
    """
    n = len(slots)
    best_score = float('inf')
    best_idx = 0

    for i in range(n):
        h = slots[i]['height']
        prev_h = slots[(i - 1) % n]['height']
        next_h = slots[(i + 1) % n]['height']

        h_err = abs(h - k_expected_height_px) / max(k_expected_height_px, 1)

        neighbor_penalty = 0.0
        if prev_h <= h:
            neighbor_penalty += 1.0
        if next_h <= h:
            neighbor_penalty += 1.0

        avg_neighbor = (prev_h + next_h) / 2.0
        triplet_ratio = h / max(avg_neighbor, 1)

        score = h_err + neighbor_penalty * 2.0 + triplet_ratio
        if score < best_score:
            best_score = score
            best_idx = i

    return best_idx


def _label_slots(
    slots: List[Dict], k_idx: int, segment_count: int
) -> Dict[int, str]:
    """Assign block labels to all slots given the K slot index.

    Labeling order from K: B1 (prev), B2 (next), then A1..A4 continuing
    clockwise from B2 until wrapping back to B1.
    """
    b1_idx = (k_idx - 1) % segment_count
    b2_idx = (k_idx + 1) % segment_count

    labels = {k_idx: 'K', b1_idx: 'B1', b2_idx: 'B2'}

    a_names = ['A1', 'A2', 'A3', 'A4']
    idx = (b2_idx + 1) % segment_count
    a_count = 0
    while idx != b1_idx and a_count < len(a_names):
        labels[idx] = a_names[a_count]
        a_count += 1
        idx = (idx + 1) % segment_count

    return labels


def _find_density_peaks(
    ys: List[float], circumference: float, bandwidth: float, n_peaks: int,
    min_spacing: float,
) -> List[float]:
    """Find top-n peaks in 1D circular kernel density of Y values.

    Uses a histogram approximation with wrap-around handling.
    Returns sorted peak Y positions.
    """
    if len(ys) == 0:
        return [i * circumference / n_peaks for i in range(n_peaks)]

    n_bins = int(circumference / bandwidth) + 1
    counts = np.zeros(n_bins, dtype=np.float64)
    for y in ys:
        b = int(y / bandwidth) % n_bins
        counts[b] += 1.0

    kernel_half = max(1, int(bandwidth / (circumference / n_bins)))
    smoothed = np.zeros_like(counts)
    for i in range(n_bins):
        for d in range(-kernel_half, kernel_half + 1):
            smoothed[i] += counts[(i + d) % n_bins]

    peaks = []
    for i in range(n_bins):
        prev_val = smoothed[(i - 1) % n_bins]
        next_val = smoothed[(i + 1) % n_bins]
        if smoothed[i] > prev_val and smoothed[i] >= next_val and smoothed[i] > 0:
            peak_y = (i + 0.5) * bandwidth
            if peak_y >= circumference:
                peak_y -= circumference
            peaks.append((peak_y, smoothed[i]))

    peaks.sort(key=lambda x: -x[1])

    def _circ_dist(a, b):
        d = abs(a - b)
        return min(d, circumference - d)

    selected: List[float] = []
    for y, strength in peaks:
        if len(selected) >= n_peaks:
            break
        if all(_circ_dist(y, s) >= min_spacing for s in selected):
            selected.append(y)

    while len(selected) < n_peaks:
        sel_sorted = sorted(selected) if selected else []
        if len(sel_sorted) == 0:
            selected.append(0.0)
            continue
        max_gap = 0
        max_i = 0
        for i in range(len(sel_sorted)):
            j = (i + 1) % len(sel_sorted)
            gap = (sel_sorted[j] - sel_sorted[i]) if j != 0 else \
                  (sel_sorted[0] + circumference - sel_sorted[-1])
            if gap > max_gap:
                max_gap = gap
                max_i = i
        j = (max_i + 1) % len(sel_sorted)
        if j == 0:
            new_y = ((sel_sorted[max_i] + sel_sorted[0] + circumference) / 2.0) % circumference
        else:
            new_y = (sel_sorted[max_i] + sel_sorted[j]) / 2.0
        selected.append(new_y)

    return sorted(selected[:n_peaks])


def detect_k_groove_pair(
    line_data: Dict,
    ring_count: int,
    params: Dict,
) -> pd.DataFrame:
    """Detect K Y position per ring via groove-pair crossing.

    For each ring band:
      1. Find where pos/neg line segments cross the ring center vertical.
      2. Merge nearby crossings.
      3. Determine K_Y from the best pos+neg pair whose gap matches
         k_expected_height_px. Falls back to single-crossing estimate
         or neighbor propagation.
    """
    L = line_data['image_height']
    W = line_data['image_width']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']

    k_expected_height_px = params.get('k_expected_height_px', 500.0)
    merge_threshold = params.get('k_merge_threshold_px', 6.0)

    def _segment_crossing_y(segment, vertical_x):
        x1, y1, x2, y2 = segment
        if x1 == x2:
            return None
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            t = (vertical_x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
        return None

    def _merge_close(points, threshold):
        if len(points) == 0:
            return []
        arr = np.array(sorted(points))
        merged = []
        while len(arr) > 0:
            p = arr[0]
            close = np.abs(arr - p) < threshold
            merged.append(float(np.mean(arr[close])))
            arr = arr[~close]
        return merged

    rows = []
    ring_spacing = W / ring_count
    gap_tolerance = k_expected_height_px * 0.3

    for ring_idx in range(ring_count):
        ring_center_x = (ring_idx + 0.5) * ring_spacing

        pos_crossings = []
        for seg in positive_lines:
            y = _segment_crossing_y(seg, ring_center_x)
            if y is not None:
                pos_crossings.append(y)

        neg_crossings = []
        for seg in negative_lines:
            y = _segment_crossing_y(seg, ring_center_x)
            if y is not None:
                neg_crossings.append(y)

        pos_merged = _merge_close(pos_crossings, merge_threshold)
        neg_merged = _merge_close(neg_crossings, merge_threshold)

        k_y = None
        det_type = 'none'
        conf = 0.0

        if pos_merged and neg_merged:
            best_gap_err = float('inf')
            for py in pos_merged:
                for ny in neg_merged:
                    d = abs(py - ny)
                    d_circ = min(d, L - d)
                    gap_err = abs(d_circ - k_expected_height_px)
                    if gap_err < best_gap_err and gap_err <= gap_tolerance:
                        best_gap_err = gap_err
                        if d <= L / 2:
                            k_y = (py + ny) / 2.0
                        else:
                            k_y = ((py + ny) / 2.0 + L / 2.0) % L
                        det_type = 'midpoint'
                        conf = max(0.6, 1.0 - gap_err / k_expected_height_px)

        if k_y is None and pos_merged:
            k_y = (pos_merged[0] - 0.5 * k_expected_height_px) % L
            det_type = 'pos_only'
            conf = 0.4

        if k_y is None and neg_merged:
            k_y = (neg_merged[0] + 0.5 * k_expected_height_px) % L
            det_type = 'neg_only'
            conf = 0.4

        rows.append((det_type, ring_center_x, k_y, conf))

    for i in range(len(rows)):
        if rows[i][2] is None:
            for nb in [i - 1, i + 1, i - 2, i + 2, i - 3, i + 3]:
                if 0 <= nb < len(rows) and rows[nb][2] is not None:
                    rows[i] = ('propagate', rows[i][1], rows[nb][2], 0.1)
                    break
            if rows[i][2] is None:
                rows[i] = ('default', rows[i][1], L / 2.0, 0.05)

    return pd.DataFrame(rows, columns=['Type', 'X', 'Y', 'Confidence'])


def detect_all_segments_from_grooves(
    line_data: Dict,
    ring_count: int,
    params: Dict,
) -> pd.DataFrame:
    """Groove-based slot detection — finds all segment positions directly.

    For each ring band: collect groove line crossings, find density peaks,
    form circular slots, identify K by height constraint, label all blocks.
    Slot heights sum to image height (circumference tiling) by construction.
    """
    L = line_data['image_height']
    W = line_data['image_width']
    all_lines = line_data['positive_lines'] + line_data['negative_lines']

    ring_offset = params.get('ring_offset', W / (2 * ring_count))
    ring_spacing_px = params.get('ring_spacing_px', W / ring_count)
    k_expected_height_px = params.get('k_expected_height_px', DEFAULT_K_EXPECTED_HEIGHT_PX)
    groove_merge_px = params.get('groove_merge_px', 40.0)
    segment_count = params.get('segment_count', 7)

    def _crossing_y_segment(x1, y1, x2, y2, x_target, margin=0.15):
        if x2 == x1:
            return None
        t = (x_target - x1) / (x2 - x1)
        if t < -margin or t > 1.0 + margin:
            return None
        y = y1 + t * (y2 - y1)
        return y if 0 <= y <= L else None

    all_rows = []
    ring_stats = []

    for ring_idx in range(ring_count):
        band_center = ring_offset + ring_idx * ring_spacing_px

        crossing_ys = []
        for x1, y1, x2, y2 in all_lines:
            y_c = _crossing_y_segment(x1, y1, x2, y2, band_center)
            if y_c is not None:
                crossing_ys.append(y_c)

        min_spacing = max(k_expected_height_px * 0.3, 90.0)
        groove_ys = _find_density_peaks(
            crossing_ys, L, groove_merge_px, segment_count, min_spacing
        )
        slots = _grooves_to_slots(groove_ys, L)

        k_idx = _identify_k_slot(slots, k_expected_height_px)
        labels = _label_slots(slots, k_idx, segment_count)

        ring_stats.append({
            'crossings': len(crossing_ys),
            'k_height': slots[k_idx]['height'],
        })

        for slot_idx, slot in enumerate(slots):
            block = labels.get(slot_idx, f'X{slot_idx}')
            all_rows.append({
                'Ring': ring_idx,
                'Block': block,
                'X': round(band_center, 1),
                'Y': round(slot['centroid'], 1),
                'quality': 1.0,
            })

    crossing_counts = [s['crossings'] for s in ring_stats]
    k_heights = [s['k_height'] for s in ring_stats]
    print(f"  Groove slots: crossings {min(crossing_counts)}-{max(crossing_counts)}, "
          f"bandwidth {groove_merge_px}px, target {segment_count}")
    print(f"  K heights: {', '.join(f'{h:.0f}' for h in k_heights)} px "
          f"(expected ~{k_expected_height_px:.0f})")

    return pd.DataFrame(all_rows, columns=['Ring', 'Block', 'X', 'Y', 'quality'])


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


def expand_k_with_per_ring_offsets(
    k_positions: pd.DataFrame,
    img_height: int,
    per_ring_offsets: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Expand K positions to all segments using per-ring individual offsets.

    per_ring_offsets: {"ring_idx": {"block_name": dy, ...}, ...}
    Each offset is a signed circular distance from K_Y to the block centroid.
    """
    rows = []
    for ring_idx, (_, k_row) in enumerate(k_positions.iterrows()):
        k_x = float(k_row['X'])
        k_y = float(k_row['Y'])
        quality = float(k_row.get('Confidence', 1.0))

        rows.append({
            'Ring': ring_idx, 'Block': 'K',
            'X': k_x, 'Y': k_y % img_height, 'quality': quality,
        })

        ring_offsets = per_ring_offsets.get(str(ring_idx), {})
        for block, offset in ring_offsets.items():
            y = (k_y + offset) % img_height
            if y < 0:
                y += img_height
            rows.append({
                'Ring': ring_idx, 'Block': block,
                'X': k_x, 'Y': round(y, 1), 'quality': quality,
            })

    return pd.DataFrame(rows, columns=['Ring', 'Block', 'X', 'Y', 'quality'])


# =============================================================================
# Centroid → Boundary Conversion (warm start for boundary-based segmentation)
# =============================================================================

def centroids_to_boundaries(
    all_segments: pd.DataFrame, img_height: int
) -> Dict[str, list]:
    """Convert detected centroids to boundary positions per ring.

    For each ring, sorts blocks by centroid Y, then places each boundary at
    the circular midpoint between adjacent centroids. The boundary at position
    i marks the start of block i (the block whose centroid follows).

    Returns dict suitable for parameters_segmentation.json:
        {"0": [{"y": 528, "block": "A2"}, ...], "1": [...], ...}
    """
    result = {}
    for ring_idx in sorted(all_segments['Ring'].unique()):
        ring_segs = all_segments[all_segments['Ring'] == ring_idx].copy()
        ring_segs = ring_segs.sort_values('Y').reset_index(drop=True)

        n = len(ring_segs)
        if n == 0:
            continue

        ys = ring_segs['Y'].values.astype(float)
        blocks = ring_segs['Block'].values

        boundaries = []
        for i in range(n):
            prev_y = ys[(i - 1) % n]
            curr_y = ys[i]
            d = curr_y - prev_y
            if d < 0:
                d += img_height
            mid = (prev_y + d / 2.0) % img_height
            boundaries.append({
                'y': round(float(mid), 1),
                'block': str(blocks[i]),
            })

        result[str(ring_idx)] = boundaries
    return result


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
    """Run the detection pipeline. Returns (k_positions, all_segments) DataFrames."""
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

    detection_mode = params.get('detection_mode', 'k_and_offsets')
    k_y_override = params.get('k_y_positions', None)
    per_ring_offsets = params.get('per_ring_offsets', None)

    if detection_mode == 'k_and_offsets':
        ring_spacing = W / ring_count

        if k_y_override is not None and len(k_y_override) == ring_count:
            print(f"  K positions: using k_y_positions override ({ring_count} rings)")
            rows = []
            for i in range(ring_count):
                band_x = (i + 0.5) * ring_spacing
                rows.append(('k_override', band_x, k_y_override[i], 1.0))
            k_positions = pd.DataFrame(rows, columns=['Type', 'X', 'Y', 'Confidence'])
        else:
            print(f"  K positions: groove-pair detection ({ring_count} rings)...")
            k_positions = detect_k_groove_pair(
                line_data, ring_count, params
            )
            print(f"  K positions: {len(k_positions)} found")

        k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)

        if per_ring_offsets is not None:
            print(f"  Expanding K → all segments (per-ring offsets, "
                  f"{len(per_ring_offsets)} rings)...")
            all_segments = expand_k_with_per_ring_offsets(
                k_positions, img_height=L,
                per_ring_offsets=per_ring_offsets,
            )
        else:
            print(f"  WARNING: No per_ring_offsets found, using group_offsets fallback")
            n_rings = len(k_positions)
            stagger_groups = params.get('stagger_groups', {"A": list(range(n_rings))})
            group_offsets = params.get('group_offsets', {})
            all_segments = expand_k_with_grouped_offsets(
                k_positions, img_height=L,
                stagger_groups=stagger_groups, group_offsets=group_offsets,
            )

    elif detection_mode == 'groove_slots':
        print(f"  Detection mode: groove_slots (direct slot detection)")
        all_segments = detect_all_segments_from_grooves(
            line_data, ring_count, params
        )
        k_rows = all_segments[all_segments['Block'] == 'K']
        k_positions = pd.DataFrame({
            'Type': 'groove_slot',
            'X': k_rows['X'].values,
            'Y': k_rows['Y'].values,
            'Confidence': k_rows['quality'].values,
        })
        k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)

    else:
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

        print(f"  Expanding K → all segments...")
        n_rings = len(k_positions)
        stagger_groups = params.get('stagger_groups', {"A": list(range(n_rings))})
        group_offsets = params.get('group_offsets', {})
        all_segments = expand_k_with_grouped_offsets(
            k_positions, img_height=L,
            stagger_groups=stagger_groups, group_offsets=group_offsets,
        )

    groove_meta = {
        'groove_alignment_total': k_positions.attrs.get('groove_alignment_total', None),
        'groove_alignment_max': k_positions.attrs.get('groove_alignment_max', None),
        'groove_alignment_pct': k_positions.attrs.get('groove_alignment_pct', None),
    }
    with open(os.path.join(tunnel_dir, 'groove_alignment.json'), 'w') as f:
        json.dump(groove_meta, f, indent=2)

    output_filename = params.get('output_filename', 'all_segments.csv')
    all_segments.to_csv(os.path.join(tunnel_dir, output_filename), index=False)
    print(f"  Segments: {len(all_segments)} total (mode={detection_mode})")

    boundaries = centroids_to_boundaries(all_segments, img_height=L)
    boundaries_path = os.path.join(tunnel_dir, 'boundaries_per_ring.json')
    with open(boundaries_path, 'w') as f:
        json.dump(boundaries, f, indent=2)
    print(f"  Boundaries: {len(boundaries)} rings → {boundaries_path}")

    visualize_detection(line_data, k_positions, tunnel_dir, all_segments=all_segments)
    print(f"  Saved: detected.csv, {output_filename}, boundaries_per_ring.json, detected_lines.png")

    return k_positions, all_segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Irregular tunnel detection pipeline")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()

    run_detection(args.tunnel_id, base_dir=args.data_dir)

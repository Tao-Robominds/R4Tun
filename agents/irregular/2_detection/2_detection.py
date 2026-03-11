"""
Irregular Tunnel Detection Pipeline

Detects oblique groove lines in the depth map, finds K-block positions
via DBSCAN clustering + regulator refinement, and expands to all segment
positions via per-ring offsets.

Outputs:
  - all_segments.csv (Ring, Block, X, Y, quality) — segment centroids
  - boundaries_per_ring.json — boundary positions for downstream segmentation
  - detected_lines.png — visualization

K detection: DBSCAN clustering of pos/neg line intersections, refined by
a regulator that selects the best pos-neg pair midpoint per ring.
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import DBSCAN, AgglomerativeClustering


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
# Minimum |angle| (degrees) for a line to count as oblique pos/neg; excludes near-horizontal from K detection
DEFAULT_ANGLE_MIN_OBLIQUE_DEG = 3.0

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
    min_oblique_deg = float(params.get('angle_min_oblique_deg', DEFAULT_ANGLE_MIN_OBLIQUE_DEG))

    positive_lines = []
    negative_lines = []
    horizontal_lines = []

    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            # Exclude near-horizontal: pos must be >= min_oblique_deg, neg must be <= -min_oblique_deg
            if angle_pos_min <= angle <= angle_pos_max and angle >= min_oblique_deg:
                positive_lines.append(line[0])
            elif angle_neg_min <= angle <= angle_neg_max and angle <= -min_oblique_deg:
                negative_lines.append(line[0])

    max_line_length_px = params.get('max_line_length_px')
    if max_line_length_px is not None and max_line_length_px > 0:
        def length_ok(seg):
            x1, y1, x2, y2 = seg
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) <= max_line_length_px
        positive_lines = [s for s in positive_lines if length_ok(s)]
        negative_lines = [s for s in negative_lines if length_ok(s)]

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


def merge_close_points(points: List[float], threshold: float = None) -> List[float]:
    """Merge Y-values within threshold (px). If threshold is None, use FIXED_MERGE_CLOSE_THRESHOLD."""
    if len(points) == 0:
        return []
    th = threshold if threshold is not None else FIXED_MERGE_CLOSE_THRESHOLD
    pts = np.array(points, dtype=np.float64)
    if len(pts) == 1:
        return [float(pts[0])]

    merged = []
    while len(pts) > 0:
        p = pts[0]
        close_mask = np.abs(pts - p) < th
        merged.append(float(np.mean(pts[close_mask])))
        pts = pts[~close_mask]
    return merged


# =============================================================================
# K-Position Detection: Line Midpoint (per-ring)
# =============================================================================

def _segment_length_and_angle(seg: Tuple) -> Tuple[float, float]:
    """Return (length_px, angle_deg) for segment. Angle: atan2(-(y2-y1), x2-x1) in degrees."""
    x1, y1, x2, y2 = seg
    length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
    angle_deg = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
    return length, float(angle_deg)


def detect_k_line_midpoint(
    line_data: Dict,
    ring_count: int,
    k_height_px: float,
    params: Dict,
) -> pd.DataFrame:
    """K positions from pos/neg line crossings at each ring's vertical slice.

    1. Close enough green and red (pos/neg pair with gap in range) -> K = midpoint.
    2. Only one green or red -> K = line crossing Y +/- (k_height_px/2).
    3. Only one, or pair too far -> pick most consistent line (best angle in 6-9 or -9 to -6, then longer) as base, then K = base Y +/- (k_height_px/2).
    """
    L = line_data['image_height']
    W = line_data['image_width']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    ring_width = W / ring_count
    merge_close_fraction = float(params.get('merge_close_fraction', 0.05))
    merge_threshold = ring_width * merge_close_fraction
    target_gap = float(params.get('reg_target_gap_frac', 0.5)) * k_height_px
    max_k_gap_factor = float(params.get('max_k_gap_factor', 1.2))
    gap_cap = k_height_px * max_k_gap_factor
    half_k = k_height_px / 2.0
    angle_pos_min = float(params.get('angle_pos_min', 6.0))
    angle_pos_max = float(params.get('angle_pos_max', 9.0))
    angle_neg_min = float(params.get('angle_neg_min', -9.0))
    angle_neg_max = float(params.get('angle_neg_max', -6.0))

    def wrap_dy(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, L - d)

    def single_line_candidates(vertical_x: float):
        """List of (y, is_pos, length, angle_deg) for each crossing at vertical_x."""
        cands = []
        for seg in positive_lines:
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                length, angle_deg = _segment_length_and_angle(seg)
                cands.append((float(y_val), True, length, angle_deg))
        for seg in negative_lines:
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                length, angle_deg = _segment_length_and_angle(seg)
                cands.append((float(y_val), False, length, angle_deg))
        return cands

    def line_quality(length: float, angle_deg: float, is_pos: bool) -> float:
        """Quality = length; boost if angle in expected range (6-9 pos, -9 to -6 neg)."""
        in_range = (angle_pos_min <= angle_deg <= angle_pos_max) if is_pos else (angle_neg_min <= angle_deg <= angle_neg_max)
        return length * (1.0 if in_range else 0.2)

    def best_single_candidate(candidates: list) -> Optional[Tuple[float, bool]]:
        """Pick (y, is_pos) by: prefer angle in range (6-9 pos, -9 to -6 neg), then longer."""
        if not candidates:
            return None
        def score(c):
            y, is_pos, length, angle = c
            in_range = (angle_pos_min <= angle <= angle_pos_max) if is_pos else (angle_neg_min <= angle <= angle_neg_max)
            return (1 if in_range else 0, length)
        best = max(candidates, key=score)
        return (best[0], best[1])

    rows = []
    for i in range(ring_count):
        vertical_x = (i + 0.5) * ring_width
        # Segment-level crossings: (y, length, angle) per pos/neg so we can score pairs by line quality
        pos_cands = []
        for seg in positive_lines:
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                length, angle_deg = _segment_length_and_angle(seg)
                pos_cands.append((float(y_val), length, angle_deg))
        neg_cands = []
        for seg in negative_lines:
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                length, angle_deg = _segment_length_and_angle(seg)
                neg_cands.append((float(y_val), length, angle_deg))

        # Hierarchy: (1) When both red and green with valid pair(s) -> use pair midpoint.
        # (2) When multiple pairs: rings 0-3 use longest+right-angles; ring 4+ use central groove.
        # (3) Fallback (no valid pair): use longest red/green line with right angles.
        if pos_cands and neg_cands:
            pair_candidates = []
            for (py, len_p, angle_p) in pos_cands:
                for (ny, len_n, angle_n) in neg_cands:
                    gap = wrap_dy(py, ny)
                    if gap > gap_cap:
                        continue
                    gap_err = abs(gap - target_gap)
                    mid = (py + ny) / 2.0
                    if i <= 3:
                        # Rings 0-3: prefer pair by longest lines with right angles (no center)
                        quality = line_quality(len_p, angle_p, True) + line_quality(len_n, angle_n, False)
                        score = gap_err - quality  # lower is better
                    else:
                        # Ring 4+: prefer groove closer to image center
                        mid_dist_to_center = wrap_dy(mid, L / 2.0)
                        score = gap_err + 0.1 * mid_dist_to_center
                    pair_candidates.append((score, mid))
            best_mid = None
            if pair_candidates:
                pair_candidates.sort(key=lambda c: c[0])
                best_mid = pair_candidates[0][1]
            if best_mid is not None:
                rows.append(('midpoint', vertical_x, best_mid, 0.9))
            else:
                # Case 3: Both but too far -> pick most consistent line, then K = base Y +/- half_k
                cands = single_line_candidates(vertical_x)
                best = best_single_candidate(cands)
                if best is not None:
                    y_base, is_pos = best
                    y_k = y_base - half_k if is_pos else y_base + half_k
                    y_k = max(0.0, min(L - 1e-6, y_k))
                    rows.append(('midpoint_fallback', vertical_x, y_k, 0.6))
                else:
                    merge_pos = merge_close_points([c[0] for c in pos_cands], merge_threshold)
                    merge_neg = merge_close_points([c[0] for c in neg_cands], merge_threshold)
                    mid_fallback = (merge_pos[0] + merge_neg[0]) / 2.0 if (merge_pos and merge_neg) else L / 2.0
                    rows.append(('midpoint_fallback', vertical_x, mid_fallback, 0.5))
        else:
            # Case 2 & 3: Only one side, or none -> pick most consistent line, K = line Y +/- half_k
            cands = single_line_candidates(vertical_x)
            best = best_single_candidate(cands)
            if best is not None:
                y_base, is_pos = best
                y_k = y_base - half_k if is_pos else y_base + half_k
                y_k = max(0.0, min(L - 1e-6, y_k))
                label = 'positive_slope' if is_pos else 'negative_slope'
                rows.append((label, vertical_x, y_k, 0.6))
            else:
                rows.append(('default', vertical_x, L / 2.0, 0.2))

    df = pd.DataFrame(rows, columns=['Type', 'X', 'Y', 'Confidence'])
    return df.sort_values(by='X').reset_index(drop=True)


# =============================================================================
# K-Position Detection: DBSCAN Clustering
# =============================================================================

def detect_k_dbscan(
    line_data: Dict,
    ring_count: int,
    params: Dict,
) -> pd.DataFrame:
    """Detect K positions via DBSCAN clustering of pos/neg line intersections.

    1. Extend lines and find all pos-neg intersections.
    2. DBSCAN cluster the normalized (x/W, y/L) coordinates.
    3. Subdivide wide clusters spanning multiple ring widths.
    4. Merge intersection clusters with line midpoints via
       AgglomerativeClustering to produce exactly ring_count positions.

    Returns DataFrame with columns Type, X, Y, Confidence (sorted by X).
    """
    L = line_data['image_height']
    W = line_data['image_width']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']

    eps = params.get('eps', 0.07)
    subdivision_threshold = params.get('complex_subdivision_threshold', 1.5)
    max_subdivisions = params.get('complex_max_subdivisions', 4)

    extended_positive = [extend_line_to_bounds(*line, W, L) for line in positive_lines]
    extended_negative = [extend_line_to_bounds(*line, W, L) for line in negative_lines]
    intersections = find_line_intersections(extended_positive, extended_negative, W, L)

    expected_ring_width = W / ring_count

    if len(intersections) == 0:
        fallback = []
        for x1, y1, x2, y2 in positive_lines + negative_lines:
            fallback.append(('midpoint_fallback', (x1+x2)/2, (y1+y2)/2, 0.3))
        if not fallback:
            for i in range(ring_count):
                fallback.append(('default', (i+0.5)*expected_ring_width, L/2, 0.1))
        df = pd.DataFrame(fallback, columns=['Type', 'X', 'Y', 'Confidence'])
        return df.sort_values(by='X').reset_index(drop=True)

    intersection_array = np.array(intersections)
    x_normalized = intersection_array[:, 0] / W
    y_normalized = intersection_array[:, 1] / L
    features = np.column_stack([x_normalized, y_normalized])

    eps_candidates = [eps, eps * 1.5, eps * 2.0, 0.10, 0.15]
    eps_candidates = sorted(list(set([round(e, 2) for e in eps_candidates])))
    min_clusters = max(3, ring_count // 2)

    for eps in eps_candidates:
        clustering = DBSCAN(eps=eps, min_samples=1).fit(features)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= min(ring_count, min_clusters) or eps == eps_candidates[-1]:
            break

    unique_labels = set(labels) - {-1}

    k_positions = []
    for label in unique_labels:
        cluster_points = intersection_array[labels == label]
        x_range = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
        n_sub = max(1, int(x_range / (expected_ring_width * subdivision_threshold)) + 1)
        n_sub = min(n_sub, max_subdivisions)

        if n_sub > 1:
            sorted_pts = cluster_points[np.argsort(cluster_points[:, 0])]
            sub_size = len(sorted_pts) // n_sub
            for s in range(n_sub):
                start = s * sub_size
                end = (s + 1) * sub_size if s < n_sub - 1 else len(sorted_pts)
                sub = sorted_pts[start:end]
                conf = min(1.0, 0.5 + 0.05 * len(sub))
                k_positions.append(('intersection_sub', np.mean(sub[:, 0]),
                                    np.mean(sub[:, 1]), conf))
        else:
            conf = min(1.0, 0.5 + 0.1 * len(cluster_points))
            k_positions.append(('intersection_cluster', np.mean(cluster_points[:, 0]),
                                np.mean(cluster_points[:, 1]), conf))

    midpoint_confidence = params.get('complex_conf_midpoint', 0.7)
    intersection_confidence = params.get('complex_conf_intersection', 0.9)

    line_midpoints = []
    for x1, y1, x2, y2 in positive_lines:
        line_midpoints.append(('positive_midpoint', (x1+x2)/2, (y1+y2)/2, midpoint_confidence))
    for x1, y1, x2, y2 in negative_lines:
        line_midpoints.append(('negative_midpoint', (x1+x2)/2, (y1+y2)/2, midpoint_confidence))

    all_candidates = k_positions + line_midpoints

    if len(all_candidates) > ring_count and len(all_candidates) >= 2:
        candidate_array = np.array([[p[1], p[2]] for p in all_candidates])
        x_norm = candidate_array[:, 0] / W
        y_norm = candidate_array[:, 1] / L
        feat = np.column_stack([x_norm, y_norm])

        n_clusters = min(ring_count, len(all_candidates))
        agg = AgglomerativeClustering(n_clusters=n_clusters).fit(feat)

        final_positions = []
        for lbl in range(n_clusters):
            mask = agg.labels_ == lbl
            cluster_pts = candidate_array[mask]
            cluster_types = [all_candidates[j][0] for j in range(len(mask)) if mask[j]]
            has_intersection = any('intersection' in t for t in cluster_types)
            det_type = 'intersection_cluster' if has_intersection else 'midpoint_cluster'
            conf = intersection_confidence if has_intersection else midpoint_confidence
            final_positions.append((det_type, np.mean(cluster_pts[:, 0]),
                                    np.mean(cluster_pts[:, 1]), conf))
        k_positions = final_positions

    k_positions.sort(key=lambda p: p[1])
    df = pd.DataFrame(k_positions, columns=['Type', 'X', 'Y', 'Confidence'])
    return df.sort_values(by='X').reset_index(drop=True)


# =============================================================================
# K-Position Regulator (refines Y using oblique line pair geometry)
# =============================================================================

def apply_k_regulator(
    k_df: pd.DataFrame,
    line_data: Dict,
    ring_count: int,
    k_height_px: float,
    params: Dict,
):
    """Refine K Y positions using oblique line pair geometry.

    For each ring, find pos/neg crossing pairs whose gap matches the target;
    pick best pair by gap match or proximity to detection; optionally blend
    with detection Y. Params (with defaults):
      reg_target_gap_frac      expected pos/neg gap as fraction of k_height_px (0.5)
      reg_gap_tolerance        fractional tolerance (0.5)
      reg_blend_weight         blend line vs detection (1.0 = pure line)
      reg_max_det_line_dist_frac  max Y dist for blend as fraction of k_height_px (1.0)
      reg_use_line_when_available  use line Y when valid pair exists (True)
      reg_pick_by_gap          pick pair by best gap match (True)
      reg_relaxed_gap_fallback if no pair in tolerance, try 2x tolerance (True)

    Returns (DataFrame with Ring, Type, X, Y, Confidence; used_pos_indices; used_neg_indices).
    """
    L = line_data['image_height']
    W = line_data['image_width']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    ring_width = W / ring_count
    merge_close_fraction = float(params.get('merge_close_fraction', 0.05))
    merge_threshold = ring_width * merge_close_fraction

    reg_target_gap = float(params.get('reg_target_gap_frac', 0.5)) * k_height_px
    reg_gap_tolerance = float(params.get('reg_gap_tolerance', 0.5))
    reg_blend_weight = float(params.get('reg_blend_weight', 1.0))
    reg_max_det_line_dist = float(params.get('reg_max_det_line_dist_frac', 1.0)) * k_height_px
    reg_use_line_when_available = params.get('reg_use_line_when_available', True)
    reg_pick_by_gap = params.get('reg_pick_by_gap', True)
    gap_allow = reg_gap_tolerance * reg_target_gap
    max_k_gap_factor = float(params.get('max_k_gap_factor', 1.2))
    gap_cap = k_height_px * max_k_gap_factor
    angle_pos_min = float(params.get('angle_pos_min', 6.0))
    angle_pos_max = float(params.get('angle_pos_max', 9.0))
    angle_neg_min = float(params.get('angle_neg_min', -9.0))
    angle_neg_max = float(params.get('angle_neg_max', -6.0))

    def wrap_dy(a, b):
        d = abs(a - b)
        return min(d, L - d)

    def line_quality(length: float, angle_deg: float, is_pos: bool) -> float:
        in_range = (angle_pos_min <= angle_deg <= angle_pos_max) if is_pos else (angle_neg_min <= angle_deg <= angle_neg_max)
        return length * (1.0 if in_range else 0.2)

    det_by_ring = {}
    for idx in range(len(k_df)):
        row = k_df.iloc[idx]
        r = idx
        if 0 <= r < ring_count:
            det_by_ring[r] = {
                'Y': float(row['Y']),
                'Type': str(row.get('Type', 'detected')),
                'Confidence': float(row.get('Confidence', 0.5)),
            }
    for i in range(ring_count):
        if i not in det_by_ring:
            det_by_ring[i] = {'Y': L / 2.0, 'Type': 'fallback', 'Confidence': 0.1}

    # Per-ring: (vertical_x, pos_ys_with_idx, neg_ys_with_idx, det_y) for line-index tracking
    ring_data = []
    for i in range(ring_count):
        vertical_x = (i + 0.5) * ring_width
        pos_ys_with_idx = []
        for j, seg in enumerate(positive_lines):
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                pos_ys_with_idx.append((y_val, j))
        neg_ys_with_idx = []
        for j, seg in enumerate(negative_lines):
            y_val = line_segment_vertical_intersection(vertical_x, seg)
            if y_val is not None:
                neg_ys_with_idx.append((y_val, j))
        merge_pos = merge_close_points([y for y, _ in pos_ys_with_idx], merge_threshold)
        merge_neg = merge_close_points([y for y, _ in neg_ys_with_idx], merge_threshold)
        ring_data.append((vertical_x, pos_ys_with_idx, neg_ys_with_idx, merge_pos, merge_neg, det_by_ring[i]['Y']))

    used_pos_indices = set()
    used_neg_indices = set()
    rows = []
    for i in range(ring_count):
        vertical_x, pos_ys_with_idx, neg_ys_with_idx, merge_pos, merge_neg, det_y = ring_data[i]
        y_line = None
        best_py, best_ny = None, None

        if merge_pos and merge_neg:
            candidates = []
            for py in merge_pos:
                for ny in merge_neg:
                    gap = wrap_dy(py, ny)
                    if gap > gap_cap or abs(gap - reg_target_gap) > gap_allow:
                        continue
                    mid = (py + ny) / 2.0
                    candidates.append((mid, gap, wrap_dy(mid, det_y), py, ny))
            if not candidates and params.get('reg_relaxed_gap_fallback', True):
                gap_allow_wide = 2.0 * gap_allow
                for py in merge_pos:
                    for ny in merge_neg:
                        gap = wrap_dy(py, ny)
                        if gap > gap_cap or abs(gap - reg_target_gap) > gap_allow_wide:
                            continue
                        mid = (py + ny) / 2.0
                        candidates.append((mid, gap, wrap_dy(mid, det_y), py, ny))
            if candidates:
                # Hierarchy: rings 0-3 pick pair by longest+right-angles; ring 4+ by central groove
                def reg_score(c):
                    mid, gap, dist_to_det = c[0], c[1], c[2]
                    py, ny = c[3], c[4]
                    gap_err = abs(gap - reg_target_gap)
                    mid_dist_to_center = wrap_dy(mid, L / 2.0)
                    if i <= 3:
                        pos_q = max(
                            (line_quality(*_segment_length_and_angle(positive_lines[j]), True)
                             for y_val, j in pos_ys_with_idx if abs(y_val - py) < merge_threshold),
                            default=0.0,
                        )
                        neg_q = max(
                            (line_quality(*_segment_length_and_angle(negative_lines[j]), False)
                             for y_val, j in neg_ys_with_idx if abs(y_val - ny) < merge_threshold),
                            default=0.0,
                        )
                        quality = pos_q + neg_q
                        if reg_pick_by_gap:
                            return gap_err - quality
                        return dist_to_det - quality
                    else:
                        if reg_pick_by_gap:
                            return gap_err + 0.3 * mid_dist_to_center
                        return dist_to_det + 0.3 * mid_dist_to_center
                best = min(candidates, key=reg_score)
                best_mid, _, _, best_py, best_ny = best
                y_line = best_mid

        if y_line is not None and best_py is not None and best_ny is not None:
            for y_val, idx in pos_ys_with_idx:
                if abs(y_val - best_py) < merge_threshold:
                    used_pos_indices.add(idx)
            for y_val, idx in neg_ys_with_idx:
                if abs(y_val - best_ny) < merge_threshold:
                    used_neg_indices.add(idx)

        if y_line is not None:
            use_line = reg_use_line_when_available or wrap_dy(y_line, det_y) <= reg_max_det_line_dist
            if use_line:
                y_reg = reg_blend_weight * y_line + (1.0 - reg_blend_weight) * det_y
                reg_type = 'regulated'
            else:
                y_reg = det_y
                reg_type = 'unregulated'
        else:
            y_reg = det_y
            reg_type = 'unregulated'

        y_reg = max(0.0, min(L - 1e-6, y_reg))
        orig_type = det_by_ring[i]['Type'] + '_' + reg_type
        conf = 0.9 if reg_type == 'regulated' else det_by_ring[i]['Confidence']
        rows.append({
            'Ring': i, 'Type': orig_type, 'X': vertical_x,
            'Y': y_reg, 'Confidence': conf,
        })

    out = pd.DataFrame(rows)
    n_regulated = sum(1 for r in rows if r['Type'].endswith('_regulated'))
    print(f"  Regulator: {n_regulated}/{ring_count} rings regulated "
          f"(target_gap={reg_target_gap:.0f}, tolerance={reg_gap_tolerance}, "
          f"blend={reg_blend_weight})")
    return out, used_pos_indices, used_neg_indices


# =============================================================================
# Segment Expansion: K → All Segments
# =============================================================================

def expand_k_with_per_ring_offsets(
    k_positions: pd.DataFrame,
    img_height: int,
    per_ring_offsets: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Expand K positions to all segments using per-ring boundary offsets.

    per_ring_offsets: {"ring_idx": {"block_name": dy, ...}, ...}
    Each offset is a signed circular distance from detected K_Y to the
    block's boundary Y position.  All blocks (including K) must be present.
    """
    rows = []
    for ring_idx, (_, k_row) in enumerate(k_positions.iterrows()):
        k_x = float(k_row['X'])
        k_y = float(k_row['Y'])
        quality = float(k_row.get('Confidence', 1.0))

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
# Segments → Boundary JSON (per_ring_offsets are already boundary positions)
# =============================================================================

def segments_to_boundaries(all_segments: pd.DataFrame) -> Dict[str, list]:
    """Convert all_segments Y values directly to boundary JSON.

    per_ring_offsets store boundary offsets from K, so Y in all_segments
    already represents the boundary start position for each block.
    """
    result = {}
    for ring_idx in sorted(all_segments['Ring'].unique()):
        ring_segs = all_segments[all_segments['Ring'] == ring_idx].copy()
        ring_segs = ring_segs.sort_values('Y').reset_index(drop=True)

        boundaries = []
        for _, row in ring_segs.iterrows():
            boundaries.append({
                'y': round(float(row['Y']), 1),
                'block': str(row['Block']),
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
    used_pos_indices: Optional[set] = None,
    used_neg_indices: Optional[set] = None,
) -> None:
    """Save visualization of detected lines and segment positions.

    If used_pos_indices / used_neg_indices are provided, only draw oblique lines
    that survived regulation. Horizontal lines are unrelated to K; in regulated
    view we do not draw them.
    """
    dilated_edges = line_data['dilated_edges']
    L, W = line_data['image_height'], line_data['image_width']

    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # so plt.imshow shows R=red, G=green, B=blue

    # Oblique lines: draw all detected; highlight (thicker) those used in regulation. Red = pos, green = neg.
    pos_lines = line_data['positive_lines']
    neg_lines = line_data['negative_lines']
    for i, (x1, y1, x2, y2) in enumerate(pos_lines):
        thickness = 3 if (used_pos_indices is None or i in used_pos_indices) else 1
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness)
    for i, (x1, y1, x2, y2) in enumerate(neg_lines):
        thickness = 3 if (used_neg_indices is None or i in used_neg_indices) else 1
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
    # Horizontal lines: draw all when present (they are not used for K).
    for x1, y1, x2, y2 in line_data['horizontal_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    for _, row in k_positions.iterrows():
        cv2.circle(output_image, (int(row['X']), int(row['Y'])), 8, (255, 255, 0), -1)   # RGB yellow
        cv2.line(output_image, (int(row['X']), 0), (int(row['X']), L), (255, 0, 255), 1)   # magenta

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
    """Run the detection pipeline. Returns (k_positions, all_segments) DataFrames.

    Pipeline: detect_lines -> detect_k (line_midpoint or dbscan) -> apply_k_regulator
              -> expand_k_with_per_ring_offsets -> segments_to_boundaries.
    """
    print(f"Detection Pipeline: {tunnel_id}")

    params, params_loaded = load_parameters(tunnel_id, base_dir)

    preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
    tunnel_diameter = preprocessing_params.get('tunnel_diameter', 5.5)
    resolution = preprocessing_params.get('depth_map_resolution', 0.005)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
    k_height_px = k_height_mm / (resolution * 1000.0)

    tunnel_dir = os.path.join(base_dir, tunnel_id)

    depth_map_outlier_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_map_outlier_path):
        raise FileNotFoundError(f"depth_map_outlier.npy not found. Run preprocessing first.")

    depth_map_outlier = np.load(depth_map_outlier_path)
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    L, W = depth_map_outlier.shape

    per_ring_offsets = params.get('per_ring_offsets', None)
    if per_ring_offsets is None:
        raise ValueError("per_ring_offsets required in parameters_detection.json")

    print(f"  Lines: detecting (image {L}x{W})...")
    if params.get('max_line_length_px') is None:
        params = dict(params)
        params['max_line_length_px'] = (W / ring_count) * float(params.get('max_line_length_factor', 1.5))
    line_data = detect_lines(depth_map_outlier, params)
    print(f"  Lines: +{len(line_data['positive_lines'])} -{len(line_data['negative_lines'])} "
          f"H{len(line_data['horizontal_lines'])} V{len(line_data['vertical_lines'])}")

    k_y_override = params.get('k_y_positions', None)
    used_pos_indices = None
    used_neg_indices = None
    if k_y_override is not None and len(k_y_override) == ring_count:
        print(f"  K positions: using k_y_positions override ({ring_count} rings)")
        ring_spacing = W / ring_count
        rows = []
        for i in range(ring_count):
            band_x = (i + 0.5) * ring_spacing
            rows.append(('k_override', band_x, k_y_override[i], 1.0))
        k_positions = pd.DataFrame(rows, columns=['Type', 'X', 'Y', 'Confidence'])
    elif params.get('k_detection_method') == 'line_midpoint':
        print(f"  K positions: line-midpoint per ring ({ring_count} rings)...")
        k_positions = detect_k_line_midpoint(
            line_data, ring_count, k_height_px, params
        )
        print(f"  K positions: {len(k_positions)} raw, "
              f"types={k_positions['Type'].value_counts().to_dict()}")
        print(f"  K positions: applying regulator...")
        k_positions, used_pos_indices, used_neg_indices = apply_k_regulator(
            k_positions, line_data, ring_count, k_height_px, params
        )
    else:
        print(f"  K positions: DBSCAN detection ({ring_count} rings)...")
        k_positions = detect_k_dbscan(line_data, ring_count, params)
        print(f"  K positions: {len(k_positions)} raw, "
              f"types={k_positions['Type'].value_counts().to_dict()}")

        print(f"  K positions: applying regulator...")
        k_positions, used_pos_indices, used_neg_indices = apply_k_regulator(
            k_positions, line_data, ring_count, k_height_px, params
        )

    k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)

    # Per-ring diagnostic: oblique crossings vs outcome (detected but filtered vs not detected)
    if used_pos_indices is not None or used_neg_indices is not None:
        ring_width = W / ring_count
        pos_lines = line_data['positive_lines']
        neg_lines = line_data['negative_lines']
        print(f"  Per-ring oblique crossings vs K outcome:")
        for i in range(ring_count):
            vx = (i + 0.5) * ring_width
            n_pos = sum(1 for seg in pos_lines if line_segment_vertical_intersection(vx, seg) is not None)
            n_neg = sum(1 for seg in neg_lines if line_segment_vertical_intersection(vx, seg) is not None)
            k_type = k_positions.iloc[i]['Type'] if i < len(k_positions) else '?'
            if n_pos == 0 and n_neg == 0:
                why = "no oblique crossings"
            elif n_pos == 0 or n_neg == 0:
                why = "one side only (pos or neg missing)"
            elif "regulated" in str(k_type) and "unregulated" not in str(k_type):
                why = "regulated"
            else:
                why = "detected but filtered (no valid pair)"
            print(f"    Ring {i}: {n_pos} pos, {n_neg} neg → {k_type} ({why})")

    print(f"  Expanding K → all segments (per-ring offsets, "
          f"{len(per_ring_offsets)} rings)...")
    all_segments = expand_k_with_per_ring_offsets(
        k_positions, img_height=L, per_ring_offsets=per_ring_offsets,
    )

    output_filename = params.get('output_filename', 'all_segments.csv')
    all_segments.to_csv(os.path.join(tunnel_dir, output_filename), index=False)
    print(f"  Segments: {len(all_segments)} total")

    boundaries = segments_to_boundaries(all_segments)
    boundaries_path = os.path.join(tunnel_dir, 'boundaries_per_ring.json')
    with open(boundaries_path, 'w') as f:
        json.dump(boundaries, f, indent=2)
    print(f"  Boundaries: {len(boundaries)} rings → {boundaries_path}")

    visualize_detection(
        line_data, k_positions, tunnel_dir, all_segments=all_segments,
        used_pos_indices=used_pos_indices, used_neg_indices=used_neg_indices,
    )
    print(f"  Saved: detected.csv, {output_filename}, boundaries_per_ring.json, detected_lines.png")

    return k_positions, all_segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Irregular tunnel detection pipeline")
    parser.add_argument("tunnel_id", help="Tunnel identifier")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()

    run_detection(args.tunnel_id, base_dir=args.data_dir)

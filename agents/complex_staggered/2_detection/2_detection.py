"""
Complex Staggered Detection Pipeline: Line Detection and K-Position Calculation

This module detects oblique lines, finds intersections, and calculates K-block
positions for complex_staggered patterns (tunnels 4-1, 5-1) using DBSCAN clustering
and line subdivision.

Based on P4TUN optimization reports:
- Detection provided +6.3% mIoU improvement - the LARGEST single-stage gain
- binary_threshold, hough_oblique_threshold, angle parameters are HIGH sensitivity
- Complex patterns require wider angle ranges and intersection-based detection

Critical Parameters (~22 total):
- Base detection (14): binary_threshold, hough_oblique_threshold, angle_positive/negative_min/max,
  hough_vertical_threshold, hough_horizontal_*, horizontal_angle_tolerance, merge_distance_threshold,
  dilation_*, hough_oblique_min_length/max_gap
- Complex-specific (8+): complex_hough_threshold/min_length/max_gap, complex_angle_pos/neg_min/max,
  complex_min_y_span, complex_min_x_span, complex_eps_primary, complex_eps_secondary,
  complex_subdivision_threshold, complex_max_subdivisions, complex_conf_midpoint, complex_conf_intersection

Physical constants (k_height_mm, ab_height_mm) are read from preprocessing stage.
"""

import os
import sys
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
    """
    Load parameters from parameters_detection.json.
    
    Priority:
        1. agents/complex_staggered/2_detection/parameters/<tunnel_id>/parameters_detection.json
        2. data/<tunnel_id>/parameters_detection.json
        3. agents/complex_staggered/2_detection/parameters/sample/parameters_detection.json
        4. Hardcoded defaults
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_detection.json"
    
    if tunnel_id:
        params_path = os.path.join(script_dir, "parameters", tunnel_id, param_file)
        if os.path.exists(params_path):
            print(f"Loading parameters from {params_path}")
            with open(params_path, 'r') as f:
                return json.load(f), True
        
        tunnel_path = os.path.join(base_dir, tunnel_id, param_file)
        if os.path.exists(tunnel_path):
            print(f"Loading parameters from {tunnel_path}")
            with open(tunnel_path, 'r') as f:
                return json.load(f), True
    
    sample_path = os.path.join(script_dir, "parameters", "sample", param_file)
    if os.path.exists(sample_path):
        print(f"Loading sample parameters from {sample_path}")
        with open(sample_path, 'r') as f:
            return json.load(f), True
    
    print("Warning: No parameter file found, using hardcoded defaults")
    return {}, False


def get_param(params: Dict, key: str, default=None):
    """Get parameter value with default fallback."""
    return params.get(key, default)


# =============================================================================
# CRITICAL PARAMETERS (tunable via JSON)
# =============================================================================

# Preprocessing - HIGH sensitivity
DEFAULT_BINARY_THRESHOLD = 127

# Hough Oblique - HIGH sensitivity
DEFAULT_HOUGH_OBLIQUE_THRESHOLD = 50
DEFAULT_ANGLE_POSITIVE_MIN = 6.0
DEFAULT_ANGLE_POSITIVE_MAX = 9.0
DEFAULT_ANGLE_NEGATIVE_MIN = -9.0
DEFAULT_ANGLE_NEGATIVE_MAX = -6.0

# Hough Vertical - MEDIUM-HIGH sensitivity
DEFAULT_HOUGH_VERTICAL_THRESHOLD = 500

# Hough Horizontal - MEDIUM sensitivity (tunable)
DEFAULT_HOUGH_HORIZONTAL_THRESHOLD = 50
DEFAULT_HOUGH_HORIZONTAL_MIN_LENGTH = 100
DEFAULT_HOUGH_HORIZONTAL_MAX_GAP = 10
DEFAULT_HORIZONTAL_ANGLE_TOLERANCE = 1.0

# Merge parameters - MEDIUM sensitivity (tunable)
DEFAULT_MERGE_DISTANCE_THRESHOLD = 3.0

# Dilation - MEDIUM sensitivity
DEFAULT_DILATION_KERNEL_SIZE = 3
DEFAULT_DILATION_ITERATIONS = 1
DEFAULT_HOUGH_OBLIQUE_MIN_LENGTH = 100
DEFAULT_HOUGH_OBLIQUE_MAX_GAP = 40

# Fixed parameters (LOW sensitivity)
FIXED_MERGE_CLOSE_THRESHOLD = 6.0  # For merging close Y positions

# Complex-specific parameters (tunable, flat keys)
DEFAULT_COMPLEX_HOUGH_THRESHOLD = 30
DEFAULT_COMPLEX_HOUGH_MIN_LENGTH = 50
DEFAULT_COMPLEX_HOUGH_MAX_GAP = 100
DEFAULT_COMPLEX_ANGLE_POS_MIN = 4.0
DEFAULT_COMPLEX_ANGLE_POS_MAX = 12.0
DEFAULT_COMPLEX_ANGLE_NEG_MIN = -12.0
DEFAULT_COMPLEX_ANGLE_NEG_MAX = -4.0
DEFAULT_COMPLEX_MIN_Y_SPAN = 30
DEFAULT_COMPLEX_MIN_X_SPAN = 30
DEFAULT_COMPLEX_EPS_PRIMARY = 0.05
DEFAULT_COMPLEX_EPS_SECONDARY = 0.10
DEFAULT_COMPLEX_SUBDIVISION_THRESHOLD = 1.5
DEFAULT_COMPLEX_MAX_SUBDIVISIONS = 4
DEFAULT_COMPLEX_CONF_MIDPOINT = 0.7
DEFAULT_COMPLEX_CONF_INTERSECTION = 0.9

# Physical Constants - READ FROM PREPROCESSING STAGE
# k_height_mm = π * tunnel_diameter * 1000 / 16 (for 6-segment ring)
# ab_height_mm = 3 * k_height_mm
# tunnel_diameter and resolution come from preprocessing params

PREPROCESSING_PARAMS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "1_preprocessing", "parameters"
)


def load_preprocessing_params(tunnel_id: str, base_dir: str = "data") -> Dict:
    """
    Load preprocessing parameters for the tunnel.
    
    Physical constants (tunnel_diameter, depth_map_resolution) are defined
    in preprocessing and must be read from there - not duplicated.
    """
    # Try tunnel-specific first, then sample
    for subdir in [tunnel_id, "sample"]:
        params_path = os.path.join(PREPROCESSING_PARAMS_DIR, subdir, "parameters_preprocessing.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                return json.load(f)
    
    # Try data folder
    tunnel_path = os.path.join(base_dir, tunnel_id, "parameters_preprocessing.json")
    if os.path.exists(tunnel_path):
        with open(tunnel_path, 'r') as f:
            return json.load(f)
    
    return {}


def calculate_segment_heights(tunnel_diameter: float) -> Tuple[float, float]:
    """
    Calculate K-block and AB-block heights from tunnel diameter.
    
    For 6-segment simple staggered pattern:
    - K-block spans 1/16 of circumference
    - AB-block spans 3/16 of circumference (3x K-block)
    
    Args:
        tunnel_diameter: Tunnel diameter in meters
        
    Returns:
        (k_height_mm, ab_height_mm)
    """
    circumference_mm = np.pi * tunnel_diameter * 1000
    k_height_mm = circumference_mm / 16
    ab_height_mm = 3 * k_height_mm
    return k_height_mm, ab_height_mm


# =============================================================================
# Utility Functions
# =============================================================================

def mm_to_px(mm: float, resolution: float) -> float:
    """Convert millimeters to pixels."""
    return mm / (resolution * 1000)


# =============================================================================
# Line Detection
# =============================================================================

def detect_lines(
    depth_map_outlier: np.ndarray,
    binary_threshold: int,
    hough_oblique_threshold: int,
    angle_positive_min: float,
    angle_positive_max: float,
    angle_negative_min: float,
    angle_negative_max: float,
    hough_vertical_threshold: int,
    dilation_kernel_size: int,
    dilation_iterations: int,
    hough_oblique_min_length: int,
    hough_oblique_max_gap: int,
    hough_horizontal_threshold: int,
    hough_horizontal_min_length: int,
    hough_horizontal_max_gap: int,
    horizontal_angle_tolerance: float,
    merge_distance_threshold: float
) -> Dict:
    """
    Detect oblique, horizontal, and vertical lines from depth map.
    
    CRITICAL PARAMETERS (HIGH sensitivity):
    - binary_threshold: Edge detection sensitivity
    - hough_oblique_threshold: Line detection sensitivity
    - angle_positive/negative_min/max: Oblique line filtering
    - hough_vertical_threshold: Ring boundary detection
    
    MEDIUM SENSITIVITY PARAMETERS:
    - dilation_kernel_size, dilation_iterations: Morphological operations
    - hough_oblique_min_length, hough_oblique_max_gap: Line filtering
    - hough_horizontal_*: Horizontal line detection
    - horizontal_angle_tolerance: Horizontal line filtering
    - merge_distance_threshold: Vertical line merging
    """
    L, W = depth_map_outlier.shape
    
    # Pre-processing - Binary on NaN/non-NaN
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, binary_threshold, 255, cv2.THRESH_BINARY)
    
    # Enhanced edge detection using depth values
    depth_valid = depth_map_outlier[~np.isnan(depth_map_outlier)]
    if len(depth_valid) > 0:
        depth_min, depth_max = depth_valid.min(), depth_valid.max()
        if depth_max > depth_min:
            out = np.zeros_like(depth_map_outlier, dtype=np.float64)
            valid = ~np.isnan(depth_map_outlier)
            out[valid] = (depth_map_outlier[valid] - depth_min) / (depth_max - depth_min) * 255
            depth_normalized = out.astype(np.uint8)
            
            canny_edges = cv2.Canny(depth_normalized, 50, 150)
            combined_edges = cv2.bitwise_or(binary_image, canny_edges)
        else:
            combined_edges = binary_image
    else:
        combined_edges = binary_image
    
    # Dilation to connect broken line segments
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=dilation_iterations)
    
    # Detect oblique lines
    lines_oblique = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_oblique_threshold,
        minLineLength=hough_oblique_min_length,
        maxLineGap=hough_oblique_max_gap
    )
    
    # Detect horizontal lines (tunable parameters)
    lines_horizontal = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_horizontal_threshold,
        minLineLength=hough_horizontal_min_length,
        maxLineGap=hough_horizontal_max_gap
    )
    
    # Detect vertical lines
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, hough_vertical_threshold)
    if lines_vertical is not None:
        max_rho = W
        lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= max_rho]
    
    # Separate positive and negative slope lines
    positive_lines = []
    negative_lines = []
    horizontal_lines = []
    
    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            
            if angle_positive_min <= angle <= angle_positive_max:
                positive_lines.append(line[0])
            elif angle_negative_min <= angle <= angle_negative_max:
                negative_lines.append(line[0])
    
    if lines_horizontal is not None:
        for line in lines_horizontal:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -horizontal_angle_tolerance <= angle <= horizontal_angle_tolerance:
                horizontal_lines.append(line[0])
    
    # Process vertical lines - merge close ones (tunable threshold)
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
# Ring Center Calculation
# =============================================================================

def compute_ring_centers(
    line_data: Dict, 
    ring_count: int,
    ring_spacing: float,
    resolution: float
) -> List[float]:
    """Compute ring center X positions from vertical lines."""
    L, W = line_data['image_height'], line_data['image_width']
    expected_ring_width_px = ring_spacing / resolution  # Expected ring width in pixels
    vertical_lines = line_data['vertical_lines']
    
    if not vertical_lines:
        print("No vertical lines detected. Using fallback method.")
        block_width = W / ring_count
        return [(i + 0.5) * block_width for i in range(ring_count)]
    
    # Calculate midpoints between adjacent vertical lines
    mid_lines = []
    for i in range(len(vertical_lines) - 1):
        rho1, theta1 = vertical_lines[i]
        rho2, theta2 = vertical_lines[i + 1]
        new_rho = (rho1 + rho2) / 2
        new_theta = (theta1 + theta2) / 2
        a = np.cos(new_theta)
        x_pos = a * new_rho
        mid_lines.append((x_pos, new_theta))
    
    if len(mid_lines) == 0:
        block_width = W / ring_count
        return [(i + 0.5) * block_width for i in range(ring_count)]
    
    # Calculate average distance
    x_positions = [x for x, _ in mid_lines]
    distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
    avg_distance_detected = np.mean(distances) if distances else 0
    avg_distance_designed = W / ring_count
    
    if abs(avg_distance_detected - expected_ring_width_px) <= abs(avg_distance_designed - expected_ring_width_px):
        avg_distance = avg_distance_detected
    else:
        avg_distance = avg_distance_designed
    
    # Extend to cover all rings
    all_mid_lines = list(mid_lines)
    
    if mid_lines:
        # Extend left
        leftmost_x, leftmost_theta = mid_lines[0]
        x = leftmost_x - avg_distance
        while x >= 0:
            all_mid_lines.insert(0, (x, leftmost_theta))
            x -= avg_distance
        
        # Extend right
        rightmost_x, rightmost_theta = mid_lines[-1]
        x = rightmost_x + avg_distance
        while x <= W:
            all_mid_lines.append((x, rightmost_theta))
            x += avg_distance
    
    all_mid_lines = sorted(list(set(all_mid_lines)), key=lambda line: line[0])
    x_positions = [x for x, _ in all_mid_lines]
    
    return x_positions


# =============================================================================
# Complex-Specific Functions
# =============================================================================

def extend_line_to_bounds(x1, y1, x2, y2, W, L):
    """Extend a line segment to the image boundaries."""
    if x2 == x1:
        return x1, 0, x2, L
    
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    # Find intersections with image boundaries
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


def detect_oblique_lines_wide_angle(
    dilated_edges: np.ndarray,
    L: int,
    W: int,
    params: Dict
) -> Tuple[List, List]:
    """Detect oblique lines with wider angle range for complex_staggered patterns."""
    threshold = get_param(params, 'complex_hough_threshold', DEFAULT_COMPLEX_HOUGH_THRESHOLD)
    min_length = get_param(params, 'complex_hough_min_length', DEFAULT_COMPLEX_HOUGH_MIN_LENGTH)
    max_gap = get_param(params, 'complex_hough_max_gap', DEFAULT_COMPLEX_HOUGH_MAX_GAP)
    
    angle_pos_min = get_param(params, 'complex_angle_pos_min', DEFAULT_COMPLEX_ANGLE_POS_MIN)
    angle_pos_max = get_param(params, 'complex_angle_pos_max', DEFAULT_COMPLEX_ANGLE_POS_MAX)
    angle_neg_min = get_param(params, 'complex_angle_neg_min', DEFAULT_COMPLEX_ANGLE_NEG_MIN)
    angle_neg_max = get_param(params, 'complex_angle_neg_max', DEFAULT_COMPLEX_ANGLE_NEG_MAX)
    min_y_span = get_param(params, 'complex_min_y_span', DEFAULT_COMPLEX_MIN_Y_SPAN)
    min_x_span = get_param(params, 'complex_min_x_span', DEFAULT_COMPLEX_MIN_X_SPAN)
    
    lines_oblique = cv2.HoughLinesP(
        dilated_edges, 1, np.pi/180,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap
    )
    
    positive_lines = []
    negative_lines = []
    
    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            y_span = abs(y2 - y1)
            x_span = abs(x2 - x1)
            
            if angle_pos_min <= angle <= angle_pos_max and y_span >= min_y_span and x_span >= min_x_span:
                positive_lines.append((x1, y1, x2, y2))
            elif angle_neg_min <= angle <= angle_neg_max and y_span >= min_y_span and x_span >= min_x_span:
                negative_lines.append((x1, y1, x2, y2))
    
    return positive_lines, negative_lines


# =============================================================================
# K-Position Calculation (Complex Staggered)
# =============================================================================

def line_segment_vertical_intersection(vertical_x: float, segment: Tuple) -> Optional[float]:
    """Compute intersection of vertical line with line segment."""
    x1, y1, x2, y2 = segment
    if x1 == x2:
        return None
    if min(x1, x2) <= vertical_x <= max(x1, x2):
        t = (vertical_x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)
    return None


def merge_close_points(points: List[float]) -> List[float]:
    """Merge Y-values that are within threshold distance (FIXED)."""
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


def calculate_k_positions_complex_staggered(
    line_data: Dict,
    ring_count: int,
    k_height_mm: float,
    ab_height_mm: float,
    resolution: float,
    params: Dict
) -> pd.DataFrame:
    """
    Calculate K positions for complex_staggered patterns using oblique line intersections.
    
    Uses DBSCAN clustering and line subdivision to handle irregular patterns.
    """
    L = line_data['image_height']
    W = line_data['image_width']
    dilated_edges = line_data['dilated_edges']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    
    print(f"  [Complex Staggered Mode] Using oblique line intersections")
    print(f"    Initial lines: Positive={len(positive_lines)}, Negative={len(negative_lines)}")
    
    # Re-detect with wider angle range
    positive_lines, negative_lines = detect_oblique_lines_wide_angle(dilated_edges, L, W, params)
    print(f"    Re-detected with wider angles: Positive={len(positive_lines)}, Negative={len(negative_lines)}")
    
    # Extend lines to boundaries
    extended_positive = [extend_line_to_bounds(*line, W, L) for line in positive_lines]
    extended_negative = [extend_line_to_bounds(*line, W, L) for line in negative_lines]
    
    # Find intersections
    intersections = find_line_intersections(extended_positive, extended_negative, W, L)
    print(f"    Found {len(intersections)} line intersections")
    
    if len(intersections) == 0:
        # Fallback
        all_lines = positive_lines + negative_lines
        adjusted_points = []
        for x1, y1, x2, y2 in all_lines:
            adjusted_points.append(('midpoint_fallback', (x1+x2)/2, (y1+y2)/2, 0.3))
        if not adjusted_points:
            expected_ring_width = W / ring_count
            for i in range(ring_count):
                adjusted_points.append(('default', (i+0.5)*expected_ring_width, L/2, 0.1))
        df = pd.DataFrame(adjusted_points, columns=['Type', 'X', 'Y', 'Confidence'])
        return df.sort_values(by='X').reset_index(drop=True)
    
    # Cluster intersections
    intersection_array = np.array(intersections)
    expected_ring_width = W / ring_count
    x_normalized = intersection_array[:, 0] / W
    y_normalized = intersection_array[:, 1] / L
    features = np.column_stack([x_normalized, y_normalized])
    
    # Get clustering parameters (build eps_candidates from primary and secondary)
    eps_primary = get_param(params, 'complex_eps_primary', DEFAULT_COMPLEX_EPS_PRIMARY)
    eps_secondary = get_param(params, 'complex_eps_secondary', DEFAULT_COMPLEX_EPS_SECONDARY)
    eps_candidates = [eps_primary, eps_secondary, eps_secondary * 1.5, eps_secondary * 2.0, 0.15]
    eps_candidates = sorted(list(set([round(e, 2) for e in eps_candidates])))
    
    min_clusters = max(3, ring_count // 2)
    
    for eps in eps_candidates:
        clustering = DBSCAN(eps=eps, min_samples=1).fit(features)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= min(ring_count, min_clusters) or eps == eps_candidates[-1]:
            print(f"    Using eps={eps:.2f}, found {n_clusters} clusters")
            break
    
    # Get cluster centers
    unique_labels = set(labels)
    k_positions = []
    subdivision_threshold = get_param(params, 'complex_subdivision_threshold', DEFAULT_COMPLEX_SUBDIVISION_THRESHOLD)
    max_subdivisions = get_param(params, 'complex_max_subdivisions', DEFAULT_COMPLEX_MAX_SUBDIVISIONS)
    if max_subdivisions is None:
        max_subdivisions = ring_count // 2
    
    for label in unique_labels:
        cluster_points = intersection_array[labels == label]
        x_range = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
        n_subdivisions = max(1, int(x_range / (expected_ring_width * subdivision_threshold)) + 1)
        n_subdivisions = min(n_subdivisions, max_subdivisions)
        
        if n_subdivisions > 1:
            sorted_points = cluster_points[np.argsort(cluster_points[:, 0])]
            sub_size = len(sorted_points) // n_subdivisions
            for i in range(n_subdivisions):
                start = i * sub_size
                end = (i + 1) * sub_size if i < n_subdivisions - 1 else len(sorted_points)
                sub = sorted_points[start:end]
                conf_base = 0.5
                conf_factor = 0.05
                k_positions.append(('intersection_sub', np.mean(sub[:, 0]), np.mean(sub[:, 1]), 
                                   min(1.0, conf_base + conf_factor * len(sub))))
        else:
            conf_base = 0.5
            conf_factor = 0.1
            k_positions.append(('intersection', np.mean(cluster_points[:, 0]), np.mean(cluster_points[:, 1]),
                               min(1.0, conf_base + conf_factor * len(cluster_points))))
    
    k_positions.sort(key=lambda p: p[1])
    print(f"    Found {len(k_positions)} K position clusters from intersections")
    
    # Add line midpoints and cluster to get ring_count positions
    if len(k_positions) > 0:
        midpoint_confidence = get_param(params, 'complex_conf_midpoint', DEFAULT_COMPLEX_CONF_MIDPOINT)
        
        line_midpoints = []
        for x1, y1, x2, y2 in positive_lines:
            line_midpoints.append(('positive_midpoint', (x1+x2)/2, (y1+y2)/2, midpoint_confidence))
        for x1, y1, x2, y2 in negative_lines:
            line_midpoints.append(('negative_midpoint', (x1+x2)/2, (y1+y2)/2, midpoint_confidence))
        
        all_candidates = k_positions + line_midpoints
        
        if len(all_candidates) > ring_count:
            candidate_array = np.array([[p[1], p[2]] for p in all_candidates])
            x_norm = candidate_array[:, 0] / W
            y_norm = candidate_array[:, 1] / L
            features = np.column_stack([x_norm, y_norm])
            
            n_clusters = min(ring_count, len(all_candidates))
            clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(features)
            labels = clustering.labels_
            
            final_positions = []
            intersection_conf = get_param(params, 'complex_conf_intersection', DEFAULT_COMPLEX_CONF_INTERSECTION)
            midpoint_conf = 0.6
            for label in range(n_clusters):
                mask = labels == label
                cluster_points = candidate_array[mask]
                cluster_types = [all_candidates[i][0] for i, m in enumerate(mask) if m]
                det_type = 'intersection_cluster' if 'intersection' in str(cluster_types) else 'midpoint_cluster'
                confidence = intersection_conf if 'intersection' in str(cluster_types) else midpoint_conf
                final_positions.append((det_type, np.mean(cluster_points[:, 0]), 
                                       np.mean(cluster_points[:, 1]), confidence))
            k_positions = final_positions
        
        k_positions.sort(key=lambda p: p[1])
        print(f"    Final K positions: {len(k_positions)}")
    
    df = pd.DataFrame(k_positions, columns=['Type', 'X', 'Y', 'Confidence'])
    return df.sort_values(by='X').reset_index(drop=True)


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(
    line_data: Dict,
    k_positions: pd.DataFrame,
    tunnel_dir: str
) -> None:
    """Generate visualization of detected lines and K positions."""
    dilated_edges = line_data['dilated_edges']
    L, W = line_data['image_height'], line_data['image_width']
    
    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    
    # Colors
    color_positive = (255, 0, 0)    # Red
    color_negative = (0, 255, 0)    # Green
    color_horizontal = (0, 0, 255)  # Blue
    color_vertical = (255, 0, 255)  # Magenta
    line_thickness = 3
    
    # Draw positive slope lines (red)
    for x1, y1, x2, y2 in line_data['positive_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color_positive, line_thickness)
    
    # Draw negative slope lines (green)
    for x1, y1, x2, y2 in line_data['negative_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color_negative, line_thickness)
    
    # Draw horizontal lines (blue)
    for x1, y1, x2, y2 in line_data['horizontal_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color_horizontal, line_thickness)
    
    # Draw K positions (yellow circles) and vertical lines
    for _, row in k_positions.iterrows():
        cv2.circle(output_image, (int(row['X']), int(row['Y'])), 8, (0, 255, 255), -1)
        cv2.line(output_image, (int(row['X']), 0), (int(row['X']), L), color_vertical, 1)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(output_image)
    plt.title('Complex Staggered Detection Results')
    plt.savefig(os.path.join(tunnel_dir, 'detected_lines.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Detection Pipeline
# =============================================================================

def run_detection(tunnel_id: str, base_dir: str = "data") -> pd.DataFrame:
    """
    Execute the complete detection pipeline for complex_staggered patterns.
    
    CRITICAL PARAMETERS (~22 total):
    - Base detection (14): binary_threshold, hough_oblique_threshold, angle_positive/negative_min/max,
      hough_vertical_threshold, hough_horizontal_*, horizontal_angle_tolerance, merge_distance_threshold,
      dilation_*, hough_oblique_min_length/max_gap
    - Complex-specific (8+): complex_hough_threshold/min_length/max_gap, complex_angle_pos/neg_min/max,
      complex_min_y_span, complex_min_x_span, complex_eps_primary, complex_eps_secondary,
      complex_subdivision_threshold, complex_max_subdivisions, complex_conf_midpoint, complex_conf_intersection
    
    Args:
        tunnel_id: Identifier for the tunnel (e.g., "4-1", "5-1")
        base_dir: Base directory for data files
    """
    print(f"{'=' * 60}")
    print(f"Detection Pipeline: {tunnel_id}")
    print(f"{'=' * 60}")
    
    # Load parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    
    # Extract CRITICAL parameters (HIGH sensitivity)
    binary_threshold = get_param(params, 'binary_threshold', DEFAULT_BINARY_THRESHOLD)
    hough_oblique_threshold = get_param(params, 'hough_oblique_threshold', DEFAULT_HOUGH_OBLIQUE_THRESHOLD)
    angle_positive_min = get_param(params, 'angle_positive_min', DEFAULT_ANGLE_POSITIVE_MIN)
    angle_positive_max = get_param(params, 'angle_positive_max', DEFAULT_ANGLE_POSITIVE_MAX)
    angle_negative_min = get_param(params, 'angle_negative_min', DEFAULT_ANGLE_NEGATIVE_MIN)
    angle_negative_max = get_param(params, 'angle_negative_max', DEFAULT_ANGLE_NEGATIVE_MAX)
    hough_vertical_threshold = get_param(params, 'hough_vertical_threshold', DEFAULT_HOUGH_VERTICAL_THRESHOLD)
    
    # Physical constants - READ FROM PREPROCESSING STAGE (not duplicated here)
    preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
    tunnel_diameter = preprocessing_params.get('tunnel_diameter', 5.5)
    resolution = preprocessing_params.get('depth_map_resolution', 0.005)
    ring_spacing = preprocessing_params.get('ring_spacing', 1.2)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
    
    # Extract MEDIUM sensitivity parameters
    dilation_kernel_size = get_param(params, 'dilation_kernel_size', DEFAULT_DILATION_KERNEL_SIZE)
    dilation_iterations = get_param(params, 'dilation_iterations', DEFAULT_DILATION_ITERATIONS)
    hough_oblique_min_length = get_param(params, 'hough_oblique_min_length', DEFAULT_HOUGH_OBLIQUE_MIN_LENGTH)
    hough_oblique_max_gap = get_param(params, 'hough_oblique_max_gap', DEFAULT_HOUGH_OBLIQUE_MAX_GAP)
    
    # Extract merge/horizontal parameters (tunable)
    hough_horizontal_threshold = get_param(params, 'hough_horizontal_threshold', DEFAULT_HOUGH_HORIZONTAL_THRESHOLD)
    hough_horizontal_min_length = get_param(params, 'hough_horizontal_min_length', DEFAULT_HOUGH_HORIZONTAL_MIN_LENGTH)
    hough_horizontal_max_gap = get_param(params, 'hough_horizontal_max_gap', DEFAULT_HOUGH_HORIZONTAL_MAX_GAP)
    horizontal_angle_tolerance = get_param(params, 'horizontal_angle_tolerance', DEFAULT_HORIZONTAL_ANGLE_TOLERANCE)
    merge_distance_threshold = get_param(params, 'merge_distance_threshold', DEFAULT_MERGE_DISTANCE_THRESHOLD)
    
    # Extract complex-specific parameters (flat keys)
    complex_hough_threshold = get_param(params, 'complex_hough_threshold', DEFAULT_COMPLEX_HOUGH_THRESHOLD)
    complex_hough_min_length = get_param(params, 'complex_hough_min_length', DEFAULT_COMPLEX_HOUGH_MIN_LENGTH)
    complex_hough_max_gap = get_param(params, 'complex_hough_max_gap', DEFAULT_COMPLEX_HOUGH_MAX_GAP)
    complex_angle_pos_min = get_param(params, 'complex_angle_pos_min', DEFAULT_COMPLEX_ANGLE_POS_MIN)
    complex_angle_pos_max = get_param(params, 'complex_angle_pos_max', DEFAULT_COMPLEX_ANGLE_POS_MAX)
    complex_angle_neg_min = get_param(params, 'complex_angle_neg_min', DEFAULT_COMPLEX_ANGLE_NEG_MIN)
    complex_angle_neg_max = get_param(params, 'complex_angle_neg_max', DEFAULT_COMPLEX_ANGLE_NEG_MAX)
    complex_min_y_span = get_param(params, 'complex_min_y_span', DEFAULT_COMPLEX_MIN_Y_SPAN)
    complex_min_x_span = get_param(params, 'complex_min_x_span', DEFAULT_COMPLEX_MIN_X_SPAN)
    complex_eps_primary = get_param(params, 'complex_eps_primary', DEFAULT_COMPLEX_EPS_PRIMARY)
    complex_eps_secondary = get_param(params, 'complex_eps_secondary', DEFAULT_COMPLEX_EPS_SECONDARY)
    complex_subdivision_threshold = get_param(params, 'complex_subdivision_threshold', DEFAULT_COMPLEX_SUBDIVISION_THRESHOLD)
    complex_max_subdivisions = get_param(params, 'complex_max_subdivisions', DEFAULT_COMPLEX_MAX_SUBDIVISIONS)
    complex_conf_midpoint = get_param(params, 'complex_conf_midpoint', DEFAULT_COMPLEX_CONF_MIDPOINT)
    complex_conf_intersection = get_param(params, 'complex_conf_intersection', DEFAULT_COMPLEX_CONF_INTERSECTION)
    
    # Build complex params dict for complex functions
    complex_params = {
        'complex_hough_threshold': complex_hough_threshold,
        'complex_hough_min_length': complex_hough_min_length,
        'complex_hough_max_gap': complex_hough_max_gap,
        'complex_angle_pos_min': complex_angle_pos_min,
        'complex_angle_pos_max': complex_angle_pos_max,
        'complex_angle_neg_min': complex_angle_neg_min,
        'complex_angle_neg_max': complex_angle_neg_max,
        'complex_min_y_span': complex_min_y_span,
        'complex_min_x_span': complex_min_x_span,
        'complex_eps_primary': complex_eps_primary,
        'complex_eps_secondary': complex_eps_secondary,
        'complex_subdivision_threshold': complex_subdivision_threshold,
        'complex_max_subdivisions': complex_max_subdivisions,
        'complex_conf_midpoint': complex_conf_midpoint,
        'complex_conf_intersection': complex_conf_intersection,
    }
    
    print("\nCritical parameters (HIGH sensitivity):")
    print(f"  binary_threshold:        {binary_threshold}")
    print(f"  hough_oblique_threshold: {hough_oblique_threshold}")
    print(f"  angle_positive_min:      {angle_positive_min}°")
    print(f"  angle_positive_max:      {angle_positive_max}°")
    print(f"  angle_negative_min:      {angle_negative_min}°")
    print(f"  angle_negative_max:      {angle_negative_max}°")
    print(f"  hough_vertical_threshold: {hough_vertical_threshold}")
    print(f"\nPhysical constants (from preprocessing stage):")
    print(f"  tunnel_diameter:         {tunnel_diameter}m")
    print(f"  ring_spacing:            {ring_spacing}m")
    print(f"  resolution:              {resolution}")
    print(f"  k_height_mm (calculated): {k_height_mm:.2f}")
    print(f"  ab_height_mm (calculated): {ab_height_mm:.2f}")
    print("\nMedium sensitivity parameters:")
    print(f"  dilation_kernel_size:    {dilation_kernel_size}")
    print(f"  dilation_iterations:     {dilation_iterations}")
    print(f"  hough_oblique_min_length: {hough_oblique_min_length}")
    print(f"  hough_oblique_max_gap:   {hough_oblique_max_gap}")
    print("\nMerge/horizontal parameters:")
    print(f"  hough_horizontal_threshold: {hough_horizontal_threshold}")
    print(f"  hough_horizontal_min_length: {hough_horizontal_min_length}")
    print(f"  hough_horizontal_max_gap:   {hough_horizontal_max_gap}")
    print(f"  horizontal_angle_tolerance: {horizontal_angle_tolerance}")
    print(f"  merge_distance_threshold:   {merge_distance_threshold}")
    print("\nComplex-specific parameters:")
    print(f"  complex_hough_threshold: {complex_hough_threshold}")
    print(f"  complex_hough_min_length: {complex_hough_min_length}")
    print(f"  complex_hough_max_gap: {complex_hough_max_gap}")
    print(f"  complex_angle_pos_min/max: {complex_angle_pos_min}°/{complex_angle_pos_max}°")
    print(f"  complex_angle_neg_min/max: {complex_angle_neg_min}°/{complex_angle_neg_max}°")
    print(f"  complex_min_y_span: {complex_min_y_span}, complex_min_x_span: {complex_min_x_span}")
    print(f"  complex_eps_primary: {complex_eps_primary}, complex_eps_secondary: {complex_eps_secondary}")
    print(f"  complex_subdivision_threshold: {complex_subdivision_threshold}")
    print(f"  complex_max_subdivisions: {complex_max_subdivisions}")
    print(f"  complex_conf_midpoint: {complex_conf_midpoint}, complex_conf_intersection: {complex_conf_intersection}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load data from preprocessing stage
    depth_map_outlier_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_map_outlier_path):
        raise FileNotFoundError(f"depth_map_outlier.npy not found at {depth_map_outlier_path}. Run preprocessing first.")
    
    depth_map_outlier = np.load(depth_map_outlier_path)
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    L, W = depth_map_outlier.shape
    
    print(f"\n[Step 1] Detecting lines...")
    line_data = detect_lines(
        depth_map_outlier,
        binary_threshold=binary_threshold,
        hough_oblique_threshold=hough_oblique_threshold,
        angle_positive_min=angle_positive_min,
        angle_positive_max=angle_positive_max,
        angle_negative_min=angle_negative_min,
        angle_negative_max=angle_negative_max,
        hough_vertical_threshold=hough_vertical_threshold,
        dilation_kernel_size=dilation_kernel_size,
        dilation_iterations=dilation_iterations,
        hough_oblique_min_length=hough_oblique_min_length,
        hough_oblique_max_gap=hough_oblique_max_gap,
        hough_horizontal_threshold=hough_horizontal_threshold,
        hough_horizontal_min_length=hough_horizontal_min_length,
        hough_horizontal_max_gap=hough_horizontal_max_gap,
        horizontal_angle_tolerance=horizontal_angle_tolerance,
        merge_distance_threshold=merge_distance_threshold
    )
    print(f"  Positive slope lines: {len(line_data['positive_lines'])}")
    print(f"  Negative slope lines: {len(line_data['negative_lines'])}")
    print(f"  Horizontal lines: {len(line_data['horizontal_lines'])}")
    print(f"  Vertical lines: {len(line_data['vertical_lines'])}")
    
    print(f"\n[Step 2] Calculating K positions (complex staggered)...")
    k_positions = calculate_k_positions_complex_staggered(
        line_data, ring_count, k_height_mm, ab_height_mm, resolution, complex_params
    )
    print(f"  Calculated {len(k_positions)} K positions")
    print(f"  Detection types: {k_positions['Type'].value_counts().to_dict()}")
    if 'Confidence' in k_positions.columns:
        print(f"  Average confidence: {k_positions['Confidence'].mean():.3f}")
        print(f"  Confidence range: [{k_positions['Confidence'].min():.3f}, {k_positions['Confidence'].max():.3f}]")
    
    # Save results
    k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)
    print(f"\n  Saved: {os.path.join(tunnel_dir, 'detected.csv')}")
    
    # Generate visualization
    visualize_detection(line_data, k_positions, tunnel_dir)
    print(f"  Saved: {os.path.join(tunnel_dir, 'detected_lines.png')}")
    
    print(f"\n{'=' * 60}")
    print(f"Detection complete!")
    print(f"{'=' * 60}")
    
    print("\nK Position Summary:")
    print(k_positions.to_string(index=False))
    
    return k_positions


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Complex staggered detection pipeline")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    run_detection(args.tunnel_id, base_dir=args.data_dir)

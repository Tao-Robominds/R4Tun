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
import math
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
    params: Dict
) -> Dict:
    """
    Unified line detection: one Hough call with unified parameters.
    
    BO-tunable parameters (8D):
    - binary_threshold: Edge detection sensitivity
    - hough_threshold: Oblique line votes (replaces hough_oblique_threshold and complex_hough_threshold)
    - hough_min_length: Oblique min line length
    - hough_max_gap: Oblique max gap
    - angle_pos_min/max: Positive slope filter (wide range: 4.84-13.55°)
    - angle_neg_min/max: Negative slope filter (wide range: -14.67 to -5.82°)
    
    Baked parameters (8): dilation_kernel_size=2, dilation_iterations=4,
    hough_vertical_threshold=776, hough_horizontal_*=20/98/18,
    horizontal_angle_tolerance=0.79, merge_distance_threshold=3.85
    """
    L, W = depth_map_outlier.shape
    
    # BO-tunable edge detection
    binary_threshold = params.get('binary_threshold', 139)
    
    # Baked dilation parameters
    dilation_kernel_size = 2
    dilation_iterations = 4
    
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
    
    # Dilation to connect broken line segments (baked)
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=dilation_iterations)
    
    # ONE Hough call for oblique lines (BO-tunable)
    hough_threshold = params.get('hough_threshold', 37)
    hough_min_length = params.get('hough_min_length', 31)
    hough_max_gap = params.get('hough_max_gap', 133)
    
    lines_oblique = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_threshold,
        minLineLength=hough_min_length,
        maxLineGap=hough_max_gap
    )
    
    # Detect horizontal lines (baked parameters)
    hough_horizontal_threshold = 20
    hough_horizontal_min_length = 98
    hough_horizontal_max_gap = 18
    horizontal_angle_tolerance = 0.79
    
    lines_horizontal = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180,
        hough_horizontal_threshold,
        minLineLength=hough_horizontal_min_length,
        maxLineGap=hough_horizontal_max_gap
    )
    
    # Detect vertical lines (baked)
    hough_vertical_threshold = 776
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, hough_vertical_threshold)
    if lines_vertical is not None:
        max_rho = W
        lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= max_rho]
    
    # ONE angle filter for oblique lines (BO-tunable, wide range)
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
    
    # Process vertical lines - merge close ones (baked threshold)
    merge_distance_threshold = 3.85
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
    print(f"    Unified lines: Positive={len(positive_lines)}, Negative={len(negative_lines)}")
    
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
    
    # Get clustering parameters (using unified eps)
    eps = get_param(params, 'eps', 0.07)
    eps_candidates = [eps, eps * 1.5, eps * 2.0, 0.10, 0.15]
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


def calculate_k_positions_banded(
    line_data: Dict,
    ring_count: int,
    params: Dict
) -> pd.DataFrame:
    """Calculate K positions using evenly-spaced ring bands.

    1. X = band center = (i + 0.5) * W / ring_count
    2. Y = median Y of oblique line intersections within band (+/- 0.6 * ring_width)
    3. Fallback: line crossings at band center
    4. Fallback: interpolate Y from neighboring bands

    Args:
        line_data: Output from detect_lines().
        ring_count: Number of rings.
        params: Detection parameters dict (may include step_template).

    Returns:
        DataFrame with columns Type, X, Y, Confidence.
    """
    L = line_data['image_height']
    W = line_data['image_width']
    dilated_edges = line_data['dilated_edges']

    print(f"  [Banded K Detection] ring_count={ring_count}, image={L}x{W}")

    # Use unified oblique lines from detect_lines (already in line_data)
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    print(f"    Oblique lines: Positive={len(positive_lines)}, Negative={len(negative_lines)}")

    extended_positive = [extend_line_to_bounds(*line, W, L) for line in positive_lines]
    extended_negative = [extend_line_to_bounds(*line, W, L) for line in negative_lines]

    intersections = find_line_intersections(extended_positive, extended_negative, W, L)
    print(f"    Line intersections: {len(intersections)}")

    ring_width = W / ring_count
    band_margin = ring_width * 0.6
    all_extended = extended_positive + extended_negative

    band_ys = {}
    for i in range(ring_count):
        band_center = (i + 0.5) * ring_width
        band_left = band_center - band_margin
        band_right = band_center + band_margin

        # Primary: oblique line intersections within this band
        band_pts_y = [y for x, y in intersections
                      if band_left <= x <= band_right]

        if len(band_pts_y) >= 3:
            k_y = float(np.median(band_pts_y))
            conf = min(1.0, 0.5 + 0.05 * len(band_pts_y))
            det_type = 'band_intersection'
        else:
            # Fallback: oblique line crossings at vertical x=band_center
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

    # Interpolation fallback for bands with no detection
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
    for i, (dt, x, y, c) in enumerate(k_positions):
        print(f"    Band {i}: X={x:.0f}, Y={y:.0f}, conf={c:.2f} [{dt}]")

    df = pd.DataFrame(k_positions, columns=['Type', 'X', 'Y', 'Confidence'])
    return df.sort_values(by='X').reset_index(drop=True)


def _line_crossing_y_at_x(x1: float, y1: float, x2: float, y2: float, x: float) -> Optional[float]:
    """Y coordinate where the line through (x1,y1)-(x2,y2) crosses vertical x. Extrapolates if needed."""
    if x2 == x1:
        return None
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def _cluster_1d_gap(values: np.ndarray, gap_px: float = 150.0) -> List[np.ndarray]:
    """Cluster 1D values by gap; consecutive values within gap_px stay in same cluster."""
    if len(values) == 0:
        return []
    s = np.sort(values)
    clusters = []
    current = [s[0]]
    for i in range(1, len(s)):
        if s[i] - current[-1] <= gap_px:
            current.append(s[i])
        else:
            clusters.append(np.array(current))
            current = [s[i]]
    clusters.append(np.array(current))
    return clusters




# =============================================================================
# Segment Expansion: K → All Segments
# =============================================================================

def calculate_k_positions_groove_pair(
    line_data: Dict,
    ring_count: int,
    k_height_mm: float,
    resolution: float,
    params: Dict,
) -> pd.DataFrame:
    """Locate K Y by finding the closest positive+negative oblique line pair
    whose gap matches the expected K block height.

    Algorithm:
      1. Detect wide-angle oblique lines; for each ring X band, collect
         positive and negative line crossings at band center.
      2. Enumerate all (pos, neg) crossing pairs.  Score each by
         ``|gap - k_expected_height_px|``.
      3. Keep top-N candidates per ring.
      4. Disambiguate via **groove alignment scoring**: expand each candidate
         K Y with grouped offsets and count how many expanded positions have
         a groove crossing within ``groove_snap_px``.
      5. Select the candidate with the highest groove alignment count.

    All new parameters are BO-tunable:
      - ``k_expected_height_px``  (derived from K_height_mm, BO can fine-tune)
      - ``k_gap_tolerance_px``    (max allowed deviation from expected height)
      - ``k_candidates_per_ring`` (top-N for disambiguation)
      - ``groove_snap_px``        (proximity threshold for groove alignment)

    Returns:
        DataFrame with columns Type, X, Y, Confidence (one row per ring).
    """
    L = line_data['image_height']
    W = line_data['image_width']

    ring_offset = params.get('ring_offset', W / (2 * ring_count))
    ring_spacing_px = params.get('ring_spacing_px', W / ring_count)
    half_width = abs(ring_spacing_px) * 0.5

    # BO-tunable groove-pair parameters
    k_expected_height_px = params.get(
        'k_expected_height_px',
        (k_height_mm / 1000.0) / resolution / 2.0  # default: half the angular span → ~300px
    )
    k_gap_tolerance_px = params.get('k_gap_tolerance_px', 150.0)
    k_candidates_per_ring = int(params.get('k_candidates_per_ring', 8))
    groove_snap_px = params.get('groove_snap_px', 60.0)

    # Use unified oblique lines from detect_lines (single Hough call, wide angle range)
    positive_lines_wide = line_data['positive_lines']
    negative_lines_wide = line_data['negative_lines']
    # Same lines used for both candidate generation and groove alignment scoring
    std_oblique = positive_lines_wide + negative_lines_wide

    print(f"  [Groove-Pair K Detection] ring_count={ring_count}, image={L}x{W}")
    print(f"    Wide-angle lines: Positive={len(positive_lines_wide)}, Negative={len(negative_lines_wide)}")
    print(f"    Standard oblique lines (for groove scoring): {len(std_oblique)}")
    print(f"    k_expected_height_px={k_expected_height_px:.0f}, gap_tol={k_gap_tolerance_px:.0f}, "
          f"candidates={k_candidates_per_ring}, groove_snap={groove_snap_px:.0f}")

    # Grouped offsets for groove alignment scoring
    stagger_groups = params.get('stagger_groups', {})
    group_offsets = params.get('group_offsets', {})
    ring_to_group: Dict[int, str] = {}
    default_group = list(stagger_groups.keys())[0] if stagger_groups else "A"
    for grp, ring_list in stagger_groups.items():
        for r in ring_list:
            ring_to_group[r] = grp
    expansion_blocks = ['B1', 'B2', 'A1', 'A2', 'A3', 'A4']

    def _groove_crossings_at_x(x_center: float) -> List[float]:
        """Y positions where *standard* oblique lines cross vertical x=x_center.

        Standard lines are sparser/higher-quality than wide-angle lines,
        preventing every candidate from trivially matching all grooves.
        """
        crossings = []
        for x1, y1, x2, y2 in std_oblique:
            y_c = _line_crossing_y_at_x(x1, y1, x2, y2, x_center)
            if y_c is not None and 0 <= y_c <= L:
                crossings.append(y_c)
        return sorted(crossings)

    def _wrap_dist_y(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, L - d)

    def _groove_alignment_score(
        k_y: float,
        ring_idx: int,
        groove_ys: List[float],
    ) -> float:
        """Compute groove alignment score for a K-Y candidate.

        For each of the 6 expanded non-K block positions, find the minimum
        distance to the nearest standard groove crossing.  The score is the
        number of blocks whose nearest groove is within ``groove_snap_px``
        (the *count* component) plus a proximity bonus (sum of
        (groove_snap_px - min_dist) / groove_snap_px for each hit, giving
        continuous gradients for BO).
        """
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
        return total  # range [0, 12] (2 points per aligned block)

    def _wrap_midpoint(a: float, b: float) -> float:
        """Wrap-aware midpoint of two Y values on a cylinder of height L."""
        if abs(a - b) <= L / 2:
            return (a + b) / 2.0
        return ((a + b) / 2.0 + L / 2.0) % L

    # Fallback: banded result for rings with no viable candidates
    banded_df = calculate_k_positions_banded(line_data, ring_count, params)

    k_positions = []
    groove_scores = []
    for i in range(ring_count):
        band_center = ring_offset + i * ring_spacing_px

        # Collect wide-angle positive and negative crossings within the band
        pos_crossings = []
        for x1, y1, x2, y2 in positive_lines_wide:
            mid_x = (x1 + x2) / 2.0
            if band_center - half_width <= mid_x <= band_center + half_width:
                y_c = _line_crossing_y_at_x(x1, y1, x2, y2, band_center)
                if y_c is not None and 0 <= y_c <= L:
                    pos_crossings.append(y_c)

        neg_crossings = []
        for x1, y1, x2, y2 in negative_lines_wide:
            mid_x = (x1 + x2) / 2.0
            if band_center - half_width <= mid_x <= band_center + half_width:
                y_c = _line_crossing_y_at_x(x1, y1, x2, y2, band_center)
                if y_c is not None and 0 <= y_c <= L:
                    neg_crossings.append(y_c)

        # Enumerate all (pos, neg) pairs; filter by gap proximity to K height
        candidates = []  # (gap_err, gap, midpoint_y)
        for py in pos_crossings:
            for ny in neg_crossings:
                gap = _wrap_dist_y(py, ny)
                gap_err = abs(gap - k_expected_height_px)
                if gap_err > k_gap_tolerance_px:
                    continue
                mid = _wrap_midpoint(py, ny)
                candidates.append((gap_err, gap, mid))

        # Deduplicate: merge candidates whose midpoints are within 30px
        candidates.sort(key=lambda c: c[0])
        deduped = []
        for gap_err, gap, mid in candidates:
            is_dup = False
            for _, _, existing_mid in deduped:
                if _wrap_dist_y(mid, existing_mid) < 30:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append((gap_err, gap, mid))
            if len(deduped) >= k_candidates_per_ring:
                break
        candidates = deduped

        if candidates:
            # Score candidates with combined metric:
            # combined = groove_alignment_score - gap_err_penalty
            # groove_alignment uses *standard* oblique lines (sparse/discriminating)
            groove_ys = _groove_crossings_at_x(band_center)
            best_mid = candidates[0][2]
            best_combined = -float('inf')
            best_groove = 0.0
            for gap_err, gap, mid in candidates:
                groove = _groove_alignment_score(mid, i, groove_ys)
                # gap_err penalty: normalised to [0, 1] by tolerance
                gap_penalty = gap_err / k_gap_tolerance_px
                combined = groove - gap_penalty
                if combined > best_combined:
                    best_combined = combined
                    best_mid = mid
                    best_groove = groove
            y_k = best_mid
            conf = min(1.0, 0.5 + 0.04 * best_groove)
            det_type = 'groove_pair'
            groove_scores.append(best_groove)
        else:
            # Fallback: banded median
            row = banded_df.iloc[i]
            y_k = row['Y']
            conf = float(row['Confidence']) * 0.4
            det_type = 'groove_pair_fallback'
            groove_scores.append(0.0)

        k_positions.append((det_type, band_center, y_k, conf))
        print(f"    Ring {i}: X={band_center:.0f}, Y={y_k:.0f}, conf={conf:.2f} [{det_type}] "
              f"(cands={len(candidates)}, groove={groove_scores[-1]:.1f})")

    # Report summary
    groove_total = sum(groove_scores)
    groove_max = 12.0 * ring_count  # max 12 per ring (2 pts per aligned block × 6 blocks)
    print(f"    Groove alignment: {groove_total:.1f}/{groove_max:.0f} "
          f"({groove_total / groove_max * 100:.1f}%)")

    df = pd.DataFrame(k_positions, columns=['Type', 'X', 'Y', 'Confidence'])
    # Attach groove alignment metadata for intrinsic scoring
    df.attrs['groove_alignment_total'] = float(groove_total)
    df.attrs['groove_alignment_max'] = float(groove_max)
    df.attrs['groove_alignment_pct'] = (
        groove_total / groove_max * 100 if groove_max > 0 else 0.0
    )
    df.attrs['groove_scores_per_ring'] = [float(s) for s in groove_scores]
    return df.sort_values(by='X').reset_index(drop=True)


def calculate_k_positions_combined(
    line_data: Dict,
    ring_count: int,
    k_height_mm: float,
    resolution: float,
    params: Dict,
) -> pd.DataFrame:
    """
    Combined K Y detection: fuses DBSCAN (coverage) + groove_pair (precision).
    
    Algorithm:
      1. DBSCAN path: Find pos-neg line intersections, cluster with DBSCAN (eps),
         get (X, Y) per cluster, assign to nearest ring band.
      2. Groove-pair path: For each ring X band, find pos-neg crossing pairs with
         gap ~ k_expected_height_px, keep top candidates.
      3. Per-ring fusion: Collect candidates from both methods, score each using
         groove alignment (expand with offsets, count groove crossings within
         groove_snap_px), select best.
    
    Both paths use the SAME oblique lines from unified detect_lines().
    
    Returns:
        DataFrame with columns Type, X, Y, Confidence (one row per ring).
    """
    L = line_data['image_height']
    W = line_data['image_width']
    dilated_edges = line_data['dilated_edges']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    
    ring_offset = params.get('ring_offset', W / (2 * ring_count))
    ring_spacing_px = params.get('ring_spacing_px', W / ring_count)
    half_width = abs(ring_spacing_px) * 0.5
    
    # BO-tunable parameters
    eps = params.get('eps', 0.07)
    k_expected_height_px = params.get(
        'k_expected_height_px',
        (k_height_mm / 1000.0) / resolution / 2.0
    )
    k_gap_tolerance_px = params.get('k_gap_tolerance_px', 150.0)
    k_candidates_per_ring = int(params.get('k_candidates_per_ring', 8))
    groove_snap_px = params.get('groove_snap_px', 60.0)
    
    print(f"  [Combined K Detection] ring_count={ring_count}, image={L}x{W}")
    print(f"    Oblique lines: Positive={len(positive_lines)}, Negative={len(negative_lines)}")
    print(f"    eps={eps:.3f}, k_expected_height_px={k_expected_height_px:.0f}, "
          f"gap_tol={k_gap_tolerance_px:.0f}, candidates={k_candidates_per_ring}, "
          f"groove_snap={groove_snap_px:.0f}")
    
    # Grouped offsets for groove alignment scoring
    stagger_groups = params.get('stagger_groups', {})
    group_offsets = params.get('group_offsets', {})
    ring_to_group: Dict[int, str] = {}
    default_group = list(stagger_groups.keys())[0] if stagger_groups else "A"
    for grp, ring_list in stagger_groups.items():
        for r in ring_list:
            ring_to_group[r] = grp
    expansion_blocks = ['B1', 'B2', 'A1', 'A2', 'A3', 'A4']
    
    def _line_crossing_y_at_x(x1: float, y1: float, x2: float, y2: float, x: float) -> Optional[float]:
        """Y coordinate where line crosses vertical x."""
        if x2 == x1:
            return None
        t = (x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)
    
    def _wrap_dist_y(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, L - d)
    
    def _wrap_midpoint(a: float, b: float) -> float:
        """Wrap-aware midpoint of two Y values."""
        if abs(a - b) <= L / 2:
            return (a + b) / 2.0
        return ((a + b) / 2.0 + L / 2.0) % L
    
    def _groove_crossings_at_x(x_center: float) -> List[float]:
        """Y positions where oblique lines cross vertical x=x_center."""
        crossings = []
        all_lines = positive_lines + negative_lines
        for x1, y1, x2, y2 in all_lines:
            y_c = _line_crossing_y_at_x(x1, y1, x2, y2, x_center)
            if y_c is not None and 0 <= y_c <= L:
                crossings.append(y_c)
        return sorted(crossings)
    
    def _groove_alignment_score(
        k_y: float,
        ring_idx: int,
        groove_ys: List[float],
    ) -> float:
        """Compute groove alignment score for a K-Y candidate."""
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
    
    # DBSCAN path: find intersections, cluster, assign to ring bands
    extended_positive = [extend_line_to_bounds(*line, W, L) for line in positive_lines]
    extended_negative = [extend_line_to_bounds(*line, W, L) for line in negative_lines]
    intersections = find_line_intersections(extended_positive, extended_negative, W, L)
    print(f"    DBSCAN path: {len(intersections)} intersections found")
    
    dbscan_candidates = {}  # ring_idx -> list of (X, Y, confidence)
    if len(intersections) > 0:
        intersection_array = np.array(intersections)
        x_normalized = intersection_array[:, 0] / W
        y_normalized = intersection_array[:, 1] / L
        features = np.column_stack([x_normalized, y_normalized])
        
        # Cluster with unified eps
        clustering = DBSCAN(eps=eps, min_samples=1).fit(features)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}
        
        for label in unique_labels:
            cluster_points = intersection_array[labels == label]
            cluster_x = np.mean(cluster_points[:, 0])
            cluster_y = np.mean(cluster_points[:, 1])
            conf = min(1.0, 0.5 + 0.05 * len(cluster_points))
            
            # Assign to nearest ring band
            ring_idx = int(round((cluster_x - ring_offset) / ring_spacing_px))
            ring_idx = max(0, min(ring_count - 1, ring_idx))
            
            if ring_idx not in dbscan_candidates:
                dbscan_candidates[ring_idx] = []
            dbscan_candidates[ring_idx].append((cluster_x, cluster_y, conf))
    
    print(f"    DBSCAN candidates: {len(dbscan_candidates)} rings with candidates")
    
    # Groove-pair path: same as groove_pair method
    groove_pair_candidates = {}  # ring_idx -> list of (gap_err, gap, midpoint_y)
    for i in range(ring_count):
        band_center = ring_offset + i * ring_spacing_px
        
        pos_crossings = []
        for x1, y1, x2, y2 in positive_lines:
            mid_x = (x1 + x2) / 2.0
            if band_center - half_width <= mid_x <= band_center + half_width:
                y_c = _line_crossing_y_at_x(x1, y1, x2, y2, band_center)
                if y_c is not None and 0 <= y_c <= L:
                    pos_crossings.append(y_c)
        
        neg_crossings = []
        for x1, y1, x2, y2 in negative_lines:
            mid_x = (x1 + x2) / 2.0
            if band_center - half_width <= mid_x <= band_center + half_width:
                y_c = _line_crossing_y_at_x(x1, y1, x2, y2, band_center)
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
        
        # Deduplicate
        candidates.sort(key=lambda c: c[0])
        deduped = []
        for gap_err, gap, mid in candidates:
            is_dup = False
            for _, _, existing_mid in deduped:
                if _wrap_dist_y(mid, existing_mid) < 30:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append((gap_err, gap, mid))
            if len(deduped) >= k_candidates_per_ring:
                break
        groove_pair_candidates[i] = deduped
    
    # Per-ring fusion: collect all candidates, score with groove alignment, select best
    k_positions = []
    groove_scores = []
    banded_df = calculate_k_positions_banded(line_data, ring_count, params)
    
    for i in range(ring_count):
        band_center = ring_offset + i * ring_spacing_px
        groove_ys = _groove_crossings_at_x(band_center)
        
        # Collect all candidates
        all_candidates = []
        dbscan_count = 0
        groove_pair_count = 0
        
        # DBSCAN candidates
        if i in dbscan_candidates:
            for x, y, conf in dbscan_candidates[i]:
                all_candidates.append(('dbscan', x, y, conf))
                dbscan_count += 1
        
        # Groove-pair candidates
        if i in groove_pair_candidates:
            for gap_err, gap, mid in groove_pair_candidates[i]:
                all_candidates.append(('groove_pair', band_center, mid, 0.8))
                groove_pair_count += 1
        
        # Debug: show candidate counts
        if dbscan_count > 0 or groove_pair_count > 0:
            print(f"      Ring {i}: {dbscan_count} DBSCAN + {groove_pair_count} groove_pair candidates")
        
        if not all_candidates:
            # Fallback: banded
            row = banded_df.iloc[i]
            y_k = row['Y']
            conf = float(row['Confidence']) * 0.3
            det_type = 'combined_fallback'
            groove_scores.append(0.0)
        else:
            # Score each candidate with groove alignment
            best_candidate = None
            best_score = -float('inf')
            best_groove = 0.0
            
            for det_type, x, y, base_conf in all_candidates:
                groove = _groove_alignment_score(y, i, groove_ys)
                # Combined score: groove alignment - gap_err penalty (if groove_pair)
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
        print(f"    Ring {i}: X={band_center:.0f}, Y={y_k:.0f}, conf={conf:.2f} [{det_type}] "
              f"(cands={len(all_candidates)}, groove={groove_scores[-1]:.1f})")
    
    # Report summary
    groove_total = sum(groove_scores)
    groove_max = 12.0 * ring_count
    print(f"    Groove alignment: {groove_total:.1f}/{groove_max:.0f} "
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



# Non-K block names for grouped offset expansion
EXPANSION_BLOCKS = ['B1', 'B2', 'A1', 'A2', 'A3', 'A4']


def expand_k_with_grouped_offsets(
    k_positions: pd.DataFrame,
    img_height: int,
    stagger_groups: Dict[str, list],
    group_offsets: Dict[str, float],
) -> pd.DataFrame:
    """Derive all segment positions from K using grouped offsets + stagger assignment.

    Each ring belongs to a stagger group (e.g. "A" or "B").  All rings in the
    same group share the same 6 Y-offsets from K.  This is BO-tunable at 12D
    (2 groups x 6 blocks) + a discrete stagger assignment.

    Args:
        k_positions: DataFrame with columns Type, X, Y, Confidence (K-only).
        img_height: Depth map height in pixels (for Y wrap-around).
        stagger_groups: Maps group name -> list of ring indices, e.g.
            {"A": [0,1,2,3,4], "B": [5,6]}.
        group_offsets: Maps "{group}_{block}" -> signed pixel offset from K,
            e.g. {"A_B1": -460.9, "B_B1": 517.8, ...}.

    Returns:
        DataFrame with columns Ring, Block, X, Y, quality.
    """
    # Build reverse lookup: ring_idx -> group name
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

        # K segment
        rows.append({
            'Ring': ring_idx,
            'Block': 'K',
            'X': k_x,
            'Y': k_y % img_height,
            'quality': quality,
        })

        # Non-K blocks: look up group, then apply group offset
        group = ring_to_group.get(ring_idx, default_group)
        for block in EXPANSION_BLOCKS:
            key = f"{group}_{block}"
            offset = group_offsets.get(key, 0.0)
            y = (k_y + offset) % img_height
            if y < 0:
                y += img_height
            rows.append({
                'Ring': ring_idx,
                'Block': block,
                'X': k_x,
                'Y': round(y, 1),
                'quality': quality,
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
    """Generate visualization of detected lines, K positions, and all segment positions."""
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
    
    # Draw all segment positions if provided
    block_colors = {
        'K': (0, 255, 255),    # Yellow (already drawn above)
        'B1': (255, 165, 0),   # Orange
        'A1': (0, 200, 200),   # Teal
        'A2': (200, 0, 200),   # Purple
        'A3': (100, 255, 100), # Light green
        'A4': (100, 100, 255), # Light blue
        'B2': (255, 100, 100), # Light red
    }
    if all_segments is not None:
        for _, row in all_segments.iterrows():
            if row['Block'] == 'K':
                continue
            color = block_colors.get(row['Block'], (200, 200, 200))
            cv2.circle(output_image, (int(row['X']), int(row['Y'])), 5, color, -1)
    
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
    
    # Physical constants - READ FROM PREPROCESSING STAGE (not duplicated here)
    preprocessing_params = load_preprocessing_params(tunnel_id, base_dir)
    tunnel_diameter = preprocessing_params.get('tunnel_diameter', 5.5)
    resolution = preprocessing_params.get('depth_map_resolution', 0.005)
    ring_spacing = preprocessing_params.get('ring_spacing', 1.2)
    k_height_mm, ab_height_mm = calculate_segment_heights(tunnel_diameter)
    
    # Extract DBSCAN parameters for complex_staggered method (using unified eps)
    eps = params.get('eps', 0.07)
    complex_params = {
        'eps': eps,  # Unified eps parameter (replaces complex_eps_primary/secondary)
        'complex_subdivision_threshold': params.get('complex_subdivision_threshold', DEFAULT_COMPLEX_SUBDIVISION_THRESHOLD),
        'complex_max_subdivisions': params.get('complex_max_subdivisions', DEFAULT_COMPLEX_MAX_SUBDIVISIONS),
        'complex_conf_midpoint': params.get('complex_conf_midpoint', DEFAULT_COMPLEX_CONF_MIDPOINT),
        'complex_conf_intersection': params.get('complex_conf_intersection', DEFAULT_COMPLEX_CONF_INTERSECTION),
    }
    
    print("\nBO-tunable line detection parameters:")
    print(f"  binary_threshold:        {params.get('binary_threshold', 139)}")
    print(f"  hough_threshold:         {params.get('hough_threshold', 37)}")
    print(f"  hough_min_length:        {params.get('hough_min_length', 31)}")
    print(f"  hough_max_gap:           {params.get('hough_max_gap', 133)}")
    print(f"  angle_pos_min/max:       {params.get('angle_pos_min', 4.84)}°/{params.get('angle_pos_max', 13.55)}°")
    print(f"  angle_neg_min/max:       {params.get('angle_neg_min', -14.67)}°/{params.get('angle_neg_max', -5.82)}°")
    print(f"\nPhysical constants (from preprocessing stage):")
    print(f"  tunnel_diameter:         {tunnel_diameter}m")
    print(f"  ring_spacing:            {ring_spacing}m")
    print(f"  resolution:              {resolution}")
    print(f"  k_height_mm (calculated): {k_height_mm:.2f}")
    print(f"  ab_height_mm (calculated): {ab_height_mm:.2f}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load data from preprocessing stage
    depth_map_outlier_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_map_outlier_path):
        raise FileNotFoundError(f"depth_map_outlier.npy not found at {depth_map_outlier_path}. Run preprocessing first.")
    
    depth_map_outlier = np.load(depth_map_outlier_path)
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    L, W = depth_map_outlier.shape
    
    print(f"\n[Step 1] Detecting lines (unified)...")
    line_data = detect_lines(depth_map_outlier, params)
    print(f"  Positive slope lines: {len(line_data['positive_lines'])}")
    print(f"  Negative slope lines: {len(line_data['negative_lines'])}")
    print(f"  Horizontal lines: {len(line_data['horizontal_lines'])}")
    print(f"  Vertical lines: {len(line_data['vertical_lines'])}")
    
    k_detection_method = params.get('k_detection_method', 'complex_staggered')
    print(f"\n[Step 2] Calculating K positions ({k_detection_method})...")
    if k_detection_method == 'groove_pair':
        k_positions = calculate_k_positions_groove_pair(
            line_data, ring_count, k_height_mm, resolution, params
        )
    elif k_detection_method == 'combined':
        k_positions = calculate_k_positions_combined(
            line_data, ring_count, k_height_mm, resolution, params
        )
    else:  # complex_staggered (DBSCAN)
        k_positions = calculate_k_positions_complex_staggered(
            line_data, ring_count, k_height_mm, ab_height_mm, resolution, complex_params
        )
    print(f"  Calculated {len(k_positions)} K positions")
    print(f"  Detection types: {k_positions['Type'].value_counts().to_dict()}")
    if 'Confidence' in k_positions.columns:
        print(f"  Average confidence: {k_positions['Confidence'].mean():.3f}")
        print(f"  Confidence range: [{k_positions['Confidence'].min():.3f}, {k_positions['Confidence'].max():.3f}]")
    
    # Optional: reverse ring order so ring 0 = right (high X), ring N-1 = left (low X), matching GT convention
    reverse_ring_order = params.get('reverse_ring_order', False)
    if reverse_ring_order:
        k_positions = k_positions.iloc[::-1].reset_index(drop=True)
        print(f"  Reversed ring order (ring 0 = right / high X)")
    
    # Save K-only results (backward compatible)
    k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)
    print(f"\n  Saved: {os.path.join(tunnel_dir, 'detected.csv')}")

    # Save groove alignment metadata (for intrinsic BO objective)
    groove_meta = {
        'groove_alignment_total': k_positions.attrs.get('groove_alignment_total', None),
        'groove_alignment_max': k_positions.attrs.get('groove_alignment_max', None),
        'groove_alignment_pct': k_positions.attrs.get('groove_alignment_pct', None),
        'k_detection_method': k_detection_method,
    }
    groove_meta_path = os.path.join(tunnel_dir, 'groove_alignment.json')
    with open(groove_meta_path, 'w') as f:
        json.dump(groove_meta, f, indent=2)
    print(f"  Saved: {groove_meta_path}")

    # Step 3: Expand K positions to all segments (grouped_offsets only)
    expansion_method = params.get('expansion_method', 'grouped_offsets')
    print(f"\n[Step 3] Expanding K positions to all segments ({expansion_method})...")

    n_rings_detected = len(k_positions)
    
    if expansion_method != 'grouped_offsets':
        raise ValueError(f"Only 'grouped_offsets' expansion method is supported. Got: {expansion_method}")
    
    # Grouped offsets: 2 stagger groups x 6 blocks = 12D (BO-tunable)
    stagger_groups = params.get('stagger_groups', {"A": list(range(n_rings_detected))})
    group_offsets = params.get('group_offsets', {})
    all_segments = expand_k_with_grouped_offsets(
        k_positions,
        img_height=L,
        stagger_groups=stagger_groups,
        group_offsets=group_offsets,
    )
    
    # Output filename based on detection method
    output_filename = params.get('output_filename', 'all_segments.csv')
    all_segments.to_csv(os.path.join(tunnel_dir, output_filename), index=False)
    print(f"  Expanded {len(k_positions)} K positions → {len(all_segments)} total segments")
    print(f"  Blocks per ring: {all_segments.groupby('Ring')['Block'].count().values.tolist()}")
    print(f"  Saved: {os.path.join(tunnel_dir, output_filename)}")
    
    # Generate visualization (with all segment positions)
    visualize_detection(line_data, k_positions, tunnel_dir, all_segments=all_segments)
    print(f"  Saved: {os.path.join(tunnel_dir, 'detected_lines.png')}")
    
    print(f"\n{'=' * 60}")
    print(f"Detection complete!")
    print(f"{'=' * 60}")
    
    print("\nK Position Summary:")
    print(k_positions.to_string(index=False))
    
    print(f"\nAll Segments Summary ({len(all_segments)} total):")
    print(all_segments.to_string(index=False))
    
    return k_positions, all_segments


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

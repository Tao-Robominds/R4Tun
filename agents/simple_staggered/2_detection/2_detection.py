"""
Simplified Detection Pipeline: Line Detection and K-Position Calculation

This module detects oblique lines, finds intersections, and calculates K-block
positions with **only critical parameters** exposed for Bayesian Optimization.

Based on P4TUN optimization reports:
- Detection provided +6.3% mIoU improvement - the LARGEST single-stage gain
- binary_threshold, hough_oblique_threshold, angle parameters are HIGH sensitivity
- Post-detection tweaks cannot compensate for poor detection

Critical Parameters (14 total):
- binary_threshold: Edge detection sensitivity (HIGH)
- hough_oblique_threshold: Line detection sensitivity (HIGH)
- angle_positive_min/max: Positive slope angle range (HIGH)
- angle_negative_min/max: Negative slope angle range (HIGH)
- hough_vertical_threshold: Ring boundary detection (MEDIUM-HIGH)
- hough_horizontal_threshold/min_length/max_gap: Horizontal line detection (MEDIUM)
- horizontal_angle_tolerance: Horizontal line filtering (MEDIUM)
- merge_distance_threshold: Vertical line merging (MEDIUM)
- dilation_kernel_size, dilation_iterations: Morphological operations (MEDIUM)
- hough_oblique_min_length, hough_oblique_max_gap: Line filtering (MEDIUM)

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


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Tuple[Dict, bool]:
    """
    Load parameters from parameters_detection.json.
    
    Priority:
        1. agents/simple_staggered/2_detection/parameters/<tunnel_id>/parameters_detection.json
        2. data/<tunnel_id>/parameters_detection.json
        3. agents/simple_staggered/2_detection/parameters/sample/parameters_detection.json
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
# K-Position Calculation
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


def calculate_k_positions(
    line_data: Dict,
    ring_centers: List[float],
    k_height_mm: float,
    ab_height_mm: float,
    resolution: float
) -> pd.DataFrame:
    """
    Calculate K positions using midpoint logic.
    
    CRITICAL PARAMETERS:
    - k_height_mm: K-block height for offset calculation
    - ab_height_mm: AB-block height for alternation pattern
    """
    K_HEIGHT_PX = mm_to_px(k_height_mm, resolution)
    AB_HEIGHT_PX = mm_to_px(ab_height_mm, resolution)
    L = line_data['image_height']

    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']

    adjusted_points = []

    for vertical_x in ring_centers:
        # Find intersections with positive slope lines
        pos_intersections = []
        for x1, y1, x2, y2 in positive_lines:
            y_int = line_segment_vertical_intersection(vertical_x, (x1, y1, x2, y2))
            if y_int is not None:
                pos_intersections.append(y_int)

        # Find intersections with negative slope lines
        neg_intersections = []
        for x1, y1, x2, y2 in negative_lines:
            y_int = line_segment_vertical_intersection(vertical_x, (x1, y1, x2, y2))
            if y_int is not None:
                neg_intersections.append(y_int)

        merge_positive = merge_close_points(pos_intersections)
        merge_negative = merge_close_points(neg_intersections)
        
        # Case 1: Both positive and negative slope intersections → midpoint
        if len(merge_positive) > 0 and len(merge_negative) > 0:
            midpoint_y = (merge_positive[0] + merge_negative[0]) / 2
            adjusted_points.append(('midpoint', vertical_x, midpoint_y))
        
        # Case 2: Only positive slope → adjust by -0.5*K_height
        elif len(merge_positive) > 0:
            y = merge_positive[0] - 0.5 * K_HEIGHT_PX
            adjusted_points.append(('positive_slope', vertical_x, y))
        
        # Case 3: Only negative slope → adjust by +0.5*K_height
        elif len(merge_negative) > 0:
            y = merge_negative[0] + 0.5 * K_HEIGHT_PX
            adjusted_points.append(('negative_slope', vertical_x, y))
        
        # Case 4: No line intersections — use alternation pattern
        else:
            if adjusted_points:
                last_y = adjusted_points[-1][2]
                alternation_offset = (2.0 / 3.0) * AB_HEIGHT_PX

                low_center, low_hw = 0.25 * L, 0.10 * L
                high_center, high_hw = 0.65 * L, 0.10 * L
                low_lo, low_hi = low_center - low_hw, low_center + low_hw
                high_lo, high_hi = high_center - high_hw, high_center + high_hw

                if low_lo <= last_y <= low_hi:
                    assumed_y = last_y + alternation_offset
                elif high_lo <= last_y <= high_hi:
                    assumed_y = last_y - alternation_offset
                else:
                    if len(adjusted_points) > 1:
                        second_last_y = adjusted_points[-2][2]
                        if low_lo <= second_last_y <= low_hi:
                            assumed_y = second_last_y
                        elif high_lo <= second_last_y <= high_hi:
                            assumed_y = second_last_y
                        else:
                            assumed_y = L / 2
                    else:
                        assumed_y = L / 2

                assumed_y = max(0.0, min(L, assumed_y))
                adjusted_points.append(('assume', vertical_x, assumed_y))
            else:
                adjusted_points.append(('default', vertical_x, L / 2))
    
    df = pd.DataFrame(adjusted_points, columns=['Type', 'X', 'Y'])
    df = df.sort_values(by='X').reset_index(drop=True)
    
    return df


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(
    line_data: Dict,
    ring_centers: List[float],
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
    
    # Draw ring centers (magenta vertical lines)
    for x in ring_centers:
        cv2.line(output_image, (int(x), 0), (int(x), L), color_vertical, 1)
    
    # Draw K positions (yellow circles)
    for _, row in k_positions.iterrows():
        cv2.circle(output_image, (int(row['X']), int(row['Y'])), 8, (0, 255, 255), -1)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(output_image)
    plt.title('Detection Results')
    plt.savefig(os.path.join(tunnel_dir, 'detected_lines.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Detection Pipeline
# =============================================================================

def run_detection(tunnel_id: str, base_dir: str = "data") -> pd.DataFrame:
    """
    Execute the complete detection pipeline.
    
    CRITICAL PARAMETERS (14 total):
    - binary_threshold: Edge detection sensitivity (HIGH)
    - hough_oblique_threshold: Line detection sensitivity (HIGH)
    - angle_positive_min/max: Positive slope angle range (HIGH)
    - angle_negative_min/max: Negative slope angle range (HIGH)
    - hough_vertical_threshold: Ring boundary detection (MEDIUM-HIGH)
    - hough_horizontal_*: Horizontal line detection (MEDIUM)
    - horizontal_angle_tolerance: Horizontal line filtering (MEDIUM)
    - merge_distance_threshold: Vertical line merging (MEDIUM)
    - dilation_*: Morphological operations (MEDIUM)
    - hough_oblique_min_length/max_gap: Line filtering (MEDIUM)
    
    Args:
        tunnel_id: Identifier for the tunnel (e.g., "1-4", "2-2")
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
    
    print(f"\n[Step 2] Computing ring centers...")
    ring_centers = compute_ring_centers(line_data, ring_count, ring_spacing, resolution)
    print(f"  Found {len(ring_centers)} ring centers")
    
    print(f"\n[Step 3] Calculating K positions...")
    k_positions = calculate_k_positions(
        line_data, ring_centers, k_height_mm, ab_height_mm, resolution
    )
    print(f"  Calculated {len(k_positions)} K positions")
    print(f"  Detection types: {k_positions['Type'].value_counts().to_dict()}")
    
    # Save results
    k_positions.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)
    print(f"\n  Saved: {os.path.join(tunnel_dir, 'detected.csv')}")
    
    # Generate visualization
    visualize_detection(line_data, ring_centers, k_positions, tunnel_dir)
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
    parser = argparse.ArgumentParser(description="Simplified detection pipeline")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    run_detection(args.tunnel_id, base_dir=args.data_dir)

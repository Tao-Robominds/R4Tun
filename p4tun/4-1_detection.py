"""
Algorithm 4-1 - Combined Line Detection and Pattern Recognition

This module combines:
1. Hough-based line detection (oblique, horizontal, vertical)
2. Pattern type detection (6seg_alternating, 6seg_constant, 7seg_alternating)
3. K-position calculation with pattern-aware normalization

Outputs:
- detected.csv: Raw line detection results
- pattern.csv: Normalized K positions for SAM
- pattern.json: Pattern metadata

Note: Data should be trimmed to exactly 360° coverage before detection.
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
# Physical Constants (defaults, can be overridden by parameters JSON)
# =============================================================================

DEFAULT_K_HEIGHT_MM = 1079.92
DEFAULT_AB_HEIGHT_MM = 3239.77
DEFAULT_SEGMENT_WIDTH_MM = 1200
DEFAULT_RESOLUTION = 0.005


def mm_to_px(mm: float, resolution: float = DEFAULT_RESOLUTION) -> float:
    """Convert millimeters to pixels."""
    return mm / (resolution * 1000)


def px_to_mm(px: float, resolution: float = DEFAULT_RESOLUTION) -> float:
    """Convert pixels to millimeters."""
    return px * resolution * 1000


# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str, base_dir: str = "data") -> dict:
    """Load parameters from JSON file."""
    script_dir = os.path.dirname(__file__)
    
    params_path = os.path.join(script_dir, "parameters", tunnel_id, "parameters_detection.json")
    if os.path.exists(params_path):
        print(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as f:
            return json.load(f)
    
    # Try sample parameters
    sample_path = os.path.join(script_dir, "parameters", "sample", "parameters_detection.json")
    if os.path.exists(sample_path):
        print(f"Loading sample parameters from {sample_path}")
        with open(sample_path, 'r') as f:
            return json.load(f)
    
    print("Using hardcoded default parameters")
    return {}


def get_param(params: dict, *keys, default=None):
    """Get nested parameter value."""
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# =============================================================================
# Segment Count Detection (Geometry-based)
# =============================================================================

def detect_segment_count_from_geometry(tunnel_dir: str, resolution: float = DEFAULT_RESOLUTION,
                                        k_height_mm: float = DEFAULT_K_HEIGHT_MM,
                                        ab_height_mm: float = DEFAULT_AB_HEIGHT_MM) -> int:
    """
    Detect segment count from tunnel geometry (radius → circumference).
    
    Uses the relationship: circumference = 2π × radius
    Compares calculated circumference to expected values:
    - 6-segment: K + 5×AB = 17278.77 mm
    - 7-segment: K + 6×AB = 20518.54 mm
    
    This is a general geometric solution - no ground truth labels used.
    """
    enhanced_path = os.path.join(tunnel_dir, 'enhanced.csv')
    
    if os.path.exists(enhanced_path):
        df = pd.read_csv(enhanced_path)
        if 'r' in df.columns:
            avg_radius = df['r'].mean()
            circumference_mm = 2 * np.pi * avg_radius * 1000
            
            circ_6 = k_height_mm + 5 * ab_height_mm  # 17278.77mm
            circ_7 = k_height_mm + 6 * ab_height_mm  # 20518.54mm
            
            dist_6 = abs(circumference_mm - circ_6)
            dist_7 = abs(circumference_mm - circ_7)
            segment_count = 6 if dist_6 < dist_7 else 7
            
            print(f"Tunnel radius: {avg_radius:.3f}m → circumference: {circumference_mm:.1f}mm")
            print(f"Detected: {segment_count} segments (closest to {'6-seg' if segment_count == 6 else '7-seg'} expected)")
            return segment_count
    
    return None


def detect_segment_count_from_height(image_height: int, resolution: float = DEFAULT_RESOLUTION,
                                     k_height_mm: float = DEFAULT_K_HEIGHT_MM,
                                     ab_height_mm: float = DEFAULT_AB_HEIGHT_MM) -> int:
    """Fallback: Auto-detect 6 or 7 segments from image height."""
    height_mm = image_height * resolution * 1000
    circumference_6 = k_height_mm + 5 * ab_height_mm
    circumference_7 = k_height_mm + 6 * ab_height_mm
    
    dist_6 = abs(height_mm - circumference_6)
    dist_7 = abs(height_mm - circumference_7)
    
    detected = 6 if dist_6 < dist_7 else 7
    expected_mm = circumference_6 if detected == 6 else circumference_7
    error_pct = abs(height_mm - expected_mm) / expected_mm * 100
    
    print(f"Image height: {image_height} px = {height_mm:.1f} mm")
    print(f"Fallback detected: {detected} segments (error: {error_pct:.1f}%)")
    
    return detected


# =============================================================================
# Line Detection
# =============================================================================

def detect_lines(depth_map_outlier: np.ndarray, params: dict, resolution: float = DEFAULT_RESOLUTION) -> Dict:
    """
    Detect oblique, horizontal, and vertical lines from depth map.
    
    Returns:
        Dictionary with detected lines and metadata
    """
    L, W = depth_map_outlier.shape
    
    # Physical constants from parameters
    segment_width_mm = get_param(params, 'physical_constants', 'segment_width_mm', default=SEGMENT_WIDTH_MM)
    
    # Preprocessing parameters
    binary_threshold = get_param(params, 'preprocessing', 'binary_threshold', default=127)
    dilation_kernel_size = get_param(params, 'preprocessing', 'dilation_kernel_size', default=3)
    dilation_iterations = get_param(params, 'preprocessing', 'dilation_iterations', default=1)
    
    # Hough oblique parameters
    hough_oblique_rho = get_param(params, 'hough_oblique', 'rho', default=1)
    hough_oblique_theta = np.pi / 180 * get_param(params, 'hough_oblique', 'theta_deg', default=1.0)
    hough_oblique_threshold = get_param(params, 'hough_oblique', 'threshold', default=50)
    hough_oblique_min_length = get_param(params, 'hough_oblique', 'min_length', default=100)
    hough_oblique_max_gap = get_param(params, 'hough_oblique', 'max_gap', default=40)
    angle_pos_min = get_param(params, 'hough_oblique', 'angle_positive_min', default=6)
    angle_pos_max = get_param(params, 'hough_oblique', 'angle_positive_max', default=9)
    angle_neg_min = get_param(params, 'hough_oblique', 'angle_negative_min', default=-9)
    angle_neg_max = get_param(params, 'hough_oblique', 'angle_negative_max', default=-6)
    
    # Hough horizontal parameters
    hough_horiz_threshold = get_param(params, 'hough_horizontal', 'threshold', default=50)
    hough_horiz_min_length = get_param(params, 'hough_horizontal', 'min_length', default=100)
    hough_horiz_max_gap = get_param(params, 'hough_horizontal', 'max_gap', default=10)
    horiz_angle_tolerance = get_param(params, 'hough_horizontal', 'angle_tolerance', default=1)
    
    # Hough vertical parameters
    hough_vert_threshold = get_param(params, 'hough_vertical', 'threshold', default=500)
    hough_vert_angle_tolerance = get_param(params, 'hough_vertical', 'angle_tolerance', default=0.5)
    vert_filter_rings = get_param(params, 'hough_vertical', 'filter_rings', default=5)
    
    # Line processing parameters
    merge_distance_threshold = get_param(params, 'line_processing', 'merge_distance_threshold', default=3)
    
    # Pre-processing
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, binary_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(binary_image, kernel, iterations=dilation_iterations)
    
    # Detect oblique lines
    lines_oblique = cv2.HoughLinesP(dilated_edges, hough_oblique_rho, hough_oblique_theta, 
                                     hough_oblique_threshold, 
                                     minLineLength=hough_oblique_min_length, 
                                     maxLineGap=hough_oblique_max_gap)
    
    # Detect horizontal lines
    lines_horizontal = cv2.HoughLinesP(dilated_edges, hough_oblique_rho, hough_oblique_theta, 
                                        hough_horiz_threshold, 
                                        minLineLength=hough_horiz_min_length, 
                                        maxLineGap=hough_horiz_max_gap)
    
    # Detect vertical lines
    lines_vertical = cv2.HoughLines(dilated_edges, hough_oblique_rho, hough_oblique_theta, hough_vert_threshold)
    if lines_vertical is not None:
        lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= (vert_filter_rings * segment_width_mm / (resolution*1000))]
    
    # Separate positive and negative slope lines
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
            if -horiz_angle_tolerance <= angle <= horiz_angle_tolerance:
                horizontal_lines.append(line[0])
    
    # Process vertical lines - merge close ones
    merged_vertical = []
    if lines_vertical is not None:
        lines_vert_2d = lines_vertical[:, 0]
        for rho, theta in lines_vert_2d:
            if abs(theta) <= hough_vert_angle_tolerance * np.pi / 180:
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


def compute_ring_boundaries(line_data: Dict, ring_count: int, params: dict) -> List[float]:
    """
    Compute ring center X positions from vertical lines.
    """
    L, W = line_data['image_height'], line_data['image_width']
    vertical_lines = line_data['vertical_lines']
    
    # Calculate midpoints between adjacent vertical lines
    mid_lines = []
    for i in range(len(vertical_lines) - 1):
        rho1, _ = vertical_lines[i]
        rho2, _ = vertical_lines[i + 1]
        mid_lines.append((rho1 + rho2) / 2)
    
    if len(mid_lines) == 0:
        # Fallback: evenly spaced
        block_width = W / ring_count
        return [(i + 0.5) * block_width for i in range(ring_count)]
    
    # Calculate average distance
    distances = [mid_lines[i+1] - mid_lines[i] for i in range(len(mid_lines)-1)]
    avg_distance = np.mean(distances) if distances else W / ring_count
    
    # Extend to cover all rings
    all_ring_centers = list(mid_lines)
    
    # Extend left
    leftmost = mid_lines[0]
    x = leftmost - avg_distance
    while x >= 0:
        all_ring_centers.insert(0, x)
        x -= avg_distance
    
    # Extend right
    rightmost = mid_lines[-1]
    x = rightmost + avg_distance
    while x <= W:
        all_ring_centers.append(x)
        x += avg_distance
    
    # Filter to valid range and deduplicate
    all_ring_centers = sorted(set([x for x in all_ring_centers if 0 <= x <= W]))
    
    return all_ring_centers


# =============================================================================
# V-Pair Detection
# =============================================================================

def detect_v_pairs(line_data: Dict, ring_boundaries: List[float], params: dict, 
                   resolution: float = DEFAULT_RESOLUTION,
                   k_height_mm: float = DEFAULT_K_HEIGHT_MM) -> Dict:
    """
    Detect V-pairs (oblique line intersections) at each ring center.
    V-pairs indicate K-block boundaries.
    """
    K_HEIGHT_PX = mm_to_px(k_height_mm, resolution)
    v_pair_spacing_tolerance = get_param(params, 'pattern_detection', 'v_pair_spacing_tolerance_px', default=60)
    
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    
    def line_segment_vertical_intersection(vertical_x, x1, y1, x2, y2):
        if x1 == x2:
            return None
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            t = (vertical_x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
        return None
    
    v_pairs = {}
    for ring_idx, ring_x in enumerate(ring_boundaries):
        pos_intersections = []
        neg_intersections = []
        
        for x1, y1, x2, y2 in positive_lines:
            y_int = line_segment_vertical_intersection(ring_x, x1, y1, x2, y2)
            if y_int is not None:
                pos_intersections.append(y_int)
        
        for x1, y1, x2, y2 in negative_lines:
            y_int = line_segment_vertical_intersection(ring_x, x1, y1, x2, y2)
            if y_int is not None:
                neg_intersections.append(y_int)
        
        # Find valid V-pairs (positive/negative pairs with ~K_HEIGHT spacing)
        midpoints = []
        qualities = []
        
        for pos_y in pos_intersections:
            for neg_y in neg_intersections:
                spacing = abs(pos_y - neg_y)
                if abs(spacing - K_HEIGHT_PX) < v_pair_spacing_tolerance:
                    midpoint = (pos_y + neg_y) / 2
                    quality = 1.0 - abs(spacing - K_HEIGHT_PX) / v_pair_spacing_tolerance
                    midpoints.append(midpoint)
                    qualities.append(quality)
        
        if midpoints:
            best_idx = np.argmax(qualities)
            v_pairs[ring_idx] = {
                'midpoint': midpoints[best_idx],
                'quality': qualities[best_idx],
                'ring_x': ring_x,
                'pos_count': len(pos_intersections),
                'neg_count': len(neg_intersections)
            }
    
    return v_pairs


# =============================================================================
# Pattern Type Detection
# =============================================================================

def detect_pattern_type(v_pairs: Dict, segment_count: int, params: dict,
                        image_height: int, resolution: float = DEFAULT_RESOLUTION,
                        ab_height_mm: float = DEFAULT_AB_HEIGHT_MM) -> Dict:
    """
    Detect the K-block pattern type based on V-pair characteristics.
    
    Pattern types:
    - 6seg_alternating: K position alternates between two positions ~432px apart
    - 6seg_constant: K position is constant across all rings
    - 7seg_alternating: 7-segment tunnel with alternating K positions
    """
    # Load pattern detection parameters
    alternation_tolerance = get_param(params, 'pattern_detection', 'alternation_tolerance_px', default=100)
    constant_spread_threshold = get_param(params, 'pattern_detection', 'constant_spread_threshold_px', default=100)
    cluster_separation_7seg = get_param(params, 'pattern_detection', 'cluster_separation_threshold_7seg_px', default=300)
    confidence_scaling = get_param(params, 'pattern_detection', 'confidence_scaling_factor', default=200)
    
    AB_HEIGHT_PX = mm_to_px(ab_height_mm, resolution)
    expected_alternation = 2/3 * AB_HEIGHT_PX  # ~432px
    
    metrics = {
        'num_v_pairs': len(v_pairs),
        'spread_px': 0,
        'cluster_separation_px': 0,
        'ring_changes': []
    }
    
    # Handle sparse V-pair cases
    if len(v_pairs) < 2:
        if segment_count == 7:
            return {
                'pattern_type': '7seg_alternating',
                'confidence': 0.6,
                'metrics': metrics
            }
        else:
            return {
                'pattern_type': '6seg_alternating',
                'confidence': 0.7,
                'metrics': metrics
            }
    
    # Analyze midpoint distribution
    midpoints = [v_pairs[r]['midpoint'] for r in sorted(v_pairs.keys())]
    spread = max(midpoints) - min(midpoints)
    metrics['spread_px'] = spread
    
    # Ring-to-ring changes
    sorted_rings = sorted(v_pairs.keys())
    ring_changes = []
    for i in range(len(sorted_rings) - 1):
        r1, r2 = sorted_rings[i], sorted_rings[i + 1]
        change = v_pairs[r2]['midpoint'] - v_pairs[r1]['midpoint']
        ring_changes.append(change)
    metrics['ring_changes'] = ring_changes
    
    # Clustering analysis
    midpoints_array = np.array(midpoints)
    cluster_separation = 0
    if len(midpoints) >= 2:
        median = np.median(midpoints_array)
        lower = midpoints_array[midpoints_array <= median]
        higher = midpoints_array[midpoints_array > median]
        if len(lower) > 0 and len(higher) > 0:
            cluster_separation = np.mean(higher) - np.mean(lower)
    metrics['cluster_separation_px'] = cluster_separation
    
    # Pattern classification
    if segment_count == 7:
        if cluster_separation > cluster_separation_7seg:
            confidence = 0.8 + 0.2 * min(cluster_separation / 500, 1.0)
        else:
            confidence = 0.7
        return {
            'pattern_type': '7seg_alternating',
            'confidence': min(confidence, 1.0),
            'metrics': metrics
        }
    else:
        if abs(cluster_separation - expected_alternation) < alternation_tolerance:
            pattern_type = '6seg_alternating'
            confidence = 1.0 - abs(cluster_separation - expected_alternation) / confidence_scaling
        elif spread < constant_spread_threshold:
            pattern_type = '6seg_constant'
            confidence = 1.0 - spread / confidence_scaling
        else:
            pattern_type = '6seg_alternating'
            confidence = 0.7
        
        return {
            'pattern_type': pattern_type,
            'confidence': min(confidence, 1.0),
            'metrics': metrics
        }


# =============================================================================
# K Position Calculation
# =============================================================================

def calculate_k_positions(v_pairs: Dict, ring_boundaries: List[float], pattern_info: Dict,
                          image_height: int, image_width: int, ring_count: int,
                          resolution: float = DEFAULT_RESOLUTION,
                          k_height_mm: float = DEFAULT_K_HEIGHT_MM,
                          ab_height_mm: float = DEFAULT_AB_HEIGHT_MM) -> pd.DataFrame:
    """
    Calculate normalized K positions based on detected pattern type.
    
    For alternating patterns, uses robust position estimation:
    - Filters outlier V-pairs that don't fit the expected pattern
    - Falls back to geometric priors if V-pairs are inconsistent
    """
    pattern_type = pattern_info['pattern_type']
    
    K_HEIGHT_PX = mm_to_px(k_height_mm, resolution)
    AB_HEIGHT_PX = mm_to_px(ab_height_mm, resolution)
    ALTERNATION_OFFSET = 2/3 * AB_HEIGHT_PX  # ~432px
    CORRECTION_OFFSET = AB_HEIGHT_PX / 2
    
    # Expected K position range (center ~40% of image height ± some tolerance)
    expected_center = image_height * 0.4
    position_tolerance = ALTERNATION_OFFSET * 2  # Allow positions within 2x alternation distance
    
    results = []
    
    if 'constant' in pattern_type:
        # Constant pattern: all K positions at same Y
        if len(v_pairs) >= 1:
            midpoints = [v_pairs[r]['midpoint'] for r in v_pairs]
            # Filter outliers
            midpoints = [m for m in midpoints if abs(m + CORRECTION_OFFSET - expected_center) < position_tolerance]
            if midpoints:
                k_position = np.mean(midpoints) + CORRECTION_OFFSET
            else:
                k_position = expected_center
        else:
            k_position = expected_center
        
        for ring_idx in range(ring_count):
            ring_x = ring_boundaries[ring_idx] if ring_idx < len(ring_boundaries) else (ring_idx + 0.5) * (image_width / ring_count)
            quality = 0.7 if ring_idx in v_pairs else 0.5
            
            results.append({
                'ring': ring_idx,
                'X': ring_x,
                'Y': k_position,
                'quality': quality,
                'detection_type': 'constant',
                'position_class': 'constant'
            })
    
    else:  # Alternating pattern (6seg or 7seg)
        # Apply correction and filter outliers
        corrected_positions = {}
        valid_corrected = []
        
        for ring_idx, data in v_pairs.items():
            corrected = data['midpoint'] + CORRECTION_OFFSET
            # Filter: only keep positions within expected range
            if abs(corrected - expected_center) < position_tolerance:
                corrected_positions[ring_idx] = {
                    'corrected': corrected,
                    'quality': data['quality'],
                    'ring_x': data['ring_x']
                }
                valid_corrected.append(corrected)
            else:
                print(f"  Filtering outlier V-pair at ring {ring_idx}: Y={corrected:.1f} (expected ~{expected_center:.1f})")
        
        # Find two alternating positions
        if len(valid_corrected) >= 2:
            positions = valid_corrected
            median = np.median(positions)
            lower = [p for p in positions if p <= median]
            higher = [p for p in positions if p > median]
            
            pos1 = np.mean(lower) if lower else median - ALTERNATION_OFFSET / 2
            pos2 = np.mean(higher) if higher else median + ALTERNATION_OFFSET / 2
        else:
            # Use geometric priors
            pos1 = expected_center
            pos2 = expected_center + ALTERNATION_OFFSET
        
        # Assign positions to rings with strict alternation
        last_pos_idx = None
        for ring_idx in range(ring_count):
            ring_x = ring_boundaries[ring_idx] if ring_idx < len(ring_boundaries) else (ring_idx + 0.5) * (image_width / ring_count)
            
            if ring_idx in corrected_positions:
                k_y = corrected_positions[ring_idx]['corrected']
                quality = corrected_positions[ring_idx]['quality']
                detection_type = 'v_pair_corrected'
                
                # Determine which cluster this belongs to
                if abs(k_y - pos1) < abs(k_y - pos2):
                    last_pos_idx = 0
                else:
                    last_pos_idx = 1
            else:
                # Infer from alternation pattern
                if last_pos_idx is not None:
                    inferred_pos_idx = 1 - last_pos_idx
                else:
                    # Start with position closest to expected center
                    inferred_pos_idx = 0 if abs(pos1 - expected_center) < abs(pos2 - expected_center) else 1
                
                k_y = pos1 if inferred_pos_idx == 0 else pos2
                quality = 0.5
                detection_type = 'alternation_inferred'
                last_pos_idx = inferred_pos_idx
            
            results.append({
                'ring': ring_idx,
                'X': ring_x,
                'Y': k_y,
                'quality': quality,
                'detection_type': detection_type,
                'position_class': 'lower' if abs(k_y - pos1) < abs(k_y - pos2) else 'higher'
            })
    
    df = pd.DataFrame(results)
    df['Type'] = df['detection_type'].apply(lambda x: 'midpoint' if 'v_pair' in x else 'inferred')
    return df


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(line_data: Dict, ring_boundaries: List[float], 
                        k_positions: pd.DataFrame, tunnel_dir: str):
    """Generate visualization of detected lines and K positions."""
    dilated_edges = line_data['dilated_edges']
    L, W = line_data['image_height'], line_data['image_width']
    
    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    
    # Draw positive slope lines (red)
    for x1, y1, x2, y2 in line_data['positive_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    
    # Draw negative slope lines (green)
    for x1, y1, x2, y2 in line_data['negative_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Draw horizontal lines (blue)
    for x1, y1, x2, y2 in line_data['horizontal_lines']:
        cv2.line(output_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    
    # Draw ring boundaries (magenta)
    for x in ring_boundaries:
        cv2.line(output_image, (int(x), 0), (int(x), L), (255, 0, 255), 1)
    
    # Draw K positions (yellow circles)
    for _, row in k_positions.iterrows():
        cv2.circle(output_image, (int(row['X']), int(row['Y'])), 8, (0, 255, 255), -1)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(output_image)
    plt.title('Detection Results')
    plt.savefig(os.path.join(tunnel_dir, 'detection_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Detection Pipeline
# =============================================================================

def run_detection(tunnel_id: str, base_dir: str = "data") -> Tuple[pd.DataFrame, Dict]:
    """
    Run the complete detection pipeline:
    1. Line detection (oblique, horizontal, vertical)
    2. V-pair detection
    3. Pattern type detection
    4. K position calculation
    
    Returns:
        k_positions (DataFrame): Per-ring K positions for SAM
        pattern_info (Dict): Pattern metadata
    """
    # Load parameters
    params = load_parameters(tunnel_id, base_dir)
    
    # Physical constants (can be overridden per-tunnel)
    resolution = get_param(params, 'physical_constants', 'resolution', default=DEFAULT_RESOLUTION)
    k_height_mm = get_param(params, 'physical_constants', 'k_height_mm', default=DEFAULT_K_HEIGHT_MM)
    ab_height_mm = get_param(params, 'physical_constants', 'ab_height_mm', default=DEFAULT_AB_HEIGHT_MM)
    ring_spacing_m = get_param(params, 'physical_constants', 'ring_spacing_m', default=1.2)
    oblique_angle_deg = get_param(params, 'physical_constants', 'oblique_angle_deg', default=7.52)
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    print(f"{'=' * 60}")
    print(f"Detection Pipeline for Tunnel: {tunnel_id}")
    print(f"{'=' * 60}")
    
    # Load data
    depth_map_outlier = np.load(os.path.join(tunnel_dir, "depth_map_outlier.npy"))
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    L, W = depth_map_outlier.shape
    
    print(f"\n[Step 1] Detecting segment count from geometry...")
    segment_count = detect_segment_count_from_geometry(tunnel_dir, resolution, k_height_mm, ab_height_mm)
    if segment_count is None:
        segment_count = detect_segment_count_from_height(L, resolution, k_height_mm, ab_height_mm)
    
    print(f"\n[Step 2] Detecting lines...")
    line_data = detect_lines(depth_map_outlier, params, resolution)
    print(f"  Positive slope lines: {len(line_data['positive_lines'])}")
    print(f"  Negative slope lines: {len(line_data['negative_lines'])}")
    print(f"  Horizontal lines: {len(line_data['horizontal_lines'])}")
    print(f"  Vertical lines: {len(line_data['vertical_lines'])}")
    
    print(f"\n[Step 3] Computing ring boundaries...")
    ring_boundaries = compute_ring_boundaries(line_data, ring_count, params)
    print(f"  Found {len(ring_boundaries)} ring centers")
    
    print(f"\n[Step 4] Detecting V-pairs...")
    v_pairs = detect_v_pairs(line_data, ring_boundaries, params, resolution, k_height_mm)
    print(f"  Detected {len(v_pairs)} V-pairs out of {ring_count} rings")
    
    print(f"\n[Step 5] Detecting pattern type...")
    pattern_info = detect_pattern_type(v_pairs, segment_count, params, L, resolution, ab_height_mm)
    pattern_info['segment_count'] = segment_count
    print(f"  Pattern type: {pattern_info['pattern_type']}")
    print(f"  Confidence: {pattern_info['confidence']:.2f}")
    
    print(f"\n[Step 6] Calculating K positions...")
    k_positions = calculate_k_positions(
        v_pairs, ring_boundaries, pattern_info,
        L, W, ring_count, resolution, k_height_mm, ab_height_mm
    )
    print(f"  Calculated {len(k_positions)} K positions")
    print(f"  Detection types: {k_positions['detection_type'].value_counts().to_dict()}")
    
    # Save raw detection results (detected.csv)
    raw_detections = []
    for ring_idx in range(len(ring_boundaries)):
        ring_x = ring_boundaries[ring_idx] if ring_idx < len(ring_boundaries) else (ring_idx + 0.5) * (W / ring_count)
        if ring_idx in v_pairs:
            raw_detections.append({
                'Type': 'v_pair',
                'X': ring_x,
                'Y': v_pairs[ring_idx]['midpoint']
            })
        else:
            # Use alternation inference for raw detection too
            k_row = k_positions[k_positions['ring'] == ring_idx]
            if len(k_row) > 0:
                raw_detections.append({
                    'Type': 'inferred',
                    'X': k_row.iloc[0]['X'],
                    'Y': k_row.iloc[0]['Y']
                })
    
    detected_df = pd.DataFrame(raw_detections)
    detected_df.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)
    print(f"\n  Saved: {os.path.join(tunnel_dir, 'detected.csv')}")
    
    # Save normalized K positions (pattern.csv)
    k_positions.to_csv(os.path.join(tunnel_dir, 'pattern.csv'), index=False)
    print(f"  Saved: {os.path.join(tunnel_dir, 'pattern.csv')}")
    
    # Save pattern metadata (pattern.json)
    pattern_metadata = {
        'tunnel_id': tunnel_id,
        'pattern_type': pattern_info['pattern_type'],
        'segment_count': segment_count,
        'confidence': pattern_info['confidence'],
        'metrics': pattern_info['metrics'],
        'v_pair_count': len(v_pairs),
        'ring_count': ring_count,
        'image_height': L,
        'image_width': W,
        'resolution': resolution,
        'physical_constants': {
            'K_HEIGHT_MM': k_height_mm,
            'AB_HEIGHT_MM': ab_height_mm,
            'K_HEIGHT_PX': mm_to_px(k_height_mm, resolution),
            'AB_HEIGHT_PX': mm_to_px(ab_height_mm, resolution)
        }
    }
    
    with open(os.path.join(tunnel_dir, 'pattern.json'), 'w') as f:
        json.dump(pattern_metadata, f, indent=2)
    print(f"  Saved: {os.path.join(tunnel_dir, 'pattern.json')}")
    
    # Generate visualization
    visualize_detection(line_data, ring_boundaries, k_positions, tunnel_dir)
    print(f"  Saved: {os.path.join(tunnel_dir, 'detection_visualization.png')}")
    
    print(f"\n{'=' * 60}")
    print(f"Detection complete!")
    print(f"{'=' * 60}")
    
    return k_positions, pattern_metadata


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combined line detection and pattern recognition")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    k_positions, pattern_info = run_detection(args.tunnel_id, base_dir=args.data_dir)
    
    print("\nK Position Summary:")
    print(k_positions[['ring', 'X', 'Y', 'quality', 'detection_type']].to_string(index=False))

"""
Algorithm 4-1 - Line Detection and K-Position Calculation

Parameterized version based on sam4tun logic.
Detects oblique lines, finds intersections, and calculates K-block midpoints.

Outputs:
- detected.csv: K positions for SAM
- detected_lines.png: Visualization
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
# Line Detection
# =============================================================================

def detect_lines(depth_map_outlier: np.ndarray, params: dict, resolution: float = DEFAULT_RESOLUTION) -> Dict:
    """
    Detect oblique, horizontal, and vertical lines from depth map.
    """
    L, W = depth_map_outlier.shape
    
    # Preprocessing parameters
    binary_threshold = get_param(params, 'preprocessing', 'binary_threshold', default=127)
    dilation_kernel_size = get_param(params, 'preprocessing', 'dilation_kernel_size', default=3)
    dilation_iterations = get_param(params, 'preprocessing', 'dilation_iterations', default=1)
    
    # Hough oblique parameters
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
    vert_filter_rings = get_param(params, 'hough_vertical', 'filter_rings', default=5)
    
    # Line processing parameters
    merge_distance_threshold = get_param(params, 'line_processing', 'merge_distance_threshold', default=3)
    
    # Pre-processing - improved edge detection
    # Method 1: Binary on NaN/non-NaN (original)
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, binary_threshold, 255, cv2.THRESH_BINARY)
    
    # Method 2: Use actual depth values for better edge detection
    depth_valid = depth_map_outlier[~np.isnan(depth_map_outlier)]
    if len(depth_valid) > 0:
        depth_min, depth_max = depth_valid.min(), depth_valid.max()
        if depth_max > depth_min:
            depth_normalized = ((depth_map_outlier - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            depth_normalized[np.isnan(depth_map_outlier)] = 0
            
            # Use Canny edge detection for better line detection
            canny_edges = cv2.Canny(depth_normalized, 50, 150)
            
            # Combine both methods
            combined_edges = cv2.bitwise_or(binary_image, canny_edges)
        else:
            combined_edges = binary_image
    else:
        combined_edges = binary_image
    
    # Dilation to connect broken line segments
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=dilation_iterations)
    
    # Detect oblique lines
    lines_oblique = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 
                                     hough_oblique_threshold, 
                                     minLineLength=hough_oblique_min_length, 
                                     maxLineGap=hough_oblique_max_gap)
    
    # Detect horizontal lines
    lines_horizontal = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 
                                        hough_horiz_threshold, 
                                        minLineLength=hough_horiz_min_length, 
                                        maxLineGap=hough_horiz_max_gap)
    
    # Detect vertical lines
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, hough_vert_threshold)
    if lines_vertical is not None:
        # Filter: For vertical lines (theta ≈ 0), rho represents X position
        # rho can range from 0 to W (image width)
        # Keep all lines within image bounds
        max_rho = W  # Maximum rho = image width
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

def compute_ring_centers(line_data: Dict, ring_count: int) -> List[float]:
    """
    Compute ring center X positions from vertical lines.
    Same logic as sam4tun.
    """
    L, W = line_data['image_height'], line_data['image_width']
    vertical_lines = line_data['vertical_lines']
    
    if not vertical_lines:
        # Fallback: evenly spaced
        print("No vertical lines detected. Using fallback method.")
        block_width = W / ring_count
        return [(i + 0.5) * block_width for i in range(ring_count)]
    
    # Calculate midpoints between adjacent vertical lines
    # Convert rho,theta to X positions first (matching sam4tun logic)
    mid_lines = []
    for i in range(len(vertical_lines) - 1):
        rho1, theta1 = vertical_lines[i]
        rho2, theta2 = vertical_lines[i + 1]
        # Calculate midpoint in rho,theta space
        new_rho = (rho1 + rho2) / 2
        new_theta = (theta1 + theta2) / 2
        # Convert to X position (matching sam4tun line 158)
        a = np.cos(new_theta)
        x_pos = a * new_rho
        mid_lines.append((x_pos, new_theta))
    
    if len(mid_lines) == 0:
        block_width = W / ring_count
        return [(i + 0.5) * block_width for i in range(ring_count)]
    
    # Calculate average distance using X positions
    x_positions = [x for x, _ in mid_lines]
    distances = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
    avg_distance_detected = np.mean(distances) if distances else 0
    avg_distance_designed = W / ring_count
    
    # Choose better estimate
    if abs(avg_distance_detected - (1.2 / 0.005)) <= abs(avg_distance_designed - (1.2 / 0.005)):
        avg_distance = avg_distance_detected
    else:
        avg_distance = avg_distance_designed
    
    # Extend to cover all rings (matching sam4tun logic)
    all_mid_lines = list(mid_lines)
    
    # Extend left (matching sam4tun lines 191-205)
    if mid_lines:
        leftmost_x, leftmost_theta = mid_lines[0]
        x = leftmost_x - avg_distance
        while x >= 0:
            all_mid_lines.insert(0, (x, leftmost_theta))
            x -= avg_distance
        
        # Extend right (matching sam4tun lines 208-222)
        rightmost_x, rightmost_theta = mid_lines[-1]
        x = rightmost_x + avg_distance
        while x <= W:
            all_mid_lines.append((x, rightmost_theta))
            x += avg_distance
    
    # Sort by X position and extract X values (matching sam4tun line 224)
    all_mid_lines = sorted(list(set(all_mid_lines)), key=lambda line: line[0])
    x_positions = [x for x, _ in all_mid_lines]
    
    return x_positions


# =============================================================================
# K-Position Calculation (sam4tun logic)
# =============================================================================

def line_segment_vertical_intersection(vertical_x, segment):
    """Compute intersection of vertical line with line segment."""
    x1, y1, x2, y2 = segment
    if x1 == x2:
        return None
    if min(x1, x2) <= vertical_x <= max(x1, x2):
        t = (vertical_x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)
    return None


def merge_close_points(points, threshold=6):
    """Merge points that are within threshold distance."""
    if len(points) == 0:
        return []
    points = np.array(points)
    if len(points) == 1:
        return [points[0]]
    
    merged_points = []
    while len(points) > 0:
        p = points[0]
        close_mask = np.abs(points - p) < threshold
        merged_points.append(np.mean(points[close_mask]))
        points = points[~close_mask]
    return merged_points


def calculate_k_positions(line_data: Dict, ring_centers: List[float], 
                          k_height_mm: float, ab_height_mm: float,
                          resolution: float) -> pd.DataFrame:
    """
    Calculate K positions using sam4tun's midpoint logic.
    NO correction offset - just raw midpoint between oblique lines.
    """
    K_HEIGHT_PX = mm_to_px(k_height_mm, resolution)
    AB_HEIGHT_PX = mm_to_px(ab_height_mm, resolution)
    L = line_data['image_height']
    
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    horizontal_lines = line_data['horizontal_lines']
    
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
        
        # Case 4: Use alternation pattern based on previous point
        else:
            if adjusted_points:
                last_y = adjusted_points[-1][2]
                # Alternation offset = 2/3 * AB_height ≈ 431.87 pixels
                alternation_offset = 2/3 * AB_HEIGHT_PX
                
                if 1035 <= last_y <= 1265:  # ~1150 ± 10%
                    assumed_y = last_y + alternation_offset
                elif 1422 <= last_y <= 1738:  # ~1580 ± 10%
                    assumed_y = last_y - alternation_offset
                else:
                    # Check two points back
                    if len(adjusted_points) > 1:
                        second_last_y = adjusted_points[-2][2]
                        if 1035 <= second_last_y <= 1265:
                            assumed_y = second_last_y
                        elif 1422 <= second_last_y <= 1738:
                            assumed_y = second_last_y
                        else:
                            assumed_y = L / 2
                    else:
                        assumed_y = L / 2
                
                adjusted_points.append(('assume', vertical_x, assumed_y))
            else:
                adjusted_points.append(('default', vertical_x, L / 2))
    
    # Create DataFrame
    df = pd.DataFrame(adjusted_points, columns=['Type', 'X', 'Y'])
    df = df.sort_values(by='X').reset_index(drop=True)
    
    return df


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(line_data: Dict, ring_centers: List[float], 
                        k_positions: pd.DataFrame, tunnel_dir: str):
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
    Run the complete detection pipeline (sam4tun logic, parameterized).
    """
    # Load parameters
    params = load_parameters(tunnel_id, base_dir)
    
    # Physical constants (can be overridden per-tunnel)
    resolution = get_param(params, 'physical_constants', 'resolution', default=DEFAULT_RESOLUTION)
    k_height_mm = get_param(params, 'physical_constants', 'k_height_mm', default=DEFAULT_K_HEIGHT_MM)
    ab_height_mm = get_param(params, 'physical_constants', 'ab_height_mm', default=DEFAULT_AB_HEIGHT_MM)
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    print(f"{'=' * 60}")
    print(f"Detection Pipeline for Tunnel: {tunnel_id}")
    print(f"{'=' * 60}")
    
    # Load data
    depth_map_outlier = np.load(os.path.join(tunnel_dir, "depth_map_outlier.npy"))
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    L, W = depth_map_outlier.shape
    
    print(f"\n[Step 1] Detecting lines...")
    line_data = detect_lines(depth_map_outlier, params, resolution)
    print(f"  Positive slope lines: {len(line_data['positive_lines'])}")
    print(f"  Negative slope lines: {len(line_data['negative_lines'])}")
    print(f"  Horizontal lines: {len(line_data['horizontal_lines'])}")
    print(f"  Vertical lines: {len(line_data['vertical_lines'])}")
    
    print(f"\n[Step 2] Computing ring centers...")
    ring_centers = compute_ring_centers(line_data, ring_count)
    print(f"  Found {len(ring_centers)} ring centers")
    
    print(f"\n[Step 3] Calculating K positions...")
    k_positions = calculate_k_positions(line_data, ring_centers, k_height_mm, ab_height_mm, resolution)
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
    parser = argparse.ArgumentParser(description="Line detection and K-position calculation")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    k_positions = run_detection(args.tunnel_id, base_dir=args.data_dir)

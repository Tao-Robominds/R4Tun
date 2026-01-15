"""
Algorithm 4 - Prompt Point Generation
Faithful port of sam4tun/4-1_detection.py with:
1. External parameter loading
2. Auto-detection of 6 vs 7 segments
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str, base_dir: str = "data"):
    """Load parameters from JSON file."""
    script_dir = os.path.dirname(__file__)
    
    # Try centralized parameters first
    params_path = os.path.join(script_dir, "parameters", tunnel_id, "parameters_detection.json")
    if os.path.exists(params_path):
        print(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as f:
            return json.load(f)
    
    # Try default parameters
    default_path = os.path.join(script_dir, "parameters_detection.json")
    if os.path.exists(default_path):
        print(f"Loading default parameters from {default_path}")
        with open(default_path, 'r') as f:
            return json.load(f)
    
    print("Using hardcoded default parameters")
    return {}


def get_param(params, *keys, default=None):
    """Get nested parameter value."""
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# =============================================================================
# Segment Count Detection
# =============================================================================

def detect_segment_count(image_height: int, resolution: float = 0.005) -> int:
    """
    Auto-detect 6 or 7 segments from image height.
    
    Circumference for 6 segments: K + 5×AB = 17278.77 mm
    Circumference for 7 segments: K + 6×AB = 20518.54 mm
    """
    K_HEIGHT_MM = 1079.92
    AB_HEIGHT_MM = 3239.77
    
    height_mm = image_height * resolution * 1000
    circumference_6 = K_HEIGHT_MM + 5 * AB_HEIGHT_MM  # 17278.77
    circumference_7 = K_HEIGHT_MM + 6 * AB_HEIGHT_MM  # 20518.54
    
    dist_6 = abs(height_mm - circumference_6)
    dist_7 = abs(height_mm - circumference_7)
    
    detected = 6 if dist_6 < dist_7 else 7
    expected_mm = circumference_6 if detected == 6 else circumference_7
    error_pct = abs(height_mm - expected_mm) / expected_mm * 100
    
    print(f"Image height: {image_height} px = {height_mm:.1f} mm")
    print(f"Expected for {detected} segments: {expected_mm:.1f} mm")
    print(f"Auto-detected: {detected} segments (error: {error_pct:.1f}%)")
    
    return detected


# =============================================================================
# Main Detection (EXACT COPY of original sam4tun/4-1_detection.py logic)
# =============================================================================

def run_detection(tunnel_id: str, base_dir: str = "data", segment_count: int = None):
    """
    Run detection pipeline - FAITHFUL to original sam4tun/4-1_detection.py
    """
    # Load parameters
    params = load_parameters(tunnel_id, base_dir)
    resolution = get_param(params, 'physical_constants', 'resolution', default=0.005)
    
    # Load data
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_map_outlier = np.load(os.path.join(tunnel_dir, "depth_map_outlier.npy"))
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    
    print(f"Processing tunnel: {tunnel_id}")
    
    # Auto-detect segment count if not specified
    L, W = depth_map_outlier.shape
    if segment_count is None:
        segment_count = detect_segment_count(L, resolution)
    
    # ==========================================================================
    # EXACT COPY OF ORIGINAL CODE BELOW (sam4tun/4-1_detection.py)
    # ==========================================================================
    
    # Cell 4: pre-processing
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(binary_image, kernel, iterations=1)
    
    # Cell 5: detection
    # Oblique line segment detection parameters
    lines_oblique = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=40)
    
    # Horizontal line detection parameters (0 degrees)
    lines_horizontal = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=10)
    
    # Vertical line detection
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, 500)
    if lines_vertical is not None:
        lines_vertical = lines_vertical[lines_vertical[:, 0, 0] <= (5 * 1200 / (resolution*1000))]
    
    # Prepare output image
    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    
    # Define colors
    color_angle1 = (255, 0, 0)  # Red for positive angle lines
    color_angle2 = (0, 255, 0)  # Green for negative angle lines
    color_horizontal = (0, 0, 255)  # Blue for horizontal lines
    color_vertical = (255, 165, 0)  # Orange for vertical lines
    color_mid_lines = (255, 0, 255)  # Magenta for centered lines
    line_thickness = 3
    
    # Detect and draw oblique lines with angles between 6-9 degrees and -9 to -6 degrees
    joint_oblique_positive = []
    joint_oblique_negtive = []
    joint_horizontal = []
    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))  # Invert y-coordinates
            
            if 6 <= angle <= 9:
                joint_oblique_positive.append(line)
                cv2.line(output_image, (x1, y1), (x2, y2), color_angle1, line_thickness)
            elif -9 <= angle <= -6:
                joint_oblique_negtive.append(line)
                cv2.line(output_image, (x1, y1), (x2, y2), color_angle2, line_thickness)
    
    # Detect and draw horizontal lines
    if lines_horizontal is not None:
        for line in lines_horizontal:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -1 <= angle <= 1:
                joint_horizontal.append(line)
                cv2.line(output_image, (x1, y1), (x2, y2), color_horizontal, line_thickness)
    
    # Merge close vertical lines
    merged_lines = []
    all_mid_lines = []
    threshold_distance = 3
    
    if lines_vertical is not None:
        lines_vertical = lines_vertical[:, 0]  # Convert to 2D array
        
        for i, (rho1, theta1) in enumerate(lines_vertical):
            if -0.5 * np.pi / 180 <= abs(theta1) <= 0.5 * np.pi / 180:
                x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)
                is_merged = False
                
                for j, (rho2, theta2) in enumerate(merged_lines):
                    x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)
                    if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) < threshold_distance:
                        new_rho = (rho1 + rho2) / 2
                        new_theta = (theta1 + theta2) / 2
                        merged_lines[j] = (new_rho, new_theta)
                        is_merged = True
                        break
                
                if not is_merged:
                    merged_lines.append((rho1, theta1))
        
        merged_lines.sort(key=lambda line: line[0])
        
        # Draw merged vertical lines
        for rho, theta in merged_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2677 * (-b))
            y1 = int(y0 + 2677 * (a))
            x2 = int(x0 - 2677 * (-b))
            y2 = int(y0 - 2677 * (a))
            cv2.line(output_image, (x1, y1), (x2, y2), color_vertical, line_thickness)
        
        # Calculate centered lines between adjacent vertical lines
        mid_lines = []
        num_lines = len(merged_lines)
        for i in range(num_lines - 1):
            rho1, theta1 = merged_lines[i]
            rho2, theta2 = merged_lines[i + 1]
            new_rho = (rho1 + rho2) / 2
            new_theta = (theta1 + theta2) / 2
            mid_lines.append((new_rho, new_theta))
            
            a = np.cos(new_theta)
            b = np.sin(new_theta)
            x0 = a * new_rho
            y0 = b * new_rho
            x1 = int(x0 + L * (-b))
            y1 = int(y0 + L * (a))
            x2 = int(x0 - L * (-b))
            y2 = int(y0 - L * (a))
            cv2.line(output_image, (x1, y1), (x2, y2), color_mid_lines, line_thickness)
        
        # Calculate average distance between centered lines
        distances = []
        for i in range(len(mid_lines) - 1):
            rho1, theta1 = mid_lines[i]
            rho2, theta2 = mid_lines[i + 1]
            x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)
            x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            distances.append(distance)
        
        avg_distance_detected = np.mean(distances) if distances else 0
        avg_distance_designed = W / ring_count
        
        if np.abs(avg_distance_detected - (1.2 / resolution)) <= np.abs(avg_distance_designed - (1.2 / resolution)):
            avg_distance = avg_distance_detected
        else:
            avg_distance = avg_distance_designed
        
        all_mid_lines = mid_lines.copy()
        
        if mid_lines:
            leftmost_rho, leftmost_theta = mid_lines[0]
            a = np.cos(leftmost_theta)
            b = np.sin(leftmost_theta)
            x0 = a * leftmost_rho
            y0 = b * leftmost_rho
            
            while x0 >= 0:
                x1 = int(x0 + L * (-b))
                y1 = int(y0 + L * (a))
                x2 = int(x0 - L * (-b))
                y2 = int(y0 - L * (a))
                cv2.line(output_image, (x1, y1), (x2, y2), color_mid_lines, line_thickness)
                all_mid_lines.append((x0, leftmost_theta))
                x0 -= avg_distance
            
            rightmost_rho, rightmost_theta = mid_lines[-1]
            a = np.cos(rightmost_theta)
            b = np.sin(rightmost_theta)
            x0 = a * rightmost_rho
            y0 = b * rightmost_rho
            
            while x0 <= output_image.shape[1]:
                x1 = int(x0 + L * (-b))
                y1 = int(y0 + L * (a))
                x2 = int(x0 - L * (-b))
                y2 = int(y0 - L * (a))
                cv2.line(output_image, (x1, y1), (x2, y2), color_mid_lines, line_thickness)
                all_mid_lines.append((x0, rightmost_theta))
                x0 += avg_distance
        
        all_mid_lines = sorted(list(set(all_mid_lines)), key=lambda line: line[0])
    
    # Fallback: Generate evenly spaced vertical lines if no lines were detected
    if lines_vertical is None or len(all_mid_lines) == 0:
        print("No vertical lines detected. Using fallback method.")
        all_mid_lines = []
        block_width = W / ring_count
        
        for i in range(ring_count):
            x_pos = (i + 0.5) * block_width
            all_mid_lines.append((x_pos, 0))
            x1, y1 = int(x_pos), 0
            x2, y2 = int(x_pos), L
            cv2.line(output_image, (x1, y1), (x2, y2), color_mid_lines, line_thickness)
        
        print(f"Generated {len(all_mid_lines)} synthetic vertical lines at ring centers")
    
    # Save visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image)
    os.makedirs(tunnel_dir, exist_ok=True)
    plt.savefig(os.path.join(tunnel_dir, 'detected_lines.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cell 6: Intersection detection
    def line_segment_vertical_intersection(vertical_x, segment):
        x1, y1, x2, y2 = segment
        if x1 == x2:
            return None
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            t = (vertical_x - x1) / (x2 - x1)
            intersect_y = y1 + t * (y2 - y1)
            return (vertical_x, intersect_y)
        return None
    
    def merge_close_points(points, threshold=6):
        points = np.array(points)
        if len(points) == 0:
            return np.array([])
        if len(points) == 1:
            return points
        merged_points = []
        while len(points) > 0:
            p = points[0]
            close_points = np.linalg.norm(points - p, axis=1) < threshold
            merged_points.append(np.mean(points[close_points], axis=0))
            points = points[~close_points]
        return np.array(merged_points)
    
    def compute_midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    def check_distance_pattern(points, k, ab, tolerance=10):
        points = sorted(points, key=lambda p: p[0])
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if any(abs(distance - (k + m * ab)) < tolerance for m in [2, 4]):
                    return compute_midpoint(points[i], points[j])
        return None
    
    # Input data
    vertical_lines = all_mid_lines
    horizontal_lines = joint_horizontal
    positive_slope_lines = joint_oblique_positive
    negative_slope_lines = joint_oblique_negtive
    
    # Variables to hold results
    adjusted_points = []
    
    # Preset values for distance pattern checking
    K_height_pixel = 1079.92 / (1000 * resolution)
    AB_height_pixel = 3239.77 / (1000 * resolution)
    
    # Detect intersections
    for vertical_x, _ in vertical_lines:
        intersections_with_positive_slope = []
        intersections_with_negative_slope = []
        intersections_with_horizontal = []
        
        for segment in positive_slope_lines:
            inter_point = line_segment_vertical_intersection(vertical_x, segment[0])
            if inter_point:
                intersections_with_positive_slope.append(inter_point)
        
        for segment in negative_slope_lines:
            inter_point = line_segment_vertical_intersection(vertical_x, segment[0])
            if inter_point:
                intersections_with_negative_slope.append(inter_point)
        
        merge_positive = merge_close_points(intersections_with_positive_slope)
        merge_negative = merge_close_points(intersections_with_negative_slope)
        
        # Case 1: Intersecting two different types of slope lines
        if len(merge_positive) > 0 and len(merge_negative) > 0:
            midpoint = compute_midpoint(merge_positive[0], merge_negative[0])
            adjusted_points.append(('midpoint', midpoint))
        
        # Case 2: Only positive slope intersections
        elif len(merge_positive) > 0:
            point = merge_positive[0]
            adjusted_points.append(('positive_slope', (point[0], point[1] - 0.5 * K_height_pixel)))
        
        # Case 3: Only negative slope intersections
        elif len(merge_negative) > 0:
            point = merge_negative[0]
            adjusted_points.append(('negative_slope', (point[0], point[1] + 0.5 * K_height_pixel)))
        
        # Case 4: Check intersections with horizontal lines
        else:
            for segment in horizontal_lines:
                inter_point = line_segment_vertical_intersection(vertical_x, segment[0])
                if inter_point:
                    intersections_with_horizontal.append(inter_point)
            merge_horizontal = merge_close_points(intersections_with_horizontal)
            
            pattern_midpoint = check_distance_pattern(merge_horizontal, K_height_pixel, AB_height_pixel, tolerance=50)
            if pattern_midpoint:
                adjusted_points.append(('horizontal', pattern_midpoint))
            else:
                # Determine y-coordinate based on previous point
                if adjusted_points:
                    last_point_y = adjusted_points[-1][1][1]
                    if 1035 <= last_point_y <= 1265:
                        assumed_y = last_point_y + 431.87
                    elif 1422 <= last_point_y <= 1738:
                        assumed_y = last_point_y - 431.87
                    else:
                        if len(adjusted_points) > 1:
                            second_last_point_y = adjusted_points[-2][1][1]
                            if 1035 <= second_last_point_y <= 1265:
                                assumed_y = second_last_point_y
                            elif 1422 <= second_last_point_y <= 1738:
                                assumed_y = second_last_point_y
                            else:
                                assumed_y = None
                        else:
                            assumed_y = None
                else:
                    assumed_y = None
                
                if assumed_y is not None:
                    adjusted_points.append(('assume', (vertical_x, assumed_y)))
                else:
                    default_y = L / 2
                    adjusted_points.append(('default', (vertical_x, default_y)))
                    print(f"Warning: Using default y-coordinate ({default_y}) for vertical line at x = {vertical_x}")
    
    # Recording initial point coordinate
    df_loc = pd.DataFrame(adjusted_points, columns=['Type', 'Coordinates'])
    df_loc['X'] = df_loc['Coordinates'].apply(lambda coord: coord[0])
    df_loc['Y'] = df_loc['Coordinates'].apply(lambda coord: coord[1])
    df_loc = df_loc.drop(columns=['Coordinates'])
    df_loc = df_loc.sort_values(by='X').reset_index(drop=True)
    
    print(f"Number of vertical lines: {len(vertical_lines)}")
    print(f"Number of adjusted points: {len(adjusted_points)}")
    print("DataFrame:")
    print(df_loc)
    
    # Save results
    df_loc.to_csv(os.path.join(tunnel_dir, 'detected.csv'), index=False)
    
    return df_loc, segment_count


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tunnel segment detection")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4)")
    parser.add_argument("--segments", type=int, default=None, help="Number of segments (auto-detect if omitted)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    run_detection(args.tunnel_id, base_dir=args.data_dir, segment_count=args.segments)

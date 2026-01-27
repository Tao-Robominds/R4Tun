"""
Algorithm 4-1 - Complex Staggered Detection (T4/T5 patterns)

Complete detection pipeline for complex_staggered patterns (4-1, 5-1).
Uses oblique line intersections and midpoints instead of evenly spaced ring centers.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# Import line detection from main detection script
import importlib.util
spec = importlib.util.spec_from_file_location("detection", 
    os.path.join(os.path.dirname(__file__), "4-1_detection.py"))
detection_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection_module)

# Import clustering for complex detection
from sklearn.cluster import DBSCAN, AgglomerativeClustering


def mm_to_px(mm: float, resolution: float = 0.005) -> float:
    """Convert millimeters to pixels."""
    return mm / (resolution * 1000)


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


def detect_oblique_lines_wide_angle(dilated_edges, L, W, params=None):
    """Detect oblique lines with wider angle range for complex_staggered patterns."""
    # Get parameters with defaults
    def get_param(keys, default):
        if params is None:
            return default
        value = params
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    threshold = get_param(['complex_staggered', 'hough_re_detect', 'threshold'], 30)
    min_length = get_param(['complex_staggered', 'hough_re_detect', 'min_length'], 50)
    max_gap = get_param(['complex_staggered', 'hough_re_detect', 'max_gap'], 100)
    
    angle_pos_min = get_param(['complex_staggered', 'angle_range', 'positive_min'], 4)
    angle_pos_max = get_param(['complex_staggered', 'angle_range', 'positive_max'], 12)
    angle_neg_min = get_param(['complex_staggered', 'angle_range', 'negative_min'], -12)
    angle_neg_max = get_param(['complex_staggered', 'angle_range', 'negative_max'], -4)
    min_y_span = get_param(['complex_staggered', 'line_filtering', 'min_y_span'], 30)
    min_x_span = get_param(['complex_staggered', 'line_filtering', 'min_x_span'], 30)
    
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


def calculate_k_positions_complex_staggered(
    line_data: Dict,
    ring_count: int,
    k_height_mm: float,
    ab_height_mm: float,
    resolution: float,
    params: dict = None,
) -> pd.DataFrame:
    """Calculate K positions for complex_staggered patterns using oblique line intersections."""
    L = line_data['image_height']
    W = line_data['image_width']
    dilated_edges = line_data['dilated_edges']
    positive_lines = line_data['positive_lines']
    negative_lines = line_data['negative_lines']
    
    # Get parameter helper
    def get_param(keys, default):
        if params is None:
            return default
        value = params
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
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
    
    # Get clustering parameters
    eps_candidates = get_param(['complex_staggered', 'clustering', 'eps_candidates'], [0.03, 0.05, 0.08, 0.10, 0.15])
    min_clusters = get_param(['complex_staggered', 'clustering', 'min_clusters'], 5)
    
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
    # Get subdivision parameters
    subdivision_threshold = get_param(['complex_staggered', 'clustering', 'subdivision_threshold'], 1.5)
    max_subdivisions = get_param(['complex_staggered', 'clustering', 'max_subdivisions'], None)
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
                # Get confidence calculation parameters
                conf_base = get_param(['complex_staggered', 'confidence', 'subdivision_base'], 0.5)
                conf_factor = get_param(['complex_staggered', 'confidence', 'subdivision_factor'], 0.05)
                k_positions.append(('intersection_sub', np.mean(sub[:, 0]), np.mean(sub[:, 1]), 
                                   min(1.0, conf_base + conf_factor * len(sub))))
        else:
            conf_base = get_param(['complex_staggered', 'confidence', 'cluster_base'], 0.5)
            conf_factor = get_param(['complex_staggered', 'confidence', 'cluster_factor'], 0.1)
            k_positions.append(('intersection', np.mean(cluster_points[:, 0]), np.mean(cluster_points[:, 1]),
                               min(1.0, conf_base + conf_factor * len(cluster_points))))
    
    k_positions.sort(key=lambda p: p[1])
    print(f"    Found {len(k_positions)} K position clusters from intersections")
    
    # Add line midpoints and cluster to get ring_count positions
    if len(k_positions) > 0:
        # Get midpoint confidence
        midpoint_confidence = get_param(['complex_staggered', 'confidence', 'midpoint'], 0.7)
        
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
            for label in range(n_clusters):
                mask = labels == label
                cluster_points = candidate_array[mask]
                cluster_types = [all_candidates[i][0] for i, m in enumerate(mask) if m]
                det_type = 'intersection_cluster' if 'intersection' in str(cluster_types) else 'midpoint_cluster'
                intersection_conf = get_param(['complex_staggered', 'confidence', 'final_intersection'], 0.9)
                midpoint_conf = get_param(['complex_staggered', 'confidence', 'final_midpoint'], 0.6)
                confidence = intersection_conf if 'intersection' in str(cluster_types) else midpoint_conf
                final_positions.append((det_type, np.mean(cluster_points[:, 0]), 
                                       np.mean(cluster_points[:, 1]), confidence))
            k_positions = final_positions
        
        k_positions.sort(key=lambda p: p[1])
        print(f"    Final K positions: {len(k_positions)}")
    
    df = pd.DataFrame(k_positions, columns=['Type', 'X', 'Y', 'Confidence'])
    return df.sort_values(by='X').reset_index(drop=True)

DEFAULT_K_HEIGHT_MM = 1079.92
DEFAULT_AB_HEIGHT_MM = 3239.77
DEFAULT_RESOLUTION = 0.005


def load_parameters(tunnel_id: str, base_dir: str = "data") -> dict:
    """Load parameters from JSON."""
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_detection.json"

    params_path = os.path.join(script_dir, "parameters", tunnel_id, param_file)
    if os.path.exists(params_path):
        print(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as f:
            return json.load(f)

    tunnel_path = os.path.join(base_dir, tunnel_id, param_file)
    if os.path.exists(tunnel_path):
        print(f"Loading parameters from {tunnel_path}")
        with open(tunnel_path, 'r') as f:
            return json.load(f)

    sample_path = os.path.join(script_dir, "parameters", "sample", param_file)
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


def visualize_detection(line_data: Dict, k_positions: pd.DataFrame, tunnel_dir: str):
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
    
    # Draw K positions (yellow circles)
    for _, row in k_positions.iterrows():
        cv2.circle(output_image, (int(row['X']), int(row['Y'])), 8, (0, 255, 255), -1)
        # Draw vertical line at X position
        cv2.line(output_image, (int(row['X']), 0), (int(row['X']), L), color_vertical, 1)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(output_image)
    plt.title('Complex Staggered Detection Results')
    plt.savefig(os.path.join(tunnel_dir, 'detected_lines.png'), dpi=300, bbox_inches='tight')
    plt.close()


def run_detection(tunnel_id: str, base_dir: str = "data") -> pd.DataFrame:
    """
    Run complex staggered detection pipeline for T4/T5 patterns.
    """
    # Load parameters
    params = load_parameters(tunnel_id, base_dir)
    
    # Physical constants
    resolution = get_param(params, 'physical_constants', 'resolution', default=DEFAULT_RESOLUTION)
    k_height_mm = get_param(params, 'physical_constants', 'k_height_mm', default=DEFAULT_K_HEIGHT_MM)
    ab_height_mm = get_param(params, 'physical_constants', 'ab_height_mm', default=DEFAULT_AB_HEIGHT_MM)

    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    print(f"{'=' * 60}")
    print(f"Complex Staggered Detection Pipeline for Tunnel: {tunnel_id}")
    print(f"{'=' * 60}")
    
    # Load data
    depth_map_outlier = np.load(os.path.join(tunnel_dir, "depth_map_outlier.npy"))
    ring_count = int(open(os.path.join(tunnel_dir, 'ring_count.txt'), 'r').read())
    L, W = depth_map_outlier.shape
    
    # Use line detection from main detection script
    print(f"\n[Step 1] Detecting lines...")
    line_data = detection_module.detect_lines(depth_map_outlier, params, resolution)
    print(f"  Positive slope lines: {len(line_data['positive_lines'])}")
    print(f"  Negative slope lines: {len(line_data['negative_lines'])}")
    print(f"  Horizontal lines: {len(line_data['horizontal_lines'])}")
    print(f"  Vertical lines: {len(line_data['vertical_lines'])}")
    
    # Use complex staggered K position calculation
    print(f"\n[Step 2] Calculating K positions (complex staggered)...")
    k_positions = calculate_k_positions_complex_staggered(
        line_data, ring_count, k_height_mm, ab_height_mm, resolution,
        params=params,
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Complex staggered detection for T4/T5")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 4-1, 5-1)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    k_positions = run_detection(args.tunnel_id, base_dir=args.data_dir)

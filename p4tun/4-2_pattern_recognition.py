"""
Algorithm 4-2 - K-Block Pattern Detection
Discovers tunnel segment patterns from detection results:
1. Pattern type detection (6-seg alternating, 6-seg constant, 7-seg wrap-around)
2. K-block position calculation per ring using physical geometry
3. Outputs pattern.csv with per-ring K positions for SAM
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

# =============================================================================
# Physical Constants
# =============================================================================

# Standard segment dimensions (mm)
K_HEIGHT_MM = 1079.92
AB_HEIGHT_MM = 3239.77
SEGMENT_WIDTH_MM = 1200
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
    
    default_path = os.path.join(script_dir, "parameters_detection.json")
    if os.path.exists(default_path):
        print(f"Loading default parameters from {default_path}")
        with open(default_path, 'r') as f:
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
# Segment Count Detection
# =============================================================================

def detect_segment_count(image_height: int, resolution: float = DEFAULT_RESOLUTION) -> int:
    """
    Auto-detect 6 or 7 segments from image height.
    
    Circumference for 6 segments: K + 5×AB = 17278.77 mm
    Circumference for 7 segments: K + 6×AB = 20518.54 mm
    """
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
# V-Pair Detection (Oblique Line Intersections)
# =============================================================================

def detect_v_pairs(depth_map_path: str, params: dict, resolution: float = DEFAULT_RESOLUTION) -> Dict:
    """
    Detect V-pairs (oblique line intersections) from the depth map.
    V-pairs indicate segment boundaries and are key to K-block detection.
    
    Returns:
        Dictionary with per-ring V-pair midpoints and detection quality
    """
    # Load depth map with outliers
    depth_map_outlier = np.load(depth_map_path.replace('depth_map.png', 'depth_map_outlier.npy'))
    ring_count = int(open(depth_map_path.replace('depth_map.png', 'ring_count.txt'), 'r').read())
    
    L, W = depth_map_outlier.shape
    
    # Pre-processing
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    ret, binary_image = cv2.threshold(binary_map, 127, 255, cv2.THRESH_BINARY)
    kernel_size = get_param(params, 'preprocessing', 'dilation_kernel_size', default=3)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edges = cv2.dilate(binary_image, kernel, iterations=1)
    
    # Hough parameters
    threshold = get_param(params, 'hough_oblique', 'threshold', default=40)
    min_length = get_param(params, 'hough_oblique', 'min_length', default=120)
    max_gap = get_param(params, 'hough_oblique', 'max_gap', default=48)
    angle_pos_min = get_param(params, 'hough_oblique', 'angle_positive_min', default=6.0)
    angle_pos_max = get_param(params, 'hough_oblique', 'angle_positive_max', default=9.0)
    angle_neg_min = get_param(params, 'hough_oblique', 'angle_negative_min', default=-9.0)
    angle_neg_max = get_param(params, 'hough_oblique', 'angle_negative_max', default=-6.0)
    
    # Detect oblique lines
    lines_oblique = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, threshold, 
                                    minLineLength=min_length, maxLineGap=max_gap)
    
    # Separate positive and negative slope lines
    positive_lines = []
    negative_lines = []
    
    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            x1, x2, y1, y2 = (x2, x1, y2, y1) if x1 > x2 else (x1, x2, y1, y2)
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            
            if angle_pos_min <= angle <= angle_pos_max:
                positive_lines.append(line[0])
            elif angle_neg_min <= angle <= angle_neg_max:
                negative_lines.append(line[0])
    
    # Detect vertical ring boundaries
    lines_vertical = cv2.HoughLines(dilated_edges, 1, np.pi / 180, 500)
    ring_boundaries = []
    
    if lines_vertical is not None:
        # Filter and merge vertical lines
        merged_lines = []
        for rho, theta in lines_vertical[:, 0]:
            if abs(theta) <= 0.5 * np.pi / 180:
                x_pos = rho * np.cos(theta)
                if x_pos <= (5 * 1200 / (resolution * 1000)):
                    merged = False
                    for i, (mrho, mtheta) in enumerate(merged_lines):
                        mx = mrho * np.cos(mtheta)
                        if abs(x_pos - mx) < 3:
                            merged_lines[i] = ((rho + mrho) / 2, (theta + mtheta) / 2)
                            merged = True
                            break
                    if not merged:
                        merged_lines.append((rho, theta))
        
        # Get ring center positions
        merged_lines.sort(key=lambda l: l[0])
        for i in range(len(merged_lines) - 1):
            rho1, _ = merged_lines[i]
            rho2, _ = merged_lines[i + 1]
            ring_boundaries.append((rho1 + rho2) / 2)
    
    # Fallback: evenly spaced ring boundaries
    if len(ring_boundaries) == 0:
        block_width = W / ring_count
        ring_boundaries = [(i + 0.5) * block_width for i in range(ring_count)]
    
    # Find V-pair midpoints per ring
    K_HEIGHT_PX = mm_to_px(K_HEIGHT_MM, resolution)
    v_pairs = {}
    
    def line_segment_vertical_intersection(vertical_x, x1, y1, x2, y2):
        if x1 == x2:
            return None
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            t = (vertical_x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
        return None
    
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
        
        # Find valid V-pairs (positive/negative pairs with K_HEIGHT spacing)
        midpoints = []
        qualities = []
        
        for pos_y in pos_intersections:
            for neg_y in neg_intersections:
                spacing = abs(pos_y - neg_y)
                # V-pairs should have approximately K_HEIGHT spacing
                if abs(spacing - K_HEIGHT_PX) < 60:  # tolerance
                    midpoint = (pos_y + neg_y) / 2
                    quality = 1.0 - abs(spacing - K_HEIGHT_PX) / 60
                    midpoints.append(midpoint)
                    qualities.append(quality)
        
        if midpoints:
            # Take the best quality midpoint
            best_idx = np.argmax(qualities)
            v_pairs[ring_idx] = {
                'midpoint': midpoints[best_idx],
                'quality': qualities[best_idx],
                'ring_x': ring_x,
                'all_midpoints': midpoints
            }
    
    return {
        'v_pairs': v_pairs,
        'ring_boundaries': ring_boundaries,
        'ring_count': ring_count,
        'image_height': L,
        'image_width': W,
        'positive_lines': len(positive_lines),
        'negative_lines': len(negative_lines)
    }


# =============================================================================
# Pattern Type Detection
# =============================================================================

def detect_pattern_type(v_pair_data: Dict, resolution: float = DEFAULT_RESOLUTION) -> Dict:
    """
    Detect the K-block pattern type based on V-pair characteristics.
    
    Pattern types:
    - 6seg_alternating: K position alternates between two positions ~432px apart
    - 6seg_constant: K position is constant across all rings
    - 7seg_wraparound: K position varies significantly due to wrap-around
    
    Returns:
        Dictionary with pattern type, confidence, and detection metrics
    """
    v_pairs = v_pair_data['v_pairs']
    ring_count = v_pair_data['ring_count']
    image_height = v_pair_data['image_height']
    
    # Calculate pixel constants
    K_HEIGHT_PX = mm_to_px(K_HEIGHT_MM, resolution)
    AB_HEIGHT_PX = mm_to_px(AB_HEIGHT_MM, resolution)
    expected_alternation = 2/3 * AB_HEIGHT_PX  # ~432px
    
    # Determine segment count from image height
    segment_count = detect_segment_count(image_height, resolution)
    
    # Initialize metrics
    metrics = {
        'spread_px': 0,
        'cluster_separation_px': 0,
        'ring_changes': [],
        'num_v_pairs': len(v_pairs),
        'detection_rate': len(v_pairs) / ring_count if ring_count > 0 else 0
    }
    
    # Handle sparse V-pair cases with educated guesses based on segment count
    if len(v_pairs) < 2:
        # Not enough V-pairs for analysis - use segment count as primary indicator
        if segment_count == 7:
            # 7-segment tunnels typically have wrap-around patterns
            pattern_type = '7seg_wraparound'
            confidence = 0.6  # Lower confidence due to sparse data
            reason = 'Inferred from segment count (sparse V-pair data)'
        else:
            # 6-segment tunnels are typically alternating in this dataset
            pattern_type = '6seg_alternating'
            confidence = 0.7  # Default to alternating as it's most common
            reason = 'Default to alternating (sparse V-pair data)'
        
        return {
            'pattern_type': pattern_type,
            'segment_count': segment_count,
            'confidence': confidence,
            'reason': reason,
            'metrics': metrics
        }
    
    # Extract midpoints for analysis
    midpoints = [v_pairs[r]['midpoint'] for r in sorted(v_pairs.keys())]
    
    # Metric 1: V-pair spread (max - min midpoint position)
    spread = max(midpoints) - min(midpoints)
    metrics['spread_px'] = spread
    
    # Metric 2: Ring-to-ring changes
    ring_changes = []
    sorted_rings = sorted(v_pairs.keys())
    for i in range(len(sorted_rings) - 1):
        r1, r2 = sorted_rings[i], sorted_rings[i + 1]
        change = v_pairs[r2]['midpoint'] - v_pairs[r1]['midpoint']
        ring_changes.append(change)
    metrics['ring_changes'] = ring_changes
    
    # Metric 3: Clustering - check if midpoints cluster into 2 groups
    midpoints_array = np.array(midpoints)
    cluster_separation = 0
    if len(midpoints) >= 2:
        # Simple 2-cluster analysis
        median = np.median(midpoints_array)
        lower = midpoints_array[midpoints_array <= median]
        higher = midpoints_array[midpoints_array > median]
        if len(lower) > 0 and len(higher) > 0:
            cluster_separation = np.mean(higher) - np.mean(lower)
    metrics['cluster_separation_px'] = cluster_separation
    
    # Pattern classification
    if segment_count == 7:
        # 7-segment tunnels: check for wrap-around pattern
        # Indicators: large spread (>400px), variable ring changes
        if spread > 400 or cluster_separation > 800:
            pattern_type = '7seg_wraparound'
            confidence = 0.8 + 0.2 * min(spread / 800, 1.0)
        else:
            # Still assume wrap-around for 7-segment (most common)
            pattern_type = '7seg_wraparound'
            confidence = 0.7
    else:
        # 6-segment tunnels: alternating vs constant
        if abs(cluster_separation - expected_alternation) < 100:
            pattern_type = '6seg_alternating'
            confidence = 1.0 - abs(cluster_separation - expected_alternation) / 200
        elif spread < 100:
            pattern_type = '6seg_constant'
            confidence = 1.0 - spread / 200
        else:
            # Default to alternating (most common in dataset)
            pattern_type = '6seg_alternating'
            confidence = 0.7
    
    return {
        'pattern_type': pattern_type,
        'segment_count': segment_count,
        'confidence': min(confidence, 1.0),
        'metrics': metrics
    }


# =============================================================================
# K Position Calculation
# =============================================================================

def calculate_k_positions_alternating(v_pair_data: Dict, pattern_info: Dict, 
                                       resolution: float = DEFAULT_RESOLUTION) -> pd.DataFrame:
    """
    Calculate K positions for 6-segment alternating tunnels.
    
    Key insight: V-pairs are detected at the ALTERNATE K position.
    Correction: actual_K = V_midpoint + AB_HEIGHT/2
    """
    v_pairs = v_pair_data['v_pairs']
    ring_boundaries = v_pair_data['ring_boundaries']
    ring_count = v_pair_data['ring_count']
    image_height = v_pair_data['image_height']
    
    K_HEIGHT_PX = mm_to_px(K_HEIGHT_MM, resolution)
    AB_HEIGHT_PX = mm_to_px(AB_HEIGHT_MM, resolution)
    ALTERNATION_OFFSET = 2/3 * AB_HEIGHT_PX  # ~432px
    CORRECTION_OFFSET = AB_HEIGHT_PX / 2  # ~324px
    
    # Step 1: Apply correction to get actual K positions from V-pairs
    corrected_positions = {}
    for ring_idx, data in v_pairs.items():
        # V-pairs appear at alternate position, add AB_HEIGHT/2 for correction
        corrected = data['midpoint'] + CORRECTION_OFFSET
        corrected_positions[ring_idx] = {
            'raw_midpoint': data['midpoint'],
            'corrected': corrected,
            'quality': data['quality'],
            'ring_x': data['ring_x']
        }
    
    # Step 2: Cluster corrected positions to find the two K positions
    if len(corrected_positions) >= 2:
        positions = [d['corrected'] for d in corrected_positions.values()]
        median = np.median(positions)
        lower = [p for p in positions if p <= median]
        higher = [p for p in positions if p > median]
        
        pos1 = np.mean(lower) if lower else median - ALTERNATION_OFFSET / 2
        pos2 = np.mean(higher) if higher else median + ALTERNATION_OFFSET / 2
    else:
        # Fallback: use image center as reference
        center = image_height * 0.4
        pos1 = center
        pos2 = center + ALTERNATION_OFFSET
    
    # Step 3: Assign K position to each ring
    results = []
    last_pos_idx = None
    
    for ring_idx in range(ring_count):
        ring_x = ring_boundaries[ring_idx] if ring_idx < len(ring_boundaries) else (ring_idx + 0.5) * (v_pair_data['image_width'] / ring_count)
        
        if ring_idx in corrected_positions:
            # Use detected position
            k_y = corrected_positions[ring_idx]['corrected']
            quality = corrected_positions[ring_idx]['quality']
            detection_type = 'v_pair_corrected'
            
            # Determine which position this is closer to
            if abs(k_y - pos1) < abs(k_y - pos2):
                last_pos_idx = 0
            else:
                last_pos_idx = 1
        else:
            # Infer from alternation pattern
            if last_pos_idx is not None:
                # Alternate from last position
                inferred_pos_idx = 1 - last_pos_idx
            else:
                # No prior - use position closest to center
                inferred_pos_idx = 0 if abs(pos1 - image_height * 0.4) < abs(pos2 - image_height * 0.4) else 1
            
            k_y = pos1 if inferred_pos_idx == 0 else pos2
            quality = 0.5  # Lower quality for inferred positions
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
    
    return pd.DataFrame(results)


def calculate_k_positions_constant(v_pair_data: Dict, pattern_info: Dict,
                                   resolution: float = DEFAULT_RESOLUTION) -> pd.DataFrame:
    """
    Calculate K positions for 6-segment constant tunnels (non-alternating).
    Uses depth variance analysis as primary method.
    """
    v_pairs = v_pair_data['v_pairs']
    ring_boundaries = v_pair_data['ring_boundaries']
    ring_count = v_pair_data['ring_count']
    image_height = v_pair_data['image_height']
    
    K_HEIGHT_PX = mm_to_px(K_HEIGHT_MM, resolution)
    AB_HEIGHT_PX = mm_to_px(AB_HEIGHT_MM, resolution)
    
    # If we have V-pairs, use their mean
    if len(v_pairs) >= 1:
        midpoints = [v_pairs[r]['midpoint'] for r in v_pairs]
        k_position = np.mean(midpoints) + AB_HEIGHT_PX / 2  # Apply correction
    else:
        # Fallback: use position prior (40% of image height)
        k_position = image_height * 0.4
    
    results = []
    for ring_idx in range(ring_count):
        ring_x = ring_boundaries[ring_idx] if ring_idx < len(ring_boundaries) else (ring_idx + 0.5) * (v_pair_data['image_width'] / ring_count)
        
        quality = 0.7 if ring_idx in v_pairs else 0.5
        detection_type = 'v_pair_mean' if ring_idx in v_pairs else 'position_prior'
        
        results.append({
            'ring': ring_idx,
            'X': ring_x,
            'Y': k_position,
            'quality': quality,
            'detection_type': detection_type,
            'position_class': 'constant'
        })
    
    return pd.DataFrame(results)


def calculate_k_positions_wraparound(v_pair_data: Dict, pattern_info: Dict,
                                     resolution: float = DEFAULT_RESOLUTION) -> pd.DataFrame:
    """
    Calculate K positions for 7-segment wrap-around tunnels.
    Uses direct V-pair detection with propagation to neighbors.
    
    For 7-segment tunnels, K position can be at any of 7 "slots" due to wrap-around.
    We use detected V-pairs directly and propagate to undetected rings.
    """
    v_pairs = v_pair_data['v_pairs']
    ring_boundaries = v_pair_data['ring_boundaries']
    ring_count = v_pair_data['ring_count']
    image_height = v_pair_data['image_height']
    image_width = v_pair_data['image_width']
    
    K_HEIGHT_PX = mm_to_px(K_HEIGHT_MM, resolution)
    AB_HEIGHT_PX = mm_to_px(AB_HEIGHT_MM, resolution)
    
    # For 7-segment, calculate possible slot positions
    # K can be at positions separated by AB_HEIGHT from each other
    # Center position is approximately at 40% of image height for typical tunnels
    center_estimate = image_height * 0.4
    
    # For 7-segment, use direct V-pair midpoints where available
    results = []
    detected_positions = {}
    
    for ring_idx in v_pairs:
        # Use midpoint directly for 7-segment (V-pairs indicate K boundaries)
        detected_positions[ring_idx] = v_pairs[ring_idx]['midpoint']
    
    for ring_idx in range(ring_count):
        # Calculate ring X position
        if ring_idx < len(ring_boundaries):
            ring_x = ring_boundaries[ring_idx]
        else:
            ring_x = (ring_idx + 0.5) * (image_width / ring_count)
        
        if ring_idx in detected_positions:
            k_y = detected_positions[ring_idx]
            quality = v_pairs[ring_idx]['quality']
            detection_type = 'v_pair_direct'
        else:
            # Propagate from nearest detected neighbor
            nearest_ring = None
            min_dist = float('inf')
            for detected_ring in detected_positions:
                dist = abs(detected_ring - ring_idx)
                if dist < min_dist:
                    min_dist = dist
                    nearest_ring = detected_ring
            
            if nearest_ring is not None:
                # For 7-segment, wrap-around means K position can shift by ±AB_HEIGHT
                # between rings. Without more info, use nearest neighbor position.
                k_y = detected_positions[nearest_ring]
                quality = max(0.3, v_pairs[nearest_ring]['quality'] - 0.1 * min_dist)
                detection_type = f'propagated_from_ring_{nearest_ring}'
            else:
                # Fallback: use center estimate
                k_y = center_estimate
                quality = 0.3
                detection_type = 'center_estimate'
        
        results.append({
            'ring': ring_idx,
            'X': ring_x,
            'Y': k_y,
            'quality': quality,
            'detection_type': detection_type,
            'position_class': 'variable'
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Main Pattern Detection Pipeline
# =============================================================================

def run_pattern_detection(tunnel_id: str, base_dir: str = "data") -> Tuple[pd.DataFrame, Dict]:
    """
    Run the complete pattern detection pipeline.
    
    Input:
        - detected.csv from 4-1_detection.py
        - depth_map_outlier.npy
        - ring_count.txt
    
    Output:
        - pattern.csv: per-ring K positions with quality scores
        - pattern.json: pattern metadata and detection parameters
    """
    # Load parameters
    params = load_parameters(tunnel_id, base_dir)
    resolution = get_param(params, 'physical_constants', 'resolution', default=DEFAULT_RESOLUTION)
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    depth_map_path = os.path.join(tunnel_dir, 'depth_map.png')
    
    print(f"=" * 60)
    print(f"Pattern Detection for Tunnel: {tunnel_id}")
    print(f"=" * 60)
    
    # Step 1: Detect V-pairs from depth map
    print("\n[Step 1] Detecting V-pairs...")
    v_pair_data = detect_v_pairs(depth_map_path, params, resolution)
    print(f"  Detected {len(v_pair_data['v_pairs'])} V-pairs out of {v_pair_data['ring_count']} rings")
    print(f"  Positive slope lines: {v_pair_data['positive_lines']}")
    print(f"  Negative slope lines: {v_pair_data['negative_lines']}")
    
    # Step 2: Detect pattern type
    print("\n[Step 2] Detecting pattern type...")
    pattern_info = detect_pattern_type(v_pair_data, resolution)
    print(f"  Pattern type: {pattern_info['pattern_type']}")
    print(f"  Confidence: {pattern_info['confidence']:.2f}")
    print(f"  Segment count: {pattern_info['segment_count']}")
    if pattern_info['metrics']:
        print(f"  Spread: {pattern_info['metrics'].get('spread_px', 0):.1f} px")
        print(f"  Cluster separation: {pattern_info['metrics'].get('cluster_separation_px', 0):.1f} px")
    
    # Step 3: Calculate K positions based on pattern type
    print("\n[Step 3] Calculating K positions...")
    pattern_type = pattern_info['pattern_type']
    
    if 'alternating' in pattern_type:
        k_positions = calculate_k_positions_alternating(v_pair_data, pattern_info, resolution)
    elif 'constant' in pattern_type:
        k_positions = calculate_k_positions_constant(v_pair_data, pattern_info, resolution)
    elif 'wraparound' in pattern_type:
        k_positions = calculate_k_positions_wraparound(v_pair_data, pattern_info, resolution)
    else:
        # Fallback to alternating method
        k_positions = calculate_k_positions_alternating(v_pair_data, pattern_info, resolution)
    
    # Step 4: Add Type column for compatibility with existing SAM code
    k_positions['Type'] = k_positions['detection_type'].apply(
        lambda x: 'midpoint' if 'v_pair' in x else 'inferred'
    )
    
    # Prepare output
    print(f"\n[Step 4] K positions calculated for {len(k_positions)} rings")
    print(f"  Mean quality: {k_positions['quality'].mean():.2f}")
    print(f"  Detection types: {k_positions['detection_type'].value_counts().to_dict()}")
    
    # Save results
    pattern_csv_path = os.path.join(tunnel_dir, 'pattern.csv')
    k_positions.to_csv(pattern_csv_path, index=False)
    print(f"\n  Saved: {pattern_csv_path}")
    
    # Save pattern metadata
    pattern_metadata = {
        'tunnel_id': tunnel_id,
        'pattern_type': pattern_info['pattern_type'],
        'segment_count': pattern_info['segment_count'],
        'confidence': pattern_info['confidence'],
        'metrics': pattern_info['metrics'],
        'v_pair_count': len(v_pair_data['v_pairs']),
        'ring_count': v_pair_data['ring_count'],
        'image_height': v_pair_data['image_height'],
        'image_width': v_pair_data['image_width'],
        'resolution': resolution,
        'physical_constants': {
            'K_HEIGHT_MM': K_HEIGHT_MM,
            'AB_HEIGHT_MM': AB_HEIGHT_MM,
            'K_HEIGHT_PX': mm_to_px(K_HEIGHT_MM, resolution),
            'AB_HEIGHT_PX': mm_to_px(AB_HEIGHT_MM, resolution)
        }
    }
    
    pattern_json_path = os.path.join(tunnel_dir, 'pattern.json')
    with open(pattern_json_path, 'w') as f:
        json.dump(pattern_metadata, f, indent=2)
    print(f"  Saved: {pattern_json_path}")
    
    print(f"\n{'=' * 60}")
    print(f"Pattern detection complete!")
    print(f"{'=' * 60}")
    
    return k_positions, pattern_metadata


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="K-block pattern detection")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    k_positions, pattern_info = run_pattern_detection(args.tunnel_id, base_dir=args.data_dir)
    
    # Display results summary
    print("\nK Position Summary:")
    print(k_positions[['ring', 'X', 'Y', 'quality', 'detection_type']].to_string(index=False))


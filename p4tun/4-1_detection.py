"""
Unified Tunnel Segment Detection & Pattern Inference (GT-Free)

This module provides a UNIVERSAL approach to tunnel segment detection that
handles ALL segment arrangement patterns:

  - ROW PATTERN: K-blocks aligned in 1-2 horizontal bands, segments stacked
  - WRAPAROUND PATTERN: K-blocks rotate across rings, segments wrap around image
  - MIXED PATTERN: Combination of the above
  - (Extensible for future unknown patterns)

NO GROUND TRUTH REQUIRED.

Pipeline:
    1. Preprocess depth map to binary edge image
    2. Detect oblique lines (joint edges at ±7.5° angle)
    3. Detect horizontal lines (K-block boundaries)
    4. Detect vertical lines (ring center separations)
    5. Compute K-block center points at line intersections
    6. AUTO-DETECT pattern type from K-block distribution
    7. INFER ALL SEGMENT POSITIONS using domain knowledge + pattern-aware handling

Key Innovation:
    - Each ring is processed INDEPENDENTLY
    - Y coordinates are handled with modular arithmetic for wraparound
    - Pattern detection is automatic but can be overridden
    - Architecture is extensible for future patterns

Outputs:
    - detected.csv: K-block prompt points (legacy format)
    - inferred_from_pattern.csv: ALL segment positions (for SAM)
    - detected_lines.png: Visualization
"""

import os
import sys
import json
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Parameter Loading
# =============================================================================

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Tuple[Dict[str, Any], bool]:
    """
    Load parameters from JSON file with fallback to defaults.
    
    Priority:
        1. Centralized: p4tun/parameters/<tunnel_id>/parameters_detection.json
        2. Tunnel-specific: data/<tunnel_id>/parameters_detection.json
        3. Default: p4tun/parameters_detection.json (if exists)
        4. Hardcoded defaults (if no file found)
    
    Returns:
        Tuple of (params_dict, was_loaded_from_file)
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
    
    default_path = os.path.join(script_dir, param_file)
    if os.path.exists(default_path):
        print(f"Loading default parameters from {default_path}")
        with open(default_path, 'r') as f:
            return json.load(f), True
    
    print("Warning: No parameter file found, using hardcoded defaults")
    return {}, False


def get_param(params: Dict, *keys, default=None, allow_default: bool = True):
    """
    Get nested parameter value with optional default fallback.
    
    Args:
        params: Parameter dictionary
        keys: Nested keys to traverse
        default: Default value if key not found
        allow_default: If False, raise KeyError instead of using default
    
    Returns:
        Parameter value or default (if allow_default=True)
    """
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if allow_default:
                return default
            else:
                raise KeyError(f"Parameter not found: {' -> '.join(keys)}")
    return value


# =============================================================================
# Default Constants
# =============================================================================

# --- Physical Constants ---
DEFAULT_RING_SPACING_M = 1.2
DEFAULT_K_HEIGHT_MM = 1079.92
DEFAULT_AB_HEIGHT_MM = 3239.77
DEFAULT_SEGMENT_WIDTH_MM = 1200
DEFAULT_OBLIQUE_ANGLE_DEG = 7.52
DEFAULT_RESOLUTION = 0.005

# Computed circumferences for auto-detection
CIRCUMFERENCE_6_SEGMENTS_MM = DEFAULT_K_HEIGHT_MM + 5 * DEFAULT_AB_HEIGHT_MM
CIRCUMFERENCE_7_SEGMENTS_MM = DEFAULT_K_HEIGHT_MM + 6 * DEFAULT_AB_HEIGHT_MM

# --- Preprocessing ---
DEFAULT_BINARY_THRESHOLD = 127
DEFAULT_DILATION_KERNEL_SIZE = 3
DEFAULT_DILATION_ITERATIONS = 1

# --- Hough Oblique ---
DEFAULT_OBLIQUE_RHO = 1
DEFAULT_OBLIQUE_THETA_DEG = 1.0
DEFAULT_OBLIQUE_THRESHOLD = 50
DEFAULT_OBLIQUE_MIN_LENGTH = 100
DEFAULT_OBLIQUE_MAX_GAP = 40
DEFAULT_OBLIQUE_ANGLE_POSITIVE_MIN = 6
DEFAULT_OBLIQUE_ANGLE_POSITIVE_MAX = 9
DEFAULT_OBLIQUE_ANGLE_NEGATIVE_MIN = -9
DEFAULT_OBLIQUE_ANGLE_NEGATIVE_MAX = -6

# --- Hough Horizontal ---
DEFAULT_HORIZONTAL_THRESHOLD = 50
DEFAULT_HORIZONTAL_MIN_LENGTH = 100
DEFAULT_HORIZONTAL_MAX_GAP = 10
DEFAULT_HORIZONTAL_ANGLE_TOLERANCE = 1

# --- Hough Vertical ---
DEFAULT_VERTICAL_THRESHOLD = 500
DEFAULT_VERTICAL_ANGLE_TOLERANCE = 0.5

# --- Line Processing ---
DEFAULT_MERGE_DISTANCE_THRESHOLD = 3
DEFAULT_INTERSECTION_MERGE_THRESHOLD = 6
DEFAULT_PATTERN_TOLERANCE = 10
DEFAULT_HORIZONTAL_PATTERN_TOLERANCE = 50

# --- Pattern Detection (CONFIGURABLE) ---
# These can be overridden in parameters_detection.json under "pattern_detection" section
DEFAULT_PARTIAL_SCAN_THRESHOLD = 0.70      # ratio < this → partial scan (no wraparound)
DEFAULT_FULL_SCAN_MIN_THRESHOLD = 0.84     # ratio >= this AND <= max → likely full 360°
DEFAULT_FULL_SCAN_MAX_THRESHOLD = 1.05     # upper bound for full scan ratio
DEFAULT_EDGE_PROXIMITY_PERCENT = 0.05      # K-block within this % of edge → wraparound
DEFAULT_ROTATION_THRESHOLD_PERCENT = 30    # K-block Y range > this % → rotation detected


# =============================================================================
# Segment Count Auto-Detection
# =============================================================================

def detect_segment_count(image_height: int, resolution: float = DEFAULT_RESOLUTION) -> int:
    """
    Auto-detect segment count from image height.
    
    The tunnel circumference (image height in pixels × resolution) corresponds to:
    - 6 segments: K + 5×AB = 17,278.77 mm
    - 7 segments: K + 6×AB = 20,518.54 mm
    
    Args:
        image_height: Height of depth map in pixels.
        resolution: Image resolution in meters per pixel.
        
    Returns:
        Detected segment count (6 or 7).
    """
    # Convert image height to mm
    height_mm = image_height * resolution * 1000
    
    # Calculate distances to expected circumferences
    dist_6 = abs(height_mm - CIRCUMFERENCE_6_SEGMENTS_MM)
    dist_7 = abs(height_mm - CIRCUMFERENCE_7_SEGMENTS_MM)
    
    if dist_6 < dist_7:
        detected = 6
        expected_mm = CIRCUMFERENCE_6_SEGMENTS_MM
    else:
        detected = 7
        expected_mm = CIRCUMFERENCE_7_SEGMENTS_MM
    
    error_percent = abs(height_mm - expected_mm) / expected_mm * 100
    
    print(f"Image height: {image_height} px = {height_mm:.1f} mm")
    print(f"Expected for {detected} segments: {expected_mm:.1f} mm")
    print(f"Auto-detected: {detected} segments (error: {error_percent:.1f}%)")
    
    if error_percent > 10:
        print(f"Warning: Large deviation ({error_percent:.1f}%), detection may be incorrect")
    
    return detected


# =============================================================================
# Image Preprocessing
# =============================================================================

def preprocess_depth_map(
    depth_map: np.ndarray,
    binary_threshold: int = DEFAULT_BINARY_THRESHOLD,
    dilation_kernel_size: int = DEFAULT_DILATION_KERNEL_SIZE,
    dilation_iterations: int = DEFAULT_DILATION_ITERATIONS
) -> np.ndarray:
    """Convert depth map to binary edge image for line detection."""
    # Normalize to 8-bit
    normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Binary threshold
    _, binary = cv2.threshold(normalized, binary_threshold, 255, cv2.THRESH_BINARY)
    
    # Dilate to connect edges
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=dilation_iterations)
    
    return dilated


# =============================================================================
# Line Detection
# =============================================================================

def compute_line_angle(x1: int, y1: int, x2: int, y2: int) -> float:
    """Compute line angle in degrees from horizontal."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    return np.degrees(np.arctan2(dy, dx))


def detect_oblique_lines(
    edge_image: np.ndarray,
    rho: int = DEFAULT_OBLIQUE_RHO,
    theta: float = np.pi / 180,
    threshold: int = DEFAULT_OBLIQUE_THRESHOLD,
    min_length: int = DEFAULT_OBLIQUE_MIN_LENGTH,
    max_gap: int = DEFAULT_OBLIQUE_MAX_GAP,
    angle_pos_min: float = DEFAULT_OBLIQUE_ANGLE_POSITIVE_MIN,
    angle_pos_max: float = DEFAULT_OBLIQUE_ANGLE_POSITIVE_MAX,
    angle_neg_min: float = DEFAULT_OBLIQUE_ANGLE_NEGATIVE_MIN,
    angle_neg_max: float = DEFAULT_OBLIQUE_ANGLE_NEGATIVE_MAX
) -> Tuple[List, List]:
    """Detect oblique joint lines (positive and negative slope)."""
    lines = cv2.HoughLinesP(
        edge_image, rho, theta, threshold,
        minLineLength=min_length, maxLineGap=max_gap
    )
    
    positive_lines = []
    negative_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = compute_line_angle(x1, y1, x2, y2)
            
            if angle_pos_min <= angle <= angle_pos_max:
                positive_lines.append((x1, y1, x2, y2))
            elif angle_neg_min <= angle <= angle_neg_max:
                negative_lines.append((x1, y1, x2, y2))
    
    return positive_lines, negative_lines


def detect_horizontal_lines(
    edge_image: np.ndarray,
    threshold: int = DEFAULT_HORIZONTAL_THRESHOLD,
    min_length: int = DEFAULT_HORIZONTAL_MIN_LENGTH,
    max_gap: int = DEFAULT_HORIZONTAL_MAX_GAP,
    angle_tolerance: float = DEFAULT_HORIZONTAL_ANGLE_TOLERANCE
) -> List:
    """Detect horizontal lines (K-block boundaries)."""
    lines = cv2.HoughLinesP(
        edge_image, 1, np.pi / 180, threshold,
        minLineLength=min_length, maxLineGap=max_gap
    )
    
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = compute_line_angle(x1, y1, x2, y2)
            if abs(angle) <= angle_tolerance:
                horizontal_lines.append((x1, y1, x2, y2))
    
    return horizontal_lines


def detect_vertical_lines(
    edge_image: np.ndarray,
    resolution: float,
    ring_count: int,
    threshold: int = DEFAULT_VERTICAL_THRESHOLD,
    angle_tolerance: float = DEFAULT_VERTICAL_ANGLE_TOLERANCE,
    merge_distance: float = 10.0
) -> List:
    """Detect vertical lines (ring boundaries) and merge close ones.
    
    The old method (OA 0.516 for 5-1) merged close vertical lines and 
    used midpoints between them - NOT dense scanning.
    """
    lines = cv2.HoughLines(edge_image, 1, np.pi / 180, threshold)
    
    vertical_lines = []
    if lines is not None:
        height, width = edge_image.shape
        
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            
            if abs(angle) <= angle_tolerance:
                x = int(rho / np.cos(theta - np.pi / 2)) if abs(np.cos(theta - np.pi / 2)) > 0.01 else int(rho)
                # Keep ALL vertical lines (no filter) 
                if 0 < x < width:
                    vertical_lines.append(x)
    
    # Merge close vertical lines (like old method did)
    if not vertical_lines:
        return []
    
    vertical_lines = sorted(vertical_lines)
    merged = [vertical_lines[0]]
    for x in vertical_lines[1:]:
        if x - merged[-1] > merge_distance:
            merged.append(x)
        else:
            # Merge by averaging
            merged[-1] = (merged[-1] + x) / 2
    
    return merged


def merge_close_lines(lines: List, threshold: float = DEFAULT_MERGE_DISTANCE_THRESHOLD) -> List:
    """Merge lines that are very close together."""
    if not lines:
        return []
    
    lines = sorted(lines)
    merged = [lines[0]]
    
    for line in lines[1:]:
        if line - merged[-1] > threshold:
            merged.append(line)
    
    return merged


# =============================================================================
# Center Line Generation
# =============================================================================

def generate_center_lines(vertical_lines: List, width: int, height: int, ring_count: int) -> List:
    """Generate ring center X-coordinates from detected vertical lines."""
    expected_ring_width = width / ring_count
    
    if len(vertical_lines) < 2:
        print(f"  No valid vertical lines, using uniform distribution")
        return generate_fallback_center_lines(width, ring_count)
    
    # Estimate ring width from vertical lines
    spacings = np.diff(vertical_lines)
    detected_spacing = np.median(spacings) if len(spacings) > 0 else expected_ring_width
    
    # Validate: detected spacing should be close to expected ring width
    # If spacing is less than 50% of expected, the detection is unreliable
    if detected_spacing < expected_ring_width * 0.5:
        print(f"  Warning: Detected spacing ({detected_spacing:.1f}px) << expected ({expected_ring_width:.1f}px)")
        print(f"  Falling back to uniform distribution")
        return generate_fallback_center_lines(width, ring_count)
    
    # Generate center lines using detected spacing
    centers = []
    for i in range(ring_count):
        x = detected_spacing * (i + 0.5)
        if 0 <= x <= width:
            centers.append(x)
    
    # If we got fewer centers than expected, fall back
    if len(centers) < ring_count:
        print(f"  Warning: Only {len(centers)} centers generated, expected {ring_count}")
        return generate_fallback_center_lines(width, ring_count)
    
    return centers


def generate_fallback_center_lines(width: int, ring_count: int) -> List:
    """Generate center lines using uniform distribution."""
    ring_width = width / ring_count
    return [ring_width * (i + 0.5) for i in range(ring_count)]


# =============================================================================
# Intersection Detection
# =============================================================================

def find_intersection(line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
    """Find intersection point of two line segments."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def merge_close_points(points: List[Tuple], threshold: float) -> List[Tuple]:
    """Merge points that are close together."""
    if not points:
        return []
    
    merged = []
    used = set()
    
    for i, (px, py) in enumerate(points):
        if i in used:
            continue
        
        cluster_x = [px]
        cluster_y = [py]
        
        for j, (qx, qy) in enumerate(points):
            if j <= i or j in used:
                continue
            if np.sqrt((px - qx) ** 2 + (py - qy) ** 2) <= threshold:
                cluster_x.append(qx)
                cluster_y.append(qy)
                used.add(j)
        
        merged.append((np.mean(cluster_x), np.mean(cluster_y)))
    
    return merged


# =============================================================================
# K-Block Prompt Point Detection (Combined Method)
# =============================================================================

def find_kblock_edge_pairs(
    positive_ys: List[float],
    negative_ys: List[float],
    k_height_px: float,
    ab_height_px: float,
    tolerance_px: float = 50
) -> List[Tuple[float, float, float, float, str]]:
    """
    Find valid K-block edge pairs using domain knowledge.
    
    K-block has two oblique edges. The distance between K-block's positive
    and negative edges should be approximately K_height. But we may also
    find B-block edges which are AB_height apart.
    
    Returns:
        List of (pos_y, neg_y, mid_y, confidence, type) tuples
    """
    pairs = []
    
    for pos_y in positive_ys:
        for neg_y in negative_ys:
            distance = abs(pos_y - neg_y)
            
            # Check if distance matches K-block height
            k_diff = abs(distance - k_height_px)
            if k_diff <= tolerance_px:
                confidence = 1.0 - k_diff / tolerance_px
                mid_y = (pos_y + neg_y) / 2
                pairs.append((pos_y, neg_y, mid_y, confidence, 'K'))
            
            # Check if distance matches half-K + half-AB (B1 to K edge)
            b_to_k = (k_height_px + ab_height_px) / 2
            b_diff = abs(distance - b_to_k)
            if b_diff <= tolerance_px:
                confidence = 1.0 - b_diff / tolerance_px
                mid_y = (pos_y + neg_y) / 2
                pairs.append((pos_y, neg_y, mid_y, confidence, 'B_to_K'))
    
    # Sort by confidence (highest first), prefer K matches
    pairs.sort(key=lambda x: (-1 if x[4] == 'K' else 0, x[3]), reverse=True)
    return pairs


def compute_prompt_points(
    center_lines: List,
    positive_lines: List,
    negative_lines: List,
    horizontal_lines: List,
    resolution: float,
    height: int,
    intersection_merge_threshold: float = DEFAULT_INTERSECTION_MERGE_THRESHOLD,
    k_height_mm: float = DEFAULT_K_HEIGHT_MM
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    Compute K-block prompt points using COMBINED method:
    1. Detect all line intersections
    2. Use domain knowledge to find valid edge pairs
    3. Prefer midpoint between positive and negative slopes (proven to work)
    
    Returns list of (type, (x, y)) tuples.
    """
    prompt_points = []
    
    # Convert heights to pixels
    k_height_px = k_height_mm / (resolution * 1000)
    ab_height_mm = 3239.77  # Standard AB height
    ab_height_px = ab_height_mm / (resolution * 1000)
    
    for cx in center_lines:
        # Vertical line through center
        center_line = (cx, 0, cx, height)
        
        # Find all intersections with oblique lines
        positive_intersections = []
        for line in positive_lines:
            pt = find_intersection(center_line, line)
            if pt and 0 <= pt[1] <= height:
                positive_intersections.append(pt)
        
        negative_intersections = []
        for line in negative_lines:
            pt = find_intersection(center_line, line)
            if pt and 0 <= pt[1] <= height:
                negative_intersections.append(pt)
        
        # Merge close points
        positive_intersections = merge_close_points(positive_intersections, intersection_merge_threshold)
        negative_intersections = merge_close_points(negative_intersections, intersection_merge_threshold)
        
        # Sort by Y (important for consistency)
        pos_ys = sorted([p[1] for p in positive_intersections])
        neg_ys = sorted([n[1] for n in negative_intersections])
        
        k_center = None
        method = None
        
        # PRIMARY METHOD: Use FIRST positive and FIRST negative slope intersections
        # This is what the old method (OA 0.516) used - simple midpoint between first detections
        # Different rings naturally get different Y positions based on actual line detections
        if pos_ys and neg_ys:
            # Use FIRST (lowest Y) from each - this is what the old "midpoint" method did
            first_pos_y = pos_ys[0]
            first_neg_y = neg_ys[0]
            mid_y = (first_pos_y + first_neg_y) / 2
            k_center = (cx, mid_y)
            method = 'midpoint'
        
        # Fallback methods
        if k_center is None:
            # Find horizontal intersections
            horizontal_intersections = []
            for line in horizontal_lines:
                pt = find_intersection(center_line, line)
                if pt and 0 <= pt[1] <= height:
                    horizontal_intersections.append(pt)
            horizontal_intersections = merge_close_points(horizontal_intersections, intersection_merge_threshold)
            
            if horizontal_intersections:
                h_ys = sorted([h[1] for h in horizontal_intersections])
                mid_idx = len(h_ys) // 2
                k_center = (cx, h_ys[mid_idx])
                method = 'horizontal'
            elif pos_ys:
                k_center = (cx, pos_ys[-1])  # Use last (highest Y)
                method = 'positive_last'
            elif neg_ys:
                k_center = (cx, neg_ys[-1])  # Use last (highest Y)
                method = 'negative_last'
            else:
                k_center = (cx, height * 0.7)  # K-block typically at ~70% of height
                method = 'assume_70pct'
        
        prompt_points.append((method, k_center))
    
    # POST-PROCESSING: Ensure consistency across rings
    # NOTE: Disabled for now - was too aggressive and forced all K-blocks to same Y
    # This destroyed natural variation in K-block positions across rings (ring-to-ring variation / rotation)
    # The old method (OA 0.628) had wide Y range (883-4154) which was correct
    # prompt_points = enforce_kblock_consistency(prompt_points, height, k_height_px)
    
    return prompt_points


def enforce_kblock_consistency(
    prompt_points: List[Tuple[str, Tuple[float, float]]],
    image_height: int,
    k_height_px: float
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    Ensure K-block Y positions are consistent across rings.
    """
    if len(prompt_points) < 3:
        return prompt_points
    
    # Extract Y values
    y_values = [p[1][1] for p in prompt_points]
    
    # Use segment height for clustering
    segment_height = k_height_px * 3  # ~648 px
    
    # Group Y values into bands
    y_sorted = sorted(enumerate(y_values), key=lambda x: x[1])
    
    # Find clusters
    clusters = []
    current_cluster = [y_sorted[0]]
    
    for i in range(1, len(y_sorted)):
        if y_sorted[i][1] - y_sorted[i-1][1] < segment_height:
            current_cluster.append(y_sorted[i])
        else:
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            current_cluster = [y_sorted[i]]
    
    if len(current_cluster) >= 2:
        clusters.append(current_cluster)
    
    if not clusters:
        return prompt_points
    
    # Find the largest cluster
    largest_cluster = max(clusters, key=len)
    cluster_y_mean = np.mean([y[1] for y in largest_cluster])
    cluster_indices = {y[0] for y in largest_cluster}
    
    print(f"  K-block consistency: {len(largest_cluster)}/{len(prompt_points)} in main band (Y~{cluster_y_mean:.0f})")
    
    # Adjust outliers to the cluster mean
    adjusted_points = []
    for i, (method, (cx, cy)) in enumerate(prompt_points):
        if i in cluster_indices:
            adjusted_points.append((method, (cx, cy)))
        else:
            adjusted_points.append((f'{method}_adjusted', (cx, cluster_y_mean)))
            print(f"    Ring {i+1}: Y adjusted {cy:.0f} → {cluster_y_mean:.0f}")
    
    return adjusted_points


# =============================================================================
# Pattern Detection and Strategy
# =============================================================================

class PatternStrategy:
    """Base class for segment position inference strategies."""
    
    @staticmethod
    def detect(k_positions: List[Tuple], image_height: int, k_height_px: float) -> float:
        """Return confidence score (0-1) that this pattern applies."""
        raise NotImplementedError
    
    @staticmethod
    def infer_segments(k_positions: List, image_height: int, **kwargs) -> List[Dict]:
        """Infer all segment positions for this pattern."""
        raise NotImplementedError


def detect_pattern_type(
    k_positions: List[Tuple[str, Tuple[float, float]]],
    image_height: int,
    k_height_px: float,
    ab_height_px: float = None,
    segments_per_ring: int = 6,
    pattern_params: Dict = None
) -> Tuple[str, float]:
    """
    Auto-detect the tunnel pattern type using configurable parameters and adaptive detection.
    
    DETECTION METHODS (in order of priority):
    1. ADAPTIVE: Actually compute segment positions and check if any go out of bounds
    2. HEURISTIC: Use configurable thresholds as fallback
    
    CONFIGURABLE PARAMETERS (in pattern_params or parameters_detection.json):
    - partial_scan_threshold: ratio below which wraparound is impossible (default: 0.70)
    - full_scan_min_threshold: lower bound for full 360° scan ratio (default: 0.84)
    - full_scan_max_threshold: upper bound for full 360° scan ratio (default: 1.05)
    - edge_proximity_percent: K-block edge threshold as % of image (default: 0.05)
    - rotation_threshold_percent: K-block Y range % indicating rotation (default: 30)
    
    Returns:
        Tuple of (pattern_type, confidence):
        - 'row': All segments fit within image bounds (no wraparound needed)
        - 'wraparound': Some segments would extend beyond bounds (need wraparound)
        - 'mixed': Some rings wrap, some don't
    """
    if not k_positions:
        return 'unknown', 0.0
    
    if ab_height_px is None:
        ab_height_px = k_height_px * 3  # Default estimate
    
    # Load configurable thresholds (from params or defaults)
    if pattern_params is None:
        pattern_params = {}
    
    partial_threshold = pattern_params.get('partial_scan_threshold', DEFAULT_PARTIAL_SCAN_THRESHOLD)
    full_scan_min = pattern_params.get('full_scan_min_threshold', DEFAULT_FULL_SCAN_MIN_THRESHOLD)
    full_scan_max = pattern_params.get('full_scan_max_threshold', DEFAULT_FULL_SCAN_MAX_THRESHOLD)
    edge_percent = pattern_params.get('edge_proximity_percent', DEFAULT_EDGE_PROXIMITY_PERCENT)
    rotation_percent = pattern_params.get('rotation_threshold_percent', DEFAULT_ROTATION_THRESHOLD_PERCENT)
    
    # Calculate expected full circumference
    num_ab_blocks = segments_per_ring - 1
    expected_circumference = k_height_px + (num_ab_blocks) * ab_height_px
    circumference_ratio = image_height / expected_circumference
    
    y_values = [p[1][1] for p in k_positions]
    y_min, y_max = min(y_values), max(y_values)
    y_range_percent = (y_max - y_min) / image_height * 100
    
    print(f"  Image height: {image_height}px, Expected circumference: {expected_circumference:.0f}px")
    print(f"  Circumference ratio: {circumference_ratio:.2f}")
    print(f"  K-block Y range: [{y_min:.1f}, {y_max:.1f}] ({y_range_percent:.1f}% of image)")
    
    # =========================================================================
    # QUICK CHECK: Partial scan detection (before expensive computations)
    # =========================================================================
    
    if circumference_ratio < partial_threshold:
        print(f"  QUICK CHECK: Partial scan (ratio {circumference_ratio:.2f} < {partial_threshold})")
        print(f"  → Wraparound impossible for partial scans")
        return 'row', 0.95
    
    # =========================================================================
    # METHOD 1: ADAPTIVE DETECTION - Check if K-blocks are positioned at edges
    # This is more robust than pure threshold-based heuristics
    # =========================================================================
    
    # For full/near-full scans, check K-block edge proximity
    # K-blocks very close to top/bottom edge indicate segments MUST wrap
    edge_threshold = image_height * edge_percent
    k_near_top = y_min < edge_threshold
    k_near_bottom = y_max > (image_height - edge_threshold)
    
    print(f"  Adaptive edge check: K_min={y_min:.1f} vs edge={edge_threshold:.1f} (near_top={k_near_top})")
    print(f"                       K_max={y_max:.1f} vs edge={image_height - edge_threshold:.1f} (near_bottom={k_near_bottom})")
    
    if k_near_top or k_near_bottom:
        print(f"  ADAPTIVE: K-blocks at edge - wraparound needed")
        return 'wraparound', 0.95
    
    # =========================================================================
    # METHOD 2: HEURISTIC FALLBACK - Use configurable thresholds
    # =========================================================================
    
    # Full scan ratio check (0.84-1.05 range indicates full 360° scan)
    if full_scan_min <= circumference_ratio <= full_scan_max:
        print(f"  HEURISTIC: Circumference ratio {circumference_ratio:.2f} in full scan range "
              f"[{full_scan_min}, {full_scan_max}]")
        print(f"  → Full 360° scan likely - enabling wraparound for safety")
        return 'wraparound', 0.75
    
    # K-block rotation across rings
    if y_range_percent > rotation_percent:
        print(f"  HEURISTIC: K-block rotation {y_range_percent:.1f}% > {rotation_percent}% threshold")
        print(f"  → Significant K-block movement indicates wraparound")
        return 'wraparound', 0.70
    
    # Default: row pattern (K-blocks stable and not near edges)
    print(f"  DEFAULT: No wraparound indicators - row pattern")
    return 'row', 0.90
    

# =============================================================================
# Unified Segment Position Inference (handles ALL patterns)
# =============================================================================

def infer_all_segment_positions(
    k_positions: List[Tuple[str, Tuple[float, float]]],
    image_height: int,
    resolution: float = DEFAULT_RESOLUTION,
    segments_per_ring: int = 6,
    k_height_mm: float = DEFAULT_K_HEIGHT_MM,
    ab_height_mm: float = DEFAULT_AB_HEIGHT_MM,
    enable_wraparound: bool = True,
    pattern_params: Dict = None
) -> pd.DataFrame:
    """
    Infer ALL segment positions from K-block centers using domain knowledge.
    
    UNIFIED APPROACH: Works for row-based, wraparound, and mixed patterns.
    The key insight is that each ring is processed independently, and Y coordinates
    are wrapped modularly when they exceed image bounds.
    
    Args:
        k_positions: List of (type, (x, y)) K-block centers.
        image_height: Height of depth map in pixels.
        resolution: Image resolution in meters/pixel.
        segments_per_ring: Number of segments per ring (6 or 7).
        enable_wraparound: If True, wrap Y coordinates; if False, clip to bounds.
        
    Returns:
        DataFrame with Ring, Block, X, Y, inferred, pattern_type columns.
    """
    # Heights in pixels
    k_height_px = k_height_mm / (resolution * 1000)
    ab_height_px = ab_height_mm / (resolution * 1000)
    
    # Auto-detect pattern type (pattern_params passed from main pipeline)
    pattern_type, pattern_confidence = detect_pattern_type(
        k_positions, image_height, k_height_px, ab_height_px, segments_per_ring,
        pattern_params=pattern_params
    )
    print(f"  Pattern detected: {pattern_type} (confidence: {pattern_confidence:.2f})")
    
    # Segment order based on count
    if segments_per_ring == 7:
        segment_order = ['K', 'B1', 'A1', 'A2', 'A3', 'A4', 'B2']
    else:  # 6 segments
        segment_order = ['K', 'B1', 'A1', 'A2', 'A3', 'B2']
    
    def get_segment_offset(block: str) -> float:
        """
        Get Y offset from K-block center to segment center.
        
        Physical layout (Y increases downward in image):
            Going UP (negative Y offset from K):
                ... A3/A4 (top visible)
                A2
                A1  
                B1
            K (center, offset = 0)
            Going DOWN (positive Y offset from K):
                B2
                A3/A4, A2, A1 (bottom visible, wrapped)
        """
        if block == 'K':
            return 0
        elif block == 'B1':
            # B1 is above K: half K-height + half AB-height upward
            return -(k_height_px / 2 + ab_height_px / 2)
        elif block == 'B2':
            # B2 is below K: half K-height + half AB-height downward
            return (k_height_px / 2 + ab_height_px / 2)
        else:
            # A blocks are stacked above B1
            a_blocks = [b for b in segment_order if b.startswith('A')]
            block_idx = a_blocks.index(block)
            b1_offset = -(k_height_px / 2 + ab_height_px / 2)
            # Each A block is one AB-height above the previous
            return b1_offset - (block_idx + 1) * ab_height_px
    
    def normalize_y(y: float, height: int, wrap: bool) -> float:
        """
        Normalize Y coordinate - either wrap or clip.
        
        Args:
            y: Raw Y coordinate (can be negative or > height)
            height: Image height
            wrap: If True, wrap around; if False, clip to bounds
            
        Returns:
            Normalized Y coordinate in [0, height)
        """
        if wrap:
            # Modular arithmetic: Y wraps around image height
            y = y % height
            if y < 0:
                y += height
            return y
        else:
            # Clip to valid range
            return max(0, min(height - 1, y))
    
    segments = []
    
    for ring_idx, (k_type, (k_x, k_y)) in enumerate(k_positions):
        ring_id = ring_idx + 1
        
        for block in segment_order:
            offset = get_segment_offset(block)
            raw_y = k_y + offset
            
            # Normalize Y coordinate (wrap or clip based on pattern/setting)
            # For wraparound patterns, we always wrap
            # For row patterns, we can clip (segments should be in bounds anyway)
            should_wrap = enable_wraparound and (pattern_type in ('wraparound', 'mixed'))
            segment_y = normalize_y(raw_y, image_height, wrap=should_wrap)
            
            segments.append({
                'Ring': ring_id,
                'Block': block,
                'X': k_x,
                'Y': segment_y,
                'inferred': True,
                'pattern_type': pattern_type
            })
    
    df = pd.DataFrame(segments)
    
    # Report statistics
    print(f"  Total segments inferred: {len(df)}")
    if pattern_type == 'wraparound':
        # Count how many segments wrapped around
        wrapped_count = sum(1 for _, row in df.iterrows() 
                          if row['Block'] != 'K' and 
                          (row['Y'] < image_height * 0.2 or row['Y'] > image_height * 0.8))
        print(f"  Segments near edges (potential wraparound): {wrapped_count}")
    
    return df


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(
    edge_image: np.ndarray,
    positive_lines: List,
    negative_lines: List,
    horizontal_lines: List,
    vertical_lines: List,
    center_lines: List,
    output_path: str
) -> None:
    """Visualize all detected lines."""
    height, width = edge_image.shape
    vis = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    
    # Draw lines
    for x1, y1, x2, y2 in positive_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
    
    for x1, y1, x2, y2 in negative_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
    
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
    
    for x in vertical_lines:
        cv2.line(vis, (int(x), 0), (int(x), height), (0, 0, 255), 1)  # Red
    
    for x in center_lines:
        cv2.line(vis, (int(x), 0), (int(x), height), (255, 0, 255), 1)  # Magenta
    
    cv2.imwrite(output_path, vis)


# =============================================================================
# Main Pipeline
# =============================================================================

def detect_and_infer_patterns(
    tunnel_id: str,
    base_dir: str = "data",
    resolution: float = None,
    segments_per_ring: Optional[int] = None,
    pattern_mode: str = 'auto',
    enable_wraparound: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete detection and pattern inference pipeline.
    
    UNIFIED APPROACH: This pipeline handles ALL tunnel patterns:
    - Row-based: Segments stacked vertically with K-blocks aligned
    - Wraparound: Segments wrap around the cylindrical image
    - Mixed: Combination of the above
    - (Extensible for future patterns)
    
    Args:
        tunnel_id: Tunnel identifier.
        base_dir: Base data directory.
        resolution: Depth map resolution (loaded from params if None).
        segments_per_ring: Number of segments per ring (auto-detected if None).
        pattern_mode: 'auto' (detect), 'row', 'wraparound', or 'mixed'.
        enable_wraparound: Whether to enable wraparound Y coordinate handling.
        
    Returns:
        Tuple of (detected_df, inferred_df).
    """
    print("=" * 70)
    print("DETECTION & PATTERN INFERENCE (GT-Free)")
    print("=" * 70)
    print(f"Tunnel: {tunnel_id}")
    
    # Load parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    
    # Extract parameters - use defaults ONLY if no file was loaded
    allow_defaults = not params_loaded
    if resolution is None:
        resolution = get_param(params, 'physical_constants', 'resolution', default=DEFAULT_RESOLUTION, allow_default=allow_defaults)
    k_height_mm = get_param(params, 'physical_constants', 'k_height_mm', default=DEFAULT_K_HEIGHT_MM, allow_default=allow_defaults)
    ab_height_mm = get_param(params, 'physical_constants', 'ab_height_mm', default=DEFAULT_AB_HEIGHT_MM, allow_default=allow_defaults)
    
    # Preprocessing
    binary_threshold = get_param(params, 'preprocessing', 'binary_threshold', default=DEFAULT_BINARY_THRESHOLD, allow_default=allow_defaults)
    dilation_kernel_size = get_param(params, 'preprocessing', 'dilation_kernel_size', default=DEFAULT_DILATION_KERNEL_SIZE, allow_default=allow_defaults)
    dilation_iterations = get_param(params, 'preprocessing', 'dilation_iterations', default=DEFAULT_DILATION_ITERATIONS, allow_default=allow_defaults)
    
    # Hough oblique
    oblique_rho = get_param(params, 'hough_oblique', 'rho', default=DEFAULT_OBLIQUE_RHO, allow_default=allow_defaults)
    oblique_theta_deg = get_param(params, 'hough_oblique', 'theta_deg', default=DEFAULT_OBLIQUE_THETA_DEG, allow_default=allow_defaults)
    oblique_threshold = get_param(params, 'hough_oblique', 'threshold', default=DEFAULT_OBLIQUE_THRESHOLD, allow_default=allow_defaults)
    oblique_min_length = get_param(params, 'hough_oblique', 'min_length', default=DEFAULT_OBLIQUE_MIN_LENGTH, allow_default=allow_defaults)
    oblique_max_gap = get_param(params, 'hough_oblique', 'max_gap', default=DEFAULT_OBLIQUE_MAX_GAP, allow_default=allow_defaults)
    oblique_angle_pos_min = get_param(params, 'hough_oblique', 'angle_positive_min', default=DEFAULT_OBLIQUE_ANGLE_POSITIVE_MIN, allow_default=allow_defaults)
    oblique_angle_pos_max = get_param(params, 'hough_oblique', 'angle_positive_max', default=DEFAULT_OBLIQUE_ANGLE_POSITIVE_MAX, allow_default=allow_defaults)
    oblique_angle_neg_min = get_param(params, 'hough_oblique', 'angle_negative_min', default=DEFAULT_OBLIQUE_ANGLE_NEGATIVE_MIN, allow_default=allow_defaults)
    oblique_angle_neg_max = get_param(params, 'hough_oblique', 'angle_negative_max', default=DEFAULT_OBLIQUE_ANGLE_NEGATIVE_MAX, allow_default=allow_defaults)
    
    # Hough horizontal
    horizontal_threshold = get_param(params, 'hough_horizontal', 'threshold', default=DEFAULT_HORIZONTAL_THRESHOLD, allow_default=allow_defaults)
    horizontal_min_length = get_param(params, 'hough_horizontal', 'min_length', default=DEFAULT_HORIZONTAL_MIN_LENGTH, allow_default=allow_defaults)
    horizontal_max_gap = get_param(params, 'hough_horizontal', 'max_gap', default=DEFAULT_HORIZONTAL_MAX_GAP, allow_default=allow_defaults)
    horizontal_angle_tol = get_param(params, 'hough_horizontal', 'angle_tolerance', default=DEFAULT_HORIZONTAL_ANGLE_TOLERANCE, allow_default=allow_defaults)
    
    # Hough vertical
    vertical_threshold = get_param(params, 'hough_vertical', 'threshold', default=DEFAULT_VERTICAL_THRESHOLD, allow_default=allow_defaults)
    vertical_angle_tol = get_param(params, 'hough_vertical', 'angle_tolerance', default=DEFAULT_VERTICAL_ANGLE_TOLERANCE, allow_default=allow_defaults)
    
    # Line processing
    merge_dist_threshold = get_param(params, 'line_processing', 'merge_distance_threshold', default=DEFAULT_MERGE_DISTANCE_THRESHOLD, allow_default=allow_defaults)
    intersection_merge = get_param(params, 'line_processing', 'intersection_merge_threshold', default=DEFAULT_INTERSECTION_MERGE_THRESHOLD, allow_default=allow_defaults)
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load data
    depth_map_path = os.path.join(tunnel_dir, "depth_map_outlier.npy")
    if not os.path.exists(depth_map_path):
        # Try PNG version
        png_path = os.path.join(tunnel_dir, "depth_map.png")
        if os.path.exists(png_path):
            depth_map = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        else:
            raise FileNotFoundError(f"No depth map found in {tunnel_dir}")
    else:
        depth_map = np.load(depth_map_path)
    
    with open(os.path.join(tunnel_dir, "ring_count.txt"), 'r') as f:
        ring_count = int(f.read().strip())
    
    height, width = depth_map.shape
    
    # Auto-detect segment count if not provided
    if segments_per_ring is None:
        print("\nAuto-detecting segment count...")
        segments_per_ring = detect_segment_count(height, resolution)
    else:
        print(f"Using specified segment count: {segments_per_ring}")
    
    print(f"\nImage: {width} x {height}")
    print(f"Rings: {ring_count}")
    print(f"Segments per ring: {segments_per_ring}")
    print("=" * 70)
    
    # Preprocess
    print("Preprocessing depth map...")
    edge_image = preprocess_depth_map(
        depth_map, binary_threshold, dilation_kernel_size, dilation_iterations
    )
    
    # Detect lines
    print("Detecting lines...")
    positive_lines, negative_lines = detect_oblique_lines(
        edge_image, oblique_rho, np.pi / 180 * oblique_theta_deg,
        oblique_threshold, oblique_min_length, oblique_max_gap,
        oblique_angle_pos_min, oblique_angle_pos_max,
        oblique_angle_neg_min, oblique_angle_neg_max
    )
    print(f"  Oblique: {len(positive_lines)} positive, {len(negative_lines)} negative")
    
    horizontal_lines = detect_horizontal_lines(
        edge_image, horizontal_threshold, horizontal_min_length,
        horizontal_max_gap, horizontal_angle_tol
    )
    print(f"  Horizontal: {len(horizontal_lines)}")
    
    vertical_lines = detect_vertical_lines(
        edge_image, resolution, ring_count,
        vertical_threshold, vertical_angle_tol, merge_dist_threshold
    )
    print(f"  Vertical: {len(vertical_lines)}")
    
    # Generate center lines from detected vertical lines
    # The old method (OA 0.516) used detected vertical lines, NOT dense scanning
    # Dense scanning causes all K-blocks to have similar Y because oblique lines are similar
    print("Generating ring centers from detected vertical lines...")
    if vertical_lines:
        # Use MIDPOINTS between detected vertical lines (ring boundaries)
        vertical_lines = sorted(vertical_lines)
        center_lines = []
        for i in range(len(vertical_lines) - 1):
            mid_x = (vertical_lines[i] + vertical_lines[i + 1]) / 2
            center_lines.append(mid_x)
        # Add first and last ring centers
        if len(vertical_lines) >= 2:
            ring_width = vertical_lines[1] - vertical_lines[0]
            # First ring center (before first vertical line)
            first_center = vertical_lines[0] - ring_width / 2
            if first_center > 0:
                center_lines.insert(0, first_center)
            # Last ring center (after last vertical line)
            last_center = vertical_lines[-1] + ring_width / 2
            if last_center < width:
                center_lines.append(last_center)
    else:
        # Fallback to uniform distribution
        center_lines = generate_fallback_center_lines(width, ring_count)
    print(f"  Ring centers: {len(center_lines)} (from {len(vertical_lines)} detected vertical lines)")
    
    # Compute K-block prompt points using combined method
    # Scan all center lines to find K-blocks wherever they exist
    print("Computing K-block positions (dense scanning)...")
    print(f"  Scanning {len(center_lines)} positions across image width...")
    k_positions = compute_prompt_points(
        center_lines, positive_lines, negative_lines, horizontal_lines,
        resolution, height, intersection_merge, k_height_mm
    )
    print(f"  K-blocks found: {len(k_positions)}")
    
    # Create legacy detected.csv
    detected_df = pd.DataFrame(k_positions, columns=['Type', 'Coordinates'])
    detected_df['X'] = detected_df['Coordinates'].apply(lambda c: c[0])
    detected_df['Y'] = detected_df['Coordinates'].apply(lambda c: c[1])
    detected_df = detected_df.drop(columns=['Coordinates'])
    detected_df = detected_df.sort_values(by='X').reset_index(drop=True)
    
    # Load pattern detection parameters (configurable thresholds)
    pattern_params = get_param(params, 'pattern_detection', default={}, allow_default=True)
    
    # INFER ALL SEGMENT POSITIONS (unified approach)
    print("Inferring all segment positions...")
    print(f"  Pattern mode: {pattern_mode}")
    print(f"  Wraparound enabled: {enable_wraparound}")
    inferred_df = infer_all_segment_positions(
        k_positions, height, resolution, segments_per_ring,
        k_height_mm=k_height_mm, ab_height_mm=ab_height_mm,
        enable_wraparound=enable_wraparound,
        pattern_params=pattern_params
    )
    print(f"  Total segments: {len(inferred_df)}")
    
    # Save results
    os.makedirs(tunnel_dir, exist_ok=True)
    
    # Visualization
    visualize_detection(
        edge_image, positive_lines, negative_lines, horizontal_lines,
        vertical_lines, center_lines,
        os.path.join(tunnel_dir, "detected_lines.png")
    )
    
    # Save detected.csv (legacy K-block format)
    detected_df.to_csv(os.path.join(tunnel_dir, "detected.csv"), index=False)
    
    # Save inferred_from_pattern.csv (ALL segments for SAM)
    inferred_df.to_csv(os.path.join(tunnel_dir, "inferred_from_pattern.csv"), index=False)
    
    print("=" * 70)
    print(f"Saved to {tunnel_dir}/")
    print(f"  - detected.csv ({len(detected_df)} K-block centers)")
    print(f"  - inferred_from_pattern.csv ({len(inferred_df)} segment positions)")
    print(f"  - detected_lines.png")
    print("=" * 70)
    
    return detected_df, inferred_df


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Tunnel Segment Detection & Pattern Inference (GT-Free)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Patterns:
  - row:        K-blocks aligned horizontally (segments stacked vertically)
  - wraparound: K-blocks rotate across rings (segments wrap around image)
  - mixed:      Combination of above patterns
  - auto:       Auto-detect pattern from K-block distribution

Examples:
  python 4-1_detection.py 1-4                    # Auto-detect everything
  python 4-1_detection.py 4-1 --segments 7      # Force 7 segments
  python 4-1_detection.py 5-1 --pattern auto    # Explicit auto-detect
  python 4-1_detection.py 3-1 --no-wraparound   # Disable wraparound handling
"""
    )
    
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4, 4-1, 5-1)")
    parser.add_argument("--segments", "-s", type=int, default=None,
                        help="Number of segments per ring (auto-detect if omitted)")
    parser.add_argument("--pattern", "-p", choices=['auto', 'row', 'wraparound', 'mixed'],
                        default='auto', help="Pattern detection mode (default: auto)")
    parser.add_argument("--no-wraparound", action="store_true",
                        help="Disable wraparound Y coordinate handling")
    parser.add_argument("--data-dir", "-d", default="data",
                        help="Base data directory (default: data)")
    
    args = parser.parse_args()
    
    detect_and_infer_patterns(
        tunnel_id=args.tunnel_id,
        base_dir=args.data_dir,
        segments_per_ring=args.segments,
        pattern_mode=args.pattern,
        enable_wraparound=not args.no_wraparound
    )

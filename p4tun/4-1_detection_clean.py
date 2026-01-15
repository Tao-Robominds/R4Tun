"""
Prompt Point Detection and Segment Pattern Inference (GT-Free)

This module detects tunnel ring boundaries and infers ALL segment positions
using Hough line detection on depth maps combined with domain knowledge.

NO GROUND TRUTH REQUIRED.

Pipeline:
    1. Preprocess depth map to binary edge image
    2. Detect oblique lines (joint edges at ±7.5° angle)
    3. Detect horizontal lines (K-block boundaries)
    4. Detect vertical lines (ring center separations)
    5. Compute K-block center points at line intersections
    6. INFER ALL SEGMENT POSITIONS using domain knowledge

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

def load_parameters(tunnel_id: str = None, base_dir: str = "data") -> Dict[str, Any]:
    """
    Load parameters from JSON file with fallback to defaults.
    
    Priority:
        1. Centralized: sam4tun/parameters/<tunnel_id>/parameters_detection.json
        2. Tunnel-specific: data/<tunnel_id>/parameters_detection.json
        3. Default: sam4tun/parameters_detection.json (if exists)
        4. Hardcoded defaults (if no file found)
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_detection.json"
    
    if tunnel_id:
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
    
    default_path = os.path.join(script_dir, param_file)
    if os.path.exists(default_path):
        print(f"Loading default parameters from {default_path}")
        with open(default_path, 'r') as f:
            return json.load(f)
    
    print("Warning: No parameter file found, using hardcoded defaults")
    return {}


def get_param(params: Dict, *keys, default=None):
    """Get nested parameter value with default fallback."""
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
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
DEFAULT_VERTICAL_FILTER_RINGS = 5

# --- Line Processing ---
DEFAULT_MERGE_DISTANCE_THRESHOLD = 3
DEFAULT_INTERSECTION_MERGE_THRESHOLD = 6
DEFAULT_PATTERN_TOLERANCE = 10
DEFAULT_HORIZONTAL_PATTERN_TOLERANCE = 50


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
    filter_rings: int = DEFAULT_VERTICAL_FILTER_RINGS
) -> List:
    """Detect vertical lines (ring boundaries)."""
    lines = cv2.HoughLines(edge_image, 1, np.pi / 180, threshold)
    
    vertical_lines = []
    if lines is not None:
        height, width = edge_image.shape
        ring_width = width / ring_count
        
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            
            if abs(angle) <= angle_tolerance:
                x = int(rho / np.cos(theta - np.pi / 2)) if abs(np.cos(theta - np.pi / 2)) > 0.01 else int(rho)
                
                # Filter to keep only lines near ring boundaries
                ring_position = x / ring_width
                if ring_position <= filter_rings or ring_position >= (ring_count - filter_rings):
                    vertical_lines.append(x)
    
    # Remove duplicates
    vertical_lines = sorted(set(vertical_lines))
    return vertical_lines


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
# K-Block Prompt Point Detection
# =============================================================================

def compute_prompt_points(
    center_lines: List,
    positive_lines: List,
    negative_lines: List,
    horizontal_lines: List,
    resolution: float,
    height: int,
    intersection_merge_threshold: float = DEFAULT_INTERSECTION_MERGE_THRESHOLD
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    Compute K-block prompt points from line intersections.
    
    Returns list of (type, (x, y)) tuples.
    """
    prompt_points = []
    
    for cx in center_lines:
        # Vertical line through center
        center_line = (cx, 0, cx, height)
        
        # Find intersections with oblique lines
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
        
        # Find horizontal intersections
        horizontal_intersections = []
        for line in horizontal_lines:
            pt = find_intersection(center_line, line)
            if pt and 0 <= pt[1] <= height:
                horizontal_intersections.append(pt)
        
        # Merge close points
        positive_intersections = merge_close_points(positive_intersections, intersection_merge_threshold)
        negative_intersections = merge_close_points(negative_intersections, intersection_merge_threshold)
        horizontal_intersections = merge_close_points(horizontal_intersections, intersection_merge_threshold)
        
        # Determine K-block center
        k_center = None
        
        # Method 1: Midpoint between positive and negative oblique lines
        if positive_intersections and negative_intersections:
            pos_y = positive_intersections[0][1]
            neg_y = negative_intersections[0][1]
            mid_y = (pos_y + neg_y) / 2
            k_center = (cx, mid_y)
            prompt_points.append(('midpoint', k_center))
        
        # Method 2: From horizontal lines
        elif horizontal_intersections:
            # Take the middle horizontal line
            h_ys = sorted([h[1] for h in horizontal_intersections])
            mid_idx = len(h_ys) // 2
            k_center = (cx, h_ys[mid_idx])
            prompt_points.append(('horizontal', k_center))
        
        # Method 3: Single oblique line
        elif positive_intersections:
            k_center = (cx, positive_intersections[0][1])
            prompt_points.append(('positive_slope', k_center))
        elif negative_intersections:
            k_center = (cx, negative_intersections[0][1])
            prompt_points.append(('negative_slope', k_center))
        
        # Method 4: Fallback to center of image
        else:
            k_center = (cx, height / 2)
            prompt_points.append(('assume', k_center))
    
    return prompt_points


# =============================================================================
# Segment Position Inference (NEW - replaces row-based)
# =============================================================================

def infer_all_segment_positions(
    k_positions: List[Tuple[str, Tuple[float, float]]],
    image_height: int,
    resolution: float = DEFAULT_RESOLUTION,
    segments_per_ring: int = 6,
    k_height_mm: float = DEFAULT_K_HEIGHT_MM,
    ab_height_mm: float = DEFAULT_AB_HEIGHT_MM
) -> pd.DataFrame:
    """
    Infer ALL segment positions from K-block centers using domain knowledge.
    
    Automatically detects whether wrap-around is needed based on image coverage:
    - Coverage >= 90%: Wrap-around (segments cross image boundary)
    - Coverage < 90%: No wrap (skip segments outside image)
    
    Args:
        k_positions: List of (type, (x, y)) K-block centers.
        image_height: Height of depth map in pixels.
        resolution: Image resolution in meters/pixel.
        segments_per_ring: Number of segments per ring (6 or 7).
        
    Returns:
        DataFrame with Ring, Block, X, Y, inferred columns.
    """
    # Segment order based on count
    if segments_per_ring == 7:
        segment_order = ['K', 'B1', 'A1', 'A2', 'A3', 'A4', 'B2']
    else:  # 6 segments
        segment_order = ['K', 'B1', 'A1', 'A2', 'A3', 'B2']
    
    # Heights in pixels
    k_height_px = k_height_mm / (resolution * 1000)
    ab_height_px = ab_height_mm / (resolution * 1000)
    
    # Calculate theoretical circumference and coverage
    circumference_px = k_height_px + (segments_per_ring - 1) * ab_height_px
    coverage = image_height / circumference_px
    
    # Determine if wrap-around is needed
    # Coverage >= 90% means image covers most/all of circumference
    wrap_around = coverage >= 0.90
    
    print(f"  Image height: {image_height} px")
    print(f"  Circumference: {circumference_px:.0f} px")
    print(f"  Coverage: {coverage * 100:.1f}%")
    print(f"  Wrap-around: {'YES' if wrap_around else 'NO'}")
    
    def get_segment_offset_up(block: str, segment_order: List[str]) -> float:
        """Get Y offset going UP from K-block center.
        
        Physical layout (top to bottom of image):
            A3 (or A4 for 7-segment)  ← going UP
            A2
            A1
            B1
            K (center)
            B2
            A3 (wrap-around)  ← going DOWN
            A2
            A1
        """
        if block == 'K':
            return 0
        elif block == 'B1':
            return -(k_height_px / 2 + ab_height_px / 2)
        elif block == 'B2':
            return (k_height_px / 2 + ab_height_px / 2)
        else:
            # A blocks going UP from B1
            a_blocks = [b for b in segment_order if b.startswith('A')]
            block_idx = a_blocks.index(block)
            b1_offset = -(k_height_px / 2 + ab_height_px / 2)
            return b1_offset - (block_idx + 1) * ab_height_px
    
    def get_segment_offset_down(block: str, segment_order: List[str]) -> float:
        """Get Y offset going DOWN from B2 (for wrap-around handling)."""
        if block in ('K', 'B1', 'B2'):
            return None  # Only A blocks go down from B2
        
        # A blocks going DOWN from B2 (in reverse order: A3, A2, A1)
        a_blocks = [b for b in segment_order if b.startswith('A')]
        # Reverse order: A3 is first below B2, then A2, then A1
        block_idx = len(a_blocks) - 1 - a_blocks.index(block)
        b2_offset = (k_height_px / 2 + ab_height_px / 2)
        return b2_offset + (block_idx + 1) * ab_height_px
    
    segments = []
    skipped = 0
    margin = ab_height_px / 2
    
    for ring_idx, (k_type, (k_x, k_y)) in enumerate(k_positions):
        ring_id = ring_idx + 1
        added_blocks = set()
        
        # Pass 1: Calculate positions going UP from K
        for block in segment_order:
            offset = get_segment_offset_up(block, segment_order)
            segment_y = k_y + offset
            
            if wrap_around:
                segment_y = segment_y % image_height
                segments.append({
                    'Ring': ring_id, 'Block': block, 'X': k_x,
                    'Y': segment_y, 'inferred': True
                })
                added_blocks.add(block)
            else:
                if -margin <= segment_y <= image_height + margin:
                    segment_y = max(0, min(image_height - 1, segment_y))
                    segments.append({
                        'Ring': ring_id, 'Block': block, 'X': k_x,
                        'Y': segment_y, 'inferred': True
                    })
                    added_blocks.add(block)
                else:
                    skipped += 1
        
        # Pass 2: For non-wrap-around, also calculate A blocks going DOWN from B2
        # This handles the "reverse walk" case where A blocks appear at the bottom
        if not wrap_around:
            a_blocks = [b for b in segment_order if b.startswith('A')]
            for block in reversed(a_blocks):  # A3, A2, A1 order
                if block in added_blocks:
                    continue  # Already added from going UP
                
                offset = get_segment_offset_down(block, segment_order)
                if offset is None:
                    continue
                segment_y = k_y + offset
                
                if -margin <= segment_y <= image_height + margin:
                    segment_y = max(0, min(image_height - 1, segment_y))
                    segments.append({
                        'Ring': ring_id, 'Block': block, 'X': k_x,
                        'Y': segment_y, 'inferred': True
                    })
                    added_blocks.add(block)
                else:
                    skipped += 1
    
    if skipped > 0:
        print(f"  Skipped {skipped} out-of-bounds segments")
    
    return pd.DataFrame(segments)


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
    segments_per_ring: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete detection and pattern inference pipeline.
    
    Args:
        tunnel_id: Tunnel identifier.
        base_dir: Base data directory.
        resolution: Depth map resolution (loaded from params if None).
        segments_per_ring: Number of segments per ring (auto-detected if None).
        
    Returns:
        Tuple of (detected_df, inferred_df).
    """
    print("=" * 70)
    print("DETECTION & PATTERN INFERENCE (GT-Free)")
    print("=" * 70)
    print(f"Tunnel: {tunnel_id}")
    
    # Load parameters
    params = load_parameters(tunnel_id, base_dir)
    
    # Extract parameters with defaults
    if resolution is None:
        resolution = get_param(params, 'physical_constants', 'resolution', default=DEFAULT_RESOLUTION)
    k_height_mm = get_param(params, 'physical_constants', 'k_height_mm', default=DEFAULT_K_HEIGHT_MM)
    ab_height_mm = get_param(params, 'physical_constants', 'ab_height_mm', default=DEFAULT_AB_HEIGHT_MM)
    
    # Preprocessing
    binary_threshold = get_param(params, 'preprocessing', 'binary_threshold', default=DEFAULT_BINARY_THRESHOLD)
    dilation_kernel_size = get_param(params, 'preprocessing', 'dilation_kernel_size', default=DEFAULT_DILATION_KERNEL_SIZE)
    dilation_iterations = get_param(params, 'preprocessing', 'dilation_iterations', default=DEFAULT_DILATION_ITERATIONS)
    
    # Hough oblique
    oblique_rho = get_param(params, 'hough_oblique', 'rho', default=DEFAULT_OBLIQUE_RHO)
    oblique_theta_deg = get_param(params, 'hough_oblique', 'theta_deg', default=DEFAULT_OBLIQUE_THETA_DEG)
    oblique_threshold = get_param(params, 'hough_oblique', 'threshold', default=DEFAULT_OBLIQUE_THRESHOLD)
    oblique_min_length = get_param(params, 'hough_oblique', 'min_length', default=DEFAULT_OBLIQUE_MIN_LENGTH)
    oblique_max_gap = get_param(params, 'hough_oblique', 'max_gap', default=DEFAULT_OBLIQUE_MAX_GAP)
    oblique_angle_pos_min = get_param(params, 'hough_oblique', 'angle_positive_min', default=DEFAULT_OBLIQUE_ANGLE_POSITIVE_MIN)
    oblique_angle_pos_max = get_param(params, 'hough_oblique', 'angle_positive_max', default=DEFAULT_OBLIQUE_ANGLE_POSITIVE_MAX)
    oblique_angle_neg_min = get_param(params, 'hough_oblique', 'angle_negative_min', default=DEFAULT_OBLIQUE_ANGLE_NEGATIVE_MIN)
    oblique_angle_neg_max = get_param(params, 'hough_oblique', 'angle_negative_max', default=DEFAULT_OBLIQUE_ANGLE_NEGATIVE_MAX)
    
    # Hough horizontal
    horizontal_threshold = get_param(params, 'hough_horizontal', 'threshold', default=DEFAULT_HORIZONTAL_THRESHOLD)
    horizontal_min_length = get_param(params, 'hough_horizontal', 'min_length', default=DEFAULT_HORIZONTAL_MIN_LENGTH)
    horizontal_max_gap = get_param(params, 'hough_horizontal', 'max_gap', default=DEFAULT_HORIZONTAL_MAX_GAP)
    horizontal_angle_tol = get_param(params, 'hough_horizontal', 'angle_tolerance', default=DEFAULT_HORIZONTAL_ANGLE_TOLERANCE)
    
    # Hough vertical
    vertical_threshold = get_param(params, 'hough_vertical', 'threshold', default=DEFAULT_VERTICAL_THRESHOLD)
    vertical_angle_tol = get_param(params, 'hough_vertical', 'angle_tolerance', default=DEFAULT_VERTICAL_ANGLE_TOLERANCE)
    vertical_filter_rings = get_param(params, 'hough_vertical', 'filter_rings', default=DEFAULT_VERTICAL_FILTER_RINGS)
    
    # Line processing
    merge_dist_threshold = get_param(params, 'line_processing', 'merge_distance_threshold', default=DEFAULT_MERGE_DISTANCE_THRESHOLD)
    intersection_merge = get_param(params, 'line_processing', 'intersection_merge_threshold', default=DEFAULT_INTERSECTION_MERGE_THRESHOLD)
    
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
        vertical_threshold, vertical_angle_tol, vertical_filter_rings
    )
    print(f"  Vertical: {len(vertical_lines)}")
    
    # Generate center lines
    print("Generating ring centers...")
    if vertical_lines:
        center_lines = generate_center_lines(vertical_lines, width, height, ring_count)
    else:
        center_lines = generate_fallback_center_lines(width, ring_count)
    print(f"  Centers: {len(center_lines)}")
    
    # Compute K-block prompt points
    print("Computing K-block positions...")
    k_positions = compute_prompt_points(
        center_lines, positive_lines, negative_lines, horizontal_lines,
        resolution, height, intersection_merge
    )
    print(f"  K-blocks: {len(k_positions)}")
    
    # Create legacy detected.csv
    detected_df = pd.DataFrame(k_positions, columns=['Type', 'Coordinates'])
    detected_df['X'] = detected_df['Coordinates'].apply(lambda c: c[0])
    detected_df['Y'] = detected_df['Coordinates'].apply(lambda c: c[1])
    detected_df = detected_df.drop(columns=['Coordinates'])
    detected_df = detected_df.sort_values(by='X').reset_index(drop=True)
    
    # INFER ALL SEGMENT POSITIONS
    print("Inferring all segment positions...")
    inferred_df = infer_all_segment_positions(
        k_positions, height, resolution, segments_per_ring,
        k_height_mm=k_height_mm, ab_height_mm=ab_height_mm
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
    if len(sys.argv) < 2:
        print("Usage: python 4-1_detection_clean.py <tunnel_id> [segments_per_ring]")
        print()
        print("Arguments:")
        print("  tunnel_id         Tunnel identifier (e.g., 1-4, 4-1)")
        print("  segments_per_ring Number of segments (auto-detected if omitted)")
        print()
        print("Examples:")
        print("  python 4-1_detection_clean.py 1-4      # Auto-detect segments")
        print("  python 4-1_detection_clean.py 4-1 7    # Force 7 segments")
        sys.exit(1)
    
    tunnel_id = sys.argv[1]
    segments_per_ring = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    detect_and_infer_patterns(tunnel_id, segments_per_ring=segments_per_ring)

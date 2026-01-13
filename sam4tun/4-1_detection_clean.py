"""
Prompt Point Generation via Line Detection

This module detects tunnel ring boundaries and generates initial prompt points
for subsequent segmentation using Hough line detection on depth maps.

Algorithm Overview:
    1. Preprocess depth map to binary edge image
    2. Detect oblique lines (joint edges at ±7.5° angle)
    3. Detect horizontal lines (ring boundaries)
    4. Detect vertical lines (ring center separations)
    5. Compute prompt points at line intersections
    6. Fill gaps using geometric constraints
"""

import os
import sys
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Constants
# =============================================================================

# --- Physical Constants (fixed by tunnel design, don't change) ---
RING_SPACING = 1.2  # Nominal ring spacing (meters)
K_HEIGHT = 1079.92 / 1000  # K-block height (meters)
AB_HEIGHT = 3239.77 / 1000  # A+B block height (meters)

# --- Quality Parameters (tune these for better results) ---

# Image preprocessing
BINARY_THRESHOLD = 127  # Threshold for converting depth map to binary
                         # Lower = more sensitive to faint edges
DILATION_KERNEL_SIZE = 3  # Morphological dilation kernel size
                           # Larger = thicker edges, may merge close lines
DILATION_ITERATIONS = 1  # Number of dilation iterations
                          # More = thicker edges

# Hough line detection - Oblique lines (joint edges)
OBLIQUE_RHO = 1  # Distance resolution (pixels)
OBLIQUE_THETA = np.pi / 180  # Angle resolution (radians)
OBLIQUE_THRESHOLD = 50  # Accumulator threshold
                         # Lower = more line detections, more false positives
OBLIQUE_MIN_LENGTH = 100  # Minimum line length (pixels)
                           # Shorter = detect smaller segments
OBLIQUE_MAX_GAP = 40  # Maximum gap between line segments (pixels)
                       # Larger = connect more segments

# Oblique line angle ranges (degrees)
OBLIQUE_ANGLE_POSITIVE_MIN = 6  # Minimum positive angle
OBLIQUE_ANGLE_POSITIVE_MAX = 9  # Maximum positive angle
OBLIQUE_ANGLE_NEGATIVE_MIN = -9  # Minimum negative angle
OBLIQUE_ANGLE_NEGATIVE_MAX = -6  # Maximum negative angle

# Hough line detection - Horizontal lines
HORIZONTAL_THRESHOLD = 50  # Accumulator threshold
HORIZONTAL_MIN_LENGTH = 100  # Minimum line length (pixels)
HORIZONTAL_MAX_GAP = 10  # Maximum gap (smaller for horizontal)
HORIZONTAL_ANGLE_TOLERANCE = 1  # Angle tolerance for horizontal (degrees)

# Hough line detection - Vertical lines
VERTICAL_THRESHOLD = 500  # Accumulator threshold for HoughLines
                           # Higher = only strong vertical lines
VERTICAL_ANGLE_TOLERANCE = 0.5  # Angle tolerance (degrees)
VERTICAL_FILTER_RINGS = 5  # Only keep lines within this many rings from edge

# Line merging
MERGE_DISTANCE_THRESHOLD = 3  # Distance threshold for merging close lines (pixels)

# Intersection detection
INTERSECTION_MERGE_THRESHOLD = 6  # Threshold for merging close intersection points (pixels)
PATTERN_TOLERANCE = 10  # Tolerance for distance pattern matching (pixels)
HORIZONTAL_PATTERN_TOLERANCE = 50  # Tolerance for horizontal pattern (pixels)

# Fallback estimation
Y_ESTIMATE_TOLERANCE = 0.1  # Tolerance for Y-coordinate estimation (10%)
Y_OFFSET_BLOCKS = 431.87  # Offset between block types in pixels (at 0.005 resolution)

# --- Performance Parameters ---
# (None for this module - single-threaded OpenCV operations)


# =============================================================================
# Helper Functions
# =============================================================================

def compute_line_angle(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Compute the angle of a line segment in degrees.
    
    Angles are measured from horizontal, with positive angles
    going counter-clockwise (standard math convention with inverted Y).
    
    Args:
        x1, y1: Start point coordinates.
        x2, y2: End point coordinates.
        
    Returns:
        Angle in degrees (-180 to 180).
    """
    # Ensure consistent direction (left to right)
    if x1 > x2:
        x1, x2, y1, y2 = x2, x1, y2, y1
    
    # Invert Y to match standard angle convention
    return np.degrees(np.arctan2(-(y2 - y1), x2 - x1))


def line_vertical_intersection(
    vertical_x: float,
    segment: np.ndarray
) -> Optional[Tuple[float, float]]:
    """
    Compute intersection of a vertical line with a line segment.
    
    Args:
        vertical_x: X-coordinate of vertical line.
        segment: Line segment as [x1, y1, x2, y2].
        
    Returns:
        Intersection point (x, y) or None if no intersection.
    """
    x1, y1, x2, y2 = segment
    
    # Skip vertical segments
    if x1 == x2:
        return None
    
    # Check if vertical line crosses segment
    if min(x1, x2) <= vertical_x <= max(x1, x2):
        t = (vertical_x - x1) / (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (vertical_x, y)
    
    return None


def merge_close_points(
    points: List[Tuple[float, float]],
    threshold: float = INTERSECTION_MERGE_THRESHOLD
) -> np.ndarray:
    """
    Merge points that are within a threshold distance.
    
    Args:
        points: List of (x, y) points.
        threshold: Maximum distance for merging.
        
    Returns:
        Array of merged points.
    """
    if len(points) == 0:
        return np.array([])
    if len(points) == 1:
        return np.array(points)
    
    points = np.array(points)
    merged = []
    
    while len(points) > 0:
        p = points[0]
        distances = np.linalg.norm(points - p, axis=1)
        close_mask = distances < threshold
        merged.append(np.mean(points[close_mask], axis=0))
        points = points[~close_mask]
    
    return np.array(merged)


def compute_midpoint(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> Tuple[float, float]:
    """Compute midpoint of two points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def check_distance_pattern(
    points: List[Tuple[float, float]],
    k_height: float,
    ab_height: float,
    tolerance: float = PATTERN_TOLERANCE
) -> Optional[Tuple[float, float]]:
    """
    Check if point distances match expected block pattern.
    
    Looks for patterns where distance = k + n*ab for n in [2, 4].
    
    Args:
        points: List of intersection points.
        k_height: K-block height in pixels.
        ab_height: A+B block height in pixels.
        tolerance: Distance tolerance.
        
    Returns:
        Midpoint if pattern found, None otherwise.
    """
    if len(points) < 2:
        return None
    
    points = sorted(points, key=lambda p: p[0])
    
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            
            # Check for pattern match
            for n in [2, 4]:
                expected = k_height + n * ab_height
                if abs(distance - expected) < tolerance:
                    return compute_midpoint(points[i], points[j])
    
    return None


# =============================================================================
# Line Detection
# =============================================================================

def preprocess_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Convert depth map to binary edge image.
    
    Args:
        depth_map: Input depth map with NaN for empty pixels.
        
    Returns:
        Dilated binary edge image.
    """
    # Convert NaN to 0, valid to 255
    binary = np.where(np.isnan(depth_map), 0, 255).astype(np.uint8)
    
    # Threshold (for consistency with original)
    _, binary = cv2.threshold(binary, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Dilate to strengthen edges
    kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=DILATION_ITERATIONS)
    
    return dilated


def detect_oblique_lines(
    edge_image: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Detect oblique lines at approximately ±7.5 degrees.
    
    Args:
        edge_image: Binary edge image.
        
    Returns:
        Tuple of (positive_angle_lines, negative_angle_lines).
    """
    lines = cv2.HoughLinesP(
        edge_image,
        rho=OBLIQUE_RHO,
        theta=OBLIQUE_THETA,
        threshold=OBLIQUE_THRESHOLD,
        minLineLength=OBLIQUE_MIN_LENGTH,
        maxLineGap=OBLIQUE_MAX_GAP
    )
    
    positive_lines = []
    negative_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = compute_line_angle(x1, y1, x2, y2)
            
            if OBLIQUE_ANGLE_POSITIVE_MIN <= angle <= OBLIQUE_ANGLE_POSITIVE_MAX:
                positive_lines.append(line)
            elif OBLIQUE_ANGLE_NEGATIVE_MIN <= angle <= OBLIQUE_ANGLE_NEGATIVE_MAX:
                negative_lines.append(line)
    
    return positive_lines, negative_lines


def detect_horizontal_lines(edge_image: np.ndarray) -> List[np.ndarray]:
    """
    Detect horizontal lines (ring boundaries).
    
    Args:
        edge_image: Binary edge image.
        
    Returns:
        List of horizontal line segments.
    """
    lines = cv2.HoughLinesP(
        edge_image,
        rho=OBLIQUE_RHO,
        theta=OBLIQUE_THETA,
        threshold=HORIZONTAL_THRESHOLD,
        minLineLength=HORIZONTAL_MIN_LENGTH,
        maxLineGap=HORIZONTAL_MAX_GAP
    )
    
    horizontal_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = compute_line_angle(x1, y1, x2, y2)
            
            if -HORIZONTAL_ANGLE_TOLERANCE <= angle <= HORIZONTAL_ANGLE_TOLERANCE:
                horizontal_lines.append(line)
    
    return horizontal_lines


def detect_vertical_lines(
    edge_image: np.ndarray,
    resolution: float,
    ring_count: int
) -> List[Tuple[float, float]]:
    """
    Detect and merge vertical lines (ring separations).
    
    Args:
        edge_image: Binary edge image.
        resolution: Depth map resolution (meters/pixel).
        ring_count: Number of rings in tunnel.
        
    Returns:
        List of vertical lines as (rho, theta) tuples.
    """
    lines = cv2.HoughLines(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=VERTICAL_THRESHOLD
    )
    
    if lines is None:
        return []
    
    # Filter to near-vertical lines within expected range
    max_rho = VERTICAL_FILTER_RINGS * RING_SPACING / resolution
    filtered = lines[lines[:, 0, 0] <= max_rho]
    
    if len(filtered) == 0:
        return []
    
    # Merge close lines
    merged = []
    tolerance = VERTICAL_ANGLE_TOLERANCE * np.pi / 180
    
    for rho, theta in filtered[:, 0]:
        if abs(theta) > tolerance:
            continue
        
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        
        # Check if close to existing line
        is_merged = False
        for i, (m_rho, m_theta) in enumerate(merged):
            m_x = m_rho * np.cos(m_theta)
            m_y = m_rho * np.sin(m_theta)
            
            if np.sqrt((x - m_x)**2 + (y - m_y)**2) < MERGE_DISTANCE_THRESHOLD:
                # Average the lines
                merged[i] = ((rho + m_rho) / 2, (theta + m_theta) / 2)
                is_merged = True
                break
        
        if not is_merged:
            merged.append((rho, theta))
    
    return sorted(merged, key=lambda l: l[0])


def generate_center_lines(
    vertical_lines: List[Tuple[float, float]],
    image_width: int,
    image_height: int,
    ring_count: int
) -> List[Tuple[float, float]]:
    """
    Generate center lines between adjacent vertical lines.
    
    Extends lines to cover the full image width using average spacing.
    
    Args:
        vertical_lines: Detected vertical lines.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        ring_count: Number of rings.
        
    Returns:
        List of center lines as (x_position, theta) tuples.
    """
    if len(vertical_lines) < 2:
        return []
    
    # Compute midpoints between adjacent lines
    center_lines = []
    for i in range(len(vertical_lines) - 1):
        rho1, theta1 = vertical_lines[i]
        rho2, theta2 = vertical_lines[i + 1]
        center_lines.append(((rho1 + rho2) / 2, (theta1 + theta2) / 2))
    
    # Compute average spacing
    if len(center_lines) > 1:
        spacings = []
        for i in range(len(center_lines) - 1):
            x1 = center_lines[i][0] * np.cos(center_lines[i][1])
            x2 = center_lines[i + 1][0] * np.cos(center_lines[i + 1][1])
            spacings.append(abs(x2 - x1))
        avg_spacing_detected = np.mean(spacings)
    else:
        avg_spacing_detected = image_width / ring_count
    
    # Choose better spacing estimate
    designed_spacing = image_width / ring_count
    expected_spacing = RING_SPACING / 0.005  # At 0.005 resolution
    
    if abs(avg_spacing_detected - expected_spacing) <= abs(designed_spacing - expected_spacing):
        avg_spacing = avg_spacing_detected
    else:
        avg_spacing = designed_spacing
    
    # Extend to cover full width
    all_lines = center_lines.copy()
    
    if center_lines:
        # Extend left
        leftmost_x = center_lines[0][0] * np.cos(center_lines[0][1])
        theta = center_lines[0][1]
        x = leftmost_x - avg_spacing
        while x >= 0:
            all_lines.append((x, theta))
            x -= avg_spacing
        
        # Extend right
        rightmost_x = center_lines[-1][0] * np.cos(center_lines[-1][1])
        theta = center_lines[-1][1]
        x = rightmost_x + avg_spacing
        while x <= image_width:
            all_lines.append((x, theta))
            x += avg_spacing
    
    return sorted(list(set(all_lines)), key=lambda l: l[0])


def generate_fallback_center_lines(
    image_width: int,
    ring_count: int
) -> List[Tuple[float, float]]:
    """
    Generate evenly spaced center lines when detection fails.
    
    Args:
        image_width: Image width in pixels.
        ring_count: Number of rings.
        
    Returns:
        List of center lines at ring centers.
    """
    block_width = image_width / ring_count
    lines = []
    
    for i in range(ring_count):
        x_pos = (i + 0.5) * block_width
        lines.append((x_pos, 0))
    
    return lines


# =============================================================================
# Prompt Point Generation
# =============================================================================

def compute_prompt_points(
    center_lines: List[Tuple[float, float]],
    positive_lines: List[np.ndarray],
    negative_lines: List[np.ndarray],
    horizontal_lines: List[np.ndarray],
    resolution: float,
    image_height: int
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    Compute prompt points at line intersections.
    
    Priority:
        1. Midpoint of positive and negative oblique intersections
        2. Adjusted positive oblique intersection
        3. Adjusted negative oblique intersection
        4. Horizontal line pattern matching
        5. Estimation from previous points
    
    Args:
        center_lines: Vertical center lines.
        positive_lines: Positive angle oblique lines.
        negative_lines: Negative angle oblique lines.
        horizontal_lines: Horizontal lines.
        resolution: Depth map resolution.
        image_height: Image height in pixels.
        
    Returns:
        List of (type, (x, y)) prompt points.
    """
    # Convert physical dimensions to pixels
    k_height_px = K_HEIGHT / resolution
    ab_height_px = AB_HEIGHT / resolution
    
    prompt_points = []
    
    for vertical_x, _ in center_lines:
        positive_intersections = []
        negative_intersections = []
        
        # Find intersections with oblique lines
        for line in positive_lines:
            point = line_vertical_intersection(vertical_x, line[0])
            if point:
                positive_intersections.append(point)
        
        for line in negative_lines:
            point = line_vertical_intersection(vertical_x, line[0])
            if point:
                negative_intersections.append(point)
        
        # Merge close intersections
        merged_positive = merge_close_points(positive_intersections)
        merged_negative = merge_close_points(negative_intersections)
        
        # Determine prompt point
        if len(merged_positive) > 0 and len(merged_negative) > 0:
            # Case 1: Both types - use midpoint
            midpoint = compute_midpoint(merged_positive[0], merged_negative[0])
            prompt_points.append(('midpoint', midpoint))
        
        elif len(merged_positive) > 0:
            # Case 2: Only positive - adjust by half K height
            point = merged_positive[0]
            prompt_points.append(('positive_slope', (point[0], point[1] - 0.5 * k_height_px)))
        
        elif len(merged_negative) > 0:
            # Case 3: Only negative - adjust by half K height
            point = merged_negative[0]
            prompt_points.append(('negative_slope', (point[0], point[1] + 0.5 * k_height_px)))
        
        else:
            # Case 4: Check horizontal lines
            horizontal_intersections = []
            for line in horizontal_lines:
                point = line_vertical_intersection(vertical_x, line[0])
                if point:
                    horizontal_intersections.append(point)
            
            merged_horizontal = merge_close_points(horizontal_intersections)
            
            pattern_point = check_distance_pattern(
                list(map(tuple, merged_horizontal)),
                k_height_px, ab_height_px,
                tolerance=HORIZONTAL_PATTERN_TOLERANCE
            )
            
            if pattern_point:
                prompt_points.append(('horizontal', pattern_point))
            else:
                # Case 5: Estimate from previous points
                estimated_y = estimate_y_coordinate(prompt_points, image_height)
                
                if estimated_y is not None:
                    prompt_points.append(('assume', (vertical_x, estimated_y)))
                else:
                    # Fallback to image center
                    prompt_points.append(('default', (vertical_x, image_height / 2)))
                    print(f"Warning: Using default Y for x={vertical_x}")
    
    return prompt_points


def estimate_y_coordinate(
    previous_points: List[Tuple[str, Tuple[float, float]]],
    image_height: int
) -> Optional[float]:
    """
    Estimate Y coordinate based on previous prompt points.
    
    Uses the alternating pattern of block types to predict Y.
    
    Args:
        previous_points: List of previously determined points.
        image_height: Image height for range calculation.
        
    Returns:
        Estimated Y coordinate or None.
    """
    if not previous_points:
        return None
    
    # Expected Y ranges (at 0.005 resolution)
    # These correspond to two block type positions
    range1_center = 1150
    range2_center = 1580
    
    tolerance = Y_ESTIMATE_TOLERANCE
    range1 = (range1_center * (1 - tolerance), range1_center * (1 + tolerance))
    range2 = (range2_center * (1 - tolerance), range2_center * (1 + tolerance))
    
    last_y = previous_points[-1][1][1]
    
    if range1[0] <= last_y <= range1[1]:
        return last_y + Y_OFFSET_BLOCKS
    elif range2[0] <= last_y <= range2[1]:
        return last_y - Y_OFFSET_BLOCKS
    
    # Try second-to-last point
    if len(previous_points) > 1:
        second_last_y = previous_points[-2][1][1]
        if range1[0] <= second_last_y <= range1[1]:
            return second_last_y
        elif range2[0] <= second_last_y <= range2[1]:
            return second_last_y
    
    return None


# =============================================================================
# Visualization
# =============================================================================

def visualize_detection(
    edge_image: np.ndarray,
    positive_lines: List[np.ndarray],
    negative_lines: List[np.ndarray],
    horizontal_lines: List[np.ndarray],
    vertical_lines: List[Tuple[float, float]],
    center_lines: List[Tuple[float, float]],
    output_path: str
) -> None:
    """
    Visualize detected lines on the edge image.
    
    Args:
        edge_image: Binary edge image.
        positive_lines: Positive angle lines.
        negative_lines: Negative angle lines.
        horizontal_lines: Horizontal lines.
        vertical_lines: Vertical lines.
        center_lines: Center lines.
        output_path: Path to save visualization.
    """
    height, width = edge_image.shape
    output = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    
    # Colors
    colors = {
        'positive': (255, 0, 0),  # Red
        'negative': (0, 255, 0),  # Green
        'horizontal': (0, 0, 255),  # Blue
        'vertical': (255, 165, 0),  # Orange
        'center': (255, 0, 255)  # Magenta
    }
    thickness = 3
    
    # Draw oblique lines
    for line in positive_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), colors['positive'], thickness)
    
    for line in negative_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), colors['negative'], thickness)
    
    # Draw horizontal lines
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), colors['horizontal'], thickness)
    
    # Draw vertical lines
    for rho, theta in vertical_lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 + height * (-b))
        y1 = int(y0 + height * a)
        x2 = int(x0 - height * (-b))
        y2 = int(y0 - height * a)
        cv2.line(output, (x1, y1), (x2, y2), colors['vertical'], thickness)
    
    # Draw center lines
    for x_pos, theta in center_lines:
        x = int(x_pos)
        cv2.line(output, (x, 0), (x, height), colors['center'], thickness)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(output)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_prompt_points(
    prompt_points: List[Tuple[str, Tuple[float, float]]],
    output_path: str
) -> None:
    """
    Visualize prompt points with type-based colors.
    
    Args:
        prompt_points: List of (type, (x, y)) points.
        output_path: Path to save visualization.
    """
    colors = {
        'horizontal': 'b',
        'positive_slope': 'r',
        'negative_slope': 'c',
        'midpoint': 'm',
        'assume': 'g',
        'default': 'orange'
    }
    markers = {
        'horizontal': 'o',
        'positive_slope': '^',
        'negative_slope': 's',
        'midpoint': '*',
        'assume': 'd',
        'default': 'x'
    }
    
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    
    for label, (x, y) in prompt_points:
        ax.plot(x, y, color=colors[label], marker=markers[label], markersize=10, label=label)
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Detected Prompt Points')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.csv', '_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def detect_prompt_points(
    tunnel_id: str,
    base_dir: str = "data",
    resolution: float = 0.005
) -> pd.DataFrame:
    """
    Execute the complete prompt point detection pipeline.
    
    Args:
        tunnel_id: Tunnel identifier.
        base_dir: Base data directory.
        resolution: Depth map resolution.
        
    Returns:
        DataFrame with detected prompt points.
    """
    print(f"Processing tunnel: {tunnel_id}")
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load data
    depth_map = np.load(os.path.join(tunnel_dir, "depth_map_outlier.npy"))
    with open(os.path.join(tunnel_dir, "ring_count.txt"), 'r') as f:
        ring_count = int(f.read().strip())
    
    height, width = depth_map.shape
    print(f"Depth map size: {height} x {width}")
    print(f"Ring count: {ring_count}")
    
    # Preprocess
    print("Preprocessing depth map...")
    edge_image = preprocess_depth_map(depth_map)
    
    # Detect lines
    print("Detecting oblique lines...")
    positive_lines, negative_lines = detect_oblique_lines(edge_image)
    print(f"  Positive: {len(positive_lines)}, Negative: {len(negative_lines)}")
    
    print("Detecting horizontal lines...")
    horizontal_lines = detect_horizontal_lines(edge_image)
    print(f"  Horizontal: {len(horizontal_lines)}")
    
    print("Detecting vertical lines...")
    vertical_lines = detect_vertical_lines(edge_image, resolution, ring_count)
    print(f"  Vertical: {len(vertical_lines)}")
    
    # Generate center lines
    print("Generating center lines...")
    if vertical_lines:
        center_lines = generate_center_lines(vertical_lines, width, height, ring_count)
    else:
        print("  No vertical lines detected, using fallback method")
        center_lines = generate_fallback_center_lines(width, ring_count)
    print(f"  Center lines: {len(center_lines)}")
    
    # Compute prompt points
    print("Computing prompt points...")
    prompt_points = compute_prompt_points(
        center_lines, positive_lines, negative_lines, horizontal_lines,
        resolution, height
    )
    print(f"  Prompt points: {len(prompt_points)}")
    
    # Create output DataFrame
    df = pd.DataFrame(prompt_points, columns=['Type', 'Coordinates'])
    df['X'] = df['Coordinates'].apply(lambda c: c[0])
    df['Y'] = df['Coordinates'].apply(lambda c: c[1])
    df = df.drop(columns=['Coordinates'])
    df = df.sort_values(by='X').reset_index(drop=True)
    
    # Save results
    os.makedirs(tunnel_dir, exist_ok=True)
    
    # Visualization
    visualize_detection(
        edge_image, positive_lines, negative_lines, horizontal_lines,
        vertical_lines, center_lines,
        os.path.join(tunnel_dir, "detected_lines.png")
    )
    
    df.to_csv(os.path.join(tunnel_dir, "detected.csv"), index=False)
    
    print(f"\nResults saved to {tunnel_dir}/")
    print(df)
    
    return df


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 4-1_detection_clean.py <tunnel_id>")
        print("Example: python 4-1_detection_clean.py 1-4")
        sys.exit(1)
    
    detect_prompt_points(sys.argv[1])


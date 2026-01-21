"""
Improved Detection for Tunnel 4-1

Combines insights from:
1. Reasoning model analysis (detecting_analyser.md)
2. Pattern recognition from point cloud characteristics
3. Physical constraints (7-segment tunnel)

Key improvements from reasoning model:
- Lower binarization threshold (120 vs 127) for faint edges
- Larger morphological kernel (5x5) with 2 iterations
- Adjusted Hough parameters for sparser points
- Wider angle acceptance (5-10° vs 6-9°)
- Higher vertical line vote threshold (700)

Author: Combined Analysis
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Configuration
TUNNEL_ID = "4-1"
BASE_DIR = f"data/{TUNNEL_ID}"

# Physical constants for 7-segment tunnel
TUNNEL_DIAMETER = 5.5  # meters
K_HEIGHT_MM = 1079.92  # mm (from reasoning model)
AB_HEIGHT_MM = 3239.77  # mm (from reasoning model)
SEGMENT_WIDTH_MM = 1250  # mm (from reasoning model)
RESOLUTION = 0.005  # meters per pixel

# Convert to pixels
K_HEIGHT_PX = K_HEIGHT_MM / (1000 * RESOLUTION)  # ~216 px
AB_HEIGHT_PX = AB_HEIGHT_MM / (1000 * RESOLUTION)  # ~648 px

# Detection parameters (from reasoning model analysis)
BINARY_THRESHOLD = 120  # Lower than default 127 for faint edges
KERNEL_SIZE = 5  # Larger than default 3
DILATION_ITERATIONS = 2  # More than default 1
HOUGH_THRESHOLD_OBLIQUE = 40  # Lower than default 50
MIN_LINE_LENGTH_RATIO = 0.1  # 10% of image width
MAX_LINE_GAP = 60  # Higher than default 40
ANGLE_MIN = 5  # Wider than default 6
ANGLE_MAX = 10  # Wider than default 9
VERTICAL_VOTE_THRESHOLD = 700  # Higher than default 500


def load_data():
    """Load depth map and ring count"""
    depth_map_outlier = np.load(os.path.join(BASE_DIR, "depth_map_outlier.npy"))
    depth_map = cv2.imread(os.path.join(BASE_DIR, "depth_map.png"), cv2.IMREAD_GRAYSCALE)
    ring_count = int(open(os.path.join(BASE_DIR, "ring_count.txt"), 'r').read())
    
    print(f"Loaded tunnel {TUNNEL_ID}")
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Ring count: {ring_count}")
    
    return depth_map_outlier, depth_map, ring_count


def preprocess_image(depth_map_outlier):
    """Preprocess with reasoning model recommendations"""
    # Create binary map
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    
    # Binarization with lower threshold (reasoning model: 120)
    _, binary_image = cv2.threshold(binary_map, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Morphology with larger kernel (reasoning model: 5x5, 2 iterations)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    dilated_edges = cv2.dilate(binary_image, kernel, iterations=DILATION_ITERATIONS)
    
    return binary_map, dilated_edges


def detect_lines_hough(dilated_edges, L, W):
    """Detect lines using improved Hough parameters"""
    
    # Oblique lines with adjusted parameters
    lines_oblique = cv2.HoughLinesP(
        dilated_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD_OBLIQUE,
        minLineLength=int(MIN_LINE_LENGTH_RATIO * W),
        maxLineGap=MAX_LINE_GAP
    )
    
    # Horizontal lines
    lines_horizontal = cv2.HoughLinesP(
        dilated_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=10
    )
    
    # Vertical lines with higher threshold
    lines_vertical = cv2.HoughLines(
        dilated_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=VERTICAL_VOTE_THRESHOLD
    )
    
    return lines_oblique, lines_horizontal, lines_vertical


def filter_oblique_lines(lines_oblique):
    """Filter oblique lines with wider angle acceptance"""
    positive_slope = []
    negative_slope = []
    
    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            if x1 > x2:
                x1, x2, y1, y2 = x2, x1, y2, y1
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            
            # Wider angle acceptance (reasoning model: 5-10°)
            if ANGLE_MIN <= angle <= ANGLE_MAX:
                positive_slope.append(line)
            elif -ANGLE_MAX <= angle <= -ANGLE_MIN:
                negative_slope.append(line)
    
    return positive_slope, negative_slope


def detect_k_block_gradient(depth_map, x_start, x_end, image_height):
    """
    Improved K-block detection using gradient analysis.
    K-block is ~216px wide (physical constraint).
    """
    K_MAX_WIDTH = 250  # Maximum width to consider as K-block
    
    # Get ring intensity profile
    ring_strip = depth_map[:, x_start:x_end]
    intensity_profile = np.mean(ring_strip, axis=1)
    intensity_smooth = gaussian_filter1d(intensity_profile, sigma=10)
    
    # Use gradient to find edges
    gradient = np.abs(np.gradient(intensity_smooth))
    gradient_smooth = gaussian_filter1d(gradient, sigma=5)
    
    # Find all significant edges
    edges, _ = find_peaks(gradient_smooth, distance=50, prominence=1)
    
    if len(edges) < 3:
        return None
    
    # Calculate segments between edges
    segments = []
    for i in range(len(edges) - 1):
        y_start = edges[i]
        y_end = edges[i + 1]
        width = y_end - y_start
        center = (y_start + y_end) / 2
        avg_intensity = np.mean(intensity_smooth[y_start:y_end])
        
        segments.append({
            'y_start': y_start,
            'y_end': y_end,
            'center': center,
            'width': width,
            'intensity': avg_intensity
        })
    
    # Find K-block candidates
    margin = int(image_height * 0.1)
    k_candidates = [s for s in segments 
                    if s['width'] < K_MAX_WIDTH 
                    and margin < s['center'] < image_height - margin]
    
    if not k_candidates:
        return None
    
    # Best candidate: closest to expected K-block width
    best = min(k_candidates, key=lambda x: abs(x['width'] - K_HEIGHT_PX))
    return best


def process_vertical_lines(lines_vertical, L, W, ring_count):
    """Process and merge vertical lines - always use evenly spaced for consistency"""
    # For tunnel 4-1, use evenly spaced mid-lines based on ring count
    # This is more reliable than Hough-detected vertical lines
    print(f"Using evenly spaced mid-lines for {ring_count} rings")
    
    mid_lines = []
    block = W / ring_count
    for i in range(ring_count):
        x = (i + 0.5) * block
        mid_lines.append((x, 0))
    
    return mid_lines


def compute_intersections(mid_lines, positive_slope, negative_slope, horizontal_lines, L, W):
    """Compute intersection points with improved logic"""
    
    def line_segment_vertical_intersection(vertical_x, segment):
        x1, y1, x2, y2 = segment
        if x1 == x2:
            return None
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            t = (vertical_x - x1) / (x2 - x1)
            return (vertical_x, y1 + t * (y2 - y1))
        return None
    
    def merge_close_points(points, threshold=6):
        if len(points) < 2:
            return np.array(points) if points else np.array([])
        pts = np.array(points)
        merged = []
        while len(pts) > 0:
            p = pts[0]
            close = np.linalg.norm(pts - p, axis=1) < threshold
            merged.append(np.mean(pts[close], axis=0))
            pts = pts[~close]
        return np.array(merged)
    
    adjusted_points = []
    
    for vx, _ in mid_lines:
        pos_pts = []
        neg_pts = []
        hor_pts = []
        
        for seg in positive_slope:
            ip = line_segment_vertical_intersection(vx, seg[0])
            if ip is not None:
                pos_pts.append(ip)
        
        for seg in negative_slope:
            ip = line_segment_vertical_intersection(vx, seg[0])
            if ip is not None:
                neg_pts.append(ip)
        
        mpos = merge_close_points(pos_pts)
        mneg = merge_close_points(neg_pts)
        
        if len(mpos) > 0 and len(mneg) > 0:
            # Midpoint between positive and negative slope intersections
            mid_y = (mpos[0][1] + mneg[0][1]) / 2
            adjusted_points.append(('midpoint', (vx, mid_y)))
        elif len(mpos) > 0:
            x, y = mpos[0]
            adjusted_points.append(('positive_slope', (x, y - 0.5 * K_HEIGHT_PX)))
        elif len(mneg) > 0:
            x, y = mneg[0]
            adjusted_points.append(('negative_slope', (x, y + 0.5 * K_HEIGHT_PX)))
        else:
            # Fallback: use horizontal lines or default
            for seg in horizontal_lines if horizontal_lines is not None else []:
                ip = line_segment_vertical_intersection(vx, seg[0])
                if ip is not None:
                    hor_pts.append(ip)
            
            mhor = merge_close_points(hor_pts)
            if len(mhor) > 0:
                adjusted_points.append(('horizontal', tuple(mhor[0])))
            else:
                # Use previous point or default
                if adjusted_points:
                    last_y = adjusted_points[-1][1][1]
                    adjusted_points.append(('assume', (vx, last_y)))
                else:
                    adjusted_points.append(('default', (vx, L / 2)))
    
    return adjusted_points


def refine_with_pattern_recognition(adjusted_points, depth_map, ring_count):
    """Refine K-block positions using pattern recognition for ALL rings"""
    L, W = depth_map.shape
    ring_width = W / ring_count
    
    refined_points = []
    
    for i, (ptype, (x, y)) in enumerate(adjusted_points):
        # Always try gradient-based K-block detection for each ring
        x_center = int(x)
        x_start = int(max(0, x_center - ring_width * 0.4))
        x_end = int(min(W, x_center + ring_width * 0.4))
        
        k_block = detect_k_block_gradient(depth_map, x_start, x_end, L)
        
        if k_block:
            # Use gradient-detected position
            refined_points.append(('pattern', (x, k_block['center'])))
            print(f"Ring {i}: K-block at Y={k_block['center']:.0f}, width={k_block['width']:.0f}")
        else:
            # Fallback to original or default
            if ptype not in ['default', 'assume']:
                refined_points.append((ptype, (x, y)))
            else:
                # Use image center as last resort
                refined_points.append(('default', (x, L / 2)))
                print(f"Ring {i}: Using default Y={L/2:.0f}")
    
    return refined_points


def main():
    print("=" * 60)
    print("Improved Detection for Tunnel 4-1")
    print("Combining Reasoning Model + Pattern Recognition")
    print("=" * 60)
    
    # Load data
    depth_map_outlier, depth_map, ring_count = load_data()
    L, W = depth_map.shape
    
    # Preprocess with reasoning model recommendations
    binary_map, dilated_edges = preprocess_image(depth_map_outlier)
    print(f"Preprocessed with threshold={BINARY_THRESHOLD}, kernel={KERNEL_SIZE}x{KERNEL_SIZE}")
    
    # Detect lines with improved Hough parameters
    lines_oblique, lines_horizontal, lines_vertical = detect_lines_hough(dilated_edges, L, W)
    
    # Filter oblique lines with wider angle acceptance
    positive_slope, negative_slope = filter_oblique_lines(lines_oblique)
    print(f"Detected oblique lines: {len(positive_slope)} positive, {len(negative_slope)} negative")
    
    # Process vertical lines
    mid_lines = process_vertical_lines(lines_vertical, L, W, ring_count)
    print(f"Mid-lines: {len(mid_lines)}")
    
    # Compute intersections
    adjusted_points = compute_intersections(
        mid_lines, positive_slope, negative_slope, lines_horizontal, L, W
    )
    
    # Refine with pattern recognition
    refined_points = refine_with_pattern_recognition(adjusted_points, depth_map, ring_count)
    
    # Create DataFrame
    df_loc = pd.DataFrame(refined_points, columns=['Type', 'Coordinates'])
    df_loc['X'] = df_loc['Coordinates'].apply(lambda c: c[0])
    df_loc['Y'] = df_loc['Coordinates'].apply(lambda c: c[1])
    df_loc = df_loc.drop(columns=['Coordinates']).sort_values(by='X').reset_index(drop=True)
    
    print(f"\nDetected {len(df_loc)} points:")
    print(df_loc)
    
    # Save results
    df_loc.to_csv(os.path.join(BASE_DIR, "detected.csv"), index=False)
    print(f"\nSaved to {BASE_DIR}/detected.csv")
    
    # Visualize
    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    colors = {
        'horizontal': (0, 0, 255),
        'positive_slope': (255, 0, 0),
        'negative_slope': (0, 255, 0),
        'midpoint': (255, 0, 255),
        'assume': (0, 255, 255),
        'default': (255, 165, 0),
        'pattern': (255, 255, 0)
    }
    
    for ptype, (x, y) in refined_points:
        color = colors.get(ptype, (255, 255, 255))
        cv2.circle(output_image, (int(x), int(y)), 15, color, -1)
    
    cv2.imwrite(os.path.join(BASE_DIR, "detected_lines.png"), output_image)
    print(f"Saved visualization to {BASE_DIR}/detected_lines.png")
    
    return df_loc


if __name__ == "__main__":
    main()

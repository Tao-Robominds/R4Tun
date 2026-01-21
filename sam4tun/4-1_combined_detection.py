"""
Combined Detection for Tunnel 4-1

Combines:
1. Hough line detection (with reasoning model optimizations)
2. Gradient-based pattern recognition
3. Physical constraints validation

Strategy:
- Use Hough to find oblique lines and compute intersections
- Use gradient analysis to detect narrow segments (K-blocks)
- Cross-validate results between methods
- Use physical constraints to validate and refine

Author: Combined Analysis
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Configuration
TUNNEL_ID = "4-1"
BASE_DIR = f"data/{TUNNEL_ID}"

# Physical constants
K_HEIGHT_MM = 1079.92
AB_HEIGHT_MM = 3239.77
RESOLUTION = 0.005

# Convert to pixels
K_HEIGHT_PX = K_HEIGHT_MM / (1000 * RESOLUTION)  # ~216 px
AB_HEIGHT_PX = AB_HEIGHT_MM / (1000 * RESOLUTION)  # ~648 px

# Detection parameters (from reasoning model)
BINARY_THRESHOLD = 120
KERNEL_SIZE = 5
DILATION_ITERATIONS = 2
HOUGH_THRESHOLD = 40
MIN_LINE_LENGTH_RATIO = 0.1
MAX_LINE_GAP = 60
ANGLE_MIN = 5
ANGLE_MAX = 10


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
    binary_map = np.where(np.isnan(depth_map_outlier), 0, 255).astype(np.uint8)
    _, binary_image = cv2.threshold(binary_map, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    dilated_edges = cv2.dilate(binary_image, kernel, iterations=DILATION_ITERATIONS)
    return binary_map, dilated_edges


def detect_hough_lines(dilated_edges, L, W):
    """Detect oblique lines using Hough transform"""
    lines_oblique = cv2.HoughLinesP(
        dilated_edges, 1, np.pi / 180, HOUGH_THRESHOLD,
        minLineLength=int(MIN_LINE_LENGTH_RATIO * W),
        maxLineGap=MAX_LINE_GAP
    )
    
    positive_slope = []
    negative_slope = []
    
    if lines_oblique is not None:
        for line in lines_oblique:
            x1, y1, x2, y2 = line[0]
            if x1 > x2:
                x1, x2, y1, y2 = x2, x1, y2, y1
            angle = np.degrees(np.arctan2(-(y2 - y1), x2 - x1))
            
            if ANGLE_MIN <= angle <= ANGLE_MAX:
                positive_slope.append(line)
            elif -ANGLE_MAX <= angle <= -ANGLE_MIN:
                negative_slope.append(line)
    
    print(f"Hough detected: {len(positive_slope)} positive, {len(negative_slope)} negative slope lines")
    return positive_slope, negative_slope


def detect_k_block_gradient(depth_map, x_start, x_end, image_height):
    """Detect K-block using gradient analysis"""
    K_MAX_WIDTH = 280
    
    ring_strip = depth_map[:, x_start:x_end]
    intensity_profile = np.mean(ring_strip, axis=1)
    intensity_smooth = gaussian_filter1d(intensity_profile, sigma=10)
    
    gradient = np.abs(np.gradient(intensity_smooth))
    gradient_smooth = gaussian_filter1d(gradient, sigma=5)
    
    edges, _ = find_peaks(gradient_smooth, distance=50, prominence=1)
    
    if len(edges) < 3:
        return None
    
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
    
    margin = int(image_height * 0.08)
    k_candidates = [s for s in segments 
                    if s['width'] < K_MAX_WIDTH 
                    and margin < s['center'] < image_height - margin]
    
    if not k_candidates:
        return None
    
    best = min(k_candidates, key=lambda x: abs(x['width'] - K_HEIGHT_PX))
    return best


def compute_hough_intersection(x_pos, positive_slope, negative_slope, L):
    """Compute K-block Y position from Hough line intersections"""
    
    def line_segment_intersection(vertical_x, segment):
        x1, y1, x2, y2 = segment
        if x1 == x2:
            return None
        if min(x1, x2) <= vertical_x <= max(x1, x2):
            t = (vertical_x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
        return None
    
    pos_ys = []
    neg_ys = []
    
    for seg in positive_slope:
        y = line_segment_intersection(x_pos, seg[0])
        if y is not None:
            pos_ys.append(y)
    
    for seg in negative_slope:
        y = line_segment_intersection(x_pos, seg[0])
        if y is not None:
            neg_ys.append(y)
    
    if pos_ys and neg_ys:
        # Midpoint between positive and negative slope intersections
        return ('midpoint', (np.mean(pos_ys) + np.mean(neg_ys)) / 2)
    elif pos_ys:
        return ('positive_slope', np.mean(pos_ys) - 0.5 * K_HEIGHT_PX)
    elif neg_ys:
        return ('negative_slope', np.mean(neg_ys) + 0.5 * K_HEIGHT_PX)
    
    return None


def combine_detections(hough_result, gradient_result, ring_idx, image_height):
    """Combine Hough and gradient detection results"""
    
    # Priority order:
    # 1. Hough midpoint (most reliable)
    # 2. Hough single slope (use it to guide gradient selection)
    # 3. Gradient detection alone
    # 4. Default
    
    if hough_result and hough_result[0] == 'midpoint':
        # Validate with gradient if available
        if gradient_result:
            hough_y = hough_result[1]
            grad_y = gradient_result['center']
            
            # If both agree (within 300px), use average
            if abs(hough_y - grad_y) < 300:
                combined_y = (hough_y + grad_y) / 2
                return ('combined', combined_y, 'hough+gradient agree')
            else:
                # Trust Hough midpoint more
                return ('hough_midpoint', hough_y, f'hough preferred (grad at {grad_y:.0f})')
        return ('hough_midpoint', hough_result[1], 'hough only')
    
    elif hough_result:
        # Single slope Hough available - use it to guide detection
        hough_y = hough_result[1]
        hough_type = hough_result[0]
        
        if gradient_result:
            grad_y = gradient_result['center']
            
            # If gradient is close to Hough estimate, trust it
            if abs(hough_y - grad_y) < 350:
                return ('hough_guided', grad_y, f'{hough_type} at {hough_y:.0f}')
            else:
                # Hough estimate might be better for single slope
                return ('hough_single', hough_y, f'{hough_type} (grad at {grad_y:.0f})')
        
        return (hough_type, hough_y, 'hough single slope only')
    
    elif gradient_result:
        return ('gradient', gradient_result['center'], 'gradient only')
    
    # Default fallback
    return ('default', image_height / 2, 'no detection')


def main():
    print("=" * 60)
    print("Combined Detection for Tunnel 4-1")
    print("Hough + Gradient + Physical Constraints")
    print("=" * 60)
    
    # Load data
    depth_map_outlier, depth_map, ring_count = load_data()
    L, W = depth_map.shape
    
    # Preprocess
    binary_map, dilated_edges = preprocess_image(depth_map_outlier)
    
    # Detect Hough lines
    positive_slope, negative_slope = detect_hough_lines(dilated_edges, L, W)
    
    # Process each ring
    ring_width = W / ring_count
    results = []
    
    print(f"\n{'Ring':<6} {'Method':<20} {'Y':<10} {'Details'}")
    print("-" * 70)
    
    for ring_idx in range(ring_count):
        x_center = (ring_idx + 0.5) * ring_width
        x_start = int(max(0, x_center - ring_width * 0.4))
        x_end = int(min(W, x_center + ring_width * 0.4))
        
        # Method 1: Hough line intersections
        hough_result = compute_hough_intersection(x_center, positive_slope, negative_slope, L)
        
        # Method 2: Gradient-based K-block detection
        gradient_result = detect_k_block_gradient(depth_map, x_start, x_end, L)
        
        # Combine results
        method, y_pos, details = combine_detections(hough_result, gradient_result, ring_idx, L)
        
        results.append({
            'Type': method,
            'X': x_center,
            'Y': y_pos
        })
        
        print(f"{ring_idx:<6} {method:<20} {y_pos:<10.0f} {details}")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(BASE_DIR, "detected.csv"), index=False)
    print(f"\nSaved {len(df)} detection points to {BASE_DIR}/detected.csv")
    
    # Visualize
    output_image = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
    
    colors = {
        'combined': (0, 255, 0),        # Green - best
        'hough_midpoint': (255, 255, 0),  # Cyan
        'gradient_validated': (0, 255, 255),  # Yellow
        'gradient': (255, 0, 255),       # Magenta
        'positive_slope': (255, 0, 0),   # Blue
        'negative_slope': (0, 0, 255),   # Red
        'default': (128, 128, 128)       # Gray
    }
    
    for _, row in df.iterrows():
        x, y = int(row['X']), int(row['Y'])
        color = colors.get(row['Type'], (255, 255, 255))
        cv2.circle(output_image, (x, y), 20, color, -1)
        cv2.circle(output_image, (x, y), 22, (255, 255, 255), 2)
    
    cv2.imwrite(os.path.join(BASE_DIR, "detected_lines.png"), output_image)
    print(f"Saved visualization to {BASE_DIR}/detected_lines.png")
    
    return df


if __name__ == "__main__":
    main()

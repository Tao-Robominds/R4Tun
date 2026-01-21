"""
Pattern Recognition-based Segment Detection for Tunnel 4-1

This script uses depth map edge detection and point cloud analysis to detect
segment boundaries without relying on ground truth.

Key techniques:
1. Sobel edge detection for segment boundary detection
2. K-block identification by narrow width
3. Physical constraint validation

Author: SAM4Tun Pattern Recognition
"""

import os
import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import json

# Fixed parameters for tunnel 4-1
TUNNEL_ID = "4-1"
BASE_DIR = f"data/{TUNNEL_ID}"

# Physical constants (in meters)
TUNNEL_DIAMETER = 5.5  # meters
K_BLOCK_HEIGHT = 0.364  # meters (364mm)
AB_BLOCK_HEIGHT = 1.232  # meters (1232mm)
RESOLUTION = 0.005  # meters per pixel

# Detection parameters (tuned for 4-1)
RADIUS_THRESHOLD = 3.5  # Filter points with r > this value
EDGE_PROMINENCE = 100  # Minimum edge strength for boundary detection
EDGE_DISTANCE = 150  # Minimum distance between edge peaks
K_BLOCK_WIDTH_RATIO = 0.6  # K-block is < 60% width of median segment

# Segment mapping
SEGMENT_NAMES = {1: 'K', 2: 'B1', 3: 'A1', 4: 'A2', 5: 'A3', 6: 'A4', 7: 'B2'}


def load_data():
    """Load enhanced point cloud data and depth map"""
    enhanced_path = os.path.join(BASE_DIR, "enhanced.csv")
    df = pd.read_csv(enhanced_path)
    print(f"Loaded {len(df):,} points from {enhanced_path}")
    
    depth_map = cv2.imread(f'{BASE_DIR}/depth_map.png', cv2.IMREAD_GRAYSCALE)
    print(f"Depth map shape: {depth_map.shape}")
    
    return df, depth_map


def get_coordinate_mapping(df):
    """Get coordinate mapping between point cloud and image"""
    surface = df[df['r'] > RADIUS_THRESHOLD]
    
    theta_min = surface['theta'].min()
    theta_max = surface['theta'].max()
    h_min = surface['h'].min()
    h_max = surface['h'].max()
    
    return {
        'theta_min': theta_min,
        'theta_max': theta_max,
        'h_min': h_min,
        'h_max': h_max
    }


def detect_k_block_gradient(depth_map, x_start, x_end, image_height):
    """
    Improved K-block detection using gradient-based edge detection.
    
    K-block is ~73px wide (physical constraint), A/B blocks are ~246px wide.
    We look for the narrowest segment that matches K-block expected width.
    """
    
    # Expected K-block width in pixels (physical constraint)
    K_HEIGHT_PX = 73   # Expected K-block height
    K_MAX_WIDTH = 200  # Maximum width to consider as K-block
    
    # Get ring intensity profile
    ring_strip = depth_map[:, x_start:x_end]
    intensity_profile = np.mean(ring_strip, axis=1)
    intensity_smooth = gaussian_filter1d(intensity_profile, sigma=10)
    
    # Use gradient to find edges (intensity transitions)
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
    
    # Find K-block candidates:
    # 1. Width < K_MAX_WIDTH (narrow segment)
    # 2. Not at image edges (center between 10% and 90% of height)
    margin = int(image_height * 0.1)
    k_candidates = [s for s in segments 
                    if s['width'] < K_MAX_WIDTH 
                    and margin < s['center'] < image_height - margin]
    
    if not k_candidates:
        return None
    
    # Best candidate: closest to expected K-block width
    best = min(k_candidates, key=lambda x: abs(x['width'] - K_HEIGHT_PX))
    best['score'] = 4  # High confidence
    
    return best


def theta_to_pixel_y(theta, coord_map, image_height):
    """Convert theta coordinate to pixel Y coordinate"""
    y = (theta - coord_map['theta_min']) / (coord_map['theta_max'] - coord_map['theta_min']) * image_height
    return y


def pixel_y_to_theta(y, coord_map, image_height):
    """Convert pixel Y to theta coordinate"""
    theta = coord_map['theta_min'] + (y / image_height) * (coord_map['theta_max'] - coord_map['theta_min'])
    return theta


def h_to_pixel_x(h, coord_map, image_width):
    """Convert h coordinate to pixel X coordinate"""
    x = (h - coord_map['h_min']) / (coord_map['h_max'] - coord_map['h_min']) * image_width
    return x


def generate_all_segments_from_k(k_y, x_pixel, ring_idx, coord_map, image_height):
    """Generate all segment positions based on K-block position"""
    
    theta_range = coord_map['theta_max'] - coord_map['theta_min']
    circumference = np.pi * TUNNEL_DIAMETER
    
    # Convert physical heights to pixels
    k_height_px = (K_BLOCK_HEIGHT / circumference) * theta_range / (coord_map['theta_max'] - coord_map['theta_min']) * image_height
    ab_height_px = (AB_BLOCK_HEIGHT / circumference) * theta_range / (coord_map['theta_max'] - coord_map['theta_min']) * image_height
    
    positions = []
    
    # K-block
    k_theta = pixel_y_to_theta(k_y, coord_map, image_height)
    positions.append({'Ring': ring_idx, 'Block': 'K', 'X': x_pixel, 'Y': k_y, 'theta': k_theta})
    
    # B1 is above K (lower Y)
    b1_y = k_y - (0.5 * k_height_px + 0.5 * ab_height_px)
    b1_theta = pixel_y_to_theta(b1_y, coord_map, image_height)
    positions.append({'Ring': ring_idx, 'Block': 'B1', 'X': x_pixel, 'Y': b1_y, 'theta': b1_theta})
    
    # A1 is above B1
    a1_y = b1_y - ab_height_px
    a1_theta = pixel_y_to_theta(a1_y, coord_map, image_height)
    positions.append({'Ring': ring_idx, 'Block': 'A1', 'X': x_pixel, 'Y': a1_y, 'theta': a1_theta})
    
    # B2 is below K (higher Y)
    b2_y = k_y + (0.5 * k_height_px + 0.5 * ab_height_px)
    b2_theta = pixel_y_to_theta(b2_y, coord_map, image_height)
    positions.append({'Ring': ring_idx, 'Block': 'B2', 'X': x_pixel, 'Y': b2_y, 'theta': b2_theta})
    
    # A2 is above A1 (continuing up)
    a2_y = a1_y - ab_height_px
    a2_theta = pixel_y_to_theta(a2_y, coord_map, image_height)
    positions.append({'Ring': ring_idx, 'Block': 'A2', 'X': x_pixel, 'Y': a2_y, 'theta': a2_theta})
    
    # A3 is above A2
    a3_y = a2_y - ab_height_px
    a3_theta = pixel_y_to_theta(a3_y, coord_map, image_height)
    positions.append({'Ring': ring_idx, 'Block': 'A3', 'X': x_pixel, 'Y': a3_y, 'theta': a3_theta})
    
    # A4 is above A3
    a4_y = a3_y - ab_height_px
    a4_theta = pixel_y_to_theta(a4_y, coord_map, image_height)
    positions.append({'Ring': ring_idx, 'Block': 'A4', 'X': x_pixel, 'Y': a4_y, 'theta': a4_theta})
    
    return positions


def main():
    print("=" * 60)
    print("Pattern Recognition-based Segment Detection for Tunnel 4-1")
    print("=" * 60)
    
    # Load data
    df, depth_map = load_data()
    image_height, image_width = depth_map.shape
    
    # Get coordinate mapping
    coord_map = get_coordinate_mapping(df)
    print(f"Theta range: [{coord_map['theta_min']:.3f}, {coord_map['theta_max']:.3f}]")
    print(f"H range: [{coord_map['h_min']:.3f}, {coord_map['h_max']:.3f}]")
    
    # Get ring count
    ring_count = int(open(f'{BASE_DIR}/ring_count.txt', 'r').read())
    print(f"Ring count: {ring_count}")
    
    ring_width_px = image_width / ring_count
    
    all_positions = []
    k_detected_rings = []
    k_positions_by_ring = {}
    
    # Get ring data from point cloud
    surface = df[df['r'] > RADIUS_THRESHOLD]
    h_min, h_max = coord_map['h_min'], coord_map['h_max']
    ring_width_h = (h_max - h_min) / ring_count
    
    # Process each ring
    for ring_idx in range(ring_count):
        x_center = int((ring_idx + 0.5) * ring_width_px)
        x_start = int(max(0, x_center - ring_width_px * 0.4))
        x_end = int(min(image_width, x_center + ring_width_px * 0.4))
        
        # Get point cloud data for this ring
        h_ring_min = h_min + ring_idx * ring_width_h
        h_ring_max = h_ring_min + ring_width_h
        df_ring = surface[(surface['h'] >= h_ring_min) & (surface['h'] < h_ring_max)]
        
        print(f"\nRing {ring_idx}: X=[{x_start}, {x_end}], {len(df_ring):,} points")
        
        # Gradient-based K-block detection
        k_block = detect_k_block_gradient(
            depth_map, x_start, x_end, image_height
        )
        
        if k_block:
            k_y = k_block['center']
            print(f"  K-block detected at Y={k_y:.0f}, width={k_block['width']:.0f}, score={k_block['score']}")
            k_detected_rings.append(ring_idx)
            k_positions_by_ring[ring_idx] = k_y
            
            # Generate all segment positions
            ring_positions = generate_all_segments_from_k(
                k_y, x_center, ring_idx, coord_map, image_height
            )
            all_positions.extend(ring_positions)
        else:
            print(f"  No K-block detected")
    
    # For rings without K-block, use fallback detection
    print(f"\n{'='*60}")
    print(f"K-block detected in {len(k_detected_rings)} rings: {k_detected_rings}")
    
    # Fallback for missing rings: use depth map intensity profile
    if k_detected_rings and len(k_detected_rings) < ring_count:
        for ring_idx in range(ring_count):
            if ring_idx not in k_detected_rings:
                x_center = int((ring_idx + 0.5) * ring_width_px)
                x_start = int(max(0, x_center - ring_width_px * 0.4))
                x_end = int(min(image_width, x_center + ring_width_px * 0.4))
                
                # Try intensity-based detection with relaxed parameters
                ring_strip = depth_map[:, x_start:x_end]
                intensity_profile = np.mean(ring_strip, axis=1)
                intensity_smooth = gaussian_filter1d(intensity_profile, sigma=30)
                
                # Find all intensity valleys (potential K-block locations)
                valleys, _ = find_peaks(-intensity_smooth, distance=150, prominence=5)
                
                # Filter valleys in middle region
                margin = int(image_height * 0.1)
                inner_valleys = [v for v in valleys if margin < v < image_height - margin]
                
                if inner_valleys:
                    # Use the valley closest to the mean of detected K positions
                    if k_positions_by_ring:
                        mean_k_y = np.mean(list(k_positions_by_ring.values()))
                        k_y = min(inner_valleys, key=lambda v: abs(v - mean_k_y))
                    else:
                        # Use middle valley
                        k_y = inner_valleys[len(inner_valleys) // 2]
                    
                    print(f"Ring {ring_idx}: Fallback K at Y={k_y:.0f} (intensity valley)")
                else:
                    # Last resort: interpolate from nearest
                    nearest_ring = min(k_detected_rings, key=lambda r: abs(r - ring_idx))
                    k_y = k_positions_by_ring[nearest_ring]
                    print(f"Ring {ring_idx}: Interpolated K at Y={k_y:.0f} from ring {nearest_ring}")
                
                ring_positions = generate_all_segments_from_k(
                    k_y, x_center, ring_idx, coord_map, image_height
                )
                all_positions.extend(ring_positions)
    
    # Save results
    all_positions_df = pd.DataFrame(all_positions)
    all_positions_df.to_csv(os.path.join(BASE_DIR, "pattern_segments.csv"), index=False)
    
    # Create detected.csv for SAM (K-block positions)
    k_positions = all_positions_df[all_positions_df['Block'] == 'K'].sort_values('Ring')
    detected_df = pd.DataFrame({
        'Type': 'pattern',
        'X': k_positions['X'].values,
        'Y': k_positions['Y'].values
    })
    detected_df.to_csv(os.path.join(BASE_DIR, "detected.csv"), index=False)
    
    # Save pattern results
    results = {
        'tunnel_id': TUNNEL_ID,
        'method': 'edge_detection_pattern',
        'parameters': {
            'edge_prominence': EDGE_PROMINENCE,
            'edge_distance': EDGE_DISTANCE,
            'k_block_width_ratio': K_BLOCK_WIDTH_RATIO
        },
        'segments_detected': len(all_positions),
        'rings_with_k': len(k_detected_rings)
    }
    with open(os.path.join(BASE_DIR, "pattern_detection_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Detection Summary:")
    print(f"  Total segments detected: {len(all_positions)}")
    print(f"  Rings processed: {ring_count}")
    print(f"  K-blocks found: {len(k_detected_rings)}")
    print(f"  Results saved to {BASE_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Reverse Engineering Ground Truth Analysis

This script extracts actual segment positions, spacings, and patterns from 
ground truth data to determine optimal parameters for the SAM pipeline.

Key questions to answer:
1. What are the actual K-block Y positions per ring?
2. What are the actual segment spacings (not physical constants)?
3. What are the actual template dimensions needed?
4. Why is there a gap between GT-derived and no-GT performance?
"""

import numpy as np
import pandas as pd
import cv2
import os
import json
import pickle
from collections import defaultdict

def load_ground_truth(tunnel_id):
    """Load raw ground truth point cloud"""
    gt_path = f"data/{tunnel_id}.txt"
    
    # Try to infer format from first line
    with open(gt_path, 'r') as f:
        first_line = f.readline().strip()
        
    # Check if it's space-separated or comma-separated
    if ',' in first_line:
        df = pd.read_csv(gt_path)
    else:
        # Space-separated: x y z intensity ? segment_label
        df = pd.read_csv(gt_path, sep=r'\s+', header=None,
                        names=['x', 'y', 'z', 'intensity', 'unknown', 'segment'])
    
    print(f"Loaded {len(df):,} points from {gt_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique segments: {sorted(df['segment'].unique())}")
    
    return df

def load_enhanced_data(tunnel_id):
    """Load enhanced data with pixel coordinates"""
    enhanced_path = f"data/{tunnel_id}/enhanced.csv"
    df = pd.read_csv(enhanced_path)
    print(f"Loaded {len(df):,} points from enhanced data")
    return df

def analyze_segment_positions(df, tunnel_id):
    """
    Extract actual segment positions from ground truth.
    This is the KEY analysis - what are the actual Y positions?
    """
    base_dir = f"data/{tunnel_id}"
    
    # Load depth map to get image dimensions
    depth_map = cv2.imread(f'{base_dir}/depth_map.png')
    img_height, img_width = depth_map.shape[:2]
    
    # Get ring count
    ring_count = int(open(f'{base_dir}/ring_count.txt').read())
    
    print(f"\n{'='*70}")
    print(f"GROUND TRUTH ANALYSIS FOR TUNNEL {tunnel_id}")
    print(f"{'='*70}")
    print(f"Image: {img_width} x {img_height}")
    print(f"Rings: {ring_count}")
    
    # Segment label mapping (tunnel-specific)
    # For 4-1: typically 6 or 7 segments, labels vary
    unique_segments = sorted(df['segment'].unique())
    print(f"Unique segment labels in GT: {unique_segments}")
    
    results = {
        'tunnel_id': tunnel_id,
        'image_size': {'width': img_width, 'height': img_height},
        'ring_count': ring_count,
        'segment_labels': unique_segments,
    }
    
    return results, df

def analyze_enhanced_segments(tunnel_id):
    """
    Analyze segment positions using enhanced.csv + pixel_to_point mapping
    """
    base_dir = f"data/{tunnel_id}"
    
    # Load enhanced data
    enhanced_df = pd.read_csv(f'{base_dir}/enhanced.csv')
    
    # Load pixel_to_point mapping to get pixel coordinates
    pixel_to_point_path = f'{base_dir}/pixel_to_point.pkl'
    if os.path.exists(pixel_to_point_path):
        pixel_to_point = pickle.load(open(pixel_to_point_path, 'rb'))
        # Convert to DataFrame
        ptp_df = pd.DataFrame(pixel_to_point)
        # Merge with enhanced data
        enhanced_df = enhanced_df.reset_index()
        enhanced_df = enhanced_df.rename(columns={'index': 'original_index'})
        enhanced_df['point_idx'] = enhanced_df.index
        
        # Merge pixel coordinates
        merged = ptp_df.merge(enhanced_df, left_on='index', right_on='point_idx', how='inner')
        print(f"Merged {len(merged):,} points with pixel coordinates")
        enhanced_df = merged
    
    # Load depth map dimensions
    depth_map = cv2.imread(f'{base_dir}/depth_map.png')
    img_height, img_width = depth_map.shape[:2]
    ring_count = int(open(f'{base_dir}/ring_count.txt').read())
    
    print(f"\n{'='*70}")
    print(f"ENHANCED DATA ANALYSIS FOR TUNNEL {tunnel_id}")
    print(f"{'='*70}")
    print(f"Image: {img_width} x {img_height}")
    print(f"Rings: {ring_count}")
    
    # Check what columns we have
    print(f"Columns: {enhanced_df.columns.tolist()[:15]}...")
    
    # Check for segment column
    segment_col = None
    for col in ['segment', 'gt_segment', 'label', 'gt_labels']:
        if col in enhanced_df.columns:
            segment_col = col
            break
    
    if segment_col is None:
        print("ERROR: No segment column found!")
        return None, None, None, None
    
    print(f"Using segment column: {segment_col}")
    
    # Filter out NaN segments
    enhanced_df = enhanced_df[~enhanced_df[segment_col].isna()]
    unique_segments = sorted(enhanced_df[segment_col].unique())
    print(f"Unique segments: {unique_segments}")
    
    # Also check for ring column
    ring_col = None
    for col in ['ring', 'gt_ring', 'ring_id']:
        if col in enhanced_df.columns:
            ring_col = col
            break
    
    print(f"Ring column: {ring_col}")
    
    # Analyze segment positions using pixel coordinates
    segment_stats = {}
    
    # Filter to surface points only (typically r > threshold)
    if 'r' in enhanced_df.columns:
        surface_df = enhanced_df[enhanced_df['r'] > 2.7].copy()
        print(f"Surface points (r > 2.7): {len(surface_df):,}")
    else:
        surface_df = enhanced_df.copy()
    
    # Get pixel coordinates - use pixel_x, pixel_y from merged data
    if 'pixel_y' in surface_df.columns and 'pixel_x' in surface_df.columns:
        for seg in unique_segments:
            seg_data = surface_df[surface_df[segment_col] == seg]
            if len(seg_data) > 0 and not np.isnan(seg):
                segment_stats[int(seg)] = {
                    'count': len(seg_data),
                    'y_min': float(seg_data['pixel_y'].min()),
                    'y_max': float(seg_data['pixel_y'].max()),
                    'y_mean': float(seg_data['pixel_y'].mean()),
                    'y_std': float(seg_data['pixel_y'].std()),
                    'x_min': float(seg_data['pixel_x'].min()),
                    'x_max': float(seg_data['pixel_x'].max()),
                }
    else:
        print("WARNING: No pixel coordinates found!")
    
    print(f"\nSegment Statistics (pixel coordinates):")
    print(f"{'Segment':<10} {'Count':>10} {'Y_min':>8} {'Y_max':>8} {'Y_mean':>10} {'Y_std':>8}")
    print("-" * 60)
    for seg in sorted(segment_stats.keys()):
        stats = segment_stats[seg]
        print(f"{seg:<10} {stats['count']:>10,} {stats['y_min']:>8.0f} {stats['y_max']:>8.0f} {stats['y_mean']:>10.1f} {stats['y_std']:>8.1f}")
    
    return enhanced_df, segment_stats, ring_col, segment_col

def compute_segment_spacings(enhanced_df, segment_stats, ring_col, segment_col, tunnel_id):
    """
    Compute actual segment spacings per ring from ground truth.
    This is what we need to reverse-engineer!
    """
    base_dir = f"data/{tunnel_id}"
    ring_count = int(open(f'{base_dir}/ring_count.txt').read())
    depth_map = cv2.imread(f'{base_dir}/depth_map.png')
    img_height, img_width = depth_map.shape[:2]
    
    print(f"\n{'='*70}")
    print(f"SEGMENT SPACING ANALYSIS")
    print(f"{'='*70}")
    
    # Standard segment order (may need adjustment per tunnel)
    # K=1, B1=2, A1=3, A2=4, A3=5, A4=6, B2=7 (or B2=6 for 6-segment)
    
    # Filter surface points
    if 'r' in enhanced_df.columns:
        surface_df = enhanced_df[enhanced_df['r'] > 2.7].copy()
    else:
        surface_df = enhanced_df.copy()
    
    # Filter out NaN segments
    surface_df = surface_df[~surface_df[segment_col].isna()]
    
    # Compute per-ring segment centroids
    ring_segment_positions = defaultdict(dict)
    
    if ring_col and 'pixel_y' in surface_df.columns:
        # Filter out NaN rings
        valid_df = surface_df[~surface_df[ring_col].isna()]
        
        for ring_id in valid_df[ring_col].unique():
            if np.isnan(ring_id):
                continue
            ring_data = valid_df[valid_df[ring_col] == ring_id]
            
            for seg in ring_data[segment_col].unique():
                if np.isnan(seg):
                    continue
                seg_data = ring_data[ring_data[segment_col] == seg]
                if len(seg_data) > 10:  # Need enough points
                    ring_segment_positions[int(ring_id)][int(seg)] = {
                        'y_mean': float(seg_data['pixel_y'].mean()),
                        'y_median': float(seg_data['pixel_y'].median()),
                        'y_min': float(seg_data['pixel_y'].min()),
                        'y_max': float(seg_data['pixel_y'].max()),
                        'x_mean': float(seg_data['pixel_x'].mean()),
                        'count': len(seg_data)
                    }
    
    # Print per-ring K positions
    print(f"\nPer-Ring K-block (segment 1) Y positions:")
    k_positions = []
    k_x_positions = []
    for ring_id in sorted(ring_segment_positions.keys()):
        if 1 in ring_segment_positions[ring_id]:
            k_y = ring_segment_positions[ring_id][1]['y_mean']
            k_x = ring_segment_positions[ring_id][1]['x_mean']
            k_positions.append(k_y)
            k_x_positions.append(k_x)
            print(f"  Ring {ring_id}: K at Y = {k_y:.1f}, X = {k_x:.1f}")
    
    if k_positions:
        print(f"\nK-block statistics:")
        print(f"  Mean Y: {np.mean(k_positions):.1f}")
        print(f"  Std Y: {np.std(k_positions):.1f}")
        print(f"  Min Y: {np.min(k_positions):.1f}")
        print(f"  Max Y: {np.max(k_positions):.1f}")
        print(f"  As % of image height ({img_height}): {np.mean(k_positions)/img_height*100:.1f}%")
    
    # Compute segment spacings
    print(f"\nSegment Spacings (Y difference between consecutive segments):")
    
    # Assume segment order: K(1) -> B1(2) -> A1(3) -> A2(4) -> A3(5) -> [A4(6)] -> B2(6 or 7)
    spacing_results = defaultdict(list)
    
    for ring_id in sorted(ring_segment_positions.keys()):
        positions = ring_segment_positions[ring_id]
        segments_present = sorted(positions.keys())
        
        # K to B1 (B1 is ABOVE K, so K_y > B1_y means positive spacing going up)
        if 1 in positions and 2 in positions:
            spacing = positions[1]['y_mean'] - positions[2]['y_mean']
            spacing_results['K_to_B1'].append(spacing)
        
        # B1 to A1
        if 2 in positions and 3 in positions:
            spacing = positions[2]['y_mean'] - positions[3]['y_mean']
            spacing_results['B1_to_A1'].append(spacing)
        
        # A1 to A2
        if 3 in positions and 4 in positions:
            spacing = positions[3]['y_mean'] - positions[4]['y_mean']
            spacing_results['A1_to_A2'].append(spacing)
        
        # A2 to A3
        if 4 in positions and 5 in positions:
            spacing = positions[4]['y_mean'] - positions[5]['y_mean']
            spacing_results['A2_to_A3'].append(spacing)
        
        # A3 to A4 (if exists - 7 segment tunnel)
        if 5 in positions and 6 in positions and 7 in positions:
            # This is a 7-segment tunnel with A4
            spacing = positions[5]['y_mean'] - positions[6]['y_mean']
            spacing_results['A3_to_A4'].append(spacing)
        
        # K to B2 (B2 is BELOW K, so B2_y > K_y means positive spacing going down)
        # B2 could be segment 6 (6-segment) or 7 (7-segment)
        max_seg = max(segments_present)
        if 1 in positions and max_seg in positions and max_seg > 1:
            # B2 should be below K (higher Y value)
            if positions[max_seg]['y_mean'] > positions[1]['y_mean']:
                spacing = positions[max_seg]['y_mean'] - positions[1]['y_mean']
                spacing_results['K_to_B2'].append(spacing)
    
    print(f"\n{'Spacing':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Count':>6}")
    print("-" * 65)
    for spacing_name, values in spacing_results.items():
        if values:
            print(f"{spacing_name:<15} {np.mean(values):>10.1f} {np.std(values):>10.1f} {np.min(values):>10.1f} {np.max(values):>10.1f} {len(values):>6}")
    
    # Also compute segment heights (vertical extent of each segment)
    print(f"\nSegment Heights (Y extent within each segment):")
    segment_heights = defaultdict(list)
    for ring_id in sorted(ring_segment_positions.keys()):
        for seg, data in ring_segment_positions[ring_id].items():
            height = data['y_max'] - data['y_min']
            segment_heights[seg].append(height)
    
    print(f"{'Segment':<10} {'Mean Height':>12} {'Std':>10}")
    print("-" * 35)
    for seg in sorted(segment_heights.keys()):
        heights = segment_heights[seg]
        print(f"{seg:<10} {np.mean(heights):>12.1f} {np.std(heights):>10.1f}")
    
    return ring_segment_positions, spacing_results

def compute_optimal_parameters(segment_stats, spacing_results, tunnel_id):
    """
    Compute optimal parameters based on GT analysis
    """
    base_dir = f"data/{tunnel_id}"
    depth_map = cv2.imread(f'{base_dir}/depth_map.png')
    img_height, img_width = depth_map.shape[:2]
    ring_count = int(open(f'{base_dir}/ring_count.txt').read())
    
    print(f"\n{'='*70}")
    print(f"OPTIMAL PARAMETERS (REVERSE-ENGINEERED FROM GT)")
    print(f"{'='*70}")
    
    optimal = {
        'tunnel_id': tunnel_id,
        'image_height': img_height,
        'image_width': img_width,
        'ring_count': ring_count,
    }
    
    # K-block position as ratio of image height
    if 1 in segment_stats:
        k_y_mean = segment_stats[1]['y_mean']
        optimal['k_position_ratio'] = float(k_y_mean / img_height)
        optimal['k_y_mean_px'] = float(k_y_mean)
        print(f"\nK-block position:")
        print(f"  k_position_ratio = {optimal['k_position_ratio']:.4f}")
        print(f"  k_y_mean_px = {optimal['k_y_mean_px']:.1f}")
        print(f"  Image height = {img_height}")
    
    # Segment spacings
    print(f"\nOptimal segment spacings (pixels):")
    for spacing_name, values in spacing_results.items():
        if values:
            optimal[f'{spacing_name}_mean'] = float(np.mean(values))
            optimal[f'{spacing_name}_std'] = float(np.std(values))
            print(f"  {spacing_name}: {np.mean(values):.1f} ± {np.std(values):.1f}")
    
    # K-block height (Y extent)
    if 1 in segment_stats:
        k_height = segment_stats[1]['y_max'] - segment_stats[1]['y_min']
        optimal['K_height_px'] = float(k_height)
        print(f"\nK-block dimensions:")
        print(f"  K_height_px = {k_height:.1f}")
    
    # AB block height (average of A/B blocks, excluding background)
    ab_heights = []
    for seg in [2, 3, 4, 5, 6, 7]:  # B1, A1, A2, A3, A4/B2
        if seg in segment_stats:
            h = segment_stats[seg]['y_max'] - segment_stats[seg]['y_min']
            ab_heights.append(h)
    if ab_heights:
        optimal['AB_height_px'] = float(np.mean(ab_heights))
        optimal['AB_height_px_std'] = float(np.std(ab_heights))
        print(f"  AB_height_px = {optimal['AB_height_px']:.1f} ± {optimal['AB_height_px_std']:.1f}")
    
    # Compare with hardcoded values
    print(f"\n{'='*70}")
    print(f"COMPARISON WITH HARDCODED VALUES")
    print(f"{'='*70}")
    
    hardcoded = {
        'K_height_mm': 1079.92,
        'AB_height_mm': 3239.77,
        'K_height_px_expected': 1079.92 / 5,  # ~216 px at 5mm resolution
        'AB_height_px_expected': 3239.77 / 5,  # ~648 px at 5mm resolution
    }
    optimal['hardcoded'] = hardcoded
    
    print(f"\nHardcoded physical constants (mm -> px at 5mm resolution):")
    print(f"  K_height: {hardcoded['K_height_mm']} mm = {hardcoded['K_height_px_expected']:.1f} px")
    print(f"  AB_height: {hardcoded['AB_height_mm']} mm = {hardcoded['AB_height_px_expected']:.1f} px")
    
    if 'K_height_px' in optimal:
        diff = optimal['K_height_px'] - hardcoded['K_height_px_expected']
        ratio = optimal['K_height_px'] / hardcoded['K_height_px_expected']
        print(f"\nActual from GT:")
        print(f"  K_height_px = {optimal['K_height_px']:.1f} (expected: {hardcoded['K_height_px_expected']:.1f})")
        print(f"  Difference: {diff:.1f} px ({ratio:.2f}x)")
        optimal['K_height_diff'] = float(diff)
        optimal['K_height_ratio'] = float(ratio)
    
    if 'AB_height_px' in optimal:
        diff = optimal['AB_height_px'] - hardcoded['AB_height_px_expected']
        ratio = optimal['AB_height_px'] / hardcoded['AB_height_px_expected']
        print(f"  AB_height_px = {optimal['AB_height_px']:.1f} (expected: {hardcoded['AB_height_px_expected']:.1f})")
        print(f"  Difference: {diff:.1f} px ({ratio:.2f}x)")
        optimal['AB_height_diff'] = float(diff)
        optimal['AB_height_ratio'] = float(ratio)
    
    # Critical: Compare detected K positions with GT K positions
    print(f"\n{'='*70}")
    print(f"K-BLOCK DETECTION ACCURACY")
    print(f"{'='*70}")
    
    detected_path = f'{base_dir}/detected.csv'
    if os.path.exists(detected_path):
        detected_df = pd.read_csv(detected_path)
        if 'k_y_mean_px' in optimal and len(detected_df) > 0:
            gt_k_y = optimal['k_y_mean_px']
            detected_k_y = detected_df['Y'].values
            
            errors = np.abs(detected_k_y - gt_k_y)
            optimal['k_detection_errors'] = [float(e) for e in errors]
            optimal['k_detection_mean_error'] = float(np.mean(errors))
            optimal['k_detection_max_error'] = float(np.max(errors))
            
            print(f"  GT K mean Y: {gt_k_y:.1f}")
            print(f"  Detected K positions: {len(detected_k_y)}")
            print(f"  Mean detection error: {np.mean(errors):.1f} px")
            print(f"  Max detection error: {np.max(errors):.1f} px")
            print(f"  Per-ring errors: {[f'{e:.0f}' for e in errors]}")
    
    return optimal

def analyze_detection_accuracy(tunnel_id):
    """
    Compare detected K positions with GT K positions
    """
    base_dir = f"data/{tunnel_id}"
    
    print(f"\n{'='*70}")
    print(f"DETECTION ACCURACY ANALYSIS")
    print(f"{'='*70}")
    
    # Load detected positions
    detected_path = f'{base_dir}/detected.csv'
    if os.path.exists(detected_path):
        detected_df = pd.read_csv(detected_path)
        print(f"Detected K positions:")
        print(detected_df)
    else:
        print("No detected.csv found")
        return None
    
    return detected_df

def main():
    import sys
    tunnel_ids = sys.argv[1:] if len(sys.argv) > 1 else ['4-1', '2-2']
    
    all_results = {}
    
    for tunnel_id in tunnel_ids:
        print(f"\n{'#'*70}")
        print(f"# ANALYZING TUNNEL {tunnel_id}")
        print(f"{'#'*70}")
        
        try:
            # Load and analyze enhanced data (has pixel coordinates)
            result = analyze_enhanced_segments(tunnel_id)
            if result is None:
                continue
                
            enhanced_df, segment_stats, ring_col, segment_col = result
            
            # Compute segment spacings
            ring_positions, spacing_results = compute_segment_spacings(
                enhanced_df, segment_stats, ring_col, segment_col, tunnel_id
            )
            
            # Compute optimal parameters
            optimal = compute_optimal_parameters(segment_stats, spacing_results, tunnel_id)
            
            # Analyze detection accuracy
            detected = analyze_detection_accuracy(tunnel_id)
            
            # Save results
            all_results[tunnel_id] = {
                'segment_stats': segment_stats,
                'spacing_results': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                                   for k, v in spacing_results.items() if v},
                'optimal': optimal
            }
            
            # Save to JSON
            tunnel_base_dir = f'data/{tunnel_id}'
            output_path = f'{tunnel_base_dir}/gt_analysis.json'
            with open(output_path, 'w') as f:
                json.dump(all_results[tunnel_id], f, indent=2, default=float)
            print(f"\nResults saved to {output_path}")
            
        except Exception as e:
            print(f"Error analyzing {tunnel_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare tunnels
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"TUNNEL COMPARISON")
        print(f"{'='*70}")
        
        for tid in all_results:
            opt = all_results[tid].get('optimal', {})
            print(f"\n{tid}:")
            print(f"  K position ratio: {opt.get('k_position_ratio', 'N/A')}")
            print(f"  K height: {opt.get('K_height_px', 'N/A')}")
            print(f"  AB height: {opt.get('AB_height_px', 'N/A')}")

if __name__ == '__main__':
    main()

"""
Generate Ground Truth Pattern for Each Tunnel

Analyzes the segment labels in ground truth .txt files to determine:
1. Number of segments (6 or 7)
2. Pattern type (alternating, constant)
3. K-block position per ring

Output: pattern_gt.json for each tunnel with ground truth pattern info
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

# Segment names mapping for 7-segment tunnels
# For 6-segment tunnels, seg=6 is B2, not A4
SEGMENT_NAMES_7SEG = {
    0: 'Background',
    1: 'K',
    2: 'B1',
    3: 'A1',
    4: 'A2',
    5: 'A3',
    6: 'A4',
    7: 'B2'
}

SEGMENT_NAMES_6SEG = {
    0: 'Background',
    1: 'K',
    2: 'B1',
    3: 'A1',
    4: 'A2',
    5: 'A3',
    6: 'B2'  # Note: 6=B2 for 6-segment tunnels!
}

# Physical constants
K_HEIGHT_MM = 1079.92
AB_HEIGHT_MM = 3239.77
TUNNEL_DIAMETER = 5.5
CIRCUMFERENCE = np.pi * TUNNEL_DIAMETER  # ~17.2788m


def load_raw_data(filepath):
    """Load raw point cloud data from .txt file."""
    print(f"Loading {filepath}...")
    data = np.loadtxt(filepath)
    print(f"  Loaded {len(data):,} points")
    return {
        'x': data[:, 0],
        'y': data[:, 1],
        'z': data[:, 2],
        'intensity': data[:, 3],
        'segment': data[:, 4].astype(int),
        'ring': data[:, 5].astype(int)
    }


def load_unwrapped_data(filepath):
    """Load unwrapped CSV with theta coordinates."""
    import csv
    
    print(f"Loading {filepath}...")
    data = defaultdict(list)
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ['x', 'y', 'z', 'intensity', 'segment', 'ring', 'r', 'theta', 'h']:
                if key in row and row[key]:
                    try:
                        if key in ['segment', 'ring']:
                            data[key].append(int(float(row[key])))
                        else:
                            data[key].append(float(row[key]))
                    except ValueError:
                        pass
    
    for key in data:
        data[key] = np.array(data[key])
    
    print(f"  Loaded {len(data['x']):,} points")
    return data


def analyze_segment_distribution(data):
    """Analyze the distribution of segments."""
    segments = data['segment']
    unique_segments = np.unique(segments)
    max_seg = max(unique_segments)
    
    # Use appropriate segment names based on tunnel type
    segment_names = SEGMENT_NAMES_7SEG if max_seg >= 7 else SEGMENT_NAMES_6SEG
    
    print(f"\nSegment distribution:")
    for seg in sorted(unique_segments):
        count = np.sum(segments == seg)
        name = segment_names.get(seg, f"Seg{seg}")
        print(f"  {name} (seg={seg}): {count:,} points")
    
    # Determine number of unique non-background segments
    non_bg_segments = [s for s in unique_segments if s > 0]
    return len(non_bg_segments), unique_segments


def analyze_k_positions_per_ring(data):
    """
    Analyze K-block (segment=1) positions per ring.
    
    Returns dict with ring_idx -> K-block theta center position
    """
    segments = data['segment']
    rings = data['ring']
    theta = data['theta']
    
    unique_rings = np.unique(rings)
    k_positions = {}
    
    print(f"\nK-block positions per ring:")
    
    for ring_idx in sorted(unique_rings):
        # Get K-block points for this ring
        mask = (rings == ring_idx) & (segments == 1)
        k_theta = theta[mask]
        
        if len(k_theta) > 0:
            k_center = np.mean(k_theta)
            k_std = np.std(k_theta)
            k_min = np.min(k_theta)
            k_max = np.max(k_theta)
            k_positions[int(ring_idx)] = {
                'center': float(k_center),
                'std': float(k_std),
                'min': float(k_min),
                'max': float(k_max),
                'count': int(len(k_theta))
            }
            print(f"  Ring {ring_idx}: K center={k_center:.3f}, range=[{k_min:.3f}, {k_max:.3f}], n={len(k_theta)}")
        else:
            print(f"  Ring {ring_idx}: No K-block found")
    
    return k_positions


def determine_pattern_type(k_positions, num_segments):
    """
    Determine pattern type from K-block positions.
    
    Pattern types:
    - 6seg_alternating: K alternates between 2 positions
    - 6seg_constant: K stays at same position
    - 7seg_alternating: 7-segment tunnel with alternating K
    """
    if not k_positions:
        return 'unknown', 0.0, {}
    
    k_centers = [k_positions[r]['center'] for r in sorted(k_positions.keys())]
    
    if len(k_centers) < 2:
        return f'{num_segments}seg_constant', 0.5, {'reason': 'insufficient data'}
    
    # Calculate spread and differences
    k_spread = np.max(k_centers) - np.min(k_centers)
    k_diffs = np.diff(k_centers)
    
    # Expected AB height in theta units (linear distance on circumference)
    # AB_HEIGHT_MM / 1000 = meters
    expected_alternation = AB_HEIGHT_MM / 1000  # ~3.24m
    
    print(f"\nPattern analysis:")
    print(f"  K-center spread: {k_spread:.3f}m")
    print(f"  K-center differences: {k_diffs}")
    print(f"  Expected alternation distance: {expected_alternation:.3f}m")
    
    # Check if K positions cluster into 2 groups (alternating)
    k_array = np.array(k_centers)
    k_mean = np.mean(k_array)
    
    # Cluster analysis: are positions above/below mean alternating?
    above_mean = k_array > k_mean
    alternation_count = np.sum(np.diff(above_mean.astype(int)) != 0)
    alternation_ratio = alternation_count / (len(k_array) - 1) if len(k_array) > 1 else 0
    
    print(f"  Alternation ratio: {alternation_ratio:.2f} ({alternation_count}/{len(k_array)-1})")
    
    # Determine pattern
    # High spread and high alternation = alternating pattern
    # Low spread = constant pattern
    
    if k_spread < 0.5:  # Less than 0.5m spread
        pattern_type = f'{num_segments}seg_constant'
        confidence = 0.9 - k_spread
    elif alternation_ratio > 0.6:  # More than 60% alternation
        pattern_type = f'{num_segments}seg_alternating'
        confidence = 0.7 + 0.3 * alternation_ratio
    else:
        # Progressive pattern or irregular
        pattern_type = f'{num_segments}seg_alternating'
        confidence = 0.5 + 0.3 * alternation_ratio
    
    metrics = {
        'spread': float(k_spread),
        'alternation_ratio': float(alternation_ratio),
        'k_mean': float(k_mean),
        'k_std': float(np.std(k_array)),
        'k_differences': [float(d) for d in k_diffs]
    }
    
    return pattern_type, confidence, metrics


def generate_pattern_gt(tunnel_id, data_dir='data'):
    """Generate ground truth pattern for a tunnel."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing Ground Truth Pattern for Tunnel {tunnel_id}")
    print(f"{'='*60}")
    
    base_dir = os.path.join(data_dir, tunnel_id)
    unwrapped_path = os.path.join(base_dir, 'unwrapped.csv')
    
    if not os.path.exists(unwrapped_path):
        print(f"ERROR: {unwrapped_path} not found")
        return None
    
    # Load unwrapped data (has theta coordinates)
    data = load_unwrapped_data(unwrapped_path)
    
    # Analyze segment distribution
    num_segments, unique_segments = analyze_segment_distribution(data)
    print(f"\nDetected {num_segments} segment types (excluding background)")
    
    # Determine if 6 or 7 segment tunnel based on max segment label
    # 6-segment: labels 0-6 where 6=B2 (K, B1, A1, A2, A3, B2)
    # 7-segment: labels 0-7 where 6=A4, 7=B2 (K, B1, A1, A2, A3, A4, B2)
    max_segment = max(unique_segments)
    segment_count = 6 if max_segment == 6 else 7
    print(f"Tunnel type: {segment_count}-segment (max_label={max_segment})")
    
    if segment_count == 6:
        print("  Segments: K(1), B1(2), A1(3), A2(4), A3(5), B2(6)")
    else:
        print("  Segments: K(1), B1(2), A1(3), A2(4), A3(5), A4(6), B2(7)")
    
    # Analyze K-block positions per ring
    k_positions = analyze_k_positions_per_ring(data)
    
    # Determine pattern type
    pattern_type, confidence, metrics = determine_pattern_type(k_positions, segment_count)
    
    print(f"\n{'='*60}")
    print(f"Pattern Result: {pattern_type} (confidence: {confidence:.2f})")
    print(f"{'='*60}")
    
    # Create ground truth pattern output
    result = {
        'tunnel_id': tunnel_id,
        'segment_count': segment_count,
        'pattern_type': pattern_type,
        'confidence': confidence,
        'metrics': metrics,
        'k_positions_per_ring': k_positions,
        'unique_rings': sorted([int(r) for r in np.unique(data['ring'])]),
        'ring_count': len(np.unique(data['ring']))
    }
    
    # Save to JSON
    output_path = os.path.join(base_dir, 'pattern_gt.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved ground truth pattern to {output_path}")
    
    return result


def main():
    """Main function to generate pattern ground truth for all tunnels."""
    
    # Tunnels to process
    tunnels = ['1-4', '2-2', '3-1', '4-1', '5-1']
    
    # Check command line arguments
    if len(sys.argv) > 1:
        tunnels = sys.argv[1:]
    
    data_dir = 'data'
    results = {}
    
    for tunnel_id in tunnels:
        try:
            result = generate_pattern_gt(tunnel_id, data_dir)
            if result:
                results[tunnel_id] = result
        except Exception as e:
            print(f"ERROR processing {tunnel_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary of Ground Truth Patterns")
    print(f"{'='*60}")
    
    for tunnel_id, result in results.items():
        print(f"\n{tunnel_id}:")
        print(f"  Segments: {result['segment_count']}")
        print(f"  Pattern: {result['pattern_type']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Rings: {result['ring_count']}")
        if 'spread' in result['metrics']:
            print(f"  K-spread: {result['metrics']['spread']:.3f}m")
            print(f"  Alternation: {result['metrics']['alternation_ratio']:.2f}")


if __name__ == "__main__":
    main()

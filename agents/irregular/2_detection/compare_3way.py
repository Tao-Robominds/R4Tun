#!/usr/bin/env python3
"""
3-way comparison: DBSCAN vs Groove-Pair vs Combined K detection methods.
Compares all_segments_{method}.csv files against all_segments_gt.csv.
"""

import os
import sys
import pandas as pd
import numpy as np


def wrap_aware_distance(y1, y2, img_height):
    """Compute wrap-aware Y distance on cylindrical depth map."""
    d = abs(y1 - y2)
    return min(d, img_height - d)


def compare_segments(detected_path, gt_path, img_height):
    """Compare detected segments against GT."""
    if not os.path.exists(detected_path):
        return None
    
    detected = pd.read_csv(detected_path)
    gt = pd.read_csv(gt_path)
    
    # Normalize ring indices: detected uses 0-6, GT uses 119-125
    # Match by Block and relative position (ring index within each dataset)
    detected_sorted = detected.sort_values(['Ring', 'Block']).reset_index(drop=True)
    gt_sorted = gt.sort_values(['Ring', 'Block']).reset_index(drop=True)
    
    # Create relative ring index for matching
    detected_sorted['rel_ring'] = detected_sorted.groupby('Ring').ngroup()
    gt_sorted['rel_ring'] = gt_sorted.groupby('Ring').ngroup()
    
    # Merge on relative ring and Block
    merged = detected_sorted.merge(gt_sorted, on=['rel_ring', 'Block'], suffixes=('_det', '_gt'))
    
    if len(merged) == 0:
        return None
    
    # Compute distances
    distances = []
    k_distances = []
    per_ring_k = {}
    
    for _, row in merged.iterrows():
        dist = wrap_aware_distance(row['Y_det'], row['Y_gt'], img_height)
        distances.append(dist)
        if row['Block'] == 'K':
            k_distances.append(dist)
            ring = row['rel_ring']  # Use relative ring index
            if ring not in per_ring_k:
                per_ring_k[ring] = []
            per_ring_k[ring].append(dist)
    
    return {
        'mean_all_segment_dist': np.mean(distances),
        'std_all_segment_dist': np.std(distances),
        'mean_k_dy': np.mean(k_distances) if k_distances else None,
        'std_k_dy': np.std(k_distances) if k_distances else None,
        'worst_ring_k_dy': max(k_distances) if k_distances else None,
        'per_ring_k_dy': {ring: np.mean(dists) for ring, dists in per_ring_k.items()},
        'n_segments_matched': len(merged),
        'n_k_matched': len(k_distances),
    }


def main():
    tunnel_id = "4-1"
    base_dir = "data"
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load image dimensions
    depth_map_outlier = np.load(os.path.join(tunnel_dir, "depth_map_outlier.npy"))
    L, W = depth_map_outlier.shape
    
    # Load GT
    gt_path = os.path.join(tunnel_dir, "all_segments_gt.csv")
    if not os.path.exists(gt_path):
        print(f"ERROR: GT file not found: {gt_path}")
        return
    
    methods = [
        ('complex_staggered', 'all_segments_dbscan.csv'),
        ('groove_pair', 'all_segments_groove_pair.csv'),
        ('combined', 'all_segments_combined.csv'),
    ]
    
    results = {}
    
    for method, filename in methods:
        detected_path = os.path.join(tunnel_dir, filename)
        comparison = compare_segments(detected_path, gt_path, L)
        if comparison:
            results[method] = comparison
        else:
            print(f"WARNING: {filename} not found or empty")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("3-Way Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Mean All-Seg Dist':<20} {'Mean K |dY|':<15} {'Worst Ring K |dY|':<20}")
    print("-" * 80)
    for method, comp in results.items():
        mean_all = f"{comp['mean_all_segment_dist']:.1f}" if comp['mean_all_segment_dist'] else "N/A"
        mean_k = f"{comp['mean_k_dy']:.1f}" if comp['mean_k_dy'] else "N/A"
        worst_k = f"{comp['worst_ring_k_dy']:.1f}" if comp['worst_ring_k_dy'] else "N/A"
        print(f"{method:<20} {mean_all:<20} {mean_k:<15} {worst_k:<20}")
    
    # Per-ring K comparison
    if results:
        print(f"\n{'='*80}")
        print("Per-Ring K |dY| Comparison")
        print(f"{'='*80}")
        print(f"{'Ring':<8}", end="")
        for method, _ in methods:
            if method in results:
                print(f"{method:<20}", end="")
        print()
        print("-" * 80)
        
        all_rings = set()
        for comp in results.values():
            all_rings.update(comp.get('per_ring_k_dy', {}).keys())
        
        for ring in sorted(all_rings):
            print(f"{ring:<8}", end="")
            for method, _ in methods:
                if method in results:
                    per_ring = results[method].get('per_ring_k_dy', {})
                    val = per_ring.get(ring, None)
                    print(f"{val:.1f}" if val is not None else "N/A", end="")
                    print(" " * (20 - len(f"{val:.1f}" if val is not None else "N/A")), end="")
            print()


if __name__ == "__main__":
    main()

"""
Visualize the actual ground truth segment patterns
to understand why the current approach fails
"""

import numpy as np
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt
import os

def visualize_gt_pattern(tunnel_id):
    """Visualize GT segment positions overlaid on depth map"""
    base_dir = f"data/{tunnel_id}"
    
    # Load data
    enhanced_df = pd.read_csv(f'{base_dir}/enhanced.csv')
    pixel_to_point = pickle.load(open(f'{base_dir}/pixel_to_point.pkl', 'rb'))
    depth_map = cv2.imread(f'{base_dir}/depth_map.png')
    
    # Merge pixel coordinates
    ptp_df = pd.DataFrame(pixel_to_point)
    enhanced_df = enhanced_df.reset_index()
    enhanced_df['point_idx'] = enhanced_df.index
    merged = ptp_df.merge(enhanced_df, left_on='index', right_on='point_idx', how='inner')
    
    # Filter surface points
    surface_df = merged[merged['r'] > 2.7].copy()
    surface_df = surface_df[~surface_df['segment'].isna()]
    
    img_height, img_width = depth_map.shape[:2]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Depth map with K-block positions highlighted
    ax1 = axes[0, 0]
    ax1.imshow(depth_map, cmap='gray')
    
    # Plot K-block (segment 1) points
    k_points = surface_df[surface_df['segment'] == 1]
    if len(k_points) > 0:
        ax1.scatter(k_points['pixel_x'], k_points['pixel_y'], 
                   c='red', s=1, alpha=0.5, label='K-block (GT)')
    ax1.set_title(f'{tunnel_id}: K-block positions (GT)')
    ax1.legend()
    
    # 2. All segments colored by label
    ax2 = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    segment_names = {0: 'BG', 1: 'K', 2: 'B1', 3: 'A1', 4: 'A2', 5: 'A3', 6: 'A4/B2', 7: 'B2'}
    
    for seg in sorted(surface_df['segment'].unique()):
        if np.isnan(seg):
            continue
        seg_points = surface_df[surface_df['segment'] == seg]
        ax2.scatter(seg_points['pixel_x'], seg_points['pixel_y'],
                   c=[colors[int(seg) % 8]], s=0.5, alpha=0.3,
                   label=f'{segment_names.get(int(seg), str(int(seg)))} ({len(seg_points):,})')
    ax2.set_xlim(0, img_width)
    ax2.set_ylim(img_height, 0)  # Flip Y
    ax2.set_title(f'{tunnel_id}: All segments colored by label')
    ax2.legend(loc='upper right', fontsize=8)
    
    # 3. K-block Y position per ring
    ax3 = axes[1, 0]
    ring_col = 'ring'
    k_by_ring = surface_df[surface_df['segment'] == 1].groupby(ring_col).agg({
        'pixel_y': 'mean',
        'pixel_x': 'mean'
    }).reset_index()
    
    ax3.scatter(k_by_ring[ring_col], k_by_ring['pixel_y'], s=100, c='red')
    ax3.axhline(y=k_by_ring['pixel_y'].mean(), color='blue', linestyle='--', label=f'Mean: {k_by_ring["pixel_y"].mean():.0f}')
    ax3.set_xlabel('Ring')
    ax3.set_ylabel('K-block Y position (pixels)')
    ax3.set_title(f'{tunnel_id}: K-block Y position per ring')
    ax3.legend()
    ax3.grid(True)
    
    # Add detected K positions if available
    detected_path = f'{base_dir}/detected.csv'
    if os.path.exists(detected_path):
        detected_df = pd.read_csv(detected_path)
        ax3.scatter(range(len(detected_df)), detected_df['Y'], s=50, c='green', marker='x', label='Detected')
        ax3.legend()
    
    # 4. Segment Y distribution histogram
    ax4 = axes[1, 1]
    for seg in [1, 2, 3, 4, 5, 6]:  # Skip background
        seg_points = surface_df[surface_df['segment'] == seg]
        if len(seg_points) > 0:
            ax4.hist(seg_points['pixel_y'], bins=50, alpha=0.5, 
                    label=f'{segment_names.get(seg, str(seg))}')
    ax4.set_xlabel('Y position (pixels)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'{tunnel_id}: Segment Y distributions')
    ax4.legend()
    
    plt.tight_layout()
    output_path = f'{base_dir}/gt_pattern_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    # Print per-ring segment centroid table
    print(f"\n{'='*80}")
    print(f"PER-RING SEGMENT CENTROIDS FOR {tunnel_id}")
    print(f"{'='*80}")
    
    centroids = surface_df.groupby([ring_col, 'segment']).agg({
        'pixel_y': 'mean',
        'pixel_x': 'mean'
    }).reset_index()
    
    # Pivot to show segments as columns
    pivot = centroids.pivot(index=ring_col, columns='segment', values='pixel_y')
    print("\nY positions by Ring and Segment:")
    print(pivot.to_string())
    
    # Calculate actual spacings per ring
    print(f"\n{'='*80}")
    print(f"ACTUAL SEGMENT SPACINGS PER RING")
    print(f"{'='*80}")
    
    for ring_id in sorted(pivot.index):
        row = pivot.loc[ring_id]
        print(f"\nRing {ring_id}:")
        
        # K position
        k_y = row.get(1.0, np.nan)
        if not np.isnan(k_y):
            print(f"  K(1): Y={k_y:.0f}")
        
        # Spacings
        if not np.isnan(row.get(1.0, np.nan)) and not np.isnan(row.get(2.0, np.nan)):
            print(f"  K→B1: {row[1.0] - row[2.0]:.0f} px")
        if not np.isnan(row.get(2.0, np.nan)) and not np.isnan(row.get(3.0, np.nan)):
            print(f"  B1→A1: {row[2.0] - row[3.0]:.0f} px")
        
        # Check for wraparound
        max_seg = max([s for s in row.index if not np.isnan(row[s])])
        if not np.isnan(row.get(1.0, np.nan)) and max_seg > 1:
            b2_y = row[max_seg]
            if b2_y > k_y:  # B2 below K
                print(f"  K→B2({int(max_seg)}): {b2_y - k_y:.0f} px (below)")
            else:
                print(f"  K→B2({int(max_seg)}): {k_y - b2_y:.0f} px (WRAPPED ABOVE)")

if __name__ == '__main__':
    import sys
    tunnels = sys.argv[1:] if len(sys.argv) > 1 else ['4-1', '2-2']
    for tid in tunnels:
        try:
            visualize_gt_pattern(tid)
        except Exception as e:
            print(f"Error processing {tid}: {e}")
            import traceback
            traceback.print_exc()

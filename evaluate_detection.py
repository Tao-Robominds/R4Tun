#!/usr/bin/env python3
"""
Evaluate detection results against ground truth K positions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_detection(tunnel_id: str, base_dir: str = "data"):
    """Evaluate detected.csv against detected_gt.csv."""
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    
    # Load files
    detected_path = os.path.join(tunnel_dir, 'detected.csv')
    gt_path = os.path.join(tunnel_dir, 'detected_gt.csv')
    
    if not os.path.exists(detected_path):
        print(f"Error: {detected_path} not found")
        return
    
    if not os.path.exists(gt_path):
        print(f"Error: {gt_path} not found")
        return
    
    detected = pd.read_csv(detected_path)
    gt = pd.read_csv(gt_path)
    
    # Sort by X
    detected_sorted = detected.sort_values('X').reset_index(drop=True)
    gt_sorted = gt.sort_values('X').reset_index(drop=True)
    
    print("=" * 70)
    print(f"Detection Evaluation for Tunnel: {tunnel_id}")
    print("=" * 70)
    print(f"\nGround Truth: {len(gt_sorted)} positions")
    print(f"Detected: {len(detected_sorted)} positions")
    
    # Calculate position errors
    n_gt = len(gt_sorted)
    n_det = len(detected_sorted)
    
    # Count penalty
    count_penalty = abs(n_det - n_gt) * 50
    print(f"\nCount penalty: {count_penalty:.1f} pixels ({abs(n_det - n_gt)} extra/missing)")
    
    # Calculate X and Y errors
    if n_det >= n_gt:
        # More detected than GT - match each GT to nearest detected
        total_error = 0
        matches = []
        used_det = set()
        for i in range(n_gt):
            gt_x, gt_y = gt_sorted.iloc[i]['X'], gt_sorted.iloc[i]['Y']
            
            # Find nearest detected
            distances = np.sqrt(
                (detected_sorted['X'] - gt_x)**2 + 
                (detected_sorted['Y'] - gt_y)**2
            )
            min_idx = distances.idxmin()
            min_dist = distances[min_idx]
            total_error += min_dist
            matches.append({
                'GT_idx': i,
                'Det_idx': min_idx,
                'GT_X': gt_x,
                'GT_Y': gt_y,
                'Det_X': detected_sorted.iloc[min_idx]['X'],
                'Det_Y': detected_sorted.iloc[min_idx]['Y'],
                'Error': min_dist
            })
            used_det.add(min_idx)
    else:
        # Fewer detected than GT - match each detected to nearest GT
        total_error = 0
        matches = []
        used_gt = set()
        for i in range(n_det):
            det_x, det_y = detected_sorted.iloc[i]['X'], detected_sorted.iloc[i]['Y']
            
            distances = np.sqrt(
                (gt_sorted['X'] - det_x)**2 + 
                (gt_sorted['Y'] - det_y)**2
            )
            min_idx = distances.idxmin()
            min_dist = distances.min()
            total_error += min_dist
            matches.append({
                'GT_idx': min_idx,
                'Det_idx': i,
                'GT_X': gt_sorted.iloc[min_idx]['X'],
                'GT_Y': gt_sorted.iloc[min_idx]['Y'],
                'Det_X': det_x,
                'Det_Y': det_y,
                'Error': min_dist
            })
            used_gt.add(min_idx)
        
        # Add penalty for missing detections
        missing_count = n_gt - n_det
        total_error += missing_count * 100
        print(f"Missing detection penalty: {missing_count * 100:.1f} pixels ({missing_count} missing)")
    
    # Average error
    avg_error = total_error / max(n_gt, n_det)
    
    # Add count penalty
    total_error_with_penalty = avg_error + count_penalty
    
    # Convert to score (higher is better)
    max_error = 1000 if tunnel_id == '4-1' else 500
    score = max(0, 1 - total_error_with_penalty / max_error)
    
    print(f"\n{'=' * 70}")
    print("METRICS")
    print(f"{'=' * 70}")
    print(f"Average position error: {avg_error:.2f} pixels")
    print(f"Total error (with penalties): {total_error_with_penalty:.2f} pixels")
    print(f"Score: {score:.4f} (higher is better, max=1.0)")
    
    # Detailed match table
    print(f"\n{'=' * 70}")
    print("DETAILED MATCHES")
    print(f"{'=' * 70}")
    matches_df = pd.DataFrame(matches)
    if len(matches_df) > 0:
        print(f"\n{'GT_idx':<8} {'GT_X':<12} {'GT_Y':<12} {'Det_X':<12} {'Det_Y':<12} {'Error':<10} {'Type':<15}")
        print("-" * 90)
        for _, m in matches_df.iterrows():
            det_idx = int(m['Det_idx'])
            det_type = detected_sorted.iloc[det_idx].get('Type', 'N/A') if 'Type' in detected_sorted.columns else 'N/A'
            print(f"{int(m['GT_idx']):<8} {m['GT_X']:<12.2f} {m['GT_Y']:<12.2f} {m['Det_X']:<12.2f} {m['Det_Y']:<12.2f} {m['Error']:<10.2f} {det_type:<15}")
        
        # Statistics
        print(f"\n{'=' * 70}")
        print("ERROR STATISTICS")
        print(f"{'=' * 70}")
        print(f"Min error: {matches_df['Error'].min():.2f} pixels")
        print(f"Max error: {matches_df['Error'].max():.2f} pixels")
        print(f"Mean error: {matches_df['Error'].mean():.2f} pixels")
        print(f"Median error: {matches_df['Error'].median():.2f} pixels")
        print(f"Std error: {matches_df['Error'].std():.2f} pixels")
        
        # X and Y error breakdown
        matches_df['X_error'] = np.abs(matches_df['Det_X'] - matches_df['GT_X'])
        matches_df['Y_error'] = np.abs(matches_df['Det_Y'] - matches_df['GT_Y'])
        print(f"\nX error - Mean: {matches_df['X_error'].mean():.2f}, Std: {matches_df['X_error'].std():.2f}")
        print(f"Y error - Mean: {matches_df['Y_error'].mean():.2f}, Std: {matches_df['Y_error'].std():.2f}")
    
    # Visualization
    output_path = os.path.join(tunnel_dir, 'detection_evaluation.png')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: X-Y scatter
    ax1.scatter(gt_sorted['X'], gt_sorted['Y'], c='green', marker='o', s=100, label='Ground Truth', alpha=0.7)
    ax1.scatter(detected_sorted['X'], detected_sorted['Y'], c='red', marker='x', s=100, label='Detected', alpha=0.7)
    
    # Draw lines connecting matches
    if len(matches_df) > 0:
        for _, m in matches_df.iterrows():
            ax1.plot([m['GT_X'], m['Det_X']], [m['GT_Y'], m['Det_Y']], 
                    'b--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_title('K Position Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert Y axis to match image coordinates
    
    # Plot 2: Error distribution
    if len(matches_df) > 0:
        ax2.hist(matches_df['Error'], bins=min(10, len(matches_df)), edgecolor='black', alpha=0.7)
        ax2.axvline(matches_df['Error'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {matches_df["Error"].mean():.2f}')
        ax2.set_xlabel('Position Error (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    return {
        'score': score,
        'avg_error': avg_error,
        'total_error': total_error_with_penalty,
        'count_penalty': count_penalty,
        'matches': matches_df
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_detection.py <tunnel_id>")
        print("Example: python evaluate_detection.py 4-1")
        sys.exit(1)
    
    tunnel_id = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    
    evaluate_detection(tunnel_id, base_dir)

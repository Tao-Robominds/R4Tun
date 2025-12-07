#!/usr/bin/env python3
"""
Denoising Quality Evaluation Module

Computes intrinsic metrics to evaluate the quality of the denoising process.
These metrics compare input (unwrapped) vs output (denoised) point clouds.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from scipy.spatial import cKDTree
from scipy.stats import entropy

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python evaluate_denoising.py <tunnel_id>")
    print("Example: python evaluate_denoising.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]

# Determine base directory
if os.path.exists(f"data/{tunnel_id}/denoised.csv"):
    base_dir = "data/"
else:
    base_dir = "../data/"

unwrapped_csv = os.path.join(base_dir, f"{tunnel_id}/unwrapped.csv")
denoised_csv = os.path.join(base_dir, f"{tunnel_id}/denoised.csv")
output_dir = os.path.join(base_dir, f"{tunnel_id}/evaluation")
os.makedirs(output_dir, exist_ok=True)

print(f"=== Denoising Quality Evaluation for Tunnel: {tunnel_id} ===\n")

# Load data
if not os.path.exists(unwrapped_csv):
    print(f"❌ Error: unwrapped.csv not found at {unwrapped_csv}")
    sys.exit(1)
if not os.path.exists(denoised_csv):
    print(f"❌ Error: denoised.csv not found at {denoised_csv}")
    sys.exit(1)

df_unwrapped = pd.read_csv(unwrapped_csv)
df_denoised = pd.read_csv(denoised_csv)

print(f"✓ Loaded {len(df_unwrapped):,} points from unwrapped.csv")
print(f"✓ Loaded {len(df_denoised):,} points from denoised.csv")

# Check required columns
required_cols = ['r', 'theta', 'h']
missing_cols = [col for col in required_cols if col not in df_unwrapped.columns]
if missing_cols:
    print(f"❌ Error: Missing required columns in unwrapped.csv: {missing_cols}")
    sys.exit(1)

# Ensure pred column exists
if 'pred' not in df_denoised.columns:
    df_denoised['pred'] = 7  # Assume all valid if not present

# ============================================================================
# 1. NOISE REMOVAL METRICS
# ============================================================================

print("\n[1/5] Computing Noise Removal Metrics...")

total_input = len(df_unwrapped)
total_output = len(df_denoised)

# Count valid and noise points
if 'pred' in df_denoised.columns:
    valid_points = (df_denoised['pred'] == 7).sum()
    noise_points = (df_denoised['pred'] == 0).sum()
else:
    valid_points = total_output
    noise_points = 0

noise_removal_rate = noise_points / total_input if total_input > 0 else 0
data_retention_rate = valid_points / total_input if total_input > 0 else 0
points_removed = total_input - valid_points

noise_metrics = {
    'total_input_points': int(total_input),
    'total_output_points': int(total_output),
    'valid_points_remaining': int(valid_points),
    'noise_points_removed': int(noise_points),
    'noise_removal_rate': float(noise_removal_rate),
    'data_retention_rate': float(data_retention_rate),
    'points_removed': int(points_removed),
    'interpretation': 'Higher retention rate (while removing noise) indicates better denoising'
}

# ============================================================================
# 2. POINT DENSITY ANALYSIS
# ============================================================================

print("[2/5] Computing Point Density Metrics...")

def compute_density_stats(df, sample_size=10000):
    """Compute nearest neighbor distance statistics"""
    valid_df = df[df['pred'] == 7] if 'pred' in df.columns else df
    
    if len(valid_df) < 100:
        return {
            'mean_nn_distance': 0.1,
            'median_nn_distance': 0.1,
            'std_nn_distance': 0.05,
            'cv_nn_distance': 0.5
        }
    
    sample_size = min(sample_size, len(valid_df))
    sample_indices = np.random.choice(len(valid_df), sample_size, replace=False)
    sample_coords = valid_df.iloc[sample_indices][['h', 'theta', 'r']].values
    
    tree = cKDTree(sample_coords)
    distances, _ = tree.query(sample_coords, k=2)
    nn_distances = distances[:, 1]
    
    return {
        'mean_nn_distance': float(np.mean(nn_distances)),
        'median_nn_distance': float(np.median(nn_distances)),
        'std_nn_distance': float(np.std(nn_distances)),
        'cv_nn_distance': float(np.std(nn_distances) / np.mean(nn_distances)) if np.mean(nn_distances) > 0 else 0
    }

density_before = compute_density_stats(df_unwrapped)
density_after = compute_density_stats(df_denoised)

density_improvement = {
    'before': density_before,
    'after': density_after,
    'improvement_ratio': density_before['median_nn_distance'] / density_after['median_nn_distance'] if density_after['median_nn_distance'] > 0 else 1.0,
    'interpretation': 'Lower median NN distance after denoising indicates better density (noise removal)'
}

# ============================================================================
# 3. SPATIAL COVERAGE ANALYSIS
# ============================================================================

print("[3/5] Computing Spatial Coverage Metrics...")

def compute_coverage(df, h_bins=20, theta_bins=36):
    """Compute 2D coverage in (h, theta) space"""
    valid_df = df[df['pred'] == 7] if 'pred' in df.columns else df
    
    if len(valid_df) == 0:
        return {
            'coverage_percentage': 0.0,
            'coverage_uniformity': 0.0,
            'sparse_areas_percentage': 100.0
        }
    
    h_range = [valid_df['h'].min(), valid_df['h'].max()]
    theta_range = [valid_df['theta'].min(), valid_df['theta'].max()]
    
    h_bins_edges = np.linspace(h_range[0], h_range[1], h_bins + 1)
    theta_bins_edges = np.linspace(theta_range[0], theta_range[1], theta_bins + 1)
    
    coverage_matrix = np.zeros((h_bins, theta_bins))
    for i in range(h_bins):
        for j in range(theta_bins):
            mask = ((valid_df['h'] >= h_bins_edges[i]) & (valid_df['h'] < h_bins_edges[i+1]) &
                   (valid_df['theta'] >= theta_bins_edges[j]) & (valid_df['theta'] < theta_bins_edges[j+1]))
            coverage_matrix[i, j] = mask.sum()
    
    non_zero_cells = np.count_nonzero(coverage_matrix)
    total_cells = coverage_matrix.size
    coverage_percentage = (non_zero_cells / total_cells) * 100
    
    # Coverage uniformity (inverse of CV of non-zero densities)
    non_zero_densities = coverage_matrix[coverage_matrix > 0]
    if len(non_zero_densities) > 0:
        coverage_uniformity = 1 / (1 + np.std(non_zero_densities) / np.mean(non_zero_densities))
        sparse_threshold = np.percentile(non_zero_densities, 25)
        sparse_areas = np.sum(coverage_matrix < sparse_threshold)
        sparse_areas_percentage = (sparse_areas / total_cells) * 100
    else:
        coverage_uniformity = 0.0
        sparse_areas_percentage = 100.0
    
    return {
        'coverage_percentage': float(coverage_percentage),
        'coverage_uniformity': float(coverage_uniformity),
        'sparse_areas_percentage': float(sparse_areas_percentage)
    }

coverage_before = compute_coverage(df_unwrapped)
coverage_after = compute_coverage(df_denoised)

coverage_metrics = {
    'before': coverage_before,
    'after': coverage_after,
    'coverage_preserved': coverage_after['coverage_percentage'] / coverage_before['coverage_percentage'] if coverage_before['coverage_percentage'] > 0 else 0.0,
    'uniformity_improvement': coverage_after['coverage_uniformity'] - coverage_before['coverage_uniformity'],
    'interpretation': 'Higher coverage preserved and uniformity indicates better denoising quality'
}

# ============================================================================
# 4. RADIUS CONSISTENCY ANALYSIS
# ============================================================================

print("[4/5] Computing Radius Consistency Metrics...")

def compute_radius_stats(df):
    """Compute radius distribution statistics"""
    valid_df = df[df['pred'] == 7] if 'pred' in df.columns else df
    
    if len(valid_df) == 0 or 'r' not in valid_df.columns:
        return {
            'mean': 0.0,
            'std': 0.0,
            'cv': 0.0,
            'range': 0.0
        }
    
    r_values = valid_df['r'].values
    return {
        'mean': float(np.mean(r_values)),
        'std': float(np.std(r_values)),
        'cv': float(np.std(r_values) / np.mean(r_values)) if np.mean(r_values) > 0 else 0.0,
        'range': float(np.max(r_values) - np.min(r_values))
    }

radius_before = compute_radius_stats(df_unwrapped)
radius_after = compute_radius_stats(df_denoised)

radius_metrics = {
    'before': radius_before,
    'after': radius_after,
    'cv_improvement': radius_before['cv'] - radius_after['cv'],
    'interpretation': 'Lower CV after denoising indicates more consistent radius (better noise removal)'
}

# ============================================================================
# 5. COMPOSITE QUALITY SCORE
# ============================================================================

print("[5/5] Computing Composite Quality Score...")

# Normalize metrics to 0-1 scale (higher is better)
def normalize_metric(value, lower_bound, upper_bound, invert=False):
    """Normalize a metric to 0-1 scale"""
    if lower_bound == upper_bound:
        return 0.5
    normalized = (value - lower_bound) / (upper_bound - lower_bound)
    normalized = np.clip(normalized, 0, 1)
    if invert:
        normalized = 1 - normalized
    return normalized

# Score components
scores = {
    'data_retention': data_retention_rate,  # Should be high
    'noise_removal': normalize_metric(noise_removal_rate, 0, 0.3),  # Moderate removal is good
    'density_improvement': normalize_metric(density_improvement['improvement_ratio'], 0.8, 1.2),  # Should be ~1.0
    'coverage_preserved': normalize_metric(coverage_metrics['coverage_preserved'], 0.8, 1.0),  # Should be high
    'uniformity_improvement': normalize_metric(coverage_metrics['uniformity_improvement'], -0.2, 0.2),  # Should be positive
    'radius_consistency': normalize_metric(radius_metrics['cv_improvement'], -0.1, 0.1)  # Should be positive
}

# Weighted composite score
weights = {
    'data_retention': 0.25,
    'noise_removal': 0.20,
    'density_improvement': 0.15,
    'coverage_preserved': 0.15,
    'uniformity_improvement': 0.15,
    'radius_consistency': 0.10
}

composite_score = sum(scores[key] * weights[key] for key in scores)

# Quality rating
if composite_score >= 0.85:
    quality_rating = "Excellent"
elif composite_score >= 0.70:
    quality_rating = "Good"
elif composite_score >= 0.55:
    quality_rating = "Fair"
else:
    quality_rating = "Poor"

quality_score = {
    'composite_score': float(composite_score),
    'quality_rating': quality_rating,
    'component_scores': {k: float(v) for k, v in scores.items()},
    'weights': weights
}

# ============================================================================
# COMPILE RESULTS
# ============================================================================

results = {
    'tunnel_id': tunnel_id,
    'evaluation_timestamp': pd.Timestamp.now().isoformat(),
    'metrics': {
        'noise_removal': noise_metrics,
        'point_density': density_improvement,
        'spatial_coverage': coverage_metrics,
        'radius_consistency': radius_metrics,
    },
    'quality_score': quality_score
}

# Save JSON results
json_path = os.path.join(output_dir, 'denoising_quality.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {json_path}")

# Generate markdown report
markdown_path = os.path.join(output_dir, 'denoising_quality.md')
with open(markdown_path, 'w') as f:
    f.write(f"# Denoising Quality Evaluation for Tunnel {tunnel_id}\n\n")
    f.write(f"**Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Input Points**: {total_input:,}\n")
    f.write(f"**Output Points**: {total_output:,}\n")
    f.write(f"**Valid Points**: {valid_points:,}\n")
    f.write(f"**Noise Removed**: {noise_points:,}\n\n")
    
    f.write("## Quality Score\n\n")
    f.write(f"**Composite Score**: {composite_score:.3f}\n")
    f.write(f"**Quality Rating**: **{quality_rating}**\n\n")
    
    f.write("### Component Scores\n\n")
    for component, score in scores.items():
        f.write(f"- **{component.replace('_', ' ').title()}**: {score:.3f}\n")
    f.write("\n")
    
    f.write("## Detailed Metrics\n\n")
    
    f.write("### 1. Noise Removal\n\n")
    f.write(f"- Data Retention Rate: {data_retention_rate:.1%}\n")
    f.write(f"- Noise Removal Rate: {noise_removal_rate:.1%}\n")
    f.write(f"- Points Removed: {points_removed:,}\n")
    f.write(f"- *Interpretation*: {noise_metrics['interpretation']}\n\n")
    
    f.write("### 2. Point Density\n\n")
    f.write(f"- Before: Median NN Distance = {density_before['median_nn_distance']:.4f} m\n")
    f.write(f"- After: Median NN Distance = {density_after['median_nn_distance']:.4f} m\n")
    f.write(f"- Improvement Ratio: {density_improvement['improvement_ratio']:.3f}\n")
    f.write(f"- *Interpretation*: {density_improvement['interpretation']}\n\n")
    
    f.write("### 3. Spatial Coverage\n\n")
    f.write(f"- Coverage Before: {coverage_before['coverage_percentage']:.1f}%\n")
    f.write(f"- Coverage After: {coverage_after['coverage_percentage']:.1f}%\n")
    f.write(f"- Coverage Preserved: {coverage_metrics['coverage_preserved']:.1%}\n")
    f.write(f"- Uniformity Improvement: {coverage_metrics['uniformity_improvement']:.3f}\n")
    f.write(f"- *Interpretation*: {coverage_metrics['interpretation']}\n\n")
    
    f.write("### 4. Radius Consistency\n\n")
    f.write(f"- Radius CV Before: {radius_before['cv']:.4f}\n")
    f.write(f"- Radius CV After: {radius_after['cv']:.4f}\n")
    f.write(f"- CV Improvement: {radius_metrics['cv_improvement']:.4f}\n")
    f.write(f"- *Interpretation*: {radius_metrics['interpretation']}\n\n")
    
    f.write("## Recommendations\n\n")
    if composite_score < 0.70:
        f.write("⚠️ **Quality issues detected**. Consider:\n")
        if data_retention_rate < 0.7:
            f.write("- Adjusting denoising parameters to preserve more valid points\n")
        if noise_removal_rate < 0.05:
            f.write("- Increasing noise removal aggressiveness\n")
        if coverage_metrics['coverage_preserved'] < 0.9:
            f.write("- Checking for over-aggressive filtering that removes valid regions\n")
        if radius_metrics['cv_improvement'] < 0:
            f.write("- Reviewing radius filtering parameters\n")
    else:
        f.write("✓ **Denoising quality is acceptable**. No major issues detected.\n")

print(f"✓ Markdown report saved to {markdown_path}")

# Print summary
print("\n" + "="*60)
print("DENOISING QUALITY SUMMARY")
print("="*60)
print(f"Composite Score: {composite_score:.3f} ({quality_rating})")
print(f"\nKey Metrics:")
print(f"  Data Retention:     {data_retention_rate:.1%}")
print(f"  Noise Removal:      {noise_removal_rate:.1%}")
print(f"  Coverage Preserved: {coverage_metrics['coverage_preserved']:.1%}")
print(f"  Radius CV Change:  {radius_metrics['cv_improvement']:.4f}")
print("="*60)
print(f"\n✓ Evaluation complete. See {markdown_path} for detailed report.")


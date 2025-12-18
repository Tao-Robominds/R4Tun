#!/usr/bin/env python3
"""
Enhancing Quality Evaluation Module

Computes intrinsic metrics to evaluate the quality of the enhancing/upsampling process.
These metrics compare input (denoised) vs output (enhanced) point clouds.
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
    print("Usage: python evaluate_enhancing.py <tunnel_id>")
    print("Example: python evaluate_enhancing.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]

# Determine base directory
if os.path.exists(f"data/{tunnel_id}/enhanced.csv"):
    base_dir = "data/"
else:
    base_dir = "../data/"

denoised_csv = os.path.join(base_dir, f"{tunnel_id}/denoised.csv")
enhanced_csv = os.path.join(base_dir, f"{tunnel_id}/enhanced.csv")
output_dir = os.path.join(base_dir, f"{tunnel_id}/evaluation")
os.makedirs(output_dir, exist_ok=True)

print(f"=== Enhancing Quality Evaluation for Tunnel: {tunnel_id} ===\n")

# Load data
if not os.path.exists(denoised_csv):
    print(f"❌ Error: denoised.csv not found at {denoised_csv}")
    sys.exit(1)
if not os.path.exists(enhanced_csv):
    print(f"❌ Error: enhanced.csv not found at {enhanced_csv}")
    sys.exit(1)

df_denoised = pd.read_csv(denoised_csv)
df_enhanced = pd.read_csv(enhanced_csv)

print(f"✓ Loaded {len(df_denoised):,} points from denoised.csv")
print(f"✓ Loaded {len(df_enhanced):,} points from enhanced.csv")

# Check required columns
required_cols = ['r', 'theta', 'h']
missing_cols = [col for col in required_cols if col not in df_denoised.columns]
if missing_cols:
    print(f"❌ Error: Missing required columns: {missing_cols}")
    sys.exit(1)

# Ensure pred column exists
if 'pred' not in df_denoised.columns:
    df_denoised['pred'] = 7
if 'pred' not in df_enhanced.columns:
    df_enhanced['pred'] = 7

# ============================================================================
# 1. ENHANCEMENT METRICS
# ============================================================================

print("\n[1/5] Computing Enhancement Metrics...")

total_before = len(df_denoised)
total_after = len(df_enhanced)

# Count valid points
valid_before = (df_denoised['pred'] != 0).sum() if 'pred' in df_denoised.columns else total_before
valid_after = (df_enhanced['pred'] != 0).sum() if 'pred' in df_enhanced.columns else total_after

points_added = total_after - total_before
enhancement_ratio = total_after / total_before if total_before > 0 else 1.0
valid_enhancement_ratio = valid_after / valid_before if valid_before > 0 else 1.0

enhancement_metrics = {
    'total_points_before': int(total_before),
    'total_points_after': int(total_after),
    'valid_points_before': int(valid_before),
    'valid_points_after': int(valid_after),
    'points_added': int(points_added),
    'enhancement_ratio': float(enhancement_ratio),
    'valid_enhancement_ratio': float(valid_enhancement_ratio),
    'interpretation': 'Higher enhancement ratio indicates more points added (better upsampling)'
}

# ============================================================================
# 2. POINT DENSITY ANALYSIS
# ============================================================================

print("[2/5] Computing Point Density Metrics...")

def compute_density_stats(df, sample_size=10000):
    """Compute nearest neighbor distance statistics"""
    valid_df = df[df['pred'] != 0] if 'pred' in df.columns else df
    
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

density_before = compute_density_stats(df_denoised)
density_after = compute_density_stats(df_enhanced)

density_improvement = {
    'before': density_before,
    'after': density_after,
    'improvement_ratio': density_before['median_nn_distance'] / density_after['median_nn_distance'] if density_after['median_nn_distance'] > 0 else 1.0,
    'density_increase': (density_before['median_nn_distance'] - density_after['median_nn_distance']) / density_before['median_nn_distance'] if density_before['median_nn_distance'] > 0 else 0.0,
    'interpretation': 'Lower median NN distance after enhancing indicates better density (successful upsampling)'
}

# ============================================================================
# 3. SPATIAL COVERAGE ANALYSIS
# ============================================================================

print("[3/5] Computing Spatial Coverage Metrics...")

def compute_coverage(df, h_bins=20, theta_bins=20):
    """Compute 2D coverage in (h, theta) space"""
    valid_df = df[df['pred'] != 0] if 'pred' in df.columns else df
    
    if len(valid_df) == 0:
        return {
            'coverage_percentage': 0.0,
            'coverage_uniformity': 0.0,
            'sparse_areas_percentage': 100.0,
            'coverage_entropy': 0.0
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
    
    # Coverage uniformity
    non_zero_densities = coverage_matrix[coverage_matrix > 0]
    if len(non_zero_densities) > 0:
        coverage_uniformity = 1 / (1 + np.std(non_zero_densities) / np.mean(non_zero_densities))
        sparse_threshold = np.percentile(non_zero_densities, 25)
        sparse_areas = np.sum(coverage_matrix < sparse_threshold)
        sparse_areas_percentage = (sparse_areas / total_cells) * 100
        # Coverage entropy (higher = more uniform distribution)
        coverage_entropy = entropy(coverage_matrix.flatten() + 1)  # +1 to avoid log(0)
    else:
        coverage_uniformity = 0.0
        sparse_areas_percentage = 100.0
        coverage_entropy = 0.0
    
    return {
        'coverage_percentage': float(coverage_percentage),
        'coverage_uniformity': float(coverage_uniformity),
        'sparse_areas_percentage': float(sparse_areas_percentage),
        'coverage_entropy': float(coverage_entropy)
    }

coverage_before = compute_coverage(df_denoised)
coverage_after = compute_coverage(df_enhanced)

# Compute improvement metrics
coverage_improvement_ratio = coverage_after['coverage_percentage'] / coverage_before['coverage_percentage'] if coverage_before['coverage_percentage'] > 0 else 1.0
uniformity_improvement = coverage_after['coverage_uniformity'] - coverage_before['coverage_uniformity']
sparse_reduction = coverage_before['sparse_areas_percentage'] - coverage_after['sparse_areas_percentage']

coverage_metrics = {
    'before': coverage_before,
    'after': coverage_after,
    'coverage_improvement_ratio': float(coverage_improvement_ratio),
    'uniformity_improvement': float(uniformity_improvement),
    'sparse_reduction': float(sparse_reduction),
    'interpretation': 'Higher coverage, uniformity, and sparse reduction indicate better enhancing quality'
}

# ============================================================================
# 4. UPSAMPLING QUALITY ANALYSIS
# ============================================================================

print("[4/5] Computing Upsampling Quality Metrics...")

# Check if upsampling improved sparse regions
def analyze_upsampling_effectiveness(df_before, df_after):
    """Analyze how well upsampling filled sparse regions"""
    valid_before = df_before[df_before['pred'] != 0] if 'pred' in df_before.columns else df_before
    valid_after = df_after[df_after['pred'] != 0] if 'pred' in df_after.columns else df_after
    
    if len(valid_before) == 0 or len(valid_after) == 0:
        return {
            'target_spacing_achieved': 0.0,
            'upsampling_efficiency': 0.0
        }
    
    # Target spacing for SAM (typically 0.05-0.08m)
    target_spacing = 0.06
    
    # Check if median spacing is close to target
    median_spacing_before = density_before['median_nn_distance']
    median_spacing_after = density_after['median_nn_distance']
    
    spacing_achievement = 1.0 - abs(median_spacing_after - target_spacing) / target_spacing
    spacing_achievement = max(0.0, min(1.0, spacing_achievement))
    
    # Upsampling efficiency (how much density improved relative to points added)
    if enhancement_ratio > 1.0:
        density_gain = (median_spacing_before - median_spacing_after) / median_spacing_before
        efficiency = density_gain / (enhancement_ratio - 1.0) if enhancement_ratio > 1.0 else 0.0
    else:
        efficiency = 0.0
    
    return {
        'target_spacing': target_spacing,
        'median_spacing_after': median_spacing_after,
        'target_spacing_achieved': float(spacing_achievement),
        'upsampling_efficiency': float(efficiency),
        'interpretation': 'Higher values indicate better upsampling effectiveness'
    }

upsampling_quality = analyze_upsampling_effectiveness(df_denoised, df_enhanced)

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
# Use a logarithmic scale for enhancement_ratio since values can be very high (3x, 5x, etc.)
# Normalize log(ratio) between log(1.0) and log(6.0) to better handle high ratios
enhancement_ratio_score = normalize_metric(np.log(enhancement_ratio), np.log(1.0), np.log(6.0)) if enhancement_ratio > 0 else 0.0

scores = {
    'enhancement_ratio': enhancement_ratio_score,  # Using log scale: 1.0x=0.0, 6.0x=1.0
    'density_improvement': normalize_metric(density_improvement['density_increase'], 0.0, 0.5),  # 0-50% improvement
    'coverage_improvement': normalize_metric(coverage_improvement_ratio, 1.0, 1.2),  # 1.0-1.2x improvement
    'uniformity_improvement': normalize_metric(uniformity_improvement, -0.1, 0.3),  # Should be positive
    'sparse_reduction': normalize_metric(sparse_reduction, 0.0, 20.0),  # 0-20% reduction
    'target_spacing': upsampling_quality['target_spacing_achieved'],  # How close to target
    'upsampling_efficiency': normalize_metric(upsampling_quality['upsampling_efficiency'], 0.0, 0.3)  # Efficiency
}

# Weighted composite score
weights = {
    'enhancement_ratio': 0.15,
    'density_improvement': 0.20,
    'coverage_improvement': 0.15,
    'uniformity_improvement': 0.15,
    'sparse_reduction': 0.15,
    'target_spacing': 0.10,
    'upsampling_efficiency': 0.10
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
        'enhancement': enhancement_metrics,
        'point_density': density_improvement,
        'spatial_coverage': coverage_metrics,
        'upsampling_quality': upsampling_quality,
    },
    'quality_score': quality_score
}

# Save JSON results
json_path = os.path.join(output_dir, 'enhancing_quality.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {json_path}")

# Generate markdown report
markdown_path = os.path.join(output_dir, 'enhancing_quality.md')
with open(markdown_path, 'w') as f:
    f.write(f"# Enhancing Quality Evaluation for Tunnel {tunnel_id}\n\n")
    f.write(f"**Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Points Before**: {total_before:,}\n")
    f.write(f"**Points After**: {total_after:,}\n")
    f.write(f"**Points Added**: {points_added:,}\n")
    f.write(f"**Enhancement Ratio**: {enhancement_ratio:.2f}x\n\n")
    
    f.write("## Quality Score\n\n")
    f.write(f"**Composite Score**: {composite_score:.3f}\n")
    f.write(f"**Quality Rating**: **{quality_rating}**\n\n")
    
    f.write("### Component Scores\n\n")
    for component, score in scores.items():
        weight = weights[component]
        contribution = score * weight
        f.write(f"- **{component.replace('_', ' ').title()}**: {score:.3f} (weight: {weight:.2f}, contribution: {contribution:.3f})\n")
    f.write("\n")
    f.write(f"**Note**: The composite score is a weighted average of all component scores. ")
    f.write(f"A high enhancement ratio ({enhancement_ratio:.2f}x) indicates many points were added, ")
    f.write(f"but the overall quality rating also considers density improvement, coverage, uniformity, and efficiency.\n\n")
    
    f.write("## Detailed Metrics\n\n")
    
    f.write("### 1. Enhancement\n\n")
    f.write(f"- Enhancement Ratio: {enhancement_ratio:.2f}x\n")
    f.write(f"- Points Added: {points_added:,}\n")
    f.write(f"- Valid Enhancement Ratio: {valid_enhancement_ratio:.2f}x\n")
    f.write(f"- *Interpretation*: {enhancement_metrics['interpretation']}\n\n")
    
    f.write("### 2. Point Density\n\n")
    f.write(f"- Before: Median NN Distance = {density_before['median_nn_distance']:.4f} m\n")
    f.write(f"- After: Median NN Distance = {density_after['median_nn_distance']:.4f} m\n")
    f.write(f"- Improvement Ratio: {density_improvement['improvement_ratio']:.3f}\n")
    f.write(f"- Density Increase: {density_improvement['density_increase']:.1%}\n")
    f.write(f"- *Interpretation*: {density_improvement['interpretation']}\n\n")
    
    f.write("### 3. Spatial Coverage\n\n")
    f.write(f"- Coverage Before: {coverage_before['coverage_percentage']:.1f}%\n")
    f.write(f"- Coverage After: {coverage_after['coverage_percentage']:.1f}%\n")
    f.write(f"- Coverage Improvement: {coverage_improvement_ratio:.2f}x\n")
    f.write(f"- Uniformity Improvement: {uniformity_improvement:.3f}\n")
    f.write(f"- Sparse Reduction: {sparse_reduction:.1f}%\n")
    f.write(f"- *Interpretation*: {coverage_metrics['interpretation']}\n\n")
    
    f.write("### 4. Upsampling Quality\n\n")
    f.write(f"- Target Spacing: {upsampling_quality['target_spacing']:.3f} m\n")
    f.write(f"- Achieved Spacing: {upsampling_quality['median_spacing_after']:.4f} m\n")
    f.write(f"- Target Achievement: {upsampling_quality['target_spacing_achieved']:.3f}\n")
    f.write(f"- Upsampling Efficiency: {upsampling_quality['upsampling_efficiency']:.3f}\n")
    f.write(f"- *Interpretation*: {upsampling_quality['interpretation']}\n\n")
    
    f.write("## Recommendations\n\n")
    if composite_score < 0.70:
        f.write("⚠️ **Quality issues detected**. Consider:\n")
        if enhancement_ratio < 1.2:
            f.write("- Increasing upsampling parameters to add more points\n")
        if density_improvement['density_increase'] < 0.1:
            f.write("- Adjusting target_distance parameters for better density\n")
        if coverage_improvement_ratio < 1.05:
            f.write("- Improving coverage in sparse regions\n")
        if upsampling_quality['target_spacing_achieved'] < 0.7:
            f.write("- Fine-tuning target spacing parameters\n")
    else:
        f.write("✓ **Enhancing quality is acceptable**. No major issues detected.\n")

print(f"✓ Markdown report saved to {markdown_path}")

# Print summary
print("\n" + "="*60)
print("ENHANCING QUALITY SUMMARY")
print("="*60)
print(f"Composite Score: {composite_score:.3f} ({quality_rating})")
print(f"\nKey Metrics:")
print(f"  Enhancement Ratio:    {enhancement_ratio:.2f}x")
print(f"  Density Improvement:  {density_improvement['density_increase']:.1%}")
print(f"  Coverage Improvement:  {coverage_improvement_ratio:.2f}x")
print(f"  Target Achievement:   {upsampling_quality['target_spacing_achieved']:.3f}")
print("="*60)
print(f"\n✓ Evaluation complete. See {markdown_path} for detailed report.")


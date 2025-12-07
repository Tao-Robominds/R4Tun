#!/usr/bin/env python3
"""
Unfolding Quality Evaluation Module

Computes intrinsic metrics to evaluate the quality of the unfolding/unwrapping process.
These metrics don't require ground truth and can be computed directly from the unfolding output.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python evaluate_unfolding.py <tunnel_id>")
    print("Example: python evaluate_unfolding.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]

# Determine base directory
if os.path.exists(f"data/{tunnel_id}/unwrapped.csv"):
    base_dir = "data/"
else:
    base_dir = "../data/"

unwrapped_csv = os.path.join(base_dir, f"{tunnel_id}/unwrapped.csv")
ring_count_file = os.path.join(base_dir, f"{tunnel_id}/ring_count.txt")
output_dir = os.path.join(base_dir, f"{tunnel_id}/evaluation")
os.makedirs(output_dir, exist_ok=True)

print(f"=== Unfolding Quality Evaluation for Tunnel: {tunnel_id} ===\n")

# Load data
if not os.path.exists(unwrapped_csv):
    print(f"❌ Error: unwrapped.csv not found at {unwrapped_csv}")
    sys.exit(1)

df = pd.read_csv(unwrapped_csv)
print(f"✓ Loaded {len(df):,} points from unwrapped.csv")

# Load ring count
ring_count = None
if os.path.exists(ring_count_file):
    with open(ring_count_file, 'r') as f:
        ring_count = int(f.read().strip())
    print(f"✓ Ring count: {ring_count}")

# Check required columns
required_cols = ['r', 'theta', 'h']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ Error: Missing required columns: {missing_cols}")
    sys.exit(1)

# ============================================================================
# 1. CYLINDRICAL COORDINATE CONSISTENCY METRICS
# ============================================================================

print("\n[1/4] Computing Cylindrical Coordinate Consistency Metrics...")

r_values = df['r'].values
theta_values = df['theta'].values
h_values = df['h'].values

# Radius consistency
r_mean = np.mean(r_values)
r_std = np.std(r_values)
r_median = np.median(r_values)
r_min = np.min(r_values)
r_max = np.max(r_values)
r_range = r_max - r_min
r_cv = r_std / r_mean if r_mean > 0 else 0  # Coefficient of variation

# Radius distribution statistics
r_q25, r_q75 = np.percentile(r_values, [25, 75])
r_iqr = r_q75 - r_q25

# Theta coverage
theta_min = np.min(theta_values)
theta_max = np.max(theta_values)
theta_span = theta_max - theta_min
theta_coverage_rad = theta_span
theta_coverage_deg = np.degrees(theta_coverage_rad)
expected_coverage_rad = 2 * np.pi
theta_coverage_ratio = theta_coverage_rad / expected_coverage_rad if expected_coverage_rad > 0 else 0

# Theta distribution analysis (check for gaps)
theta_bins = 360
theta_hist, theta_edges = np.histogram(theta_values, bins=theta_bins, range=(theta_min, theta_max))
theta_gaps = np.sum(theta_hist == 0)  # Number of empty bins
theta_gap_ratio = theta_gaps / theta_bins

# Height (h) spacing consistency
h_sorted = np.sort(h_values)
h_unique = np.unique(h_sorted)
if len(h_unique) > 1:
    h_spacings = np.diff(h_unique)
    h_spacing_mean = np.mean(h_spacings)
    h_spacing_std = np.std(h_spacings)
    h_spacing_cv = h_spacing_std / h_spacing_mean if h_spacing_mean > 0 else 0
    h_spacing_min = np.min(h_spacings)
    h_spacing_max = np.max(h_spacings)
else:
    h_spacing_mean = h_spacing_std = h_spacing_cv = h_spacing_min = h_spacing_max = 0

h_min = np.min(h_values)
h_max = np.max(h_values)
h_span = h_max - h_min

cylindrical_metrics = {
    'radius_consistency': {
        'mean': float(r_mean),
        'std': float(r_std),
        'median': float(r_median),
        'min': float(r_min),
        'max': float(r_max),
        'range': float(r_range),
        'coefficient_of_variation': float(r_cv),
        'iqr': float(r_iqr),
        'interpretation': 'Lower CV and range indicate more consistent radius (better for circular tunnels)'
    },
    'theta_coverage': {
        'min_rad': float(theta_min),
        'max_rad': float(theta_max),
        'span_rad': float(theta_span),
        'span_degrees': float(theta_coverage_deg),
        'coverage_ratio': float(theta_coverage_ratio),
        'expected_coverage_rad': float(expected_coverage_rad),
        'gaps_count': int(theta_gaps),
        'gap_ratio': float(theta_gap_ratio),
        'interpretation': 'Coverage ratio should be ~1.0 for full 360° coverage. Lower gap_ratio is better.'
    },
    'height_spacing': {
        'min': float(h_min),
        'max': float(h_max),
        'span': float(h_span),
        'spacing_mean': float(h_spacing_mean),
        'spacing_std': float(h_spacing_std),
        'spacing_cv': float(h_spacing_cv),
        'spacing_min': float(h_spacing_min),
        'spacing_max': float(h_spacing_max),
        'interpretation': 'Lower spacing CV indicates more uniform ring spacing'
    }
}

# ============================================================================
# 2. SPATIAL COVERAGE METRICS
# ============================================================================

print("[2/4] Computing Spatial Coverage Metrics...")

# Create 2D grid in (theta, h) space
theta_bins_2d = 72  # 5 degree bins
h_bins_2d = ring_count if ring_count else 50
theta_edges_2d = np.linspace(theta_min, theta_max, theta_bins_2d + 1)
h_edges_2d = np.linspace(h_min, h_max, h_bins_2d + 1)

# Compute 2D histogram
coverage_hist, _, _ = np.histogram2d(theta_values, h_values, bins=[theta_edges_2d, h_edges_2d])

# Coverage completeness
covered_cells = np.sum(coverage_hist > 0)
total_cells = theta_bins_2d * h_bins_2d
coverage_completeness = covered_cells / total_cells if total_cells > 0 else 0

# Density uniformity (coefficient of variation of cell densities)
non_zero_densities = coverage_hist[coverage_hist > 0]
if len(non_zero_densities) > 0:
    density_mean = np.mean(non_zero_densities)
    density_std = np.std(non_zero_densities)
    density_cv = density_std / density_mean if density_mean > 0 else 0
else:
    density_cv = 0

# Point distribution per ring (if ring information available)
if 'ring' in df.columns:
    ring_counts = df['ring'].value_counts().sort_index()
    ring_count_mean = ring_counts.mean()
    ring_count_std = ring_counts.std()
    ring_count_cv = ring_count_std / ring_count_mean if ring_count_mean > 0 else 0
else:
    ring_count_mean = ring_count_std = ring_count_cv = None

coverage_metrics = {
    'coverage_completeness': {
        'value': float(coverage_completeness),
        'covered_cells': int(covered_cells),
        'total_cells': int(total_cells),
        'interpretation': 'Should be close to 1.0 for complete coverage'
    },
    'density_uniformity': {
        'coefficient_of_variation': float(density_cv),
        'interpretation': 'Lower CV indicates more uniform point distribution'
    },
    'ring_point_distribution': {
        'mean_points_per_ring': float(ring_count_mean) if ring_count_mean is not None else None,
        'std_points_per_ring': float(ring_count_std) if ring_count_std is not None else None,
        'cv_points_per_ring': float(ring_count_cv) if ring_count_cv is not None else None,
        'interpretation': 'Lower CV indicates more uniform point distribution across rings'
    }
}

# ============================================================================
# 3. GEOMETRIC CONSISTENCY METRICS
# ============================================================================

print("[3/4] Computing Geometric Consistency Metrics...")

# If we have original x, y, z coordinates, we can compute additional metrics
if all(col in df.columns for col in ['x', 'y', 'z']):
    xyz = df[['x', 'y', 'z']].values
    
    # Compute distance from points to centerline (using mean radius as proxy)
    # For a perfect circular tunnel, all points should be at approximately the same distance
    center_distances = np.linalg.norm(xyz[:, :2], axis=1)  # Distance in XY plane
    center_distance_mean = np.mean(center_distances)
    center_distance_std = np.std(center_distances)
    center_distance_cv = center_distance_std / center_distance_mean if center_distance_mean > 0 else 0
    
    # Compare with r values (should be similar for good unfolding)
    r_xyz_correlation = np.corrcoef(center_distances, r_values)[0, 1]
    
    geometric_metrics = {
        'centerline_consistency': {
            'mean_distance_to_centerline': float(center_distance_mean),
            'std_distance_to_centerline': float(center_distance_std),
            'cv_distance_to_centerline': float(center_distance_cv),
            'r_xyz_correlation': float(r_xyz_correlation),
            'interpretation': 'High correlation and low CV indicate consistent centerline extraction'
        }
    }
else:
    geometric_metrics = {
        'centerline_consistency': {
            'note': 'Original xyz coordinates not available for centerline analysis'
        }
    }

# ============================================================================
# 4. COMPOSITE QUALITY SCORE
# ============================================================================

print("[4/4] Computing Composite Quality Score...")

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

# Score components (0-1, higher is better)
scores = {
    'radius_consistency': normalize_metric(r_cv, 0, 0.2, invert=True),  # CV < 0.2 is good
    'theta_coverage': theta_coverage_ratio,  # Should be ~1.0
    'theta_gaps': normalize_metric(theta_gap_ratio, 0, 0.3, invert=True),  # < 30% gaps is good
    'height_spacing': normalize_metric(h_spacing_cv, 0, 0.3, invert=True),  # CV < 0.3 is good
    'coverage_completeness': coverage_completeness,  # Should be ~1.0
    'density_uniformity': normalize_metric(density_cv, 0, 1.0, invert=True),  # Lower CV is better
}

# Weighted composite score
weights = {
    'radius_consistency': 0.20,
    'theta_coverage': 0.20,
    'theta_gaps': 0.15,
    'height_spacing': 0.15,
    'coverage_completeness': 0.15,
    'density_uniformity': 0.15,
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
    'weights': weights,
    'interpretation': {
        'Excellent (≥0.85)': 'All metrics within acceptable ranges',
        'Good (0.70-0.85)': 'Most metrics acceptable, minor issues',
        'Fair (0.55-0.70)': 'Some metrics outside acceptable ranges',
        'Poor (<0.55)': 'Multiple metrics indicate significant problems'
    }
}

# ============================================================================
# COMPILE RESULTS
# ============================================================================

results = {
    'tunnel_id': tunnel_id,
    'evaluation_timestamp': pd.Timestamp.now().isoformat(),
    'total_points': int(len(df)),
    'ring_count': ring_count,
    'metrics': {
        'cylindrical_coordinates': cylindrical_metrics,
        'spatial_coverage': coverage_metrics,
        'geometric_consistency': geometric_metrics,
    },
    'quality_score': quality_score
}

# Save JSON results
json_path = os.path.join(output_dir, 'unfolding_quality.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {json_path}")

# Generate markdown report
markdown_path = os.path.join(output_dir, 'unfolding_quality.md')
with open(markdown_path, 'w') as f:
    f.write(f"# Unfolding Quality Evaluation for Tunnel {tunnel_id}\n\n")
    f.write(f"**Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Total Points**: {len(df):,}\n")
    f.write(f"**Ring Count**: {ring_count if ring_count else 'N/A'}\n\n")
    
    f.write("## Quality Score\n\n")
    f.write(f"**Composite Score**: {composite_score:.3f}\n")
    f.write(f"**Quality Rating**: **{quality_rating}**\n\n")
    
    f.write("### Component Scores\n\n")
    for component, score in scores.items():
        f.write(f"- **{component.replace('_', ' ').title()}**: {score:.3f}\n")
    f.write("\n")
    
    f.write("## Detailed Metrics\n\n")
    
    f.write("### 1. Radius Consistency\n\n")
    f.write(f"- Mean: {r_mean:.4f} m\n")
    f.write(f"- Std: {r_std:.4f} m\n")
    f.write(f"- Coefficient of Variation: {r_cv:.4f}\n")
    f.write(f"- Range: {r_range:.4f} m\n")
    f.write(f"- *Interpretation*: {cylindrical_metrics['radius_consistency']['interpretation']}\n\n")
    
    f.write("### 2. Theta Coverage\n\n")
    f.write(f"- Coverage: {theta_coverage_deg:.1f}° ({theta_coverage_ratio:.3f} of expected 360°)\n")
    f.write(f"- Gap Ratio: {theta_gap_ratio:.3f} ({theta_gaps} empty bins out of {theta_bins})\n")
    f.write(f"- *Interpretation*: {cylindrical_metrics['theta_coverage']['interpretation']}\n\n")
    
    f.write("### 3. Height Spacing\n\n")
    f.write(f"- Span: {h_span:.2f} m\n")
    f.write(f"- Spacing CV: {h_spacing_cv:.4f}\n")
    f.write(f"- *Interpretation*: {cylindrical_metrics['height_spacing']['interpretation']}\n\n")
    
    f.write("### 4. Spatial Coverage\n\n")
    f.write(f"- Coverage Completeness: {coverage_completeness:.3f}\n")
    f.write(f"- Density Uniformity CV: {density_cv:.4f}\n")
    f.write(f"- *Interpretation*: Higher completeness and lower CV indicate better coverage\n\n")
    
    if 'centerline_consistency' in geometric_metrics:
        if 'r_xyz_correlation' in geometric_metrics['centerline_consistency']:
            f.write("### 5. Centerline Consistency\n\n")
            f.write(f"- R-XYZ Correlation: {geometric_metrics['centerline_consistency']['r_xyz_correlation']:.4f}\n")
            f.write(f"- Distance CV: {geometric_metrics['centerline_consistency']['cv_distance_to_centerline']:.4f}\n")
            f.write(f"- *Interpretation*: {geometric_metrics['centerline_consistency']['interpretation']}\n\n")
    
    f.write("## Recommendations\n\n")
    if composite_score < 0.70:
        f.write("⚠️ **Quality issues detected**. Consider:\n")
        if r_cv > 0.2:
            f.write("- Adjusting ellipse fitting parameters (ransac_threshold, ransac_inlier_ratio)\n")
        if theta_coverage_ratio < 0.9:
            f.write("- Checking point cloud completeness and coverage\n")
        if h_spacing_cv > 0.3:
            f.write("- Adjusting slice_spacing_factor for more uniform ring spacing\n")
        if coverage_completeness < 0.8:
            f.write("- Investigating missing regions in the point cloud\n")
    else:
        f.write("✓ **Unfolding quality is acceptable**. No major issues detected.\n")

print(f"✓ Markdown report saved to {markdown_path}")

# Print summary
print("\n" + "="*60)
print("UNFOLDING QUALITY SUMMARY")
print("="*60)
print(f"Composite Score: {composite_score:.3f} ({quality_rating})")
print(f"\nKey Metrics:")
print(f"  Radius CV:        {r_cv:.4f}")
print(f"  Theta Coverage:    {theta_coverage_deg:.1f}° ({theta_coverage_ratio:.3f})")
print(f"  Height Spacing CV: {h_spacing_cv:.4f}")
print(f"  Coverage:          {coverage_completeness:.3f}")
print("="*60)
print(f"\n✓ Evaluation complete. See {markdown_path} for detailed report.")


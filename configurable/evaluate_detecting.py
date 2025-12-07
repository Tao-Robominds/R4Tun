#!/usr/bin/env python3
"""
Detecting Quality Evaluation Module

Computes intrinsic metrics to evaluate the quality of the detecting/prompt point generation process.
These metrics assess detection completeness, spatial distribution, and geometric consistency.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from scipy.spatial import cKDTree
from scipy.stats import entropy, linregress

# Check if tunnel_id is provided
if len(sys.argv) != 2:
    print("Usage: python evaluate_detecting.py <tunnel_id>")
    print("Example: python evaluate_detecting.py 1-4")
    sys.exit(1)

tunnel_id = sys.argv[1]

# Determine base directory
if os.path.exists(f"data/{tunnel_id}/detected.csv"):
    base_dir = "data/"
else:
    base_dir = "../data/"

detected_csv = os.path.join(base_dir, f"{tunnel_id}/detected.csv")
depth_map_file = os.path.join(base_dir, f"{tunnel_id}/depth_map.png")
ring_count_file = os.path.join(base_dir, f"{tunnel_id}/ring_count.txt")
output_dir = os.path.join(base_dir, f"{tunnel_id}/evaluation")
os.makedirs(output_dir, exist_ok=True)

print(f"=== Detecting Quality Evaluation for Tunnel: {tunnel_id} ===\n")

# Load data
if not os.path.exists(detected_csv):
    print(f"❌ Error: detected.csv not found at {detected_csv}")
    sys.exit(1)

df_detected = pd.read_csv(detected_csv)

# Load ring count if available
ring_count = None
if os.path.exists(ring_count_file):
    try:
        ring_count = int(open(ring_count_file, 'r').read().strip())
        print(f"✓ Ring count: {ring_count}")
    except:
        pass

print(f"✓ Loaded {len(df_detected):,} detected points from detected.csv")

# Check required columns
required_cols = ['X', 'Y']
missing_cols = [col for col in required_cols if col not in df_detected.columns]
if missing_cols:
    print(f"❌ Error: Missing required columns: {missing_cols}")
    sys.exit(1)

# ============================================================================
# 1. DETECTION COMPLETENESS METRICS
# ============================================================================

print("\n[1/5] Computing Detection Completeness Metrics...")

total_detected = len(df_detected)
valid_detected = df_detected.dropna(subset=['X', 'Y'])
valid_count = len(valid_detected)

# Detection type distribution
type_distribution = {}
if 'Type' in df_detected.columns:
    type_distribution = df_detected['Type'].value_counts().to_dict()

# Expected types based on detection algorithm
expected_types = ['midpoint', 'positive_slope', 'negative_slope', 'horizontal', 'assume', 'default']
detected_type_count = len([t for t in type_distribution.keys() if t in expected_types])

# Quality of detection methods (prefer geometric methods over assumptions)
quality_types = ['midpoint', 'positive_slope', 'negative_slope', 'horizontal']
low_quality_types = ['assume', 'default']
quality_type_count = sum(type_distribution.get(t, 0) for t in quality_types)
low_quality_type_count = sum(type_distribution.get(t, 0) for t in low_quality_types)

quality_ratio = quality_type_count / valid_count if valid_count > 0 else 0.0

completeness_metrics = {
    'total_points_detected': int(total_detected),
    'valid_points_detected': int(valid_count),
    'invalid_points': int(total_detected - valid_count),
    'type_distribution': {k: int(v) for k, v in type_distribution.items()},
    'detected_type_count': int(detected_type_count),
    'quality_type_count': int(quality_type_count),
    'low_quality_type_count': int(low_quality_type_count),
    'quality_ratio': float(quality_ratio),
    'interpretation': 'Higher quality_ratio indicates more geometric detections vs. assumptions/defaults'
}

# ============================================================================
# 2. SPATIAL DISTRIBUTION METRICS
# ============================================================================

print("[2/5] Computing Spatial Distribution Metrics...")

def compute_spatial_stats(df):
    """Compute spatial distribution statistics"""
    if len(df) == 0:
        return {
            'x_range': [0, 0],
            'y_range': [0, 0],
            'coverage_area': 0.0,
            'x_spread': 0.0,
            'y_spread': 0.0,
            'aspect_ratio': 1.0
        }
    
    x_range = [float(df['X'].min()), float(df['X'].max())]
    y_range = [float(df['Y'].min()), float(df['Y'].max())]
    
    x_spread = x_range[1] - x_range[0]
    y_spread = y_range[1] - y_range[0]
    coverage_area = x_spread * y_spread
    aspect_ratio = x_spread / y_spread if y_spread > 0 else 1.0
    
    return {
        'x_range': x_range,
        'y_range': y_range,
        'coverage_area': float(coverage_area),
        'x_spread': float(x_spread),
        'y_spread': float(y_spread),
        'aspect_ratio': float(aspect_ratio)
    }

spatial_stats = compute_spatial_stats(valid_detected)

# Spatial uniformity (entropy of 2D grid distribution)
def compute_spatial_uniformity(df, bins=10):
    """Compute spatial uniformity using 2D histogram entropy"""
    if len(df) == 0:
        return 0.0
    
    x_bins = np.linspace(df['X'].min(), df['X'].max(), bins + 1)
    y_bins = np.linspace(df['Y'].min(), df['Y'].max(), bins + 1)
    
    hist, _, _ = np.histogram2d(df['X'], df['Y'], bins=[x_bins, y_bins])
    hist_flat = hist.flatten()
    hist_flat = hist_flat[hist_flat > 0]  # Remove zeros
    
    if len(hist_flat) == 0:
        return 0.0
    
    return float(entropy(hist_flat))

uniformity_score = compute_spatial_uniformity(valid_detected)

distribution_metrics = {
    'spatial_coverage': spatial_stats,
    'spatial_uniformity': float(uniformity_score),
    'interpretation': 'Higher uniformity indicates more even spatial distribution of detected points'
}

# ============================================================================
# 3. GEOMETRIC CONSISTENCY METRICS
# ============================================================================

print("[3/5] Computing Geometric Consistency Metrics...")

def compute_spacing_consistency(df):
    """Compute spacing consistency between adjacent points"""
    if len(df) < 2:
        return {
            'mean_spacing': 0.0,
            'median_spacing': 0.0,
            'std_spacing': 0.0,
            'cv_spacing': 0.0,
            'spacing_regularity': 0.0
        }
    
    # Sort by X coordinate
    sorted_df = df.sort_values('X').reset_index(drop=True)
    
    # Compute Euclidean distances between consecutive points
    x_diffs = np.diff(sorted_df['X'].values)
    y_diffs = np.diff(sorted_df['Y'].values)
    spacings = np.sqrt(x_diffs**2 + y_diffs**2)
    
    mean_spacing = float(np.mean(spacings))
    median_spacing = float(np.median(spacings))
    std_spacing = float(np.std(spacings))
    cv_spacing = std_spacing / mean_spacing if mean_spacing > 0 else 0.0
    
    # Spacing regularity (1 - CV, higher is better)
    spacing_regularity = max(0.0, 1.0 - cv_spacing)
    
    return {
        'mean_spacing': mean_spacing,
        'median_spacing': median_spacing,
        'std_spacing': std_spacing,
        'cv_spacing': float(cv_spacing),
        'spacing_regularity': float(spacing_regularity),
        'min_spacing': float(np.min(spacings)),
        'max_spacing': float(np.max(spacings))
    }

spacing_stats = compute_spacing_consistency(valid_detected)

# X-coordinate linearity (points should be roughly evenly spaced in X)
def compute_x_linearity(df):
    """Compute how well X coordinates follow a linear progression"""
    if len(df) < 3:
        return {
            'x_linearity_r2': 0.0,
            'x_spacing_consistency': 0.0
        }
    
    sorted_df = df.sort_values('X').reset_index(drop=True)
    x_coords = sorted_df['X'].values
    indices = np.arange(len(x_coords))
    
    # Linear regression to check if X coordinates are evenly spaced
    slope, intercept, r_value, p_value, std_err = linregress(indices, x_coords)
    r2 = r_value**2
    
    # X spacing consistency (should be constant)
    x_spacings = np.diff(x_coords)
    x_spacing_cv = np.std(x_spacings) / np.mean(x_spacings) if np.mean(x_spacings) > 0 else 0.0
    x_spacing_consistency = max(0.0, 1.0 - x_spacing_cv)
    
    return {
        'x_linearity_r2': float(r2),
        'x_spacing_consistency': float(x_spacing_consistency),
        'x_slope': float(slope),
        'x_intercept': float(intercept)
    }

x_linearity = compute_x_linearity(valid_detected)

geometric_metrics = {
    'spacing_consistency': spacing_stats,
    'x_linearity': x_linearity,
    'interpretation': 'Higher spacing_regularity and x_linearity indicate more consistent geometric structure'
}

# ============================================================================
# 4. RING ALIGNMENT METRICS (if ring_count available)
# ============================================================================

print("[4/5] Computing Ring Alignment Metrics...")

ring_alignment_metrics = {
    'ring_count_provided': ring_count is not None,
    'ring_count': int(ring_count) if ring_count else None,
    'points_per_ring': float(valid_count / ring_count) if ring_count and ring_count > 0 else None,
    'interpretation': 'Compares detected points to expected ring structure'
}

if ring_count:
    # Expected spacing based on ring count and image width
    if len(valid_detected) > 0:
        x_range = valid_detected['X'].max() - valid_detected['X'].min()
        expected_spacing = x_range / ring_count if ring_count > 0 else 0.0
        actual_mean_spacing = spacing_stats['mean_spacing']
        
        spacing_ratio = actual_mean_spacing / expected_spacing if expected_spacing > 0 else 0.0
        spacing_accuracy = max(0.0, min(1.0, 1.0 - abs(1.0 - spacing_ratio)))
        
        ring_alignment_metrics['expected_spacing'] = float(expected_spacing)
        ring_alignment_metrics['actual_mean_spacing'] = float(actual_mean_spacing)
        ring_alignment_metrics['spacing_ratio'] = float(spacing_ratio)
        ring_alignment_metrics['spacing_accuracy'] = float(spacing_accuracy)
        
        # Point count alignment
        expected_points = ring_count
        point_count_ratio = valid_count / expected_points if expected_points > 0 else 0.0
        point_count_accuracy = 1.0 - abs(1.0 - point_count_ratio) if point_count_ratio <= 1.0 else 1.0 / point_count_ratio
        
        ring_alignment_metrics['expected_points'] = int(expected_points)
        ring_alignment_metrics['point_count_ratio'] = float(point_count_ratio)
        ring_alignment_metrics['point_count_accuracy'] = float(point_count_accuracy)

# ============================================================================
# 5. DETECTION QUALITY METRICS
# ============================================================================

print("[5/5] Computing Detection Quality Metrics...")

# Coverage efficiency (how well points cover the depth map area)
def estimate_depth_map_size():
    """Estimate depth map size from detected points"""
    if len(valid_detected) == 0:
        return None
    
    # Assume resolution is 0.005 (typical for this pipeline)
    resolution = 0.005
    
    # Estimate dimensions from point spread
    x_range = valid_detected['X'].max() - valid_detected['X'].min()
    y_range = valid_detected['Y'].max() - valid_detected['Y'].min()
    
    # Add some margin
    estimated_width = x_range * 1.2
    estimated_height = y_range * 1.2
    
    return {
        'estimated_width': float(estimated_width),
        'estimated_height': float(estimated_height),
        'estimated_area': float(estimated_width * estimated_height),
        'resolution': resolution
    }

depth_map_estimate = estimate_depth_map_size()

coverage_efficiency = 0.0
if depth_map_estimate and depth_map_estimate['estimated_area'] > 0:
    detection_coverage = spatial_stats['coverage_area']
    coverage_efficiency = min(1.0, detection_coverage / depth_map_estimate['estimated_area'])

# Detection density (points per unit area)
detection_density = valid_count / spatial_stats['coverage_area'] if spatial_stats['coverage_area'] > 0 else 0.0

quality_metrics = {
    'coverage_efficiency': float(coverage_efficiency),
    'detection_density': float(detection_density),
    'depth_map_estimate': depth_map_estimate,
    'interpretation': 'Higher coverage_efficiency and appropriate detection_density indicate better detection quality'
}

# ============================================================================
# COMPOSITE QUALITY SCORE
# ============================================================================

print("[6/6] Computing Composite Quality Score...")

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
    'completeness': normalize_metric(valid_count, 5, 30),  # Expect 5-30 points
    'quality_ratio': quality_ratio,  # Already 0-1
    'spatial_uniformity': normalize_metric(uniformity_score, 0, 3),  # Entropy-based
    'spacing_regularity': spacing_stats['spacing_regularity'],  # Already 0-1
    'x_linearity': x_linearity['x_linearity_r2'],  # R² is already 0-1
    'coverage_efficiency': coverage_efficiency  # Already 0-1
}

# Add ring alignment if available
if ring_count:
    scores['spacing_accuracy'] = ring_alignment_metrics.get('spacing_accuracy', 0.0)
    scores['point_count_accuracy'] = ring_alignment_metrics.get('point_count_accuracy', 0.0)

# Weighted composite score
weights = {
    'completeness': 0.15,
    'quality_ratio': 0.20,
    'spatial_uniformity': 0.15,
    'spacing_regularity': 0.20,
    'x_linearity': 0.15,
    'coverage_efficiency': 0.15
}

if ring_count:
    weights['spacing_accuracy'] = 0.10
    weights['point_count_accuracy'] = 0.10
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

composite_score = sum(scores[key] * weights.get(key, 0) for key in scores)

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
        'completeness': completeness_metrics,
        'spatial_distribution': distribution_metrics,
        'geometric_consistency': geometric_metrics,
        'ring_alignment': ring_alignment_metrics,
        'detection_quality': quality_metrics,
    },
    'quality_score': quality_score
}

# Save JSON results
json_path = os.path.join(output_dir, 'detecting_quality.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Results saved to {json_path}")

# Generate markdown report
markdown_path = os.path.join(output_dir, 'detecting_quality.md')
with open(markdown_path, 'w') as f:
    f.write(f"# Detecting Quality Evaluation for Tunnel {tunnel_id}\n\n")
    f.write(f"**Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Points Detected**: {valid_count:,}\n")
    if ring_count:
        f.write(f"**Expected Rings**: {ring_count}\n")
        f.write(f"**Points per Ring**: {ring_alignment_metrics.get('points_per_ring', 0):.2f}\n")
    f.write("\n")
    
    f.write("## Quality Score\n\n")
    f.write(f"**Composite Score**: {composite_score:.3f}\n")
    f.write(f"**Quality Rating**: **{quality_rating}**\n\n")
    
    f.write("### Component Scores\n\n")
    for component, score in scores.items():
        f.write(f"- **{component.replace('_', ' ').title()}**: {score:.3f}\n")
    f.write("\n")
    
    f.write("## Detailed Metrics\n\n")
    
    f.write("### 1. Detection Completeness\n\n")
    f.write(f"- Total Points: {total_detected}\n")
    f.write(f"- Valid Points: {valid_count}\n")
    f.write(f"- Quality Types: {quality_type_count} ({quality_ratio:.1%})\n")
    f.write(f"- Low Quality Types: {low_quality_type_count}\n")
    f.write(f"- *Interpretation*: {completeness_metrics['interpretation']}\n\n")
    
    f.write("**Type Distribution:**\n")
    for type_name, count in type_distribution.items():
        f.write(f"- {type_name}: {count}\n")
    f.write("\n")
    
    f.write("### 2. Spatial Distribution\n\n")
    f.write(f"- Coverage Area: {spatial_stats['coverage_area']:.1f} pixels²\n")
    f.write(f"- X Spread: {spatial_stats['x_spread']:.1f} pixels\n")
    f.write(f"- Y Spread: {spatial_stats['y_spread']:.1f} pixels\n")
    f.write(f"- Aspect Ratio: {spatial_stats['aspect_ratio']:.3f}\n")
    f.write(f"- Spatial Uniformity: {uniformity_score:.3f}\n")
    f.write(f"- *Interpretation*: {distribution_metrics['interpretation']}\n\n")
    
    f.write("### 3. Geometric Consistency\n\n")
    f.write(f"- Mean Spacing: {spacing_stats['mean_spacing']:.2f} pixels\n")
    f.write(f"- Median Spacing: {spacing_stats['median_spacing']:.2f} pixels\n")
    f.write(f"- Spacing Regularity: {spacing_stats['spacing_regularity']:.3f}\n")
    f.write(f"- X Linearity (R²): {x_linearity['x_linearity_r2']:.3f}\n")
    f.write(f"- X Spacing Consistency: {x_linearity['x_spacing_consistency']:.3f}\n")
    f.write(f"- *Interpretation*: {geometric_metrics['interpretation']}\n\n")
    
    if ring_count:
        f.write("### 4. Ring Alignment\n\n")
        f.write(f"- Expected Rings: {ring_count}\n")
        f.write(f"- Expected Spacing: {ring_alignment_metrics.get('expected_spacing', 0):.2f} pixels\n")
        f.write(f"- Actual Mean Spacing: {ring_alignment_metrics.get('actual_mean_spacing', 0):.2f} pixels\n")
        f.write(f"- Spacing Accuracy: {ring_alignment_metrics.get('spacing_accuracy', 0):.3f}\n")
        f.write(f"- Point Count Ratio: {ring_alignment_metrics.get('point_count_ratio', 0):.2f}x\n")
        f.write(f"- Point Count Accuracy: {ring_alignment_metrics.get('point_count_accuracy', 0):.3f}\n")
        f.write(f"- *Interpretation*: {ring_alignment_metrics['interpretation']}\n\n")
    
    f.write("### 5. Detection Quality\n\n")
    f.write(f"- Coverage Efficiency: {coverage_efficiency:.3f}\n")
    f.write(f"- Detection Density: {detection_density:.4f} points/pixel²\n")
    if depth_map_estimate:
        f.write(f"- Estimated Depth Map Area: {depth_map_estimate['estimated_area']:.1f} pixels²\n")
    f.write(f"- *Interpretation*: {quality_metrics['interpretation']}\n\n")
    
    f.write("## Recommendations\n\n")
    if composite_score < 0.70:
        f.write("⚠️ **Quality issues detected**. Consider:\n")
        if scores['completeness'] < 0.7:
            f.write("- Adjusting line detection parameters to detect more points\n")
        if scores['quality_ratio'] < 0.7:
            f.write("- Improving line detection to reduce reliance on 'assume' or 'default' methods\n")
        if scores['spacing_regularity'] < 0.7:
            f.write("- Checking for missing detections or irregular spacing\n")
        if ring_count and scores.get('spacing_accuracy', 1.0) < 0.7:
            f.write("- Verifying detection alignment with expected ring structure\n")
    else:
        f.write("✓ **Detection quality is acceptable**. No major issues detected.\n")

print(f"✓ Markdown report saved to {markdown_path}")

# Print summary
print("\n" + "="*60)
print("DETECTING QUALITY SUMMARY")
print("="*60)
print(f"Composite Score: {composite_score:.3f} ({quality_rating})")
print(f"\nKey Metrics:")
print(f"  Points Detected:       {valid_count}")
print(f"  Quality Ratio:         {quality_ratio:.1%}")
print(f"  Spacing Regularity:    {spacing_stats['spacing_regularity']:.3f}")
print(f"  X Linearity:           {x_linearity['x_linearity_r2']:.3f}")
if ring_count:
    print(f"  Spacing Accuracy:      {ring_alignment_metrics.get('spacing_accuracy', 0):.3f}")
print("="*60)
print(f"\n✓ Evaluation complete. See {markdown_path} for detailed report.")


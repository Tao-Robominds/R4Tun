"""
Algorithm 3 Results Characterizer

This script analyzes the enhanced point cloud from Algorithm 3 (Geometry Guided Up Sampling)
and outputs comprehensive characteristics for analysis.

Key outputs:
- Enhanced point density analysis
- Upsampling quality assessment  
- Segmentation readiness metrics

Author: SAM4Tun Implementation Team
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import entropy
from typing import Dict, Tuple, List
import sys

def load_enhanced_data(tunnel_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load enhanced point cloud data from Algorithm 3"""
    base_dir = f"data/{tunnel_id}"
    
    # Load pre-enhancement and post-enhancement data
    enhanced_files = [
        os.path.join(base_dir, "enhanced.csv"),
    ]
    
    pre_enhancement_files = [
        os.path.join(base_dir, "denoised.csv"),
        os.path.join(base_dir, "unwrapped.csv")
    ]
    
    # Load enhanced data
    enhanced_df = None
    for file_path in enhanced_files:
        if os.path.exists(file_path):
            print(f"Loading enhanced data from: {file_path}")
            enhanced_df = pd.read_csv(file_path, comment='#')
            break
    
    # Load pre-enhancement data for comparison
    pre_enhanced_df = None
    for file_path in pre_enhancement_files:
        if os.path.exists(file_path):
            print(f"Loading pre-enhancement data from: {file_path}")
            pre_enhanced_df = pd.read_csv(file_path, comment='#')
            break
    
    if enhanced_df is None or pre_enhanced_df is None:
        raise FileNotFoundError(f"Enhanced or pre-enhancement data not found in {base_dir}")
    
    return enhanced_df, pre_enhanced_df

def analyze_enhanced_density(enhanced_df: pd.DataFrame, pre_enhanced_df: pd.DataFrame) -> Dict:
    """A. Enhanced Point Density Analysis"""
    print("  Analyzing enhanced point density...")
    
    # Basic counts
    total_enhanced = len(enhanced_df)
    total_pre_enhanced = len(pre_enhanced_df)
    enhancement_ratio = total_enhanced / total_pre_enhanced if total_pre_enhanced > 0 else 0.0
    
    # Filter valid points (pred != 0)
    if 'pred' in enhanced_df.columns:
        valid_enhanced = enhanced_df[enhanced_df['pred'] != 0]
        valid_pre_enhanced = pre_enhanced_df[pre_enhanced_df['pred'] != 0] if 'pred' in pre_enhanced_df.columns else pre_enhanced_df
    else:
        valid_enhanced = enhanced_df
        valid_pre_enhanced = pre_enhanced_df
    
    # Point density analysis
    if len(valid_enhanced) > 1000:
        sample_size = min(10000, len(valid_enhanced))
        sample_indices = np.random.choice(len(valid_enhanced), sample_size, replace=False)
        sample_coords = valid_enhanced.iloc[sample_indices][['h', 'theta', 'r']].values
        
        tree = cKDTree(sample_coords)
        distances, _ = tree.query(sample_coords, k=2)
        nn_distances = distances[:, 1]
    else:
        nn_distances = np.array([0.1])
    
    density_stats = {
        'total_points_before': int(total_pre_enhanced),
        'total_points_after': int(total_enhanced),
        'overall_enhancement_ratio': float(enhancement_ratio),
        'points_added': int(total_enhanced - total_pre_enhanced),
        'final_nn_distances': {
            'mean': float(np.mean(nn_distances)),
            'median': float(np.median(nn_distances)),
            'std': float(np.std(nn_distances)),
            'percentiles': {
                '25th': float(np.percentile(nn_distances, 25)),
                '75th': float(np.percentile(nn_distances, 75)),
                '95th': float(np.percentile(nn_distances, 95))
            }
        }
    }
    
    return density_stats

def analyze_upsampling_quality(enhanced_df: pd.DataFrame, pre_enhanced_df: pd.DataFrame) -> Dict:
    """B. Upsampling Quality Assessment"""
    print("  Analyzing upsampling quality...")
    
    # Surface coverage analysis
    valid_enhanced = enhanced_df[enhanced_df['pred'] != 0] if 'pred' in enhanced_df.columns else enhanced_df
    valid_pre_enhanced = pre_enhanced_df[pre_enhanced_df['pred'] != 0] if 'pred' in pre_enhanced_df.columns else pre_enhanced_df
    
    # Diameter estimation from enhanced point cloud
    diameter_estimation = {}
    if len(valid_enhanced) > 0 and 'r' in valid_enhanced.columns:
        r_values = valid_enhanced['r']
        diameter_estimation = {
            'inner_diameter': float(2 * r_values.min()),
            'outer_diameter': float(2 * r_values.max()),
            'average_diameter': float(2 * r_values.mean()),
            'median_diameter': float(2 * r_values.median()),
            'ring_thickness': float(r_values.max() - r_values.min()),
            'description': 'Estimated tunnel/ring diameters based on radial distances from enhanced point cloud'
        }
    
    # Check if we have enough data
    if len(valid_enhanced) == 0:
        return {
            'coverage_uniformity': 0.0,
            'improvement_effectiveness': 0.0,
            'remaining_sparse_percentage': 100.0,
            'diameter_estimation': diameter_estimation,
            'coverage_matrices': {
                'enhanced': [],
                'pre_enhanced': [],
                'improvement': []
            },
            'spatial_quality_metrics': {
                'coverage_entropy': 0.0,
                'improvement_variance': 0.0,
                'max_improvement_factor': 0.0
            }
        }
    
    # Coverage grid analysis
    h_range = [valid_enhanced['h'].min(), valid_enhanced['h'].max()]
    theta_range = [valid_enhanced['theta'].min(), valid_enhanced['theta'].max()]
    
    # Create coverage matrices
    h_bins = np.linspace(h_range[0], h_range[1], 20)  # 20x20 grid for analysis
    theta_bins = np.linspace(theta_range[0], theta_range[1], 20)
    
    # Enhanced coverage
    enhanced_coverage = np.zeros((len(h_bins)-1, len(theta_bins)-1))
    for i, (h_min, h_max) in enumerate(zip(h_bins[:-1], h_bins[1:])):
        for j, (theta_min, theta_max) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):
            mask = ((valid_enhanced['h'] >= h_min) & (valid_enhanced['h'] < h_max) & 
                   (valid_enhanced['theta'] >= theta_min) & (valid_enhanced['theta'] < theta_max))
            enhanced_coverage[i, j] = mask.sum()
    
    # Pre-enhanced coverage
    pre_enhanced_coverage = np.zeros((len(h_bins)-1, len(theta_bins)-1))
    if len(valid_pre_enhanced) > 0:
        for i, (h_min, h_max) in enumerate(zip(h_bins[:-1], h_bins[1:])):
            for j, (theta_min, theta_max) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):
                mask = ((valid_pre_enhanced['h'] >= h_min) & (valid_pre_enhanced['h'] < h_max) & 
                       (valid_pre_enhanced['theta'] >= theta_min) & (valid_pre_enhanced['theta'] < theta_max))
                pre_enhanced_coverage[i, j] = mask.sum()
    
    # Coverage improvement metrics
    coverage_improvement = enhanced_coverage - pre_enhanced_coverage
    relative_improvement = np.divide(coverage_improvement, pre_enhanced_coverage + 1, 
                                   out=np.zeros_like(coverage_improvement), where=(pre_enhanced_coverage + 1) != 0)
    
    # Quality metrics
    enhanced_flat = enhanced_coverage.flatten()
    enhanced_nonzero = enhanced_flat[enhanced_flat > 0]
    coverage_uniformity = 1 / (1 + np.std(enhanced_nonzero)) if len(enhanced_nonzero) > 0 else 0.0
    
    positive_improvements = relative_improvement[relative_improvement > 0]
    improvement_effectiveness = np.mean(positive_improvements) if len(positive_improvements) > 0 else 0.0
    
    # Remaining sparse areas
    sparse_threshold = np.percentile(enhanced_nonzero, 25) if len(enhanced_nonzero) > 0 else 0
    remaining_sparse_percentage = (np.sum(enhanced_coverage < sparse_threshold) / enhanced_coverage.size) * 100
    
    quality_stats = {
        'coverage_uniformity': float(coverage_uniformity),
        'improvement_effectiveness': float(improvement_effectiveness),
        'remaining_sparse_percentage': float(remaining_sparse_percentage),
        'diameter_estimation': diameter_estimation,
        'coverage_matrices': {
            'enhanced': enhanced_coverage.tolist(),
            'pre_enhanced': pre_enhanced_coverage.tolist(),
            'improvement': coverage_improvement.tolist()
        },
        'spatial_quality_metrics': {
            'coverage_entropy': float(entropy(enhanced_flat + 1)),  # +1 to avoid log(0)
            'improvement_variance': float(np.var(relative_improvement)),
            'max_improvement_factor': float(np.max(relative_improvement)) if relative_improvement.size > 0 else 0.0
        }
    }
    
    return quality_stats

def analyze_segmentation_readiness(enhanced_df: pd.DataFrame, density_stats: Dict) -> Dict:
    """C. Segmentation Readiness Metrics"""
    print("  Analyzing segmentation readiness...")
    
    valid_enhanced = enhanced_df[enhanced_df['pred'] != 0] if 'pred' in enhanced_df.columns else enhanced_df
    
    # Check if we have enough data
    if len(valid_enhanced) == 0:
        return {
            'overall_readiness_score': 0.0,
            'template_spacing_suitability': 0.0,
            'optimal_template_spacing': 0.05,
            'current_median_spacing': 1.0,
            'region_difficulty_assessment': {},
            'segmentation_complexity': {
                'easy_regions': 0,
                'moderate_regions': 0,
                'hard_regions': 0
            }
        }
    
    # Point distribution analysis for template matching
    median_spacing = density_stats['final_nn_distances']['median']
    
    # Template matching suitability
    optimal_template_spacing = 0.05  # Typical SAM template spacing
    spacing_suitability = min(1.0, optimal_template_spacing / median_spacing) if median_spacing > 0 else 0.0
    
    # Spatial uniformity analysis
    h_range = [valid_enhanced['h'].min(), valid_enhanced['h'].max()]
    theta_range = [valid_enhanced['theta'].min(), valid_enhanced['theta'].max()]
    
    # Create analysis grid
    h_bins = np.linspace(h_range[0], h_range[1], 10)  # 10x10 grid
    theta_bins = np.linspace(theta_range[0], theta_range[1], 10)
    
    region_densities = []
    region_difficulty = {}
    
    for i, (h_min, h_max) in enumerate(zip(h_bins[:-1], h_bins[1:])):
        for j, (theta_min, theta_max) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):
            mask = ((valid_enhanced['h'] >= h_min) & (valid_enhanced['h'] < h_max) & 
                   (valid_enhanced['theta'] >= theta_min) & (valid_enhanced['theta'] < theta_max))
            region_data = valid_enhanced[mask]
            
            if len(region_data) > 10:
                # Local density variation within region
                if len(region_data) > 100:
                    sample_coords = region_data[['h', 'theta', 'r']].values
                    tree = cKDTree(sample_coords)
                    distances, _ = tree.query(sample_coords, k=min(6, len(sample_coords)))
                    mean_distances = np.mean(distances[:, 1:], axis=1)
                    valid_distances = mean_distances[mean_distances > 0]  # Filter out zero distances
                    
                    if len(valid_distances) > 0:
                        local_densities = 1 / (valid_distances + 1e-6)
                        mean_density = np.mean(local_densities)
                        if mean_density > 0:
                            density_cv = np.std(local_densities) / mean_density
                        else:
                            density_cv = 0.5
                    else:
                        density_cv = 0.5
                else:
                    density_cv = 0.5  # Moderate difficulty for small regions
                
                region_densities.append(density_cv)
                
                # Difficulty classification
                if density_cv < 0.3:
                    difficulty = 'easy'
                elif density_cv < 0.6:
                    difficulty = 'moderate'
                else:
                    difficulty = 'hard'
                
                region_key = f"region_{i}_{j}"
                region_difficulty[region_key] = {
                    'density_coefficient_variation': float(density_cv),
                    'difficulty_level': difficulty,
                    'point_count': len(region_data),
                    'h_range': [float(h_min), float(h_max)],
                    'theta_range': [float(theta_min), float(theta_max)]
                }
    
    # Calculate coverage uniformity from region difficulty stats
    if len(region_densities) > 0:
        coverage_uniformity = 1 / (1 + np.std(region_densities))
    else:
        coverage_uniformity = 0.5  # Default moderate uniformity
    
    # Overall segmentation readiness score
    readiness_score = (spacing_suitability + coverage_uniformity) / 2
    
    segmentation_stats = {
        'overall_readiness_score': float(readiness_score),
        'template_spacing_suitability': float(spacing_suitability),
        'optimal_template_spacing': optimal_template_spacing,
        'current_median_spacing': median_spacing,
        'region_difficulty_assessment': region_difficulty,
        'segmentation_complexity': {
            'easy_regions': len([r for r in region_difficulty.values() if r['difficulty_level'] == 'easy']),
            'moderate_regions': len([r for r in region_difficulty.values() if r['difficulty_level'] == 'moderate']),
            'hard_regions': len([r for r in region_difficulty.values() if r['difficulty_level'] == 'hard'])
        }
    }
    
    return segmentation_stats

def save_algorithm3_characteristics(characteristics: Dict, base_dir: str):
    """Save all Algorithm 3 characteristics"""
    print("  Saving Algorithm 3 characteristics...")
    
    # Create characteristics subdirectory
    characteristics_dir = os.path.join(base_dir, 'characteristics')
    os.makedirs(characteristics_dir, exist_ok=True)
    
    # Save main characteristics
    characteristics_file = os.path.join(characteristics_dir, 'enhanced_characteristics.json')
    with open(characteristics_file, 'w') as f:
        json.dump(characteristics, f, indent=2, default=str)
    
    print(f"    Enhanced characteristics saved to: {characteristics_file}")

def characterize_algorithm3_results(tunnel_id: str) -> Dict:
    """Main function to characterize Algorithm 3 results"""
    
    print(f"\n=== Algorithm 3 Characterizer ===")
    print(f"Processing Tunnel ID: {tunnel_id}")
    print(f"=== Characterizing Algorithm 3 Results for Tunnel {tunnel_id} ===")
    
    base_dir = f"data/{tunnel_id}"
    print(f"Working directory: {base_dir}")
    
    # Load enhanced and pre-enhanced data
    enhanced_df, pre_enhanced_df = load_enhanced_data(tunnel_id)
    print(f"Enhanced: {len(enhanced_df):,} points, Pre-enhanced: {len(pre_enhanced_df):,} points")
    
    # Perform all analyses
    density_stats = analyze_enhanced_density(enhanced_df, pre_enhanced_df)
    quality_stats = analyze_upsampling_quality(enhanced_df, pre_enhanced_df)
    segmentation_stats = analyze_segmentation_readiness(enhanced_df, density_stats)
    
    # Compile comprehensive characteristics
    characteristics = {
        'tunnel_id': tunnel_id,
        'algorithm_3_results': {
            'enhanced_density': density_stats,
            'upsampling_quality': quality_stats,
            'segmentation_readiness': segmentation_stats
        },
        'processing_metadata': {
            'tunnel_id': tunnel_id,
            'source_algorithm': 'Algorithm 3 - Geometry Guided Up Sampling',
            'output_directory': base_dir,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save all outputs
    save_algorithm3_characteristics(characteristics, base_dir)
    
    # Print summary
    print(f"\n=== Algorithm 3 Results Summary ===")
    print(f"Enhancement ratio: {density_stats['overall_enhancement_ratio']:.2f}x")
    print(f"Points added: {density_stats['points_added']:,}")
    print(f"Segmentation readiness: {segmentation_stats['overall_readiness_score']:.2f}")
    print(f"Coverage uniformity: {quality_stats['coverage_uniformity']:.2f}")
    print(f"Improvement effectiveness: {quality_stats['improvement_effectiveness']:.2f}")
    
    return characteristics

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        tunnel_id = sys.argv[1]
    else:
        tunnel_id = "4-1"  # Default
    
    print(f"\n=== Starting Algorithm 3 Analysis ===")
    print(f"Target Tunnel ID: {tunnel_id}")
    
    # Run characterization
    characteristics = characterize_algorithm3_results(tunnel_id)
    
    print(f"\n✅ Algorithm 3 characterization complete for tunnel {tunnel_id}")
    print(f"📁 Results saved in: data/{tunnel_id}/characteristics/")
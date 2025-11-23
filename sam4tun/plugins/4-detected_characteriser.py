"""
Algorithm 4 Results Characterizer - SAM Prompt Effectiveness Analysis

This script analyzes how effectively the detection points from Algorithm 4 (Depth Image Detection)
serve as SAM prompt centers for segmenting the enhanced point cloud from Algorithm 3.

Key analysis:
- Detection point distribution as SAM template centers
- Prompt effectiveness for enhanced point cloud segmentation
- SAM parameter optimization based on prompt-to-target relationship

Data relationship:
- detected.csv: Detection points used as SAM template centers (2D depth map coordinates)
- enhanced.csv: Target point cloud for receiving segmentation labels (3D coordinates)
- Process: Detection Points → SAM Templates → 2D Segmentation → Project to Enhanced Cloud

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

def load_sam_workflow_data(tunnel_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load detection points (SAM prompts) and enhanced point cloud (SAM target)"""
    base_dir = f"data/{tunnel_id}"
    
    # Load detection points (SAM template centers)
    detection_points_file = os.path.join(base_dir, "detected.csv")
    
    # Load enhanced point cloud (SAM segmentation target)
    enhanced_file = os.path.join(base_dir, "enhanced.csv")
    
    # Load detection points
    if not os.path.exists(detection_points_file):
        raise FileNotFoundError(f"Detection points data not found: {detection_points_file}")
    
    print(f"Loading detection points (SAM prompts) from: {detection_points_file}")
    detection_points_df = pd.read_csv(detection_points_file, comment='#')
    
    # Load enhanced point cloud
    if not os.path.exists(enhanced_file):
        raise FileNotFoundError(
            f"Enhanced point cloud not found: {enhanced_file}\n"
            f"This script analyzes SAM prompt effectiveness for enhanced point cloud segmentation.\n"
            f"Please ensure Algorithm 3 (Geometry Guided Up Sampling) has been run to generate enhanced.csv"
        )
    
    print(f"Loading enhanced point cloud (SAM target) from: {enhanced_file}")
    enhanced_df = pd.read_csv(enhanced_file, comment='#')
    
    return detection_points_df, enhanced_df

def analyze_prompt_distribution_for_sam(detection_points_df: pd.DataFrame) -> Dict:
    """A. Detection Point Distribution Analysis for SAM Template Generation"""
    print("  Analyzing detection points as SAM template centers...")
    
    # Basic statistics
    total_points = len(detection_points_df)
    
    # Detection points should have X, Y coordinates and Type information
    if not ('X' in detection_points_df.columns and 'Y' in detection_points_df.columns):
        raise ValueError(f"Expected X, Y coordinates in detection points. Found: {detection_points_df.columns.tolist()}")
    
    valid_points = detection_points_df.dropna(subset=['X', 'Y'])
    valid_count = len(valid_points)
    
    if valid_count == 0:
        return {
            'total_detection_points': int(total_points),
            'valid_detection_points': 0,
            'sam_template_distribution': {},
            'prompt_spacing_analysis': {},
            'type_distribution': {}
        }
    
    # Analyze detection point types (important for SAM template selection)
    type_distribution = valid_points['Type'].value_counts().to_dict() if 'Type' in valid_points.columns else {}
    
    # Spatial bounds for SAM template coverage
    spatial_bounds = {
        'x_range': [float(valid_points['X'].min()), float(valid_points['X'].max())],
        'y_range': [float(valid_points['Y'].min()), float(valid_points['Y'].max())]
    }
    
    # Calculate coverage area for SAM templates
    coverage_area = (spatial_bounds['x_range'][1] - spatial_bounds['x_range'][0]) * \
                   (spatial_bounds['y_range'][1] - spatial_bounds['y_range'][0])
    
    # Prompt spacing analysis (critical for SAM performance)
    if valid_count > 1:
        coords = valid_points[['X', 'Y']].values
        tree = cKDTree(coords)
        
        # Nearest neighbor distances between prompts
        distances, _ = tree.query(coords, k=min(3, valid_count))
        nn_distances = distances[:, 1] if distances.shape[1] > 1 else np.array([0.1])
        
        # Template overlap analysis
        # Standard SAM template size is roughly 1250x3240 pixels (from generate_template_mask)
        template_spacing_adequacy = np.mean(nn_distances) / 1250  # Ratio to template width
        
        spacing_stats = {
            'mean_prompt_spacing': float(np.mean(nn_distances)),
            'median_prompt_spacing': float(np.median(nn_distances)),
            'min_prompt_spacing': float(np.min(nn_distances)),
            'max_prompt_spacing': float(np.max(nn_distances)),
            'spacing_std': float(np.std(nn_distances)),
            'template_spacing_ratio': float(template_spacing_adequacy),
            'potential_template_overlap': float(np.sum(nn_distances < 1250) / len(nn_distances))
        }
    else:
        spacing_stats = {
            'mean_prompt_spacing': 0.0,
            'median_prompt_spacing': 0.0,
            'min_prompt_spacing': 0.0,
            'max_prompt_spacing': 0.0,
            'spacing_std': 0.0,
            'template_spacing_ratio': 0.0,
            'potential_template_overlap': 0.0
        }
    
    # Coverage grid analysis for SAM template distribution
    x_range = spatial_bounds['x_range']
    y_range = spatial_bounds['y_range']
    
    # Create grid to assess template coverage uniformity
    if x_range[1] > x_range[0] and y_range[1] > y_range[0]:
        x_bins = np.linspace(x_range[0], x_range[1], 8)  # 8x8 grid for coverage
        y_bins = np.linspace(y_range[0], y_range[1], 8)
        
        coverage_matrix = np.zeros((7, 7))
        for i, (x_min, x_max) in enumerate(zip(x_bins[:-1], x_bins[1:])):
            for j, (y_min, y_max) in enumerate(zip(y_bins[:-1], y_bins[1:])):
                mask = ((valid_points['X'] >= x_min) & (valid_points['X'] < x_max) & 
                       (valid_points['Y'] >= y_min) & (valid_points['Y'] < y_max))
                coverage_matrix[i, j] = mask.sum()
        
        # Coverage uniformity for SAM
        covered_cells = np.sum(coverage_matrix > 0)
        total_cells = coverage_matrix.size
        coverage_percentage = (covered_cells / total_cells) * 100
        
        non_zero_coverage = coverage_matrix[coverage_matrix > 0]
        coverage_uniformity = 1 / (1 + np.std(non_zero_coverage)) if len(non_zero_coverage) > 0 else 0.0
    else:
        coverage_matrix = np.zeros((7, 7))
        coverage_percentage = 0.0
        coverage_uniformity = 0.0
        covered_cells = 0
        total_cells = 49
    
    distribution_stats = {
        'total_detection_points': int(total_points),
        'valid_detection_points': int(valid_count),
        'sam_template_distribution': {
            'spatial_bounds': spatial_bounds,
            'coverage_area': float(coverage_area),
            'prompt_density': float(valid_count / coverage_area) if coverage_area > 0 else 0.0,
            'coverage_matrix': coverage_matrix.tolist(),
            'coverage_percentage': float(coverage_percentage),
            'coverage_uniformity': float(coverage_uniformity),
            'uncovered_regions': int(total_cells - covered_cells)
        },
        'prompt_spacing_analysis': spacing_stats,
        'type_distribution': {str(k): int(v) for k, v in type_distribution.items()}
    }
    
    return distribution_stats

def analyze_sam_prompt_effectiveness(detection_points_df: pd.DataFrame, enhanced_df: pd.DataFrame) -> Dict:
    """B. SAM Prompt Effectiveness for Enhanced Point Cloud Segmentation"""
    print("  Analyzing SAM prompt effectiveness for enhanced point cloud...")
    
    # Get valid detection points
    valid_detection = detection_points_df.dropna(subset=['X', 'Y'])
    
    # Get valid enhanced points (those that will receive segmentation labels)
    if 'pred' in enhanced_df.columns:
        # Focus on points that are currently labeled as tunnels (pred == 7) 
        # These are the target points for SAM segmentation
        target_points = enhanced_df[enhanced_df['pred'] == 7]
    else:
        target_points = enhanced_df
    
    if len(valid_detection) == 0 or len(target_points) == 0:
        return {
            'prompt_to_target_ratio': 0.0,
            'diameter_estimation': {},
            'sam_coverage_analysis': {},
            'segmentation_effectiveness': {},
            'workflow_assessment': {}
        }
    
    # Basic ratio analysis
    prompt_to_target_ratio = len(valid_detection) / len(target_points)
    
    print(f"    Detection points (prompts): {len(valid_detection)}")
    print(f"    Target points (tunnel points): {len(target_points)}")
    print(f"    Prompt-to-target ratio: {prompt_to_target_ratio:.6f}")
    
    # SAM coverage analysis
    # Since detection points are in 2D (depth map) and enhanced points are in 3D,
    # we analyze the coverage conceptually
    
    # Calculate expected SAM template coverage
    # Each detection point generates a template of ~1250x3240 pixels
    template_area = 1250 * 3240  # pixels
    total_detection_coverage = len(valid_detection) * template_area
    
    # Estimate depth map area (rough approximation)
    detection_bounds = {
        'x_range': [valid_detection['X'].min(), valid_detection['X'].max()],
        'y_range': [valid_detection['Y'].min(), valid_detection['Y'].max()]
    }
    depth_map_area = (detection_bounds['x_range'][1] - detection_bounds['x_range'][0]) * \
                     (detection_bounds['y_range'][1] - detection_bounds['y_range'][0])
    
    # Coverage efficiency
    coverage_efficiency = min(1.0, total_detection_coverage / depth_map_area) if depth_map_area > 0 else 0.0
    
    # Analyze prompt distribution adequacy
    prompt_types = valid_detection['Type'].value_counts() if 'Type' in valid_detection.columns else {}
    
    # Different prompt types serve different tunnel structures
    expected_types = ['midpoint', 'negative_slope', 'positive_slope', 'default', 'assume']
    type_coverage = len(set(prompt_types.keys()) & set(expected_types)) / len(expected_types)
    
    # Enhanced point cloud characteristics
    enhanced_spatial_range = {
        'h_range': [target_points['h'].min(), target_points['h'].max()] if 'h' in target_points.columns else [0, 0],
        'theta_range': [target_points['theta'].min(), target_points['theta'].max()] if 'theta' in target_points.columns else [0, 0],
        'r_range': [target_points['r'].min(), target_points['r'].max()] if 'r' in target_points.columns else [0, 0]
    }
    
    # Diameter estimation from enhanced point cloud (target for SAM segmentation)
    diameter_estimation = {}
    if len(target_points) > 0 and 'r' in target_points.columns:
        r_values = target_points['r']
        diameter_estimation = {
            'inner_diameter': float(2 * r_values.min()),
            'outer_diameter': float(2 * r_values.max()),
            'average_diameter': float(2 * r_values.mean()),
            'median_diameter': float(2 * r_values.median()),
            'ring_thickness': float(r_values.max() - r_values.min()),
            'description': 'Estimated tunnel/ring diameters based on radial distances from enhanced point cloud (SAM segmentation target)'
        }
    
    effectiveness_stats = {
        'prompt_to_target_ratio': float(prompt_to_target_ratio),
        'diameter_estimation': diameter_estimation,
        'sam_coverage_analysis': {
            'detection_bounds': detection_bounds,
            'enhanced_spatial_range': enhanced_spatial_range,
            'estimated_template_coverage': float(coverage_efficiency),
            'prompt_type_coverage': float(type_coverage),
            'total_template_area': float(total_detection_coverage),
            'estimated_depth_map_area': float(depth_map_area)
        },
        'segmentation_effectiveness': {
            'prompt_type_distribution': {str(k): int(v) for k, v in prompt_types.items()},
            'target_point_count': int(len(target_points)),
            'points_per_prompt': float(len(target_points) / len(valid_detection)),
            'expected_segmentation_load': 'high' if len(target_points) / len(valid_detection) > 100000 else 
                                         'medium' if len(target_points) / len(valid_detection) > 50000 else 'low'
        },
        'workflow_assessment': {
            'coordinate_system_compatibility': '2D_prompts_to_3D_targets',
            'projection_required': True,
            'pixel_to_point_mapping_critical': True
        }
    }
    
    return effectiveness_stats

def analyze_sam_configuration_optimization(detection_stats: Dict, effectiveness_stats: Dict) -> Dict:
    """C. SAM Configuration Optimization Based on Prompt Analysis"""
    print("  Analyzing optimal SAM configuration...")
    
    # Extract key metrics
    num_prompts = detection_stats['valid_detection_points']
    template_spacing_ratio = detection_stats['prompt_spacing_analysis']['template_spacing_ratio']
    coverage_percentage = detection_stats['sam_template_distribution']['coverage_percentage']
    prompt_to_target_ratio = effectiveness_stats['prompt_to_target_ratio']
    
    # SAM parameter optimization based on prompt analysis
    
    # Template overlap assessment
    if template_spacing_ratio < 0.8:  # Significant overlap
        template_overlap_level = 'high'
        overlap_score = 0.6
    elif template_spacing_ratio < 1.5:  # Some overlap
        template_overlap_level = 'moderate'
        overlap_score = 0.8
    else:  # Good spacing
        template_overlap_level = 'minimal'
        overlap_score = 1.0
    
    # Coverage adequacy
    if coverage_percentage > 80:
        coverage_adequacy = 'excellent'
        coverage_score = 1.0
    elif coverage_percentage > 60:
        coverage_adequacy = 'good'
        coverage_score = 0.8
    elif coverage_percentage > 40:
        coverage_adequacy = 'moderate'
        coverage_score = 0.6
    else:
        coverage_adequacy = 'poor'
        coverage_score = 0.4
    
    # Overall SAM readiness
    readiness_score = (overlap_score + coverage_score) / 2
    
    # SAM parameter recommendations
    if readiness_score > 0.8:
        sam_strategy = 'template_based_segmentation'
        mask_threshold = 0.5
        stability_threshold = 0.95
        confidence = 'high'
    elif readiness_score > 0.6:
        sam_strategy = 'adaptive_template_segmentation'
        mask_threshold = 0.4
        stability_threshold = 0.90
        confidence = 'medium'
    else:
        sam_strategy = 'enhanced_preprocessing_required'
        mask_threshold = 0.3
        stability_threshold = 0.85
        confidence = 'low'
    
    # Template-specific recommendations
    template_recommendations = {
        'use_multi_scale_templates': template_spacing_ratio < 1.0,
        'template_size_optimization': 'reduce' if template_spacing_ratio < 0.5 else 'standard',
        'overlap_handling': 'required' if template_overlap_level == 'high' else 'minimal',
        'parallel_processing_suitable': num_prompts > 5 and readiness_score > 0.7
    }
    
    optimization_stats = {
        'overall_sam_readiness_score': float(readiness_score),
        'template_analysis': {
            'spacing_ratio': float(template_spacing_ratio),
            'overlap_level': template_overlap_level,
            'overlap_score': float(overlap_score),
            'coverage_adequacy': coverage_adequacy,
            'coverage_score': float(coverage_score)
        },
        'recommended_sam_strategy': {
            'strategy': sam_strategy,
            'confidence_level': confidence,
            'mask_threshold': mask_threshold,
            'stability_threshold': stability_threshold
        },
        'template_optimization': template_recommendations,
        'processing_recommendations': {
            'batch_processing': num_prompts > 3,
            'sequential_processing': num_prompts <= 3,
            'memory_optimization_needed': prompt_to_target_ratio < 0.00001,  # Very high target load
            'gpu_acceleration_beneficial': True
        }
    }
    
    return optimization_stats

def save_sam_analysis_characteristics(characteristics: Dict, base_dir: str):
    """Save SAM analysis characteristics"""
    print("  Saving SAM analysis characteristics...")
    
    # Create characteristics subdirectory
    characteristics_dir = os.path.join(base_dir, 'characteristics')
    os.makedirs(characteristics_dir, exist_ok=True)
    
    # Save main characteristics
    characteristics_file = os.path.join(characteristics_dir, 'detected_characteristics.json')
    with open(characteristics_file, 'w') as f:
        json.dump(characteristics, f, indent=2, default=str)
    
    print(f"    Detected characteristics saved to: {characteristics_file}")

def characterize_sam_workflow(tunnel_id: str) -> Dict:
    """Main function to characterize SAM workflow effectiveness"""
    
    print(f"\n=== Algorithm 4 Characterizer ===")
    print(f"Processing Tunnel ID: {tunnel_id}")
    print(f"\n=== Characterizing SAM Workflow for Tunnel {tunnel_id} ===")
    print("Analyzing detection points as SAM prompts for enhanced point cloud segmentation")
    
    base_dir = f"data/{tunnel_id}"
    print(f"Working directory: {base_dir}")
    
    # Load SAM workflow data
    detection_points_df, enhanced_df = load_sam_workflow_data(tunnel_id)
    print(f"Detection points (SAM prompts): {len(detection_points_df):,} points")
    print(f"Enhanced point cloud (SAM target): {len(enhanced_df):,} points")
    
    # Perform SAM-specific analyses
    detection_stats = analyze_prompt_distribution_for_sam(detection_points_df)
    effectiveness_stats = analyze_sam_prompt_effectiveness(detection_points_df, enhanced_df)
    optimization_stats = analyze_sam_configuration_optimization(detection_stats, effectiveness_stats)
    
    # Compile comprehensive characteristics
    characteristics = {
        'tunnel_id': tunnel_id,
        'sam_workflow_analysis': {
            'prompt_distribution': detection_stats,
            'prompt_effectiveness': effectiveness_stats,
            'sam_optimization': optimization_stats
        },
        'processing_metadata': {
            'tunnel_id': tunnel_id,
            'analysis_type': 'SAM Workflow Effectiveness Analysis',
            'prompt_source': 'Algorithm 4 - Depth Image Detection',
            'target_data': 'Algorithm 3 - Enhanced Point Cloud',
            'workflow': 'Detection Points → SAM Templates → 2D Segmentation → 3D Projection',
            'output_directory': base_dir,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save all outputs
    save_sam_analysis_characteristics(characteristics, base_dir)
    
    # Print summary
    print(f"\n=== SAM Workflow Analysis Summary ===")
    optimization = characteristics['sam_workflow_analysis']['sam_optimization']
    print(f"Detection prompts: {detection_stats['valid_detection_points']}")
    print(f"Template coverage: {detection_stats['sam_template_distribution']['coverage_percentage']:.1f}%")
    print(f"SAM readiness: {optimization['overall_sam_readiness_score']:.2f}")
    print(f"Template strategy: {optimization['recommended_sam_strategy']['strategy']}")
    print(f"Confidence level: {optimization['recommended_sam_strategy']['confidence_level']}")
    print(f"Template overlap: {optimization['template_analysis']['overlap_level']}")
    
    return characteristics

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        tunnel_id = sys.argv[1]
    else:
        tunnel_id = "5-1"  # Default
    
    print(f"\n=== Starting Algorithm 4 Analysis ===")
    print(f"Target Tunnel ID: {tunnel_id}")
    
    # Run characterization
    characteristics = characterize_sam_workflow(tunnel_id)
    
    print(f"\n✅ SAM workflow analysis complete for tunnel {tunnel_id}")
    print(f"📁 Results saved in: data/{tunnel_id}/characteristics/") 
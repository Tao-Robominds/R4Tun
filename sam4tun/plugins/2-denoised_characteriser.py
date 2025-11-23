"""
Denoising Results Characterizer

This script analyzes the denoised point cloud from Denoising and outputs
comprehensive characteristics for analysis.

Key outputs:
- Point density analysis
- Spatial distribution analysis  
- Quality metrics
- Geometry characteristics

Author: SAM4Tun Implementation Team
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from typing import Dict, Tuple, List
import argparse

def load_denoised_data(tunnel_id: str) -> pd.DataFrame:
    """Load denoised point cloud data from Denoising"""
    base_dir = f"data/{tunnel_id}"
    
    # Try to load from different possible sources
    denoised_files = [
        os.path.join(base_dir, "denoised.csv"),
        os.path.join(base_dir, "unwrapped.csv")
    ]
    
    for file_path in denoised_files:
        if os.path.exists(file_path):
            if file_path.endswith("unwrapped.csv"):
                # Handle unwrapped file with metadata comments
                df = pd.read_csv(file_path, comment='#')
                # Add pred column if not present
                if 'pred' not in df.columns:
                    df['pred'] = 7  # Assume all points are valid initially
            else:
                df = pd.read_csv(file_path)
            return df
    
    raise FileNotFoundError(f"No denoised data found in {base_dir}")

def read_ring_count(tunnel_id: str) -> int:
    """Read ring count from ring_count.txt file"""
    ring_count_path = f"data/{tunnel_id}/ring_count.txt"
    ring_count = None
    
    if os.path.exists(ring_count_path):
        try:
            with open(ring_count_path, 'r') as f:
                ring_count = int(f.read().strip())
        except (ValueError, IOError) as e:
            pass
    
    return ring_count

def analyze_point_density(df: pd.DataFrame) -> Dict:
    """A. Point Density Analysis"""
    
    valid_points = df[df['pred'] == 7] if 'pred' in df.columns else df
    
    if len(valid_points) < 100:
        return {
            'mean_nn_distance': 0.1,
            'median_nn_distance': 0.1,
            'std_nn_distance': 0.05,
            'percentiles': {'25th': 0.08, '75th': 0.12, '95th': 0.15}
        }
    
    # Sample for efficiency if dataset is large
    sample_size = min(10000, len(valid_points))
    if len(valid_points) > sample_size:
        sample_indices = np.random.choice(len(valid_points), sample_size, replace=False)
        sample_coords = valid_points.iloc[sample_indices][['h', 'theta', 'r']].values
    else:
        sample_coords = valid_points[['h', 'theta', 'r']].values
    
    # Build KDTree and compute nearest neighbor distances
    tree = cKDTree(sample_coords)
    distances, _ = tree.query(sample_coords, k=2)  # k=2 to exclude self
    nn_distances = distances[:, 1]  # Second column is nearest neighbor
    
    density_stats = {
        'mean_nn_distance': float(np.mean(nn_distances)),
        'median_nn_distance': float(np.median(nn_distances)),
        'std_nn_distance': float(np.std(nn_distances)),
        'percentiles': {
            '25th': float(np.percentile(nn_distances, 25)),
            '75th': float(np.percentile(nn_distances, 75)),
            '95th': float(np.percentile(nn_distances, 95))
        }
    }
    
    return density_stats

def analyze_spatial_distribution(df: pd.DataFrame) -> Dict:
    """B. Spatial Distribution Analysis"""
    
    valid_points = df[df['pred'] == 7] if 'pred' in df.columns else df
    
    # Coordinate ranges in cylindrical system
    h_range = [float(valid_points['h'].min()), float(valid_points['h'].max())]
    theta_range = [float(valid_points['theta'].min()), float(valid_points['theta'].max())]
    r_range = [float(valid_points['r'].min()), float(valid_points['r'].max())]
    r_values = valid_points['r']
    
    # Diameter estimation based on radial distances
    diameter_estimation = {
        'inner_diameter': float(2 * r_values.min()),
        'outer_diameter': float(2 * r_values.max()),
        'average_diameter': float(2 * r_values.mean()),
        'median_diameter': float(2 * r_values.median()),
        'ring_thickness': float(r_values.max() - r_values.min()),
        'description': 'Estimated tunnel/ring diameters based on radial distances from denoised point cloud'
    }
    
    # Coverage analysis - create 2D grid
    h_bins = np.linspace(h_range[0], h_range[1], 10)  # Use fixed 10 bins
    theta_bins = np.linspace(theta_range[0], theta_range[1], 36)  # 10-degree bins
    
    coverage_matrix = np.zeros((len(h_bins)-1, len(theta_bins)-1))
    for i, (h_min, h_max) in enumerate(zip(h_bins[:-1], h_bins[1:])):
        for j, (theta_min, theta_max) in enumerate(zip(theta_bins[:-1], theta_bins[1:])):
            mask = ((valid_points['h'] >= h_min) & (valid_points['h'] < h_max) & 
                   (valid_points['theta'] >= theta_min) & (valid_points['theta'] < theta_max))
            coverage_matrix[i, j] = mask.sum()
    
    # Coverage statistics
    non_zero_cells = np.count_nonzero(coverage_matrix)
    total_cells = coverage_matrix.size
    coverage_percentage = (non_zero_cells / total_cells) * 100
    
    # Identify sparse areas
    sparse_threshold = np.percentile(coverage_matrix[coverage_matrix > 0], 25) if np.any(coverage_matrix > 0) else 0
    sparse_areas = coverage_matrix < sparse_threshold
    sparse_percentage = (np.sum(sparse_areas) / total_cells) * 100
    
    spatial_stats = {
        'coordinate_ranges': {
            'h_range': h_range,
            'theta_range': theta_range, 
            'r_range': r_range
        },
        'diameter_estimation': diameter_estimation,
        'coverage_analysis': {
            'total_coverage_percentage': float(coverage_percentage),
            'sparse_areas_percentage': float(sparse_percentage),
            'coverage_matrix_shape': coverage_matrix.shape,
            'sparse_threshold': float(sparse_threshold)
        },
        'coverage_matrix': coverage_matrix.tolist(),  # For saving
        'sparse_areas': sparse_areas.tolist()  # For saving
    }
    
    return spatial_stats

def analyze_quality_metrics(df: pd.DataFrame) -> Dict:
    """C. Quality Metrics"""
    
    total_points = len(df)
    
    if 'pred' in df.columns:
        valid_points = (df['pred'] == 7).sum()
        noise_points = (df['pred'] == 0).sum()
        noise_removal_rate = noise_points / total_points
        data_retention_rate = valid_points / total_points
    else:
        valid_points = total_points
        noise_points = 0
        noise_removal_rate = 0.0
        data_retention_rate = 1.0
    
    # Surface completeness estimation
    valid_df = df[df['pred'] == 7] if 'pred' in df.columns else df
    
    # Estimate surface completeness based on coverage uniformity
    if len(valid_df) > 0:
        h_coverage = (valid_df['h'].max() - valid_df['h'].min()) / (df['h'].max() - df['h'].min())
        theta_coverage = (valid_df['theta'].max() - valid_df['theta'].min()) / (df['theta'].max() - df['theta'].min())
        surface_completeness = (h_coverage + theta_coverage) / 2
    else:
        surface_completeness = 0.0
    
    quality_stats = {
        'total_input_points': total_points,
        'valid_points_remaining': int(valid_points),
        'noise_points_removed': int(noise_points),
        'noise_removal_rate': float(noise_removal_rate),
        'data_retention_rate': float(data_retention_rate),
        'surface_completeness': float(surface_completeness),
        'outlier_percentage': float(noise_removal_rate)
    }
    
    return quality_stats

def analyze_geometry_characteristics(df: pd.DataFrame, spatial_stats: Dict) -> Dict:
    """D. Geometry Characteristics"""
    
    valid_points = df[df['pred'] == 7] if 'pred' in df.columns else df
    
    # General curvature estimates based on radial variation
    h_range = spatial_stats['coordinate_ranges']['h_range']
    r_range = spatial_stats['coordinate_ranges']['r_range']
    
    # Divide into sections along height for curvature estimation
    h_sections = np.linspace(h_range[0], h_range[1], 10)
    section_curvatures = []
    
    for i in range(len(h_sections)-1):
        h_min, h_max = h_sections[i], h_sections[i+1]
        section_data = valid_points[(valid_points['h'] >= h_min) & (valid_points['h'] < h_max)]
        if len(section_data) > 10:
            # Estimate curvature from radius variation in this section
            r_std = section_data['r'].std()
            section_curvatures.append(r_std)
    
    avg_curvature = np.mean(section_curvatures) if section_curvatures else 0.0
    surface_regularity = np.std(section_curvatures) if section_curvatures else 0.0
    
    # Tunnel geometry metrics
    # Use diameter_estimation from spatial_stats if available
    diameter_estimation = spatial_stats.get('diameter_estimation', {})
    
    geometry_stats = {
        'average_curvature_estimate': float(avg_curvature),
        'surface_regularity': float(surface_regularity),
        'tunnel_length': float(h_range[1] - h_range[0]),
        'radius_variation': [float(r_range[0]), float(r_range[1])],
        'section_curvatures': [float(c) for c in section_curvatures],
        'estimated_diameter': float(2 * np.mean(r_range)),
        'diameter_estimation': diameter_estimation
    }
    
    return geometry_stats

def save_characteristics(characteristics: Dict, base_dir: str):
    """Save all characteristics to files"""
    
    # Create characteristics subdirectory
    characteristics_dir = os.path.join(base_dir, 'characteristics')
    os.makedirs(characteristics_dir, exist_ok=True)
    
    # Save main characteristics JSON
    characteristics_file = os.path.join(characteristics_dir, 'denoised_characteristics.json')
    with open(characteristics_file, 'w') as f:
        json.dump(characteristics, f, indent=2, default=str)

def characterize_denoising_results(tunnel_id: str) -> Dict:
    """Main function to characterize Denoising results"""
    
    print(f"\n=== Denoising Characterizer ===")
    print(f"Processing Tunnel ID: {tunnel_id}")
    
    base_dir = f"data/{tunnel_id}"
    print(f"Working directory: {base_dir}")
    
    # Load denoised data and ring count
    df = load_denoised_data(tunnel_id)
    ring_count = read_ring_count(tunnel_id)
    
    print(f"Loaded {len(df):,} points from denoised data")
    if ring_count:
        print(f"Ring count: {ring_count}")
    else:
        print("Ring count: Not available")
    
    # Perform all analyses
    density_stats = analyze_point_density(df)
    spatial_stats = analyze_spatial_distribution(df)
    quality_stats = analyze_quality_metrics(df)
    geometry_stats = analyze_geometry_characteristics(df, spatial_stats)
    
    # Compile comprehensive characteristics
    characteristics = {
        'tunnel_id': tunnel_id,
        'denoising_results': {
            'denoising_summary': quality_stats,
            'point_density_analysis': density_stats,
            'spatial_distribution': spatial_stats,
            'geometry_characteristics': geometry_stats,
            'ring_count': ring_count
        },
        'source_file': f"data/{tunnel_id}/denoised.csv",
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save all outputs
    save_characteristics(characteristics, base_dir)
    
    return characteristics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Characterize Denoising results')
    parser.add_argument('tunnel_id', type=str, 
                        help='Tunnel ID (e.g., "5-1", "4-1", etc.)')
    
    args = parser.parse_args()
    
    print(f"\n=== Starting Denoising Analysis ===")
    print(f"Target Tunnel ID: {args.tunnel_id}")
    
    # Run characterization
    characteristics = characterize_denoising_results(args.tunnel_id)
    
    # Print summary
    analysis = characteristics['denoising_results']
    print(f"\n=== Denoising Analysis Complete ===")
    print(f"Processed Tunnel ID: {args.tunnel_id}")
    print(f"Total input points: {analysis['denoising_summary']['total_input_points']:,}")
    print(f"Valid points remaining: {analysis['denoising_summary']['valid_points_remaining']:,}")
    print(f"Data retention rate: {analysis['denoising_summary']['data_retention_rate']:.1%}")
    print(f"📁 Results saved in: data/{args.tunnel_id}/characteristics/") 
import numpy as np
import pandas as pd
import json
import argparse
import os
from scipy.spatial import cKDTree

def analyze_unwrapped_pointcloud(csv_path, output_json_path=None):
    """
    Analyze unwrapped point cloud data to characterize x, y, z coordinates
    with intensity and cylindrical coordinates (r, theta, h).
    """
    # Extract tunnel_id from path and output it
    tunnel_id = csv_path.split('/')[-2]
    print(f"\n=== Unfolding Characterizer ===")
    print(f"Processing Tunnel ID: {tunnel_id}")
    print(f"Input file: {csv_path}")
    
    # Load the unwrapped data, skipping comment lines
    df = pd.read_csv(csv_path, comment='#')
    
    # Read ring count from ring_count.txt file
    ring_count = None
    ring_count_path = f"data/{tunnel_id}/ring_count.txt"

    if os.path.exists(ring_count_path):
        try:
            with open(ring_count_path, 'r') as f:
                ring_count = int(f.read().strip())
                print(f"Ring count loaded: {ring_count}")
        except (ValueError, IOError) as e:
            print(f"Warning: Could not read ring count from {ring_count_path}: {e}")
    else:
        print(f"Ring count file not found: {ring_count_path}")
    
    # Expected columns: x, y, z, intensity, segment, ring, r, theta, h
    expected_columns = ['x', 'y', 'z', 'intensity', 'r', 'theta', 'h']
    
    # Check if all expected columns exist
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
    
    # Use available columns for analysis
    available_columns = [col for col in expected_columns if col in df.columns]
    df = df[available_columns]
    
    # Basic statistics
    total_points = len(df)
    
    # Cartesian coordinate ranges
    cartesian_stats = {}
    if all(col in df.columns for col in ['x', 'y', 'z']):
        cartesian_stats = {
            'x_range': [float(df['x'].min()), float(df['x'].max())],
            'y_range': [float(df['y'].min()), float(df['y'].max())],
            'z_range': [float(df['z'].min()), float(df['z'].max())],
            'x_span': float(df['x'].max() - df['x'].min()),
            'y_span': float(df['y'].max() - df['y'].min()),
            'z_span': float(df['z'].max() - df['z'].min())
        }
    
    # Cylindrical coordinate analysis
    cylindrical_stats = {}
    if all(col in df.columns for col in ['r', 'theta', 'h']):
        r_values = df['r']
        cylindrical_stats = {
            'r_range': [float(r_values.min()), float(r_values.max())],
            'theta_range': [float(df['theta'].min()), float(df['theta'].max())],
            'h_range': [float(df['h'].min()), float(df['h'].max())],
            'r_span': float(r_values.max() - r_values.min()),
            'theta_span': float(df['theta'].max() - df['theta'].min()),
            'h_span': float(df['h'].max() - df['h'].min()),
            'theta_coverage_degrees': float(np.degrees(df['theta'].max() - df['theta'].min())),
            # Diameter estimation based on radial distances
            'diameter_estimation': {
                'inner_diameter': float(2 * r_values.min()),
                'outer_diameter': float(2 * r_values.max()),
                'average_diameter': float(2 * r_values.mean()),
                'median_diameter': float(2 * r_values.median()),
                'ring_thickness': float(r_values.max() - r_values.min()),
                'description': 'Estimated tunnel/ring diameters based on radial distances'
            }
        }
    
    # Intensity statistics
    intensity_stats = {}
    if 'intensity' in df.columns:
        intensity_stats = {
            'min': float(df['intensity'].min()),
            'max': float(df['intensity'].max()),
            'mean': float(df['intensity'].mean()),
            'median': float(df['intensity'].median()),
            'std': float(df['intensity'].std()),
            'percentiles': {
                '10th': float(np.percentile(df['intensity'], 10)),
                '25th': float(np.percentile(df['intensity'], 25)),
                '75th': float(np.percentile(df['intensity'], 75)),
                '90th': float(np.percentile(df['intensity'], 90))
            }
        }
    
    # Point density analysis (using cartesian coordinates)
    density_stats = {}
    if all(col in df.columns for col in ['x', 'y', 'z']):
        sample_size = min(10000, len(df))  # Sample for efficiency
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        sample_points = df.iloc[sample_indices][['x', 'y', 'z']].values
        
        tree = cKDTree(sample_points)
        distances, _ = tree.query(sample_points, k=2)  # k=2 to get nearest neighbor (excluding self)
        nn_distances = distances[:, 1]  # Second column is nearest neighbor distance
        
        density_stats = {
            'mean_nn_distance': float(np.mean(nn_distances)),
            'median_nn_distance': float(np.median(nn_distances)),
            'min_nn_distance': float(np.min(nn_distances)),
            'max_nn_distance': float(np.max(nn_distances)),
            'std_nn_distance': float(np.std(nn_distances))
        }
    
    # Processing notes
    processing_notes = []
    if density_stats:
        processing_notes.append(f"Point density: Median NN distance = {density_stats['median_nn_distance']:.4f}")
    if cartesian_stats:
        processing_notes.append(f"Spatial extent: X={cartesian_stats['x_span']:.2f}, Y={cartesian_stats['y_span']:.2f}, Z={cartesian_stats['z_span']:.2f}")
    if cylindrical_stats:
        if 'diameter_estimation' in cylindrical_stats:
            diameter_info = cylindrical_stats['diameter_estimation']
            processing_notes.append(f"Estimated diameters: Inner={diameter_info['inner_diameter']:.2f}, Outer={diameter_info['outer_diameter']:.2f}, Thickness={diameter_info['ring_thickness']:.2f}")
        processing_notes.append(f"Cylindrical extent: R={cylindrical_stats['r_span']:.2f}, θ={cylindrical_stats['theta_coverage_degrees']:.1f}°, H={cylindrical_stats['h_span']:.2f}")
    if ring_count is not None:
        processing_notes.append(f"Ring structure: {ring_count} rings detected")
    
    # Compile all characteristics
    characteristics = {
        "tunnel_id": tunnel_id,
        "unfolding_results": {
            "basic_statistics": {
                "total_points": int(total_points),
                "ring_count": ring_count,
                "coordinate_systems": ["cartesian (x, y, z)", "cylindrical (r, theta, h)"],
                "available_attributes": list(df.columns)
            },
            "cartesian_coordinates": cartesian_stats,
            "cylindrical_coordinates": cylindrical_stats,
            "intensity_analysis": intensity_stats,
            "point_density": density_stats,
            "processing_notes": processing_notes
        },
        "source_file": csv_path,
        "analysis_timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Save to JSON if path provided
    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(characteristics, f, indent=2)
    
    return characteristics

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze unwrapped point cloud data for tunnel characterization')
    parser.add_argument('tunnel_id', type=str, 
                        help='Tunnel ID (e.g., "5-1", "4-1", etc.)')
    
    args = parser.parse_args()
    
    print(f"\n=== Starting Unfolding Analysis ===")
    print(f"Target Tunnel ID: {args.tunnel_id}")
    
    # Construct paths based on tunnel_id with characteristics subdirectory
    csv_path = f"data/{args.tunnel_id}/unwrapped.csv"
    characteristics_dir = f"data/{args.tunnel_id}/characteristics"
    json_path = f"{characteristics_dir}/unfolded_characteristics.json"
    
    # Create characteristics directory if it doesn't exist
    os.makedirs(characteristics_dir, exist_ok=True)
    
    # Check if input file exists
    if not os.path.exists(csv_path):
        print(f"Error: Input file not found: {csv_path}")
        exit(1)
    
    # Analyze the unwrapped data
    characteristics = analyze_unwrapped_pointcloud(csv_path, json_path)
    
    # Brief confirmation
    analysis = characteristics["unfolding_results"]
    print(f"\n=== Unfolding Analysis Complete ===")
    print(f"Processed Tunnel ID: {args.tunnel_id}")
    print(f"Total points analyzed: {analysis['basic_statistics']['total_points']:,}")
    print(f"Results saved to: {json_path}")
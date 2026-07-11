import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import json
import os
import argparse
from scipy.spatial import cKDTree
from collections import Counter
import glob

from sam4tun.plugins.paths import tunnel_characteristics_dir

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def analyze_point_cloud(file_path, tunnel_id=None):
    # Read point cloud data
    point_cloud_data = np.loadtxt(file_path)
    points_xyz = point_cloud_data[:, :3]
    intensity = point_cloud_data[:, 3]
    # Note: We load segment and ring data but don't analyze them (ground truth exclusion)
    # segment = point_cloud_data[:, 4].astype(int)  # Not used in analysis
    # ring = point_cloud_data[:, 5].astype(int)     # Not used in analysis
    
    # Create DataFrame for easier analysis - only x, y, z, intensity
    df = pd.DataFrame({
        'x': points_xyz[:, 0],
        'y': points_xyz[:, 1],
        'z': points_xyz[:, 2],
        'intensity': intensity
    })
    
    # Basic statistics - updated for 4 columns only
    basic_stats = {
        "total_points": int(len(df)),
        "data_structure": {
            "columns": 4,  # Updated from 6 to 4
            "description": "x, y, z, intensity"  # Removed segment_type, ring_number
        },
        "coordinate_ranges": {
            "x_range": [float(df['x'].min()), float(df['x'].max())],
            "y_range": [float(df['y'].min()), float(df['y'].max())],
            "z_range": [float(df['z'].min()), float(df['z'].max())],
            "intensity_range": [float(df['intensity'].min()), float(df['intensity'].max())]
        }
    }
    
    # Tunnel geometry analysis (based on x,y,z coordinates only)
    points_2d_xoy = points_xyz[:, :2]
    convex_hull = ConvexHull(points_2d_xoy)
    convex_hull_points = points_2d_xoy[convex_hull.vertices]
    convex_polygon = Polygon(convex_hull_points)
    min_bounding_rect = convex_polygon.minimum_rotated_rectangle
    
    # Get dimensions from minimum bounding rectangle
    rect_vertices = np.array(min_bounding_rect.exterior.coords)[:-1]
    edges = [np.linalg.norm(rect_vertices[i] - rect_vertices[(i + 1) % 4]) for i in range(4)]
    length = max(edges)
    width = min(edges)
    height = df['z'].max() - df['z'].min()
    
    # Diameter estimation based on geometry analysis
    diameter_estimation = {
        "inner_diameter": float(width),
        "outer_diameter": float(width),
        "average_diameter": float(width),
        "median_diameter": float(width),
        "ring_thickness": 0.0,
        "description": "Estimated tunnel diameter based on minimum bounding rectangle width (2D XOY projection). May include surrounding infrastructure.",
        "method": "minimum_bounding_rectangle",
        "note": "This is a 2D projection-based estimate. For more accurate diameter estimation, use cylindrical coordinate analysis (r values) from unfolded point cloud."
    }
    
    tunnel_geometry = {
        "dimensions": {
            "length_x_axis": float(length),
            "width_y_axis": float(width),
            "height_z_axis": float(height),
            "units": "meters"
        },
        "estimated_diameter": float(width),
        "diameter_estimation": diameter_estimation,
        "actual_tunnel_diameter": 5.5,
        "diameter_discrepancy_note": "Estimated diameter may include surrounding infrastructure"
    }
    
    # Point density analysis (based on x,y,z coordinates only)
    tree = cKDTree(points_xyz)
    distances, _ = tree.query(points_xyz, k=2)  # k=2 because first point is the point itself
    nearest_distances = distances[:, 1]
    
    point_density = {
        "mean_nearest_neighbor_distance": float(np.mean(nearest_distances)),
        "median_nearest_neighbor_distance": float(np.median(nearest_distances)),
        "min_nearest_neighbor_distance": float(np.min(nearest_distances)),
        "max_nearest_neighbor_distance": float(np.max(nearest_distances)),
        "units": "meters"
    }
    

    
    # Compile results with only non-ground truth characteristics
    results = {
        "tunnel_id": tunnel_id if tunnel_id else "unknown",
        "input_file": file_path,
        "filtered_note": "Contains only characteristics for x, y, z, intensity columns. Ground truth data (segment_type, ring_number) excluded.",
        "point_cloud_analysis": {
            "basic_statistics": basic_stats,
            "tunnel_geometry": tunnel_geometry,
            "point_density": point_density           # Note: segment_analysis section completely removed
        }
    }
    
    return results

def process_all_datasets(data_dir='data'):
    """Process all .txt files in data_dir; write JSON via tunnel_characteristics_dir (see sam4tun.plugins.paths)."""
    
    # Find all .txt files in the data directory
    pattern = os.path.join(data_dir, '*.txt')
    data_files = glob.glob(pattern)
    
    if not data_files:
        print(f"No .txt files found in {data_dir}")
        return
    
    print(f"Found {len(data_files)} datasets to process")
    print("Note: Generating characteristics for x, y, z, intensity only (excluding ground truth data)")
    print("Output: data/sample/characteristics/ for sample; data/ablation/memory/{tunnel_id}/characteristics/ otherwise")
    
    # Process each dataset
    all_results = {}
    
    for data_file in data_files:
        # Extract tunnel ID from filename (remove path and extension)
        tunnel_id = os.path.splitext(os.path.basename(data_file))[0]
        
        print(f"Processing {tunnel_id}...")
        
        try:
            results = analyze_point_cloud(data_file, tunnel_id)
            characteristics_dir = tunnel_characteristics_dir(tunnel_id)
            os.makedirs(characteristics_dir, exist_ok=True)
            output_file = os.path.join(characteristics_dir, "raw_characteristics.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
            
            print(f"✓ {tunnel_id} - Non-GT characteristics saved to {output_file}")
            all_results[tunnel_id] = results
            
        except Exception as e:
            print(f"✗ Error processing {tunnel_id}: {str(e)}")
            continue
    

    
    print(f"\nProcessing complete!")
    print(f"Individual results under data/sample/characteristics/ or data/ablation/memory/[tunnel_id]/characteristics/")
    print(f"Total datasets processed: {len(all_results)}")
    print(f"📊 Generated characteristics include:")
    print(f"   ✓ Basic statistics (total_points, coordinate_ranges, intensity_range)")
    print(f"   ✓ Tunnel geometry (dimensions, diameter estimates)")
    print(f"   ✓ Point density analysis (nearest neighbor distances)")
    print(f"🚫 Excluded ground truth characteristics:")
    print(f"   ✗ Segment analysis (all segment_type related statistics)")
    print(f"   ✗ Ring number references")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze point cloud characteristics for x, y, z, intensity only (excluding ground truth data)',
        epilog="""
This script generates characteristics for x, y, z, and intensity columns only,
excluding all ground truth data (segment_type, ring_number) from analysis.

Examples:
  # Process specific tunnel
  python raw_characteristics.py --tunnel_id 1-4
  
  # Process all datasets in data directory
  python raw_characteristics.py
  
  # Process all .txt in a directory (paths follow tunnel_id; sample → data/sample/characteristics)
  python raw_characteristics.py --data_dir custom_data
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--tunnel_id', type=str, help='Specific tunnel ID to process (e.g., 3-1). If not provided, processes all datasets.')
    parser.add_argument('--data_dir', type=str, default='data', help='Base directory for data files (default: data)')
    
    args = parser.parse_args()
    
    print("🔍 Point Cloud Characteristics Analyzer (Non-Ground Truth Mode)")
    print("📋 Will analyze: x, y, z, intensity characteristics only")
    print("🚫 Will exclude: segment_type, ring_number (ground truth data)")
    print()
    
    if args.tunnel_id:
        # Process single tunnel - read from data_dir/tunnel_id.txt (e.g. data/sample.txt for tunnel_id sample)
        data_path = os.path.join(args.data_dir, f"{args.tunnel_id}.txt")
        
        if not os.path.exists(data_path):
            print(f"❌ Error: Input file not found at {data_path}")
            return
        
        print(f"🔬 Analyzing point cloud for tunnel {args.tunnel_id} (x,y,z,intensity only)...")
        results = analyze_point_cloud(data_path, args.tunnel_id)
        characteristics_dir = tunnel_characteristics_dir(args.tunnel_id)
        os.makedirs(characteristics_dir, exist_ok=True)
        output_file = os.path.join(characteristics_dir, "raw_characteristics.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        print(f"✅ Analysis complete. Non-ground truth characteristics saved to {output_file}")
        print(f"📊 Total points analyzed: {results['point_cloud_analysis']['basic_statistics']['total_points']:,}")
    
    else:
        # Process all datasets
        process_all_datasets(args.data_dir)

if __name__ == "__main__":
    main() 
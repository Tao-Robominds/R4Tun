import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import json
import os
import argparse
from scipy.spatial import cKDTree
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

def _fit_circle_3pts(p):
    """Fit a circle through 3 points. Returns (cx, cy, r) or None."""
    A = np.array([
        [2 * p[0, 0], 2 * p[0, 1], 1],
        [2 * p[1, 0], 2 * p[1, 1], 1],
        [2 * p[2, 0], 2 * p[2, 1], 1],
    ])
    b = np.array([p[0, 0] ** 2 + p[0, 1] ** 2,
                  p[1, 0] ** 2 + p[1, 1] ** 2,
                  p[2, 0] ** 2 + p[2, 1] ** 2])
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    return cx, cy, np.sqrt(r2)


def _ransac_circle(points_2d, n_iter=500, inlier_thresh=0.2, r_bounds=(1.0, 10.0)):
    """RANSAC circle fitting on 2D points. Returns (cx, cy, r) or None."""
    best_inliers = 0
    best_params = None
    n = len(points_2d)
    if n < 10:
        return None
    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        result = _fit_circle_3pts(points_2d[idx])
        if result is None:
            continue
        cx, cy, r = result
        if r < r_bounds[0] or r > r_bounds[1]:
            continue
        dists = np.abs(np.sqrt((points_2d[:, 0] - cx) ** 2 + (points_2d[:, 1] - cy) ** 2) - r)
        inliers = int(np.sum(dists < inlier_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_params = (cx, cy, r)
    return best_params


def _estimate_diameter_ransac_cross_sections(points_xyz, n_sections=10, n_iter=300,
                                              inlier_thresh=0.2):
    """Estimate tunnel diameter via RANSAC circle fitting on cross-sections.

    Takes *n_sections* slices perpendicular to the tunnel's long axis (from the
    MBR), fits a circle to each (across, z) cross-section, and returns the median
    diameter plus per-section diameters.
    """
    points_2d_xoy = points_xyz[:, :2]
    hull = ConvexHull(points_2d_xoy)
    poly = Polygon(points_2d_xoy[hull.vertices])
    rect = poly.minimum_rotated_rectangle
    rv = np.array(rect.exterior.coords)[:-1]
    edge_lens = [np.linalg.norm(rv[i] - rv[(i + 1) % 4]) for i in range(4)]

    if edge_lens[0] > edge_lens[1]:
        axis = (rv[1] - rv[0]) / edge_lens[0]
    else:
        axis = (rv[2] - rv[1]) / edge_lens[1]
    perp = np.array([-axis[1], axis[0]])

    center_xy = np.mean(points_2d_xoy, axis=0)
    along = (points_2d_xoy - center_xy) @ axis
    across = (points_2d_xoy - center_xy) @ perp
    z = points_xyz[:, 2]

    along_lo, along_hi = np.percentile(along, [10, 90])
    positions = np.linspace(along_lo, along_hi, n_sections)
    half_w = (along_hi - along_lo) / (2 * n_sections)

    diameters = []
    for pos in positions:
        mask = np.abs(along - pos) < half_w
        if np.sum(mask) < 50:
            continue
        sec = np.column_stack([across[mask], z[mask]])
        if len(sec) > 2000:
            sec = sec[np.random.choice(len(sec), 2000, replace=False)]
        result = _ransac_circle(sec, n_iter=n_iter, inlier_thresh=inlier_thresh)
        if result is not None:
            diameters.append(float(2 * result[2]))

    if not diameters:
        return None, []
    return float(np.median(diameters)), diameters


def analyze_point_cloud(file_path, tunnel_id=None):
    point_cloud_data = np.loadtxt(file_path)
    points_xyz = point_cloud_data[:, :3]
    intensity = point_cloud_data[:, 3]

    df = pd.DataFrame({
        'x': points_xyz[:, 0],
        'y': points_xyz[:, 1],
        'z': points_xyz[:, 2],
        'intensity': intensity,
    })

    basic_stats = {
        "total_points": int(len(df)),
        "data_structure": {
            "columns": 4,
            "description": "x, y, z, intensity",
        },
        "coordinate_ranges": {
            "x_range": [float(df['x'].min()), float(df['x'].max())],
            "y_range": [float(df['y'].min()), float(df['y'].max())],
            "z_range": [float(df['z'].min()), float(df['z'].max())],
            "intensity_range": [float(df['intensity'].min()), float(df['intensity'].max())],
        },
    }

    # Tunnel axis length from MBR
    points_2d_xoy = points_xyz[:, :2]
    hull = ConvexHull(points_2d_xoy)
    poly = Polygon(points_2d_xoy[hull.vertices])
    rect = poly.minimum_rotated_rectangle
    rv = np.array(rect.exterior.coords)[:-1]
    edge_lens = [np.linalg.norm(rv[i] - rv[(i + 1) % 4]) for i in range(4)]
    length = max(edge_lens)
    height = float(df['z'].max() - df['z'].min())

    # RANSAC circle fitting on cross-sections for diameter
    np.random.seed(42)
    ransac_diameter, section_diameters = _estimate_diameter_ransac_cross_sections(points_xyz)

    if ransac_diameter is not None:
        estimated_diameter = ransac_diameter
    else:
        estimated_diameter = height  # z-range fallback

    diameter_estimation = {
        "estimated_diameter": float(estimated_diameter),
        "estimated_radius": float(estimated_diameter / 2),
        "method": "ransac_circle_cross_sections",
        "n_sections_used": len(section_diameters),
        "section_diameters": [round(d, 4) for d in section_diameters],
        "description": "RANSAC circle fitting on cross-sections perpendicular to the tunnel axis. Median of per-section diameters.",
    }

    tunnel_geometry = {
        "dimensions": {
            "tunnel_length": float(length),
            "tunnel_height": height,
            "units": "meters",
        },
        "estimated_diameter": float(estimated_diameter),
        "diameter_estimation": diameter_estimation,
    }

    tree = cKDTree(points_xyz)
    distances, _ = tree.query(points_xyz, k=2)
    nearest_distances = distances[:, 1]

    point_density = {
        "mean_nearest_neighbor_distance": float(np.mean(nearest_distances)),
        "median_nearest_neighbor_distance": float(np.median(nearest_distances)),
        "min_nearest_neighbor_distance": float(np.min(nearest_distances)),
        "max_nearest_neighbor_distance": float(np.max(nearest_distances)),
        "units": "meters",
    }

    results = {
        "tunnel_id": tunnel_id if tunnel_id else "unknown",
        "input_file": file_path,
        "filtered_note": "Contains only characteristics for x, y, z, intensity columns. Ground truth data (segment_type, ring_number) excluded.",
        "point_cloud_analysis": {
            "basic_statistics": basic_stats,
            "tunnel_geometry": tunnel_geometry,
            "point_density": point_density,
        },
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
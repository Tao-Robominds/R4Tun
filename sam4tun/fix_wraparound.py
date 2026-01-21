"""
Fix wrap-around issues in tunnel data by trimming to exactly 360° coverage.

This script:
1. Trims enhanced.csv to exactly 360° theta coverage
2. Regenerates depth maps with correct dimensions
3. Updates detection_results.json with accurate coverage values
"""

import os
import csv
import json
import math
import shutil
from datetime import datetime

# Physical constants
TUNNEL_DIAMETER = 5.5  # meters
CIRCUMFERENCE = math.pi * TUNNEL_DIAMETER  # Full 360° = 17.2788m
RESOLUTION = 0.005  # meters per pixel

# K and AB block heights in mm (for detection_results.json)
K_HEIGHT_MM = 1079.92
AB_HEIGHT_MM = 3239.77


def analyze_csv(filepath):
    """Analyze a CSV file to get theta range and coverage."""
    theta_values = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'theta' in row and row['theta']:
                try:
                    theta_values.append(float(row['theta']))
                except ValueError:
                    pass
    
    if not theta_values:
        return None
    
    theta_min = min(theta_values)
    theta_max = max(theta_values)
    theta_range = theta_max - theta_min
    coverage = (theta_range / CIRCUMFERENCE) * 100
    
    return {
        'theta_min': theta_min,
        'theta_max': theta_max,
        'theta_range': theta_range,
        'coverage_percent': coverage,
        'point_count': len(theta_values)
    }


def trim_csv_to_360(input_path, output_path, target_coverage=100.0):
    """Trim CSV file to exactly 360° (target_coverage) by removing excess theta values."""
    
    analysis = analyze_csv(input_path)
    if analysis is None:
        print(f"  ERROR: Could not analyze {input_path}")
        return False
    
    current_coverage = analysis['coverage_percent']
    print(f"  Current coverage: {current_coverage:.2f}%")
    
    if current_coverage <= target_coverage + 0.01:  # Within tolerance
        print(f"  Coverage is within tolerance, no trimming needed")
        if input_path != output_path:
            shutil.copy(input_path, output_path)
        return True
    
    # Calculate new theta bounds for exactly 360°
    theta_min = analysis['theta_min']
    theta_max = analysis['theta_max']
    target_range = CIRCUMFERENCE * (target_coverage / 100.0)
    
    # Center the trim - remove equal amounts from both ends
    excess = analysis['theta_range'] - target_range
    new_theta_min = theta_min + excess / 2
    new_theta_max = theta_max - excess / 2
    
    print(f"  Trimming theta from [{theta_min:.4f}, {theta_max:.4f}] to [{new_theta_min:.4f}, {new_theta_max:.4f}]")
    
    # Read and filter the CSV
    rows_kept = 0
    rows_removed = 0
    
    with open(input_path, 'r') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        
        with open(output_path, 'w', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                if 'theta' in row and row['theta']:
                    try:
                        theta = float(row['theta'])
                        if new_theta_min <= theta <= new_theta_max:
                            writer.writerow(row)
                            rows_kept += 1
                        else:
                            rows_removed += 1
                    except ValueError:
                        writer.writerow(row)  # Keep rows without valid theta
                        rows_kept += 1
                else:
                    writer.writerow(row)
                    rows_kept += 1
    
    print(f"  Kept {rows_kept:,} rows, removed {rows_removed:,} rows")
    return True


def update_detection_results(base_dir, tunnel_id):
    """Update detection_results.json with accurate coverage values."""
    
    results_path = os.path.join(base_dir, "detection_results.json")
    
    # Analyze enhanced.csv for accurate coverage
    enhanced_path = os.path.join(base_dir, "enhanced.csv")
    if os.path.exists(enhanced_path):
        analysis = analyze_csv(enhanced_path)
    else:
        # Fallback to unwrapped.csv
        unwrapped_path = os.path.join(base_dir, "unwrapped.csv")
        analysis = analyze_csv(unwrapped_path)
    
    if analysis is None:
        print(f"  ERROR: Could not analyze data for {tunnel_id}")
        return False
    
    # Get depth map dimensions
    try:
        from PIL import Image
        depth_map_path = os.path.join(base_dir, "depth_map.png")
        img = Image.open(depth_map_path)
        width, height = img.size
        img.close()
    except:
        # Estimate from theta range
        height = int(analysis['theta_range'] / RESOLUTION)
        width = 3000  # Default estimate
    
    # Get ring count
    ring_count_path = os.path.join(base_dir, "ring_count.txt")
    try:
        with open(ring_count_path, 'r') as f:
            ring_count = int(f.read().strip())
    except:
        ring_count = 9  # Default
    
    # Determine pattern type based on tunnel characteristics
    coverage = analysis['coverage_percent']
    has_wraparound = coverage > 100.5  # More than 0.5% over is considered wraparound
    
    # Load existing results if available
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {'tunnel_id': tunnel_id}
    
    # Update with accurate values
    results['tunnel_id'] = tunnel_id
    results['detections'] = results.get('detections', {})
    results['detections']['wraparound'] = {
        'has_wraparound': has_wraparound,
        'coverage_percent': round(coverage, 4)
    }
    
    # Update image info
    results['image_info'] = {
        'height': height,
        'width': width,
        'resolution': RESOLUTION
    }
    
    # Update physical constants
    results['physical_constants'] = {
        'K_HEIGHT_MM': K_HEIGHT_MM,
        'AB_HEIGHT_MM': AB_HEIGHT_MM,
        'K_height_px': K_HEIGHT_MM / (1000 * RESOLUTION),
        'AB_height_px': AB_HEIGHT_MM / (1000 * RESOLUTION)
    }
    
    # Add metadata
    results['_metadata'] = {
        'last_updated': datetime.now().isoformat(),
        'updated_by': 'fix_wraparound.py'
    }
    
    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Updated detection_results.json: coverage={coverage:.2f}%, wraparound={has_wraparound}")
    return True


def process_tunnel(tunnel_id, base_dir):
    """Process a single tunnel to fix wrap-around issues."""
    print(f"\n{'='*60}")
    print(f"Processing tunnel: {tunnel_id}")
    print(f"{'='*60}")
    
    # 1. Check unwrapped.csv
    unwrapped_path = os.path.join(base_dir, "unwrapped.csv")
    if os.path.exists(unwrapped_path):
        print(f"\n1. Checking unwrapped.csv...")
        analysis = analyze_csv(unwrapped_path)
        if analysis:
            print(f"   Coverage: {analysis['coverage_percent']:.2f}%")
    
    # 2. Check denoised.csv
    denoised_path = os.path.join(base_dir, "denoised.csv")
    if os.path.exists(denoised_path):
        print(f"\n2. Checking denoised.csv...")
        analysis = analyze_csv(denoised_path)
        if analysis:
            print(f"   Coverage: {analysis['coverage_percent']:.2f}%")
    
    # 3. Check and fix enhanced.csv
    enhanced_path = os.path.join(base_dir, "enhanced.csv")
    if os.path.exists(enhanced_path):
        print(f"\n3. Checking and fixing enhanced.csv...")
        analysis = analyze_csv(enhanced_path)
        if analysis and analysis['coverage_percent'] > 100.01:
            # Create backup
            backup_path = enhanced_path + '.backup'
            if not os.path.exists(backup_path):
                shutil.copy(enhanced_path, backup_path)
                print(f"   Created backup: {backup_path}")
            
            # Trim to 100%
            trim_csv_to_360(enhanced_path, enhanced_path, target_coverage=100.0)
        else:
            print(f"   Coverage: {analysis['coverage_percent']:.2f}% - OK")
    
    # 4. Update detection_results.json
    print(f"\n4. Updating detection_results.json...")
    update_detection_results(base_dir, tunnel_id)
    
    print(f"\nDone processing {tunnel_id}")


def main():
    """Main function to fix wrap-around issues in all specified tunnels."""
    print("="*60)
    print("Wrap-around Fix Script")
    print("="*60)
    
    # Tunnels to process
    tunnels = ['4-1', '5-1']
    data_dir = '/home/boringtao/Projects/P4Tun_Off/data'
    
    for tunnel_id in tunnels:
        base_dir = os.path.join(data_dir, tunnel_id)
        if os.path.exists(base_dir):
            process_tunnel(tunnel_id, base_dir)
        else:
            print(f"\nWARNING: Directory not found: {base_dir}")
    
    print("\n" + "="*60)
    print("All tunnels processed!")
    print("="*60)
    print("\nNote: If enhanced.csv was trimmed, you may need to regenerate:")
    print("  - depth_map.png (run 3_enhancing.py or just the depth map generation)")
    print("  - detection files (run 4-1_detection.py)")


if __name__ == "__main__":
    main()

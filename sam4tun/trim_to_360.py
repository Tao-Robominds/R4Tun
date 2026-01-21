"""
Trim tunnel data to exactly 360° coverage.

Simple, robust script to trim enhanced.csv files that have >100% theta coverage.
"""

import os
import math
import shutil

# Physical constants
TUNNEL_DIAMETER = 5.5  # meters
CIRCUMFERENCE = math.pi * TUNNEL_DIAMETER  # 17.2788m for 360°
RESOLUTION = 0.005  # meters per pixel


def trim_csv_to_360(tunnel_id, target_coverage=100.0):
    """Trim enhanced.csv to exactly target_coverage% theta coverage."""
    
    base_dir = f'/home/boringtao/Projects/P4Tun_Off/data/{tunnel_id}'
    enhanced_path = os.path.join(base_dir, 'enhanced.csv')
    backup_path = enhanced_path + '.bak'
    
    print(f"\nProcessing tunnel {tunnel_id}...")
    
    # Read all lines
    with open(enhanced_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[0]
    data_lines = lines[1:]
    
    # Parse header to find theta column index
    header_fields = header.strip().split(',')
    try:
        theta_idx = header_fields.index('theta')
    except ValueError:
        print(f"  ERROR: 'theta' column not found in {enhanced_path}")
        return False
    
    print(f"  Theta column index: {theta_idx}")
    print(f"  Total data rows: {len(data_lines):,}")
    
    # Extract theta values and their line indices
    theta_data = []
    for i, line in enumerate(data_lines):
        fields = line.strip().split(',')
        if len(fields) > theta_idx:
            try:
                theta = float(fields[theta_idx])
                theta_data.append((i, theta))
            except (ValueError, IndexError):
                pass
    
    print(f"  Rows with valid theta: {len(theta_data):,}")
    
    if not theta_data:
        print("  ERROR: No valid theta values found")
        return False
    
    # Calculate current coverage
    theta_values = [t[1] for t in theta_data]
    theta_min = min(theta_values)
    theta_max = max(theta_values)
    theta_range = theta_max - theta_min
    current_coverage = (theta_range / CIRCUMFERENCE) * 100
    
    print(f"  Current theta range: [{theta_min:.4f}, {theta_max:.4f}]")
    print(f"  Current coverage: {current_coverage:.2f}%")
    
    if current_coverage <= target_coverage + 0.01:
        print(f"  Coverage is already within tolerance, no trimming needed")
        return True
    
    # Calculate new bounds (trim equally from both ends)
    target_range = CIRCUMFERENCE * (target_coverage / 100.0)
    excess = theta_range - target_range
    new_theta_min = theta_min + excess / 2
    new_theta_max = theta_max - excess / 2
    
    print(f"  Target theta range: [{new_theta_min:.4f}, {new_theta_max:.4f}]")
    
    # Create backup
    if not os.path.exists(backup_path):
        shutil.copy(enhanced_path, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Find indices of rows to keep
    indices_to_keep = set()
    for i, theta in theta_data:
        if new_theta_min <= theta <= new_theta_max:
            indices_to_keep.add(i)
    
    # Also keep rows without valid theta (they might be important)
    rows_without_theta = set(range(len(data_lines))) - set(i for i, _ in theta_data)
    # Actually, let's remove upsampled points that don't have theta in range
    # We need to check all rows
    
    print(f"  Rows to keep: {len(indices_to_keep):,}")
    print(f"  Rows to remove: {len(theta_data) - len(indices_to_keep):,}")
    
    # Write filtered data
    with open(enhanced_path, 'w') as f:
        f.write(header)
        for i, line in enumerate(data_lines):
            if i in indices_to_keep:
                f.write(line)
    
    # Verify the result
    with open(enhanced_path, 'r') as f:
        new_line_count = sum(1 for _ in f) - 1  # Subtract header
    
    print(f"  New file has {new_line_count:,} data rows")
    
    # Verify coverage
    theta_values_new = []
    with open(enhanced_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            fields = line.strip().split(',')
            if len(fields) > theta_idx:
                try:
                    theta_values_new.append(float(fields[theta_idx]))
                except:
                    pass
    
    if theta_values_new:
        new_range = max(theta_values_new) - min(theta_values_new)
        new_coverage = (new_range / CIRCUMFERENCE) * 100
        print(f"  New coverage: {new_coverage:.2f}%")
    
    return True


def update_detection_results(tunnel_id):
    """Update detection_results.json with accurate coverage."""
    import json
    
    base_dir = f'/home/boringtao/Projects/P4Tun_Off/data/{tunnel_id}'
    enhanced_path = os.path.join(base_dir, 'enhanced.csv')
    results_path = os.path.join(base_dir, 'detection_results.json')
    
    # Calculate coverage from enhanced.csv
    theta_values = []
    with open(enhanced_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        theta_idx = header.index('theta')
        
        for line in lines[1:]:
            fields = line.strip().split(',')
            if len(fields) > theta_idx:
                try:
                    theta_values.append(float(fields[theta_idx]))
                except:
                    pass
    
    if not theta_values:
        print(f"  ERROR: No theta values in enhanced.csv")
        return False
    
    theta_range = max(theta_values) - min(theta_values)
    coverage = (theta_range / CIRCUMFERENCE) * 100
    has_wraparound = coverage > 100.5
    
    # Load and update detection_results.json
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {'tunnel_id': tunnel_id, 'detections': {}}
    
    results['detections']['wraparound'] = {
        'has_wraparound': has_wraparound,
        'coverage_percent': round(coverage, 2)
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Updated detection_results.json: coverage={coverage:.2f}%, wraparound={has_wraparound}")
    return True


def main():
    """Main function."""
    print("="*60)
    print("Trim to 360° Coverage")
    print("="*60)
    
    tunnels = ['4-1', '5-1']
    
    for tunnel_id in tunnels:
        trim_csv_to_360(tunnel_id)
        update_detection_results(tunnel_id)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()

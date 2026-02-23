#!/usr/bin/env python3
"""Run all 3 detection methods and compare against GT."""

import os
import sys
import json
import shutil
import subprocess

script_dir = os.path.dirname(__file__)
detection_script = os.path.join(script_dir, "2_detection.py")
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

tunnel_id = "4-1"
base_dir = "data"
param_file = os.path.join(script_dir, "parameters", tunnel_id, "parameters_detection.json")

# Backup original params
backup_file = param_file + ".backup"
shutil.copy(param_file, backup_file)

methods = [
    ('complex_staggered', 'all_segments_dbscan.csv'),
    ('groove_pair', 'all_segments_groove_pair.csv'),
    ('combined', 'all_segments_combined.csv'),
]

try:
    for method, output_filename in methods:
        print(f"\n{'='*60}")
        print(f"Running {method} method -> {output_filename}")
        print(f"{'='*60}")
        
        # Load and update params
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        params['k_detection_method'] = method
        params['output_filename'] = output_filename
        
        # Save updated params
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Run detection via subprocess
        try:
            result = subprocess.run(
                [sys.executable, detection_script, tunnel_id, "--data-dir", base_dir],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✓ {method} completed")
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                if lines:
                    print("\n".join(lines[-5:]))
            else:
                print(f"✗ {method} failed with return code {result.returncode}")
                if result.stderr:
                    print("STDERR:", result.stderr[-500:])
        except Exception as e:
            print(f"✗ {method} failed: {e}")
            import traceback
            traceback.print_exc()

finally:
    # Restore original params
    shutil.copy(backup_file, param_file)
    os.remove(backup_file)

# Run comparison
print(f"\n{'='*60}")
print("Running comparison...")
print(f"{'='*60}")
subprocess.run([sys.executable, os.path.join(script_dir, "compare_3way.py")], cwd=project_root)

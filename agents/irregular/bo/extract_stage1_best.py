"""
Extract Stage 1 best detection results and prepare all_segments.csv for Stage 2.

This script:
1. Finds the best groove-pair BO trial (lowest mean |dY|)
2. Loads the best parameters
3. Runs detection with those parameters
4. Saves all_segments.csv for Stage 2 input
"""

import os
import sys
import json
import glob
import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import detection
detection_dir = PROJECT_ROOT / 'agents' / 'irregular' / '2_detection'
sys.path.insert(0, str(detection_dir))
spec = importlib.util.spec_from_file_location("detection", detection_dir / "2_detection.py")
detection_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection_module)
run_detection = detection_module.run_detection


def find_best_trial(logs_dir: str, tunnel_id: str) -> dict:
    """Find the best trial (lowest mean_k_y_distance)."""
    pattern = os.path.join(logs_dir, f"groove_pair_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    
    if not log_files:
        raise FileNotFoundError(f"No log files found: {pattern}")
    
    best_score = float('inf')
    best_trial = None
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        if 'error' in data:
            continue
        
        score = data.get('mean_k_y_distance', float('inf'))
        if score < best_score:
            best_score = score
            best_trial = data
    
    if best_trial is None:
        raise ValueError("No valid trials found")
    
    return best_trial


def main(tunnel_id: str, data_dir: str = "data/wrap"):
    """Extract Stage 1 best and prepare Stage 2 input."""
    logs_dir = str(PROJECT_ROOT / "bo" / "complex_staggered" / f"logs_groove_pair_{tunnel_id}")
    tunnel_dir = os.path.join(data_dir, tunnel_id)
    params_file = os.path.join(
        PROJECT_ROOT, 'agents', 'irregular', '2_detection',
        'parameters', tunnel_id, 'parameters_detection.json'
    )
    
    print(f"Finding best Stage 1 trial for {tunnel_id}...")
    best_trial = find_best_trial(logs_dir, tunnel_id)
    
    print(f"Best trial: {best_trial['trial_id']}")
    print(f"  Mean |dY|: {best_trial['mean_k_y_distance']:.1f}px")
    print(f"  Matched: {best_trial['num_matched']}/7")
    
    # Load current params and update with best values
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Update with best groove-pair params
    best_params = best_trial['params']
    for key in ['k_expected_height_px', 'k_gap_tolerance_px', 'k_candidates_per_ring', 'groove_snap_px']:
        if key in best_params:
            params[key] = best_params[key]
    
    # Update group offsets
    for key in best_params:
        if key.startswith('A_') or key.startswith('B_'):
            if 'group_offsets' not in params:
                params['group_offsets'] = {}
            params['group_offsets'][key] = best_params[key]
    
    # Save updated params
    import tempfile
    import shutil
    with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(params_file), delete=False, suffix='.json') as f:
        json.dump(params, f, indent=2)
        temp_path = f.name
    shutil.move(temp_path, params_file)
    print(f"\nUpdated parameters file: {params_file}")
    
    # Run detection with best params
    print(f"\nRunning detection with best parameters...")
    import io
    from contextlib import redirect_stdout, redirect_stderr
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        k_positions, all_segments = run_detection(tunnel_id, data_dir)
    
    # Save all_segments.csv for Stage 2
    segments_file = os.path.join(tunnel_dir, 'all_segments_stage1_best.csv')
    all_segments.to_csv(segments_file, index=False)
    print(f"Saved Stage 1 best all_segments.csv: {segments_file}")
    print(f"  Total segments: {len(all_segments)}")
    print(f"  Blocks per ring: {all_segments.groupby('Ring')['Block'].count().values.tolist()}")
    
    # Also save best trial summary
    summary_file = os.path.join(logs_dir, 'stage1_best_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'trial_id': best_trial['trial_id'],
            'mean_k_y_distance': best_trial['mean_k_y_distance'],
            'params': best_trial['params'],
            'segments_file': segments_file,
        }, f, indent=2)
    print(f"\nSaved summary: {summary_file}")
    
    return segments_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Stage 1 best detection")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data/wrap")
    args = parser.parse_args()
    
    main(args.tunnel_id, args.data_dir)

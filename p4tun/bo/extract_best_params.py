#!/usr/bin/env python3
"""
Extract best parameters from BO run history or terminal logs.
This script helps recover the best parameter set when BO results weren't properly saved.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def extract_from_terminal_log(log_file: str) -> Optional[Tuple[float, Dict]]:
    """Extract best mIoU and parameters from terminal log."""
    if not os.path.exists(log_file):
        return None
    
    best_miou = 0.0
    best_params = None
    in_params = False
    current_params = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Look for "New best mIoU" lines
        match = re.search(r'\[(\d+)\]\s+New best mIoU:\s+([\d.]+)', line)
        if match:
            miou = float(match.group(2))
            if miou > best_miou:
                best_miou = miou
        
        # Look for "Best mIoU:" final result
        match = re.search(r'Best mIoU:\s+([\d.]+)', line)
        if match:
            miou = float(match.group(1))
            if miou > best_miou:
                best_miou = miou
        
        # Look for "Best parameters:" section
        if 'Best parameters:' in line:
            in_params = True
            current_params = {}
            continue
        
        if in_params:
            # Parse parameter lines like "  param_name: value"
            match = re.match(r'\s+(\w+):\s+(.+)', line)
            if match:
                param_name = match.group(1)
                param_value = match.group(2).strip()
                # Try to convert to number
                try:
                    if '.' in param_value:
                        param_value = float(param_value)
                    else:
                        param_value = int(param_value)
                except:
                    pass
                current_params[param_name] = param_value
            elif line.strip() == '' or line.startswith('Saved'):
                # End of parameters section
                if current_params:
                    best_params = current_params.copy()
                in_params = False
    
    if best_miou > 0:
        return best_miou, best_params
    return None

def extract_from_history_json(history_file: str) -> Optional[Tuple[float, Dict]]:
    """Extract best parameters from history JSON file."""
    if not os.path.exists(history_file):
        return None
    
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    best_miou = 0.0
    best_params = None
    
    # Check if best_params exists
    if 'best_params' in data and data['best_params']:
        # Find best score from history
        if 'history' in data:
            for entry in data['history']:
                if 'miou' in entry:
                    if entry['miou'] > best_miou:
                        best_miou = entry['miou']
                        # Get params for this evaluation
                        if 'params' in entry:
                            best_params = entry['params']
        
        # If we have best_params but no history, use it
        if best_params is None and data['best_params']:
            best_params = data['best_params']
    
    if best_miou > 0 or best_params:
        return best_miou, best_params
    return None

def find_best_in_results_dir(tunnel_id: str, stage: str, results_dir: str = 'p4tun/bo/results') -> Optional[Tuple[float, Dict]]:
    """Find best parameters from results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return None
    
    best_miou = 0.0
    best_params = None
    best_file = None
    
    # Look for JSON files matching pattern
    pattern = f'{tunnel_id}_{stage}_*.json'
    for json_file in results_path.glob(pattern):
        if '_history.json' in json_file.name:
            continue
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'best_score' in data and data['best_score'] > best_miou:
                best_miou = data['best_score']
                if 'best_params' in data:
                    best_params = data['best_params']
                    best_file = json_file
        except:
            continue
    
    if best_miou > 0:
        return best_miou, best_params
    return None

def save_best_params(tunnel_id: str, stage: str, best_params: Dict, output_file: str):
    """Save best parameters to parameter file."""
    from p4tun.bo.search_space import params_to_detection_dict, params_to_sam_dict, save_parameters
    
    if stage == 'detection' or 'detection' in stage:
        detection_params = params_to_detection_dict(
            list(best_params.values()),
            list(best_params.keys())
        )
        save_parameters(detection_params, tunnel_id, 'detection')
        print(f"Saved detection parameters to p4tun/parameters/{tunnel_id}/parameters_detection.json")
    
    if stage == 'sam' or 'sam' in stage:
        sam_params = params_to_sam_dict(
            list(best_params.values()),
            list(best_params.keys())
        )
        save_parameters(sam_params, tunnel_id, 'sam')
        print(f"Saved SAM parameters to p4tun/parameters/{tunnel_id}/parameters_sam.json")

def main():
    """Main function to extract and save best parameters."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract best parameters from BO run')
    parser.add_argument('tunnel_id', help='Tunnel ID (e.g., 1-4, 3-1)')
    parser.add_argument('stage', help='Stage (detection, sam, combined)')
    parser.add_argument('--log-file', help='Terminal log file path')
    parser.add_argument('--history-file', help='History JSON file path')
    parser.add_argument('--results-dir', default='p4tun/bo/results', help='Results directory')
    parser.add_argument('--save', action='store_true', help='Save parameters to parameter files')
    
    args = parser.parse_args()
    
    best_miou = 0.0
    best_params = None
    source = None
    
    # Try different sources
    if args.log_file:
        result = extract_from_terminal_log(args.log_file)
        if result:
            miou, params = result
            if miou > best_miou:
                best_miou = miou
                best_params = params
                source = 'terminal_log'
    
    if args.history_file:
        result = extract_from_history_json(args.history_file)
        if result:
            miou, params = result
            if miou > best_miou:
                best_miou = miou
                best_params = params
                source = 'history_json'
    
    # Try results directory
    result = find_best_in_results_dir(args.tunnel_id, args.stage, args.results_dir)
    if result:
        miou, params = result
        if miou > best_miou:
            best_miou = miou
            best_params = params
            source = 'results_dir'
    
    if best_params:
        print(f"Found best parameters from {source}:")
        print(f"  Best mIoU: {best_miou:.4f}")
        print(f"\nBest parameters:")
        for name, value in best_params.items():
            print(f"  {name}: {value}")
        
        if args.save:
            save_best_params(args.tunnel_id, args.stage, best_params, None)
    else:
        print("No best parameters found!")
        sys.exit(1)

if __name__ == '__main__':
    main()

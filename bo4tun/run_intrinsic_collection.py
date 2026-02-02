#!/usr/bin/env python3
"""
Run Intrinsic Metrics Collection Experiments

This script:
1. Loads diverse parameter configurations from BO history
2. Runs the pipeline with each configuration
3. Saves intermediate files (detected.csv, final.csv)
4. Computes all intrinsic metrics
5. Saves comprehensive training data

Usage:
    python bo4tun/run_intrinsic_collection.py --tunnel 2-2 --n 10
    python bo4tun/run_intrinsic_collection.py --all --n 10
"""

import os
import sys
import json
import subprocess
import shutil
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bo4tun.intrinsic_metrics import (
    compute_all_metrics,
    compute_all_complex_metrics,
    compute_preprocessing_guardrails,
    load_expected_rings,
)


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = project_root / 'data'
PARAM_DIR = project_root / 'p4tun' / 'parameters'
OUTPUT_DIR = project_root / 'bo4tun' / 'training'
PYTHON = project_root / 'venv' / 'bin' / 'python'


# =============================================================================
# Parameter Management
# =============================================================================

def load_selected_configs(config_file: Path = None) -> pd.DataFrame:
    """Load selected configurations from CSV."""
    if config_file is None:
        config_file = OUTPUT_DIR / 'selected_experiment_configs.csv'
    return pd.read_csv(config_file)


def safe_int(val, default):
    """Convert value to int safely."""
    try:
        return int(val) if pd.notna(val) else default
    except:
        return default

def safe_float(val, default):
    """Convert value to float safely."""
    try:
        return float(val) if pd.notna(val) else default
    except:
        return default


def create_detection_params(row: pd.Series) -> dict:
    """Create detection parameters JSON from config row."""
    params = {
        "preprocessing": {
            "binary_threshold": safe_int(row.get('param_binary_threshold'), 127),
            "dilation_kernel_size": safe_int(row.get('param_dilation_kernel_size'), 5),
            "dilation_iterations": safe_int(row.get('param_dilation_iterations'), 2),
        },
        "hough_oblique": {
            "threshold": safe_int(row.get('param_hough_oblique_threshold'), 50),
            "minLineLength": safe_int(row.get('param_hough_oblique_min_length'), 100),
            "maxLineGap": safe_int(row.get('param_hough_oblique_max_gap'), 50),
        },
        "hough_horizontal": {
            "threshold": safe_int(row.get('param_hough_horizontal_threshold'), 50),
            "minLineLength": safe_int(row.get('param_hough_horizontal_min_length'), 100),
            "maxLineGap": safe_int(row.get('param_hough_horizontal_max_gap'), 50),
        },
        "hough_vertical": {
            "threshold": safe_int(row.get('param_hough_vertical_threshold'), 50),
        },
        "angles": {
            "positive_min": safe_float(row.get('param_angle_positive_min'), 2),
            "positive_max": safe_float(row.get('param_angle_positive_max'), 15),
        },
        "merge_distance_threshold": safe_float(row.get('param_merge_distance_threshold'), 3),
    }
    return params


def create_sam_params(row: pd.Series) -> dict:
    """Create SAM parameters JSON from config row."""
    params = {
        "segment_width": safe_float(row.get('param_segment_width'), 1200),
        "k_height": safe_float(row.get('param_k_height'), 1080),
        "ab_height": safe_float(row.get('param_ab_height'), 3240),
        "angle_deg": safe_float(row.get('param_angle_deg'), 7.5),
        "padding": safe_int(row.get('param_padding'), 50),
        "crop_margin": safe_int(row.get('param_crop_margin'), 100),
        "k_block": {
            "outer_ring": safe_float(row.get('param_k_outer_ring'), 700),
            "middle_ring": safe_float(row.get('param_k_middle_ring'), 500),
            "inner_ring": safe_float(row.get('param_k_inner_ring'), 350),
            "center_ring": safe_float(row.get('param_k_center_ring'), 150),
        },
        "ab_block": {
            "outer_ring": safe_float(row.get('param_ab_outer_ring'), 700),
            "middle_ring": safe_float(row.get('param_ab_middle_ring'), 500),
            "inner_ring": safe_float(row.get('param_ab_inner_ring'), 350),
            "center_ring": safe_float(row.get('param_ab_center_ring'), 150),
            "fine_spacing": safe_float(row.get('param_ab_fine_spacing'), 250),
            "ultra_fine": safe_float(row.get('param_ab_ultra_fine'), 150),
            "edge_ring": safe_float(row.get('param_ab_edge_ring'), 100),
            "edge_spacing": safe_float(row.get('param_ab_edge_spacing'), 350),
        },
        "masks": {
            "k_width": safe_float(row.get('param_k_mask_width'), 650),
            "k_height_pos": safe_float(row.get('param_k_mask_height_pos'), 650),
            "k_height_neg": safe_float(row.get('param_k_mask_height_neg'), 550),
            "ab_width": safe_float(row.get('param_ab_mask_width'), 600),
            "ab_height": safe_float(row.get('param_ab_mask_height'), 1600),
        },
        "min_quality_threshold": safe_float(row.get('param_min_quality_threshold'), 0.3),
    }
    return params


def create_preprocessing_params(row: pd.Series) -> dict:
    """Create preprocessing parameters JSON from config row."""
    params = {
        "unfolding": {
            "theta_offset": row.get('param_unfold_theta_offset', 0),
            "slice_half_thickness": row.get('param_unfold_slice_half_thickness', 0.5),
        },
        "denoising": {
            "radius_center": row.get('param_denoise_radius_center', 2.7),
            "radius_half_width": row.get('param_denoise_radius_half_width', 0.7),
            "theta_step": row.get('param_denoise_theta_step', 0.02),
            "radial_step": row.get('param_denoise_radial_step', 0.02),
            "gradient_threshold": row.get('param_denoise_gradient_threshold', 0.15),
        },
        "enhancing": {
            "curvature_neighbors": row.get('param_enhance_curvature_neighbors', 30),
            "resolution": row.get('param_enhance_resolution', 0.005),
            "interpolation_radius": row.get('param_enhance_interpolation_radius', 0.02),
        },
    }
    return params


def save_params_to_dir(tunnel_id: str, config_idx: int, row: pd.Series) -> Path:
    """Save parameter files to a temporary directory."""
    # Create experiment output directory
    exp_dir = DATA_DIR / 'intrinsic_experiments' / tunnel_id / f'config_{config_idx:03d}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create parameter directory for the pipeline
    param_tunnel_dir = PARAM_DIR / tunnel_id
    param_tunnel_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detection parameters
    det_params = create_detection_params(row)
    with open(param_tunnel_dir / 'parameters_detection.json', 'w') as f:
        json.dump(det_params, f, indent=2)
    
    # Save SAM parameters
    sam_params = create_sam_params(row)
    with open(param_tunnel_dir / 'parameters_sam.json', 'w') as f:
        json.dump(sam_params, f, indent=2)
    
    # Save preprocessing parameters
    pre_params = create_preprocessing_params(row)
    with open(param_tunnel_dir / 'parameters_preprocessing.json', 'w') as f:
        json.dump(pre_params, f, indent=2)
    
    return exp_dir


# =============================================================================
# Pipeline Execution
# =============================================================================

def run_pipeline_stage(tunnel_id: str, stage: int, timeout: int = 300) -> bool:
    """Run a single pipeline stage."""
    stage_scripts = {
        1: 'p4tun/1_unfolding.py',
        2: 'p4tun/2_denoising.py',
        3: 'p4tun/3_enhancing.py',
        4: 'p4tun/4-1_detection.py',
        5: 'p4tun/4-2_sam.py',  # Will be routed properly
        6: 'p4tun/evaluation.py',
    }
    
    script = stage_scripts.get(stage)
    if not script:
        return False
    
    # For stage 5, use sam_router to get correct script
    if stage == 5:
        try:
            result = subprocess.run(
                [str(PYTHON), '-m', 'p4tun.sam_router', str(DATA_DIR / tunnel_id)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                sam_script = result.stdout.strip()
                script = f'p4tun/{sam_script}'
        except:
            pass
    
    try:
        result = subprocess.run(
            [str(PYTHON), str(project_root / script), tunnel_id],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(project_root)
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  Stage {stage} timed out")
        return False
    except Exception as e:
        print(f"  Stage {stage} error: {e}")
        return False


def run_experiment(tunnel_id: str, config_idx: int, row: pd.Series, 
                   start_stage: int = 4, end_stage: int = 6) -> Dict:
    """Run a single experiment and collect metrics."""
    print(f"\n--- Experiment {tunnel_id}/config_{config_idx:03d} ---")
    print(f"  Expected mIoU: {row.get('mIoU', 'N/A'):.3f}")
    
    # Save parameters
    exp_dir = save_params_to_dir(tunnel_id, config_idx, row)
    
    # Run pipeline stages
    success = True
    for stage in range(start_stage, end_stage + 1):
        print(f"  Running stage {stage}...", end=' ')
        if run_pipeline_stage(tunnel_id, stage):
            print("OK")
        else:
            print("FAILED")
            success = False
            break
    
    # Collect results
    result = {
        'tunnel_id': tunnel_id,
        'config_idx': config_idx,
        'config_rank': row.get('rank', 'unknown'),
        'expected_mIoU': row.get('mIoU'),
        'pipeline_success': success,
    }
    
    # Copy intermediate files to experiment directory
    tunnel_dir = DATA_DIR / tunnel_id
    for filename in ['detected.csv', 'final.csv', 'unwrapped.csv', 'denoised.csv']:
        src = tunnel_dir / filename
        if src.exists():
            shutil.copy(src, exp_dir / filename)
    
    # Copy depth map if exists
    depth_map = tunnel_dir / 'depth_map_outlier.npy'
    if depth_map.exists():
        shutil.copy(depth_map, exp_dir / 'depth_map_outlier.npy')
    
    if success:
        # Compute intrinsic metrics
        detected_csv = str(exp_dir / 'detected.csv')
        final_csv = str(exp_dir / 'final.csv')
        
        is_complex = tunnel_id in ['4-1', '5-1']
        
        # Base metrics
        try:
            base_metrics = compute_all_metrics(
                tunnel_id, detected_csv, final_csv, str(DATA_DIR)
            )
            result.update(base_metrics)
        except Exception as e:
            print(f"  Error computing base metrics: {e}")
        
        # Complex metrics
        if is_complex:
            try:
                complex_metrics = compute_all_complex_metrics(
                    tunnel_id, detected_csv, final_csv, str(DATA_DIR)
                )
                for k, v in complex_metrics.items():
                    result[f'complex_{k}'] = v
            except Exception as e:
                print(f"  Error computing complex metrics: {e}")
        
        # Preprocessing guardrails
        try:
            unwrapped = str(exp_dir / 'unwrapped.csv')
            denoised = str(exp_dir / 'denoised.csv')
            depth_map = str(exp_dir / 'depth_map_outlier.npy')
            
            if os.path.exists(unwrapped):
                pre_metrics = compute_preprocessing_guardrails(
                    unwrapped,
                    denoised if os.path.exists(denoised) else None,
                    depth_map if os.path.exists(depth_map) else None
                )
                for k, v in pre_metrics.items():
                    result[f'pre_{k}'] = v
        except Exception as e:
            print(f"  Error computing preprocessing metrics: {e}")
        
        # Get actual mIoU from evaluation
        eval_file = tunnel_dir / 'evaluation_results.json'
        if eval_file.exists():
            try:
                with open(eval_file) as f:
                    eval_data = json.load(f)
                    result['actual_mIoU'] = eval_data.get('mIoU', eval_data.get('miou'))
            except:
                pass
    
    return result


# =============================================================================
# Main Collection Loop
# =============================================================================

def run_collection(
    tunnels: List[str] = None,
    n_per_tunnel: int = 10,
    start_stage: int = 4,
    end_stage: int = 6,
    config_file: Path = None
) -> pd.DataFrame:
    """Run the full collection process."""
    print("=" * 70)
    print("INTRINSIC METRICS COLLECTION")
    print("=" * 70)
    
    # Load configurations
    configs = load_selected_configs(config_file)
    
    if tunnels is None:
        tunnels = configs['tunnel_id'].unique().tolist()
    
    print(f"Tunnels: {tunnels}")
    print(f"Configs per tunnel: {n_per_tunnel}")
    print(f"Stages: {start_stage} -> {end_stage}")
    print()
    
    all_results = []
    
    for tunnel in tunnels:
        print(f"\n{'='*70}")
        print(f"TUNNEL: {tunnel}")
        print(f"{'='*70}")
        
        # Get configs for this tunnel
        tunnel_configs = configs[configs['tunnel_id'] == tunnel].head(n_per_tunnel)
        
        if len(tunnel_configs) == 0:
            print(f"No configurations found for {tunnel}")
            continue
        
        for idx, (_, row) in enumerate(tunnel_configs.iterrows()):
            result = run_experiment(tunnel, idx, row, start_stage, end_stage)
            all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f'intrinsic_collection_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {len(results_df)}")
    print(f"Successful: {results_df['pipeline_success'].sum()}")
    print(f"Failed: {(~results_df['pipeline_success']).sum()}")
    
    return results_df


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run intrinsic metrics collection')
    parser.add_argument('--tunnel', type=str, help='Single tunnel to run')
    parser.add_argument('--all', action='store_true', help='Run all tunnels')
    parser.add_argument('--n', type=int, default=10, help='Configs per tunnel')
    parser.add_argument('--start-stage', type=int, default=4, help='Start stage (1-6)')
    parser.add_argument('--end-stage', type=int, default=6, help='End stage (1-6)')
    parser.add_argument('--config-file', type=str, help='Custom config file')
    
    args = parser.parse_args()
    
    tunnels = None
    if args.tunnel:
        tunnels = [args.tunnel]
    elif args.all:
        tunnels = ['1-4', '2-2', '3-1', '4-1', '5-1']
    
    config_file = Path(args.config_file) if args.config_file else None
    
    results = run_collection(
        tunnels=tunnels,
        n_per_tunnel=args.n,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        config_file=config_file
    )
    
    return results


if __name__ == '__main__':
    main()

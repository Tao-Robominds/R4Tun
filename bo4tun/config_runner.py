"""
Config Runner - Run historical BO configs and collect intrinsic metrics.

For each historical config:
1. Apply params to parameter files
2. Run the relevant pipeline stage
3. Compute intrinsic metrics from outputs
4. Return metrics dict (to be paired with mIoU from BO logs)
"""

import os
import sys
import json
import subprocess
from typing import Dict, List, Optional, Any
import pandas as pd

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from p4tun.bo.search_space import (
    params_to_sam_dict,
    load_default_parameters,
)
from p4tun.bo.detection_bo import params_to_detection_json
from p4tun.bo.detection_complex_bo import params_to_complex_detection_json
from p4tun.bo.sam_complex_bo import params_to_complex_sam_json

from bo4tun.intrinsic_metrics import (
    compute_detection_metrics,
    compute_sam_metrics,
    compute_all_metrics,
)


# Tunnel -> stage -> script mapping
DETECTION_SCRIPT = {
    'standard': '4-1_detection.py',
    'complex': '4-1_detection_complex.py',
}
SAM_SCRIPT = {
    'standard': '4-2_sam.py',
    'complex': '4-2_sam_complex.py',
}

# Tunnels that use complex detection/SAM
COMPLEX_TUNNELS = {'5-1'}
WRAPAROUND_TUNNELS = {'4-1'}  # sam_wraparound stage


def _get_param_dict_from_row(row: pd.Series, param_cols: List[str]) -> Dict[str, Any]:
    """Extract param dict from a training data row, stripping param_ prefix."""
    CATEGORICAL_PARAMS = {'k_block_points', 'ab_block_points'}
    param_dict = {}
    for col in param_cols:
        if col in row.index and pd.notna(row[col]):
            name = col.replace('param_', '')
            val = row[col]
            if isinstance(val, (int, float)):
                param_dict[name] = val
            elif name in CATEGORICAL_PARAMS and isinstance(val, str):
                param_dict[name] = val
            elif isinstance(val, str):
                try:
                    param_dict[name] = float(val)
                except ValueError:
                    param_dict[name] = val
    return param_dict


def _get_detection_script(tunnel_id: str, stage: str) -> str:
    """Get detection script for tunnel/stage."""
    if tunnel_id in COMPLEX_TUNNELS or stage == 'complex_detection':
        return DETECTION_SCRIPT['complex']
    return DETECTION_SCRIPT['standard']


def _get_sam_script(tunnel_id: str, stage: str) -> str:
    """Get SAM script for tunnel/stage."""
    if tunnel_id in COMPLEX_TUNNELS or stage in ('complex_sam', 'sam_wraparound'):
        return SAM_SCRIPT['complex']
    return SAM_SCRIPT['standard']


def _params_row_to_detection_config(row: pd.Series, param_cols: List[str], tunnel_id: str) -> Dict:
    """Convert training row to detection parameter config."""
    param_dict = _get_param_dict_from_row(row, param_cols)
    
    if tunnel_id in COMPLEX_TUNNELS:
        names = list(param_dict.keys())
        params = list(param_dict.values())
        return params_to_complex_detection_json(params, names, tunnel_id)
    
    # Standard detection - use detection_bo format
    names = list(param_dict.keys())
    params = list(param_dict.values())
    return params_to_detection_json(params, names)


def _params_row_to_sam_config(row: pd.Series, param_cols: List[str], tunnel_id: str) -> Dict:
    """Convert training row to SAM parameter config."""
    param_dict = _get_param_dict_from_row(row, param_cols)
    
    if tunnel_id in COMPLEX_TUNNELS or 'complex' in str(row.get('stage', '')):
        names = list(param_dict.keys())
        params = list(param_dict.values())
        return params_to_complex_sam_json(params, names, tunnel_id)
    
    names = list(param_dict.keys())
    params = list(param_dict.values())
    return params_to_sam_dict(params, names)


def _run_script(script_name: str, tunnel_id: str, data_dir: str = 'data', timeout: int = 300) -> bool:
    """Run a p4tun script and return success."""
    script_path = os.path.join(PROJECT_ROOT, 'p4tun', script_name)
    if not os.path.exists(script_path):
        print(f"  Warning: Script not found {script_path}")
        return False
    
    venv_python = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')
    if not os.path.exists(venv_python):
        venv_python = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python')
    if not os.path.exists(venv_python):
        venv_python = sys.executable
    
    # Run script directly (module names with hyphens are problematic)
    cmd = [venv_python, script_path, tunnel_id, '--data-dir', data_dir]
    
    env = os.environ.copy()
    script_dir = os.path.join(PROJECT_ROOT, 'p4tun')
    segment_path = os.path.join(script_dir, 'segment-anything')
    pythonpath = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f"{segment_path}:{script_dir}:{PROJECT_ROOT}:{pythonpath}" if pythonpath else f"{segment_path}:{script_dir}:{PROJECT_ROOT}"
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_ROOT,
            env=env
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(f"  Error running {script_name}: {e}")
        return False


def run_config_and_collect_metrics(
    tunnel_id: str,
    stage: str,
    row: pd.Series,
    param_cols: List[str],
    data_dir: str = 'data',
    params_base_dir: str = 'p4tun/parameters',
    detection_only: bool = False
) -> Dict[str, float]:
    """
    Run pipeline with historical config and collect intrinsic metrics.
    
    Args:
        tunnel_id: Tunnel identifier
        stage: Stage name (detection, sam, combined, complex_sam, sam_wraparound)
        row: Training data row with param_* columns
        param_cols: List of param column names for this stage
        data_dir: Base data directory
        params_base_dir: Base directory for parameter files
        
    Returns:
        Dict of intrinsic metrics
    """
    tunnel_dir = os.path.join(data_dir, tunnel_id)
    params_dir = os.path.join(PROJECT_ROOT, params_base_dir, tunnel_id)
    os.makedirs(params_dir, exist_ok=True)
    
    # Load existing params to merge (for stages that need both detection + sam)
    existing_detection = load_default_parameters(tunnel_id, 'detection', params_base_dir)
    existing_sam = load_default_parameters(tunnel_id, 'sam', params_base_dir)
    
    ran_detection = False
    ran_sam = False
    
    # Detection stage or combined: need to run detection with config params
    if stage in ('detection', 'combined', 'complex_detection'):
        det_param_cols = [c for c in param_cols if any(
            p in c for p in ['binary_threshold', 'hough_oblique', 'angle_positive',
                            'hough_horizontal', 'dilation', 'merge_distance',
                            'complex_hough', 'complex_angle', 'complex_eps',
                            'complex_conf', 'complex_min']
        )]
        if not det_param_cols:
            det_param_cols = [c for c in param_cols if 'param_' in c][:25]
        
        det_config = _params_row_to_detection_config(row, det_param_cols, tunnel_id)
        det_path = os.path.join(params_dir, 'parameters_detection.json')
        with open(det_path, 'w') as f:
            json.dump(det_config, f, indent=4)
        
        det_script = _get_detection_script(tunnel_id, stage)
        ran_detection = _run_script(det_script, tunnel_id, data_dir)
    
    # SAM stage or combined: need detection output, then run SAM
    # For detection-only stage: we still run SAM with defaults to get sam_metrics
    # Skip SAM if detection_only=True (faster validation)
    if not detection_only and stage in ('sam', 'combined', 'complex_sam', 'sam_wraparound', 'detection'):
        # If we didn't run detection above, run it with existing params
        if not ran_detection:
            det_script = _get_detection_script(tunnel_id, stage)
            _run_script(det_script, tunnel_id, data_dir)
        
        # For detection-only: use default SAM params. For sam/combined: use row params
        if stage == 'detection':
            # Keep existing SAM params (from load_default)
            pass  # Don't overwrite parameters_sam.json
        else:
            sam_param_cols = [c for c in param_cols if any(
                p in c for p in ['segment_width', 'k_height', 'ab_height', 'angle_deg',
                                'k_mask', 'ab_mask', 'padding', 'crop_margin',
                                'k_outer', 'ab_outer', 'ab_level', 'min_quality',
                                'k_block', 'ab_block']
            )]
            if not sam_param_cols:
                sam_param_cols = [c for c in param_cols if 'param_' in c]
            
            sam_config = _params_row_to_sam_config(row, sam_param_cols, tunnel_id)
            sam_path = os.path.join(params_dir, 'parameters_sam.json')
            with open(sam_path, 'w') as f:
                json.dump(sam_config, f, indent=4)
        
        sam_script = _get_sam_script(tunnel_id, stage)
        ran_sam = _run_script(sam_script, tunnel_id, data_dir)
    
    # Compute intrinsic metrics
    detected_csv = os.path.join(tunnel_dir, 'detected.csv')
    final_csv = os.path.join(tunnel_dir, 'final.csv')
    
    metrics = {}
    
    if ran_detection or os.path.exists(detected_csv):
        det_metrics = compute_detection_metrics(tunnel_id, detected_csv, data_dir=data_dir)
        for k, v in det_metrics.items():
            metrics[f'det_{k}'] = v
    
    # Only include sam_metrics if we actually ran SAM (avoid stale data in detection_only mode)
    if ran_sam:
        sam_metrics = compute_sam_metrics(
            tunnel_id, final_csv, detected_csv, data_dir=data_dir
        )
        for k, v in sam_metrics.items():
            metrics[f'sam_{k}'] = v
    
    return metrics

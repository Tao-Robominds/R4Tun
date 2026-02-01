"""
Combined Preprocessing: Unfolding + Denoising + Enhancing

This module runs the full preprocessing pipeline in a single stage:
1. Unfolding: Transform point cloud to cylindrical coordinates
2. Denoising: Remove noise using density-based surface detection
3. Enhancing: Upsample, interpolate boundaries, generate depth maps

Data is passed between stages in memory (no intermediate file reads).
Outputs are still saved for downstream stages and debugging.
"""

import os
import sys
import json
import importlib.util
from typing import Dict, Any, Tuple

import pandas as pd

# Import from sibling modules (names start with digits, use importlib)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec_unf = importlib.util.spec_from_file_location("_unfolding", os.path.join(_script_dir, "1_unfolding.py"))
_unf_mod = importlib.util.module_from_spec(_spec_unf)
_spec_unf.loader.exec_module(_unf_mod)
load_point_cloud = _unf_mod.load_point_cloud
unfold_from_df = _unf_mod.unfold_from_df

_spec_den = importlib.util.spec_from_file_location("_denoising", os.path.join(_script_dir, "2_denoising.py"))
_den_mod = importlib.util.module_from_spec(_spec_den)
_spec_den.loader.exec_module(_den_mod)
denoise_point_cloud = _den_mod.denoise_point_cloud

_spec_enh = importlib.util.spec_from_file_location("_enhancing", os.path.join(_script_dir, "3_enhancing.py"))
_enh_mod = importlib.util.module_from_spec(_spec_enh)
_spec_enh.loader.exec_module(_enh_mod)
enhance_from_df = _enh_mod.enhance_from_df


def get_param(params: Dict, *keys, default=None, allow_default: bool = True):
    """Get nested parameter value with optional default fallback."""
    value = params
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if allow_default:
                return default
            else:
                raise KeyError(f"Parameter not found: {' -> '.join(keys)}")
    return value


def load_parameters(tunnel_id: str, base_dir: str = "data") -> Tuple[Dict[str, Any], bool]:
    """
    Load combined preprocessing parameters from parameters_preprocessing.json.
    
    Priority:
        1. p4tun/parameters/<tunnel_id>/parameters_preprocessing.json
        2. data/<tunnel_id>/parameters_preprocessing.json
        3. p4tun/parameters/sample/parameters_preprocessing.json
        4. {} (will use hardcoded defaults)
    """
    script_dir = os.path.dirname(__file__)
    param_file = "parameters_preprocessing.json"
    
    params_path = os.path.join(script_dir, "parameters", tunnel_id, param_file)
    if os.path.exists(params_path):
        print(f"Loading parameters from {params_path}")
        with open(params_path, 'r') as f:
            return json.load(f), True
    
    tunnel_path = os.path.join(base_dir, tunnel_id, param_file)
    if os.path.exists(tunnel_path):
        print(f"Loading parameters from {tunnel_path}")
        with open(tunnel_path, 'r') as f:
            return json.load(f), True
    
    sample_path = os.path.join(script_dir, "parameters", "sample", param_file)
    if os.path.exists(sample_path):
        print(f"Loading sample parameters from {sample_path}")
        with open(sample_path, 'r') as f:
            return json.load(f), True
    
    print("Warning: No parameters_preprocessing.json found, using hardcoded defaults")
    return {}, False


# Denoising defaults (for param extraction)
DEFAULT_RADIUS_MIN = 2.7
DEFAULT_RADIUS_MAX = 2.8
DEFAULT_THETA_STEP = 0.5
DEFAULT_RADIAL_STEP = 0.001
DEFAULT_GRADIENT_THRESHOLD = 0.2
DEFAULT_GRADIENT_EPSILON = 1e-6
DEFAULT_SMOOTHING_WINDOW = 3
DEFAULT_SMOOTHING_OFFSET = 0.003


def run_preprocessing(tunnel_id: str, base_dir: str = "data") -> None:
    """
    Execute the combined preprocessing pipeline: unfolding + denoising + enhancing.
    
    Data flows in memory between stages. Outputs are saved for downstream use.
    
    Args:
        tunnel_id: Identifier for the tunnel (e.g., "1-4", "5-1").
        base_dir: Base directory for data files.
    """
    print(f"{'=' * 60}")
    print(f"Preprocessing Pipeline: {tunnel_id}")
    print(f"{'=' * 60}")
    
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    os.makedirs(tunnel_dir, exist_ok=True)
    
    # Load combined parameters
    params, params_loaded = load_parameters(tunnel_id, base_dir)
    allow_defaults = not params_loaded
    
    # Extract subsection params (fallback to empty dict if missing)
    params_unfolding = params.get("unfolding", {})
    params_denoising = params.get("denoising", {})
    params_enhancing = params.get("enhancing", {})
    
    # ---- Stage 1: Unfolding ----
    print("\n[Stage 1] Unfolding...")
    filepath = os.path.join(base_dir, f"{tunnel_id}.txt")
    df_raw = load_point_cloud(filepath)
    df_unwrapped, ring_count = unfold_from_df(df_raw, params_unfolding, allow_defaults=allow_defaults)
    
    df_unwrapped.to_csv(os.path.join(tunnel_dir, "unwrapped.csv"), index=False)
    with open(os.path.join(tunnel_dir, "ring_count.txt"), 'w') as f:
        f.write(str(ring_count))
    print(f"  Saved unwrapped.csv, ring_count={ring_count}")
    
    # ---- Stage 2: Denoising (in-memory) ----
    print("\n[Stage 2] Denoising...")
    radius_min = get_param(params_denoising, 'radius_filtering', 'radius_min', default=DEFAULT_RADIUS_MIN, allow_default=allow_defaults)
    radius_max = get_param(params_denoising, 'radius_filtering', 'radius_max', default=DEFAULT_RADIUS_MAX, allow_default=allow_defaults)
    theta_step = get_param(params_denoising, 'grid_resolution', 'theta_step', default=DEFAULT_THETA_STEP, allow_default=allow_defaults)
    radial_step = get_param(params_denoising, 'grid_resolution', 'radial_step', default=DEFAULT_RADIAL_STEP, allow_default=allow_defaults)
    gradient_threshold = get_param(params_denoising, 'gradient_detection', 'gradient_threshold', default=DEFAULT_GRADIENT_THRESHOLD, allow_default=allow_defaults)
    gradient_epsilon = get_param(params_denoising, 'gradient_detection', 'gradient_epsilon', default=DEFAULT_GRADIENT_EPSILON, allow_default=allow_defaults)
    smoothing_window = get_param(params_denoising, 'cutoff_smoothing', 'smoothing_window', default=DEFAULT_SMOOTHING_WINDOW, allow_default=allow_defaults)
    smoothing_offset = get_param(params_denoising, 'cutoff_smoothing', 'smoothing_offset', default=DEFAULT_SMOOTHING_OFFSET, allow_default=allow_defaults)
    
    df_denoised = denoise_point_cloud(
        df_unwrapped, ring_count,
        radius_min=radius_min, radius_max=radius_max,
        theta_step=theta_step, radial_step=radial_step,
        gradient_threshold=gradient_threshold, gradient_epsilon=gradient_epsilon,
        smoothing_window=smoothing_window, smoothing_offset=smoothing_offset
    )
    df_denoised.to_csv(os.path.join(tunnel_dir, "denoised.csv"), index=False)
    valid_count = (df_denoised['pred'] != 0).sum()
    print(f"  Saved denoised.csv, valid points: {valid_count}/{len(df_denoised)}")
    
    # ---- Stage 3: Enhancing (in-memory) ----
    print("\n[Stage 3] Enhancing...")
    df_enhanced = enhance_from_df(df_denoised, tunnel_dir, params_enhancing, allow_defaults=allow_defaults)
    
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete: {len(df_enhanced)} points in enhanced.csv")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 1_preprocessing.py <tunnel_id>")
        print("Example: python 1_preprocessing.py 1-4")
        sys.exit(1)

    run_preprocessing(sys.argv[1])

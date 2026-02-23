"""
Bayesian Optimization for Multisource K Y Detection Parameters

Optimizes multisource fusion/DP parameters (10D) while fixing groove-pair (4D) + grouped offsets (12D) = 16D total
to minimize mean wrap-aware |dY| between detected K positions and GT K positions.

Line detection parameters and the 16D groove-pair/offsets are fixed at previous BO values.
"""

import os
import sys
import json
import time
import argparse
import importlib.util
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import forest_minimize
from skopt.space import Real, Integer

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import detection functions
DEFAULT_AGENT_TYPE = Path(__file__).parent.name
detection_dir = PROJECT_ROOT / 'agents' / 'irregular' / '2_detection'
sys.path.insert(0, str(detection_dir))

spec = importlib.util.spec_from_file_location(
    "detection",
    os.path.join(detection_dir, "2_detection.py")
)
detection_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detection_module)

run_detection = detection_module.run_detection
load_parameters = detection_module.load_parameters


# =============================================================================
# K Y Distance Computation
# =============================================================================

def compute_k_y_distance(
    detected_k: pd.DataFrame,
    gt_k: pd.DataFrame,
    img_height: int,
) -> Dict:
    """
    Compute wrap-aware Y distance between detected and GT K positions.
    
    Args:
        detected_k: DataFrame with columns Ring, X, Y
        gt_k: DataFrame with columns Ring, X, Y
        img_height: Image height for wrap-around calculations
    
    Returns:
        Dict with mean_distance, per-ring distances, and details
    """
    # Sort by Ring to ensure alignment
    det_sorted = detected_k.sort_values('Ring').reset_index(drop=True)
    gt_sorted = gt_k.sort_values('Ring').reset_index(drop=True)
    
    distances = []
    details = []
    
    for i in range(min(len(det_sorted), len(gt_sorted))):
        det_y = det_sorted.iloc[i]['Y']
        gt_y = gt_sorted.iloc[i]['Y']
        det_x = det_sorted.iloc[i]['X']
        gt_x = gt_sorted.iloc[i]['X']
        
        # Wrap-aware Y distance
        dy_raw = abs(det_y - gt_y)
        dy = min(dy_raw, img_height - dy_raw)
        
        distances.append(dy)
        details.append({
            'ring': int(det_sorted.iloc[i]['Ring']),
            'det_y': float(det_y),
            'gt_y': float(gt_y),
            'dy': float(dy),
            'det_x': float(det_x),
            'gt_x': float(gt_x),
            'dx': float(abs(det_x - gt_x)),
        })
    
    mean_distance = float(np.mean(distances)) if distances else float('inf')
    
    return {
        'mean_k_y_distance': mean_distance,
        'k_y_distances': distances,
        'details': details,
        'num_matched': len(distances),
        'num_detected': len(detected_k),
        'num_gt': len(gt_k),
    }


# =============================================================================
# Objective Function
# =============================================================================

class MultisourceObjective:
    """Objective function for multisource BO: minimize mean |dY| vs GT K positions."""
    
    def __init__(
        self,
        config: dict,
        tunnel_id: str,
        data_dir: str,
        logs_dir: str,
        verbose: bool = True,
    ):
        self.config = config
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        self.logs_dir = logs_dir
        self.verbose = verbose
        self.img_height = config['img_height']
        
        # Load GT K positions
        gt_file = os.path.join(self.tunnel_dir, 'all_segments_gt.csv')
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"GT file not found: {gt_file}")
        gt_segments = pd.read_csv(gt_file)
        self.gt_k = gt_segments[gt_segments['Block'] == 'K'].sort_values('Ring').reset_index(drop=True)
        
        # Load base detection params (line detection + other fixed params)
        params_file = os.path.join(
            PROJECT_ROOT, 'agents', 'irregular', '2_detection',
            'parameters', tunnel_id, 'parameters_detection.json'
        )
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Detection params not found: {params_file}")
        with open(params_file, 'r') as f:
            self.base_params = json.load(f)
        
        # Get fixed 16D params from config (groove-pair + group offsets)
        search_space = config['search_space']
        self.fixed_16d = search_space.get('fixed_params', {})
        
        # Build search space from config (10D multisource params)
        self.dimensions = []
        self.param_names = []
        
        # Multisource params (10D)
        for name, (lo, hi) in search_space['multisource_bounds'].items():
            if name in ['col_blur_ksize', 'top_n_candidates']:
                self.dimensions.append(Integer(int(lo), int(hi), name=name))
            else:
                self.dimensions.append(Real(lo, hi, name=name))
            self.param_names.append(name)
        
        # Warmstart
        warmstart = search_space.get('warmstart', {})
        self.x0 = [warmstart.get(n, (self.dimensions[i].low + self.dimensions[i].high) / 2)
                   for i, n in enumerate(self.param_names)]
        
        os.makedirs(logs_dir, exist_ok=True)
        self.eval_count = 0
        self.best_score = float('inf')
        self.best_params = None
        
        if verbose:
            print(f"Multisource BO ({len(self.param_names)}D)")
            print(f"  Multisource params: {len(self.param_names)}")
            print(f"  Fixed groove-pair (4D) + group offsets (12D): 16D")
            print(f"  Line detection params: fixed")
            print(f"  Objective: minimize mean |dY| vs GT K positions")
            print(f"  GT K positions: {len(self.gt_k)}")
    
    def __call__(self, params: List) -> float:
        """Evaluate multisource parameters."""
        self.eval_count += 1
        t0 = time.time()
        
        try:
            # Build param dict from 10D tuned params
            param_dict = {}
            for name, val in zip(self.param_names, params):
                # Convert numpy types to native Python types
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    param_dict[name] = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    param_dict[name] = float(val)
                elif hasattr(val, 'item'):
                    param_dict[name] = val.item()
                else:
                    param_dict[name] = val
            
            # Ensure col_blur_ksize is odd (Gaussian blur requires odd kernel size)
            if 'col_blur_ksize' in param_dict:
                param_dict['col_blur_ksize'] = int(param_dict['col_blur_ksize']) | 1
            
            # Merge: base params (line detection + other) + fixed 16D + 10D tuned
            full_params = self.base_params.copy()
            full_params.update(self.fixed_16d)  # Fixed groove-pair + offsets
            full_params.update(param_dict)      # Tuned multisource params
            full_params['k_detection_method'] = 'multisource'
            full_params['expansion_method'] = 'grouped_offsets'
            
            # Save params (atomically via temp file)
            params_file = os.path.join(
                PROJECT_ROOT, 'agents', 'irregular', '2_detection',
                'parameters', self.tunnel_id, 'parameters_detection.json'
            )
            import tempfile
            import shutil
            with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(params_file), delete=False, suffix='.json') as f:
                json.dump(full_params, f, indent=2)
                temp_path = f.name
            shutil.move(temp_path, params_file)
            
            # Run detection
            import io
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                detection_result = run_detection(self.tunnel_id, self.data_dir)
            
            # Extract K positions
            if isinstance(detection_result, tuple):
                k_positions, _ = detection_result
            else:
                k_positions = detection_result
            
            # Convert to DataFrame if needed
            if isinstance(k_positions, pd.DataFrame):
                detected_k = k_positions.copy()
            else:
                detected_k = pd.DataFrame(k_positions)
            
            # Read detected.csv (more reliable than return value)
            detected_csv = os.path.join(self.tunnel_dir, 'detected.csv')
            if os.path.exists(detected_csv):
                detected_k = pd.read_csv(detected_csv)
            
            # Ensure X, Y columns exist
            if 'X' not in detected_k.columns or 'Y' not in detected_k.columns:
                raise ValueError(f"detected.csv missing X or Y columns")
            
            # Match rings by X position (GT and detected should have similar X values)
            # Sort both by X and assign ring indices
            detected_k = detected_k.sort_values('X').reset_index(drop=True)
            gt_k_sorted = self.gt_k.sort_values('X').reset_index(drop=True)
            
            # Add Ring column to detected_k by matching X positions
            if len(detected_k) == len(gt_k_sorted):
                detected_k['Ring'] = gt_k_sorted['Ring'].values
            else:
                # Fallback: assign sequential rings
                detected_k['Ring'] = range(len(detected_k))
            
            # Compute distance
            results = compute_k_y_distance(detected_k, self.gt_k, self.img_height)
            score = results['mean_k_y_distance']
            runtime = time.time() - t0
            
            if score < self.best_score:
                self.best_score = score
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"  [#{self.eval_count}] NEW BEST mean |dY|={score:.1f}px  "
                          f"(matched={results['num_matched']}/{len(self.gt_k)})  ({runtime:.1f}s)")
            elif self.verbose and (self.eval_count <= 5 or self.eval_count % 20 == 0):
                print(f"  [#{self.eval_count}] mean |dY|={score:.1f}px  best={self.best_score:.1f}px  ({runtime:.1f}s)")
            
            # Log
            self._log(param_dict, results, runtime)
            return score  # minimize
            
        except Exception as e:
            runtime = time.time() - t0
            if self.verbose:
                print(f"  [#{self.eval_count}] ERROR: {e}  ({runtime:.1f}s)")
            self._log({}, {}, runtime, error=str(e))
            return 10000.0  # large penalty
    
    def _log(self, params, results, runtime, error=None):
        """Log trial to JSON file."""
        trial_id = f"multisource_{self.tunnel_id}_{self.eval_count:03d}"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            return obj
        
        log_data = {
            "trial_id": trial_id,
            "params": convert_to_native(params),
            "mean_k_y_distance": convert_to_native(results.get('mean_k_y_distance', float('inf'))),
            "k_y_distances": convert_to_native(results.get('k_y_distances', [])),
            "details": convert_to_native(results.get('details', [])),
            "num_matched": convert_to_native(results.get('num_matched', 0)),
            "runtime_sec": runtime,
        }
        if error:
            log_data["error"] = error
        
        log_file = os.path.join(self.logs_dir, f"{trial_id}.json")
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)


# =============================================================================
# Main Optimization
# =============================================================================

def run_multisource_bo(
    tunnel_id: str,
    data_dir: str = "data/wrap",
    n_calls: int = 150,
    n_initial: int = 20,
    verbose: bool = True,
) -> Dict:
    """
    Run multisource detection BO.
    
    Args:
        tunnel_id: Tunnel id (e.g. '4-1')
        data_dir: Base data directory
        n_calls: Total BO iterations
        n_initial: Random initial points
        verbose: Print progress
    
    Returns:
        Dictionary with best parameters and score
    """
    config_path = PROJECT_ROOT / "bo" / "complex_staggered" / "configs" / f"multisource_{tunnel_id}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    
    logs_dir = str(PROJECT_ROOT / "bo" / "complex_staggered" / f"logs_multisource_{tunnel_id}")
    
    print(f"\n{'=' * 70}")
    print(f"MULTISOURCE DETECTION BO — Tunnel {tunnel_id}")
    print(f"N calls: {n_calls}, N initial: {n_initial}")
    print(f"{'=' * 70}")
    
    objective = MultisourceObjective(config, tunnel_id, data_dir, logs_dir, verbose)
    x0 = [objective.x0]
    y0 = None
    
    print(f"\nSearch space: {len(objective.param_names)} dimensions")
    for i, (name, dim) in enumerate(zip(objective.param_names, objective.dimensions)):
        ws = x0[0][i] if x0 else "?"
        if isinstance(dim, Integer):
            print(f"  {name:25s}  [{dim.low}, {dim.high}]  warmstart={ws:.0f}")
        else:
            print(f"  {name:25s}  [{dim.low:.1f}, {dim.high:.1f}]  warmstart={ws:.1f}")
    print(f"\nStarting optimization...")
    
    result = forest_minimize(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        x0=x0,
        y0=y0,
        random_state=42,
        verbose=False,
    )
    
    best_score = result.fun
    best_dict = {}
    for name, val in zip(objective.param_names, result.x):
        # Convert numpy types to native Python types
        if isinstance(val, (np.integer, np.int64, np.int32)):
            best_dict[name] = int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            best_dict[name] = float(val)
        elif hasattr(val, 'item'):
            best_dict[name] = val.item()
        else:
            best_dict[name] = val
    
    # Ensure col_blur_ksize is odd in best_dict
    if 'col_blur_ksize' in best_dict:
        best_dict['col_blur_ksize'] = int(best_dict['col_blur_ksize']) | 1
    
    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Best mean |dY| = {best_score:.1f}px")
    print(f"{'=' * 70}")
    for name, val in best_dict.items():
        if isinstance(val, (int, float)):
            print(f"  {name:25s}: {val:.2f}")
    
    # Save results
    results_file = os.path.join(logs_dir, f"best_multisource_{tunnel_id}.json")
    with open(results_file, "w") as f:
        json.dump({
            "mean_k_y_distance": float(best_score) if isinstance(best_score, (np.floating, np.float64)) else best_score,
            "params": best_dict
        }, f, indent=2)
    print(f"\nSaved best params to {results_file}")
    
    return {"mean_k_y_distance": best_score, "params": best_dict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multisource detection BO for 4-1")
    parser.add_argument("tunnel_id", help="Tunnel id (e.g. 4-1)")
    parser.add_argument("--data-dir", default="data/wrap")
    parser.add_argument("--n-calls", type=int, default=150)
    parser.add_argument("--n-initial", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    run_multisource_bo(
        args.tunnel_id,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        verbose=not args.quiet,
    )

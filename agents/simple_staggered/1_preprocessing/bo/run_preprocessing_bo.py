"""
Bayesian Optimization for Preprocessing Parameters with F2 Objective

Optimizes preprocessing parameters (ring_spacing, radius_min, radius_max,
gradient_threshold) to maximize Retention F2 score.

F2 weights recall 4x more than precision, appropriate because false negatives
(removing true lining points) are irreversible for downstream detection/SAM.
"""

import os
import sys
import json
import glob
import time
import argparse
import importlib.util
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import preprocessing functions
preprocessing_dir = os.path.join(PROJECT_ROOT, 'agents', 'simple_staggered', '1_preprocessing')
sys.path.insert(0, preprocessing_dir)

spec = importlib.util.spec_from_file_location(
    "preprocessing",
    os.path.join(preprocessing_dir, "1_preprocessing.py")
)
preprocessing_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing_module)

load_point_cloud = preprocessing_module.load_point_cloud
run_preprocessing = preprocessing_module.run_preprocessing
load_parameters = preprocessing_module.load_parameters
get_param = preprocessing_module.get_param
DEFAULT_RING_SPACING = preprocessing_module.DEFAULT_RING_SPACING
DEFAULT_TUNNEL_DIAMETER = preprocessing_module.DEFAULT_TUNNEL_DIAMETER


# =============================================================================
# Search Space Definition
# =============================================================================

def get_preprocessing_dimensions() -> Tuple[List, List[str]]:
    """
    Define search space for preprocessing parameters (4 total).
    
    Returns:
        Tuple of (dimensions list, parameter names list)
    """
    dimensions = [
        Real(1.0, 1.4, name='ring_spacing'),             # Ring spacing in meters
        Real(2.60, 2.77, name='radius_min'),              # Inner radius filter
        Real(2.77, 2.95, name='radius_max'),              # Outer radius filter
        Real(0.05, 0.5, name='gradient_threshold'),       # Surface cutoff aggressiveness
    ]
    
    param_names = [
        'ring_spacing',
        'radius_min',
        'radius_max',
        'gradient_threshold',
    ]
    
    return dimensions, param_names


# =============================================================================
# F2 Score Computation
# =============================================================================

def compute_f2_score(
    df_denoised: pd.DataFrame,
    beta: float = 2.0
) -> Dict:
    """
    Compute Retention F2 score from denoised point cloud.
    
    Args:
        df_denoised: DataFrame with 'segment' (GT) and 'pred' (kept/removed) columns
        beta: F-score beta parameter (default 2.0 for F2)
    
    Returns:
        Dictionary with confusion matrix, metrics, and per-segment/ring breakdowns
    """
    # Ground truth: segment > 0 = lining (positive), segment == 0 = non-lining (negative)
    # Prediction: pred != 0 = kept, pred == 0 = removed
    
    lining_mask = df_denoised['segment'] > 0
    kept_mask = df_denoised['pred'] != 0
    
    # Confusion matrix
    tp = int((lining_mask & kept_mask).sum())   # Lining kept
    fp = int((~lining_mask & kept_mask).sum())  # Non-lining kept
    fn = int((lining_mask & ~kept_mask).sum())  # Lining removed
    tn = int((~lining_mask & ~kept_mask).sum()) # Non-lining removed
    
    # Overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F-beta score
    if beta**2 * precision + recall > 0:
        f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    else:
        f_beta = 0.0
    
    # Per-segment breakdown
    by_segment = {}
    for seg_id in sorted(df_denoised['segment'].unique()):
        seg_mask = df_denoised['segment'] == seg_id
        if seg_id > 0:  # Lining segment
            seg_tp = int((seg_mask & kept_mask).sum())
            seg_fn = int((seg_mask & ~kept_mask).sum())
            seg_recall = seg_tp / (seg_tp + seg_fn) if (seg_tp + seg_fn) > 0 else 0.0
            seg_f2 = (1 + beta**2) * seg_recall / (beta**2 + seg_recall) if seg_recall > 0 else 0.0
            by_segment[str(int(seg_id))] = {
                'tp': seg_tp,
                'fn': seg_fn,
                'recall': float(seg_recall),
                'f2': float(seg_f2),
            }
        else:  # Background/non-lining
            seg_fp = int((seg_mask & kept_mask).sum())
            seg_tn = int((seg_mask & ~kept_mask).sum())
            by_segment['0'] = {
                'fp': seg_fp,
                'tn': seg_tn,
            }
    
    # Per-ring breakdown
    by_ring = {}
    for ring_id in sorted(df_denoised['ring'].unique()):
        ring_mask = df_denoised['ring'] == ring_id
        ring_lining = ring_mask & lining_mask
        ring_tp = int((ring_lining & kept_mask).sum())
        ring_fn = int((ring_lining & ~kept_mask).sum())
        ring_recall = ring_tp / (ring_tp + ring_fn) if (ring_tp + ring_fn) > 0 else 0.0
        ring_f2 = (1 + beta**2) * ring_recall / (beta**2 + ring_recall) if ring_recall > 0 else 0.0
        by_ring[str(int(ring_id))] = {
            'tp': ring_tp,
            'fn': ring_fn,
            'recall': float(ring_recall),
            'f2': float(ring_f2),
        }
    
    return {
        'confusion': {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        },
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f2': float(f_beta),
        },
        'by_segment': by_segment,
        'by_ring': by_ring,
    }


# =============================================================================
# Objective Function
# =============================================================================

class PreprocessingObjective:
    """
    Objective function that evaluates preprocessing parameters using F2 score.
    """
    
    def __init__(
        self,
        tunnel_id: str,
        data_dir: str = 'data',
        verbose: bool = True,
        eval_offset: int = 0,
    ):
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.verbose = verbose
        
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        self.params_dir = os.path.join(
            PROJECT_ROOT,
            'agents', 'simple_staggered', '1_preprocessing',
            'parameters', tunnel_id
        )
        os.makedirs(self.params_dir, exist_ok=True)
        
        # Load raw point cloud to get input count
        raw_file = os.path.join(data_dir, f"{tunnel_id}.txt")
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw point cloud not found: {raw_file}")
        
        df_raw = load_point_cloud(raw_file)
        self.num_points_input = len(df_raw)
        
        # Get search space
        self.dimensions, self.param_names = get_preprocessing_dimensions()
        
        # Tracking — eval_offset allows continuing numbering from previous runs
        self.eval_offset = eval_offset
        self.eval_count = 0
        self.best_score = -np.inf
        self.best_params = None
        self.history = []
        self.logs_dir = os.path.join(
            PROJECT_ROOT,
            'agents', 'simple_staggered', '1_preprocessing', 'bo', 'logs'
        )
        os.makedirs(self.logs_dir, exist_ok=True)
        
        if verbose:
            print(f"Preprocessing BO for tunnel {tunnel_id}")
            print(f"Input points: {self.num_points_input:,}")
            print(f"Parameters: {len(self.param_names)}")
            print(f"Eval numbering starts at: {self.eval_offset + 1}")
            print(f"Logs directory: {self.logs_dir}")
    
    @property
    def global_eval_index(self) -> int:
        """Current global eval index (offset + local count)."""
        return self.eval_offset + self.eval_count
    
    def __call__(self, params: List) -> float:
        """
        Evaluate preprocessing parameters.
        
        Args:
            params: List of parameter values in order of param_names
        
        Returns:
            Negative F2 score (for minimization)
        """
        self.eval_count += 1
        start_time = time.time()
        
        try:
            # Convert params to dict
            param_dict = dict(zip(self.param_names, params))
            
            # Load existing params to preserve fixed values
            existing_params, _ = load_parameters(self.tunnel_id, self.data_dir)
            
            # Merge tunable params with fixed params
            params_to_save = {
                'ring_spacing': param_dict['ring_spacing'],
                'tunnel_diameter': get_param(existing_params, 'tunnel_diameter', default=DEFAULT_TUNNEL_DIAMETER, allow_default=True),
                'radius_min': param_dict['radius_min'],
                'radius_max': param_dict['radius_max'],
                'gradient_threshold': param_dict['gradient_threshold'],
            }
            
            # Save parameters
            params_file = os.path.join(self.params_dir, 'parameters_preprocessing.json')
            with open(params_file, 'w') as f:
                json.dump(params_to_save, f, indent=4)
            
            # Run preprocessing (suppress output)
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_preprocessing(self.tunnel_id, self.data_dir)
            
            # Load denoised.csv to compute F2
            denoised_file = os.path.join(self.tunnel_dir, 'denoised.csv')
            if not os.path.exists(denoised_file):
                raise FileNotFoundError(f"Denoised file not found: {denoised_file}")
            
            df_denoised = pd.read_csv(denoised_file)
            
            # Compute F2 score
            results = compute_f2_score(df_denoised, beta=2.0)
            f2_score = results['metrics']['f2']
            num_points_kept = int((df_denoised['pred'] != 0).sum())
            kept_ratio = num_points_kept / len(df_denoised) * 100.0
            
            runtime = time.time() - start_time
            
            # Track best
            if f2_score > self.best_score:
                self.best_score = f2_score
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"  [Eval {self.global_eval_index}] New best F2: {f2_score:.4f} "
                          f"(P={results['metrics']['precision']:.4f}, "
                          f"R={results['metrics']['recall']:.4f}, "
                          f"kept={kept_ratio:.1f}%)")
            
            # Log trial
            self._log_trial(
                param_dict,
                results,
                num_points_kept,
                kept_ratio,
                runtime,
            )
            
            # Record history
            self.history.append({
                'eval': self.global_eval_index,
                'params': param_dict,
                'f2': f2_score,
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'kept_ratio': kept_ratio,
            })
            
            if self.verbose and self.eval_count % 10 == 0:
                print(f"  [Eval {self.global_eval_index}] F2: {f2_score:.4f}, "
                      f"kept: {kept_ratio:.1f}%")
            
            return -f2_score  # Negative for minimization
            
        except Exception as e:
            runtime = time.time() - start_time
            if self.verbose:
                print(f"  [Eval {self.global_eval_index}] Error: {e}")
            # Log failed trial
            self._log_trial(
                dict(zip(self.param_names, params)),
                None,
                0,
                0.0,
                runtime,
                error=str(e),
            )
            return 0.0  # Return worst score on error
    
    def _log_trial(
        self,
        params: Dict,
        results: Optional[Dict],
        num_points_kept: int,
        kept_ratio: float,
        runtime: float,
        error: Optional[str] = None,
    ):
        """Log trial to JSON file."""
        global_idx = self.global_eval_index
        trial_id = f"preproc_{self.tunnel_id}_{global_idx:03d}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        log_data = {
            'schema_version': 'r4tun.stageA.v1',
            'trial': {
                'trial_id': trial_id,
                'timestamp_utc': timestamp,
                'tunnel_id': self.tunnel_id,
                'assembly_type': 'simple_staggered',
            },
            'params': {
                'ring_spacing': float(params['ring_spacing']),
                'radius_min': float(params['radius_min']),
                'radius_max': float(params['radius_max']),
                'gradient_threshold': float(params['gradient_threshold']),
            },
        }
        
        if error:
            log_data['trace'] = {'warnings': [f"Error: {error}"]}
            log_data['bo'] = {
                'objective_name': 'retention_f2',
                'objective_value': 0.0,
                'eval_index': global_idx,
                'runtime_sec': runtime,
                'is_feasible': False,
            }
        else:
            log_data['outputs'] = {
                'num_points_input': self.num_points_input,
                'num_points_kept': num_points_kept,
                'kept_ratio_pct': float(kept_ratio),
                'confusion': results['confusion'],
                'metrics': results['metrics'],
                'by_segment': results['by_segment'],
                'by_ring': results['by_ring'],
            }
            log_data['bo'] = {
                'objective_name': 'retention_f2',
                'objective_value': float(results['metrics']['f2']),
                'eval_index': global_idx,
                'runtime_sec': float(runtime),
                'is_feasible': True,
            }
        
        # Save log file
        log_file = os.path.join(self.logs_dir, f"{trial_id}.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def save_best_params(self) -> Optional[str]:
        """Save best parameters to JSON file."""
        if self.best_params is None:
            return None
        
        # Load existing params to preserve fixed values
        existing_params, _ = load_parameters(self.tunnel_id, self.data_dir)
        
        params_to_save = {
            'ring_spacing': float(self.best_params['ring_spacing']),
            'tunnel_diameter': get_param(existing_params, 'tunnel_diameter', default=DEFAULT_TUNNEL_DIAMETER, allow_default=True),
            'radius_min': float(self.best_params['radius_min']),
            'radius_max': float(self.best_params['radius_max']),
            'gradient_threshold': float(self.best_params['gradient_threshold']),
            'target_distances': get_param(existing_params, 'target_distances', default=DEFAULT_TARGET_DISTANCES, allow_default=True),
        }
        
        params_file = os.path.join(self.params_dir, 'parameters_preprocessing.json')
        with open(params_file, 'w') as f:
            json.dump(params_to_save, f, indent=4)
        
        return params_file


# =============================================================================
# Utilities
# =============================================================================

def find_max_trial_index(logs_dir: str, tunnel_id: str) -> int:
    """Find the highest trial index from existing log files."""
    pattern = os.path.join(logs_dir, f"preproc_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    max_idx = 0
    for f in log_files:
        basename = os.path.basename(f)
        # e.g. preproc_1-4_035.json -> 035
        try:
            idx = int(basename.split('_')[-1].replace('.json', ''))
            max_idx = max(max_idx, idx)
        except ValueError:
            pass
    return max_idx


def load_best_from_logs(logs_dir: str, tunnel_id: str) -> Optional[Tuple[List[float], float]]:
    """
    Load the best trial from existing logs to use as warm-start x0/y0.
    
    Returns:
        Tuple of (param_values_list, negative_f2) or None if no logs found.
        param_values_list is in the order of get_preprocessing_dimensions().
    """
    pattern = os.path.join(logs_dir, f"preproc_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    
    best_f2 = -1
    best_params = None
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        if 'bo' not in data or 'objective_value' not in data['bo']:
            continue
        
        f2 = data['bo']['objective_value']
        if f2 > best_f2:
            best_f2 = f2
            best_params = data.get('params', {})
    
    if best_params is None or best_f2 <= 0:
        return None
    
    # Build param list in dimension order
    # If ring_spacing is missing from old logs, use default 1.2
    param_values = [
        best_params.get('ring_spacing', 1.2),
        best_params['radius_min'],
        best_params['radius_max'],
        best_params['gradient_threshold'],
    ]
    
    return param_values, -best_f2  # negative for minimization


# =============================================================================
# Main Optimization
# =============================================================================

def run_preprocessing_bo(
    tunnel_id: str,
    data_dir: str = 'data',
    n_calls: int = 30,
    n_initial_points: int = 5,
    verbose: bool = True,
) -> Dict:
    """Run Bayesian Optimization for preprocessing parameters."""
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING BAYESIAN OPTIMIZATION - Tunnel {tunnel_id}")
    print(f"{'='*70}")
    
    logs_dir = os.path.join(
        PROJECT_ROOT,
        'agents', 'simple_staggered', '1_preprocessing', 'bo', 'logs'
    )
    
    # Determine eval offset from existing logs
    eval_offset = find_max_trial_index(logs_dir, tunnel_id)
    
    # Initialize objective
    objective = PreprocessingObjective(
        tunnel_id=tunnel_id,
        data_dir=data_dir,
        verbose=verbose,
        eval_offset=eval_offset,
    )
    
    print(f"\nSearch space: {len(objective.param_names)} parameters")
    print(f"N calls: {n_calls}, N initial: {n_initial_points}")
    print(f"Objective: Retention F2 (beta=2.0)")
    
    # Warm-start from best previous trial
    x0 = None
    y0 = None
    warm_start = load_best_from_logs(logs_dir, tunnel_id)
    if warm_start is not None:
        x0_vals, y0_val = warm_start
        x0 = [x0_vals]
        y0 = [y0_val]
        print(f"\nWarm-starting from previous best (F2={-y0_val:.4f}):")
        for name, val in zip(objective.param_names, x0_vals):
            print(f"  {name}: {val}")
    
    # Run optimization
    print(f"\nStarting optimization...")
    result = gp_minimize(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        x0=x0,
        y0=y0,
        random_state=42,
        verbose=False,
    )
    
    # Results
    best_params = dict(zip(objective.param_names, result.x))
    best_f2 = -result.fun  # Negate back
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best F2 score: {best_f2:.4f}")
    print(f"\nBest parameters:")
    for name, value in best_params.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.6f}")
        else:
            print(f"  {name}: {value}")
    
    # Save best parameters
    filepath = objective.save_best_params()
    if filepath:
        print(f"\nSaved parameters to: {filepath}")
    
    return {
        'tunnel_id': tunnel_id,
        'best_f2': best_f2,
        'best_params': best_params,
        'n_evaluations': objective.eval_count,
        'history': objective.history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing BO with F2 objective')
    parser.add_argument('tunnel_id', type=str, help='Tunnel identifier (e.g., 1-4)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--n-calls', type=int, default=30, help='Total evaluations')
    parser.add_argument('--n-initial', type=int, default=5, help='Initial random points')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    run_preprocessing_bo(
        tunnel_id=args.tunnel_id,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        verbose=args.verbose,
    )

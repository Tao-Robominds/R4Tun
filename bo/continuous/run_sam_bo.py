"""
Bayesian Optimization for SAM Segmentation Parameters with mIoU Objective

Optimizes SAM segmentation parameters to maximize mIoU (mean Intersection over Union)
against ground truth segment labels.

Uses forest_minimize (Random Forest surrogate) for 10D SAM search space.
SAM reads detected.csv from detection stage and produces final.csv with pred column.
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
from skopt import forest_minimize
from skopt.space import Real, Integer
from sklearn.metrics import jaccard_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Extract agent_type from directory name
DEFAULT_AGENT_TYPE = Path(__file__).parent.name

# Import SAM functions
sam_dir = PROJECT_ROOT / 'agents' / DEFAULT_AGENT_TYPE / '3_segmentation'
sys.path.insert(0, str(sam_dir))

spec = importlib.util.spec_from_file_location(
    "sam",
    os.path.join(sam_dir, "3_sam.py")
)
sam_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sam_module)

run_sam = sam_module.run_sam

# Import evaluation for mIoU computation
eval_dir = PROJECT_ROOT / 'agents' / DEFAULT_AGENT_TYPE
sys.path.insert(0, str(eval_dir))

spec = importlib.util.spec_from_file_location(
    "evaluation",
    os.path.join(eval_dir, "evaluation.py")
)
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

calculate_metrics = eval_module.calculate_metrics
get_class_names = eval_module.get_class_names
detect_segment_count = eval_module.detect_segment_count


# =============================================================================
# Search Space Definition (10 parameters)
# =============================================================================

def get_sam_dimensions(
    tunnel_id: str = None,
    agent_type: str = DEFAULT_AGENT_TYPE
) -> Tuple[List, List[str], Dict]:
    """
    Define search space for SAM parameters (10D).
    
    Loads per-tunnel config from bo/{agent_type}/configs/sam_{tunnel_id}.json if it exists.
    If config exists:
        - Excludes parameters listed in fixed_params from search space
        - Overrides bounds from narrowed_bounds
    If no config: returns full 10D default space.
    
    Args:
        tunnel_id: Tunnel identifier (e.g., '1-4')
        agent_type: Agent type (e.g., 'simple_staggered')
    
    Returns:
        Tuple of (dimensions list, parameter names list, fixed_params dict)
    """
    # Default bounds (10D SAM space)
    default_bounds = {
        'segment_width': (1050.0, 1350.0),
        'angle_deg': (5.5, 9.0),
        'k_mask_width': (500.0, 750.0),
        'k_mask_height_pos': (500.0, 750.0),
        'k_mask_height_neg': (350.0, 650.0),
        'ab_mask_width': (500.0, 750.0),
        'ab_mask_height': (1400.0, 1800.0),
        'padding': (80, 200),
        'crop_margin': (25, 90),
        'min_quality_threshold': (0.2, 0.6),
    }
    
    # Load per-tunnel config if it exists
    fixed_params = {}
    narrowed_bounds = {}
    
    if tunnel_id:
        config_file = PROJECT_ROOT / 'bo' / agent_type / 'configs' / f'sam_{tunnel_id}.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            fixed_params = config.get('fixed_params', {})
            narrowed_bounds = config.get('narrowed_bounds', {})
            # Convert JSON arrays to tuples
            narrowed_bounds_converted = {}
            for key, value in narrowed_bounds.items():
                if isinstance(value, list):
                    narrowed_bounds_converted[key] = tuple(value)
                else:
                    narrowed_bounds_converted[key] = value
            narrowed_bounds = narrowed_bounds_converted
    
    # Build bounds dict (narrowed overrides default)
    bounds = default_bounds.copy()
    bounds.update(narrowed_bounds)
    
    # Build dimensions and param_names, excluding fixed params
    dimensions = []
    param_names = []
    
    param_defs = [
        ('segment_width', Real, bounds['segment_width']),
        ('angle_deg', Real, bounds['angle_deg']),
        ('k_mask_width', Real, bounds['k_mask_width']),
        ('k_mask_height_pos', Real, bounds['k_mask_height_pos']),
        ('k_mask_height_neg', Real, bounds['k_mask_height_neg']),
        ('ab_mask_width', Real, bounds['ab_mask_width']),
        ('ab_mask_height', Real, bounds['ab_mask_height']),
        ('padding', Integer, bounds['padding']),
        ('crop_margin', Integer, bounds['crop_margin']),
        ('min_quality_threshold', Real, bounds['min_quality_threshold']),
    ]
    
    for name, param_type, (low, high) in param_defs:
        if name not in fixed_params:
            dimensions.append(param_type(low, high, name=name))
            param_names.append(name)
    
    return dimensions, param_names, fixed_params


def params_to_sam_json(params: List, param_names: List[str], fixed_params: Dict = None) -> Dict:
    """
    Convert BO parameters to SAM JSON structure.
    
    Args:
        params: List of BO-tuned parameter values
        param_names: List of parameter names corresponding to params
        fixed_params: Dict of fixed parameter values (not in BO search space)
    
    Returns:
        SAM parameters dict
    """
    if fixed_params is None:
        fixed_params = {}
    
    param_dict = dict(zip(param_names, params))
    
    # Merge fixed params into param_dict (fixed params take precedence)
    param_dict.update(fixed_params)
    
    return {
        'segment_width': float(param_dict['segment_width']),
        'angle_deg': float(param_dict['angle_deg']),
        'k_mask_width': float(param_dict['k_mask_width']),
        'k_mask_height_pos': float(param_dict['k_mask_height_pos']),
        'k_mask_height_neg': float(param_dict['k_mask_height_neg']),
        'ab_mask_width': float(param_dict['ab_mask_width']),
        'ab_mask_height': float(param_dict['ab_mask_height']),
        'padding': int(param_dict['padding']),
        'crop_margin': int(param_dict['crop_margin']),
        'min_quality_threshold': float(param_dict['min_quality_threshold']),
    }


# =============================================================================
# mIoU Computation
# =============================================================================

def compute_miou(tunnel_id: str, tunnel_dir: str, segment_count: int) -> Dict:
    """
    Compute mIoU from final.csv (pred vs segment columns).
    
    Args:
        tunnel_id: Tunnel identifier
        tunnel_dir: Path to tunnel data directory
        segment_count: Number of segments per ring (6 or 7)
    
    Returns:
        Dictionary with mIoU, OA, F1, and per-class IoU
    """
    final_csv = os.path.join(tunnel_dir, 'final.csv')
    if not os.path.exists(final_csv):
        raise FileNotFoundError(f"final.csv not found at {final_csv}. Run SAM first.")
    
    df = pd.read_csv(final_csv)
    
    if 'segment' not in df.columns:
        raise ValueError(f"final.csv missing 'segment' column (ground truth)")
    if 'pred' not in df.columns:
        raise ValueError(f"final.csv missing 'pred' column (predictions)")
    
    gt_labels = df['segment'].values
    pred_labels = df['pred'].values
    
    # Convert to int, handling NaN
    gt_labels = np.nan_to_num(gt_labels, nan=-1).astype(int)
    pred_labels = np.nan_to_num(pred_labels, nan=-1).astype(int)
    
    # Get class names
    class_names = get_class_names(segment_count)
    max_class = segment_count  # B2-block (6) or B2-block (7)
    
    # Calculate metrics
    results = calculate_metrics(gt_labels, pred_labels, class_names, max_class)
    
    return results


# =============================================================================
# Objective Function
# =============================================================================

class SamObjective:
    """Objective function for SAM BO: maximize mIoU."""
    
    def __init__(
        self,
        tunnel_id: str,
        data_dir: str = 'data',
        verbose: bool = True,
        eval_offset: int = 0,
        agent_type: str = DEFAULT_AGENT_TYPE,
    ):
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.verbose = verbose
        self.eval_offset = eval_offset
        self.agent_type = agent_type
        
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        self.params_dir = os.path.join(
            PROJECT_ROOT,
            'agents', agent_type, '3_segmentation',
            'parameters', tunnel_id
        )
        os.makedirs(self.params_dir, exist_ok=True)
        
        self.logs_dir = os.path.join(
            PROJECT_ROOT,
            'bo', agent_type, 'logs'
        )
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Verify inputs exist
        detected_csv = os.path.join(self.tunnel_dir, 'detected.csv')
        if not os.path.exists(detected_csv):
            raise FileNotFoundError(
                f"detected.csv not found at {detected_csv}. Run detection first."
            )
        
        # Get search space
        self.dimensions, self.param_names, self.fixed_params = get_sam_dimensions(
            tunnel_id, agent_type
        )
        
        # Detect segment count (default 6 for simple_staggered/continuous, 7 for complex_staggered)
        # For continuous, detect_segment_count doesn't have a default parameter
        if agent_type == 'continuous':
            self.segment_count = detect_segment_count(self.tunnel_dir)
        else:
            default_segments = 6 if agent_type != 'complex_staggered' else 7
            self.segment_count = detect_segment_count(self.tunnel_dir, default=default_segments)
        
        # Track best
        self.best_score = -1.0
        self.best_params = None
        self.eval_count = 0
        self.history = []
        
        if verbose:
            print(f"SAM BO for tunnel {tunnel_id}")
            print(f"Segment count: {self.segment_count} (auto-detected)")
            print(f"Parameters: {len(self.param_names)} (fixed: {len(self.fixed_params)})")
            if self.fixed_params:
                print(f"Fixed parameters: {list(self.fixed_params.keys())}")
            print(f"Eval numbering starts at: {self.eval_offset + 1}")
            print(f"Logs directory: {self.logs_dir}")
    
    @property
    def global_eval_index(self) -> int:
        """Current global eval index (offset + local count)."""
        return self.eval_offset + self.eval_count
    
    def __call__(self, params: List) -> float:
        """
        Evaluate SAM parameters.
        
        Args:
            params: List of parameter values in order of param_names
        
        Returns:
            Negative mIoU (for minimization)
        """
        self.eval_count += 1
        start_time = time.time()
        
        try:
            # Convert params to dict (merge with fixed params)
            param_dict = params_to_sam_json(params, self.param_names, self.fixed_params)
            
            # Save parameters
            params_file = os.path.join(self.params_dir, 'parameters_sam.json')
            with open(params_file, 'w') as f:
                json.dump(param_dict, f, indent=4)
            
            # Run SAM (suppress output)
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_sam(self.tunnel_id, self.data_dir)
            
            # Compute mIoU
            results = compute_miou(self.tunnel_id, self.tunnel_dir, self.segment_count)
            miou = results['mIoU']
            
            runtime = time.time() - start_time
            
            # Track best
            if miou > self.best_score:
                self.best_score = miou
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"  [Eval {self.global_eval_index}] New best mIoU: {miou:.4f} "
                          f"(OA={results['OA']:.4f}, F1={results['F1']:.4f})")
            
            # Log trial
            self._log_trial(
                param_dict,
                results,
                runtime,
                False,
            )
            
            # Record history
            self.history.append({
                'eval': self.global_eval_index,
                'params': param_dict,
                'miou': miou,
                'oa': results['OA'],
                'f1': results['F1'],
            })
            
            if self.verbose and self.eval_count % 10 == 0:
                print(f"  [Eval {self.global_eval_index}] mIoU: {miou:.4f}, "
                      f"OA={results['OA']:.4f}, F1={results['F1']:.4f}")
            
            return -miou  # Negative for minimization
            
        except Exception as e:
            runtime = time.time() - start_time
            if self.verbose:
                print(f"  [Eval {self.global_eval_index}] Error: {e}")
            # Log failed trial
            self._log_trial(
                params_to_sam_json(params, self.param_names, self.fixed_params),
                None,
                runtime,
                False,
                error=str(e),
            )
            return 0.0  # Return worst score on error
    
    def _log_trial(
        self,
        params: Dict,
        results: Optional[Dict],
        runtime: float,
        cached: bool,
        error: Optional[str] = None,
    ):
        """Log trial to JSON file."""
        global_idx = self.global_eval_index
        trial_id = f"sam_{self.tunnel_id}_{global_idx:03d}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        log_data = {
            'schema_version': 'r4tun.sam.v1',
            'trial': {
                'trial_id': trial_id,
                'timestamp_utc': timestamp,
                'tunnel_id': self.tunnel_id,
                'assembly_type': self.agent_type,
            },
            'params': params,
        }
        
        if error:
            log_data['trace'] = {'warnings': [f"Error: {error}"]}
            log_data['bo'] = {
                'objective_name': 'miou',
                'objective_value': 0.0,
                'eval_index': global_idx,
                'runtime_sec': runtime,
                'is_feasible': False,
                'cached': cached,
            }
        else:
            log_data['outputs'] = {
                'metrics': {
                    'OA': results['OA'],
                    'F1': results['F1'],
                    'mIoU': results['mIoU'],
                },
                'iou_per_class': results['IoU_per_class'].tolist(),
                'classes': results['classes'].tolist(),
            }
            log_data['bo'] = {
                'objective_name': 'miou',
                'objective_value': float(results['mIoU']),
                'eval_index': global_idx,
                'runtime_sec': float(runtime),
                'is_feasible': True,
                'cached': cached,
            }
        
        # Save log file
        log_file = os.path.join(self.logs_dir, f"{trial_id}.json")
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def save_best_params(self) -> Optional[str]:
        """Save best parameters to JSON file."""
        if self.best_params is None:
            return None
        
        params_file = os.path.join(self.params_dir, 'parameters_sam.json')
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        return params_file


# =============================================================================
# Utilities
# =============================================================================

def find_max_trial_index(logs_dir: str, tunnel_id: str) -> int:
    """Find maximum trial index for eval offset."""
    pattern = os.path.join(logs_dir, f"sam_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    
    max_idx = -1
    for log_file in log_files:
        filename = os.path.basename(log_file)
        # Extract index from sam_{tunnel_id}_{idx:03d}.json
        try:
            idx_str = filename.split('_')[-1].replace('.json', '')
            idx = int(idx_str)
            max_idx = max(max_idx, idx)
        except (ValueError, IndexError):
            continue
    
    return max_idx


def load_best_from_logs(
    logs_dir: str,
    tunnel_id: str,
    agent_type: str = DEFAULT_AGENT_TYPE,
    fixed_params: Dict = None
) -> Optional[Tuple[List[float], float]]:
    """
    Load the best trial from existing logs to use as warm-start x0/y0.
    Falls back to current parameters_sam.json if no logs exist.
    
    Args:
        logs_dir: Directory containing BO log files
        tunnel_id: Tunnel identifier
        agent_type: Agent type
        fixed_params: Dict of fixed parameters (to exclude from warm-start vector)
    
    Returns:
        Tuple of (param_values_list, negative_miou) or None if no params found.
    """
    if fixed_params is None:
        fixed_params = {}
    
    pattern = os.path.join(logs_dir, f"sam_{tunnel_id}_*.json")
    log_files = glob.glob(pattern)
    
    best_miou = -1
    best_params = None
    
    # First, try to load from previous BO logs
    for log_file in log_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        if 'bo' not in data or 'objective_value' not in data['bo']:
            continue
        
        miou = data['bo']['objective_value']
        if miou > best_miou:
            best_miou = miou
            best_params = data.get('params', {})
    
    # If no logs found, try to load from current parameters file
    if best_params is None or best_miou <= 0:
        params_file = os.path.join(
            PROJECT_ROOT,
            'agents', agent_type, '3_segmentation',
            'parameters', tunnel_id, 'parameters_sam.json'
        )
        
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                best_params = json.load(f)
            # Use a conservative mIoU estimate (0.5) for parameters file
            best_miou = 0.5
            print(f"  No previous BO logs found, using current parameters_sam.json as warm-start")
    
    if best_params is None:
        return None
    
    # Build param list in dimension order, EXCLUDING fixed params
    _, param_names, _ = get_sam_dimensions(tunnel_id, agent_type)
    
    param_extractors = {
        'segment_width': lambda p: p.get('segment_width', 1200.0),
        'angle_deg': lambda p: p.get('angle_deg', 7.5),
        'k_mask_width': lambda p: p.get('k_mask_width', 625.0),
        'k_mask_height_pos': lambda p: p.get('k_mask_height_pos', 620.0),
        'k_mask_height_neg': lambda p: p.get('k_mask_height_neg', 460.0),
        'ab_mask_width': lambda p: p.get('ab_mask_width', 625.0),
        'ab_mask_height': lambda p: p.get('ab_mask_height', 1620.0),
        'padding': lambda p: p.get('padding', 150),
        'crop_margin': lambda p: p.get('crop_margin', 50),
        'min_quality_threshold': lambda p: p.get('min_quality_threshold', 0.3),
    }
    
    param_values = []
    for name in param_names:
        if name in fixed_params:
            continue
        extractor = param_extractors.get(name, lambda p: 0.0)
        param_values.append(extractor(best_params))
    
    return param_values, -best_miou  # negative for minimization


# =============================================================================
# Main Optimization
# =============================================================================

def run_sam_bo(
    tunnel_id: str,
    data_dir: str = 'data',
    n_calls: int = 60,
    n_initial_points: int = 10,
    verbose: bool = True,
    agent_type: str = DEFAULT_AGENT_TYPE,
) -> Dict:
    """
    Run SAM Bayesian Optimization.
    
    Args:
        tunnel_id: Tunnel identifier (e.g., '1-4')
        data_dir: Base data directory
        n_calls: Number of BO iterations
        n_initial_points: Number of initial random points
        verbose: Print progress
        agent_type: Agent type (defaults to script's directory name)
    
    Returns:
        Dictionary with best parameters and mIoU
    """
    print(f"\n{'='*70}")
    print(f"SAM BAYESIAN OPTIMIZATION - Tunnel {tunnel_id} ({agent_type})")
    print(f"{'='*70}")
    
    logs_dir = os.path.join(
        PROJECT_ROOT,
        'bo', agent_type, 'logs'
    )
    os.makedirs(logs_dir, exist_ok=True)
    
    # Determine eval offset from existing logs
    eval_offset = find_max_trial_index(logs_dir, tunnel_id)
    
    # Initialize objective
    objective = SamObjective(
        tunnel_id=tunnel_id,
        data_dir=data_dir,
        verbose=verbose,
        eval_offset=eval_offset,
        agent_type=agent_type,
    )
    
    print(f"\nSearch space: {len(objective.param_names)} parameters")
    print(f"N calls: {n_calls}, N initial: {n_initial_points}")
    print(f"Objective: mIoU (mean Intersection over Union)")
    print(f"Algorithm: forest_minimize (Random Forest surrogate)")
    
    x0 = None
    y0 = None
    warm_start = load_best_from_logs(logs_dir, tunnel_id, agent_type, objective.fixed_params)
    if warm_start is not None:
        x0_vals, y0_val = warm_start
        # Clamp warm-start values to current search space bounds
        clamped_x0 = []
        for i, (name, val) in enumerate(zip(objective.param_names, x0_vals)):
            dim = objective.dimensions[i]
            # Get bounds from dimension
            if hasattr(dim, 'low') and hasattr(dim, 'high'):
                clamped_val = max(dim.low, min(dim.high, val))
                clamped_x0.append(clamped_val)
            else:
                clamped_x0.append(val)
        
        x0 = [clamped_x0]
        y0 = [y0_val]
        source = "previous BO logs" if eval_offset > 0 else "current parameters_sam.json"
        print(f"\nWarm-starting from {source} (estimated mIoU={-y0_val:.4f}):")
        for name, val, clamped in zip(objective.param_names, x0_vals, clamped_x0):
            if abs(val - clamped) > 1e-6:
                print(f"  {name}: {val:.4f} -> {clamped:.4f} (clamped to bounds)")
            else:
                print(f"  {name}: {val:.4f}")
    
    # Run optimization
    print(f"\nStarting optimization...")
    result = forest_minimize(
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
    best_params = params_to_sam_json(result.x, objective.param_names, objective.fixed_params)
    best_miou = -result.fun  # Negate back
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best mIoU score: {best_miou:.4f}")
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
        'best_miou': best_miou,
        'best_params': best_params,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM Bayesian Optimization")
    parser.add_argument("tunnel_id", help="Tunnel identifier (e.g., 1-4)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--n-calls", type=int, default=60, help="Number of BO iterations")
    parser.add_argument("--n-initial", type=int, default=10, help="Number of initial random points")
    parser.add_argument("--agent-type", default=DEFAULT_AGENT_TYPE, help="Agent type")
    
    args = parser.parse_args()
    
    run_sam_bo(
        tunnel_id=args.tunnel_id,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        agent_type=args.agent_type,
    )

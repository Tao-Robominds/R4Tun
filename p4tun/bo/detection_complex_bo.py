"""
Complex Staggered Detection Bayesian Optimization

Optimizes complex_staggered detection parameters to match ground truth K positions.
Uses 4-1_detection_complex.py for T4/T5 patterns.

Usage:
    python -m p4tun.bo.detection_complex_bo --tunnel 5-1 --n-calls 50
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer
from skopt.callbacks import DeltaYStopper

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Complex Staggered Detection Search Space
# =============================================================================

COMPLEX_DETECTION_SEARCH_SPACE = {
    # Standard preprocessing (shared)
    'binary_threshold': Integer(80, 140, name='binary_threshold'),
    'dilation_kernel_size': Integer(2, 5, name='dilation_kernel_size'),
    'dilation_iterations': Integer(1, 4, name='dilation_iterations'),
    
    # Standard Hough oblique (initial detection)
    'hough_oblique_threshold': Integer(20, 60, name='hough_oblique_threshold'),
    'hough_oblique_min_length': Integer(40, 120, name='hough_oblique_min_length'),
    'hough_oblique_max_gap': Integer(30, 80, name='hough_oblique_max_gap'),
    'angle_positive_min': Real(4.0, 7.0, name='angle_positive_min'),
    'angle_positive_max': Real(8.0, 12.0, name='angle_positive_max'),
    
    # Complex_staggered specific: Hough re-detection
    'complex_hough_threshold': Integer(20, 50, name='complex_hough_threshold'),
    'complex_hough_min_length': Integer(30, 100, name='complex_hough_min_length'),
    'complex_hough_max_gap': Integer(50, 150, name='complex_hough_max_gap'),
    
    # Complex_staggered specific: Wider angle range
    'complex_angle_pos_min': Real(3.0, 6.0, name='complex_angle_pos_min'),
    'complex_angle_pos_max': Real(10.0, 15.0, name='complex_angle_pos_max'),
    'complex_angle_neg_min': Real(-15.0, -10.0, name='complex_angle_neg_min'),
    'complex_angle_neg_max': Real(-6.0, -3.0, name='complex_angle_neg_max'),
    
    # Complex_staggered specific: Line filtering
    'complex_min_y_span': Integer(20, 50, name='complex_min_y_span'),
    'complex_min_x_span': Integer(20, 50, name='complex_min_x_span'),
    
    # Complex_staggered specific: Clustering
    'complex_eps_primary': Real(0.02, 0.10, name='complex_eps_primary'),
    'complex_eps_secondary': Real(0.05, 0.15, name='complex_eps_secondary'),
    'complex_subdivision_threshold': Real(1.0, 2.5, name='complex_subdivision_threshold'),
    'complex_max_subdivisions': Integer(2, 5, name='complex_max_subdivisions'),
    
    # Complex_staggered specific: Confidence
    'complex_conf_midpoint': Real(0.5, 0.9, name='complex_conf_midpoint'),
    'complex_conf_intersection': Real(0.7, 1.0, name='complex_conf_intersection'),
    'complex_conf_midpoint_final': Real(0.4, 0.8, name='complex_conf_midpoint_final'),
}


def get_complex_detection_dimensions():
    """Get search space dimensions and names."""
    dimensions = list(COMPLEX_DETECTION_SEARCH_SPACE.values())
    names = list(COMPLEX_DETECTION_SEARCH_SPACE.keys())
    return dimensions, names


def params_to_complex_detection_json(params: List, names: List[str], tunnel_id: str) -> Dict:
    """Convert BO parameters to complex_staggered detection JSON structure."""
    param_dict = dict(zip(names, params))
    
    # Build eps_candidates from primary and secondary eps
    eps_primary = float(param_dict.get('complex_eps_primary', 0.05))
    eps_secondary = float(param_dict.get('complex_eps_secondary', 0.10))
    eps_candidates = [eps_primary, eps_secondary, eps_secondary * 1.5, eps_secondary * 2.0, 0.15]
    eps_candidates = sorted(list(set([round(e, 2) for e in eps_candidates])))
    
    # Determine min_clusters based on tunnel
    ring_count = 9 if tunnel_id == '4-1' else 7
    min_clusters = max(3, ring_count // 2)
    
    return {
        'preprocessing': {
            'binary_threshold': int(param_dict.get('binary_threshold', 103)),
            'dilation_kernel_size': int(param_dict.get('dilation_kernel_size', 2)),
            'dilation_iterations': int(param_dict.get('dilation_iterations', 3)),
            'use_morphological_closing': True,
            'use_depth_gradients': True,
        },
        'hough_oblique': {
            'threshold': int(param_dict.get('hough_oblique_threshold', 45)),
            'min_length': int(param_dict.get('hough_oblique_min_length', 107)),
            'max_gap': int(param_dict.get('hough_oblique_max_gap', 80)),
            'angle_positive_min': float(param_dict.get('angle_positive_min', 6.5)),
            'angle_positive_max': float(param_dict.get('angle_positive_max', 9.4)),
            'angle_negative_min': -float(param_dict.get('angle_positive_max', 9.4)),
            'angle_negative_max': -float(param_dict.get('angle_positive_min', 6.5)),
        },
        'hough_horizontal': {
            'threshold': 30,
            'min_length': 129,
            'max_gap': 5,
            'angle_tolerance': 1,
        },
        'hough_vertical': {
            'threshold': 447,
        },
        'line_processing': {
            'merge_distance_threshold': 8,
            'merge_close_threshold': 12,
            'oblique_min_y_span': 50,
            'oblique_min_x_span': 50,
            'use_line_clustering': True,
            'use_vertical_clustering': True,
            'use_horizontal_constraint': True,
        },
        'physical_constants': {
            'resolution': 0.005,
            'k_height_mm': 1079.92,
            'ab_height_mm': 3239.77,
        },
        'complex_staggered': {
            'hough_re_detect': {
                'threshold': int(param_dict.get('complex_hough_threshold', 30)),
                'min_length': int(param_dict.get('complex_hough_min_length', 50)),
                'max_gap': int(param_dict.get('complex_hough_max_gap', 100)),
            },
            'angle_range': {
                'positive_min': float(param_dict.get('complex_angle_pos_min', 4.0)),
                'positive_max': float(param_dict.get('complex_angle_pos_max', 12.0)),
                'negative_min': float(param_dict.get('complex_angle_neg_min', -12.0)),
                'negative_max': float(param_dict.get('complex_angle_neg_max', -4.0)),
            },
            'line_filtering': {
                'min_y_span': int(param_dict.get('complex_min_y_span', 30)),
                'min_x_span': int(param_dict.get('complex_min_x_span', 30)),
            },
            'clustering': {
                'eps_candidates': eps_candidates,
                'min_clusters': min_clusters,
                'subdivision_threshold': float(param_dict.get('complex_subdivision_threshold', 1.5)),
                'max_subdivisions': int(param_dict.get('complex_max_subdivisions', 4)),
            },
            'confidence': {
                'subdivision_base': 0.5,
                'subdivision_factor': 0.05,
                'cluster_base': 0.5,
                'cluster_factor': 0.1,
                'midpoint': float(param_dict.get('complex_conf_midpoint', 0.7)),
                'final_intersection': float(param_dict.get('complex_conf_intersection', 0.9)),
                'final_midpoint': float(param_dict.get('complex_conf_midpoint_final', 0.6)),
            },
        },
    }


# =============================================================================
# Complex Detection Objective Function
# =============================================================================

class ComplexDetectionObjective:
    """
    Objective function for complex_staggered detection optimization.
    Evaluates against ground truth K positions.
    """
    
    def __init__(
        self,
        tunnel_id: str,
        data_dir: str = 'data',
        verbose: bool = True,
    ):
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.verbose = verbose
        
        self.tunnel_dir = os.path.join(data_dir, tunnel_id)
        self.params_dir = os.path.join(PROJECT_ROOT, 'p4tun', 'parameters', tunnel_id)
        
        # Load ground truth
        self.gt_positions = self._load_gt_positions()
        self.ring_count = len(self.gt_positions)
        
        # Get search space
        self.dimensions, self.param_names = get_complex_detection_dimensions()
        
        # Detection script path
        self.detection_script = os.path.join(PROJECT_ROOT, 'p4tun', '4-1_detection_complex.py')
        
        # Tracking
        self.eval_count = 0
        self.best_score = -np.inf
        self.best_params = None
        self.history = []
        
        if verbose:
            print(f"Loaded GT with {self.ring_count} K positions")
            print(f"GT Y range: {self.gt_positions['Y'].min():.1f} - {self.gt_positions['Y'].max():.1f}")
            print(f"Search space: {len(self.param_names)} parameters")
    
    def _load_gt_positions(self) -> pd.DataFrame:
        """Load ground truth K positions."""
        gt_path = os.path.join(self.tunnel_dir, 'detected_gt.csv')
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth not found: {gt_path}")
        
        df = pd.read_csv(gt_path)
        return df.sort_values('X').reset_index(drop=True)
    
    def _save_temp_params(self, config: Dict):
        """Save parameters to JSON file for detection script."""
        os.makedirs(self.params_dir, exist_ok=True)
        filepath = os.path.join(self.params_dir, 'parameters_detection.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    
    def _run_detection_script(self) -> pd.DataFrame:
        """Run the complex detection script and read results."""
        venv_python = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')
        cmd = [venv_python, self.detection_script, self.tunnel_id, '--data-dir', self.data_dir]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            if self.verbose:
                print(f"  Detection script error: {result.stderr[:200]}")
            return pd.DataFrame(columns=['Type', 'X', 'Y'])
        
        # Read detected.csv
        detected_path = os.path.join(self.tunnel_dir, 'detected.csv')
        if os.path.exists(detected_path):
            return pd.read_csv(detected_path)
        else:
            return pd.DataFrame(columns=['Type', 'X', 'Y'])
    
    def _calculate_score(self, detected: pd.DataFrame) -> float:
        """
        Calculate score based on how well detected positions match GT.
        Higher score is better.
        """
        if len(detected) == 0:
            return 0.0
        
        gt = self.gt_positions
        detected_sorted = detected.sort_values('X').reset_index(drop=True)
        gt_sorted = gt.sort_values('X').reset_index(drop=True)
        
        n_gt = len(gt_sorted)
        n_det = len(detected_sorted)
        
        # Penalize wrong count
        count_penalty = abs(n_det - n_gt) * 50
        
        # Calculate position errors
        if n_det >= n_gt:
            total_error = 0
            for i in range(n_gt):
                gt_x, gt_y = gt_sorted.iloc[i]['X'], gt_sorted.iloc[i]['Y']
                distances = np.sqrt(
                    (detected_sorted['X'] - gt_x)**2 + 
                    (detected_sorted['Y'] - gt_y)**2
                )
                total_error += distances.min()
        else:
            total_error = 0
            for i in range(n_det):
                det_x, det_y = detected_sorted.iloc[i]['X'], detected_sorted.iloc[i]['Y']
                distances = np.sqrt(
                    (gt_sorted['X'] - det_x)**2 + 
                    (gt_sorted['Y'] - det_y)**2
                )
                total_error += distances.min()
            total_error += (n_gt - n_det) * 100
        
        avg_error = total_error / max(n_gt, n_det)
        total_error_with_penalty = avg_error + count_penalty
        
        # Convert to score (higher is better)
        max_error = 1000
        score = max(0, 1 - total_error_with_penalty / max_error)
        
        return score
    
    def __call__(self, params: List) -> float:
        """Evaluate detection parameters against ground truth."""
        self.eval_count += 1
        
        try:
            # Convert params to detection config
            detection_config = params_to_complex_detection_json(params, self.param_names, self.tunnel_id)
            
            # Save parameters
            self._save_temp_params(detection_config)
            
            # Run detection
            detected_positions = self._run_detection_script()
            
            # Calculate score
            score = self._calculate_score(detected_positions)
            
            # Track best
            param_dict = dict(zip(self.param_names, params))
            if score > self.best_score:
                self.best_score = score
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"  [Eval {self.eval_count}] New best: {score:.4f} (n={len(detected_positions)})")
            
            # Record history
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'score': score,
                'n_detected': len(detected_positions),
            })
            
            if self.verbose and self.eval_count % 10 == 0:
                print(f"  [Eval {self.eval_count}] Score: {score:.4f}, Detected: {len(detected_positions)}")
            
            return -score  # Negative for minimization
            
        except Exception as e:
            if self.verbose:
                print(f"  [Eval {self.eval_count}] Error: {e}")
            return 0.0
    
    def save_best_params(self):
        """Save best parameters to JSON file."""
        if self.best_params is None:
            return None
        
        os.makedirs(self.params_dir, exist_ok=True)
        
        # Convert to full config
        config = params_to_complex_detection_json(
            [self.best_params[n] for n in self.param_names],
            self.param_names,
            self.tunnel_id
        )
        
        filepath = os.path.join(self.params_dir, 'parameters_detection.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        return filepath


# =============================================================================
# Main Optimization
# =============================================================================

def run_complex_detection_bo(
    tunnel_id: str,
    data_dir: str = 'data',
    n_calls: int = 50,
    n_initial: int = 10,
    verbose: bool = True,
    optimizer: str = 'gp',
) -> Dict:
    """Run Bayesian Optimization for complex_staggered detection parameters."""
    
    print(f"\n{'='*70}")
    print(f"COMPLEX STAGGERED DETECTION BO - Tunnel {tunnel_id}")
    print(f"{'='*70}")
    
    # Initialize objective
    objective = ComplexDetectionObjective(
        tunnel_id=tunnel_id,
        data_dir=data_dir,
        verbose=verbose,
    )
    
    print(f"\nSearch space: {len(objective.param_names)} parameters")
    print(f"N calls: {n_calls}, N initial: {n_initial}")
    
    # Select optimizer
    minimize_func = gp_minimize if optimizer == 'gp' else forest_minimize
    
    # Run optimization
    print(f"\nStarting optimization...")
    callbacks = []
    
    result = minimize_func(
        objective,
        objective.dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=42,
        verbose=False,
        callback=callbacks,
    )
    
    # Results
    best_params = dict(zip(objective.param_names, result.x))
    best_score = -result.fun
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best score: {best_score:.4f}")
    print(f"\nBest parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    
    # Save best parameters
    filepath = objective.save_best_params()
    if filepath:
        print(f"\nSaved parameters to: {filepath}")
    
    # Save history (convert numpy types to native Python types)
    history_file = os.path.join(PROJECT_ROOT, 'p4tun', 'bo', 'results', 
                                f'{tunnel_id}_complex_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    history_data = {
        'tunnel_id': tunnel_id,
        'best_score': float(best_score),
        'best_params': convert_types(best_params),
        'history': convert_types(objective.history),
    }
    
    with open(history_file, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved history to: {history_file}")
    
    return {
        'best_score': best_score,
        'best_params': best_params,
        'history': objective.history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complex Staggered Detection BO")
    parser.add_argument("--tunnel", required=True, help="Tunnel ID (e.g., 5-1)")
    parser.add_argument("--n-calls", type=int, default=50, help="Number of BO iterations")
    parser.add_argument("--n-initial", type=int, default=10, help="Number of initial random points")
    parser.add_argument("--optimizer", choices=['gp', 'forest'], default='gp', help="Optimizer type")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    args = parser.parse_args()
    
    run_complex_detection_bo(
        tunnel_id=args.tunnel,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        optimizer=args.optimizer,
    )

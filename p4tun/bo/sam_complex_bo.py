"""
Complex Staggered SAM Bayesian Optimization

Optimizes SAM parameters for complex_staggered patterns (4-1, 5-1).
Uses optimized detection, then runs SAM and evaluates with mIoU.

Usage:
    python -m p4tun.bo.sam_complex_bo --tunnel 5-1 --n-calls 30
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
from skopt.space import Real, Integer, Categorical
from skopt.callbacks import DeltaYStopper

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Complex Staggered SAM Search Space
# =============================================================================

COMPLEX_SAM_SEARCH_SPACE = {
    # Segment geometry
    'segment_width': Real(1150.0, 1350.0, name='segment_width'),
    'k_height': Real(950.0, 1200.0, name='k_height'),
    'ab_height': Real(3100.0, 3500.0, name='ab_height'),
    'angle_deg': Real(6.0, 9.0, name='angle_deg'),
    
    # Template mask - K block
    'k_mask_width': Real(550.0, 750.0, name='k_mask_width'),
    'k_mask_height_pos': Real(550.0, 750.0, name='k_mask_height_pos'),
    'k_mask_height_neg': Real(400.0, 650.0, name='k_mask_height_neg'),
    
    # Template mask - A/B blocks
    'ab_mask_width': Real(550.0, 700.0, name='ab_mask_width'),
    'ab_mask_height': Real(1500.0, 1750.0, name='ab_mask_height'),
    
    # Complex_staggered specific: Template sizing factors
    'k_block_width_factor': Real(0.8, 1.2, name='k_block_width_factor'),
    'k_block_height_factor': Real(0.8, 1.2, name='k_block_height_factor'),
    'ab_block_width_factor': Real(0.8, 1.2, name='ab_block_width_factor'),
    'ab_block_height_factor': Real(0.8, 1.2, name='ab_block_height_factor'),
    
    # Complex_staggered specific: Prompt density
    'k_block_points': Categorical(['standard', 'dense', 'sparse'], name='k_block_points'),
    'ab_block_points': Categorical(['standard', 'dense', 'sparse'], name='ab_block_points'),
}


def get_complex_sam_dimensions():
    """Get search space dimensions and names."""
    dimensions = list(COMPLEX_SAM_SEARCH_SPACE.values())
    names = list(COMPLEX_SAM_SEARCH_SPACE.keys())
    return dimensions, names


def params_to_complex_sam_json(params: List, names: List[str], tunnel_id: str) -> Dict:
    """Convert BO parameters to complex_staggered SAM JSON structure."""
    param_dict = dict(zip(names, params))
    
    return {
        'description': f'SAM complex_staggered BO parameters for Tunnel {tunnel_id}',
        'segment_geometry': {
            'segment_width': float(param_dict.get('segment_width', 1300.0)),
            'k_height': float(param_dict.get('k_height', 1079.92)),
            'ab_height': float(param_dict.get('ab_height', 3239.77)),
            'angle_deg': float(param_dict.get('angle_deg', 7.52)),
        },
        'image': {
            'resolution': 0.005,
        },
        'pattern_aware': {
            'use_quality_weighting': True,
            'min_quality_threshold': 0.3,
        },
        'prompt_points': {
            'template_mask': {
                'k_block': {
                    'width': float(param_dict.get('k_mask_width', 625.0)),
                    'height_pos': float(param_dict.get('k_mask_height_pos', 619.16)),
                    'height_neg': float(param_dict.get('k_mask_height_neg', 460.77)),
                },
                'b1_block': {
                    'width': float(param_dict.get('ab_mask_width', 700.0)),
                    'height_top': 1500.0,
                    'height_bottom_pos': 1540.69,
                    'height_bottom_neg': 1699.08,
                },
                'b2_block': {
                    'width': float(param_dict.get('ab_mask_width', 700.0)),
                    'height_top_pos': 1540.69,
                    'height_top_neg': 1699.08,
                    'height_bottom': 1500.0,
                },
                'a_blocks': {
                    'width': float(param_dict.get('ab_mask_width', 700.0)),
                    'height': float(param_dict.get('ab_mask_height', 1619.89)),
                },
            },
        },
        'complex_staggered': {
            '_note': 'Parameters specific to complex_staggered pattern processing',
            'template_sizing': {
                'k_block_width_factor': float(param_dict.get('k_block_width_factor', 1.0)),
                'k_block_height_factor': float(param_dict.get('k_block_height_factor', 1.0)),
                'ab_block_width_factor': float(param_dict.get('ab_block_width_factor', 1.0)),
                'ab_block_height_factor': float(param_dict.get('ab_block_height_factor', 1.0)),
            },
            'prompt_density': {
                'k_block_points': param_dict.get('k_block_points', 'standard'),
                'ab_block_points': param_dict.get('ab_block_points', 'standard'),
            },
        },
    }


# =============================================================================
# Complex SAM Objective Function
# =============================================================================

class ComplexSAMObjective:
    """
    Objective function for complex_staggered SAM optimization.
    Runs detection (using optimized params) → SAM → evaluation (mIoU).
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
        
        # Get search space
        self.dimensions, self.param_names = get_complex_sam_dimensions()
        
        # Script paths
        self.detection_script = os.path.join(PROJECT_ROOT, 'p4tun', '4-1_detection_complex.py')
        self.sam_script = os.path.join(PROJECT_ROOT, 'p4tun', '4-2_sam_complex.py')
        self.eval_script = os.path.join(PROJECT_ROOT, 'p4tun', 'evaluation.py')
        
        # Tracking
        self.eval_count = 0
        self.best_score = -np.inf
        self.best_params = None
        self.history = []
        
        if verbose:
            print(f"Search space: {len(self.param_names)} parameters")
            print(f"Detection script: {self.detection_script}")
            print(f"SAM script: {self.sam_script}")
    
    def _save_temp_params(self, config: Dict):
        """Save parameters to JSON file for SAM script."""
        os.makedirs(self.params_dir, exist_ok=True)
        filepath = os.path.join(self.params_dir, 'parameters_sam.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
    
    def _run_detection(self):
        """Run complex detection (uses optimized parameters)."""
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
                print(f"  Detection error: {result.stderr[:200]}")
            return False
        return True
    
    def _run_sam(self):
        """Run SAM wraparound script."""
        venv_python = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')
        cmd = [venv_python, self.sam_script, self.tunnel_id]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            if self.verbose:
                print(f"  SAM error: {result.stderr[:200]}")
            return False
        return True
    
    def _evaluate(self) -> Dict:
        """Run evaluation script and extract mIoU."""
        venv_python = os.path.join(PROJECT_ROOT, 'venv', 'bin', 'python3')
        cmd = [venv_python, self.eval_script, self.tunnel_id, '--data-dir', self.data_dir]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            if self.verbose:
                print(f"  Evaluation error: {result.stderr[:200]}")
            return {'mIoU': 0.0, 'OA': 0.0, 'F1': 0.0}
        
        # Parse output for metrics
        metrics = {'mIoU': 0.0, 'OA': 0.0, 'F1': 0.0}
        for line in result.stdout.split('\n'):
            if 'mIoU:' in line or 'Mean IoU:' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'mIoU' in part or 'IoU' in part:
                            if i + 1 < len(parts):
                                metrics['mIoU'] = float(parts[i + 1].rstrip(','))
                except:
                    pass
            if 'OA:' in line or 'Overall Accuracy:' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'OA' in part or 'Accuracy' in part:
                            if i + 1 < len(parts):
                                metrics['OA'] = float(parts[i + 1].rstrip(','))
                except:
                    pass
            if 'F1:' in line or 'F1 Score:' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'F1' in part:
                            if i + 1 < len(parts):
                                metrics['F1'] = float(parts[i + 1].rstrip(','))
                except:
                    pass
        
        # Try reading from evaluation output file
        eval_file = os.path.join(self.tunnel_dir, 'evaluation', 'performance.md')
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        # Parse markdown table format: | Metric | Value |
                        if 'mIoU' in line or 'Mean IoU' in line:
                            try:
                                parts = [p.strip() for p in line.split('|')]
                                for p in parts:
                                    try:
                                        val = float(p)
                                        metrics['mIoU'] = val
                                        break
                                    except:
                                        pass
                            except:
                                pass
                        if 'Overall Accuracy' in line:
                            try:
                                parts = [p.strip() for p in line.split('|')]
                                for p in parts:
                                    try:
                                        val = float(p)
                                        metrics['OA'] = val
                                        break
                                    except:
                                        pass
                            except:
                                pass
                        if 'F1 Score' in line:
                            try:
                                parts = [p.strip() for p in line.split('|')]
                                for p in parts:
                                    try:
                                        val = float(p)
                                        metrics['F1'] = val
                                        break
                                    except:
                                        pass
                            except:
                                pass
            except:
                pass
        
        return metrics
    
    def __call__(self, params: List) -> float:
        """Evaluate SAM parameters against ground truth mIoU."""
        self.eval_count += 1
        
        try:
            # Convert params to SAM config
            sam_config = params_to_complex_sam_json(params, self.param_names, self.tunnel_id)
            
            # Save parameters
            self._save_temp_params(sam_config)
            
            # Run detection (uses optimized detection params)
            if not self._run_detection():
                return 0.0
            
            # Run SAM
            if not self._run_sam():
                return 0.0
            
            # Evaluate
            metrics = self._evaluate()
            score = metrics.get('mIoU', 0.0)
            
            # Track best
            param_dict = dict(zip(self.param_names, params))
            if score > self.best_score:
                self.best_score = score
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"  [Eval {self.eval_count}] New best mIoU: {score:.4f}")
            
            # Record history
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'score': score,
                'metrics': metrics,
            })
            
            if self.verbose and self.eval_count % 5 == 0:
                print(f"  [Eval {self.eval_count}] mIoU: {score:.4f}, OA: {metrics.get('OA', 0):.4f}, F1: {metrics.get('F1', 0):.4f}")
            
            return -score  # Negative for minimization
            
        except Exception as e:
            if self.verbose:
                print(f"  [Eval {self.eval_count}] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def save_best_params(self):
        """Save best parameters to JSON file."""
        if self.best_params is None:
            return None
        
        os.makedirs(self.params_dir, exist_ok=True)
        
        # Convert to full config
        config = params_to_complex_sam_json(
            [self.best_params[n] for n in self.param_names],
            self.param_names,
            self.tunnel_id
        )
        
        filepath = os.path.join(self.params_dir, 'parameters_sam.json')
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        return filepath


# =============================================================================
# Main Optimization
# =============================================================================

def run_complex_sam_bo(
    tunnel_id: str,
    data_dir: str = 'data',
    n_calls: int = 30,
    n_initial: int = 5,
    verbose: bool = True,
    optimizer: str = 'gp',
) -> Dict:
    """Run Bayesian Optimization for complex_staggered SAM parameters."""
    
    print(f"\n{'='*70}")
    print(f"COMPLEX STAGGERED SAM BO - Tunnel {tunnel_id}")
    print(f"{'='*70}")
    
    # Initialize objective
    objective = ComplexSAMObjective(
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
    print(f"Best mIoU: {best_score:.4f}")
    print(f"\nBest parameters:")
    for name, value in best_params.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    # Save best parameters
    filepath = objective.save_best_params()
    if filepath:
        print(f"\nSaved parameters to: {filepath}")
    
    # Save history (convert numpy types to native Python types)
    history_file = os.path.join(PROJECT_ROOT, 'p4tun', 'bo', 'results', 
                                f'{tunnel_id}_complex_sam_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    
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
    parser = argparse.ArgumentParser(description="Complex Staggered SAM BO")
    parser.add_argument("--tunnel", required=True, help="Tunnel ID (e.g., 5-1)")
    parser.add_argument("--n-calls", type=int, default=30, help="Number of BO iterations")
    parser.add_argument("--n-initial", type=int, default=5, help="Number of initial random points")
    parser.add_argument("--optimizer", choices=['gp', 'forest'], default='gp', help="Optimizer type")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    args = parser.parse_args()
    
    run_complex_sam_bo(
        tunnel_id=args.tunnel,
        data_dir=args.data_dir,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        optimizer=args.optimizer,
    )

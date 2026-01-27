"""
Bayesian Optimization for SAM Wraparound Script

Runs BO on the 4-2_sam_wraparound.py script with GT segment positions.
This version uses ground truth detection to isolate SAM parameter tuning.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

from skopt import gp_minimize
from skopt.callbacks import DeltaYStopper, CheckpointSaver
from skopt.space import Real, Integer


# Simplified SAM search space focused on segment geometry and template mask
SAM_WRAPAROUND_SPACE = {
    # Segment geometry - critical for wraparound
    'segment_width': Real(1150.0, 1350.0, name='segment_width'),
    'k_height': Real(950.0, 1200.0, name='k_height'),
    'ab_height': Real(3100.0, 3500.0, name='ab_height'),
    'angle_deg': Real(6.0, 9.0, name='angle_deg'),
    
    # Template mask dimensions - K block
    'k_mask_width': Real(550.0, 750.0, name='k_mask_width'),
    'k_mask_height_pos': Real(550.0, 750.0, name='k_mask_height_pos'),
    'k_mask_height_neg': Real(400.0, 650.0, name='k_mask_height_neg'),
    
    # Template mask dimensions - A/B blocks
    'ab_mask_width': Real(550.0, 700.0, name='ab_mask_width'),
    'ab_mask_height': Real(1500.0, 1750.0, name='ab_mask_height'),
}


class SAMWraparoundObjective:
    """Objective function for SAM wraparound optimization."""
    
    def __init__(
        self,
        tunnel_id: str,
        data_dir: str = 'data',
        verbose: bool = True,
        timeout: int = 300,
    ):
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.verbose = verbose
        self.timeout = timeout
        
        self.eval_count = 0
        self.best_score = -np.inf
        self.best_params = None
        self.history = []
        
        # Script paths
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sam_script = os.path.join(self.script_dir, '4-2_sam_wraparound.py')
        self.eval_script = os.path.join(self.script_dir, 'evaluation.py')
        self.segment_anything_path = os.path.join(self.script_dir, 'segment-anything')
        self.project_root = os.path.dirname(self.script_dir)
        
        # Verify scripts exist
        if not os.path.exists(self.sam_script):
            raise FileNotFoundError(f"SAM wraparound script not found: {self.sam_script}")
        if not os.path.exists(self.eval_script):
            raise FileNotFoundError(f"Evaluation script not found: {self.eval_script}")
    
    def __call__(self, params: List) -> float:
        """Evaluate a parameter set."""
        self.eval_count += 1
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluation {self.eval_count}")
            print(f"{'='*60}")
        
        try:
            # Convert to dict
            param_names = list(SAM_WRAPAROUND_SPACE.keys())
            param_dict = dict(zip(param_names, params))
            
            if self.verbose:
                for name, val in param_dict.items():
                    print(f"  {name}: {val:.2f}")
            
            # Update the SAM wraparound script inline (via environment or config)
            # For now, we update the parameters JSON file
            self._update_sam_params(param_dict)
            
            # Run SAM wraparound
            self._run_sam_wraparound()
            
            # Evaluate
            metrics = self._evaluate()
            score = metrics.get('mIoU', 0.0)
            
            # Track best
            if score > self.best_score:
                self.best_score = score
                self.best_params = param_dict.copy()
                if self.verbose:
                    print(f"*** New best mIoU: {score:.4f} ***")
            
            # Record history
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'score': score,
                'metrics': metrics
            })
            
            if self.verbose:
                print(f"mIoU: {score:.4f}, OA: {metrics.get('OA', 0):.4f}, F1: {metrics.get('F1', 0):.4f}")
            
            return -score  # Negative for minimization
            
        except Exception as e:
            print(f"Error in evaluation {self.eval_count}: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _update_sam_params(self, param_dict: Dict):
        """Update SAM parameters JSON file."""
        params_dir = os.path.join(self.script_dir, 'parameters', self.tunnel_id)
        os.makedirs(params_dir, exist_ok=True)
        
        sam_params = {
            'description': f'SAM wraparound BO parameters for Tunnel {self.tunnel_id}',
            'segment_geometry': {
                'segment_width': float(param_dict['segment_width']),
                'k_height': float(param_dict['k_height']),
                'ab_height': float(param_dict['ab_height']),
                'angle_deg': float(param_dict['angle_deg']),
            },
            'image': {
                'resolution': 0.005
            },
            'pattern_aware': {
                'use_quality_weighting': True,
                'min_quality_threshold': 0.3
            },
            'processing': {
                'padding': 150,
                'crop_margin': 50,
                'mask_eps': 0.001,
                'y_bounds': [4200, 13100]
            },
            'prompt_points': {
                'k_block': {
                    'outer_ring': 700,
                    'middle_ring': 500,
                    'inner_ring': 348.16,
                    'center_ring': 325,
                },
                'ab_blocks': {
                    'outer_ring': 700,
                    'middle_ring': 511.06,
                    'inner_ring': 500,
                    'center_ring': 325,
                    'fine_spacing': 250,
                    'ultra_fine': 162.5,
                },
                'template_mask': {
                    'k_block': {
                        'width': float(param_dict['k_mask_width']),
                        'height_pos': float(param_dict['k_mask_height_pos']),
                        'height_neg': float(param_dict['k_mask_height_neg']),
                    },
                    'b1_block': {
                        'width': float(param_dict['ab_mask_width']),
                        'height_top': float(param_dict['ab_mask_height']),
                        'height_bottom_pos': 1540.69,
                        'height_bottom_neg': 1699.08,
                    },
                    'b2_block': {
                        'width': float(param_dict['ab_mask_width']),
                        'height_top_pos': 1540.69,
                        'height_top_neg': 1699.08,
                        'height_bottom': float(param_dict['ab_mask_height']),
                    },
                    'a_blocks': {
                        'width': float(param_dict['ab_mask_width']),
                        'height': float(param_dict['ab_mask_height']),
                    }
                }
            }
        }
        
        filepath = os.path.join(params_dir, 'parameters_sam.json')
        with open(filepath, 'w') as f:
            json.dump(sam_params, f, indent=4)
    
    def _run_sam_wraparound(self):
        """Run SAM wraparound segmentation."""
        if self.verbose:
            print("Running SAM wraparound...")
        
        cmd = [sys.executable, self.sam_script, self.tunnel_id]
        
        env = os.environ.copy()
        pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{self.segment_anything_path}:{pythonpath}" if pythonpath else self.segment_anything_path
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.project_root,
            env=env
        )
        
        if result.returncode != 0:
            if self.verbose:
                print(f"SAM stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"SAM wraparound failed")
    
    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation and parse metrics."""
        if self.verbose:
            print("Running evaluation...")
        
        cmd = [sys.executable, self.eval_script, self.tunnel_id, '--data-dir', self.data_dir]
        
        env = os.environ.copy()
        pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f"{self.segment_anything_path}:{pythonpath}" if pythonpath else self.segment_anything_path
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=self.project_root,
            env=env
        )
        
        # Parse metrics from output
        metrics = {}
        for line in result.stdout.split('\n'):
            if 'OA' in line and 'F1' in line and 'mIoU' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'OA' and i + 1 < len(parts):
                        try:
                            metrics['OA'] = float(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'F1' and i + 1 < len(parts):
                        try:
                            metrics['F1'] = float(parts[i + 1])
                        except ValueError:
                            pass
                    elif part == 'mIoU' and i + 1 < len(parts):
                        try:
                            metrics['mIoU'] = float(parts[i + 1])
                        except ValueError:
                            pass
        
        return metrics
    
    def save_history(self, filepath: str):
        """Save optimization history."""
        with open(filepath, 'w') as f:
            json.dump({
                'tunnel_id': self.tunnel_id,
                'best_score': self.best_score,
                'best_params': self.best_params,
                'history': self.history
            }, f, indent=2, default=float)


def run_bo(
    tunnel_id: str,
    n_calls: int = 30,
    n_initial: int = 10,
    output_dir: str = 'p4tun/bo/results',
    verbose: bool = True
) -> Dict:
    """Run Bayesian Optimization for SAM wraparound."""
    
    print(f"\n{'='*70}")
    print(f"SAM WRAPAROUND BAYESIAN OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Tunnel: {tunnel_id}")
    print(f"N calls: {n_calls}")
    print(f"N initial: {n_initial}")
    print(f"{'='*70}\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get search space
    dimensions = list(SAM_WRAPAROUND_SPACE.values())
    param_names = list(SAM_WRAPAROUND_SPACE.keys())
    
    # Create objective
    objective = SAMWraparoundObjective(
        tunnel_id=tunnel_id,
        verbose=verbose
    )
    
    # Setup callbacks
    checkpoint_path = os.path.join(output_dir, f'{tunnel_id}_sam_wraparound_checkpoint.pkl')
    callbacks = [
        DeltaYStopper(delta=0.005, n_best=5),
        CheckpointSaver(checkpoint_path, compress=9)
    ]
    
    # Run optimization
    result = gp_minimize(
        objective,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=42,
        verbose=verbose,
        callback=callbacks
    )
    
    # Get best results
    best_params = dict(zip(param_names, result.x))
    best_score = -result.fun
    
    # Print results
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best mIoU: {best_score:.4f}")
    print(f"\nBest parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'tunnel_id': tunnel_id,
        'best_score': best_score,
        'best_params': best_params,
        'n_calls': n_calls,
        'all_scores': [-v for v in result.func_vals],
        'convergence': _get_convergence(result),
        'timestamp': timestamp
    }
    
    # Save JSON
    json_path = os.path.join(output_dir, f'{tunnel_id}_sam_wraparound_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {json_path}")
    
    # Save history
    history_path = os.path.join(output_dir, f'{tunnel_id}_sam_wraparound_{timestamp}_history.json')
    objective.save_history(history_path)
    
    # Plot convergence
    _plot_convergence(results, output_dir, tunnel_id, timestamp)
    
    return results


def _get_convergence(result) -> List[float]:
    """Get convergence curve."""
    scores = [-v for v in result.func_vals]
    best_so_far = []
    current_best = -np.inf
    for score in scores:
        if score > current_best:
            current_best = score
        best_so_far.append(current_best)
    return best_so_far


def _plot_convergence(results: Dict, output_dir: str, tunnel_id: str, timestamp: str):
    """Plot convergence curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        convergence = results.get('convergence', [])
        if not convergence:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(convergence) + 1), convergence, 'b-', linewidth=2)
        plt.xlabel('Evaluation')
        plt.ylabel('Best mIoU')
        plt.title(f'SAM Wraparound BO Convergence: {tunnel_id}')
        plt.grid(True, alpha=0.3)
        
        # Mark best point
        best_idx = np.argmax(convergence)
        plt.scatter([best_idx + 1], [convergence[best_idx]], color='red', s=100, zorder=5)
        plt.annotate(f'Best: {convergence[best_idx]:.4f}',
                    xy=(best_idx + 1, convergence[best_idx]),
                    xytext=(10, 10), textcoords='offset points')
        
        plot_path = os.path.join(output_dir, f'{tunnel_id}_sam_wraparound_{timestamp}_convergence.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Convergence plot saved to {plot_path}")
        
    except ImportError:
        print("matplotlib not available, skipping convergence plot")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SAM Wraparound BO')
    parser.add_argument('tunnel_id', help='Tunnel ID (e.g., 4-1)')
    parser.add_argument('--n-calls', type=int, default=30, help='Number of evaluations')
    parser.add_argument('--n-initial', type=int, default=10, help='Initial random points')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    
    args = parser.parse_args()
    
    run_bo(
        tunnel_id=args.tunnel_id,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        verbose=not args.quiet
    )

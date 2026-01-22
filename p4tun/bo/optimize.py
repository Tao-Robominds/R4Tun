"""
Bayesian Optimization Runner for P4Tun Pipeline

Uses scikit-optimize to find optimal parameters for detection and SAM stages.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.callbacks import DeltaYStopper, CheckpointSaver
from skopt.utils import use_named_args

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .search_space import get_search_space, save_parameters
from .objective import PipelineObjective


class BayesianOptimizer:
    """
    Bayesian Optimization for P4Tun pipeline parameters.
    """
    
    def __init__(
        self,
        tunnel_id: str,
        stage: str = 'combined',
        data_dir: str = 'data',
        output_dir: str = 'p4tun/bo/results',
        metric: str = 'mIoU',
        n_calls: int = 50,
        n_initial_points: int = 10,
        random_state: int = 42,
        verbose: bool = True,
        optimizer: str = 'gp',  # 'gp', 'forest', 'gbrt'
    ):
        """
        Initialize the Bayesian Optimizer.
        
        Args:
            tunnel_id: Tunnel identifier (e.g., '4-1', '2-2')
            stage: Which parameters to optimize ('detection', 'sam', 'combined')
            data_dir: Base data directory
            output_dir: Directory to save results
            metric: Evaluation metric ('mIoU', 'OA', 'F1')
            n_calls: Total number of evaluations
            n_initial_points: Number of random initial points
            random_state: Random seed for reproducibility
            verbose: Print progress information
            optimizer: Optimization algorithm ('gp', 'forest', 'gbrt')
        """
        self.tunnel_id = tunnel_id
        self.stage = stage
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.metric = metric
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer_type = optimizer
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get search space
        self.dimensions, self.param_names = get_search_space(stage)
        
        # Initialize objective function
        self.objective = PipelineObjective(
            tunnel_id=tunnel_id,
            stage=stage,
            data_dir=data_dir,
            metric=metric,
            verbose=verbose,
        )
        
        self.result = None
    
    def optimize(self) -> Dict:
        """
        Run Bayesian Optimization.
        
        Returns:
            Dictionary with best parameters and optimization results
        """
        print(f"\n{'='*70}")
        print(f"BAYESIAN OPTIMIZATION FOR TUNNEL {self.tunnel_id}")
        print(f"{'='*70}")
        print(f"Stage: {self.stage}")
        print(f"Metric: {self.metric}")
        print(f"Parameters: {len(self.param_names)}")
        print(f"N calls: {self.n_calls}")
        print(f"N initial points: {self.n_initial_points}")
        print(f"Optimizer: {self.optimizer_type}")
        print(f"{'='*70}\n")
        
        # Select optimizer
        if self.optimizer_type == 'gp':
            minimize_func = gp_minimize
        elif self.optimizer_type == 'forest':
            minimize_func = forest_minimize
        elif self.optimizer_type == 'gbrt':
            minimize_func = gbrt_minimize
        else:
            minimize_func = gp_minimize
        
        # Set up callbacks
        callbacks = []
        
        # Early stopping if no improvement
        callbacks.append(DeltaYStopper(delta=0.001, n_best=10))
        
        # Checkpoint saver
        checkpoint_path = os.path.join(self.output_dir, f'{self.tunnel_id}_{self.stage}_checkpoint.pkl')
        callbacks.append(CheckpointSaver(checkpoint_path, compress=9))
        
        # Run optimization
        self.result = minimize_func(
            self.objective,
            self.dimensions,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=self.verbose,
            callback=callbacks,
        )
        
        # Get best parameters
        best_params = dict(zip(self.param_names, self.result.x))
        best_score = -self.result.fun  # Negate because we minimized negative score
        
        # Print results
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best {self.metric}: {best_score:.4f}")
        print(f"\nBest parameters:")
        for name, value in best_params.items():
            print(f"  {name}: {value}")
        
        # Save results
        results = {
            'tunnel_id': self.tunnel_id,
            'stage': self.stage,
            'metric': self.metric,
            'best_score': best_score,
            'best_params': best_params,
            'n_calls': self.n_calls,
            'n_evaluations': len(self.result.func_vals),
            'all_scores': [-v for v in self.result.func_vals],
            'convergence': self._get_convergence(),
            'timestamp': datetime.now().isoformat(),
        }
        
        self._save_results(results)
        
        return results
    
    def _get_convergence(self) -> List[float]:
        """Get convergence curve (best score at each iteration)."""
        if self.result is None:
            return []
        
        scores = [-v for v in self.result.func_vals]
        best_so_far = []
        current_best = -np.inf
        
        for score in scores:
            if score > current_best:
                current_best = score
            best_so_far.append(current_best)
        
        return best_so_far
    
    def _save_results(self, results: Dict):
        """Save optimization results."""
        # Save JSON results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(
            self.output_dir, 
            f'{self.tunnel_id}_{self.stage}_{timestamp}.json'
        )
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        print(f"\nResults saved to {json_path}")
        
        # Save best parameters to tunnel-specific directory
        if results['best_params']:
            # Save detection parameters
            if self.stage in ['detection', 'combined']:
                from .search_space import params_to_detection_dict
                detection_params = params_to_detection_dict(
                    list(results['best_params'].values()),
                    list(results['best_params'].keys())
                )
                save_parameters(detection_params, self.tunnel_id, 'detection')
            
            # Save SAM parameters
            if self.stage in ['sam', 'combined']:
                from .search_space import params_to_sam_dict
                sam_params = params_to_sam_dict(
                    list(results['best_params'].values()),
                    list(results['best_params'].keys())
                )
                save_parameters(sam_params, self.tunnel_id, 'sam')
        
        # Save objective history
        history_path = os.path.join(
            self.output_dir,
            f'{self.tunnel_id}_{self.stage}_{timestamp}_history.json'
        )
        self.objective.save_history(history_path)
        
        # Plot convergence
        self._plot_convergence(results, timestamp)
    
    def _plot_convergence(self, results: Dict, timestamp: str):
        """Plot and save convergence curve."""
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
            plt.ylabel(f'Best {self.metric}')
            plt.title(f'BO Convergence: {self.tunnel_id} ({self.stage})')
            plt.grid(True, alpha=0.3)
            
            # Mark best point
            best_idx = np.argmax(convergence)
            plt.scatter([best_idx + 1], [convergence[best_idx]], color='red', s=100, zorder=5)
            plt.annotate(f'Best: {convergence[best_idx]:.4f}', 
                        xy=(best_idx + 1, convergence[best_idx]),
                        xytext=(10, 10), textcoords='offset points')
            
            plot_path = os.path.join(
                self.output_dir,
                f'{self.tunnel_id}_{self.stage}_{timestamp}_convergence.png'
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Convergence plot saved to {plot_path}")
            
        except ImportError:
            print("matplotlib not available, skipping convergence plot")


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization for P4Tun Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m p4tun.bo.optimize --tunnel 4-1 --n-calls 50
  python -m p4tun.bo.optimize --tunnel 2-2 --stage detection --n-calls 30
  python -m p4tun.bo.optimize --tunnel 4-1 --metric OA --optimizer forest
"""
    )
    
    parser.add_argument('--tunnel', '-t', required=True,
                       help='Tunnel ID (e.g., 4-1, 2-2)')
    parser.add_argument('--stage', '-s', default='combined',
                       choices=['detection', 'sam', 'combined'],
                       help='Stage to optimize (default: combined)')
    parser.add_argument('--metric', '-m', default='mIoU',
                       choices=['mIoU', 'OA', 'F1'],
                       help='Optimization metric (default: mIoU)')
    parser.add_argument('--n-calls', '-n', type=int, default=50,
                       help='Number of evaluations (default: 50)')
    parser.add_argument('--n-initial', type=int, default=10,
                       help='Number of initial random points (default: 10)')
    parser.add_argument('--optimizer', '-o', default='gp',
                       choices=['gp', 'forest', 'gbrt'],
                       help='Optimizer type (default: gp)')
    parser.add_argument('--data-dir', default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--output-dir', default='p4tun/bo/results',
                       help='Output directory (default: p4tun/bo/results)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        tunnel_id=args.tunnel,
        stage=args.stage,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metric=args.metric,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        random_state=args.seed,
        verbose=not args.quiet,
        optimizer=args.optimizer,
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    return results


if __name__ == '__main__':
    main()

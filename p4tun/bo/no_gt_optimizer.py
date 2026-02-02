"""
No-Ground-Truth Bayesian Optimization

Uses intrinsic metrics to predict mIoU without needing ground truth labels.

Architecture:
  Layer A: Guardrails (hard constraints on intrinsic metrics)
  Layer B: Learned mIoU predictor from intrinsic metrics

Usage:
  python -m p4tun.bo.no_gt_optimizer --tunnel 1-4 --n-calls 20
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from p4tun.bo.search_space import (
    DETECTION_SPACE, SAM_SPACE,
    params_to_detection_dict, params_to_sam_dict,
)


# =============================================================================
# LAYER A: Guardrails (Hard Constraints)
# =============================================================================

# Thresholds derived from evaluation experiments
GUARDRAIL_THRESHOLDS = {
    # Detection quality
    'det_k_count_match': {'min': 0.8, 'max': None},      # K-block count should match
    'det_midpoint_ratio': {'min': 0.4, 'max': None},     # Segments at expected positions
    'det_real_detection_ratio': {'min': 0.5, 'max': None},  # Not too many assumed defaults
    'det_x_spacing_cv': {'min': None, 'max': 0.15},      # X-spacing must be uniform (critical for 2-2!)
    
    # SAM quality (avoid over-segmentation)
    'sam_mask_fill_rate': {'min': None, 'max': 0.95},    # Don't over-fill
}

# Tunnel-specific guardrail overrides
TUNNEL_GUARDRAIL_OVERRIDES = {
    '2-2': {
        'det_x_spacing_cv': {'min': None, 'max': 0.10},  # Stricter for 2-2
    },
}


def check_guardrails(metrics: Dict[str, float], tunnel_id: str = None) -> Tuple[bool, List[str]]:
    """
    Check if metrics pass all guardrail constraints.
    
    Args:
        metrics: Dictionary of intrinsic metrics
        tunnel_id: Tunnel ID for tunnel-specific thresholds
        
    Returns:
        (passed, violations): Whether passed and list of violated constraints
    """
    violations = []
    
    # Get effective thresholds (base + tunnel-specific overrides)
    effective_thresholds = dict(GUARDRAIL_THRESHOLDS)
    if tunnel_id and tunnel_id in TUNNEL_GUARDRAIL_OVERRIDES:
        for metric, override in TUNNEL_GUARDRAIL_OVERRIDES[tunnel_id].items():
            effective_thresholds[metric] = override
    
    for metric, thresholds in effective_thresholds.items():
        value = metrics.get(metric)
        if value is None or np.isnan(value):
            continue
            
        min_val = thresholds.get('min')
        max_val = thresholds.get('max')
        
        if min_val is not None and value < min_val:
            violations.append(f"{metric}={value:.3f} < {min_val}")
        if max_val is not None and value > max_val:
            violations.append(f"{metric}={value:.3f} > {max_val}")
    
    return len(violations) == 0, violations


# =============================================================================
# LAYER B: Learned mIoU Predictor
# =============================================================================

# Features used by the predictor (from evaluation)
PREDICTOR_FEATURES = [
    'det_midpoint_ratio',        # +0.87 correlation
    'det_real_detection_ratio',  # +0.69 correlation
    'det_k_count_match',         # +0.52 correlation
    'det_x_spacing_cv',          # critical for uniform detection
    'sam_mask_fill_rate',        # -0.82 correlation
]

# Simple linear model coefficients (from Ridge regression)
# Trained on: n=20, R²=0.72, Spearman=0.84
# These are approximate coefficients - ideally load from trained model
PREDICTOR_COEFFICIENTS = {
    'intercept': 0.45,
    'det_midpoint_ratio': 0.35,       # positive: higher → better mIoU
    'det_real_detection_ratio': 0.15, # positive
    'det_k_count_match': 0.10,        # positive
    'det_x_spacing_cv': -0.20,        # negative: lower CV → better mIoU
    'sam_mask_fill_rate': -0.25,      # negative: lower → better mIoU
}

# Tunnel-specific predictor coefficients (overrides)
# 2-2 requires VERY uniform X-spacing - heavy penalty for spacing CV
TUNNEL_PREDICTOR_COEFFICIENTS = {
    '2-2': {
        'intercept': 0.50,
        'det_midpoint_ratio': 0.25,           # slightly less weight
        'det_real_detection_ratio': 0.10,
        'det_k_count_match': 0.10,
        'det_x_spacing_cv': -0.50,            # HEAVY penalty for irregular spacing!
        'sam_mask_fill_rate': -0.20,
    },
}


def predict_miou(metrics: Dict[str, float], tunnel_id: str = None) -> float:
    """
    Predict mIoU from intrinsic metrics using tunnel-specific model or trained model.
    
    Args:
        metrics: Dictionary of intrinsic metrics
        tunnel_id: Tunnel ID for tunnel-specific predictor
        
    Returns:
        Predicted mIoU (0-1)
    """
    # PRIORITY 1: Use tunnel-specific coefficients if available
    # These are hand-tuned based on validation experiments
    if tunnel_id and tunnel_id in TUNNEL_PREDICTOR_COEFFICIENTS:
        coefficients = TUNNEL_PREDICTOR_COEFFICIENTS[tunnel_id]
        return _linear_predict(metrics, coefficients)
    
    # PRIORITY 2: Try to load trained model
    model_path = PROJECT_ROOT / "p4tun" / "bo" / "models" / "miou_predictor.pkl"
    if model_path.exists():
        try:
            from p4tun.bo.predictor import predict
            tid = tunnel_id or metrics.get('tunnel_id', '1-4')
            return predict(tid, metrics, model_path)
        except Exception:
            pass
    
    # PRIORITY 3: Fallback to generic linear model
    return _linear_predict(metrics, PREDICTOR_COEFFICIENTS)


def _linear_predict(metrics: Dict[str, float], coefficients: Dict[str, float]) -> float:
    """
    Simple linear prediction from intrinsic metrics.
    
    Args:
        metrics: Dictionary of intrinsic metrics
        coefficients: Dictionary of feature coefficients
        
    Returns:
        Predicted mIoU (0-1)
    """
    pred = coefficients.get('intercept', 0.45)
    
    for feature in PREDICTOR_FEATURES:
        value = metrics.get(feature)
        
        # Handle missing/nan values with feature-specific defaults
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if feature == 'det_x_spacing_cv':
                value = 0.3  # assume moderate variance if unknown
            else:
                value = 0.5  # neutral default
        
        coef = coefficients.get(feature, 0)
        pred += coef * value
    
    # Clip to valid range
    return np.clip(pred, 0.0, 1.0)


# =============================================================================
# No-GT Objective Function
# =============================================================================

class NoGTObjective:
    """
    Objective function for no-GT Bayesian Optimization.
    
    Runs pipeline, computes intrinsic metrics, applies guardrails,
    and returns predicted mIoU as the objective.
    """
    
    def __init__(
        self,
        tunnel_id: str,
        stage: str = 'combined',
        data_dir: str = 'data',
        verbose: bool = True
    ):
        self.tunnel_id = tunnel_id
        self.stage = stage
        self.data_dir = data_dir
        self.verbose = verbose
        self.eval_count = 0
        self.history = []
        
        # Import here to avoid circular imports
        from bo4tun.intrinsic_metrics import compute_all_metrics
        self.compute_metrics = compute_all_metrics
    
    def _get_search_space(self):
        """Get search space based on stage."""
        if self.stage == 'detection':
            space_dict = DETECTION_SPACE
        elif self.stage == 'sam':
            space_dict = SAM_SPACE
        else:  # combined
            space_dict = {**DETECTION_SPACE, **SAM_SPACE}
        
        # Convert dict to list of dimensions
        names = list(space_dict.keys())
        dims = [space_dict[n] for n in names]
        return dims, names
    
    def _update_params(self, params: List, names: List[str]) -> Dict:
        """Convert BO params to config dict and write to files."""
        param_dict = dict(zip(names, params))
        
        # Write detection params
        if self.stage in ['detection', 'combined']:
            det_config = params_to_detection_dict(params, names)
            det_path = PROJECT_ROOT / "p4tun" / "parameters" / self.tunnel_id / "parameters_detection.json"
            with open(det_path, 'w') as f:
                json.dump(det_config, f, indent=2)
        
        # Write SAM params
        if self.stage in ['sam', 'combined']:
            sam_config = params_to_sam_dict(params, names)
            sam_path = PROJECT_ROOT / "p4tun" / "parameters" / self.tunnel_id / "parameters_sam.json"
            with open(sam_path, 'w') as f:
                json.dump(sam_config, f, indent=2)
        
        return param_dict
    
    def _run_pipeline(self) -> bool:
        """Run detection and/or SAM pipeline."""
        import subprocess
        
        scripts = []
        if self.stage in ['detection', 'combined']:
            scripts.append(('4-1_detection.py', 'detection'))
        if self.stage in ['sam', 'combined']:
            scripts.append(('4-2_sam.py', 'SAM'))
        
        for script, name in scripts:
            script_path = PROJECT_ROOT / "p4tun" / script
            cmd = [sys.executable, str(script_path), self.tunnel_id, '--data-dir', self.data_dir]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300)
                if result.returncode != 0:
                    if self.verbose:
                        print(f"    {name} failed: {result.stderr.decode()[:100]}")
                    return False
            except subprocess.TimeoutExpired:
                if self.verbose:
                    print(f"    {name} timeout")
                return False
            except Exception as e:
                if self.verbose:
                    print(f"    {name} error: {e}")
                return False
        
        return True
    
    def _get_output_paths(self) -> Tuple[str, str]:
        """Get paths to pipeline outputs."""
        base = Path(self.data_dir) / self.tunnel_id
        detected = base / "detected.csv"
        final = base / "final.csv"
        return str(detected), str(final)
    
    def __call__(self, params: List) -> float:
        """
        Evaluate params and return negative predicted mIoU.
        
        (Negative because skopt minimizes)
        """
        self.eval_count += 1
        
        if self.verbose:
            print(f"\n[Eval {self.eval_count}] Tunnel: {self.tunnel_id}")
        
        # Get space names
        _, names = self._get_search_space()
        
        # Update params
        param_dict = self._update_params(params, names)
        
        # Run pipeline
        if not self._run_pipeline():
            if self.verbose:
                print("    Pipeline failed → penalty score")
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'status': 'failed',
                'predicted_miou': 0.0,
            })
            return 0.0  # Worst score (we negate at the end)
        
        # Compute intrinsic metrics
        detected_csv, final_csv = self._get_output_paths()
        try:
            metrics = self.compute_metrics(
                self.tunnel_id, detected_csv, final_csv, self.data_dir
            )
            # Add prefix for consistency
            metrics_prefixed = {}
            for k, v in metrics.items():
                if not k.startswith('det_') and not k.startswith('sam_'):
                    # Assume detection metrics if no prefix
                    metrics_prefixed[f'det_{k}'] if 'prompt' not in k and 'segment' not in k and 'mask' not in k and 'template' not in k else metrics_prefixed[f'sam_{k}']
                metrics_prefixed[k] = v
            metrics = metrics_prefixed
        except Exception as e:
            if self.verbose:
                print(f"    Metrics computation failed: {e}")
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'status': 'metrics_failed',
                'predicted_miou': 0.0,
            })
            return 0.0
        
        # Layer A: Check guardrails (with tunnel-specific thresholds)
        passed, violations = check_guardrails(metrics, self.tunnel_id)
        if not passed:
            if self.verbose:
                print(f"    Guardrails FAILED: {violations}")
            # Penalize but don't completely reject
            penalty = 0.1 * len(violations)
        else:
            if self.verbose:
                print("    Guardrails passed ✓")
            penalty = 0.0
        
        # Layer B: Predict mIoU (with tunnel-specific predictor)
        predicted_miou = predict_miou(metrics, self.tunnel_id)
        
        # Apply penalty
        final_score = max(0.0, predicted_miou - penalty)
        
        if self.verbose:
            print(f"    Predicted mIoU: {predicted_miou:.4f}")
            if penalty > 0:
                print(f"    After penalty: {final_score:.4f}")
        
        # Record history
        self.history.append({
            'eval': self.eval_count,
            'params': param_dict,
            'metrics': {k: float(v) if not np.isnan(v) else None for k, v in metrics.items() if k != 'tunnel_id'},
            'guardrails_passed': passed,
            'violations': violations,
            'predicted_miou': float(predicted_miou),
            'final_score': float(final_score),
            'status': 'success',
        })
        
        # Return negative (skopt minimizes)
        return -final_score


# =============================================================================
# Main Optimization Loop
# =============================================================================

def run_no_gt_optimization(
    tunnel_id: str,
    stage: str = 'combined',
    n_calls: int = 20,
    n_initial: int = 5,
    data_dir: str = 'data',
    verbose: bool = True
) -> Dict:
    """
    Run no-GT Bayesian Optimization.
    
    Args:
        tunnel_id: Tunnel to optimize
        stage: Stage to optimize ('detection', 'sam', 'combined')
        n_calls: Total number of evaluations
        n_initial: Initial random evaluations
        data_dir: Data directory
        verbose: Print progress
        
    Returns:
        Results dictionary with best params and history
    """
    print("=" * 70)
    print(f"No-GT Bayesian Optimization")
    print(f"Tunnel: {tunnel_id}, Stage: {stage}")
    print(f"Evaluations: {n_calls} (initial: {n_initial})")
    print("=" * 70)
    
    # Create objective
    objective = NoGTObjective(
        tunnel_id=tunnel_id,
        stage=stage,
        data_dir=data_dir,
        verbose=verbose
    )
    
    # Get search space
    dims, names = objective._get_search_space()
    
    # Run BO
    result = gp_minimize(
        objective,
        dims,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=42,
        verbose=verbose,
    )
    
    # Extract best
    best_params = dict(zip(names, result.x))
    best_score = -result.fun  # Negate back
    
    # Find best from history
    best_eval = max(objective.history, key=lambda x: x.get('final_score', 0))
    
    print("\n" + "=" * 70)
    print("Optimization Complete")
    print("=" * 70)
    print(f"Best predicted mIoU: {best_score:.4f}")
    print(f"Best params: {best_params}")
    
    # Save results
    results = {
        'tunnel_id': tunnel_id,
        'stage': stage,
        'n_calls': n_calls,
        'best_predicted_miou': best_score,
        'best_params': best_params,
        'history': objective.history,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save to file
    output_dir = PROJECT_ROOT / "p4tun" / "bo" / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"no_gt_bo_{tunnel_id}_{stage}_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='No-GT Bayesian Optimization')
    parser.add_argument('--tunnel', default='1-4', help='Tunnel ID')
    parser.add_argument('--stage', default='combined', 
                        choices=['detection', 'sam', 'combined'],
                        help='Stage to optimize')
    parser.add_argument('--n-calls', type=int, default=20, help='Total evaluations')
    parser.add_argument('--n-initial', type=int, default=5, help='Initial random evals')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output')
    
    args = parser.parse_args()
    
    run_no_gt_optimization(
        tunnel_id=args.tunnel,
        stage=args.stage,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        data_dir=args.data_dir,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()

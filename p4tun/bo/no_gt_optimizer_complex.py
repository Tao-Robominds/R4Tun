"""
No-Ground-Truth Bayesian Optimization for Complex Staggered Patterns

Uses intrinsic metrics to predict mIoU without needing ground truth labels.
Specific for tunnels 4-1 and 5-1 which use complex_staggered detection and SAM.

Architecture:
  Layer A: Guardrails (hard constraints on intrinsic metrics)
  Layer B: Learned mIoU predictor from intrinsic metrics

Usage:
  python -m p4tun.bo.no_gt_optimizer_complex --tunnel 5-1 --n-calls 20
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

from p4tun.bo.detection_complex_bo import (
    COMPLEX_DETECTION_SEARCH_SPACE,
    get_complex_detection_dimensions,
    params_to_complex_detection_json,
)
from p4tun.bo.sam_complex_bo import (
    COMPLEX_SAM_SEARCH_SPACE,
    get_complex_sam_dimensions,
    params_to_complex_sam_json,
)


# =============================================================================
# LAYER A: Guardrails (Hard Constraints) for Complex Staggered
# =============================================================================

# =============================================================================
# LAYER A: Detection Guardrails for Complex Staggered
# =============================================================================

# Detection quality thresholds - filter out clearly bad detection configs
# NOTE: Complex detection uses intersection_cluster (real) and midpoint_cluster (fallback)
COMPLEX_GUARDRAIL_THRESHOLDS = {
    # Detection count - must be reasonable
    'det_k_count': {'min': 4, 'max': 12},  # Expected: 6-9 depending on tunnel
    
    # X-spacing coefficient of variation - uniformity check
    'det_x_spacing_cv': {'min': None, 'max': 0.60},  # Allow some non-uniformity
    
    # Y-position range - shouldn't be too extreme
    'det_y_range': {'min': 200, 'max': 1500},  # Reasonable Y spread
}

# Tunnel-specific guardrail overrides
COMPLEX_TUNNEL_GUARDRAIL_OVERRIDES = {
    '4-1': {
        'det_k_count': {'min': 7, 'max': 12},         # 4-1 expects ~9 K-blocks
        'det_x_spacing_cv': {'min': None, 'max': 0.50},  # More uniform for 4-1
        'det_y_range': {'min': 200, 'max': 2000},     # 4-1 may have larger Y range
    },
    '5-1': {
        'det_k_count': {'min': 5, 'max': 10},         # 5-1 expects 6-7 K-blocks
        'det_x_spacing_cv': {'min': None, 'max': 0.80},  # Very non-uniform expected (large gap)
        'det_y_range': {'min': 200, 'max': 3500},     # 5-1 has large Y spread
    },
}

# =============================================================================
# SAM Parameter Guardrails (for combined optimization)
# =============================================================================

# Based on historical analysis: segment_width has -0.789 correlation with mIoU
# Lower segment_width = higher mIoU, but too low causes crashes
SAM_PARAM_GUARDRAILS = {
    'segment_width': {'min': 1150, 'max': 1350},      # Optimal range
    'k_height': {'min': 900, 'max': 1200},            # Lower is better but not too low
    'ab_height': {'min': 3000, 'max': 3500},          # Lower is better
    'angle_deg': {'min': 6.0, 'max': 9.0},            # Reasonable range
}


def check_complex_guardrails(
    metrics: Dict[str, float], 
    tunnel_id: str = None,
    sam_params: Dict[str, float] = None
) -> Tuple[bool, List[str]]:
    """
    Check if metrics and SAM params pass all complex staggered guardrail constraints.
    
    Args:
        metrics: Dictionary of detection intrinsic metrics
        tunnel_id: Tunnel ID for tunnel-specific thresholds
        sam_params: SAM geometry parameters (optional, for combined optimization)
        
    Returns:
        (passed, violations): Whether passed and list of violated constraints
    """
    violations = []
    
    # ===== Detection Guardrails =====
    effective_thresholds = dict(COMPLEX_GUARDRAIL_THRESHOLDS)
    if tunnel_id and tunnel_id in COMPLEX_TUNNEL_GUARDRAIL_OVERRIDES:
        for metric, override in COMPLEX_TUNNEL_GUARDRAIL_OVERRIDES[tunnel_id].items():
            effective_thresholds[metric] = override
    
    for metric, thresholds in effective_thresholds.items():
        value = metrics.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
            
        min_val = thresholds.get('min')
        max_val = thresholds.get('max')
        
        if min_val is not None and value < min_val:
            violations.append(f"{metric}={value:.3f} < {min_val}")
        if max_val is not None and value > max_val:
            violations.append(f"{metric}={value:.3f} > {max_val}")
    
    # ===== SAM Parameter Guardrails (if provided) =====
    if sam_params:
        for param, thresholds in SAM_PARAM_GUARDRAILS.items():
            value = sam_params.get(param)
            if value is None:
                continue
            
            min_val = thresholds.get('min')
            max_val = thresholds.get('max')
            
            if min_val is not None and value < min_val:
                violations.append(f"sam_{param}={value:.1f} < {min_val}")
            if max_val is not None and value > max_val:
                violations.append(f"sam_{param}={value:.1f} > {max_val}")
    
    return len(violations) == 0, violations


# =============================================================================
# LAYER B: Learned mIoU Predictor for Complex Staggered
# =============================================================================

# Trained model uses SAM geometry parameters as features
# (These are INPUT params, not output metrics, but correlate strongly with mIoU)
# Model trained on n=70 samples, CV MAE=0.0125, Spearman=0.87
COMPLEX_SAM_FEATURES = ['segment_width', 'ab_height', 'k_height', 'angle_deg']

# Trained model coefficients (Ridge regression, scaled features)
# segment_width has -0.789 correlation with mIoU (dominant!)
COMPLEX_MODEL_COEFFICIENTS = {
    'segment_width': -0.0352,
    'ab_height': -0.0067,
    'k_height': -0.0012,
    'angle_deg': -0.0008,
    'intercept': 0.3763,  # Mean mIoU
}

# Feature scaling parameters (from training)
COMPLEX_FEATURE_MEANS = {
    'segment_width': 1230.0,
    'ab_height': 3300.0,
    'k_height': 1050.0,
    'angle_deg': 7.5,
}
COMPLEX_FEATURE_STDS = {
    'segment_width': 55.0,
    'ab_height': 100.0,
    'k_height': 80.0,
    'angle_deg': 0.8,
}


def predict_complex_miou_from_params(sam_params: Dict[str, float]) -> float:
    """
    Predict mIoU from SAM parameters using trained model.
    
    This is the primary prediction method for complex staggered patterns.
    Uses SAM geometry parameters which have strong correlation with mIoU.
    
    Args:
        sam_params: Dictionary with segment_width, ab_height, k_height, angle_deg
        
    Returns:
        Predicted mIoU (0-1)
    """
    # Try to load trained model first
    model_path = PROJECT_ROOT / "p4tun" / "bo" / "models" / "complex_miou_predictor.pkl"
    if model_path.exists():
        try:
            import pickle
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)
            
            model = bundle['model']
            scaler = bundle['scaler']
            features = bundle['features']
            
            # Extract features
            X = np.array([[sam_params.get(f, COMPLEX_FEATURE_MEANS.get(f, 0)) for f in features]])
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            return float(np.clip(pred, 0.0, 1.0))
        except Exception as e:
            pass  # Fall back to manual calculation
    
    # Manual calculation using stored coefficients
    pred = COMPLEX_MODEL_COEFFICIENTS['intercept']
    for feat in COMPLEX_SAM_FEATURES:
        value = sam_params.get(feat, COMPLEX_FEATURE_MEANS.get(feat, 0))
        # Scale feature
        mean = COMPLEX_FEATURE_MEANS.get(feat, 0)
        std = COMPLEX_FEATURE_STDS.get(feat, 1)
        scaled = (value - mean) / std if std > 0 else 0
        # Apply coefficient
        coef = COMPLEX_MODEL_COEFFICIENTS.get(feat, 0)
        pred += coef * scaled
    
    return float(np.clip(pred, 0.0, 1.0))


def predict_complex_miou(metrics: Dict[str, float], tunnel_id: str = None, sam_params: Dict = None) -> float:
    """
    Predict mIoU for complex staggered patterns.
    
    Uses SAM parameters (primary) and detection metrics (secondary) for prediction.
    
    Args:
        metrics: Dictionary of detection intrinsic metrics
        tunnel_id: Tunnel ID (4-1 or 5-1)
        sam_params: SAM geometry parameters (segment_width, k_height, etc.)
        
    Returns:
        Predicted mIoU (0-1)
    """
    # Primary: Use SAM parameters if available (strong correlation)
    if sam_params:
        base_pred = predict_complex_miou_from_params(sam_params)
    else:
        # Fallback: use mean mIoU
        base_pred = COMPLEX_MODEL_COEFFICIENTS['intercept']
    
    # Secondary: Adjust based on detection quality
    # Bad detection can reduce mIoU even with good SAM params
    adjustment = 0.0
    
    # Penalize very bad detection
    det_k_count = metrics.get('det_k_count', 0)
    expected_k = 9 if tunnel_id == '4-1' else 6  # 5-1 has 6 K-blocks
    if det_k_count > 0:
        count_error = abs(det_k_count - expected_k) / expected_k
        if count_error > 0.3:  # More than 30% error
            adjustment -= 0.05 * count_error
    
    # Penalize very high x_spacing_cv
    x_spacing_cv = metrics.get('det_x_spacing_cv', 0)
    if x_spacing_cv > 0.6:
        adjustment -= 0.02 * (x_spacing_cv - 0.6)
    
    final_pred = base_pred + adjustment
    return float(np.clip(final_pred, 0.0, 1.0))


# =============================================================================
# Complex Staggered No-GT Objective Function
# =============================================================================

class ComplexNoGTObjective:
    """
    Objective function for no-GT Bayesian Optimization of complex staggered patterns.
    
    Runs pipeline, computes intrinsic metrics, applies guardrails,
    and returns predicted mIoU as the objective.
    """
    
    def __init__(
        self,
        tunnel_id: str,
        stage: str = 'detection',
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
        
        # Determine expected K count for this tunnel
        self.expected_k = 9 if tunnel_id == '4-1' else 6  # 5-1 has 6 K blocks
    
    def _get_search_space(self):
        """Get search space based on stage."""
        if self.stage == 'detection':
            dims, names = get_complex_detection_dimensions()
        elif self.stage == 'sam':
            dims, names = get_complex_sam_dimensions()
        else:  # combined
            det_dims, det_names = get_complex_detection_dimensions()
            sam_dims, sam_names = get_complex_sam_dimensions()
            dims = det_dims + sam_dims
            names = det_names + sam_names
        
        return dims, names
    
    def _update_params(self, params: List, names: List[str]) -> Dict:
        """Convert BO params to config dict and write to files."""
        param_dict = dict(zip(names, params))
        
        # Write detection params
        if self.stage in ['detection', 'combined']:
            det_config = params_to_complex_detection_json(params, names, self.tunnel_id)
            det_path = PROJECT_ROOT / "p4tun" / "parameters" / self.tunnel_id / "parameters_detection.json"
            det_path.parent.mkdir(parents=True, exist_ok=True)
            with open(det_path, 'w') as f:
                json.dump(det_config, f, indent=2)
        
        # Write SAM params
        if self.stage in ['sam', 'combined']:
            sam_config = params_to_complex_sam_json(params, names, self.tunnel_id)
            sam_path = PROJECT_ROOT / "p4tun" / "parameters" / self.tunnel_id / "parameters_sam.json"
            sam_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sam_path, 'w') as f:
                json.dump(sam_config, f, indent=2)
        
        return param_dict
    
    def _run_pipeline(self) -> bool:
        """Run complex detection and/or SAM pipeline."""
        import subprocess
        
        scripts = []
        
        # Always run detection first (SAM needs detected.csv)
        scripts.append(('4-1_detection_complex.py', 'complex detection', ['--data-dir', self.data_dir]))
        
        # Run SAM if stage includes it
        if self.stage in ['sam', 'combined']:
            scripts.append(('4-2_sam_complex.py', 'complex SAM', []))
        
        for script, name, extra_args in scripts:
            script_path = PROJECT_ROOT / "p4tun" / script
            cmd = [sys.executable, str(script_path), self.tunnel_id] + extra_args
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=300, cwd=PROJECT_ROOT)
                if result.returncode != 0:
                    if self.verbose:
                        stderr = result.stderr.decode()[:100] if result.stderr else "no stderr"
                        print(f"    {name} failed: {stderr}")
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
    
    def _extract_sam_params(self, param_dict: Dict) -> Dict[str, float]:
        """Extract SAM geometry parameters from param dict."""
        return {
            'segment_width': param_dict.get('segment_width', 1200.0),
            'k_height': param_dict.get('k_height', 1079.92),
            'ab_height': param_dict.get('ab_height', 3239.77),
            'angle_deg': param_dict.get('angle_deg', 7.52),
        }
    
    def __call__(self, params: List) -> float:
        """
        Evaluate params and return negative predicted mIoU.
        
        (Negative because skopt minimizes)
        """
        self.eval_count += 1
        
        if self.verbose:
            print(f"\n[Eval {self.eval_count}] Tunnel: {self.tunnel_id} (complex_staggered)")
        
        # Get space names
        _, names = self._get_search_space()
        
        # Update params
        param_dict = self._update_params(params, names)
        
        # Extract SAM params for prediction
        sam_params = self._extract_sam_params(param_dict) if self.stage in ['sam', 'combined'] else None
        
        # Run pipeline
        if not self._run_pipeline():
            if self.verbose:
                print("    Pipeline failed → penalty score")
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'sam_params': sam_params,
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
        except Exception as e:
            if self.verbose:
                print(f"    Metrics computation failed: {e}")
            self.history.append({
                'eval': self.eval_count,
                'params': param_dict,
                'sam_params': sam_params,
                'status': 'metrics_failed',
                'predicted_miou': 0.0,
            })
            return 0.0
        
        # Layer A: Check guardrails (detection metrics + SAM params)
        passed, violations = check_complex_guardrails(metrics, self.tunnel_id, sam_params)
        if not passed:
            if self.verbose:
                print(f"    Guardrails FAILED: {violations}")
            # Penalize but don't completely reject
            penalty = 0.1 * len(violations)
        else:
            if self.verbose:
                print("    Guardrails passed ✓")
            penalty = 0.0
        
        # Layer B: Predict mIoU using trained model
        # SAM params are primary predictor (r=0.87 correlation with mIoU)
        # Detection metrics provide secondary adjustment
        predicted_miou = predict_complex_miou(metrics, self.tunnel_id, sam_params)
        
        # Apply penalty
        final_score = max(0.0, predicted_miou - penalty)
        
        if self.verbose:
            print(f"    Detection: k_count={metrics.get('det_k_count', 0):.0f}, "
                  f"x_spacing_cv={metrics.get('det_x_spacing_cv', 0):.3f}")
            if sam_params:
                print(f"    SAM params: seg_width={sam_params['segment_width']:.0f}, "
                      f"k_height={sam_params['k_height']:.0f}")
            print(f"    Predicted mIoU: {predicted_miou:.4f}")
            if penalty > 0:
                print(f"    After penalty: {final_score:.4f}")
        
        # Record history
        self.history.append({
            'eval': self.eval_count,
            'params': param_dict,
            'sam_params': sam_params,
            'metrics': {k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else None 
                       for k, v in metrics.items()},
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

def run_complex_no_gt_optimization(
    tunnel_id: str,
    stage: str = 'detection',
    n_calls: int = 20,
    n_initial: int = 5,
    data_dir: str = 'data',
    verbose: bool = True
) -> Dict:
    """
    Run no-GT Bayesian Optimization for complex staggered patterns.
    
    Args:
        tunnel_id: Tunnel to optimize (4-1 or 5-1)
        stage: Stage to optimize ('detection', 'sam', 'combined')
        n_calls: Total number of evaluations
        n_initial: Initial random evaluations
        data_dir: Data directory
        verbose: Print progress
        
    Returns:
        Results dictionary with best params and history
    """
    print("=" * 70)
    print(f"No-GT Bayesian Optimization (Complex Staggered)")
    print(f"Tunnel: {tunnel_id}, Stage: {stage}")
    print(f"Evaluations: {n_calls} (initial: {n_initial})")
    print("=" * 70)
    
    # Create objective
    objective = ComplexNoGTObjective(
        tunnel_id=tunnel_id,
        stage=stage,
        data_dir=data_dir,
        verbose=verbose
    )
    
    # Get search space
    dims, names = objective._get_search_space()
    
    print(f"Search space: {len(names)} parameters")
    
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
    print(f"Best params (subset):")
    # Show key params
    key_params = ['binary_threshold', 'hough_oblique_threshold', 'angle_positive_min', 
                  'angle_positive_max', 'complex_hough_threshold', 'complex_eps_primary']
    for p in key_params:
        if p in best_params:
            print(f"  {p}: {best_params[p]}")
    
    # Save results
    results = {
        'tunnel_id': tunnel_id,
        'stage': stage,
        'pattern': 'complex_staggered',
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
    output_path = output_dir / f"no_gt_complex_{tunnel_id}_{stage}_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='No-GT Bayesian Optimization (Complex Staggered)')
    parser.add_argument('--tunnel', default='5-1', help='Tunnel ID (4-1 or 5-1)')
    parser.add_argument('--stage', default='detection', 
                        choices=['detection', 'sam', 'combined'],
                        help='Stage to optimize')
    parser.add_argument('--n-calls', type=int, default=20, help='Total evaluations')
    parser.add_argument('--n-initial', type=int, default=5, help='Initial random evals')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output')
    
    args = parser.parse_args()
    
    run_complex_no_gt_optimization(
        tunnel_id=args.tunnel,
        stage=args.stage,
        n_calls=args.n_calls,
        n_initial=args.n_initial,
        data_dir=args.data_dir,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()

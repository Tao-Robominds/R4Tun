"""
Inverse Search Module for Target-Driven Parameter Optimization

Given a target metric (e.g., mIoU >= 0.75), search the parameter space
using the GP surrogate to find candidates that meet the target.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm

from .gp_surrogate import GPSurrogate, PredictionResult


@dataclass
class SearchCandidate:
    """Container for a parameter candidate from inverse search."""
    params: Dict[str, float]
    predicted_mean: float
    predicted_std: float
    expected_improvement: float = 0.0
    acquisition_value: float = 0.0
    
    @property
    def lower_bound(self, alpha: float = 1.96) -> float:
        """95% confidence lower bound."""
        return self.predicted_mean - alpha * self.predicted_std
    
    @property
    def upper_bound(self, alpha: float = 1.96) -> float:
        """95% confidence upper bound."""
        return self.predicted_mean + alpha * self.predicted_std
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'params': self.params,
            'predicted_mean': self.predicted_mean,
            'predicted_std': self.predicted_std,
            'expected_improvement': self.expected_improvement,
            'acquisition_value': self.acquisition_value,
        }


@dataclass
class SearchResult:
    """Container for inverse search results."""
    candidates: List[SearchCandidate]
    best_candidate: SearchCandidate
    target_metric: float
    search_config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'candidates': [c.to_dict() for c in self.candidates],
            'best_candidate': self.best_candidate.to_dict(),
            'target_metric': self.target_metric,
            'search_config': self.search_config,
            'timestamp': self.timestamp,
        }


class InverseSearch:
    """
    Inverse parameter search using GP surrogate.
    
    Searches for parameter configurations that are predicted to achieve
    a target metric value, using acquisition functions that balance
    exploitation (high predicted mean) and exploration (high uncertainty).
    """
    
    def __init__(
        self,
        surrogate: GPSurrogate,
        acquisition: str = 'ei',  # 'ei', 'ucb', 'poi', 'mean'
        exploration_weight: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize InverseSearch.
        
        Args:
            surrogate: Trained GP surrogate model
            acquisition: Acquisition function type
                - 'ei': Expected Improvement
                - 'ucb': Upper Confidence Bound
                - 'poi': Probability of Improvement
                - 'mean': Pure exploitation (predicted mean)
            exploration_weight: Trade-off parameter (xi for EI, kappa for UCB)
            random_state: Random seed
        """
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.exploration_weight = exploration_weight
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Get best observed from surrogate
        self._best_params, self._best_score = surrogate.get_best_observed()
    
    def _acquisition_function(
        self,
        X: np.ndarray,
        target: float,
    ) -> np.ndarray:
        """
        Compute acquisition function value.
        
        For inverse search, we want to maximize the probability/value
        of achieving the target metric.
        
        Args:
            X: Parameter vectors
            target: Target metric value
            
        Returns:
            Acquisition values (higher is better)
        """
        prediction = self.surrogate.predict(X)
        mean = prediction.mean
        std = prediction.std
        
        # Reference value for improvement calculations
        best_so_far = max(self._best_score, target - 0.1)
        
        if self.acquisition == 'ei':
            # Expected Improvement
            xi = self.exploration_weight
            with np.errstate(divide='warn'):
                improvement = mean - best_so_far - xi
                Z = improvement / (std + 1e-8)
                ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
                ei[std < 1e-8] = 0.0
            return ei
        
        elif self.acquisition == 'ucb':
            # Upper Confidence Bound
            kappa = self.exploration_weight
            return mean + kappa * std
        
        elif self.acquisition == 'poi':
            # Probability of Improvement over target
            with np.errstate(divide='warn'):
                Z = (mean - target) / (std + 1e-8)
                poi = norm.cdf(Z)
            return poi
        
        elif self.acquisition == 'mean':
            # Pure exploitation
            return mean
        
        else:
            return mean
    
    def _objective_to_minimize(self, x: np.ndarray, target: float) -> float:
        """Objective function for scipy optimizers (minimization)."""
        x = x.reshape(1, -1)
        acq_value = self._acquisition_function(x, target)
        return -acq_value[0]  # Negate for minimization
    
    def search(
        self,
        target_metric: float,
        n_candidates: int = 10,
        n_restarts: int = 20,
        method: str = 'de',  # 'de', 'local', 'random'
        bounds_expansion: float = 0.0,
    ) -> SearchResult:
        """
        Search for parameters that achieve the target metric.
        
        Args:
            target_metric: Target score to achieve (e.g., mIoU >= 0.75)
            n_candidates: Number of candidate configurations to return
            n_restarts: Number of optimization restarts (for local method)
            method: Search method
                - 'de': Differential Evolution (global)
                - 'local': Multi-start local optimization (L-BFGS-B)
                - 'random': Random sampling with evaluation
            bounds_expansion: Factor to expand parameter bounds (0.0 = no expansion)
            
        Returns:
            SearchResult with candidate configurations
        """
        param_names = self.surrogate.param_names
        param_bounds = self.surrogate.param_bounds
        n_params = len(param_names)
        
        # Build bounds array
        bounds = []
        for name in param_names:
            low, high = param_bounds[name]
            # Optionally expand bounds
            if bounds_expansion > 0:
                range_val = high - low
                low = low - bounds_expansion * range_val
                high = high + bounds_expansion * range_val
            bounds.append((low, high))
        
        bounds_array = np.array(bounds)
        
        candidates = []
        
        if method == 'de':
            # Differential Evolution - global optimization
            candidates = self._search_differential_evolution(
                target_metric, bounds_array, n_candidates, param_names
            )
        
        elif method == 'local':
            # Multi-start L-BFGS-B
            candidates = self._search_local_multistart(
                target_metric, bounds_array, n_restarts, n_candidates, param_names
            )
        
        elif method == 'random':
            # Random sampling
            candidates = self._search_random(
                target_metric, bounds_array, n_restarts * 100, n_candidates, param_names
            )
        
        else:
            # Default to DE
            candidates = self._search_differential_evolution(
                target_metric, bounds_array, n_candidates, param_names
            )
        
        # Sort by acquisition value (descending)
        candidates.sort(key=lambda c: c.acquisition_value, reverse=True)
        
        # Select top candidates
        top_candidates = candidates[:n_candidates]
        
        # Find best candidate (highest predicted mean among those above target)
        above_target = [c for c in top_candidates if c.predicted_mean >= target_metric * 0.95]
        if above_target:
            best_candidate = max(above_target, key=lambda c: c.predicted_mean)
        else:
            best_candidate = max(top_candidates, key=lambda c: c.predicted_mean)
        
        search_config = {
            'method': method,
            'acquisition': self.acquisition,
            'exploration_weight': self.exploration_weight,
            'n_restarts': n_restarts,
            'bounds_expansion': bounds_expansion,
        }
        
        return SearchResult(
            candidates=top_candidates,
            best_candidate=best_candidate,
            target_metric=target_metric,
            search_config=search_config,
        )
    
    def _search_differential_evolution(
        self,
        target: float,
        bounds: np.ndarray,
        n_candidates: int,
        param_names: List[str],
    ) -> List[SearchCandidate]:
        """Search using Differential Evolution."""
        candidates = []
        
        # Run DE multiple times with different seeds to get diverse candidates
        for i in range(max(n_candidates, 5)):
            result = differential_evolution(
                lambda x: self._objective_to_minimize(x, target),
                bounds,
                seed=self.random_state + i,
                maxiter=200,
                tol=1e-6,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                workers=1,
            )
            
            if result.success or result.fun < -1e-6:
                x_opt = result.x
                prediction = self.surrogate.predict(x_opt.reshape(1, -1))
                
                params = dict(zip(param_names, x_opt))
                candidate = SearchCandidate(
                    params=params,
                    predicted_mean=float(prediction.mean[0]),
                    predicted_std=float(prediction.std[0]),
                    acquisition_value=float(-result.fun),
                )
                candidates.append(candidate)
        
        return candidates
    
    def _search_local_multistart(
        self,
        target: float,
        bounds: np.ndarray,
        n_restarts: int,
        n_candidates: int,
        param_names: List[str],
    ) -> List[SearchCandidate]:
        """Search using multi-start local optimization."""
        candidates = []
        
        for i in range(n_restarts):
            # Random starting point
            x0 = np.array([
                np.random.uniform(low, high)
                for low, high in bounds
            ])
            
            result = minimize(
                lambda x: self._objective_to_minimize(x, target),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100},
            )
            
            if result.success or result.fun < -1e-6:
                x_opt = result.x
                prediction = self.surrogate.predict(x_opt.reshape(1, -1))
                
                params = dict(zip(param_names, x_opt))
                candidate = SearchCandidate(
                    params=params,
                    predicted_mean=float(prediction.mean[0]),
                    predicted_std=float(prediction.std[0]),
                    acquisition_value=float(-result.fun),
                )
                candidates.append(candidate)
        
        return candidates
    
    def _search_random(
        self,
        target: float,
        bounds: np.ndarray,
        n_samples: int,
        n_candidates: int,
        param_names: List[str],
    ) -> List[SearchCandidate]:
        """Search using random sampling."""
        # Generate random samples
        X = np.array([
            np.random.uniform(low, high, n_samples)
            for low, high in bounds
        ]).T
        
        # Evaluate acquisition function
        acq_values = self._acquisition_function(X, target)
        
        # Get top candidates
        top_indices = np.argsort(acq_values)[-n_candidates * 2:][::-1]
        
        candidates = []
        for idx in top_indices:
            x = X[idx]
            prediction = self.surrogate.predict(x.reshape(1, -1))
            
            params = dict(zip(param_names, x))
            candidate = SearchCandidate(
                params=params,
                predicted_mean=float(prediction.mean[0]),
                predicted_std=float(prediction.std[0]),
                acquisition_value=float(acq_values[idx]),
            )
            candidates.append(candidate)
        
        return candidates
    
    def grid_search_sensitive_params(
        self,
        target_metric: float,
        sensitive_params: List[str],
        n_points_per_param: int = 10,
        base_params: Optional[Dict[str, float]] = None,
    ) -> SearchResult:
        """
        Grid search over most sensitive parameters.
        
        Args:
            target_metric: Target score
            sensitive_params: List of parameter names to search
            n_points_per_param: Number of grid points per parameter
            base_params: Base parameter values for non-searched params
            
        Returns:
            SearchResult with evaluated grid points
        """
        param_names = self.surrogate.param_names
        param_bounds = self.surrogate.param_bounds
        
        # Use best observed as base if not provided
        if base_params is None:
            base_params = dict(zip(param_names, self._best_params))
        
        # Build grid
        grid_arrays = []
        for name in sensitive_params:
            if name in param_bounds:
                low, high = param_bounds[name]
                grid_arrays.append(np.linspace(low, high, n_points_per_param))
        
        # Create meshgrid
        meshes = np.meshgrid(*grid_arrays)
        grid_points = np.column_stack([m.ravel() for m in meshes])
        
        # Evaluate each grid point
        candidates = []
        for point in grid_points:
            # Build full parameter vector
            params = base_params.copy()
            for i, name in enumerate(sensitive_params):
                if name in params:
                    params[name] = point[i]
            
            # Predict
            X = np.array([[params[n] for n in param_names]])
            prediction = self.surrogate.predict(X)
            acq_value = self._acquisition_function(X, target_metric)[0]
            
            candidate = SearchCandidate(
                params=params,
                predicted_mean=float(prediction.mean[0]),
                predicted_std=float(prediction.std[0]),
                acquisition_value=float(acq_value),
            )
            candidates.append(candidate)
        
        # Sort by predicted mean
        candidates.sort(key=lambda c: c.predicted_mean, reverse=True)
        
        # Best candidate
        best_candidate = candidates[0]
        
        return SearchResult(
            candidates=candidates[:20],  # Top 20
            best_candidate=best_candidate,
            target_metric=target_metric,
            search_config={
                'method': 'grid',
                'sensitive_params': sensitive_params,
                'n_points_per_param': n_points_per_param,
            },
        )


def save_search_results(result: SearchResult, filepath: str):
    """Save search results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Search results saved to {filepath}")


def main():
    """Example usage of InverseSearch."""
    import argparse
    from .data_extractor import DataExtractor
    from .gp_surrogate import GPSurrogate
    
    parser = argparse.ArgumentParser(description='Inverse search for target metrics')
    parser.add_argument('--stage', '-s', default='detection',
                       choices=['detection', 'sam'],
                       help='Stage to search parameters for')
    parser.add_argument('--model', '-m', default=None,
                       help='Path to trained GP model (.pkl)')
    parser.add_argument('--target', '-t', type=float, default=0.75,
                       help='Target metric value')
    parser.add_argument('--n-candidates', '-n', type=int, default=10,
                       help='Number of candidates to return')
    parser.add_argument('--method', default='de',
                       choices=['de', 'local', 'random'],
                       help='Search method')
    parser.add_argument('--acquisition', '-a', default='ei',
                       choices=['ei', 'ucb', 'poi', 'mean'],
                       help='Acquisition function')
    parser.add_argument('--output', '-o', default=None,
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Load or train model
    if args.model and os.path.exists(args.model):
        surrogate = GPSurrogate.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print("Training new surrogate model...")
        extractor = DataExtractor(stage=args.stage)
        data = extractor.extract_all()
        surrogate = GPSurrogate()
        surrogate.fit(data)
    
    # Run inverse search
    searcher = InverseSearch(
        surrogate=surrogate,
        acquisition=args.acquisition,
    )
    
    print(f"\nSearching for parameters with target {args.stage} >= {args.target}")
    result = searcher.search(
        target_metric=args.target,
        n_candidates=args.n_candidates,
        method=args.method,
    )
    
    # Print results
    print(f"\nTop {len(result.candidates)} candidates:")
    for i, candidate in enumerate(result.candidates):
        print(f"\n  Candidate {i+1}:")
        print(f"    Predicted: {candidate.predicted_mean:.4f} ± {candidate.predicted_std:.4f}")
        print(f"    Acquisition: {candidate.acquisition_value:.4f}")
    
    print(f"\nBest candidate:")
    print(f"  Predicted: {result.best_candidate.predicted_mean:.4f}")
    print(f"  Parameters:")
    for k, v in list(result.best_candidate.params.items())[:10]:
        print(f"    {k}: {v:.4f}")
    
    # Save results
    if args.output:
        save_search_results(result, args.output)


if __name__ == '__main__':
    main()

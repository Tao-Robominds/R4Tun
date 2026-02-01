"""
Practical Usage Patterns for Surrogate Model

This module provides ready-to-use functions for the most common
surrogate model use cases.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .data_extractor import DataExtractor
from .gp_surrogate import GPSurrogate
from .inverse_search import InverseSearch


@dataclass
class ScreeningResult:
    """Result of pre-screening a parameter configuration."""
    params: Dict[str, float]
    predicted_mean: float
    predicted_std: float
    confidence_lower: float
    confidence_upper: float
    recommendation: str  # 'evaluate', 'skip', 'explore'


class SurrogateScreener:
    """
    Pre-screen parameter configurations before expensive full evaluation.
    
    Use Case: You have candidate parameters and want to know which ones
    are worth the 30+ second full pipeline evaluation.
    
    Example:
        screener = SurrogateScreener(stage='detection')
        screener.fit()
        
        # Check if params are worth evaluating
        result = screener.screen(params, target=0.70)
        if result.recommendation == 'evaluate':
            actual_score = run_full_pipeline(params)  # Only when promising
    """
    
    def __init__(
        self,
        stage: str = 'detection',
        results_dir: str = 'p4tun/bo/results',
    ):
        self.stage = stage
        self.results_dir = results_dir
        self.surrogate: Optional[GPSurrogate] = None
        self._is_fitted = False
    
    def fit(self, tunnel_ids: Optional[List[str]] = None) -> 'SurrogateScreener':
        """Train the screening model on existing BO data."""
        extractor = DataExtractor(results_dir=self.results_dir, stage=self.stage)
        data = extractor.extract_all(tunnel_ids=tunnel_ids)
        
        self.surrogate = GPSurrogate(kernel_type='matern')
        self.surrogate.fit(data, cv_folds=3, verbose=False)
        self._is_fitted = True
        
        print(f"Screener ready: trained on {data.n_samples} samples")
        return self
    
    def screen(
        self,
        params: Dict[str, float],
        target: float = 0.70,
        confidence: float = 0.95,
    ) -> ScreeningResult:
        """
        Screen a parameter configuration.
        
        Args:
            params: Parameter dictionary
            target: Target score threshold
            confidence: Confidence level for bounds
            
        Returns:
            ScreeningResult with recommendation
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        result = self.surrogate.predict_single(params)
        mean = float(result.mean[0])
        std = float(result.std[0])
        
        # Z-score for confidence level
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        
        lower = mean - z * std
        upper = mean + z * std
        
        # Decision logic
        if lower >= target * 0.9:
            # High confidence of meeting target
            recommendation = 'evaluate'
        elif upper < target * 0.8:
            # Unlikely to meet target
            recommendation = 'skip'
        elif std > 0.05:
            # High uncertainty - worth exploring
            recommendation = 'explore'
        else:
            # Borderline - might be worth evaluating
            recommendation = 'evaluate'
        
        return ScreeningResult(
            params=params,
            predicted_mean=mean,
            predicted_std=std,
            confidence_lower=lower,
            confidence_upper=upper,
            recommendation=recommendation,
        )
    
    def batch_screen(
        self,
        param_list: List[Dict[str, float]],
        target: float = 0.70,
    ) -> List[ScreeningResult]:
        """Screen multiple configurations and sort by potential."""
        results = [self.screen(p, target) for p in param_list]
        # Sort: evaluate first, then explore, then skip
        order = {'evaluate': 0, 'explore': 1, 'skip': 2}
        results.sort(key=lambda r: (order[r.recommendation], -r.predicted_mean))
        return results


class BOWarmStarter:
    """
    Generate warm-start points for Bayesian Optimization.
    
    Use Case: Starting BO on a new tunnel or with new constraints.
    Use surrogate to suggest promising initial points instead of random.
    
    Example:
        starter = BOWarmStarter(stage='detection')
        starter.fit()
        
        # Get initial points for BO
        initial_params = starter.suggest_initial_points(
            target=0.72,
            n_points=5,
        )
        
        # Use as x0 in skopt
        gp_minimize(objective, dimensions, x0=initial_params, ...)
    """
    
    def __init__(
        self,
        stage: str = 'detection',
        results_dir: str = 'p4tun/bo/results',
    ):
        self.stage = stage
        self.results_dir = results_dir
        self.surrogate: Optional[GPSurrogate] = None
        self.param_names: Optional[List[str]] = None
        self._is_fitted = False
    
    def fit(self, tunnel_ids: Optional[List[str]] = None) -> 'BOWarmStarter':
        """Train on existing BO data."""
        extractor = DataExtractor(results_dir=self.results_dir, stage=self.stage)
        data = extractor.extract_all(tunnel_ids=tunnel_ids)
        
        self.surrogate = GPSurrogate(kernel_type='matern')
        self.surrogate.fit(data, cv_folds=3, verbose=False)
        self.param_names = data.param_names
        self._is_fitted = True
        
        print(f"Warm starter ready: {data.n_samples} samples, {len(self.param_names)} params")
        return self
    
    def suggest_initial_points(
        self,
        target: float = 0.70,
        n_points: int = 5,
        diversity_weight: float = 0.3,
    ) -> List[List[float]]:
        """
        Suggest diverse initial points for BO.
        
        Args:
            target: Target score to aim for
            n_points: Number of initial points
            diversity_weight: Weight for diversity vs quality (0-1)
            
        Returns:
            List of parameter vectors (for use as x0 in gp_minimize)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        searcher = InverseSearch(
            self.surrogate,
            acquisition='ei',
            exploration_weight=diversity_weight,
        )
        
        # Search for diverse candidates
        result = searcher.search(
            target_metric=target,
            n_candidates=n_points * 3,  # Get more, then diversify
            method='de',
        )
        
        # Select diverse subset
        selected = self._select_diverse(result.candidates, n_points)
        
        # Convert to ordered vectors
        initial_points = []
        for candidate in selected:
            vec = [candidate.params.get(name, 0.0) for name in self.param_names]
            initial_points.append(vec)
        
        return initial_points
    
    def _select_diverse(self, candidates, n_select):
        """Select diverse candidates using greedy max-min distance."""
        if len(candidates) <= n_select:
            return candidates
        
        # Convert to vectors
        vectors = np.array([
            [c.params.get(n, 0.0) for n in self.param_names]
            for c in candidates
        ])
        
        # Normalize
        vectors = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-8)
        
        # Greedy selection
        selected_idx = [0]  # Start with best
        while len(selected_idx) < n_select:
            # Find point with maximum minimum distance to selected
            best_dist = -1
            best_idx = -1
            for i in range(len(candidates)):
                if i in selected_idx:
                    continue
                min_dist = min(
                    np.linalg.norm(vectors[i] - vectors[j])
                    for j in selected_idx
                )
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            selected_idx.append(best_idx)
        
        return [candidates[i] for i in selected_idx]


class SensitivityAnalyzer:
    """
    Analyze parameter sensitivity from trained surrogate.
    
    Use Case: Understand which parameters matter most for performance.
    Focus manual tuning or future BO on sensitive parameters.
    
    Example:
        analyzer = SensitivityAnalyzer(stage='detection')
        analyzer.fit()
        
        # Get sensitivity ranking
        sensitivity = analyzer.get_sensitivity_ranking()
        # {'angle_positive_min': 0.27, 'hough_vertical_threshold': 0.26, ...}
        
        # Get top N to focus on
        top_params = analyzer.get_top_params(n=5)
    """
    
    def __init__(
        self,
        stage: str = 'detection',
        results_dir: str = 'p4tun/bo/results',
    ):
        self.stage = stage
        self.results_dir = results_dir
        self.surrogate: Optional[GPSurrogate] = None
        self._is_fitted = False
    
    def fit(self, tunnel_ids: Optional[List[str]] = None) -> 'SensitivityAnalyzer':
        """Train on existing BO data."""
        extractor = DataExtractor(results_dir=self.results_dir, stage=self.stage)
        data = extractor.extract_all(tunnel_ids=tunnel_ids)
        
        self.surrogate = GPSurrogate(kernel_type='matern')
        self.surrogate.fit(data, cv_folds=3, verbose=False)
        self._is_fitted = True
        
        return self
    
    def get_sensitivity_ranking(self) -> Dict[str, float]:
        """Get parameter sensitivity ranking (higher = more important)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        return self.surrogate.get_feature_importance()
    
    def get_top_params(self, n: int = 5) -> List[str]:
        """Get top N most sensitive parameters."""
        ranking = self.get_sensitivity_ranking()
        sorted_params = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_params[:n]]
    
    def print_report(self):
        """Print formatted sensitivity report."""
        ranking = self.get_sensitivity_ranking()
        sorted_imp = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'='*60}")
        print(f"PARAMETER SENSITIVITY: {self.stage.upper()}")
        print(f"{'='*60}")
        print(f"{'Parameter':<35} {'Importance':>10} {'Bar':<20}")
        print(f"{'-'*60}")
        
        for name, imp in sorted_imp:
            bar = '█' * int(imp * 40)
            print(f"{name:<35} {imp:>9.1%} {bar}")


def quick_screen(params: Dict[str, float], stage: str = 'detection') -> str:
    """
    Quick one-liner to screen a parameter configuration.
    
    Returns: 'evaluate', 'skip', or 'explore'
    """
    screener = SurrogateScreener(stage=stage)
    screener.fit()
    result = screener.screen(params)
    return result.recommendation


# ============================================================================
# Command-line interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Practical surrogate usage')
    parser.add_argument('--mode', '-m', required=True,
                       choices=['screen', 'warmstart', 'sensitivity'],
                       help='Usage mode')
    parser.add_argument('--stage', '-s', default='detection',
                       choices=['detection', 'sam'],
                       help='Pipeline stage')
    parser.add_argument('--target', '-t', type=float, default=0.70,
                       help='Target mIoU')
    parser.add_argument('--n-points', '-n', type=int, default=5,
                       help='Number of points')
    
    args = parser.parse_args()
    
    if args.mode == 'sensitivity':
        analyzer = SensitivityAnalyzer(stage=args.stage)
        analyzer.fit()
        analyzer.print_report()
    
    elif args.mode == 'warmstart':
        starter = BOWarmStarter(stage=args.stage)
        starter.fit()
        points = starter.suggest_initial_points(
            target=args.target,
            n_points=args.n_points,
        )
        print(f"\nSuggested {len(points)} initial points for BO:")
        for i, p in enumerate(points):
            print(f"  {i+1}. {p[:5]}... ({len(p)} values)")
    
    elif args.mode == 'screen':
        print("Use SurrogateScreener in Python code with your params dict")


if __name__ == '__main__':
    main()

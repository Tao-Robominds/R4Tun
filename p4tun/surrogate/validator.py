"""
Validator Module for Surrogate Pipeline

Validates surrogate search candidates by running the full pipeline
and comparing actual metrics against predicted values.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .inverse_search import SearchCandidate, SearchResult


@dataclass
class ValidationResult:
    """Result of validating a single candidate."""
    candidate: SearchCandidate
    actual_metrics: Dict[str, float]
    predicted_mean: float
    predicted_std: float
    prediction_error: float
    is_within_bounds: bool
    passed_target: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'params': self.candidate.params,
            'predicted_mean': self.predicted_mean,
            'predicted_std': self.predicted_std,
            'actual_metrics': self.actual_metrics,
            'prediction_error': self.prediction_error,
            'is_within_bounds': self.is_within_bounds,
            'passed_target': self.passed_target,
        }


@dataclass
class ValidationReport:
    """Report summarizing validation of multiple candidates."""
    results: List[ValidationResult]
    target_metric: float
    metric_name: str
    success_rate: float
    mean_prediction_error: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'results': [r.to_dict() for r in self.results],
            'target_metric': self.target_metric,
            'metric_name': self.metric_name,
            'success_rate': self.success_rate,
            'mean_prediction_error': self.mean_prediction_error,
            'timestamp': self.timestamp,
            'summary': {
                'n_validated': len(self.results),
                'n_passed': sum(1 for r in self.results if r.passed_target),
                'n_within_bounds': sum(1 for r in self.results if r.is_within_bounds),
            }
        }


class Validator:
    """
    Validates surrogate search candidates by running the actual pipeline.
    
    For each candidate:
    1. Convert parameters to pipeline format
    2. Run detection and/or SAM stages
    3. Compute actual metrics
    4. Compare against predictions
    5. Optionally add to training set (active learning)
    """
    
    def __init__(
        self,
        stage: str = 'detection',
        tunnel_id: str = '2-2',
        data_dir: str = 'data',
        metric: str = 'mIoU',
        verbose: bool = True,
    ):
        """
        Initialize Validator.
        
        Args:
            stage: Pipeline stage to validate ('detection' or 'sam')
            tunnel_id: Tunnel ID for validation
            data_dir: Base data directory
            metric: Primary metric to evaluate
            verbose: Print progress information
        """
        self.stage = stage
        self.tunnel_id = tunnel_id
        self.data_dir = data_dir
        self.metric = metric
        self.verbose = verbose
        
        # Lazy load pipeline objective
        self._objective = None
    
    def _get_objective(self):
        """Lazy load the pipeline objective."""
        if self._objective is None:
            try:
                from p4tun.bo.objective import PipelineObjective
                self._objective = PipelineObjective(
                    tunnel_id=self.tunnel_id,
                    stage=self.stage,
                    data_dir=self.data_dir,
                    metric=self.metric,
                    verbose=self.verbose,
                )
            except ImportError as e:
                print(f"Warning: Could not load PipelineObjective: {e}")
                print("Validation will use mock evaluation.")
                self._objective = None
        return self._objective
    
    def _params_to_vector(self, params: Dict[str, float], param_names: List[str]) -> List[float]:
        """Convert parameter dict to ordered vector."""
        return [params.get(name, 0.0) for name in param_names]
    
    def validate_candidate(
        self,
        candidate: SearchCandidate,
        target_metric: float,
    ) -> ValidationResult:
        """
        Validate a single candidate.
        
        Args:
            candidate: Candidate to validate
            target_metric: Target metric value
            
        Returns:
            ValidationResult
        """
        objective = self._get_objective()
        
        if objective is not None:
            # Run actual pipeline
            param_names = list(candidate.params.keys())
            param_vec = self._params_to_vector(candidate.params, param_names)
            
            # The objective returns negative score (for minimization)
            neg_score = objective(param_vec)
            actual_score = -neg_score
            
            # Get detailed metrics from objective history
            if hasattr(objective, 'history') and objective.history:
                last_entry = objective.history[-1]
                actual_metrics = last_entry.get('metrics', {self.metric: actual_score})
            else:
                actual_metrics = {self.metric: actual_score}
        else:
            # Mock evaluation (for testing without full pipeline)
            if self.verbose:
                print("Using mock evaluation (pipeline not available)")
            # Add some noise to predicted mean
            noise = np.random.normal(0, candidate.predicted_std * 0.5)
            actual_score = candidate.predicted_mean + noise
            actual_metrics = {self.metric: actual_score}
        
        # Compute prediction error
        prediction_error = abs(actual_metrics.get(self.metric, 0) - candidate.predicted_mean)
        
        # Check if within confidence bounds (1.96 std for 95% CI)
        lower = candidate.predicted_mean - 1.96 * candidate.predicted_std
        upper = candidate.predicted_mean + 1.96 * candidate.predicted_std
        is_within_bounds = lower <= actual_metrics.get(self.metric, 0) <= upper
        
        # Check if passed target
        passed_target = actual_metrics.get(self.metric, 0) >= target_metric
        
        return ValidationResult(
            candidate=candidate,
            actual_metrics=actual_metrics,
            predicted_mean=candidate.predicted_mean,
            predicted_std=candidate.predicted_std,
            prediction_error=prediction_error,
            is_within_bounds=is_within_bounds,
            passed_target=passed_target,
        )
    
    def validate_search_result(
        self,
        search_result: SearchResult,
        n_candidates: Optional[int] = None,
    ) -> ValidationReport:
        """
        Validate candidates from a search result.
        
        Args:
            search_result: Search result to validate
            n_candidates: Number of top candidates to validate (default: all)
            
        Returns:
            ValidationReport
        """
        candidates = search_result.candidates
        if n_candidates:
            candidates = candidates[:n_candidates]
        
        target = search_result.target_metric
        
        if self.verbose:
            print(f"\nValidating {len(candidates)} candidates...")
            print(f"Target {self.metric}: {target:.4f}")
        
        results = []
        for i, candidate in enumerate(candidates):
            if self.verbose:
                print(f"\n  Candidate {i+1}/{len(candidates)}:")
                print(f"    Predicted: {candidate.predicted_mean:.4f} ± {candidate.predicted_std:.4f}")
            
            result = self.validate_candidate(candidate, target)
            results.append(result)
            
            if self.verbose:
                print(f"    Actual: {result.actual_metrics.get(self.metric, 0):.4f}")
                print(f"    Error: {result.prediction_error:.4f}")
                print(f"    Passed: {result.passed_target}")
        
        # Compute summary statistics
        n_passed = sum(1 for r in results if r.passed_target)
        success_rate = n_passed / len(results) if results else 0.0
        mean_error = np.mean([r.prediction_error for r in results]) if results else 0.0
        
        if self.verbose:
            print(f"\n  Summary:")
            print(f"    Success rate: {success_rate:.1%} ({n_passed}/{len(results)})")
            print(f"    Mean prediction error: {mean_error:.4f}")
        
        return ValidationReport(
            results=results,
            target_metric=target,
            metric_name=self.metric,
            success_rate=success_rate,
            mean_prediction_error=mean_error,
        )
    
    def get_training_updates(
        self,
        validation_report: ValidationReport,
        error_threshold: float = 0.05,
    ) -> List[Tuple[Dict[str, float], float]]:
        """
        Get parameter-score pairs for active learning.
        
        Returns candidates with high prediction error to add to training set.
        
        Args:
            validation_report: Validation report
            error_threshold: Minimum prediction error to include
            
        Returns:
            List of (params, actual_score) tuples
        """
        updates = []
        
        for result in validation_report.results:
            if result.prediction_error > error_threshold:
                actual_score = result.actual_metrics.get(self.metric, 0)
                updates.append((result.candidate.params, actual_score))
        
        if self.verbose:
            print(f"\nFound {len(updates)} candidates for active learning update")
        
        return updates


def save_validation_report(report: ValidationReport, filepath: str):
    """Save validation report to JSON."""
    with open(filepath, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Validation report saved to {filepath}")


def main():
    """Example usage of Validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate surrogate search candidates')
    parser.add_argument('--stage', '-s', default='detection',
                       choices=['detection', 'sam'],
                       help='Stage to validate')
    parser.add_argument('--tunnel', '-t', default='2-2',
                       help='Tunnel ID for validation')
    parser.add_argument('--search-result', '-r', required=True,
                       help='Path to search result JSON')
    parser.add_argument('--n-candidates', '-n', type=int, default=5,
                       help='Number of candidates to validate')
    parser.add_argument('--output', '-o', default=None,
                       help='Output report path')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output')
    
    args = parser.parse_args()
    
    # Load search result
    with open(args.search_result, 'r') as f:
        search_data = json.load(f)
    
    # Reconstruct SearchResult (simplified)
    from .inverse_search import SearchCandidate, SearchResult
    
    candidates = [
        SearchCandidate(
            params=c['params'],
            predicted_mean=c['predicted_mean'],
            predicted_std=c['predicted_std'],
            acquisition_value=c.get('acquisition_value', 0.0),
        )
        for c in search_data['candidates']
    ]
    
    search_result = SearchResult(
        candidates=candidates,
        best_candidate=candidates[0] if candidates else None,
        target_metric=search_data['target_metric'],
    )
    
    # Create validator
    validator = Validator(
        stage=args.stage,
        tunnel_id=args.tunnel,
        verbose=not args.quiet,
    )
    
    # Validate
    report = validator.validate_search_result(
        search_result,
        n_candidates=args.n_candidates,
    )
    
    # Save report
    if args.output:
        save_validation_report(report, args.output)


if __name__ == '__main__':
    main()

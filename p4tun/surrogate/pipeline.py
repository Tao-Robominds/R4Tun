"""
Surrogate Pipeline - End-to-End Interface

Combines data extraction, GP training, inverse search, and validation
into a single cohesive pipeline.

Usage:
    from p4tun.surrogate import SurrogatePipeline
    
    # Initialize pipeline for detection stage
    pipeline = SurrogatePipeline(stage='detection')
    
    # Extract data and train surrogate
    pipeline.fit()
    
    # Search for parameters achieving target mIoU
    candidates = pipeline.search(target_miou=0.75)
    
    # Validate top candidates
    report = pipeline.validate(candidates, n_validate=5)
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .data_extractor import DataExtractor, ExtractedData
from .gp_surrogate import GPSurrogate
from .inverse_search import InverseSearch, SearchResult, save_search_results
from .validator import Validator, ValidationReport, save_validation_report


@dataclass
class PipelineConfig:
    """Configuration for surrogate pipeline."""
    stage: str = 'detection'
    results_dir: str = 'p4tun/bo/results'
    output_dir: str = 'p4tun/surrogate/outputs'
    model_dir: str = 'p4tun/surrogate/models'
    
    # GP settings
    kernel_type: str = 'matern'
    
    # Search settings
    acquisition: str = 'ei'
    exploration_weight: float = 0.1
    search_method: str = 'de'
    
    # Validation settings
    validation_tunnel: str = '2-2'
    data_dir: str = 'data'
    
    # Active learning
    active_learning: bool = True
    error_threshold: float = 0.05


class SurrogatePipeline:
    """
    End-to-end surrogate model pipeline.
    
    Workflow:
    1. Extract training data from existing BO logs
    2. Train GP surrogate model
    3. Use inverse search to find parameters for target metrics
    4. Validate candidates with full pipeline
    5. Optionally update training data (active learning)
    """
    
    def __init__(
        self,
        stage: str = 'detection',
        config: Optional[PipelineConfig] = None,
        verbose: bool = True,
    ):
        """
        Initialize SurrogatePipeline.
        
        Args:
            stage: Pipeline stage ('detection' or 'sam')
            config: Pipeline configuration (uses defaults if not provided)
            verbose: Print progress information
        """
        self.stage = stage
        self.config = config or PipelineConfig(stage=stage)
        self.verbose = verbose
        
        # Components (initialized lazily)
        self._extractor: Optional[DataExtractor] = None
        self._surrogate: Optional[GPSurrogate] = None
        self._searcher: Optional[InverseSearch] = None
        self._validator: Optional[Validator] = None
        
        # State
        self._training_data: Optional[ExtractedData] = None
        self._is_fitted: bool = False
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.model_dir, exist_ok=True)
    
    @property
    def extractor(self) -> DataExtractor:
        if self._extractor is None:
            self._extractor = DataExtractor(
                results_dir=self.config.results_dir,
                stage=self.stage,
            )
        return self._extractor
    
    @property
    def surrogate(self) -> GPSurrogate:
        if self._surrogate is None:
            self._surrogate = GPSurrogate(
                kernel_type=self.config.kernel_type,
            )
        return self._surrogate
    
    @property
    def searcher(self) -> InverseSearch:
        if self._searcher is None:
            if not self._is_fitted:
                raise RuntimeError("Pipeline not fitted. Call fit() first.")
            self._searcher = InverseSearch(
                surrogate=self._surrogate,
                acquisition=self.config.acquisition,
                exploration_weight=self.config.exploration_weight,
            )
        return self._searcher
    
    @property
    def validator(self) -> Validator:
        if self._validator is None:
            self._validator = Validator(
                stage=self.stage,
                tunnel_id=self.config.validation_tunnel,
                data_dir=self.config.data_dir,
                verbose=self.verbose,
            )
        return self._validator
    
    def extract_data(
        self,
        tunnel_ids: Optional[List[str]] = None,
        use_key_params: bool = False,
    ) -> ExtractedData:
        """
        Extract training data from BO logs.
        
        Args:
            tunnel_ids: Optional list of tunnel IDs to include
            use_key_params: Extract only sensitive parameters
            
        Returns:
            ExtractedData object
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STEP 1: DATA EXTRACTION")
            print(f"{'='*60}")
        
        data = self.extractor.extract_all(tunnel_ids=tunnel_ids)
        
        if use_key_params:
            data = self.extractor.extract_key_parameters(data)
        
        self._training_data = data
        
        if self.verbose:
            print(f"\nExtracted {data.n_samples} samples, {data.n_features} features")
            print(f"Score range: [{data.y.min():.3f}, {data.y.max():.3f}]")
        
        return data
    
    def fit(
        self,
        data: Optional[ExtractedData] = None,
        tunnel_ids: Optional[List[str]] = None,
        use_key_params: bool = False,
        cv_folds: int = 5,
    ) -> 'SurrogatePipeline':
        """
        Extract data and train the surrogate model.
        
        Args:
            data: Pre-extracted training data (extracts if not provided)
            tunnel_ids: Tunnel IDs to include
            use_key_params: Use only sensitive parameters
            cv_folds: Cross-validation folds
            
        Returns:
            Self for method chaining
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SURROGATE PIPELINE: {self.stage.upper()}")
            print(f"{'='*60}")
        
        # Extract data if not provided
        if data is None:
            data = self.extract_data(
                tunnel_ids=tunnel_ids,
                use_key_params=use_key_params,
            )
        else:
            self._training_data = data
        
        # Train surrogate
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STEP 2: GP SURROGATE TRAINING")
            print(f"{'='*60}")
        
        self._surrogate = GPSurrogate(kernel_type=self.config.kernel_type)
        self._surrogate.fit(data, cv_folds=cv_folds, verbose=self.verbose)
        
        self._is_fitted = True
        
        # Reset searcher to use new surrogate
        self._searcher = None
        
        return self
    
    def search(
        self,
        target_miou: float = 0.75,
        n_candidates: int = 10,
        n_restarts: int = 20,
        method: Optional[str] = None,
    ) -> SearchResult:
        """
        Search for parameters achieving target metric.
        
        Args:
            target_miou: Target mIoU value
            n_candidates: Number of candidates to return
            n_restarts: Number of optimization restarts
            method: Search method ('de', 'local', 'random')
            
        Returns:
            SearchResult with candidate configurations
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STEP 3: INVERSE SEARCH")
            print(f"{'='*60}")
            print(f"Target mIoU: {target_miou}")
        
        method = method or self.config.search_method
        
        result = self.searcher.search(
            target_metric=target_miou,
            n_candidates=n_candidates,
            n_restarts=n_restarts,
            method=method,
        )
        
        if self.verbose:
            print(f"\nFound {len(result.candidates)} candidates")
            print(f"Best predicted: {result.best_candidate.predicted_mean:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(
            self.config.output_dir,
            f'{self.stage}_search_{timestamp}.json'
        )
        save_search_results(result, output_path)
        
        return result
    
    def validate(
        self,
        search_result: SearchResult,
        n_validate: int = 5,
    ) -> ValidationReport:
        """
        Validate top candidates from search result.
        
        Args:
            search_result: Search result to validate
            n_validate: Number of candidates to validate
            
        Returns:
            ValidationReport
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STEP 4: VALIDATION")
            print(f"{'='*60}")
        
        report = self.validator.validate_search_result(
            search_result,
            n_candidates=n_validate,
        )
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(
            self.config.output_dir,
            f'{self.stage}_validation_{timestamp}.json'
        )
        save_validation_report(report, output_path)
        
        # Active learning update
        if self.config.active_learning:
            updates = self.validator.get_training_updates(
                report,
                error_threshold=self.config.error_threshold,
            )
            if updates and self.verbose:
                print(f"\n{len(updates)} candidates flagged for training update")
        
        return report
    
    def run(
        self,
        target_miou: float = 0.75,
        n_candidates: int = 10,
        n_validate: int = 5,
        tunnel_ids: Optional[List[str]] = None,
    ) -> Tuple[SearchResult, ValidationReport]:
        """
        Run the complete pipeline.
        
        Args:
            target_miou: Target mIoU value
            n_candidates: Number of candidates to search
            n_validate: Number of candidates to validate
            tunnel_ids: Tunnel IDs to include in training
            
        Returns:
            Tuple of (SearchResult, ValidationReport)
        """
        # Fit if not already fitted
        if not self._is_fitted:
            self.fit(tunnel_ids=tunnel_ids)
        
        # Search
        search_result = self.search(
            target_miou=target_miou,
            n_candidates=n_candidates,
        )
        
        # Validate
        validation_report = self.validate(
            search_result,
            n_validate=n_validate,
        )
        
        # Summary
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PIPELINE COMPLETE")
            print(f"{'='*60}")
            print(f"Stage: {self.stage}")
            print(f"Target mIoU: {target_miou}")
            print(f"Best predicted: {search_result.best_candidate.predicted_mean:.4f}")
            print(f"Validation success rate: {validation_report.success_rate:.1%}")
            print(f"Mean prediction error: {validation_report.mean_prediction_error:.4f}")
        
        return search_result, validation_report
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the trained surrogate model."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")
        
        if filepath is None:
            filepath = os.path.join(
                self.config.model_dir,
                f'{self.stage}_gp.pkl'
            )
        
        self._surrogate.save(filepath)
    
    def load_model(self, filepath: Optional[str] = None) -> 'SurrogatePipeline':
        """Load a pre-trained surrogate model."""
        if filepath is None:
            filepath = os.path.join(
                self.config.model_dir,
                f'{self.stage}_gp.pkl'
            )
        
        self._surrogate = GPSurrogate.load(filepath)
        self._is_fitted = True
        self._searcher = None  # Reset to use loaded surrogate
        
        if self.verbose:
            print(f"Loaded model from {filepath}")
        
        return self
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained surrogate."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")
        return self._surrogate.get_feature_importance()
    
    def predict(self, params: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict score for given parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Tuple of (predicted_mean, predicted_std)
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")
        
        result = self._surrogate.predict_single(params)
        return float(result.mean[0]), float(result.std[0])


def main():
    """Main entry point for running the surrogate pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='P4Tun Surrogate Model Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run detection surrogate pipeline targeting mIoU >= 0.75
  python -m p4tun.surrogate.pipeline --stage detection --target 0.75
  
  # Run SAM surrogate pipeline with validation
  python -m p4tun.surrogate.pipeline --stage sam --target 0.78 --n-validate 5
  
  # Load existing model and search only
  python -m p4tun.surrogate.pipeline --stage detection --load-model --target 0.80
"""
    )
    
    parser.add_argument('--stage', '-s', default='detection',
                       choices=['detection', 'sam'],
                       help='Pipeline stage (default: detection)')
    parser.add_argument('--target', '-t', type=float, default=0.75,
                       help='Target mIoU value (default: 0.75)')
    parser.add_argument('--n-candidates', '-n', type=int, default=10,
                       help='Number of candidates (default: 10)')
    parser.add_argument('--n-validate', type=int, default=5,
                       help='Number to validate (default: 5)')
    parser.add_argument('--tunnel', nargs='+', default=None,
                       help='Tunnel IDs to include')
    parser.add_argument('--validation-tunnel', default='2-2',
                       help='Tunnel ID for validation (default: 2-2)')
    parser.add_argument('--load-model', action='store_true',
                       help='Load existing model instead of training')
    parser.add_argument('--model-path', default=None,
                       help='Path to model file')
    parser.add_argument('--search-only', action='store_true',
                       help='Skip validation (search only)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output')
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig(
        stage=args.stage,
        validation_tunnel=args.validation_tunnel,
    )
    
    # Create pipeline
    pipeline = SurrogatePipeline(
        stage=args.stage,
        config=config,
        verbose=not args.quiet,
    )
    
    # Load or train model
    if args.load_model:
        pipeline.load_model(args.model_path)
    else:
        pipeline.fit(tunnel_ids=args.tunnel)
        pipeline.save_model()
    
    # Search
    search_result = pipeline.search(
        target_miou=args.target,
        n_candidates=args.n_candidates,
    )
    
    # Validate (unless skipped)
    if not args.search_only:
        validation_report = pipeline.validate(
            search_result,
            n_validate=args.n_validate,
        )
    
    # Print feature importance
    if not args.quiet:
        print("\nFeature Importance (top 10):")
        importance = pipeline.get_feature_importance()
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:10]:
            print(f"  {name}: {imp:.4f}")


if __name__ == '__main__':
    main()

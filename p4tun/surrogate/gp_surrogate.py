"""
Gaussian Process Surrogate Model for P4Tun Pipeline

Trains a GP model on extracted BO data to enable fast parameter search
without running the full pipeline.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel as C
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from .data_extractor import ExtractedData


@dataclass
class PredictionResult:
    """Container for GP prediction results."""
    mean: np.ndarray
    std: np.ndarray
    
    @property
    def lower_bound(self, alpha: float = 1.96) -> np.ndarray:
        """Lower confidence bound (default 95%)."""
        return self.mean - alpha * self.std
    
    @property
    def upper_bound(self, alpha: float = 1.96) -> np.ndarray:
        """Upper confidence bound (default 95%)."""
        return self.mean + alpha * self.std
    
    def expected_improvement(
        self,
        best_observed: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        """
        Compute Expected Improvement acquisition function.
        
        Args:
            best_observed: Best observed score so far
            xi: Exploration-exploitation trade-off parameter
            
        Returns:
            Expected improvement values
        """
        from scipy.stats import norm
        
        with np.errstate(divide='warn'):
            improvement = self.mean - best_observed - xi
            Z = improvement / self.std
            ei = improvement * norm.cdf(Z) + self.std * norm.pdf(Z)
            ei[self.std == 0.0] = 0.0
        
        return ei


class GPSurrogate:
    """
    Gaussian Process surrogate model for fast parameter evaluation.
    
    Uses the same GP framework as the original BO but trained on
    historical data for inverse search.
    """
    
    def __init__(
        self,
        kernel_type: str = 'matern',
        normalize_y: bool = True,
        n_restarts: int = 10,
        alpha: float = 1e-6,
        random_state: int = 42,
    ):
        """
        Initialize GP Surrogate.
        
        Args:
            kernel_type: Kernel type ('rbf', 'matern', 'matern52')
            normalize_y: Whether to normalize target values
            n_restarts: Number of optimizer restarts for hyperparameter tuning
            alpha: Value added to diagonal for numerical stability
            random_state: Random seed
        """
        self.kernel_type = kernel_type
        self.normalize_y = normalize_y
        self.n_restarts = n_restarts
        self.alpha = alpha
        self.random_state = random_state
        
        # Will be set during training
        self.gp: Optional[GaussianProcessRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.param_names: Optional[List[str]] = None
        self.training_data: Optional[ExtractedData] = None
        self.param_bounds: Optional[Dict[str, Tuple[float, float]]] = None
        
        # Training statistics
        self.cv_scores: Optional[np.ndarray] = None
        self.log_marginal_likelihood: Optional[float] = None
    
    def _build_kernel(self, n_features: int):
        """
        Build GP kernel based on kernel_type.
        
        Args:
            n_features: Number of input features
            
        Returns:
            Kernel object
        """
        # Length scale bounds relative to feature space
        length_scale_bounds = (1e-2, 1e3)
        
        if self.kernel_type == 'rbf':
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=np.ones(n_features),
                length_scale_bounds=length_scale_bounds
            )
        elif self.kernel_type == 'matern':
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=length_scale_bounds,
                nu=2.5  # Matern 5/2 kernel
            )
        elif self.kernel_type == 'matern52':
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=length_scale_bounds,
                nu=2.5
            )
        else:
            # Default: Matern with white noise
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=length_scale_bounds,
                nu=2.5
            ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        return kernel
    
    def fit(
        self,
        data: ExtractedData,
        cv_folds: int = 5,
        verbose: bool = True,
    ) -> 'GPSurrogate':
        """
        Train the GP surrogate model.
        
        Args:
            data: Extracted training data
            cv_folds: Number of cross-validation folds
            verbose: Print training progress
            
        Returns:
            Self for method chaining
        """
        self.training_data = data
        self.param_names = data.param_names
        
        X = data.X.copy()
        y = data.y.copy()
        
        if verbose:
            print(f"Training GP surrogate on {len(y)} samples, {X.shape[1]} features")
        
        # Scale input features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute parameter bounds from training data
        self.param_bounds = {}
        for i, name in enumerate(self.param_names):
            self.param_bounds[name] = (float(X[:, i].min()), float(X[:, i].max()))
        
        # Build kernel
        kernel = self._build_kernel(X.shape[1])
        
        # Create GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=self.normalize_y,
            alpha=self.alpha,
            random_state=self.random_state,
        )
        
        # Fit the model
        self.gp.fit(X_scaled, y)
        self.log_marginal_likelihood = self.gp.log_marginal_likelihood_value_
        
        if verbose:
            print(f"  Log marginal likelihood: {self.log_marginal_likelihood:.3f}")
            print(f"  Kernel: {self.gp.kernel_}")
        
        # Cross-validation
        if cv_folds > 1 and len(y) >= cv_folds:
            self.cv_scores = cross_val_score(
                self.gp, X_scaled, y,
                cv=cv_folds,
                scoring='r2'
            )
            if verbose:
                print(f"  CV R² score: {self.cv_scores.mean():.3f} ± {self.cv_scores.std():.3f}")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> PredictionResult:
        """
        Predict scores for parameter vectors.
        
        Args:
            X: Parameter vectors (n_samples, n_features)
            return_std: Whether to return standard deviation
            
        Returns:
            PredictionResult with mean and std
        """
        if self.gp is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Ensure 2D
        X = np.atleast_2d(X)
        
        # Scale inputs
        X_scaled = self.scaler.transform(X)
        
        # Predict
        if return_std:
            mean, std = self.gp.predict(X_scaled, return_std=True)
        else:
            mean = self.gp.predict(X_scaled)
            std = np.zeros_like(mean)
        
        return PredictionResult(mean=mean, std=std)
    
    def predict_single(
        self,
        params: Dict[str, float],
    ) -> PredictionResult:
        """
        Predict score for a single parameter dictionary.
        
        Args:
            params: Dictionary of parameter name -> value
            
        Returns:
            PredictionResult
        """
        X = np.array([[params.get(name, 0.0) for name in self.param_names]])
        return self.predict(X)
    
    def get_best_observed(self) -> Tuple[np.ndarray, float]:
        """
        Get the best observed parameters and score from training data.
        
        Returns:
            Tuple of (best_params, best_score)
        """
        # Try from training data first
        if self.training_data is not None:
            best_idx = np.argmax(self.training_data.y)
            best_params = self.training_data.X[best_idx]
            best_score = self.training_data.y[best_idx]
            return best_params, best_score
        
        # Try from loaded model cache
        if hasattr(self, '_best_params') and self._best_params is not None:
            return np.array(self._best_params), self._best_score
        
        raise RuntimeError("Model not fitted and no cached best observed")
    
    def save(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Output file path (.pkl)
        """
        # Store best observed for inverse search
        best_params = None
        best_score = None
        if self.training_data is not None:
            best_idx = np.argmax(self.training_data.y)
            best_params = self.training_data.X[best_idx].tolist()
            best_score = float(self.training_data.y[best_idx])
        
        save_dict = {
            'gp': self.gp,
            'scaler': self.scaler,
            'param_names': self.param_names,
            'param_bounds': self.param_bounds,
            'kernel_type': self.kernel_type,
            'normalize_y': self.normalize_y,
            'cv_scores': self.cv_scores,
            'log_marginal_likelihood': self.log_marginal_likelihood,
            'training_metadata': self.training_data.metadata if self.training_data else None,
            'best_params': best_params,
            'best_score': best_score,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'GPSurrogate':
        """
        Load a trained model.
        
        Args:
            filepath: Path to saved model (.pkl)
            
        Returns:
            Loaded GPSurrogate instance
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        instance = cls(
            kernel_type=save_dict['kernel_type'],
            normalize_y=save_dict['normalize_y'],
        )
        instance.gp = save_dict['gp']
        instance.scaler = save_dict['scaler']
        instance.param_names = save_dict['param_names']
        instance.param_bounds = save_dict['param_bounds']
        instance.cv_scores = save_dict['cv_scores']
        instance.log_marginal_likelihood = save_dict['log_marginal_likelihood']
        
        # Load best observed for inverse search
        instance._best_params = save_dict.get('best_params')
        instance._best_score = save_dict.get('best_score')
        
        return instance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Estimate feature importance from GP length scales.
        Shorter length scale = more important feature.
        
        Returns:
            Dictionary of param_name -> importance score
        """
        if self.gp is None or self.param_names is None:
            raise RuntimeError("Model not fitted")
        
        # Get length scales from kernel
        # For Matern kernel: kernel.k2.length_scale
        # For RBF: kernel.k2.length_scale
        try:
            if hasattr(self.gp.kernel_, 'k2'):
                length_scales = self.gp.kernel_.k2.length_scale
            else:
                length_scales = np.ones(len(self.param_names))
        except AttributeError:
            length_scales = np.ones(len(self.param_names))
        
        # Ensure array
        length_scales = np.atleast_1d(length_scales)
        if len(length_scales) == 1:
            length_scales = np.repeat(length_scales, len(self.param_names))
        
        # Importance inversely proportional to length scale
        importance = 1.0 / (length_scales + 1e-8)
        importance = importance / importance.sum()  # Normalize
        
        return dict(zip(self.param_names, importance))


def main():
    """Example usage of GPSurrogate."""
    import argparse
    from .data_extractor import DataExtractor
    
    parser = argparse.ArgumentParser(description='Train GP surrogate model')
    parser.add_argument('--stage', '-s', default='detection',
                       choices=['detection', 'sam'],
                       help='Stage to train model for')
    parser.add_argument('--results-dir', default='p4tun/bo/results',
                       help='Directory containing BO results')
    parser.add_argument('--output', '-o', default=None,
                       help='Output model file path (.pkl)')
    parser.add_argument('--kernel', '-k', default='matern',
                       choices=['rbf', 'matern', 'matern52'],
                       help='GP kernel type')
    
    args = parser.parse_args()
    
    # Extract data
    extractor = DataExtractor(results_dir=args.results_dir, stage=args.stage)
    data = extractor.extract_all()
    
    # Train model
    surrogate = GPSurrogate(kernel_type=args.kernel)
    surrogate.fit(data, verbose=True)
    
    # Show feature importance
    importance = surrogate.get_feature_importance()
    print("\nTop 10 most important parameters:")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_imp[:10]:
        print(f"  {name}: {imp:.4f}")
    
    # Save model
    if args.output:
        surrogate.save(args.output)
    else:
        default_path = f'p4tun/surrogate/models/{args.stage}_gp.pkl'
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        surrogate.save(default_path)


if __name__ == '__main__':
    main()

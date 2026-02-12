#!/usr/bin/env python3
"""
Train mIoU Predictors Using Validated Intrinsic Metrics

This script trains two separate predictors:
1. Simple patterns (1-4, 2-2, 3-1): f(det_*, sam_mask_fill_rate)
2. Complex patterns (4-1, 5-1): f(sam_complex_*)

Based on validated metrics from COMPREHENSIVE_METRICS_VALIDATION.md
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# =============================================================================
# Feature Definitions (from validation study)
# =============================================================================

SIMPLE_FEATURES = [
    'det_midpoint_ratio',       # r=0.55 - detection method quality
    'det_real_detection_ratio', # r=0.50 - actual vs fallback detections
    'det_x_spacing_cv',         # r=-0.47 - spacing uniformity
    'sam_mask_fill_rate',       # r=0.46 - segmentation coverage
    'det_y_std',                # r=-0.45 - Y position variance
]

COMPLEX_FEATURES = [
    'complex_sam_ring_completeness',  # r=0.80 - rings with >50% coverage
    'complex_sam_segment_height_cv',  # r=0.80 - height consistency
    'complex_sam_segment_width_cv',   # r=-0.40 - width consistency
    'complex_sam_segment_area_cv',    # r=-0.40 - size uniformity
    'complex_sam_ring_coverage_cv',   # r=-0.40 - cross-ring consistency
    'complex_sam_aspect_ratio_mean',  # r=0.40 - shape ratio
]

# =============================================================================
# Data Loading
# =============================================================================

def load_training_data():
    """Load comprehensive metrics data."""
    data_file = project_root / 'bo4tun' / 'training' / 'comprehensive_all_metrics.csv'
    
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        print("Run comprehensive_ablation.py first to generate the data")
        return None
    
    df = pd.read_csv(data_file)
    
    # Filter for valid mIoU
    df = df[df['mIoU'].notna() & (df['mIoU'] > 0)]
    
    return df


def prepare_simple_data(df):
    """Prepare data for simple pattern predictor."""
    simple_df = df[df['pattern_type'] == 'simple'].copy()
    
    # Check feature availability
    available = [f for f in SIMPLE_FEATURES if f in simple_df.columns]
    missing = [f for f in SIMPLE_FEATURES if f not in simple_df.columns]
    
    if missing:
        print(f"Warning: Missing simple features: {missing}")
    
    # Drop rows with NaN in features
    valid_df = simple_df.dropna(subset=available + ['mIoU'])
    
    return valid_df, available


def prepare_complex_data(df):
    """Prepare data for complex pattern predictor."""
    complex_df = df[df['pattern_type'] == 'complex'].copy()
    
    # Check feature availability
    available = [f for f in COMPLEX_FEATURES if f in complex_df.columns]
    missing = [f for f in COMPLEX_FEATURES if f not in complex_df.columns]
    
    if missing:
        print(f"Warning: Missing complex features: {missing}")
    
    # Drop rows with NaN in features
    valid_df = complex_df.dropna(subset=available + ['mIoU'])
    
    return valid_df, available


# =============================================================================
# Model Training
# =============================================================================

def train_predictor(X, y, alpha=1.0):
    """Train Ridge regression predictor."""
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def evaluate_predictor(model, X, y, feature_names):
    """Evaluate predictor with leave-one-out CV."""
    # In-sample predictions
    y_pred_train = model.predict(X)
    
    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    y_pred_cv = cross_val_predict(model, X, y, cv=loo)
    
    # Metrics
    results = {
        'n_samples': len(y),
        'n_features': len(feature_names),
        'features': feature_names,
        
        # In-sample metrics
        'train_mae': float(mean_absolute_error(y, y_pred_train)),
        'train_r2': float(r2_score(y, y_pred_train)),
        'train_spearman': float(spearmanr(y, y_pred_train)[0]),
        
        # LOO-CV metrics
        'cv_mae': float(mean_absolute_error(y, y_pred_cv)),
        'cv_r2': float(r2_score(y, y_pred_cv)),
        'cv_spearman': float(spearmanr(y, y_pred_cv)[0]),
        
        # Feature importance (coefficients)
        'coefficients': dict(zip(feature_names, model.coef_.tolist())),
        'intercept': float(model.intercept_),
    }
    
    return results


# =============================================================================
# Main Training
# =============================================================================

def main():
    print("=" * 70)
    print("TRAINING mIoU PREDICTORS WITH VALIDATED INTRINSIC METRICS")
    print("=" * 70)
    
    # Load data
    df = load_training_data()
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} samples")
    print(f"  Simple patterns: {len(df[df['pattern_type'] == 'simple'])}")
    print(f"  Complex patterns: {len(df[df['pattern_type'] == 'complex'])}")
    
    output_dir = project_root / 'bo4tun' / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # Train Simple Pattern Predictor
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. SIMPLE PATTERN PREDICTOR (1-4, 2-2, 3-1)")
    print("=" * 70)
    
    simple_df, simple_features = prepare_simple_data(df)
    print(f"\nTraining samples: {len(simple_df)}")
    print(f"Features: {simple_features}")
    
    if len(simple_df) >= 3:
        X_simple = simple_df[simple_features].values
        y_simple = simple_df['mIoU'].values
        
        # Train model
        simple_model = train_predictor(X_simple, y_simple, alpha=1.0)
        
        # Evaluate
        simple_results = evaluate_predictor(simple_model, X_simple, y_simple, simple_features)
        results['simple'] = simple_results
        
        print(f"\n--- Simple Predictor Results ---")
        print(f"In-sample:  MAE={simple_results['train_mae']:.4f}, Spearman={simple_results['train_spearman']:.4f}")
        print(f"LOO-CV:     MAE={simple_results['cv_mae']:.4f}, Spearman={simple_results['cv_spearman']:.4f}")
        print(f"\nCoefficients:")
        for feat, coef in simple_results['coefficients'].items():
            print(f"  {feat}: {coef:+.4f}")
        print(f"  intercept: {simple_results['intercept']:.4f}")
        
        # Save model
        model_path = output_dir / 'simple_miou_predictor.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': simple_model,
                'features': simple_features,
                'pattern_type': 'simple',
                'trained_on': datetime.now().isoformat(),
            }, f)
        print(f"\nModel saved: {model_path}")
    else:
        print("Insufficient data for simple predictor")
    
    # =========================================================================
    # Train Complex Pattern Predictor
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. COMPLEX PATTERN PREDICTOR (4-1, 5-1)")
    print("=" * 70)
    
    complex_df, complex_features = prepare_complex_data(df)
    print(f"\nTraining samples: {len(complex_df)}")
    print(f"Features: {complex_features}")
    
    if len(complex_df) >= 3:
        X_complex = complex_df[complex_features].values
        y_complex = complex_df['mIoU'].values
        
        # Train model
        complex_model = train_predictor(X_complex, y_complex, alpha=1.0)
        
        # Evaluate
        complex_results = evaluate_predictor(complex_model, X_complex, y_complex, complex_features)
        results['complex'] = complex_results
        
        print(f"\n--- Complex Predictor Results ---")
        print(f"In-sample:  MAE={complex_results['train_mae']:.4f}, Spearman={complex_results['train_spearman']:.4f}")
        print(f"LOO-CV:     MAE={complex_results['cv_mae']:.4f}, Spearman={complex_results['cv_spearman']:.4f}")
        print(f"\nCoefficients:")
        for feat, coef in complex_results['coefficients'].items():
            print(f"  {feat}: {coef:+.4f}")
        print(f"  intercept: {complex_results['intercept']:.4f}")
        
        # Save model
        model_path = output_dir / 'complex_miou_predictor.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': complex_model,
                'features': complex_features,
                'pattern_type': 'complex',
                'trained_on': datetime.now().isoformat(),
            }, f)
        print(f"\nModel saved: {model_path}")
    else:
        print("Insufficient data for complex predictor")
    
    # =========================================================================
    # Save Summary
    # =========================================================================
    summary_path = output_dir / 'predictor_training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    
    # =========================================================================
    # Print Formula
    # =========================================================================
    print("\n" + "=" * 70)
    print("PREDICTOR FORMULAS")
    print("=" * 70)
    
    if 'simple' in results:
        print("\n--- Simple Patterns ---")
        print("mIoU = ", end="")
        terms = []
        for feat, coef in results['simple']['coefficients'].items():
            sign = '+' if coef >= 0 else ''
            terms.append(f"{sign}{coef:.4f}*{feat}")
        terms.append(f"{results['simple']['intercept']:+.4f}")
        print("\n      ".join(terms[:3]))
        if len(terms) > 3:
            print("      " + "\n      ".join(terms[3:]))
    
    if 'complex' in results:
        print("\n--- Complex Patterns ---")
        print("mIoU = ", end="")
        terms = []
        for feat, coef in results['complex']['coefficients'].items():
            sign = '+' if coef >= 0 else ''
            terms.append(f"{sign}{coef:.4f}*{feat}")
        terms.append(f"{results['complex']['intercept']:+.4f}")
        print("\n      ".join(terms[:3]))
        if len(terms) > 3:
            print("      " + "\n      ".join(terms[3:]))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()

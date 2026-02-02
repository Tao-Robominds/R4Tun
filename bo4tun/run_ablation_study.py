"""
Ablation Study for Intrinsic Metrics
=====================================
Identifies which intrinsic metrics are most important for predicting mIoU.

Method: Leave-One-Out Feature Ablation
- Train baseline model with all features
- Remove each feature one at a time
- Measure increase in MAE (ΔMAE)
- Higher ΔMAE = more important feature
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import json
from pathlib import Path

# Configuration
TRAINING_DATA_PATH = Path(__file__).parent / "training" / "intrinsic_training_data.csv"
OUTPUT_PATH = Path(__file__).parent / "training" / "ablation_results.json"

# Pattern classification
SIMPLE_TUNNELS = ['1-4', '2-2', '3-1']
COMPLEX_TUNNELS = ['4-1', '5-1']

# Candidate features for each pattern type
SIMPLE_DETECTION_FEATURES = [
    'det_midpoint_ratio',
    'det_real_detection_ratio', 
    'det_k_count_match',
    'det_assume_default_ratio',
    'det_x_spacing_cv',
]

SIMPLE_SAM_FEATURES = [
    'sam_mask_fill_rate',
    'sam_segment_count_match',
]

COMPLEX_SAM_PARAMS = [
    # These would come from parameters, not intrinsic metrics
    # For now, we'll use detection metrics as available
    'det_k_count',
    'det_x_spacing_cv',
    'det_y_range',
    'det_y_std',
]


def load_data():
    """Load and prepare training data."""
    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"Loaded {len(df)} samples from {TRAINING_DATA_PATH}")
    print(f"Tunnels: {df['tunnel_id'].unique()}")
    print(f"mIoU range: {df['mIoU'].min():.3f} - {df['mIoU'].max():.3f}")
    return df


def separate_patterns(df):
    """Separate into simple and complex patterns."""
    simple_df = df[df['tunnel_id'].isin(SIMPLE_TUNNELS)].copy()
    complex_df = df[df['tunnel_id'].isin(COMPLEX_TUNNELS)].copy()
    
    print(f"\nSimple patterns: {len(simple_df)} samples ({simple_df['tunnel_id'].unique()})")
    print(f"Complex patterns: {len(complex_df)} samples ({complex_df['tunnel_id'].unique()})")
    
    return simple_df, complex_df


def get_available_features(df, candidate_features):
    """Get features that exist and have non-null values."""
    available = []
    for feat in candidate_features:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            if non_null > 0:
                available.append(feat)
                print(f"  {feat}: {non_null}/{len(df)} non-null")
            else:
                print(f"  {feat}: ALL NULL - skipping")
        else:
            print(f"  {feat}: NOT IN DATA - skipping")
    return available


def run_ablation(df, features, target='mIoU', alpha=1.0):
    """
    Run leave-one-out feature ablation.
    
    Returns dict with feature importance results.
    """
    # Prepare data - drop rows with any null in features or target
    cols_needed = features + [target]
    clean_df = df[cols_needed].dropna()
    
    if len(clean_df) < 5:
        print(f"  WARNING: Only {len(clean_df)} samples after dropping nulls - too few for reliable analysis")
        return None
    
    X = clean_df[features].values
    y = clean_df[target].values
    
    print(f"\n  Using {len(clean_df)} samples with {len(features)} features")
    
    # Baseline: train with all features
    baseline_model = Ridge(alpha=alpha)
    baseline_model.fit(X, y)
    baseline_pred = baseline_model.predict(X)
    baseline_mae = mean_absolute_error(y, baseline_pred)
    baseline_spearman, _ = spearmanr(y, baseline_pred)
    
    print(f"  Baseline MAE: {baseline_mae:.4f}")
    print(f"  Baseline Spearman: {baseline_spearman:.4f}")
    
    # Feature coefficients (for interpretability)
    coefficients = dict(zip(features, baseline_model.coef_))
    
    # Leave-one-out ablation
    importance_results = {}
    
    for i, remove_feat in enumerate(features):
        subset_features = [f for f in features if f != remove_feat]
        X_subset = clean_df[subset_features].values
        
        model = Ridge(alpha=alpha)
        model.fit(X_subset, y)
        pred = model.predict(X_subset)
        mae = mean_absolute_error(y, pred)
        spearman, _ = spearmanr(y, pred)
        
        delta_mae = mae - baseline_mae
        delta_spearman = baseline_spearman - spearman  # Positive = feature helps
        
        # Classify importance
        if delta_mae > 0.03:
            importance = 'HIGH'
        elif delta_mae > 0.01:
            importance = 'MEDIUM'
        else:
            importance = 'LOW'
        
        importance_results[remove_feat] = {
            'mae_without': round(mae, 4),
            'delta_mae': round(delta_mae, 4),
            'spearman_without': round(spearman, 4),
            'delta_spearman': round(delta_spearman, 4),
            'coefficient': round(coefficients[remove_feat], 4),
            'importance': importance,
        }
    
    return {
        'n_samples': len(clean_df),
        'n_features': len(features),
        'baseline_mae': round(baseline_mae, 4),
        'baseline_spearman': round(baseline_spearman, 4),
        'coefficients': {k: round(v, 4) for k, v in coefficients.items()},
        'feature_importance': importance_results,
    }


def print_importance_table(results, title):
    """Print feature importance as a formatted table."""
    if results is None:
        print(f"\n{title}: SKIPPED (insufficient data)")
        return
    
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Samples: {results['n_samples']}, Features: {results['n_features']}")
    print(f"Baseline MAE: {results['baseline_mae']:.4f}, Spearman: {results['baseline_spearman']:.4f}")
    print()
    
    # Sort by delta_mae (most important first)
    sorted_features = sorted(
        results['feature_importance'].items(),
        key=lambda x: x[1]['delta_mae'],
        reverse=True
    )
    
    print(f"{'Feature':<30} {'ΔMAE':>8} {'ΔSpearman':>10} {'Coef':>10} {'Importance':>12}")
    print("-" * 70)
    
    for feat, info in sorted_features:
        print(f"{feat:<30} {info['delta_mae']:>+8.4f} {info['delta_spearman']:>+10.4f} {info['coefficient']:>+10.4f} {info['importance']:>12}")
    
    print()
    
    # Recommendations
    high_importance = [f for f, i in sorted_features if i['importance'] == 'HIGH']
    medium_importance = [f for f, i in sorted_features if i['importance'] == 'MEDIUM']
    low_importance = [f for f, i in sorted_features if i['importance'] == 'LOW']
    
    print("RECOMMENDATIONS:")
    print(f"  KEEP (HIGH):   {high_importance if high_importance else 'None'}")
    print(f"  KEEP (MEDIUM): {medium_importance if medium_importance else 'None'}")
    print(f"  DROP (LOW):    {low_importance if low_importance else 'None'}")


def main():
    print("=" * 70)
    print("ABLATION STUDY: Intrinsic Metrics Feature Importance")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Separate patterns
    simple_df, complex_df = separate_patterns(df)
    
    results = {}
    
    # =========================================================================
    # SIMPLE PATTERNS ABLATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("SIMPLE PATTERNS (1-4, 2-2, 3-1)")
    print("=" * 70)
    
    # Check available features
    print("\nDetection features:")
    det_features = get_available_features(simple_df, SIMPLE_DETECTION_FEATURES)
    
    print("\nSAM features:")
    sam_features = get_available_features(simple_df, SIMPLE_SAM_FEATURES)
    
    all_simple_features = det_features + sam_features
    
    # Run ablation with ALL features (combined)
    if len(all_simple_features) >= 2:
        print(f"\nRunning ablation with {len(all_simple_features)} features (combined)...")
        simple_results = run_ablation(simple_df, all_simple_features)
        print_importance_table(simple_results, "SIMPLE PATTERNS - Combined Features")
        results['simple_patterns_combined'] = simple_results
    
    # Run ablation with DETECTION-ONLY features (more samples available)
    if len(det_features) >= 2:
        print(f"\nRunning ablation with {len(det_features)} detection-only features...")
        det_only_results = run_ablation(simple_df, det_features)
        print_importance_table(det_only_results, "SIMPLE PATTERNS - Detection Only")
        results['simple_patterns_detection'] = det_only_results
    
    # Run ablation with SAM-ONLY features
    if len(sam_features) >= 2:
        print(f"\nRunning ablation with {len(sam_features)} SAM-only features...")
        sam_only_results = run_ablation(simple_df, sam_features)
        print_importance_table(sam_only_results, "SIMPLE PATTERNS - SAM Only")
        results['simple_patterns_sam'] = sam_only_results
    
    # =========================================================================
    # COMPLEX PATTERNS ABLATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPLEX PATTERNS (4-1, 5-1)")
    print("=" * 70)
    
    print("\nDetection features:")
    complex_det_features = get_available_features(complex_df, COMPLEX_SAM_PARAMS)
    
    if len(complex_det_features) >= 2:
        print(f"\nRunning ablation with {len(complex_det_features)} features...")
        complex_results = run_ablation(complex_df, complex_det_features)
        print_importance_table(complex_results, "COMPLEX PATTERNS - Feature Importance")
        results['complex_patterns'] = complex_results
    else:
        print("\nInsufficient features for complex pattern ablation")
        print("NOTE: Complex patterns typically use SAM geometry params (segment_width, k_height)")
        print("      which are not in the intrinsic_training_data.csv")
        results['complex_patterns'] = None
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {OUTPUT_PATH}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Compare all simple pattern models
    print("\nSimple Patterns Model Comparison:")
    print(f"{'Model':<25} {'Samples':>8} {'Features':>10} {'Spearman':>10} {'MAE':>8}")
    print("-" * 65)
    
    for key in ['simple_patterns_combined', 'simple_patterns_detection', 'simple_patterns_sam']:
        if results.get(key):
            sr = results[key]
            model_name = key.replace('simple_patterns_', '').title()
            print(f"{model_name:<25} {sr['n_samples']:>8} {sr['n_features']:>10} {sr['baseline_spearman']:>10.3f} {sr['baseline_mae']:>8.3f}")
    
    # Show best model details
    best_key = None
    best_spearman = -1
    for key in ['simple_patterns_combined', 'simple_patterns_detection', 'simple_patterns_sam']:
        if results.get(key) and results[key]['baseline_spearman'] > best_spearman:
            best_spearman = results[key]['baseline_spearman']
            best_key = key
    
    if best_key:
        sr = results[best_key]
        high = [f for f, i in sr['feature_importance'].items() if i['importance'] in ['HIGH', 'MEDIUM']]
        print(f"\n  Best model: {best_key}")
        print(f"  Important features: {high if high else 'Need more data to determine'}")
    
    if results.get('complex_patterns'):
        cr = results['complex_patterns']
        high = [f for f, i in cr['feature_importance'].items() if i['importance'] == 'HIGH']
        print(f"\nComplex Patterns:")
        print(f"  Best model: Spearman={cr['baseline_spearman']:.3f}, MAE={cr['baseline_mae']:.3f}")
        print(f"  Most important features: {high}")
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()

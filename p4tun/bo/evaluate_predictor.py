"""
Evaluate Intrinsic Metrics → mIoU Predictor

Experiment 1: Predictive Validity (Regression with CV)
  - R², MAE, RMSE with 5-fold cross-validation

Experiment 2: Ablation Study
  - All metrics vs det_* only vs sam_* only
  - Report ΔRMSE, ΔR²

Usage:
  python -m p4tun.bo.evaluate_predictor
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA = PROJECT_ROOT / "bo4tun" / "training" / "intrinsic_training_data.csv"

# Tunnel context
TUNNEL_CONTEXT = {
    "1-4": {"pattern_type": "simple_staggered", "expected_rings": 10},
    "2-2": {"pattern_type": "simple_staggered", "expected_rings": 10},
    "3-1": {"pattern_type": "continuous", "expected_rings": 6},
    "4-1": {"pattern_type": "complex_staggered", "expected_rings": 10},
    "5-1": {"pattern_type": "complex_staggered", "expected_rings": 7},
}

DET_COLS = [
    "det_k_count", "det_k_count_match", "det_assume_default_ratio",
    "det_midpoint_ratio", "det_real_detection_ratio",
    "det_y_range", "det_y_std", "det_x_spacing_cv",
]
SAM_COLS = [
    "sam_prompt_count", "sam_segment_count", "sam_segment_count_match",
    "sam_mask_fill_rate", "sam_template_coverage",
]


def load_data(filter_tunnels=None):
    """Load training data and prepare features."""
    df = pd.read_csv(TRAINING_DATA)
    
    # Add tunnel context
    df["expected_rings"] = df["tunnel_id"].map(
        lambda t: TUNNEL_CONTEXT.get(t, {}).get("expected_rings", 10)
    )
    df["pattern_type"] = df["tunnel_id"].map(
        lambda t: TUNNEL_CONTEXT.get(t, {}).get("pattern_type", "simple_staggered")
    )
    
    # Optional filter
    if filter_tunnels:
        df = df[df["tunnel_id"].isin(filter_tunnels)].copy()
    
    return df


def prepare_features(df, metric_cols):
    """Prepare feature matrix X and target y."""
    # Encode pattern_type
    le = LabelEncoder()
    le.fit(["simple_staggered", "continuous", "complex_staggered"])
    pattern_enc = le.transform(df["pattern_type"].fillna("simple_staggered"))
    
    # Context features
    context = np.column_stack([pattern_enc, df["expected_rings"].values])
    
    # Metric features
    available_cols = [c for c in metric_cols if c in df.columns]
    
    if available_cols:
        Z = df[available_cols].astype(float)
        imputer = SimpleImputer(strategy="median")
        Z_imputed = imputer.fit_transform(Z)
        X = np.hstack([context, Z_imputed])
    else:
        # Context only - no intrinsic metrics
        X = context
    
    y = df["mIoU"].values.astype(float)
    
    feature_names = ["pattern_type_enc", "expected_rings"] + available_cols
    return X, y, feature_names


def experiment_predictive_validity(df):
    """
    Experiment 1: Predictive Validity via Regression
    
    Fit mIoU = f(det_*, sam_*) with CV and report R², MAE, RMSE, Spearman.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Predictive Validity (Regression with CV)")
    print("=" * 70)
    
    all_metric_cols = DET_COLS + SAM_COLS
    X, y, feature_names = prepare_features(df, all_metric_cols)
    
    print(f"Samples: {len(y)}")
    print(f"Features: {len(feature_names)}")
    print(f"  Context: pattern_type_enc, expected_rings")
    print(f"  det_* metrics: {len([c for c in feature_names if c.startswith('det_')])}")
    print(f"  sam_* metrics: {len([c for c in feature_names if c.startswith('sam_')])}")
    
    # Model
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    
    # Cross-validation
    cv = KFold(n_splits=min(5, len(y) // 2), shuffle=True, random_state=42)
    
    # CV predictions
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    
    # Metrics
    mae = mean_absolute_error(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
    r2 = r2_score(y, y_pred_cv)
    spearman, p_value = spearmanr(y, y_pred_cv)
    
    print("\n--- Cross-Validation Results ---")
    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  R²:       {r2:.4f}")
    print(f"  Spearman: {spearman:.4f} (p={p_value:.4f})")
    
    # Per-tunnel breakdown
    print("\n--- Per-Tunnel Predictions ---")
    df_eval = df.copy()
    df_eval["predicted_mIoU"] = y_pred_cv
    df_eval["error"] = df_eval["predicted_mIoU"] - df_eval["mIoU"]
    
    for tid in sorted(df_eval["tunnel_id"].unique()):
        sub = df_eval[df_eval["tunnel_id"] == tid]
        t_mae = mean_absolute_error(sub["mIoU"], sub["predicted_mIoU"])
        t_spearman, _ = spearmanr(sub["mIoU"], sub["predicted_mIoU"]) if len(sub) > 2 else (0, 1)
        print(f"  {tid}: n={len(sub):2d}, MAE={t_mae:.4f}, Spearman={t_spearman:.3f}")
    
    return {
        "mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman,
        "n_samples": len(y), "n_features": len(feature_names),
    }


def experiment_ablation(df):
    """
    Experiment 2: Ablation Study
    
    Compare: all metrics vs det_* only vs sam_* only.
    Report ΔRMSE, ΔR².
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Ablation Study")
    print("=" * 70)
    
    results = {}
    cv = KFold(n_splits=min(5, len(df) // 2), shuffle=True, random_state=42)
    
    configs = [
        ("All metrics (det_* + sam_*)", DET_COLS + SAM_COLS),
        ("det_* only", DET_COLS),
        ("sam_* only", SAM_COLS),
        ("Context only (no intrinsic)", []),
    ]
    
    for name, metric_cols in configs:
        X, y, feat_names = prepare_features(df, metric_cols)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "n_features": len(feat_names)}
        print(f"\n{name}:")
        print(f"  Features: {len(feat_names)}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
    
    # Compute deltas
    baseline = results["All metrics (det_* + sam_*)"]
    print("\n--- Ablation Deltas (vs All metrics) ---")
    print(f"{'Configuration':<30} {'ΔRMSE':>10} {'ΔR²':>10}")
    print("-" * 50)
    for name, r in results.items():
        delta_rmse = r["rmse"] - baseline["rmse"]
        delta_r2 = r["r2"] - baseline["r2"]
        print(f"{name:<30} {delta_rmse:>+10.4f} {delta_r2:>+10.4f}")
    
    return results


def main(filter_tunnels=None):
    print("=" * 70)
    print("INTRINSIC METRICS → mIoU PREDICTOR EVALUATION")
    print("=" * 70)
    
    df = load_data(filter_tunnels=filter_tunnels)
    print(f"\nLoaded {len(df)} samples from {TRAINING_DATA}")
    if filter_tunnels:
        print(f"Filtered to tunnels: {filter_tunnels}")
    print(f"Tunnels: {sorted(df['tunnel_id'].unique())}")
    print(f"mIoU range: [{df['mIoU'].min():.3f}, {df['mIoU'].max():.3f}]")
    
    # Experiment 1
    exp1 = experiment_predictive_validity(df)
    
    # Experiment 2
    exp2 = experiment_ablation(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Predictive validity: R²={exp1['r2']:.3f}, MAE={exp1['mae']:.4f}, Spearman={exp1['spearman']:.3f}")
    print(f"Ablation: Removing det_* increases RMSE by {exp2['sam_* only']['rmse'] - exp2['All metrics (det_* + sam_*)']['rmse']:+.4f}")
    print(f"          Removing sam_* increases RMSE by {exp2['det_* only']['rmse'] - exp2['All metrics (det_* + sam_*)']['rmse']:+.4f}")
    
    # Save results
    out_path = PROJECT_ROOT / "p4tun" / "bo" / "results" / "predictor_evaluation.json"
    import json
    with open(out_path, "w") as f:
        json.dump({"experiment1": exp1, "experiment2": {k: v for k, v in exp2.items()}}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tunnels', nargs='+', default=None,
                        help='Filter to specific tunnels (e.g., --tunnels 1-4 2-2)')
    args = parser.parse_args()
    main(filter_tunnels=args.tunnels)

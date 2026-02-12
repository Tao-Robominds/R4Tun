#!/usr/bin/env python3
"""
Run full ablation experiment with ALL 5 no-GT intrinsic metrics.
Computes both ΔMAE and ΔR² for each metric.
Output: JSON + markdown table for INTRINSIC_METRICS_REPORT.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# All 5 no-GT metrics from the report (correlation rank order)
FULL_FEATURES = [
    'det_midpoint_ratio',
    'sam_mask_fill_rate',
    'det_real_detection_ratio',
    'det_x_spacing_cv',
    'det_y_std',
]


def load_data() -> pd.DataFrame:
    """Load training data from available sources."""
    candidates = [
        project_root / 'bo4tun' / 'training' / 'intrinsic_training_data.csv',
        project_root / 'bo4tun' / 'report' / 'training_data_simple.csv',
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df['pattern_type'] = df['tunnel_id'].apply(
                lambda x: 'complex' if x in ['4-1', '5-1'] else 'simple'
            )
            return df
    raise FileNotFoundError("No training data found")


def run_full_ablation(df: pd.DataFrame, features: list, target: str = 'mIoU', alpha: float = 1.0):
    """Run leave-one-out ablation; return ΔMAE and ΔR² for each feature."""
    available = [f for f in features if f in df.columns]
    cols = available + [target]
    valid = df[cols].dropna()
    valid = valid[valid[target] > 0]

    if len(valid) < 5:
        return None

    X = valid[available].values
    y = valid[target].values

    baseline = Ridge(alpha=alpha)
    baseline.fit(X, y)
    pred_all = baseline.predict(X)
    baseline_mae = mean_absolute_error(y, pred_all)
    baseline_r2 = r2_score(y, pred_all)
    baseline_spearman = spearmanr(y, pred_all)[0]

    results = {}
    for i, feat in enumerate(available):
        idx = [j for j, f in enumerate(available) if f != feat]
        X_sub = X[:, idx]
        model = Ridge(alpha=alpha)
        model.fit(X_sub, y)
        pred_sub = model.predict(X_sub)
        mae = mean_absolute_error(y, pred_sub)
        r2 = r2_score(y, pred_sub)
        delta_mae = mae - baseline_mae
        delta_r2 = r2 - baseline_r2  # negative when removing helps
        results[feat] = {
            'delta_mae': round(delta_mae, 4),
            'delta_r2': round(delta_r2, 4),
            'mae_without': round(mae, 4),
            'r2_without': round(r2, 4),
        }

    return {
        'n_samples': len(valid),
        'n_features': len(available),
        'baseline_mae': round(baseline_mae, 4),
        'baseline_r2': round(baseline_r2, 4),
        'baseline_spearman': round(baseline_spearman, 4),
        'feature_results': results,
    }


def main():
    print("=" * 70)
    print("FULL ABLATION: All 5 no-GT intrinsic metrics (ΔMAE + ΔR²)")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} rows")

    # Simple patterns (1-4, 2-2, 3-1) - combined model
    simple = df[df['pattern_type'] == 'simple'].copy()
    res_simple = run_full_ablation(simple, FULL_FEATURES)
    if res_simple:
        print(f"\nSimple patterns: n={res_simple['n_samples']}, baseline MAE={res_simple['baseline_mae']}, R²={res_simple['baseline_r2']}")
    else:
        print("\nSimple: insufficient data")

    # All patterns (more samples)
    res_all = run_full_ablation(df, FULL_FEATURES)
    if res_all:
        print(f"All patterns:   n={res_all['n_samples']}, baseline MAE={res_all['baseline_mae']}, R²={res_all['baseline_r2']}")

    # Use best available
    result = res_simple if res_simple and res_simple['n_samples'] >= 8 else res_all
    if not result:
        print("ERROR: Insufficient data for ablation")
        return 1

    print("\n" + "=" * 70)
    print("ABLATION RESULTS (ΔMAE, ΔR² when feature removed)")
    print("=" * 70)
    for feat in FULL_FEATURES:
        r = result['feature_results'].get(feat, {})
        dmae = r.get('delta_mae', float('nan'))
        dr2 = r.get('delta_r2', float('nan'))
        print(f"  {feat}: ΔMAE={dmae:+.4f}, ΔR²={dr2:+.4f}")

    out_dir = project_root / 'bo4tun' / 'report'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'full_ablation_results.json'
    with open(out_path, 'w') as f:
        json.dump({'simple': res_simple, 'all': res_all}, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

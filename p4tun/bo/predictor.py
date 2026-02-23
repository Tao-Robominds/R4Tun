"""
Layer B: Learned mIoU Predictor

Trains a regressor to predict mIoU from intrinsic metrics + tunnel context.
At runtime without GT: compute intrinsic metrics from pipeline outputs, predict mIoU.

Features:
  - Tunnel context: pattern_type, expected_rings
  - Intrinsic metrics: det_*, sam_* from intrinsic_metrics.py

Usage:
  python -m p4tun.bo.predictor --train
  python -m p4tun.bo.predictor --predict  # demo with loaded model
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DATA = PROJECT_ROOT / "bo4tun" / "training" / "intrinsic_training_data.csv"
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "miou_predictor.pkl"

# Tunnel context (pattern_type, expected_rings)
TUNNEL_CONTEXT = {
    "1-4": {"pattern_type": "simple_staggered", "expected_rings": 10},
    "2-2": {"pattern_type": "simple_staggered", "expected_rings": 10},
    "3-1": {"pattern_type": "continuous", "expected_rings": 6},
    "4-1": {"pattern_type": "complex_staggered", "expected_rings": 10},
    "5-1": {"pattern_type": "complex_staggered", "expected_rings": 7},
}

INTRINSIC_COLUMNS = [
    "det_k_count", "det_k_count_match", "det_assume_default_ratio",
    "det_midpoint_ratio", "det_real_detection_ratio",
    "det_y_range", "det_y_std", "det_x_spacing_cv",
    "sam_prompt_count", "sam_segment_count", "sam_segment_count_match",
    "sam_mask_fill_rate", "sam_template_coverage",
]


def _add_tunnel_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add pattern_type and expected_rings from tunnel_id."""
    out = df.copy()
    out["expected_rings"] = out["tunnel_id"].map(
        lambda t: TUNNEL_CONTEXT.get(t, {}).get("expected_rings", 10)
    )
    out["pattern_type"] = out["tunnel_id"].map(
        lambda t: TUNNEL_CONTEXT.get(t, {}).get("pattern_type", "simple_staggered")
    )
    return out


def _prepare_features(
    df: pd.DataFrame,
    intrinsic_cols: list,
    pattern_encoder: LabelEncoder = None,
    imputer: SimpleImputer = None,
) -> tuple:
    """
    Build feature matrix X from tunnel context + intrinsic metrics.
    Returns (X, pattern_encoder, imputer) for training, or (X,) when encoders provided.
    """
    df = _add_tunnel_context(df)

    # Context: pattern_type (one-hot via label encoding), expected_rings
    pattern_vals = df["pattern_type"].fillna("simple_staggered").astype(str)
    if pattern_encoder is None:
        pattern_encoder = LabelEncoder()
        pattern_encoder.fit(
            ["simple_staggered", "continuous", "complex_staggered"]
        )
    def safe_label(x):
        x = str(x) if pd.notna(x) else "simple_staggered"
        return x if x in pattern_encoder.classes_ else "simple_staggered"

    pattern_encoded = pattern_encoder.transform(
        [safe_label(v) for v in pattern_vals.values]
    )

    context = np.column_stack([
        pattern_encoded,
        df["expected_rings"].values.astype(np.float64),
    ])

    # Intrinsic metrics
    metric_cols = [c for c in intrinsic_cols if c in df.columns]
    Z = df[metric_cols].copy()
    Z = Z.astype(float)

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(Z)
    Z_imputed = imputer.transform(Z)

    X = np.hstack([context, Z_imputed])
    feature_names = (
        ["pattern_type_encoded", "expected_rings"] + metric_cols
    )
    return X, feature_names, pattern_encoder, imputer


def load_training_data(csv_path: Path = None) -> pd.DataFrame:
    """Load intrinsic training data CSV."""
    path = csv_path or DEFAULT_TRAINING_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "Run: python -m bo4tun.build_training_data --sample 50"
        )
    return pd.read_csv(path)


def train(
    csv_path: Path = None,
    model_path: Path = None,
    model_type: str = "gradient_boosting",
) -> dict:
    """
    Train mIoU predictor and save to pkl.

    Returns dict with metrics (train_score, cv_score, etc.).
    """
    df = load_training_data(csv_path)
    y = df["mIoU"].values.astype(np.float64)

    intrinsic_cols = [c for c in INTRINSIC_COLUMNS if c in df.columns]
    if not intrinsic_cols:
        raise ValueError(
            f"No intrinsic metric columns found. Expected some of: {INTRINSIC_COLUMNS}"
        )

    X, feature_names, pattern_encoder, imputer = _prepare_features(
        df, intrinsic_cols
    )

    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=42
        )

    model.fit(X, y)
    train_pred = model.predict(X)
    train_mae = np.mean(np.abs(train_pred - y))
    train_rmse = np.sqrt(np.mean((train_pred - y) ** 2))

    cv_scores = cross_val_score(
        model, X, y, cv=min(5, len(df) // 2), scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores.mean()

    model_path = model_path or DEFAULT_MODEL_PATH
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "pattern_encoder": pattern_encoder,
        "imputer": imputer,
        "feature_names": feature_names,
        "intrinsic_cols": intrinsic_cols,
        "model_type": model_type,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"Model saved to {model_path}")
    print(f"  Train MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
    print(f"  CV MAE (5-fold): {cv_mae:.4f}")
    print(f"  Features: {len(feature_names)}")

    return {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "cv_mae": cv_mae,
        "n_samples": len(df),
        "n_features": len(feature_names),
    }


def load_model(model_path: Path = None):
    """Load trained predictor bundle."""
    path = Path(model_path or DEFAULT_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun: python -m p4tun.bo.predictor --train"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(
    tunnel_id: str,
    intrinsic_metrics: dict,
    model_path: Path = None,
) -> float:
    """
    Predict mIoU from tunnel context + intrinsic metrics.

    intrinsic_metrics: dict with keys like det_k_count, det_y_range, sam_prompt_count, etc.
    """
    bundle = load_model(model_path)
    pattern_encoder = bundle["pattern_encoder"]
    imputer = bundle["imputer"]
    model = bundle["model"]
    intrinsic_cols = bundle["intrinsic_cols"]

    ctx = TUNNEL_CONTEXT.get(
        tunnel_id, {"pattern_type": "simple_staggered", "expected_rings": 10}
    )
    pattern_encoded = pattern_encoder.transform([ctx["pattern_type"]])[0]
    context = np.array([[pattern_encoded, ctx["expected_rings"]]], dtype=np.float64)

    z = np.array(
        [intrinsic_metrics.get(c, np.nan) for c in intrinsic_cols],
        dtype=np.float64,
    ).reshape(1, -1)
    z_df = pd.DataFrame(z, columns=intrinsic_cols)
    z_imputed = imputer.transform(z_df)

    X = np.hstack([context, z_imputed])
    return float(model.predict(X)[0])


def main():
    parser = argparse.ArgumentParser(description="Layer B: mIoU predictor")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model from intrinsic_training_data.csv",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to intrinsic_training_data.csv",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to save/load model pkl",
    )
    parser.add_argument(
        "--model-type",
        choices=["gradient_boosting", "random_forest"],
        default="gradient_boosting",
    )
    args = parser.parse_args()

    if args.train:
        train(
            csv_path=Path(args.data) if args.data else None,
            model_path=Path(args.model) if args.model else None,
            model_type=args.model_type,
        )
    else:
        # Demo: load model and predict on a sample
        bundle = load_model(Path(args.model) if args.model else None)
        sample = {
            "det_k_count": 10,
            "det_k_count_match": 1.0,
            "det_assume_default_ratio": 0.0,
            "det_midpoint_ratio": 0.8,
            "det_real_detection_ratio": 1.0,
            "det_y_range": 435.0,
            "det_y_std": 212.0,
            "det_x_spacing_cv": 0.48,
        }
        pred = predict("2-2", sample, Path(args.model) if args.model else None)
        print(f"Demo prediction (2-2, sample metrics): mIoU ≈ {pred:.4f}")


if __name__ == "__main__":
    main()

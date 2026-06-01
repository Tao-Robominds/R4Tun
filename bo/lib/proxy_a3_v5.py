"""Frozen Ridge proxy loaders and predictors (A3-slim + p11)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import REPO_ROOT
from lib.held_out_common import A3_SLIM_MANIFEST, A3_V5_P11_MANIFEST


def load_proxy_model(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = Path(manifest["model"])
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    model = json.loads(model_path.read_text(encoding="utf-8"))
    cols = manifest.get("spec", {}).get("columns") or model.get("feature_columns", [])
    model["feature_columns"] = list(cols)
    model["manifest"] = str(manifest_path.relative_to(REPO_ROOT))
    model["variant"] = manifest.get("variant_id") or manifest.get("canonical_proxy", "unknown")
    return model


def load_a3_slim_model() -> dict[str, Any]:
    return load_proxy_model(A3_SLIM_MANIFEST)


def load_p11_model() -> dict[str, Any]:
    return load_proxy_model(A3_V5_P11_MANIFEST)


def predict_proxy(model: dict[str, Any], row: pd.Series | dict[str, Any]) -> float:
    feats = model["feature_columns"]
    if isinstance(row, dict):
        row = pd.Series(row)
    vals = []
    for f in feats:
        v = row.get(f, 0.0)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if not np.isfinite(fv):
            fv = 0.0
        vals.append(fv)
    x = np.asarray(vals, dtype=float)
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    xs = (x - mean) / np.where(scale == 0, 1.0, scale)
    coef = np.asarray(model["coef"], dtype=float)
    return float(np.dot(xs, coef) + model["intercept"])

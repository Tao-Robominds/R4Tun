"""V5-style proxy observables from post-det+seg sandbox artifacts (GT-free at runtime)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.pre_depth_qa import load_depth_3a

EXPECTED_BLOCK_TYPES = 7


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return f


def _load_boundaries_y(ring_dir: Path) -> list[float]:
    path = ring_dir / "boundaries_per_ring.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    ring0 = data.get("0", data.get(0, []))
    ys = [_safe_float(item.get("y")) for item in ring0 if item.get("y") is not None]
    return sorted(ys)


def _circular_gaps(ys: list[float], span: float) -> list[float]:
    if len(ys) < 2:
        return []
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    wrap = (span - ys[-1]) + ys[0]
    if wrap > 0:
        gaps.append(wrap)
    return [g for g in gaps if g > 0]


def compute_class_distribution_features(ring_dir: Path) -> dict[str, float]:
    final_path = ring_dir / "final.csv"
    out = {
        "present_ratio": 0.0,
        "entropy": 0.0,
        "cv": 0.0,
        "max_share": 0.0,
        "balance_norm": 0.0,
        "struct_missing_ids_before_n": float(EXPECTED_BLOCK_TYPES),
    }
    if not final_path.exists():
        return out
    df = pd.read_csv(final_path, usecols=["pred"])
    pred = pd.to_numeric(df["pred"], errors="coerce").fillna(0).astype(int)
    pos = pred[pred > 0]
    if pos.empty:
        return out
    counts = pos.value_counts()
    present = int(len(counts))
    total = float(counts.sum())
    shares = (counts / total).to_numpy(dtype=float)
    ent = float(-np.sum(shares * np.log(shares + 1e-12)))
    ent_norm = ent / math.log(max(present, 2))
    mean_c = float(counts.mean())
    std_c = float(counts.std(ddof=0))
    cv = std_c / mean_c if mean_c > 0 else 0.0
    max_share = float(shares.max()) if len(shares) else 0.0
    balance = float(1.0 / (1.0 + cv)) if cv >= 0 else 0.0
    out.update(
        {
            "present_ratio": present / EXPECTED_BLOCK_TYPES,
            "entropy": ent_norm,
            "cv": cv,
            "max_share": max_share,
            "balance_norm": balance,
            "struct_missing_ids_before_n": float(max(0, EXPECTED_BLOCK_TYPES - present)),
        }
    )
    return out


def compute_boundary_geometry_features(ring_dir: Path) -> dict[str, float]:
    d3a = load_depth_3a(ring_dir)
    height = max(int(_safe_float(d3a.get("shape_h"), 4712)), 1)
    ys = _load_boundaries_y(ring_dir)
    gaps = _circular_gaps(ys, float(height))
    if not gaps:
        return {
            "geom_boundary_gap_cv": 1.0,
            "geom_boundary_min_gap_frac": 0.0,
            "geom_boundary_max_gap_frac": 0.0,
            "geom_boundary_mean_gap_frac": 0.0,
            "n_boundaries": float(len(ys)),
        }
    arr = np.asarray(gaps, dtype=float)
    mean_g = float(arr.mean())
    std_g = float(arr.std(ddof=0))
    cv = std_g / mean_g if mean_g > 0 else 1.0
    fracs = arr / float(height)
    return {
        "geom_boundary_gap_cv": cv,
        "geom_boundary_min_gap_frac": float(fracs.min()),
        "geom_boundary_max_gap_frac": float(fracs.max()),
        "geom_boundary_mean_gap_frac": float(fracs.mean()),
        "n_boundaries": float(len(ys)),
    }


def compute_s_boundary_features(ring_dir: Path) -> dict[str, float]:
    geom = compute_boundary_geometry_features(ring_dir)
    ys = _load_boundaries_y(ring_dir)

    det_path = ring_dir / "detected.csv"
    confidences: list[float] = []
    k_conf = 0.0
    if det_path.exists():
        det = pd.read_csv(det_path)
        if "Confidence" in det.columns:
            confidences = pd.to_numeric(det["Confidence"], errors="coerce").dropna().tolist()
        if "Type" in det.columns and "Confidence" in det.columns:
            k_rows = det[det["Type"].astype(str).str.contains("k", case=False, na=False)]
            if not k_rows.empty:
                k_conf = _safe_float(k_rows["Confidence"].iloc[0])

    s_continuity = float(np.clip(np.mean(confidences) if confidences else 0.0, 0.0, 1.0))
    s_k = float(np.clip(k_conf if k_conf > 0 else 0.25, 0.0, 1.0))
    if k_conf <= 0 and ys:
        s_k = 0.25 if len(ys) >= 4 else 0.1

    gap_cv = geom["geom_boundary_gap_cv"]
    s_spacing = float(np.clip(1.0 - min(gap_cv, 1.0), 0.0, 1.0))
    n_blocks = geom["n_boundaries"]
    s_layout = float(np.clip(n_blocks / EXPECTED_BLOCK_TYPES, 0.0, 1.2))
    s_layout = min(s_layout, 1.0)

    s_boundary = s_continuity * s_k * s_spacing * s_layout
    return {
        "S_continuity": s_continuity,
        "S_K": s_k,
        "S_spacing": s_spacing,
        "S_layout_coverage": s_layout,
        "S_boundary": s_boundary,
    }


def compute_depth_audit_features(ring_dir: Path) -> dict[str, float]:
    d3a = load_depth_3a(ring_dir)
    return {
        "depth_row_nonempty_ratio_audit": _safe_float(d3a.get("row_nonempty_ratio")),
        "finite_ratio_audit": _safe_float(d3a.get("finite_ratio")),
    }


def compute_v5_proxy_features(ring_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(compute_depth_audit_features(ring_dir))
    out.update(compute_class_distribution_features(ring_dir))
    out.update(compute_boundary_geometry_features(ring_dir))
    out.update(compute_s_boundary_features(ring_dir))
    return out


def prefix_features(raw: dict[str, float], prefix: str = "v5_") -> dict[str, float]:
    return {f"{prefix}{k}": v for k, v in raw.items()}

"""Ring descriptors for held-out panel (depth QA, density, k-span, direction)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import REPO_ROOT
from lib.ring_site_params import resolve_ring_site_params


def _load_depth_audit(ring_dir: Path) -> dict[str, Any]:
    contract = ring_dir / "depth_contract_selected.json"
    if contract.is_file():
        data = json.loads(contract.read_text(encoding="utf-8"))
        audit = data.get("audit") or data
        return audit
    depth = ring_dir / "depth_map.npy"
    if depth.is_file():
        arr = np.load(depth)
        finite = np.isfinite(arr)
        fr = float(finite.sum() / max(arr.size, 1))
        row_ok = float((finite.sum(axis=1) > 0).mean())
        return {
            "finite_ratio": fr,
            "row_nonempty_ratio": row_ok,
            "largest_empty_vertical_gap_frac": 0.0,
            "height_px": int(arr.shape[0]),
            "width_px": int(arr.shape[1]) if arr.ndim > 1 else 1,
        }
    return {}


def _density_score(ring_dir: Path) -> float:
    enh = ring_dir / "enhanced.csv"
    if not enh.is_file():
        return 0.5
    df = pd.read_csv(enh, usecols=lambda c: c in {"h", "theta"})
    if df.empty or "h" not in df.columns:
        return 0.5
    h = pd.to_numeric(df["h"], errors="coerce").dropna()
    if h.empty:
        return 0.5
    span = float(h.max() - h.min())
    n = len(h)
    return float(np.clip(np.log1p(n) / (np.log1p(500_000) * max(span, 0.01)), 0.0, 1.0))


def _k_span_degrees(ring_dir: Path, segment_count: int) -> float:
    enh = ring_dir / "enhanced.csv"
    if not enh.is_file():
        return 0.5
    df = pd.read_csv(enh, usecols=lambda c: c in {"theta", "h"})
    if "theta" in df.columns:
        t = pd.to_numeric(df["theta"], errors="coerce").dropna()
        if len(t) > 10:
            span = float(t.max() - t.min())
            return float(np.clip(span / 360.0, 0.0, 1.0))
    if "h" in df.columns:
        h = pd.to_numeric(df["h"], errors="coerce").dropna()
        if len(h) > 10:
            return float(np.clip((h.max() - h.min()) * float(segment_count) / 6.0, 0.0, 1.0))
    return 0.5


def _is_rotation(order: list[int], template: list[int]) -> bool:
    if len(order) != len(template) or not order:
        return False
    n = len(template)
    for i in range(n):
        if order == template[i:] + template[:i]:
            return True
    return False


def _load_spatial_order(ring_dir: Path, segment_count: int) -> list[int] | None:
    gt_path = ring_dir / "gt_layout.json"
    if gt_path.is_file():
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        order = data.get("spatial_order_by_label")
        if isinstance(order, list) and order:
            return [int(x) for x in order]
    from lib.ceiling_gate import derive_gt_layout

    try:
        layout = derive_gt_layout(ring_dir, ring_dir, segment_count)
    except Exception:
        return None
    order = layout.get("spatial_order_by_label")
    if not isinstance(order, list) or not order:
        return None
    return [int(x) for x in order]


def _direction_from_spatial_order(order: list[int] | None, segment_count: int) -> tuple[str, float]:
    if not order:
        return "unknown", 0.5
    forward = list(range(1, segment_count + 1))
    backward = list(reversed(forward))
    if _is_rotation(order, forward):
        return "plus", 1.0
    if _is_rotation(order, backward):
        return "minus", 0.0
    return "unknown", 0.5


def _tier_density(score: float) -> str:
    if score >= 0.55:
        return "dense"
    if score >= 0.35:
        return "medium"
    return "low"


def _tier_k_span(score: float) -> str:
    if score < 0.25:
        return "narrow"
    if score < 0.65:
        return "normal"
    return "wide"


def _tier_coverage(finite_ratio: float) -> str:
    return "full" if finite_ratio >= 0.85 else "partial"


def _diameter_bin(diameter_m: float) -> float:
    bins = [5.5, 5.8, 7.4, 7.5]
    return float(min(bins, key=lambda b: abs(b - diameter_m)))


def build_ring_descriptor(
    ring_key: str,
    held_out_root: Path,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    tunnel_id, ring_part = ring_key.split("/")
    ring_id = int(ring_part.lstrip("r"))
    ring_dir = held_out_root / tunnel_id / f"r{ring_id}"
    if not ring_dir.is_dir():
        raise FileNotFoundError(ring_dir)

    site = resolve_ring_site_params(ring_key, ring_dir, registry_path=registry_path)
    seg = int(site["segment_count"])
    diam = float(site["tunnel_diameter"])
    audit = _load_depth_audit(ring_dir)
    finite = float(audit.get("finite_ratio", 0.0))
    row_ne = float(audit.get("row_nonempty_ratio", 0.0))
    gap_frac = float(audit.get("largest_empty_vertical_gap_frac", 0.0))
    cov = finite

    density = _density_score(ring_dir)
    k_span = _k_span_degrees(ring_dir, seg)
    spatial_order = _load_spatial_order(ring_dir, seg)
    direction_tier, direction_score = _direction_from_spatial_order(spatial_order, seg)

    return {
        "ring_key": ring_key,
        "tunnel_id": tunnel_id,
        "ring_id": ring_id,
        "segment_count": seg,
        "tunnel_diameter_m": diam,
        "diameter_bin": _diameter_bin(diam),
        "finite_ratio": round(finite, 6),
        "row_nonempty_ratio": round(row_ne, 6),
        "depth_coverage_ratio": round(cov, 6),
        "blank_band_ratio": round(gap_frac, 6),
        "density_score": round(density, 6),
        "k_span_score": round(k_span, 6),
        "direction_score": round(direction_score, 6),
        "spatial_order_by_label": spatial_order,
        "density_tier": _tier_density(density),
        "k_span_tier": _tier_k_span(k_span),
        "direction_tier": direction_tier,
        "coverage_tier": _tier_coverage(finite),
        "image_height": int(audit.get("height_px", 0)),
        "image_width": int(audit.get("width_px", 0)),
    }


def build_panel_descriptors(
    panel_csv: Path,
    held_out_root: Path,
    *,
    registry_path: Path | None = None,
) -> pd.DataFrame:
    panel = pd.read_csv(panel_csv)
    rows = []
    for ring_key in panel["ring_key"].astype(str):
        rows.append(build_ring_descriptor(ring_key, held_out_root, registry_path=registry_path))
    return pd.DataFrame(rows)

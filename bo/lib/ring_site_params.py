"""Pre-defined ring site parameters (segment_count, tunnel_diameter).

These MUST be resolved before preprocessing replay, BO, or agent runs.
Runtime code must not infer segment_count from GT labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lib.ceiling_gate import REPO_ROOT

DEFAULT_REGISTRY = REPO_ROOT / "data" / "ring_site_params.json"
SITE_PARAMS_FILENAME = "ring_site_params.json"


def load_registry(path: Path | None = None) -> dict[str, Any]:
    reg_path = (path or DEFAULT_REGISTRY).resolve()
    if not reg_path.is_file():
        raise FileNotFoundError(f"Ring site params registry missing: {reg_path}")
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    rings = data.get("rings", {})
    if not rings:
        raise ValueError(f"Registry has no rings: {reg_path}")
    return data


def registry_entry(registry: dict[str, Any], ring_key: str) -> dict[str, Any]:
    rings = registry.get("rings", {})
    if ring_key not in rings:
        raise KeyError(f"Ring {ring_key} not in ring_site_params registry")
    entry = dict(rings[ring_key])
    entry["ring_key"] = ring_key
    return entry


def _read_preprocessing_diameter(src_ring: Path) -> float | None:
    prep = src_ring / "parameters_preprocessing.json"
    if not prep.is_file():
        return None
    data = json.loads(prep.read_text(encoding="utf-8"))
    val = data.get("tunnel_diameter")
    return float(val) if val is not None else None


def resolve_ring_site_params(
    ring_key: str,
    src_ring: Path,
    *,
    segment_count: int | None = None,
    tunnel_diameter: float | None = None,
    manifest_entry: dict[str, Any] | None = None,
    registry_path: Path | None = None,
    diameter_tol: float = 0.05,
) -> dict[str, Any]:
    """Resolve and validate site params. Explicit args override registry."""
    registry = load_registry(registry_path)
    base = registry_entry(registry, ring_key)

    seg = segment_count
    if seg is None and manifest_entry is not None:
        seg = manifest_entry.get("segment_count")
    if seg is None:
        seg = base.get("segment_count")
    if seg is None:
        raise ValueError(f"segment_count required for {ring_key}")
    seg = int(seg)
    if seg not in (6, 7):
        raise ValueError(f"segment_count must be 6 or 7 for {ring_key}, got {seg}")

    diam = tunnel_diameter
    if diam is None and manifest_entry is not None:
        diam = manifest_entry.get("tunnel_diameter")
    if diam is None:
        diam = base.get("tunnel_diameter")
    if diam is None:
        raise ValueError(f"tunnel_diameter required for {ring_key}")
    diam = float(diam)

    prep_diam = _read_preprocessing_diameter(src_ring)
    if prep_diam is not None and abs(prep_diam - diam) > diameter_tol:
        raise ValueError(
            f"{ring_key}: tunnel_diameter mismatch — registry={diam}, "
            f"parameters_preprocessing.json={prep_diam}"
        )

    if int(base["segment_count"]) != seg:
        raise ValueError(
            f"{ring_key}: segment_count override {seg} != registry {base['segment_count']}"
        )
    if abs(float(base["tunnel_diameter"]) - diam) > diameter_tol:
        raise ValueError(
            f"{ring_key}: tunnel_diameter override {diam} != registry {base['tunnel_diameter']}"
        )

    return {
        "ring_key": ring_key,
        "segment_count": seg,
        "tunnel_diameter": round(diam, 4),
        "corpus": base.get("corpus"),
        "registry_path": str((registry_path or DEFAULT_REGISTRY).resolve()),
        "preprocessing_diameter": prep_diam,
    }


def write_ring_site_params(dest_dir: Path, params: dict[str, Any]) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / SITE_PARAMS_FILENAME
    payload = {
        "ring_key": params["ring_key"],
        "segment_count": int(params["segment_count"]),
        "tunnel_diameter": float(params["tunnel_diameter"]),
        "corpus": params.get("corpus"),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def confirm_ring_site_params(
    ring_key: str,
    *,
    source_root: Path,
    registry_path: Path | None = None,
    manifest_entry: dict[str, Any] | None = None,
    write_dir: Path | None = None,
) -> dict[str, Any]:
    """Lightweight gate: resolve site params; optionally write ring_site_params.json under write_dir."""
    tunnel_id, ring_id = ring_key.split("/")
    src_ring = source_root / tunnel_id / ring_id
    if not src_ring.is_dir():
        raise FileNotFoundError(f"No ring corpus at {src_ring}")
    params = resolve_ring_site_params(
        ring_key,
        src_ring,
        manifest_entry=manifest_entry,
        registry_path=registry_path,
    )
    written = None
    if write_dir is not None:
        written = write_ring_site_params(write_dir, params)
    return {
        "ring_key": ring_key,
        "segment_count": params["segment_count"],
        "tunnel_diameter": params["tunnel_diameter"],
        "corpus": params.get("corpus"),
        "ring_site_params_path": str(written) if written else None,
        "passed": True,
    }

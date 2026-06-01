#!/usr/bin/env python3
"""Build data/ring_site_params.json from BO-calibration + held-out corpora.

Design-time only: held-out segment_count is taken from enhanced.csv GT labels once
to populate the registry. Runtime must read the registry, not re-derive from GT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import REPO_ROOT, detect_segment_count  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "ring_site_params.json"
BO_MANIFEST = REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json"
HELD_PANEL = REPO_ROOT / "data" / "held-out" / "_manifests" / "data_v6_50ring_calibration_panel.csv"


def _diameter_from_prep(ring_dir: Path) -> float:
    prep = ring_dir / "parameters_preprocessing.json"
    if not prep.is_file():
        raise FileNotFoundError(prep)
    return float(json.loads(prep.read_text(encoding="utf-8"))["tunnel_diameter"])


def _add_ring(rings: dict, ring_key: str, *, segment_count: int, tunnel_diameter: float, corpus: str) -> None:
    if ring_key in rings:
        prev = rings[ring_key]
        if prev["segment_count"] != segment_count or abs(prev["tunnel_diameter"] - tunnel_diameter) > 0.05:
            raise ValueError(f"Conflicting registry entry for {ring_key}: {prev} vs new")
        return
    rings[ring_key] = {
        "segment_count": int(segment_count),
        "tunnel_diameter": round(float(tunnel_diameter), 4),
        "corpus": corpus,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rings: dict[str, dict] = {}

    bo = json.loads(BO_MANIFEST.read_text(encoding="utf-8"))
    for entry in bo.get("rings", []):
        rk = entry["ring_key"]
        tid, rid = rk.split("/")
        ring_dir = REPO_ROOT / "data" / "bo_calibration" / tid / rid
        _add_ring(
            rings,
            rk,
            segment_count=int(entry["segment_count"]),
            tunnel_diameter=_diameter_from_prep(ring_dir),
            corpus="bo_calibration",
        )

    import pandas as pd

    panel = pd.read_csv(HELD_PANEL)
    for rk in panel["ring_key"].astype(str):
        tid, rid = rk.split("/")
        ring_dir = REPO_ROOT / "data" / "held-out" / tid / rid
        if not ring_dir.is_dir():
            raise FileNotFoundError(ring_dir)
        _add_ring(
            rings,
            rk,
            segment_count=detect_segment_count(ring_dir),
            tunnel_diameter=_diameter_from_prep(ring_dir),
            corpus="held-out",
        )

    payload = {
        "schema_version": 1,
        "description": "Pre-defined segment_count and tunnel_diameter per ring. Required before BO/agents.",
        "n_rings": len(rings),
        "rings": dict(sorted(rings.items())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_rings": len(rings)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

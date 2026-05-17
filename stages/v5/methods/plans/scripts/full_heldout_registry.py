#!/usr/bin/env python3
"""Build full-heldout eligibility registry.

Outputs
-------
- logs/gravity_v1/full_heldout/full_heldout_registry.csv
- logs/gravity_v1/full_heldout/eligible_panel_now.json
- logs/gravity_v1/full_heldout/eligible_panel_post_calib.json
- logs/gravity_v1/full_heldout/calibration_gap_tunnels.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PANEL_JSON = REPO_ROOT / "data" / "panels" / "heldout" / "heldout_reflection_test_set.json"
CALIB_ROOT = REPO_ROOT / "logs" / "gravity_v1" / "calibration"
BO_ARTIFACTS = REPO_ROOT / "logs" / "detection_boundary_bo_v1" / "artifacts"
OUT_ROOT = REPO_ROOT / "logs" / "gravity_v1" / "full_heldout"


def _best_bo_ring_by_tunnel() -> dict[str, str]:
    out: dict[str, str] = {}
    if not BO_ARTIFACTS.exists():
        return out
    for tunnel_dir in sorted([p for p in BO_ARTIFACTS.iterdir() if p.is_dir()]):
        tunnel = tunnel_dir.name
        best_rings: list[str] = []
        for ring_dir in sorted([p for p in tunnel_dir.iterdir() if p.is_dir()]):
            p = ring_dir / "best" / tunnel / ring_dir.name / "parameters_detection.json"
            if p.exists():
                best_rings.append(ring_dir.name)
        if best_rings:
            out[tunnel] = best_rings[0]
    return out


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    panel = json.loads(PANEL_JSON.read_text())
    bo_best = _best_bo_ring_by_tunnel()

    rows: list[dict[str, object]] = []
    for rec in panel:
        tunnel = str(rec["tunnel_id"])
        ring_key = str(rec["ring_key"])
        ring_name = f"r{int(rec['ring_id'])}"
        calib_params = CALIB_ROOT / tunnel / "parameters_detection_gravity.json"
        has_calib = calib_params.exists()
        has_bo_source = tunnel in bo_best
        can_generate_calib = (not has_calib) and has_bo_source

        if has_calib:
            exclusion = ""
        elif can_generate_calib:
            exclusion = "calibration_missing_but_generatable"
        else:
            exclusion = "missing_calibration_no_bo_source"

        rows.append(
            {
                "ring_key": ring_key,
                "tunnel_id": tunnel,
                "ring_id": int(rec["ring_id"]),
                "segment_count": int(rec.get("segment_count", 0)),
                "difficulty": str(rec.get("difficulty", "")),
                "density_group": str(rec.get("density_group", "")),
                "has_calibration_template": bool(has_calib),
                "has_bo_calibration_source": bool(has_bo_source),
                "can_generate_calibration": bool(can_generate_calib),
                "is_evaluable_now": bool(has_calib),
                "is_evaluable_post_calib": bool(has_calib or can_generate_calib),
                "exclusion_reason": exclusion,
                "bo_calib_ring": bo_best.get(tunnel, ""),
                "source_txt": str(rec.get("source_txt", "")),
            }
        )

    df = pd.DataFrame(rows).sort_values(["tunnel_id", "ring_id"]).reset_index(drop=True)
    df.to_csv(OUT_ROOT / "full_heldout_registry.csv", index=False)

    eligible_now = [
        {"tunnel_id": r["tunnel_id"], "ring_id": int(r["ring_id"]), "ring_key": r["ring_key"]}
        for r in rows
        if bool(r["is_evaluable_now"])
    ]
    eligible_post = [
        {"tunnel_id": r["tunnel_id"], "ring_id": int(r["ring_id"]), "ring_key": r["ring_key"]}
        for r in rows
        if bool(r["is_evaluable_post_calib"])
    ]
    gaps = sorted(
        {
            str(r["tunnel_id"])
            for r in rows
            if (not bool(r["has_calibration_template"])) and bool(r["has_bo_calibration_source"])
        }
    )
    (OUT_ROOT / "eligible_panel_now.json").write_text(json.dumps(eligible_now, indent=2) + "\n")
    (OUT_ROOT / "eligible_panel_post_calib.json").write_text(json.dumps(eligible_post, indent=2) + "\n")
    (OUT_ROOT / "calibration_gap_tunnels.json").write_text(json.dumps(gaps, indent=2) + "\n")

    print(f"panel_rings={len(df)} tunnels={df['tunnel_id'].nunique()}")
    print(f"evaluable_now={int(df['is_evaluable_now'].sum())}")
    print(f"evaluable_post_calib={int(df['is_evaluable_post_calib'].sum())}")
    print(f"gap_tunnels={len(gaps)} -> {gaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

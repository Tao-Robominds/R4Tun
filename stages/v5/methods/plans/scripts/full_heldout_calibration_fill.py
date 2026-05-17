#!/usr/bin/env python3
"""Fill missing gravity calibration templates for full-heldout run.

Uses BO best artifacts under:
  logs/detection_boundary_bo_v1/artifacts/<tunnel>/<ring>/best/<tunnel>/<ring>/
for tunnels flagged as `can_generate_calibration` in full-heldout registry.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(SCRIPT_DIR))
from gravity_align_unwrap import _gravity_align_theta, _build_reference_profile  # noqa: E402


REGISTRY_CSV = REPO_ROOT / "logs" / "gravity_v1" / "full_heldout" / "full_heldout_registry.csv"
BO_BASE = REPO_ROOT / "logs" / "detection_boundary_bo_v1" / "artifacts"
CALIB_OUT = REPO_ROOT / "logs" / "gravity_v1" / "calibration"


def _safe_float(v: Any, default: float) -> float:
    try:
        f = float(v)
        if np.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return default


def _promote_one(tunnel: str, ring: str) -> dict[str, Any]:
    calib_dir = BO_BASE / tunnel / ring / "best" / tunnel / ring
    params_path = calib_dir / "parameters_detection.json"
    unwrap_path = calib_dir / "unwrapped.csv"
    depth_path = calib_dir / "depth_map.npy"

    if not (params_path.exists() and unwrap_path.exists() and depth_path.exists()):
        return {"status": "missing_files", "calib_dir": str(calib_dir)}

    det_params = json.loads(params_path.read_text())
    template = det_params.get("single_ring_visual_slot_template") or []

    df = pd.read_csv(unwrap_path)
    _, meta = _gravity_align_theta(df, ref_profile=None)
    theta_shift = float(meta["theta_shift"])
    dm = np.load(depth_path)
    h_px = int(dm.shape[0])

    res = 0.005
    pp_path = calib_dir / "parameters_preprocessing.json"
    if pp_path.exists():
        try:
            pp = json.loads(pp_path.read_text())
            res = _safe_float(pp.get("depth_map_resolution"), 0.005)
        except Exception:  # noqa: BLE001
            pass
    row_shift = int(round(theta_shift / res)) % max(1, h_px)
    shift_frac = float(row_shift) / float(max(1, h_px))

    new_template = []
    if template:
        for rec in template:
            y = _safe_float(rec.get("y_frac"), 0.0)
            y_new = (y - shift_frac) % 1.0
            new_rec = copy.deepcopy(rec)
            new_rec["y_frac"] = float(y_new)
            new_template.append(new_rec)
        new_template.sort(key=lambda r: float(r.get("y_frac", 0.0)))

    out_dir = CALIB_OUT / tunnel
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = out_dir / "template.json"
    out_params = out_dir / "parameters_detection_gravity.json"

    gravity_det = copy.deepcopy(det_params)
    if template:
        gravity_det["single_ring_visual_slot_template"] = new_template

    out_template.write_text(
        json.dumps(
            {
                "tunnel": tunnel,
                "calib_ring": ring,
                "calib_dir": str(calib_dir),
                "calib_theta_shift": theta_shift,
                "calib_row_shift": row_shift,
                "calib_H": h_px,
                "calib_resolution": res,
                "template": new_template,
                "n_blocks": len(new_template),
                "source_template_n": len(template),
                "template_shift_applied": bool(template),
                "source": "detection_boundary_bo_v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    out_params.write_text(json.dumps(gravity_det, indent=2, sort_keys=True) + "\n")

    # reference z-profile for direction disambiguation
    try:
        ref = _build_reference_profile(tunnel, BO_BASE)
        if ref is not None:
            np.save(out_dir / "ref_z_profile.npy", ref)
    except Exception:  # noqa: BLE001
        pass

    return {
        "status": "ok",
        "out": str(out_template),
        "row_shift": row_shift,
        "n_blocks": len(new_template),
        "template_shift_applied": bool(template),
    }


def main() -> int:
    if not REGISTRY_CSV.exists():
        raise FileNotFoundError(f"missing registry: {REGISTRY_CSV}")
    df = pd.read_csv(REGISTRY_CSV)
    gaps = (
        df[(df["can_generate_calibration"] == True) & (df["has_calibration_template"] == False)]  # noqa: E712
        .sort_values(["tunnel_id", "ring_id"])
        .drop_duplicates(subset=["tunnel_id"])
    )

    summary = {}
    for _, row in gaps.iterrows():
        tunnel = str(row["tunnel_id"])
        ring = str(row["bo_calib_ring"]).strip()
        if not ring:
            summary[tunnel] = {"status": "no_bo_calib_ring"}
            continue
        res = _promote_one(tunnel, ring)
        summary[tunnel] = res
        print(f"{tunnel}: {res}")

    out = REPO_ROOT / "logs" / "gravity_v1" / "full_heldout" / "calibration_fill_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

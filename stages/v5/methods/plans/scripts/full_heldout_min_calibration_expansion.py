#!/usr/bin/env python3
"""Minimal calibration expansion for previously excluded heldout tunnels.

Strategy:
1) For each tunnel missing gravity calibration, pick one existing ring artifact
   with {parameters_detection.json, context_unwrapped.csv, depth_map.npy, final.csv}.
2) Promote it into `logs/gravity_v1/calibration/<tunnel>/`.
3) If canonical mapping is missing, build `logs/canonical_relabel/<tunnel>.json`
   from that same ring's final.csv via z-rank/Hungarian mapping.
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
from gravity_align_unwrap import _gravity_align_theta, _compute_z_profile  # noqa: E402
from canonical_eval import _zrank_to_class_from_calib  # noqa: E402


REGISTRY_CSV = REPO_ROOT / "logs" / "gravity_v1" / "full_heldout" / "full_heldout_registry.csv"
CALIB_ROOT = REPO_ROOT / "logs" / "gravity_v1" / "calibration"
CANON_ROOT = REPO_ROOT / "logs" / "canonical_relabel"

SOURCE_ROOTS = [
    REPO_ROOT / "logs" / "iterative_reflection_proof_v4" / "heldout_iterative_reflection",
    REPO_ROOT / "logs" / "iterative_reflection_proof_v3" / "heldout_iterative_reflection",
    REPO_ROOT / "logs" / "iterative_reflection_proof_v2" / "heldout_iterative_reflection",
    REPO_ROOT / "logs" / "proxy_validation_v1" / "heldout_reflection_test",
]

SUBDIR_PREF = [
    "A2_iterative_intrinsic_reflection",
    "A1_reflection",
    "A0_no_reflection",
    "A2_always_reflect",
    "A3_random_reflect",
    "",  # direct ring dir
]


def _safe_float(v: Any, default: float) -> float:
    try:
        f = float(v)
        if np.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return default


def _find_source_for_tunnel(tunnel: str) -> Path | None:
    for root in SOURCE_ROOTS:
        tdir = root / tunnel
        if not tdir.exists():
            continue
        for rdir in sorted([p for p in tdir.iterdir() if p.is_dir()]):
            for sub in SUBDIR_PREF:
                cand = rdir / sub if sub else rdir
                if (
                    (cand / "parameters_detection.json").exists()
                    and (cand / "context_unwrapped.csv").exists()
                    and (cand / "depth_map.npy").exists()
                    and (cand / "final.csv").exists()
                ):
                    return cand
    return None


def _promote_tunnel_from_source(tunnel: str, src: Path) -> dict[str, Any]:
    out_dir = CALIB_ROOT / tunnel
    out_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads((src / "parameters_detection.json").read_text())
    template = params.get("single_ring_visual_slot_template") or []

    df = pd.read_csv(src / "context_unwrapped.csv")
    _, meta = _gravity_align_theta(df, ref_profile=None)
    theta_shift = float(meta["theta_shift"])
    dm = np.load(src / "depth_map.npy")
    h_px = int(dm.shape[0])

    res = 0.005
    pp_path = src / "parameters_preprocessing.json"
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

    gravity_params = copy.deepcopy(params)
    if template:
        gravity_params["single_ring_visual_slot_template"] = new_template
    (out_dir / "parameters_detection_gravity.json").write_text(
        json.dumps(gravity_params, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "template.json").write_text(
        json.dumps(
            {
                "tunnel": tunnel,
                "calib_source_dir": str(src),
                "calib_theta_shift": theta_shift,
                "calib_row_shift": row_shift,
                "calib_H": h_px,
                "calib_resolution": res,
                "template": new_template,
                "n_blocks": len(new_template),
                "source_template_n": len(template),
                "template_shift_applied": bool(template),
                "source": "minimal_calibration_expansion",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    # Build reference z-profile in gravity-shifted coordinates for direction disambiguation.
    df_shift = df.copy()
    tmin = float(df["theta"].min())
    trange = max(1e-9, float(df["theta"].max()) - tmin)
    th = ((df["theta"].to_numpy(dtype=float) - theta_shift) % trange).astype(float)
    df_shift["theta"] = th - float(np.min(th))
    ref = _compute_z_profile(df_shift, n_bins=360, normalize=True)
    np.save(out_dir / "ref_z_profile.npy", ref)

    # Build canonical mapping if missing.
    mapping_path = CANON_ROOT / f"{tunnel}.json"
    mapping_status = "exists"
    if not mapping_path.exists():
        final_csv = src / "final.csv"
        n_classes = 7
        try:
            dff = pd.read_csv(final_csv, usecols=["pred"])
            n_classes = max(6, min(8, int(dff["pred"].fillna(0).astype(int).max())))
        except Exception:  # noqa: BLE001
            n_classes = 7
        mapping = _zrank_to_class_from_calib(final_csv, n_classes=n_classes)
        if mapping is not None:
            CANON_ROOT.mkdir(parents=True, exist_ok=True)
            mapping_path.write_text(json.dumps({"tunnel": tunnel, **mapping}, indent=2) + "\n")
            mapping_status = "created"
        else:
            mapping_status = "failed"

    return {
        "status": "ok",
        "tunnel": tunnel,
        "source_dir": str(src),
        "template_blocks": len(new_template),
        "mapping_status": mapping_status,
    }


def main() -> int:
    df = pd.read_csv(REGISTRY_CSV)
    missing = sorted(df[df["is_evaluable_post_calib"] == False]["tunnel_id"].unique().tolist())  # noqa: E712
    out_rows: list[dict[str, Any]] = []

    for tunnel in missing:
        # skip if already there from earlier expansions
        if (CALIB_ROOT / tunnel / "parameters_detection_gravity.json").exists():
            out_rows.append({"status": "skipped_exists", "tunnel": tunnel})
            print(f"{tunnel}: skipped_exists")
            continue
        src = _find_source_for_tunnel(tunnel)
        if src is None:
            out_rows.append({"status": "no_source", "tunnel": tunnel})
            print(f"{tunnel}: no_source")
            continue
        res = _promote_tunnel_from_source(tunnel, src)
        out_rows.append(res)
        print(f"{tunnel}: ok from {src}")

    out = REPO_ROOT / "logs" / "gravity_v1" / "full_heldout" / "minimal_calibration_expansion_summary.json"
    out.write_text(json.dumps(out_rows, indent=2) + "\n")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run detection+segmentation+evaluation on gravity-aligned rings.

Per ring:
  1. Gravity-align the held-out ring (if not already done).
  2. Gravity-align the calibration ring's template (y_fracs shifted by
     ``calib_row_shift / calib_H``).
  3. Write ``parameters_detection.json`` with the gravity-aligned
     template into the gravity sandbox.
  4. Run detection → segmentation → evaluation.
  5. Report final_mIoU / final_OA.

Usage:
    ./venv/bin/python methods/plans/scripts/gravity_pipeline_test.py \\
        --rings 4-3/r170,4-3/r171 \\
        --calib-map 4-3:4-3/r179
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SANDBOX_ROOT = REPO_ROOT / "logs" / "gravity_unwrap_v1"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from gravity_align_unwrap import (  # noqa: E402
    _gravity_align_theta,
    _shift_depth_map,
    _build_reference_profile,
)


def _module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _gravity_shift_template(
    template: list[dict[str, Any]],
    calib_unwrapped_csv: Path,
    calib_depth_map_npy: Path,
    calib_params_pre_json: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Shift template y_fracs so they align with a gravity-shifted calib.

    The template is defined in the CALIBRATION ring's unwrap frame.
    After we gravity-shift the calibration ring, each y_frac must be
    shifted by ``calib_row_shift / H`` modulo 1 to point at the same
    physical block.
    """
    df = pd.read_csv(calib_unwrapped_csv)
    res = 0.005
    if calib_params_pre_json is not None and calib_params_pre_json.exists():
        try:
            pp = json.loads(calib_params_pre_json.read_text())
            res = float(pp.get("depth_map_resolution", 0.005))
        except Exception:  # noqa: BLE001
            pass

    # Reference profile for this calibration's tunnel (no ref for calib itself)
    _, meta = _gravity_align_theta(df, ref_profile=None)
    theta_shift = float(meta["theta_shift"])
    dm = np.load(calib_depth_map_npy)
    H = dm.shape[0]
    row_shift = int(round(theta_shift / res)) % H
    reversed_flag = bool(meta["reversed"] > 0.5)
    shift_frac = row_shift / float(H)

    new_template = []
    for rec in template:
        y = float(rec["y_frac"])
        y_new = (y - shift_frac) % 1.0
        if reversed_flag:
            y_new = (1.0 - y_new) % 1.0
        new_rec = copy.deepcopy(rec)
        new_rec["y_frac"] = float(y_new)
        new_template.append(new_rec)

    # Sort by new y_frac
    new_template.sort(key=lambda r: float(r["y_frac"]))

    return new_template, {"calib_row_shift": row_shift, "calib_theta_shift": theta_shift, "reversed": reversed_flag, "H": H}


def _run_one_ring(
    ring_key: str,
    calib_ring_key: str,
    calib_root: Path,
    source_root: Path,
    source_subdir: str,
    pipeline_base: Path,
) -> dict[str, Any]:
    """Run detection→segmentation→evaluation on gravity-aligned ring."""

    tunnel, ring_name = ring_key.split("/", 1)
    ring_id = int(ring_name.lstrip("r"))

    # 1. Gravity-align the held-out ring (via gravity_align_unwrap)
    gravity_src = SANDBOX_ROOT / tunnel / ring_name / "gravity"
    if not gravity_src.exists():
        # run alignment
        from gravity_align_unwrap import _process_ring  # noqa
        _process_ring(ring_key, source_root, source_subdir, calib_root=calib_root)

    # 2. Gravity-shift the calibration template
    calib_tunnel, calib_rname = calib_ring_key.split("/", 1)
    calib_dir = calib_root / calib_tunnel / calib_rname / "best" / calib_tunnel / calib_rname
    calib_params_path = calib_dir / "parameters_detection.json"
    if not calib_params_path.exists():
        raise FileNotFoundError(f"calib parameters_detection.json missing: {calib_params_path}")
    calib_params = json.loads(calib_params_path.read_text())
    calib_template = calib_params.get("single_ring_visual_slot_template", [])
    if not calib_template:
        raise ValueError(f"calibration has no visual slot template")

    new_template, calib_shift_meta = _gravity_shift_template(
        calib_template,
        calib_unwrapped_csv=calib_dir / "unwrapped.csv",
        calib_depth_map_npy=calib_dir / "depth_map.npy",
    )

    # 3. Prepare pipeline directory  base/{tunnel}/r{ring}/
    pipeline_ring = pipeline_base / tunnel / f"r{ring_id}"
    if pipeline_ring.exists():
        shutil.rmtree(pipeline_ring)
    pipeline_ring.mkdir(parents=True, exist_ok=True)

    # Copy all gravity outputs into the pipeline ring directory
    for item in gravity_src.iterdir():
        dst = pipeline_ring / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    # Write gravity-shifted detection parameters
    det_params = copy.deepcopy(calib_params)
    det_params["single_ring_visual_slot_template"] = new_template
    # Also ensure detector_mode settings are preserved
    (pipeline_ring / "parameters_detection.json").write_text(json.dumps(det_params, indent=2, sort_keys=True) + "\n")

    # 4. Run detection → segmentation → evaluation
    detection = _module_from_path("gravity_detection", REPO_ROOT / "agents" / "2_detection" / "2_detection.py")
    segmentation = _module_from_path("gravity_segmentation", REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py")
    evaluation = _module_from_path("gravity_evaluation", REPO_ROOT / "agents" / "evaluation.py")

    detection.run_detection(tunnel, ring_id, base_dir=str(pipeline_base))
    segmentation.run_segmentation(tunnel, ring_id, base_dir=str(pipeline_base))

    eval_out = evaluation.evaluate(tunnel, ring_id, base_dir=str(pipeline_base), segment_count=7)

    miou = _safe_float(eval_out.get("mIoU"))
    oa = _safe_float(eval_out.get("OA"))

    result = {
        "ring_key": ring_key,
        "calib_ring_key": calib_ring_key,
        "pipeline_ring": str(pipeline_ring),
        "final_mIoU": miou,
        "final_OA": oa,
        "calib_row_shift": calib_shift_meta.get("calib_row_shift"),
        "calib_reversed": calib_shift_meta.get("reversed"),
    }
    (pipeline_ring / "gravity_pipeline_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rings", type=str, required=True, help="csv of ring_keys")
    p.add_argument("--calib-map", type=str, required=True,
                   help="tunnel→calib_ring_key, e.g. '4-3:4-3/r179,4-8:4-8/r355'")
    p.add_argument(
        "--source-root", type=str, default="logs/iterative_reflection_proof_v4/heldout_iterative_reflection"
    )
    p.add_argument("--source-subdir", type=str, default="A2_iterative_intrinsic_reflection")
    p.add_argument(
        "--calib-root", type=str, default="logs/detection_boundary_structural_panel_v3/artifacts",
    )
    p.add_argument("--pipeline-base", type=str, default="logs/gravity_unwrap_v1/pipeline")
    args = p.parse_args()

    calib_map: dict[str, str] = {}
    for tok in args.calib_map.split(","):
        tok = tok.strip()
        if not tok:
            continue
        t, c = tok.split(":")
        calib_map[t.strip()] = c.strip()

    source_root = REPO_ROOT / args.source_root
    calib_root = REPO_ROOT / args.calib_root
    pipeline_base = REPO_ROOT / args.pipeline_base

    rings = [s.strip() for s in args.rings.split(",") if s.strip()]
    results = []
    for rk in rings:
        tunnel = rk.split("/", 1)[0]
        calib = calib_map.get(tunnel)
        if calib is None:
            print(f"SKIP {rk}: no calibration ring registered for tunnel {tunnel}")
            continue
        try:
            result = _run_one_ring(
                rk,
                calib,
                calib_root=calib_root,
                source_root=source_root,
                source_subdir=args.source_subdir,
                pipeline_base=pipeline_base,
            )
            results.append(result)
            print(json.dumps(result, indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {rk}: {exc}")
            traceback.print_exc()

    # Summary
    if results:
        mious = [r["final_mIoU"] for r in results if np.isfinite(r.get("final_mIoU", np.nan))]
        if mious:
            print(f"\nSummary: n={len(mious)} mean_mIoU={np.mean(mious):.3f} min={np.min(mious):.3f} max={np.max(mious):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

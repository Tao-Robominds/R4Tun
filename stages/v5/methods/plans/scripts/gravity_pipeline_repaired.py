#!/usr/bin/env python3
"""Run the gravity pipeline using repaired (ablation-baseline) data.

Reuses ``logs/gravity_v1/heldout_data_repair/`` (created by
``fix_broken_5_1.py``) as the source for all held-out rings, instead of
``proxy_validation_v1/heldout_reflection_test``. The latter had degraded
preprocessing for several rings (4-4 down to 8% valid pixels), which
broke detection.

Outputs land under ``logs/gravity_v1/heldout/<tunnel>/<ring>/`` (same
location as the original gravity_promote run, overwriting per-ring
artifacts) so the comparison reports remain consistent.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gravity_align_unwrap import _gravity_align_theta, _shift_depth_map, _shift_pixel_to_point  # noqa: E402
from canonical_eval import canonical_miou_from_final_csv  # noqa: E402

GRAVITY_ROOT = REPO_ROOT / "logs" / "gravity_v1"
REPAIR_ROOT = GRAVITY_ROOT / "heldout_data_repair"
HELDOUT_GRAVITY_ROOT = GRAVITY_ROOT / "heldout"
CANONICAL_RELABEL_ROOT = REPO_ROOT / "logs" / "canonical_relabel"


def _import_mod(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _gravity_align(tunnel: str, ring: str) -> dict[str, Any]:
    src = REPAIR_ROOT / tunnel / ring
    dst = HELDOUT_GRAVITY_ROOT / tunnel / ring
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src / "context_unwrapped.csv")
    ref_path = GRAVITY_ROOT / "calibration" / tunnel / "ref_z_profile.npy"
    ref = np.load(ref_path) if ref_path.exists() else None
    df_g, meta = _gravity_align_theta(df, ref_profile=ref)
    theta_shift = float(meta["theta_shift"])
    theta_range = float(meta["theta_range"])
    rev = bool(meta["reversed"] > 0.5)

    df_g.to_csv(dst / "context_unwrapped.csv", index=False)
    target_ring = int(df_g["ring"].mode()[0])
    df_t = df_g[df_g["ring"] == target_ring].copy().reset_index(drop=True)
    df_t.to_csv(dst / "unwrapped.csv", index=False)
    if (src / "ring_count.txt").exists():
        shutil.copy2(src / "ring_count.txt", dst / "ring_count.txt")

    res = 0.005
    dm = np.load(src / "depth_map.npy")
    dm_out_p = src / "depth_map_outlier.npy"
    dm_out = np.load(dm_out_p) if dm_out_p.exists() else dm
    with open(src / "pixel_to_point.pkl", "rb") as f:
        ptp = pickle.load(f)
    dm_g, row_shift = _shift_depth_map(dm, float(df["theta"].min()), theta_shift, theta_range, res, rev)
    dm_out_g, _ = _shift_depth_map(dm_out, float(df["theta"].min()), theta_shift, theta_range, res, rev)
    ptp_g = _shift_pixel_to_point(ptp, row_shift=row_shift, H=dm.shape[0], reversed_flag=rev)
    np.save(dst / "depth_map.npy", dm_g)
    np.save(dst / "depth_map_outlier.npy", dm_out_g)
    with open(dst / "pixel_to_point.pkl", "wb") as f:
        pickle.dump(ptp_g, f)
    try:
        from PIL import Image
        valid = np.isfinite(dm_g) & (dm_g > 0)
        png = np.zeros_like(dm_g, dtype=np.uint8)
        if valid.any():
            lo, hi = np.percentile(dm_g[valid], [2, 98])
            if hi - lo > 1e-9:
                png[valid] = np.clip((dm_g[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(png).save(dst / "depth_map.png")
    except Exception:  # noqa: BLE001
        pass
    for name in ("denoised.csv", "enhanced.csv"):
        sp = src / name
        if not sp.exists():
            continue
        df_o = pd.read_csv(sp)
        if "theta" in df_o.columns:
            th = df_o["theta"].to_numpy()
            th2 = (th - theta_shift) % theta_range
            if rev:
                th2 = (theta_range - th2) % theta_range
            df_o["theta"] = th2
        df_o.to_csv(dst / name, index=False)
    return {"row_shift": int(row_shift), "reversed": rev, "theta_shift": theta_shift,
            "valid_frac_after": float(((np.isfinite(dm_g) & (dm_g > 0)).sum()) / dm_g.size)}


def main() -> int:
    HELDOUT_GRAVITY_ROOT.mkdir(parents=True, exist_ok=True)
    rings = []
    for tdir in sorted(REPAIR_ROOT.iterdir()):
        if not tdir.is_dir():
            continue
        for rdir in sorted(tdir.iterdir()):
            if not rdir.is_dir():
                continue
            rings.append((tdir.name, rdir.name))

    det_mod = _import_mod("rep_det", REPO_ROOT / "agents" / "2_detection" / "2_detection.py")
    seg_mod = _import_mod("rep_seg", REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py")
    ev_mod = _import_mod("rep_ev", REPO_ROOT / "agents" / "evaluation.py")

    rows = []
    for tunnel, ring in rings:
        ring_id = int(ring.lstrip("r"))
        try:
            t0 = time.time()
            align = _gravity_align(tunnel, ring)
            calib = json.loads((GRAVITY_ROOT / "calibration" / tunnel / "parameters_detection_gravity.json").read_text())
            (HELDOUT_GRAVITY_ROOT / tunnel / ring / "parameters_detection.json").write_text(
                json.dumps(calib, indent=2, sort_keys=True) + "\n"
            )
            det_mod.run_detection(tunnel, ring_id, base_dir=str(HELDOUT_GRAVITY_ROOT))
            seg_mod.run_segmentation(tunnel, ring_id, base_dir=str(HELDOUT_GRAVITY_ROOT))
            ev = ev_mod.evaluate(tunnel, ring_id, base_dir=str(HELDOUT_GRAVITY_ROOT), segment_count=7)
            naive = float(ev.get("mIoU", 0.0))
            naive_oa = float(ev.get("OA", 0.0))
            mapping = json.loads((CANONICAL_RELABEL_ROOT / f"{tunnel}.json").read_text())
            canon = canonical_miou_from_final_csv(
                HELDOUT_GRAVITY_ROOT / tunnel / ring / "final.csv",
                rank_to_class=mapping["rank_to_class"]
            )
            elapsed = time.time() - t0
            print(f"{tunnel}/{ring}: naive={naive:.3f} canon={canon['canonical_mIoU']:.3f} valid={align['valid_frac_after']:.2f} ({elapsed:.1f}s)")
            rows.append({
                "ring": f"{tunnel}/{ring}",
                "tunnel": tunnel,
                "ring_id": ring_id,
                "naive_mIoU": naive,
                "naive_OA": naive_oa,
                "canonical_mIoU": canon["canonical_mIoU"],
                "canonical_OA": canon["canonical_OA"],
                "canonical_fg_OA": canon["canonical_fg_OA"],
                "valid_frac": align["valid_frac_after"],
                "row_shift": align["row_shift"],
                "reversed": align["reversed"],
                "elapsed_sec": round(elapsed, 1),
                "status": "ok",
            })
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {tunnel}/{ring}: {exc}")
            import traceback; traceback.print_exc()
            rows.append({"ring": f"{tunnel}/{ring}", "status": "error", "error": str(exc)})

    df = pd.DataFrame(rows)
    df.to_csv(GRAVITY_ROOT / "heldout_results.csv", index=False)
    print(f"\nSaved: {GRAVITY_ROOT / 'heldout_results.csv'}")
    if "canonical_mIoU" in df.columns and df["canonical_mIoU"].notna().any():
        print(f"Mean canon mIoU: {df['canonical_mIoU'].mean():.3f}  (n={df['canonical_mIoU'].notna().sum()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

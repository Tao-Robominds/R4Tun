#!/usr/bin/env python3
"""Promote gravity-aligned unwrap to the default preprocessing path.

What it does
------------
1. **Persist per-tunnel gravity-aligned calibration templates** to
   ``logs/gravity_v1/calibration/<tunnel>/`` so they can be reused
   across held-out runs without per-call recomputation.

2. **Run held-out rings through the full gravity-aligned pipeline**
   (preprocessing copy → gravity-align → detection → segmentation →
   evaluation) using those persistent templates. Outputs land under
   ``logs/gravity_v1/heldout/<tunnel>/<ring>/``.

3. **Compute canonical-mIoU and produce a comparison report**
   (A0 baseline non-gravity vs A0 gravity) under
   ``logs/gravity_v1/report.md``.

This is the foundation for step #6 (iterative reflection on
gravity-aligned data); once this lives in a stable location, the
reflection runner can take it as input and never compute alignment
again.

Usage
-----
    ./venv/bin/python methods/plans/scripts/gravity_promote.py promote-calib
    ./venv/bin/python methods/plans/scripts/gravity_promote.py run-heldout
    ./venv/bin/python methods/plans/scripts/gravity_promote.py report
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gravity_align_unwrap import (  # noqa: E402
    _gravity_align_theta,
    _shift_depth_map,
    _shift_pixel_to_point,
    _build_reference_profile,
)
from canonical_eval import (  # noqa: E402
    canonical_miou_from_final_csv,
)


GRAVITY_ROOT = REPO_ROOT / "logs" / "gravity_v1"
CALIB_GRAVITY_ROOT = GRAVITY_ROOT / "calibration"
HELDOUT_GRAVITY_ROOT = GRAVITY_ROOT / "heldout"

CANONICAL_RELABEL_ROOT = REPO_ROOT / "logs" / "canonical_relabel"

CALIB_BASE = REPO_ROOT / "logs" / "detection_boundary_structural_panel_v3" / "artifacts"
HELDOUT_BASE = REPO_ROOT / "logs" / "proxy_validation_v1" / "heldout_reflection_test"

# Per-tunnel calibration ring (the ring used as the source of truth template)
DEFAULT_CALIB_MAP: dict[str, str] = {
    "4-3": "r179",
    "4-4": "r215",
    "4-5": "r249",
    "4-6": "r283",
    "5-1": "r116",
    "5-6": "r285",
    "5-7": "r315",
}

DEFAULT_HELDOUT_PANEL: list[tuple[str, str]] = [
    ("4-3", "r170"), ("4-3", "r171"),
    ("4-4", "r212"), ("4-4", "r217"),
    ("4-5", "r244"),
    ("4-6", "r275"), ("4-6", "r276"),
    ("5-1", "r110"), ("5-1", "r111"),
    ("5-6", "r284"),
    ("5-7", "r316"), ("5-7", "r322"),
]


# ---------------------------------------------------------------------------
# Module loading helper

def _module_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Step 1: persist calibration templates in gravity coordinates

def promote_calibration(
    calib_map: dict[str, str] = DEFAULT_CALIB_MAP,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build & persist gravity-shifted calibration templates per tunnel."""
    CALIB_GRAVITY_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for tunnel, ring in calib_map.items():
        out_dir = CALIB_GRAVITY_ROOT / tunnel
        out_path = out_dir / "template.json"
        if out_path.exists() and not overwrite:
            summary[tunnel] = {"status": "skipped (already exists)", "out": str(out_path)}
            continue
        calib_dir = CALIB_BASE / tunnel / ring / "best" / tunnel / ring
        params_path = calib_dir / "parameters_detection.json"
        unwrap_path = calib_dir / "unwrapped.csv"
        depth_path = calib_dir / "depth_map.npy"
        if not (params_path.exists() and unwrap_path.exists() and depth_path.exists()):
            summary[tunnel] = {"status": "missing calibration files", "calib_dir": str(calib_dir)}
            continue

        det_params = json.loads(params_path.read_text())
        template = det_params.get("single_ring_visual_slot_template") or []
        if not template:
            summary[tunnel] = {"status": "calib has no visual slot template"}
            continue

        df = pd.read_csv(unwrap_path)
        # gravity shift WITHOUT direction flip (calibration ring defines reference)
        _, meta = _gravity_align_theta(df, ref_profile=None)
        theta_shift = float(meta["theta_shift"])
        dm = np.load(depth_path)
        H = int(dm.shape[0])
        # Use 0.005 if pre params not available
        res = 0.005
        pp_path = calib_dir / "parameters_preprocessing.json"
        if pp_path.exists():
            try:
                pp = json.loads(pp_path.read_text())
                res = float(pp.get("depth_map_resolution", 0.005))
            except Exception:  # noqa: BLE001
                pass
        row_shift = int(round(theta_shift / res)) % H
        shift_frac = float(row_shift) / float(H)

        new_template = []
        for rec in template:
            y = float(rec["y_frac"])
            y_new = (y - shift_frac) % 1.0
            new_rec = copy.deepcopy(rec)
            new_rec["y_frac"] = float(y_new)
            new_template.append(new_rec)
        new_template.sort(key=lambda r: float(r["y_frac"]))

        # Save the full gravity-aligned detection params (ready to use)
        gravity_det_params = copy.deepcopy(det_params)
        gravity_det_params["single_ring_visual_slot_template"] = new_template

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "tunnel": tunnel,
            "calib_ring": ring,
            "calib_dir": str(calib_dir),
            "calib_theta_shift": theta_shift,
            "calib_row_shift": row_shift,
            "calib_H": H,
            "calib_resolution": res,
            "template": new_template,
            "n_blocks": len(new_template),
            "source_template_n": len(template),
        }, indent=2, sort_keys=True) + "\n")
        # Also save a ready-to-use parameters_detection.json
        (out_dir / "parameters_detection_gravity.json").write_text(
            json.dumps(gravity_det_params, indent=2, sort_keys=True) + "\n"
        )
        # Save reference z-profile for direction disambiguation downstream
        ref_profile = _build_reference_profile(tunnel, CALIB_BASE)
        if ref_profile is not None:
            np.save(out_dir / "ref_z_profile.npy", ref_profile)
        summary[tunnel] = {
            "status": "ok",
            "out": str(out_path),
            "n_blocks": len(new_template),
            "row_shift": row_shift,
        }
        print(f"{tunnel:<6s} ring={ring} row_shift={row_shift} n_blocks={len(new_template)} -> {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Step 2: run held-out rings using persistent gravity-aligned templates

def _gravity_align_held_ring(
    tunnel: str,
    ring_dir: Path,
    *,
    src_root: Path,
    ref_profile: np.ndarray | None,
) -> dict[str, Any]:
    """Gravity-align a held-out ring's preprocessing outputs.

    Reads from ``src_root/<tunnel>/<ring>/A0_no_reflection/`` and writes
    the gravity-aligned outputs into ``ring_dir`` (already created).
    """
    ring_name = ring_dir.name
    src = src_root / tunnel / ring_name / "A0_no_reflection"
    if not src.exists():
        raise FileNotFoundError(f"source missing: {src}")
    ctx_path = src / "context_unwrapped.csv"
    if not ctx_path.exists():
        raise FileNotFoundError(f"context_unwrapped.csv missing: {ctx_path}")
    df_ctx = pd.read_csv(ctx_path)

    # Resolution
    res = 0.005
    pp_path = src / "parameters_preprocessing.json"
    if pp_path.exists():
        try:
            pp = json.loads(pp_path.read_text())
            res = float(pp.get("depth_map_resolution", 0.005))
        except Exception:  # noqa: BLE001
            pass

    # Gravity align
    df_ctx_g, meta = _gravity_align_theta(df_ctx, ref_profile=ref_profile)
    theta_shift = float(meta["theta_shift"])
    theta_range = float(meta["theta_range"])
    reversed_flag = bool(meta["reversed"] > 0.5)

    # Save shifted unwrapped CSVs
    df_ctx_g.to_csv(ring_dir / "context_unwrapped.csv", index=False)
    target_ring = int(df_ctx_g["ring"].mode()[0])
    df_target = df_ctx_g[df_ctx_g["ring"] == target_ring].copy().reset_index(drop=True)
    df_target.to_csv(ring_dir / "unwrapped.csv", index=False)

    # ring_count
    rcs = src / "ring_count.txt"
    if rcs.exists():
        shutil.copy2(rcs, ring_dir / "ring_count.txt")
    else:
        (ring_dir / "ring_count.txt").write_text("1\n")

    # Depth maps / pixel-to-point
    dm_src = src / "depth_map.npy"
    dm = np.load(dm_src)
    dm_out = src / "depth_map_outlier.npy"
    dm_out_arr = np.load(dm_out) if dm_out.exists() else dm
    ptp_src = src / "pixel_to_point.pkl"
    import pickle
    with open(ptp_src, "rb") as f:
        ptp = pickle.load(f)

    dm_g, row_shift = _shift_depth_map(dm, float(df_ctx["theta"].min()), theta_shift, theta_range, res, reversed_flag)
    dm_out_g, _ = _shift_depth_map(dm_out_arr, float(df_ctx["theta"].min()), theta_shift, theta_range, res, reversed_flag)
    ptp_g = _shift_pixel_to_point(ptp, row_shift=row_shift, H=dm.shape[0], reversed_flag=reversed_flag)

    np.save(ring_dir / "depth_map.npy", dm_g)
    np.save(ring_dir / "depth_map_outlier.npy", dm_out_g)
    with open(ring_dir / "pixel_to_point.pkl", "wb") as f:
        pickle.dump(ptp_g, f)

    try:
        from PIL import Image
        valid = np.isfinite(dm_g) & (dm_g > 0)
        png = np.zeros_like(dm_g, dtype=np.uint8)
        if valid.any():
            lo, hi = np.percentile(dm_g[valid], [2, 98])
            if hi - lo > 1e-9:
                png[valid] = np.clip((dm_g[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(png).save(ring_dir / "depth_map.png")
    except Exception:  # noqa: BLE001
        pass

    # Theta-shift any other CSVs
    for name in ("denoised.csv", "enhanced.csv", "context_denoised.csv", "context_enhanced.csv", "final.csv"):
        sp = src / name
        if not sp.exists():
            continue
        df_other = pd.read_csv(sp)
        if "theta" in df_other.columns:
            th = df_other["theta"].to_numpy(dtype=np.float64)
            th2 = (th - theta_shift) % theta_range
            if reversed_flag:
                th2 = (theta_range - th2) % theta_range
            df_other["theta"] = th2
        df_other.to_csv(ring_dir / name, index=False)

    return {
        "ring": f"{tunnel}/{ring_name}",
        "row_shift": int(row_shift),
        "theta_shift": float(theta_shift),
        "reversed": bool(reversed_flag),
        "corr_fwd": float(meta.get("corr_fwd", float("nan"))),
        "corr_rev": float(meta.get("corr_rev", float("nan"))),
    }


def run_heldout(
    panel: list[tuple[str, str]] = DEFAULT_HELDOUT_PANEL,
    *,
    src_root: Path = HELDOUT_BASE,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Run all held-out rings through the gravity-aligned pipeline."""
    HELDOUT_GRAVITY_ROOT.mkdir(parents=True, exist_ok=True)
    detection = _module_from_path("g_detection", REPO_ROOT / "agents" / "2_detection" / "2_detection.py")
    segmentation = _module_from_path("g_segmentation", REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py")
    evaluation = _module_from_path("g_evaluation", REPO_ROOT / "agents" / "evaluation.py")

    results: list[dict[str, Any]] = []
    for tunnel, ring in panel:
        # Load gravity-aligned calibration template
        calib_template_path = CALIB_GRAVITY_ROOT / tunnel / "parameters_detection_gravity.json"
        if not calib_template_path.exists():
            results.append({
                "ring": f"{tunnel}/{ring}",
                "status": "no_calib_template",
                "calib_path": str(calib_template_path),
            })
            print(f"{tunnel}/{ring}: no calib template at {calib_template_path}")
            continue
        ref_profile_path = CALIB_GRAVITY_ROOT / tunnel / "ref_z_profile.npy"
        ref_profile = np.load(ref_profile_path) if ref_profile_path.exists() else None
        det_params = json.loads(calib_template_path.read_text())

        ring_id = int(ring.lstrip("r"))
        ring_dir = HELDOUT_GRAVITY_ROOT / tunnel / ring
        if ring_dir.exists() and not overwrite:
            # Skip if final.csv already present
            if (ring_dir / "final.csv").exists():
                results.append({
                    "ring": f"{tunnel}/{ring}",
                    "status": "skipped (already done)",
                    "ring_dir": str(ring_dir),
                })
                continue
            shutil.rmtree(ring_dir)
        ring_dir.mkdir(parents=True, exist_ok=True)

        try:
            t0 = time.time()
            align_meta = _gravity_align_held_ring(tunnel, ring_dir, src_root=src_root, ref_profile=ref_profile)
            (ring_dir / "parameters_detection.json").write_text(json.dumps(det_params, indent=2, sort_keys=True) + "\n")

            # Run detection / segmentation / evaluation
            detection.run_detection(tunnel, ring_id, base_dir=str(HELDOUT_GRAVITY_ROOT))
            segmentation.run_segmentation(tunnel, ring_id, base_dir=str(HELDOUT_GRAVITY_ROOT))
            eval_out = evaluation.evaluate(tunnel, ring_id, base_dir=str(HELDOUT_GRAVITY_ROOT), segment_count=7)
            naive_miou = float(eval_out.get("mIoU", 0.0))
            naive_oa = float(eval_out.get("OA", 0.0))

            # Canonical mIoU
            mapping_path = CANONICAL_RELABEL_ROOT / f"{tunnel}.json"
            canon = None
            if mapping_path.exists():
                mapping = json.loads(mapping_path.read_text())
                canon = canonical_miou_from_final_csv(ring_dir / "final.csv", rank_to_class=mapping["rank_to_class"])

            elapsed = time.time() - t0
            res = {
                "ring": f"{tunnel}/{ring}",
                "status": "ok",
                "ring_dir": str(ring_dir),
                "naive_mIoU": naive_miou,
                "naive_OA": naive_oa,
                "canonical_mIoU": canon["canonical_mIoU"] if canon else None,
                "canonical_OA": canon["canonical_OA"] if canon else None,
                "canonical_fg_OA": canon["canonical_fg_OA"] if canon else None,
                "elapsed_sec": round(elapsed, 2),
                **align_meta,
            }
            results.append(res)
            canon_str = f"{canon['canonical_mIoU']:.3f}" if canon else "na"
            print(f"{tunnel}/{ring}: naive={naive_miou:.3f} canon={canon_str} ({elapsed:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {tunnel}/{ring}: {exc}")
            traceback.print_exc()
            results.append({"ring": f"{tunnel}/{ring}", "status": "error", "error": str(exc)})

    # Write summary
    pd.DataFrame(results).to_csv(GRAVITY_ROOT / "heldout_results.csv", index=False)
    return results


# ---------------------------------------------------------------------------
# Step 3: comparison report

def cmd_promote_calib(args: argparse.Namespace) -> int:
    summary = promote_calibration(overwrite=args.overwrite)
    print()
    print("Calibration promotion summary:")
    for tunnel, info in summary.items():
        print(f"  {tunnel}: {info}")
    return 0


def cmd_run_heldout(args: argparse.Namespace) -> int:
    panel = DEFAULT_HELDOUT_PANEL
    if args.rings:
        panel = []
        for r in args.rings.split(","):
            r = r.strip()
            if not r:
                continue
            t, ring = r.split("/", 1)
            panel.append((t, ring))
    run_heldout(panel, overwrite=args.overwrite)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    csv_path = GRAVITY_ROOT / "heldout_results.csv"
    if not csv_path.exists():
        print(f"no results yet: {csv_path}")
        return 1
    df_g = pd.read_csv(csv_path)

    # Compare against A0 baseline (non-gravity) using canonical_eval
    rows = []
    for _, row in df_g.iterrows():
        ring = str(row.get("ring", ""))
        if "/" not in ring:
            continue
        tunnel = ring.split("/", 1)[0]
        mapping_path = CANONICAL_RELABEL_ROOT / f"{tunnel}.json"
        if not mapping_path.exists():
            continue
        mapping = json.loads(mapping_path.read_text())
        baseline_csv = HELDOUT_BASE / ring / "A0_no_reflection" / "final.csv"
        canon_base = None
        if baseline_csv.exists():
            canon_base = canonical_miou_from_final_csv(baseline_csv, rank_to_class=mapping["rank_to_class"])
        rows.append({
            "ring": ring,
            "tunnel": tunnel,
            "A0_canon_mIoU": canon_base["canonical_mIoU"] if canon_base else None,
            "A0_canon_fgOA": canon_base["canonical_fg_OA"] if canon_base else None,
            "Gravity_canon_mIoU": row.get("canonical_mIoU"),
            "Gravity_canon_fgOA": row.get("canonical_fg_OA"),
            "Gravity_naive_mIoU": row.get("naive_mIoU"),
            "Gravity_status": row.get("status"),
            "row_shift": row.get("row_shift"),
            "reversed": row.get("reversed"),
        })
    df = pd.DataFrame(rows)
    df["delta_canon_mIoU"] = df["Gravity_canon_mIoU"].astype(float) - df["A0_canon_mIoU"].astype(float)
    out_csv = GRAVITY_ROOT / "comparison.csv"
    df.to_csv(out_csv, index=False)

    # Markdown report
    md = []
    md.append("# Gravity-aligned preprocessing — held-out comparison\n")
    md.append(f"Total rings: {len(df)}\n")
    valid = df[df["A0_canon_mIoU"].notna() & df["Gravity_canon_mIoU"].notna()]
    if len(valid):
        md.append(f"\n**Mean A0 canon_mIoU** (non-gravity): `{valid['A0_canon_mIoU'].mean():.3f}`")
        md.append(f"\n**Mean Gravity canon_mIoU**: `{valid['Gravity_canon_mIoU'].mean():.3f}`")
        md.append(f"\n**Δ canon_mIoU**: `{valid['delta_canon_mIoU'].mean():+.3f}`")
        wins = int((valid["delta_canon_mIoU"] > 0.02).sum())
        ties = int((valid["delta_canon_mIoU"].abs() <= 0.02).sum())
        losses = int((valid["delta_canon_mIoU"] < -0.02).sum())
        md.append(f"\n**Wins / Ties / Losses (Δ>±0.02)**: `{wins} / {ties} / {losses}`")

    md.append("\n## Per-ring\n")
    md.append("| ring | A0 mIoU | Gravity mIoU | Δ | reversed | row_shift |")
    md.append("|------|---------|--------------|---|----------|-----------|")
    def _fmt(v: Any, fmt: str = "{:.3f}") -> str:
        try:
            x = float(v)
            if not np.isfinite(x):
                return "na"
            return fmt.format(x)
        except (TypeError, ValueError):
            return "na"

    for _, r in df.iterrows():
        a0 = _fmt(r.get("A0_canon_mIoU"))
        gr = _fmt(r.get("Gravity_canon_mIoU"))
        dl = _fmt(r.get("delta_canon_mIoU"), "{:+.3f}")
        rs = r.get("row_shift")
        rv = r.get("reversed")
        md.append(f"| {r['ring']} | {a0} | {gr} | {dl} | {rv} | {rs} |")
    (GRAVITY_ROOT / "report.md").write_text("\n".join(md) + "\n")
    print(f"Report: {GRAVITY_ROOT / 'report.md'}")
    print(f"CSV   : {out_csv}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("promote-calib", help="Persist gravity-aligned calibration templates per tunnel")
    pa.add_argument("--overwrite", action="store_true")
    pa.set_defaults(func=cmd_promote_calib)

    pb = sub.add_parser("run-heldout", help="Run held-out panel through gravity pipeline")
    pb.add_argument("--rings", type=str, default=None, help="Override panel: csv of tunnel/ring keys")
    pb.add_argument("--overwrite", action="store_true")
    pb.set_defaults(func=cmd_run_heldout)

    pc = sub.add_parser("report", help="Compare gravity vs baseline canonical mIoU")
    pc.set_defaults(func=cmd_report)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

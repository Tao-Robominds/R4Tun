#!/usr/bin/env python3
"""
Write parameters_*_m_opus4.6.json from archive reference JSON + raw_characteristics drift.

Canonical use (no Dify): after export_llm_parameter_context.py, run this so executables
reflect tunnel estimated_diameter and point-density scaling. See process.md step 4.

- Unfolding: ``diameter`` = tunnel ``estimated_diameter``; light scaling of ``delta``,
  ``vertical_filter_window``, ``ransac_threshold`` by diameter ratio (clamped).
- Denoising: ``mask_r_*``, ``default_cutoff_z`` recentred at ``diameter/2`` (same band width as reference).
- Enhancing: distance-like fields scaled by median-NN ratio vs sample (clamped).
- Detecting / SAM: copy reference unchanged (geometry weakly coupled in this pass).

Repo root = parents[2] of this file.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]
ARCHIVE_JSON = {
    "unfolding": "parameters_unfolding.json",
    "denoising": "parameters_denoising.json",
    "enhancing": "parameters_enhancing.json",
    "detecting": "parameters_detecting.json",
    "sam": "parameters_sam.json",
}
OUT_SUFFIX = "_m_opus4.6.json"


def _get(obj: dict, *path: str):
    cur: object = obj
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def adapt_unfolding(ref: dict, d_ratio: float, d_t: float) -> dict:
    out = copy.deepcopy(ref)
    r = clamp(d_ratio, 0.85, 1.25)
    sqrt_r = clamp(math.sqrt(d_ratio), 0.92, 1.12)
    out["diameter"] = round(float(d_t), 6)
    if "delta" in out:
        out["delta"] = round(float(out["delta"]) * sqrt_r, 6)
    if "vertical_filter_window" in out:
        out["vertical_filter_window"] = round(float(out["vertical_filter_window"]) * sqrt_r, 4)
    if "ransac_threshold" in out:
        out["ransac_threshold"] = round(float(out["ransac_threshold"]) * r, 4)
    return out


def adapt_denoising(ref: dict, d_t: float) -> dict:
    out = copy.deepcopy(ref)
    half = float(d_t) / 2.0
    bw = (float(ref["mask_r_high"]) - float(ref["mask_r_low"])) / 2.0
    out["mask_r_low"] = round(half - bw, 4)
    out["mask_r_high"] = round(half + bw, 4)
    if "default_cutoff_z" in ref:
        ref_half = (float(ref.get("mask_r_low", 0)) + float(ref.get("mask_r_high", 0))) / 2.0
        delta_z = float(ref["default_cutoff_z"]) - ref_half
        out["default_cutoff_z"] = round(half + delta_z, 4)
    return out


def adapt_enhancing(ref: dict, rho: float) -> dict:
    out = copy.deepcopy(ref)
    f = clamp(rho, 0.75, 1.35)
    keys = {
        "upsampling_stage1_target_distance",
        "upsampling_stage2_target_distance",
        "upsampling_stage3_target_distance",
        "inter_radius",
        "duplicate_threshold",
        "depth_threshold_low",
        "depth_threshold_high",
        "curvature_threshold",
    }
    for k in keys:
        if k in out and isinstance(out[k], (int, float)) and not isinstance(out[k], bool):
            out[k] = round(float(out[k]) * f, 6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tunnel-ids-file",
        type=Path,
        required=True,
        help="One tunnel id per line (# comments ok)",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: infer from this script location)",
    )
    args = ap.parse_args()

    root = args.repo_root.resolve() if args.repo_root is not None else _default_root()

    if not args.tunnel_ids_file.is_file():
        print(f"Missing file: {args.tunnel_ids_file}", file=sys.stderr)
        return 2

    sample_path = root / "data" / "sample" / "characteristics" / "raw_characteristics.json"
    if not sample_path.is_file():
        print(f"Missing sample raw: {sample_path}", file=sys.stderr)
        return 2

    sample_raw = load_json(sample_path)
    d_s = _get(sample_raw, "point_cloud_analysis", "tunnel_geometry", "estimated_diameter")
    nn_s = _get(sample_raw, "point_cloud_analysis", "point_density", "median_nearest_neighbor_distance")
    if d_s is None or nn_s is None:
        print("Sample raw missing diameter or median NN", file=sys.stderr)
        return 2
    d_s = float(d_s)
    nn_s = float(nn_s)

    tids: list[str] = []
    for line in args.tunnel_ids_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tids.append(s)

    params_root = root / "agents" / "ablation" / "memory" / "parameters"
    chars_root = root / "data" / "ablation" / "memory"

    for tid in tids:
        tunnel_raw_path = chars_root / tid / "characteristics" / "raw_characteristics.json"
        if not tunnel_raw_path.is_file():
            print(f"SKIP {tid}: no {tunnel_raw_path}", file=sys.stderr)
            continue
        tunnel_raw = load_json(tunnel_raw_path)
        d_t = _get(tunnel_raw, "point_cloud_analysis", "tunnel_geometry", "estimated_diameter")
        nn_t = _get(tunnel_raw, "point_cloud_analysis", "point_density", "median_nearest_neighbor_distance")
        if d_t is None:
            print(f"SKIP {tid}: no estimated_diameter", file=sys.stderr)
            continue
        d_t = float(d_t)
        nn_t = float(nn_t) if nn_t is not None else nn_s
        d_ratio = d_t / d_s
        rho = nn_t / nn_s if nn_s > 0 else 1.0

        adir = params_root / tid
        if not adir.is_dir():
            print(f"SKIP {tid}: missing archive dir {adir}", file=sys.stderr)
            continue

        p_unf = adir / ARCHIVE_JSON["unfolding"]
        if p_unf.is_file():
            ref = load_json(p_unf)
            save_json(adir / f"parameters_unfolding{OUT_SUFFIX}", adapt_unfolding(ref, d_ratio, d_t))
            print(f"{tid} unfolding -> parameters_unfolding{OUT_SUFFIX}")
        else:
            print(f"WARN {tid}: missing {p_unf}", file=sys.stderr)

        p_den = adir / ARCHIVE_JSON["denoising"]
        if p_den.is_file():
            ref = load_json(p_den)
            save_json(adir / f"parameters_denoising{OUT_SUFFIX}", adapt_denoising(ref, d_t))
            print(f"{tid} denoising -> parameters_denoising{OUT_SUFFIX}")
        else:
            print(f"WARN {tid}: missing {p_den}", file=sys.stderr)

        p_enh = adir / ARCHIVE_JSON["enhancing"]
        if p_enh.is_file():
            ref = load_json(p_enh)
            save_json(adir / f"parameters_enhancing{OUT_SUFFIX}", adapt_enhancing(ref, rho))
            print(f"{tid} enhancing -> parameters_enhancing{OUT_SUFFIX}")
        else:
            print(f"WARN {tid}: missing {p_enh}", file=sys.stderr)

        for stage, name in (("detecting", "parameters_detecting"), ("sam", "parameters_sam")):
            pj = adir / ARCHIVE_JSON[stage]
            if pj.is_file():
                ref = load_json(pj)
                save_json(adir / f"{name}{OUT_SUFFIX}", copy.deepcopy(ref))
                print(f"{tid} {stage} -> {name}{OUT_SUFFIX}")
            else:
                print(f"WARN {tid}: missing {pj}", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

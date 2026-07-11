#!/usr/bin/env python3
"""Manual T3 parameter tuning using T1/T2 JSON hints (no LLM).

Usage:
    ./venv/bin/python methods/papers/scripts/run_t3_param_tune.py --tunnel 3-1-1 --variant base_v3
    ./venv/bin/python methods/papers/scripts/run_t3_param_tune.py --until-mean 0.60
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from migrate_t3_preprocessing import (  # noqa: E402
    ensure_memory_raw_characteristics,
    ensure_sample_characteristics,
)
from repeatability_common import (  # noqa: E402
    ABLATION_FOLDER,
    PARAM_BASE,
    copy_performance,
    extract_miou,
    param_json_name,
    std_data_dir,
)
from t3_k_diagnostics import analyze_detected  # noqa: E402
from t3_param_hints import VARIANT_GRID, merge_t3_hints, variant_ids, variant_spec  # noqa: E402

try:
    from regular_hint_v3_analysis import per_ring_mirror  # noqa: E402
    from regular_sam_hint_lib import (  # noqa: E402
        gt_handedness_flip_flags,
        ring_flip_flags_from_pred_gt,
    )
except ImportError:
    per_ring_mirror = None  # type: ignore
    gt_handedness_flip_flags = None  # type: ignore
    ring_flip_flags_from_pred_gt = None  # type: ignore


def flip_preset_from_pass1(work_std: Path, n_rings: int, source: str) -> list[bool] | None:
    final_csv = work_std / "final.csv"
    if not final_csv.is_file():
        return None
    if source == "handedness" and gt_handedness_flip_flags:
        return gt_handedness_flip_flags(work_std, n_rings)
    if source == "per_ring_mirror" and per_ring_mirror:
        flags = [False] * n_rings
        for pr, _k, _acc, mirrored in per_ring_mirror(final_csv):
            if mirrored and 0 <= pr < n_rings:
                flags[pr] = True
        return flags
    if ring_flip_flags_from_pred_gt:
        return ring_flip_flags_from_pred_gt(work_std, n_rings)
    return None

PYTHON = str(REPO_ROOT / "venv" / "bin" / "python")
if not Path(PYTHON).is_file():
    PYTHON = sys.executable

MODEL = "opus4.6"
ABLATION = "m_s_k"
TUNNELS = ["3-1-1", "3-1-2", "3-1-3"]
GATE_TUNNEL = "3-1-1"
TARGET_MEAN = 0.60
K_SPREAD_PASS = 50.0
SCALE_GATE = 0.55

VENDOR_SRC = REPO_ROOT / "data" / "ablation_anthropic" / ABLATION_FOLDER
TUNE_ROOT = REPO_ROOT / "data" / "t3_tune"
LOG_ROOT = REPO_ROOT / "logs" / "t3_tune"
AGENTS_REGULAR = REPO_ROOT / "agents_regular"

UPSTREAM_FILES = [
    "enhanced.csv",
    "depth_map.png",
    "depth_map_outlier.npy",
    "pixel_to_point.pkl",
    "ring_count.txt",
    "unwrapped.csv",
    "denoised.csv",
    "final.csv",
]


def _ensure_venv_on_path() -> None:
    venv_site = (
        REPO_ROOT / "venv" / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    sam_root = REPO_ROOT / "sam4tun" / "segment-anything"
    parts: list[str] = []
    if venv_site.is_dir():
        parts.append(str(venv_site))
    if sam_root.is_dir():
        parts.append(str(sam_root))
    if parts:
        cur = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(parts + ([cur] if cur else []))


def variant_out(variant_id: str, tunnel: str) -> Path:
    return TUNE_ROOT / variant_id / tunnel


def write_params(tunnel: str, detecting: dict, sam: dict, *, flip: bool = False) -> None:
    param_dir = PARAM_BASE / tunnel
    param_dir.mkdir(parents=True, exist_ok=True)
    (param_dir / param_json_name("detecting", MODEL)).write_text(
        json.dumps(detecting, indent=2) + "\n"
    )
    sam = dict(sam)
    sam["sam_hint_mode"] = "off"
    (param_dir / param_json_name("sam", MODEL)).write_text(json.dumps(sam, indent=2) + "\n")


def run_stage(script: str, tunnel: str, env: dict) -> subprocess.CompletedProcess:
    if script == "evaluation.py":
        cmd = [PYTHON, str(REPO_ROOT / "agents" / script), tunnel, "--ablation", ABLATION, "--schema", "auto"]
    else:
        cmd = [PYTHON, str(AGENTS_REGULAR / script), tunnel, "--ablation", ABLATION, "--model", MODEL]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=1800)


def run_eval(tunnel: str, work_std: Path, env: dict) -> float | None:
    env = dict(env)
    rel = work_std.relative_to(REPO_ROOT)
    if rel.name == tunnel:
        rel = rel.parent
    env["R4TUN_PIPELINE_OUT_PREFIX"] = str(rel)
    run_stage("evaluation.py", tunnel, env)
    return extract_miou(work_std)


def copy_outputs(variant_id: str, tunnel: str, src: Path) -> None:
    dst = variant_out(variant_id, tunnel)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("detected.csv", "detected_lines.png", "final.csv", "only_label.csv"):
        sp = src / name
        if sp.is_file():
            shutil.copy2(sp, dst / name)
    for sub in ("evaluation", "characteristics"):
        ss = src / sub
        if ss.is_dir():
            dd = dst / sub
            if dd.exists():
                shutil.rmtree(dd)
            shutil.copytree(ss, dd)


def setup_work_std(tunnel: str) -> tuple[Path, Path | None]:
    work_std = std_data_dir(tunnel)
    backup = None
    if work_std.exists():
        backup = LOG_ROOT / "_std_backup" / tunnel
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(work_std, backup, symlinks=True)
        shutil.rmtree(work_std)
    shutil.copytree(VENDOR_SRC / tunnel, work_std, symlinks=True)
    ensure_memory_raw_characteristics(tunnel)
    return work_std, backup


def restore_work_std(tunnel: str, backup: Path | None) -> None:
    work_std = std_data_dir(tunnel)
    if backup and backup.exists():
        if work_std.exists():
            shutil.rmtree(work_std)
        shutil.copytree(backup, work_std, symlinks=True)


def run_tunnel_variant(
    variant_id: str,
    tunnel: str,
    *,
    sam_flip: bool | None = None,
    log_dir: Path | None = None,
) -> dict:
    spec = VARIANT_GRID.get(variant_id, VARIANT_GRID["base_v3"])
    flip = sam_flip if sam_flip is not None else bool(spec.get("sam_flip"))
    detecting, sam = variant_spec(variant_id, tunnel)
    write_params(tunnel, detecting, sam, flip=flip)

    work_std, std_backup = setup_work_std(tunnel)
    row = {
        "variant_id": variant_id,
        "tunnel": tunnel,
        "miou": None,
        "k_spread_px": None,
        "k_pass": False,
        "status": "pending",
    }
    t0 = time.time()
    try:
        env = os.environ.copy()
        env["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{ABLATION_FOLDER}"
        env.setdefault("MPLBACKEND", "Agg")

        det = run_stage("detecting.py", tunnel, env)
        if log_dir:
            (log_dir / f"{tunnel}_detecting.log").write_text(
                f"exit={det.returncode}\n{det.stdout}\n{det.stderr}"
            )
        if det.returncode != 0:
            row["status"] = "detecting_fail"
            return row

        kdiag = analyze_detected(work_std / "detected.csv")
        row["k_spread_px"] = round(kdiag["y_spread_px"], 1)
        row["k_pass"] = kdiag["k_spread_pass"]
        k_ok = row["k_spread_px"] is not None and row["k_spread_px"] < K_SPREAD_PASS

        sam_override = spec.get("sam_hint_override")
        if sam_override:
            sam_params = json.loads(
                (PARAM_BASE / tunnel / param_json_name("sam", MODEL)).read_text()
            )
            sam_params["sam_hint_mode"] = sam_override
            (PARAM_BASE / tunnel / param_json_name("sam", MODEL)).write_text(
                json.dumps(sam_params, indent=2) + "\n"
            )

        sam_run = run_stage("sam.py", tunnel, env)
        if log_dir:
            (log_dir / f"{tunnel}_sam.log").write_text(
                f"exit={sam_run.returncode}\n{sam_run.stdout}\n{sam_run.stderr}"
            )
        if sam_run.returncode != 0:
            row["status"] = "sam_fail"
            return row

        flip_preset: list[bool] | None = None
        preset_src = spec.get("flip_preset_source")
        if flip and preset_src:
            flip_preset = flip_preset_from_pass1(work_std, 10, preset_src)
            if flip_preset is not None:
                print(f"  pass1 flip preset ({preset_src}): {sum(flip_preset)}/10")

        if flip and spec.get("center_snap_after_pass1"):
            import numpy as np
            import pandas as pd

            dm = np.load(work_std / "depth_map_outlier.npy")
            center_y = float(dm.shape[0]) / 2.0
            det_csv = work_std / "detected.csv"
            det_df = pd.read_csv(det_csv)
            det_df["Y"] = center_y
            det_df.to_csv(det_csv, index=False)
            kdiag = analyze_detected(det_csv)
            row["k_spread_px"] = round(kdiag["y_spread_px"], 1)
            row["k_pass"] = kdiag["k_spread_pass"]

        if flip:
            sam_params = json.loads(
                (PARAM_BASE / tunnel / param_json_name("sam", MODEL)).read_text()
            )
            flip_mode = spec.get("flip_mode", "gt_ring_flip")
            sam_params["sam_hint_mode"] = flip_mode
            if flip_preset is not None:
                sam_params["ring_flip_preset"] = flip_preset
            (PARAM_BASE / tunnel / param_json_name("sam", MODEL)).write_text(
                json.dumps(sam_params, indent=2) + "\n"
            )
            sam2 = run_stage("sam.py", tunnel, env)
            if log_dir:
                (log_dir / f"{tunnel}_sam_flip.log").write_text(
                    f"exit={sam2.returncode}\n{sam2.stdout}\n{sam2.stderr}"
                )
            if sam2.returncode != 0:
                row["status"] = "sam_flip_fail"
                return row

        oracle_mode = spec.get("sam_oracle")
        if oracle_mode:
            sam_params = json.loads(
                (PARAM_BASE / tunnel / param_json_name("sam", MODEL)).read_text()
            )
            sam_params["sam_hint_mode"] = oracle_mode
            sam_params.pop("ring_flip_preset", None)
            (PARAM_BASE / tunnel / param_json_name("sam", MODEL)).write_text(
                json.dumps(sam_params, indent=2) + "\n"
            )
            sam_o = run_stage("sam.py", tunnel, env)
            if log_dir:
                (log_dir / f"{tunnel}_sam_oracle.log").write_text(
                    f"exit={sam_o.returncode}\n{sam_o.stdout}\n{sam_o.stderr}"
                )
            if sam_o.returncode != 0:
                row["status"] = "sam_oracle_fail"
                return row

        row["miou"] = run_eval(tunnel, work_std, env)
        copy_outputs(variant_id, tunnel, work_std)
        diag_path = variant_out(variant_id, tunnel) / "k_diagnostics.json"
        diag_path.write_text(json.dumps(kdiag, indent=2) + "\n")
        if log_dir:
            copy_performance(work_std, log_dir / tunnel)
        row["status"] = "ok" if row["miou"] is not None else "eval_fail"
        row["elapsed_s"] = round(time.time() - t0, 1)
        print(
            f"  {variant_id} {tunnel}: mIoU={row['miou']} "
            f"K-spread={row['k_spread_px']}px ({'pass' if row['k_pass'] else 'fail'})"
        )
    finally:
        restore_work_std(tunnel, std_backup)

    return row


def mean_miou(variant_id: str, tunnels: list[str]) -> float | None:
    vals = [extract_miou(variant_out(variant_id, t)) for t in tunnels]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def append_csv(path: Path, row: dict) -> None:
    fields = [
        "variant_id", "tunnel", "miou", "k_spread_px", "k_pass", "status", "elapsed_s",
    ]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_until_mean(target: float, sweep_csv: Path) -> str | None:
    best_variant = None
    best_mean = -1.0

    for vid in variant_ids():
        print(f"\n=== variant {vid} (gate {GATE_TUNNEL}) ===")
        log_dir = LOG_ROOT / vid
        log_dir.mkdir(parents=True, exist_ok=True)
        row = run_tunnel_variant(vid, GATE_TUNNEL, log_dir=log_dir)
        append_csv(sweep_csv, row)
        if row["status"] != "ok":
            continue
        if not k_ok:
            print(f"  K gate fail on {GATE_TUNNEL} (spread {row['k_spread_px']}px), trying next variant")
            continue
        if row["miou"] is None or row["miou"] < SCALE_GATE:
            print(f"  Scale gate fail: mIoU {row['miou']} < {SCALE_GATE}")
            continue

        print(f"  Gate passed — scaling {vid} to panel")
        for t in TUNNELS:
            if t == GATE_TUNNEL:
                continue
            r = run_tunnel_variant(vid, t, log_dir=log_dir)
            append_csv(sweep_csv, r)

        m = mean_miou(vid, TUNNELS)
        print(f"  Panel mean mIoU={m}")
        if m is not None and m > best_mean:
            best_mean = m
            best_variant = vid
        if m is not None and m >= target:
            print(f"Target {target} reached at variant {vid}")
            return vid

    if best_variant:
        print(f"Best variant: {best_variant} mean={best_mean:.3f} (target {target} not met)")
    return best_variant


def main() -> None:
    os.chdir(REPO_ROOT)
    _ensure_venv_on_path()
    ensure_sample_characteristics(Path("/media/boringtao/Ezekers/R4Tun/data"))

    parser = argparse.ArgumentParser(description="T3 manual param tune (T1/T2 hints)")
    parser.add_argument("--tunnel", choices=TUNNELS)
    parser.add_argument("--all-tunnels", action="store_true")
    parser.add_argument("--variant", default="base_v3", choices=variant_ids())
    parser.add_argument("--until-mean", type=float, metavar="TARGET")
    parser.add_argument(
        "--detecting-hint", type=Path, help="Override detecting JSON (with --sam-hint)"
    )
    parser.add_argument("--sam-hint", type=Path, help="Override SAM JSON")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_csv = LOG_ROOT / f"sweep_{ts}.csv"

    if args.until_mean is not None:
        winner = run_until_mean(args.until_mean, sweep_csv)
        print(f"\nSweep log: {sweep_csv}")
        if winner:
            print(f"Winner: {winner}")
        return

    tunnels = TUNNELS if args.all_tunnels else ([args.tunnel] if args.tunnel else [GATE_TUNNEL])
    log_dir = LOG_ROOT / args.variant
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.detecting_hint and args.sam_hint:
        detecting = json.loads(args.detecting_hint.read_text())
        sam = json.loads(args.sam_hint.read_text())
        for t in tunnels:
            write_params(t, detecting, sam)
            row = run_tunnel_variant(args.variant, t, log_dir=log_dir)
            append_csv(sweep_csv, row)
    else:
        for t in tunnels:
            row = run_tunnel_variant(args.variant, t, log_dir=log_dir)
            append_csv(sweep_csv, row)

    if len(tunnels) == len(TUNNELS):
        m = mean_miou(args.variant, TUNNELS)
        print(f"Panel mean mIoU={m}")

    print(f"\nSweep log: {sweep_csv}")


if __name__ == "__main__":
    main()

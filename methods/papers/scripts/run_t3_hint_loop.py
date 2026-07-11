#!/usr/bin/env python3
"""Graded T3 hint loop: preprocessing fix + few-shot GT-free hints (T0–T5).

Usage:
    ./venv/bin/python methods/papers/scripts/run_t3_hint_loop.py --level T0 --tunnel 3-1-1
    ./venv/bin/python methods/papers/scripts/run_t3_hint_loop.py --level T1 --all-tunnels
    ./venv/bin/python methods/papers/scripts/run_t3_hint_loop.py --until-mean 0.60
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
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from few_shot_hint_lib import (  # noqa: E402
    exemplars_for_level,
    seed_exemplar_params_to_tunnel,
)
from migrate_t3_preprocessing import (  # noqa: E402
    DEFAULT_SOURCE_ROOT,
    ensure_memory_raw_characteristics,
    ensure_sample_characteristics,
)
from regular_sam_hint_lib import sam_hint_level_to_mode  # noqa: E402
from repeatability_common import (  # noqa: E402
    ABLATION_FOLDER,
    ORCHESTRATORS,
    PARAM_BASE,
    copy_performance,
    extract_miou,
    param_json_name,
    std_data_dir,
)

PYTHON = str(REPO_ROOT / "venv" / "bin" / "python")
if not Path(PYTHON).is_file():
    PYTHON = sys.executable

MODEL = "opus4.6"
ABLATION = "m_s_k"
TS = os.environ.get("T3_HINT_LOOP_TS") or datetime.now().strftime("%Y%m%d_%H%M%S")

CONTINUOUS_TUNNELS = ["3-1-1", "3-1-2", "3-1-3"]
HINT_LEVELS = ["T0", "T1", "T2", "T3", "T4", "T5"]
GATE_TUNNEL = "3-1-1"
TARGET_MEAN = 0.60
GATE_THRESHOLDS = {"T1": 0.45, "scale": 0.55}

VENDOR_SRC = REPO_ROOT / "data" / "ablation_anthropic" / ABLATION_FOLDER
LOOP_ROOT = REPO_ROOT / "data" / "t3_hint_loop"
LOG_ROOT = REPO_ROOT / "logs" / "t3_hint_loop"

AGENTS_REGULAR = REPO_ROOT / "agents_regular"
LIVE_MSK = REPO_ROOT / "agents" / "ablation" / ABLATION_FOLDER
HINT_AGENTS = AGENTS_REGULAR / "ablation" / ABLATION_FOLDER / "agents"

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

CHARACTERISERS = [
    ("unfolding", "1-unfolded_characteriser.py"),
    ("denoising", "2-denoised_characteriser.py"),
    ("enhancing", "3-enhanced_characteriser.py"),
    ("detecting", "4-detected_characteriser.py"),
]

STAGES_ALL = ["unfolding", "denoising", "enhancing", "detecting", "sam"]
STAGES_DS = ["detecting", "sam"]


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


def level_out(tunnel: str, level: str) -> Path:
    return LOOP_ROOT / level / tunnel


def seed_upstream_symlinks(level: str, tunnel: str) -> Path:
    dst = level_out(tunnel, level)
    src = VENDOR_SRC / tunnel
    dst.mkdir(parents=True, exist_ok=True)
    for name in UPSTREAM_FILES:
        sp = src / name
        dp = dst / name
        if not sp.is_file():
            continue
        if dp.exists() or dp.is_symlink():
            dp.unlink()
        dp.symlink_to(sp.resolve())
    return dst


def validate_vendor_tree(tunnel: str) -> list[str]:
    errors: list[str] = []
    root = VENDOR_SRC / tunnel
    for name in ("enhanced.csv", "depth_map.png", "pixel_to_point.pkl"):
        if not (root / name).is_file():
            errors.append(f"missing {name}")
    return errors


def ensure_characteristics(tunnel: str) -> None:
    char_dir = VENDOR_SRC / tunnel / "characteristics"
    targets = [
        "unfolded_characteristics.json",
        "denoised_characteristics.json",
        "enhanced_characteristics.json",
    ]
    if all((char_dir / n).is_file() for n in targets):
        return
    char_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["R4TUN_PIPELINE_OUT_PREFIX"] = str(VENDOR_SRC.relative_to(REPO_ROOT))
    plugins = REPO_ROOT / "sam4tun" / "plugins"
    for script in (
        "1-unfolded_characteriser.py",
        "2-denoised_characteriser.py",
        "3-enhanced_characteriser.py",
    ):
        subprocess.run(
            [PYTHON, str(plugins / script), tunnel],
            env=env, cwd=str(REPO_ROOT), capture_output=True, timeout=600,
        )


def clear_stage_params(tunnel: str, stages: list[str]) -> None:
    param_dir = PARAM_BASE / tunnel
    for stage in stages:
        pf = param_dir / param_json_name(stage, MODEL)
        if pf.is_file():
            pf.unlink()


@contextmanager
def swapped_hint_assets(
    level: str,
    *,
    swap_docs: bool,
    swap_analysts: bool,
    swap_detecting_py: bool,
):
    ts = TS
    backup = LOG_ROOT / f"_swap_backup_{ts}"
    backup.mkdir(parents=True, exist_ok=True)
    restored: list[tuple[Path, Path]] = []

    def _swap(src: Path, live: Path) -> None:
        if not src.is_file():
            raise FileNotFoundError(src)
        if live.is_file():
            bak = backup / f"{live.parent.name}_{live.name}"
            bak.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live, bak)
            restored.append((bak, live))
        shutil.copy2(src, live)

    try:
        if swap_docs:
            for stage in ("detecting",):
                live_d = LIVE_MSK / "agents" / stage
                for name in ("knowledge.md", "cot.md"):
                    _swap(HINT_AGENTS / stage / name, live_d / name)
        if swap_analysts:
            for stage, analyst in (("detecting", "analyst.py"), ("segmenting", "analyst.py")):
                _swap(HINT_AGENTS / stage / analyst, LIVE_MSK / "agents" / stage / analyst)
        if swap_detecting_py:
            live_det = REPO_ROOT / "agents" / "detecting.py"
            reg_det = AGENTS_REGULAR / "detecting.py"
            if live_det.is_file():
                bak = backup / "detecting.py"
                shutil.copy2(live_det, bak)
                restored.append((bak, live_det))
            shutil.copy2(reg_det, live_det)
        os.environ["T3_HINT_LEVEL"] = level
        os.environ["R4TUN_MODEL_TAG"] = MODEL
        yield
    finally:
        for bak, live in restored:
            shutil.copy2(bak, live)


def inject_sam_hint_mode(tunnel: str, mode: str) -> dict | None:
    path = PARAM_BASE / tunnel / param_json_name("sam", MODEL)
    backup = None
    if path.is_file():
        backup = json.loads(path.read_text())
    else:
        backup = {}
    params = dict(backup)
    params["sam_hint_mode"] = mode
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2) + "\n")
    return backup


def restore_sam_params(tunnel: str, backup: dict | None) -> None:
    path = PARAM_BASE / tunnel / param_json_name("sam", MODEL)
    if backup is None:
        return
    if backup:
        path.write_text(json.dumps(backup, indent=2) + "\n")
    elif path.is_file():
        path.unlink()


def run_orchestrator(tunnel: str, stages: list[str], env: dict) -> subprocess.CompletedProcess:
    script = ORCHESTRATORS[MODEL]
    cmd = [PYTHON, str(REPO_ROOT / script), tunnel, "--model", MODEL, "--stages", *stages]
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=3600,
    )


def run_pipeline_stage(script: str, tunnel: str, env: dict) -> subprocess.CompletedProcess:
    if script == "evaluation.py":
        cmd = [PYTHON, str(REPO_ROOT / "agents" / "evaluation.py"), tunnel, "--ablation", ABLATION, "--schema", "auto"]
    elif script in ("detecting.py", "sam.py"):
        cmd = [PYTHON, str(AGENTS_REGULAR / script), tunnel, "--ablation", ABLATION, "--model", MODEL]
    else:
        cmd = [PYTHON, str(REPO_ROOT / "agents" / script), tunnel, "--ablation", ABLATION, "--model", MODEL]
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=1800,
    )


STAGE_SCRIPT = {
    "unfolding": "unfolding.py",
    "denoising": "denoising.py",
    "enhancing": "enhancing.py",
    "detecting": "detecting.py",
    "sam": "sam.py",
}


def run_eval(tunnel: str, out_prefix: Path, env: dict) -> float | None:
    env = dict(env)
    rel = out_prefix.relative_to(REPO_ROOT)
    if rel.name == tunnel:
        rel = rel.parent
    env["R4TUN_PIPELINE_OUT_PREFIX"] = str(rel)
    ev = run_pipeline_stage("evaluation.py", tunnel, env)
    return extract_miou(out_prefix)


def copy_outputs_to_loop(level: str, tunnel: str, src: Path) -> None:
    dst = level_out(tunnel, level)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("detected.csv", "detected_lines.png", "final.csv", "only_label.csv"):
        sp = src / name
        if sp.is_file():
            dp = dst / name
            if dp.exists() or dp.is_symlink():
                dp.unlink()
            shutil.copy2(sp, dp)
    for sub in ("evaluation", "characteristics"):
        ss = src / sub
        if ss.is_dir():
            dd = dst / sub
            if dd.exists():
                shutil.rmtree(dd)
            shutil.copytree(ss, dd)


def level_config(level: str) -> dict:
    cfg = {
        "level": level,
        "swap_docs": False,
        "swap_analysts": False,
        "swap_detecting_py": False,
        "stages": None,
        "sam_hint": "off",
        "use_orchestrator": False,
        "use_frozen_exemplar": False,
        "eval_only": False,
        "rerun_sam_after": False,
    }
    if level == "T0":
        cfg["eval_only"] = True
    elif level == "T1":
        cfg.update(
            swap_docs=True, swap_analysts=False, stages=STAGES_DS,
            use_frozen_exemplar=True,
        )
    elif level == "T2":
        cfg.update(
            swap_docs=True, swap_analysts=False, swap_detecting_py=True,
            stages=STAGES_DS, use_frozen_exemplar=True,
        )
    elif level == "T3":
        cfg.update(
            swap_docs=True, swap_analysts=False, swap_detecting_py=True,
            stages=STAGES_DS, use_frozen_exemplar=True,
        )
    elif level == "T4":
        cfg.update(
            swap_docs=True, swap_analysts=False, stages=STAGES_ALL,
            use_frozen_exemplar=True,
        )
    elif level == "T5":
        cfg.update(
            swap_docs=True, swap_analysts=False, swap_detecting_py=True,
            stages=STAGES_DS, sam_hint=sam_hint_level_to_mode("T5"),
            use_frozen_exemplar=True, rerun_sam_after=True,
        )
    return cfg


def run_tunnel_level(
    level: str,
    tunnel: str,
    skip_existing: bool = False,
) -> dict:
    cfg = level_config(level)
    out = level_out(tunnel, level)
    log_dir = LOG_ROOT / level / tunnel
    log_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "level": level,
        "tunnel": tunnel,
        "miou": None,
        "status": "pending",
        "exemplars": ",".join(exemplars_for_level(level)),
    }

    vendor_errors = validate_vendor_tree(tunnel)
    if vendor_errors:
        row["status"] = f"vendor_invalid:{';'.join(vendor_errors)}"
        return row

    if skip_existing and extract_miou(out) is not None:
        row["miou"] = extract_miou(out)
        row["status"] = "skipped"
        return row

    seed_upstream_symlinks(level, tunnel)

    work_std = std_data_dir(tunnel)
    std_backup = log_dir / "_std_data_backup"
    param_backup_dir = log_dir / "_param_backup"

    if work_std.exists():
        if std_backup.exists():
            shutil.rmtree(std_backup)
        shutil.copytree(work_std, std_backup, symlinks=True)
        shutil.rmtree(work_std)
    shutil.copytree(VENDOR_SRC / tunnel, work_std, symlinks=True)
    ensure_characteristics(tunnel)
    ensure_sample_characteristics(DEFAULT_SOURCE_ROOT)
    ensure_memory_raw_characteristics(tunnel)

    sam_backup = None
    try:
        t0 = time.time()

        if cfg["eval_only"]:
            miou = extract_miou(work_std)
            if miou is None and (work_std / "only_label.csv").is_file():
                env_eval = os.environ.copy()
                env_eval["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{ABLATION_FOLDER}"
                row["miou"] = run_eval(tunnel, work_std, env_eval)
            else:
                row["miou"] = miou
            row["status"] = "ok" if row["miou"] is not None else "eval_fail"
            copy_outputs_to_loop(level, tunnel, work_std)
            copy_performance(work_std, log_dir)
            row["elapsed_s"] = round(time.time() - t0, 1)
            return row

        if cfg["stages"] and not cfg.get("rerun_sam_after"):
            clear_stage_params(tunnel, cfg["stages"])
        elif cfg.get("rerun_sam_after"):
            clear_stage_params(tunnel, ["detecting", "sam"])

        exemplars = exemplars_for_level(level)
        if cfg.get("use_frozen_exemplar") and cfg["stages"]:
            primary = exemplars[-1]
            seed_exemplar_params_to_tunnel(
                tunnel, cfg["stages"], exemplars, MODEL, primary=primary,
            )

        if cfg["sam_hint"] != "off" and not cfg.get("rerun_sam_after"):
            sam_backup = inject_sam_hint_mode(tunnel, cfg["sam_hint"])

        need_swap = cfg["swap_docs"] or cfg["swap_analysts"] or cfg["swap_detecting_py"]
        swap_ctx = (
            swapped_hint_assets(
                level,
                swap_docs=cfg["swap_docs"],
                swap_analysts=cfg["swap_analysts"],
                swap_detecting_py=cfg["swap_detecting_py"],
            )
            if need_swap
            else nullcontext()
        )

        with swap_ctx:
            env_pipe = os.environ.copy()
            env_pipe["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{ABLATION_FOLDER}"
            env_pipe.setdefault("MPLBACKEND", "Agg")

            if cfg.get("use_frozen_exemplar") and cfg["stages"]:
                scripts = []
                for stage in cfg["stages"]:
                    if cfg.get("rerun_sam_after") and stage == "sam":
                        continue
                    scripts.append(STAGE_SCRIPT[stage])
                for script in scripts:
                    st = run_pipeline_stage(script, tunnel, env_pipe)
                    (log_dir / f"{script}.log").write_text(
                        f"exit={st.returncode}\n{st.stdout}\n{st.stderr}"
                    )
                    if st.returncode != 0:
                        row["status"] = f"{script.replace('.py','')}_fail"
                        return row

            elif cfg["use_orchestrator"]:
                if PARAM_BASE.joinpath(tunnel).exists():
                    if param_backup_dir.exists():
                        shutil.rmtree(param_backup_dir)
                    shutil.copytree(PARAM_BASE / tunnel, param_backup_dir)

                result = run_orchestrator(tunnel, cfg["stages"], os.environ.copy())
                (log_dir / "orchestrator.log").write_text(
                    f"exit={result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                )
                if result.returncode != 0:
                    row["status"] = "orchestrator_fail"
                    return row

            if cfg.get("rerun_sam_after"):
                if cfg["sam_hint"] != "off":
                    sam_backup = inject_sam_hint_mode(tunnel, cfg["sam_hint"])
                env_pipe = os.environ.copy()
                env_pipe["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{ABLATION_FOLDER}"
                env_pipe.setdefault("MPLBACKEND", "Agg")
                sam = run_pipeline_stage("sam.py", tunnel, env_pipe)
                (log_dir / "sam.log").write_text(
                    f"exit={sam.returncode}\nSTDOUT:\n{sam.stdout}\nSTDERR:\n{sam.stderr}"
                )
                if sam.returncode != 0:
                    row["status"] = "sam_fail"
                    return row

        env_eval = os.environ.copy()
        env_eval["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{ABLATION_FOLDER}"
        row["miou"] = run_eval(tunnel, work_std, env_eval)
        copy_outputs_to_loop(level, tunnel, work_std)
        copy_performance(work_std, log_dir)
        row["status"] = "ok" if row["miou"] is not None else "eval_fail"
        row["elapsed_s"] = round(time.time() - t0, 1)
        print(f"  {level} {tunnel}: mIoU={row['miou']} ({row.get('elapsed_s')}s)")

    finally:
        restore_sam_params(tunnel, sam_backup)
        if param_backup_dir.exists() and PARAM_BASE.joinpath(tunnel).exists():
            shutil.rmtree(PARAM_BASE / tunnel)
            shutil.copytree(param_backup_dir, PARAM_BASE / tunnel)
        if std_backup.exists():
            if work_std.exists():
                shutil.rmtree(work_std)
            shutil.copytree(std_backup, work_std, symlinks=True)

    return row


def mean_miou(level: str, tunnels: list[str]) -> float | None:
    vals = [extract_miou(level_out(t, level)) for t in tunnels]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def gate_passes(level: str, miou: float | None, for_scale: bool = False) -> bool:
    if miou is None:
        return False
    if for_scale:
        return miou >= GATE_THRESHOLDS["scale"]
    if level == "T0":
        return True
    return miou >= GATE_THRESHOLDS.get(level, GATE_THRESHOLDS["scale"])


def append_csv(path: Path, row: dict) -> None:
    fields = ["level", "tunnel", "miou", "status", "exemplars", "elapsed_s"]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    os.chdir(REPO_ROOT)
    _ensure_venv_on_path()

    parser = argparse.ArgumentParser(description="T3 hint loop T0–T5")
    parser.add_argument("--level", choices=HINT_LEVELS)
    parser.add_argument("--tunnel", choices=CONTINUOUS_TUNNELS)
    parser.add_argument("--all-tunnels", action="store_true")
    parser.add_argument("--until-mean", type=float, metavar="TARGET")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    summary_csv = LOG_ROOT / f"summary_{TS}.csv"

    if args.until_mean is not None:
        target = args.until_mean
        for level in HINT_LEVELS:
            print(f"\n{'='*70}\n  LEVEL {level}\n{'='*70}")
            row = run_tunnel_level(level, GATE_TUNNEL, skip_existing=args.skip_existing)
            append_csv(summary_csv, row)
            if row["status"] not in ("ok", "skipped"):
                print(f"Warning at {level} gate: {row['status']}")
            gate_miou = row.get("miou")
            if gate_miou is not None and not gate_passes(level, gate_miou):
                print(f"Gate mIoU {gate_miou} below threshold for {level} (continuing)")
            for tunnel in CONTINUOUS_TUNNELS:
                if tunnel == GATE_TUNNEL:
                    continue
                r = run_tunnel_level(level, tunnel, skip_existing=args.skip_existing)
                append_csv(summary_csv, r)
            m = mean_miou(level, CONTINUOUS_TUNNELS)
            print(f"  {level} mean mIoU={m}")
            if m is not None and m >= target:
                print(f"Target {target} reached at {level}")
                break
        print(f"\nSummary: {summary_csv}")
        return

    levels = [args.level] if args.level else HINT_LEVELS
    tunnels = CONTINUOUS_TUNNELS if args.all_tunnels else (
        [args.tunnel] if args.tunnel else [GATE_TUNNEL]
    )

    for level in levels:
        print(f"\n=== {level} ===")
        for tunnel in tunnels:
            row = run_tunnel_level(level, tunnel, skip_existing=args.skip_existing)
            append_csv(summary_csv, row)

    print(f"\nSummary: {summary_csv}")


if __name__ == "__main__":
    main()

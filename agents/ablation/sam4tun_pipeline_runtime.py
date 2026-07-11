"""Shared sam4tun/agents pipeline runtime for LLM ablation orchestrators.

Matches run_ablation_glm.py: stages run under sam4tun/agents/, scratch work dir
sam4tun/data/{tunnel_id}, params under sam4tun/agents/parameters/, archive to
data/ablation/{condition}/{tunnel_id}, evaluate via evaluate_static.py.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

STAGE_CHARS = {
    "unfolding": "unfolded_characteristics.json",
    "denoising": "denoised_characteristics.json",
    "enhancing": "enhanced_characteristics.json",
    "detecting": "detected_characteristics.json",
}

WORK_ARTIFACTS = {
    "unfolding": "unwrapped.csv",
    "denoising": "denoised.csv",
    "enhancing": "enhanced.csv",
    "detecting": "detected.csv",
    "sam": "only_label.csv",
}


def param_base(ablation_folder: str) -> Path:
    return REPO_ROOT / "sam4tun" / "agents" / "parameters" / ablation_folder


def out_root(ablation_folder: str) -> Path:
    return REPO_ROOT / "data" / "ablation" / ablation_folder


def work_dir(tunnel_id: str) -> Path:
    return REPO_ROOT / "sam4tun" / "data" / tunnel_id


def setup_env(ablation_folder: str) -> dict[str, str]:
    os.environ["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{ablation_folder}"
    os.environ["R4TUN_ABLATION_TUNNEL_SUBROOT"] = ablation_folder
    os.environ["R4TUN_PIPELINE_WORK_DIR"] = "sam4tun/data"
    venv_site = REPO_ROOT / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    sam_sa = REPO_ROOT / "sam4tun" / "segment-anything"
    pp = [str(REPO_ROOT), str(REPO_ROOT / "sam4tun")]
    if venv_site.is_dir():
        pp.append(str(venv_site))
    if sam_sa.is_dir():
        pp.append(str(sam_sa))
    os.environ["PYTHONPATH"] = os.pathsep.join(pp)
    return os.environ.copy()


def symlink_input(tunnel_id: str) -> None:
    subset = REPO_ROOT / "data" / "subsets" / f"{tunnel_id}.txt"
    link = REPO_ROOT / "sam4tun" / "data" / f"{tunnel_id}.txt"
    if not subset.is_file():
        raise FileNotFoundError(subset)
    (REPO_ROOT / "sam4tun" / "data").mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(subset, link.parent))


def ensure_tunnel_characteristics(tunnel_id: str, ablation_folder: str) -> None:
    dest = out_root(ablation_folder) / tunnel_id / "characteristics"
    if (dest / "raw_characteristics.json").is_file():
        return
    src = REPO_ROOT / "data" / "ablation" / "memory" / tunnel_id / "characteristics"
    if not (src / "raw_characteristics.json").is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    print(f"  Copied characteristics memory → {ablation_folder}/{tunnel_id}")


def run_pipeline_stage(
    tunnel_id: str,
    stage_script: str,
    ablation_code: str,
    model_tag: str,
    env: dict,
) -> None:
    cmd = [
        PYTHON,
        str(REPO_ROOT / "sam4tun" / "agents" / stage_script),
        tunnel_id,
        "--ablation", ablation_code,
        "--model", model_tag,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-30:]:
            print(f"    {line}")
    if result.returncode != 0:
        print(result.stderr[-4000:] if result.stderr else "")
        raise RuntimeError(f"Stage {stage_script} failed (exit {result.returncode})")


def run_characteriser(tunnel_id: str, script: str, env: dict) -> None:
    cmd = [PYTHON, str(REPO_ROOT / "sam4tun" / "plugins" / script), tunnel_id]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-15:]:
            print(f"    {line}")
    if result.returncode != 0:
        print(result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(f"Characteriser {script} failed")


def run_raw_characteriser(tunnel_id: str, env: dict) -> None:
    subset = REPO_ROOT / "data" / "subsets" / f"{tunnel_id}.txt"
    if tunnel_id == "sample":
        fp = REPO_ROOT / "sam4tun" / "data" / "sample.txt"
    else:
        fp = subset
    cmd = [
        PYTHON, "-c",
        f"""
import os, json, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from sam4tun.plugins.raw_characteristics import NumpyEncoder, analyze_point_cloud
from sam4tun.plugins.paths import tunnel_characteristics_dir
tid = {tunnel_id!r}
fp = {str(fp)!r}
results = analyze_point_cloud(fp, tid)
out_dir = tunnel_characteristics_dir(tid)
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'raw_characteristics.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)
print('wrote', out)
""",
    ]
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0:
        print(result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError("raw characteriser failed")


def archive_pipeline_output(tunnel_id: str, ablation_folder: str) -> None:
    src = work_dir(tunnel_id)
    dest = out_root(ablation_folder) / tunnel_id
    if not src.is_dir():
        raise FileNotFoundError(f"Missing pipeline output: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    analysis_backup = dest.parent / f".{tunnel_id}_analysis_tmp"
    analysis_src = dest / "analysis"
    if analysis_src.is_dir():
        if analysis_backup.exists():
            shutil.rmtree(analysis_backup)
        shutil.copytree(analysis_src, analysis_backup)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    link = REPO_ROOT / "sam4tun" / "data" / f"{tunnel_id}.txt"
    if link.is_symlink():
        link.unlink()
    mem_chars = out_root(ablation_folder) / tunnel_id / "characteristics"
    if mem_chars.is_dir():
        shutil.copytree(mem_chars, dest / "characteristics", dirs_exist_ok=True)
    if analysis_backup.is_dir():
        shutil.copytree(analysis_backup, dest / "analysis", dirs_exist_ok=True)
        shutil.rmtree(analysis_backup)


def run_evaluation(tunnel_id: str, ablation_folder: str, env: dict) -> None:
    root = out_root(ablation_folder)
    only_label = root / tunnel_id / "only_label.csv"
    if not only_label.is_file():
        print(f"  Skipping evaluation: {only_label} not found")
        return
    cmd = [
        PYTHON,
        str(REPO_ROOT / "sam4tun" / "agents" / "evaluate_static.py"),
        tunnel_id,
        "--data-root", str(root),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")
    if result.returncode != 0:
        print(result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError("evaluation failed")


def save_parameters(
    tunnel_id: str,
    stage_name: str,
    params: dict,
    ablation_folder: str,
    param_suffix: str,
    stage_to_param: dict[str, str],
    model_tag: str,
) -> Path:
    pname = stage_to_param[stage_name]
    filename = f"parameters_{pname}{param_suffix}{model_tag}.json"
    out_dir = param_base(ablation_folder) / tunnel_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


def prepare_work_dir(tunnel_id: str, stage_filter: list[str] | None) -> None:
    fresh_run = not stage_filter or "unfolding" in stage_filter
    wd = work_dir(tunnel_id)
    if fresh_run and wd.exists():
        shutil.rmtree(wd)


def should_skip_pipeline(
    tunnel_id: str,
    ablation_folder: str,
    stage_name: str,
    characteriser: str | None,
    params_match: bool,
    upstream_pipeline_ran: bool,
) -> bool:
    if not params_match or upstream_pipeline_ran:
        return False
    wd = work_dir(tunnel_id)
    state_ok = (wd / "state.pkl").is_file()
    char_file = STAGE_CHARS.get(stage_name, "")
    char_ok = not characteriser or (
        out_root(ablation_folder) / tunnel_id / "characteristics" / char_file
    ).is_file()
    return state_ok and char_ok

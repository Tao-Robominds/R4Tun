#!/usr/bin/env python
"""
GLM ablation orchestrator for sam4tun/agents pipeline.

Interleaves GLM (Zhipu glm-4-plus) inference with sam4tun/agents stages and
characteriser plugins. Params → sam4tun/agents/parameters/{condition}/{tid}/.

Usage:
    ./venv/bin/python run_ablation_glm.py --ablation m_s_k 1-1 2-1
    ./venv/bin/python run_ablation_glm.py --ablation m_s_k --t1-t2
    ./venv/bin/python run_ablation_glm.py --ablation m_s_k --sanity   # requires t1_t2_gate PASS
    ./venv/bin/python run_ablation_glm.py --ablation m 1-1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
DEFAULT_MODEL_TAG = "glm"
GLM_API_MODEL = "glm-4-plus"
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 3
API_TIMEOUT = 300

T1_T2_IDS = ["1-1", "2-1"]
SANITY_IDS = ["1-1", "2-1", "3-1-1", "4-1", "5-1"]
T1_T2_GATE_THRESHOLD = 0.70
T1_T2_GATE_FILE = REPO_ROOT / "data" / "ablation" / "t1_t2_gate.md"

ABLATION_CONFIGS = {
    "m": {
        "folder": "memory",
        "code": "m",
        "suffix": "_m_",
        "analyst_pkg": "agents/ablation/memory/agents",
        "analyst_module_prefix": {
            "unfolding": "unfolding.analyst",
            "denoising": "denoising.analyst",
            "enhancing": "enhancing.analyser" if False else "enhancing.analyst",
            "detecting": "detecting.analyst",
            "sam": "segmenting.analyst",
        },
    },
    "m_s": {
        "folder": "memory+state",
        "code": "m_s",
        "suffix": "_m_s_",
        "analyst_pkg": "agents/ablation/memory+state/agents",
    },
    "m_s_k": {
        "folder": "memory+state+knowledge",
        "code": "m_s_k",
        "suffix": "_m_s_k_",
        "analyst_pkg": "agents/ablation/memory+state+knowledge/agents",
    },
}

STAGES = [
    ("unfolding", "unfolding.py", "1-unfolded_characteriser.py"),
    ("denoising", "denoising.py", "2-denoised_characteriser.py"),
    ("enhancing", "enhancing.py", "3-enhanced_characteriser.py"),
    ("detecting", "detecting.py", "4-detected_characteriser.py"),
    ("sam", "sam.py", None),
]

ANALYST_CLASSES = {
    "unfolding": ("unfolding.analyst", "UnfoldingAnalyser"),
    "denoising": ("denoising.analyst", "DenoisingAnalyser"),
    "enhancing": ("enhancing.analyst", "EnhancingAnalyser"),
    "detecting": ("detecting.analyst", "DetectingAnalyser"),
    "sam": ("segmenting.analyst", "SegmentingAnalyser"),
}

STAGE_TO_PARAM = {
    "unfolding": "unfolding",
    "denoising": "denoising",
    "enhancing": "enhancing",
    "detecting": "detecting",
    "sam": "sam",
}


def cfg_for(ablation: str) -> dict:
    if ablation not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation {ablation!r}")
    return ABLATION_CONFIGS[ablation]


def param_base(ablation: str) -> Path:
    return REPO_ROOT / "sam4tun" / "agents" / "parameters" / cfg_for(ablation)["folder"]


def out_root(ablation: str) -> Path:
    return REPO_ROOT / "data" / "ablation" / cfg_for(ablation)["folder"]


def setup_env(ablation: str) -> dict[str, str]:
    folder = cfg_for(ablation)["folder"]
    os.environ["R4TUN_PIPELINE_OUT_PREFIX"] = f"data/ablation/{folder}"
    os.environ["R4TUN_ABLATION_TUNNEL_SUBROOT"] = folder
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


def _import_analyst(ablation: str, stage_name: str):
    agents_abs = str((REPO_ROOT / cfg_for(ablation)["analyst_pkg"]).resolve())
    if agents_abs not in sys.path:
        sys.path.insert(0, agents_abs)
    import importlib
    module_rel, class_name = ANALYST_CLASSES[stage_name]
    mod = importlib.import_module(module_rel)
    return getattr(mod, class_name)


def build_prompt(tunnel_id: str, ablation: str, stage_name: str, model_tag: str) -> str:
    cls = _import_analyst(ablation, stage_name)
    analyser = cls(tunnel_id)
    if hasattr(analyser, "build_llm_prompt_markdown"):
        try:
            return analyser.build_llm_prompt_markdown(model_tag=model_tag)
        except TypeError:
            return analyser.build_llm_prompt_markdown()
    raise RuntimeError(f"No prompt builder on {cls}")


def call_glm(prompt: str, dry_run: bool = False) -> str:
    if dry_run:
        print("  [dry-run] Skipping GLM API call")
        return ""
    key = os.getenv("GLM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GLM_API_KEY not set in .env")
    client = OpenAI(api_key=key, base_url=GLM_BASE_URL, timeout=API_TIMEOUT)
    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=GLM_API_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            print(
                f"  GLM API: {time.time() - t0:.1f}s, "
                f"{getattr(usage, 'prompt_tokens', '?')} in / "
                f"{getattr(usage, 'completion_tokens', '?')} out"
            )
            return text
        except Exception as e:
            print(f"  GLM attempt {attempt}/{MAX_RETRIES} failed: {type(e).__name__}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(10 * attempt)
    raise RuntimeError("unreachable")


def extract_json_from_response(response_text: str) -> dict:
    pattern = r"```json\s*\n(.*?)\n\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if not matches:
        raise ValueError("No ```json``` fence in GLM response")
    return json.loads(matches[-1].strip())


def save_parameters(
    tunnel_id: str, stage_name: str, params: dict, ablation: str, model_tag: str
) -> Path:
    suffix = cfg_for(ablation)["suffix"]
    pname = STAGE_TO_PARAM[stage_name]
    filename = f"parameters_{pname}{suffix}{model_tag}.json"
    out_dir = param_base(ablation) / tunnel_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved: {out_path}")
    return out_path


def run_pipeline_stage(tunnel_id: str, stage_script: str, ablation: str, model_tag: str, env: dict) -> None:
    cmd = [
        PYTHON,
        str(REPO_ROOT / "sam4tun" / "agents" / stage_script),
        tunnel_id,
        "--ablation", cfg_for(ablation)["code"],
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
    print(result.stdout or result.stderr)
    if result.returncode != 0:
        raise RuntimeError("raw characteriser failed")


def archive_pipeline_output(tunnel_id: str, ablation: str) -> None:
    work = REPO_ROOT / "sam4tun" / "data" / tunnel_id
    dest = out_root(ablation) / tunnel_id
    if not work.is_dir():
        raise FileNotFoundError(f"Missing pipeline output: {work}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(work), str(dest))
    link = REPO_ROOT / "sam4tun" / "data" / f"{tunnel_id}.txt"
    if link.is_symlink():
        link.unlink()
    # Keep stage characteristics alongside archived run
    mem_chars = REPO_ROOT / "data" / "ablation" / cfg_for(ablation)["folder"] / tunnel_id / "characteristics"
    if mem_chars.is_dir():
        shutil.copytree(mem_chars, dest / "characteristics", dirs_exist_ok=True)


def run_evaluation(tunnel_id: str, ablation: str, env: dict) -> None:
    root = out_root(ablation)
    cmd = [
        PYTHON,
        str(REPO_ROOT / "sam4tun" / "agents" / "evaluate_static.py"),
        tunnel_id,
        "--data-root", str(root),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr or "")
        raise RuntimeError("evaluation failed")


def extract_miou(tunnel_id: str, ablation: str) -> float | None:
    perf = out_root(ablation) / tunnel_id / "evaluation" / "performance.md"
    if not perf.is_file():
        return None
    for line in perf.read_text().splitlines():
        if "Mean IoU (mIoU):" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                return None
    return None


def ensure_tunnel_characteristics(tunnel_id: str, ablation: str) -> None:
    """Copy raw characteristics from memory/ when this condition folder lacks them."""
    folder = cfg_for(ablation)["folder"]
    dest = REPO_ROOT / "data" / "ablation" / folder / tunnel_id / "characteristics"
    if (dest / "raw_characteristics.json").is_file():
        return
    src = REPO_ROOT / "data" / "ablation" / "memory" / tunnel_id / "characteristics"
    if not (src / "raw_characteristics.json").is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    print(f"  Copied characteristics memory → {folder}/{tunnel_id}")


def symlink_input(tunnel_id: str) -> None:
    subset = REPO_ROOT / "data" / "subsets" / f"{tunnel_id}.txt"
    link = REPO_ROOT / "sam4tun" / "data" / f"{tunnel_id}.txt"
    if not subset.is_file():
        raise FileNotFoundError(subset)
    REPO_ROOT / "sam4tun" / "data"
    (REPO_ROOT / "sam4tun" / "data").mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(os.path.relpath(subset, link.parent))


def t1_t2_gate_passed() -> bool:
    if not T1_T2_GATE_FILE.is_file():
        return False
    return "Status: PASS" in T1_T2_GATE_FILE.read_text()


def write_t1_t2_gate(results: dict[str, float | None], status: str) -> None:
    T1_T2_GATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# T1/T2 GLM ablation gate",
        "",
        f"- **Threshold:** mIoU ≥ {T1_T2_GATE_THRESHOLD}",
        f"- **Finished:** {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"- **Status:** {status}",
        "",
        "| tunnel | mIoU | pass |",
        "|--------|------|------|",
    ]
    for tid in T1_T2_IDS:
        m = results.get(tid)
        ok = m is not None and m >= T1_T2_GATE_THRESHOLD
        lines.append(f"| {tid} | {m if m is not None else 'n/a'} | {'yes' if ok else 'no'} |")
    T1_T2_GATE_FILE.write_text("\n".join(lines) + "\n")


def check_t1_t2_gate(ablation: str) -> bool:
    results = {tid: extract_miou(tid, ablation) for tid in T1_T2_IDS}
    ok = all(v is not None and v >= T1_T2_GATE_THRESHOLD for v in results.values())
    write_t1_t2_gate(results, "PASS" if ok else "FAIL")
    if ok:
        print(f"T1/T2 gate PASS — see {T1_T2_GATE_FILE}")
    else:
        print(f"T1/T2 gate FAIL — do not run 3-1-1/4-1/5-1. See {T1_T2_GATE_FILE}")
    return ok


def process_tunnel(
    tunnel_id: str,
    ablation: str,
    model_tag: str,
    dry_run: bool,
    env: dict,
    stage_filter: list[str] | None = None,
) -> None:
    print(f"\n{'='*60}\nTUNNEL {tunnel_id}  ablation={ablation}  model={model_tag}\n{'='*60}")
    out_root(ablation).mkdir(parents=True, exist_ok=True)
    (out_root(ablation) / tunnel_id).mkdir(parents=True, exist_ok=True)
    ensure_tunnel_characteristics(tunnel_id, ablation)

    if not dry_run:
        symlink_input(tunnel_id)
        work = REPO_ROOT / "sam4tun" / "data" / tunnel_id
        fresh_run = not stage_filter or "unfolding" in stage_filter
        if fresh_run and work.exists():
            shutil.rmtree(work)

    needs_unfolding = not stage_filter or "unfolding" in stage_filter
    if needs_unfolding and not dry_run:
        print("\n--- Raw characteristics ---")
        run_raw_characteriser(tunnel_id, env)

    upstream_ran = False
    for stage_name, stage_script, char_script in STAGES:
        if stage_filter and stage_name not in stage_filter:
            continue
        print(f"\n--- Stage: {stage_name} ---")
        suffix = cfg_for(ablation)["suffix"]
        pname = STAGE_TO_PARAM[stage_name]
        param_path = param_base(ablation) / tunnel_id / f"parameters_{pname}{suffix}{model_tag}.json"
        old_params = None
        if param_path.is_file():
            old_params = json.loads(param_path.read_text())

        prompt = build_prompt(tunnel_id, ablation, stage_name, model_tag)
        print(f"  Prompt: {len(prompt):,} chars")
        response = call_glm(prompt, dry_run=dry_run)
        if dry_run:
            continue

        analysis_dir = out_root(ablation) / tunnel_id / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / f"{stage_name}_reasoning_{model_tag}.md").write_text(response)

        params = extract_json_from_response(response)
        save_parameters(tunnel_id, stage_name, params, ablation, model_tag)

        if old_params == params and not upstream_ran:
            work = REPO_ROOT / "sam4tun" / "data" / tunnel_id
            state_ok = (work / "state.pkl").is_file()
            char_file = STAGE_CHARS.get(stage_name, "")
            char_ok = not char_script or (
                out_root(ablation) / tunnel_id / "characteristics" / char_file
            ).is_file()
            if state_ok and char_ok:
                print(f"  SKIP pipeline: params unchanged for {stage_name}")
                continue
            print(f"  Re-running pipeline: work state missing for {stage_name}")
        if not dry_run:
            run_pipeline_stage(tunnel_id, stage_script, ablation, model_tag, env)
            if char_script:
                run_characteriser(tunnel_id, char_script, env)
            upstream_ran = True

    if dry_run or not upstream_ran:
        return

    print("\n--- Archive + evaluation ---")
    archive_pipeline_output(tunnel_id, ablation)
    run_evaluation(tunnel_id, ablation, env)
    miou = extract_miou(tunnel_id, ablation)
    print(f"  mIoU: {miou if miou is not None else 'n/a'}")


STAGE_CHARS = {
    "unfolding": "unfolded_characteristics.json",
    "denoising": "denoised_characteristics.json",
    "enhancing": "enhanced_characteristics.json",
    "detecting": "detected_characteristics.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="GLM ablation orchestrator (sam4tun/agents)")
    parser.add_argument("tunnel_ids", nargs="*", help="Tunnel IDs")
    parser.add_argument("--ablation", "-a", required=True, choices=list(ABLATION_CONFIGS))
    parser.add_argument("--model", default=DEFAULT_MODEL_TAG)
    parser.add_argument("--t1-t2", action="store_true", help="Run 1-1 and 2-1 only")
    parser.add_argument("--sanity", action="store_true", help="Full sanity subset (needs T1/T2 gate)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-t1-t2-gate", action="store_true", help="Evaluate gate from existing runs")
    parser.add_argument("--stages", nargs="+", choices=[s[0] for s in STAGES])
    args = parser.parse_args()

    if args.sanity and not t1_t2_gate_passed():
        print("ERROR: --sanity requires T1/T2 gate PASS. Run --t1-t2 with m_s_k first.")
        sys.exit(1)

    if args.t1_t2:
        tunnel_ids = T1_T2_IDS
    elif args.sanity:
        tunnel_ids = SANITY_IDS
    elif args.tunnel_ids:
        tunnel_ids = args.tunnel_ids
    else:
        parser.error("Provide tunnel IDs, --t1-t2, or --sanity")

    env = setup_env(args.ablation)
    log_dir = REPO_ROOT / "logs" / "ablation"
    log_dir.mkdir(parents=True, exist_ok=True)

    for tid in tunnel_ids:
        log_file = log_dir / f"{tid}_{args.ablation}_{args.model}.log"
        try:
            process_tunnel(tid, args.ablation, args.model, args.dry_run, env, args.stages)
        except Exception as e:
            print(f"\nFAILED {tid}: {e}")
            if len(tunnel_ids) == 1:
                sys.exit(1)

    if args.check_t1_t2_gate or (args.t1_t2 and not args.dry_run):
        if not check_t1_t2_gate(args.ablation):
            sys.exit(1)


if __name__ == "__main__":
    main()

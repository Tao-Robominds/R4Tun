"""Smoke run: preprocessing + detection + eval on reference panel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.ablation.scripts._labelmap_viz import render_labelmap_png


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_report(results_path: Path, out_md: Path) -> None:
    data = json.loads(results_path.read_text())
    rows = [
        "| tunnel | ring | mIoU | OA | detection PNG | gt PNG |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in data.get("rings", []):
        tid = r["tunnel_id"]
        rid = int(r["ring_id"])
        rows.append(
            f"| {tid} | {rid} | {r['mIoU']:.4f} | {r['OA']:.4f} | "
            f"`data/ablation/{tid}/r{rid}/detection/labelmap.png` | "
            f"`data/ablation/{tid}/r{rid}/gt_ceiling/labelmap.png` |"
        )
    md = f"""# Detection baseline (pre-BO)

| metric | value |
|---|---:|
| median mIoU | {data.get('median_mIoU', 0.0):.4f} |
| min mIoU | {data.get('min_mIoU', 0.0):.4f} |
| max mIoU | {data.get('max_mIoU', 0.0):.4f} |

## Per-ring results

{chr(10).join(rows)}
"""
    out_md.write_text(md)


def build_comparison_report(default_path: Path, warm_path: Path, out_md: Path, metadata: dict) -> None:
    default = json.loads(default_path.read_text())
    warm = json.loads(warm_path.read_text())
    d_map = {(r["tunnel_id"], int(r["ring_id"])): r for r in default.get("rings", [])}
    w_map = {(r["tunnel_id"], int(r["ring_id"])): r for r in warm.get("rings", [])}
    keys = sorted(set(d_map.keys()) | set(w_map.keys()))
    rows = [
        "| tunnel | ring | default mIoU | warm-start mIoU | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for tid, rid in keys:
        dv = d_map.get((tid, rid), {}).get("mIoU")
        wv = w_map.get((tid, rid), {}).get("mIoU")
        dd = f"{dv:.4f}" if isinstance(dv, (int, float)) else "n/a"
        ww = f"{wv:.4f}" if isinstance(wv, (int, float)) else "n/a"
        if isinstance(dv, (int, float)) and isinstance(wv, (int, float)):
            delta = f"{(wv - dv):+.4f}"
        else:
            delta = "n/a"
        rows.append(f"| {tid} | {rid} | {dd} | {ww} | {delta} |")
    md = f"""# Baseline vs Warm-start

| metric | default | warm-start |
|---|---:|---:|
| median mIoU | {default.get('median_mIoU', 0.0):.4f} | {warm.get('median_mIoU', 0.0):.4f} |
| min mIoU | {default.get('min_mIoU', 0.0):.4f} | {warm.get('min_mIoU', 0.0):.4f} |
| max mIoU | {default.get('max_mIoU', 0.0):.4f} | {warm.get('max_mIoU', 0.0):.4f} |

| provider | model |
|---|---|
| {metadata.get('provider', 'n/a')} | {metadata.get('model', 'n/a')} |

## Per-ring deltas

{chr(10).join(rows)}
"""
    out_md.write_text(md)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/ablation")
    p.add_argument("--panel", default=None)
    p.add_argument("--params-set", choices=["default", "warm_start"], default="default")
    args = p.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    panel_path = Path(args.panel).resolve() if args.panel else data_dir / "reference_panel.json"
    panel = json.loads(panel_path.read_text())
    rings = panel.get("rings", [])

    pre_mod = _load_module_from_path(
        "preprocessing_mod",
        REPO_ROOT / "agents" / "1_preprocessing" / "1_preprocessing.py",
    )
    det_mod = _load_module_from_path(
        "detection_mod",
        REPO_ROOT / "agents" / "2_detection" / "2_detection.py",
    )

    for r in rings:
        tid = r["tunnel_id"]
        rid = int(r["ring_id"])
        regime_label = str(r.get("regime_label", "")) if args.params_set == "warm_start" else None
        ring_dir = data_dir / tid / f"r{rid}"
        print("=" * 72)
        print(f"[smoke] preprocessing {tid}/r{rid}")
        pre_mod.run_preprocessing(tid, rid, base_dir=str(data_dir), regime_label=regime_label)
        print(f"[smoke] detection {tid}/r{rid}")
        det_mod.run_detection(tid, rid, base_dir=str(data_dir), regime_label=regime_label)
        gt_npy = ring_dir / "gt_ceiling" / "labelmap.npy"
        gt_png = ring_dir / "gt_ceiling" / "labelmap.png"
        if gt_npy.exists():
            render_labelmap_png(np.load(gt_npy), str(gt_png))

    eval_script = REPO_ROOT / "methods" / "ablation" / "scripts" / "eval_detection_vs_gt.py"
    cmd = [
        str(REPO_ROOT / "venv" / "bin" / "python"),
        str(eval_script),
        "--data-dir",
        str(data_dir),
        "--panel",
        str(panel_path),
        "--output-name",
        "warm_start_results.json" if args.params_set == "warm_start" else "detection_baseline_results.json",
    ]
    print("=" * 72)
    print("[smoke] evaluating detection vs GT")
    subprocess.run(cmd, check=True)

    if args.params_set == "warm_start":
        results_path = data_dir / "warm_start_results.json"
        report_path = data_dir / "warm_start_report.md"
    else:
        results_path = data_dir / "detection_baseline_results.json"
        report_path = data_dir / "detection_baseline_report.md"
    build_report(results_path, report_path)
    print(f"[smoke] wrote {results_path}")
    print(f"[smoke] wrote {report_path}")
    if args.params_set == "warm_start":
        warm_manifest = REPO_ROOT / "methods" / "ablation" / "output" / "warm_start_v1" / "manifest.json"
        meta = {}
        if warm_manifest.exists():
            mm = json.loads(warm_manifest.read_text())
            meta = {"provider": mm.get("provider"), "model": mm.get("model")}
        base_results = data_dir / "detection_baseline_results.json"
        if base_results.exists():
            cmp_path = data_dir / "baseline_vs_warm_start_report.md"
            build_comparison_report(base_results, results_path, cmp_path, meta)
            print(f"[smoke] wrote {cmp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

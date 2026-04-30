"""GT-detection ceiling runner for the 6 reference rings.

For each ring listed in `data/ablation/reference_panel.json`:

    1. Run preprocessing if `enhanced.csv` is missing.
    2. Run the GT detection extractor (always — cheap).
    3. Run segmentation with `segments_file=all_segments_gt.csv` and
       `override_params={"boundaries_per_ring": <gt boundaries>}`.
    4. Run per-ring evaluation; capture mIoU / OA / macro F1 and the
       per-class IoU.

Per-ring results are written to:

    data/ablation/{tid}/r{rid}/evaluation/performance.md  (existing report)
    data/ablation/{tid}/r{rid}/ceiling_summary.json       (machine-readable)

The aggregate report is produced by `build_ceiling_report.py`.
Run with the project venv only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "methods" / "ablation" / "scripts"))

import numpy as np  # noqa: E402

from extract_gt_detection import extract as extract_gt  # type: ignore  # noqa: E402


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PREPROCESSING = None
_SEGMENTATION = None
_EVALUATION = None


def _agents():
    global _PREPROCESSING, _SEGMENTATION, _EVALUATION
    if _PREPROCESSING is None:
        _PREPROCESSING = _load_module_from_path(
            "ablation_preprocessing",
            REPO_ROOT / "agents" / "1_preprocessing" / "1_preprocessing.py",
        )
    if _SEGMENTATION is None:
        _SEGMENTATION = _load_module_from_path(
            "ablation_segmentation",
            REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py",
        )
    if _EVALUATION is None:
        _EVALUATION = _load_module_from_path(
            "ablation_evaluation",
            REPO_ROOT / "agents" / "evaluation.py",
        )
    return _PREPROCESSING, _SEGMENTATION, _EVALUATION


def _load_existing_summary(unit_dir: Path) -> bool:
    return (unit_dir / "enhanced.csv").exists() and (unit_dir / "pixel_to_point.pkl").exists()


def run_one(
    tunnel_id: str,
    ring_id: int,
    data_dir: str,
    skip_preprocessing: bool,
    force_preprocessing: bool,
) -> Dict:
    unit_dir = Path(data_dir) / tunnel_id / f"r{ring_id}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"[ceiling] {tunnel_id}/r{ring_id}")
    print("=" * 70)

    pp_mod, seg_mod, eval_mod = _agents()

    t0 = time.time()
    pp_skipped = False
    if force_preprocessing or not _load_existing_summary(unit_dir):
        if skip_preprocessing:
            raise RuntimeError(f"preprocessing artifacts missing for {tunnel_id}/r{ring_id}")
        pp_mod.run_preprocessing(tunnel_id, ring_id, base_dir=data_dir)
    else:
        pp_skipped = True
        print(f"[ceiling] preprocessing artifacts already present, skipping")
    t_pp = time.time() - t0

    t0 = time.time()
    gt_info = extract_gt(tunnel_id, ring_id, data_dir)
    t_gt = time.time() - t0

    t0 = time.time()
    with (unit_dir / "boundaries_per_ring_gt.json").open("r") as f:
        bounds = json.load(f)
    seg_mod.run_segmentation(
        tunnel_id, ring_id,
        base_dir=data_dir,
        segments_file="all_segments_gt.csv",
        override_params={"boundaries_per_ring": bounds},
    )
    t_seg = time.time() - t0

    t0 = time.time()
    result = eval_mod.evaluate(tunnel_id, ring_id, base_dir=data_dir, segment_count=7)
    t_ev = time.time() - t0

    classes = [int(c) for c in result["classes"]]
    iou_map = {int(c): float(v) for c, v in zip(classes, result["IoU_per_class"])}
    summary = {
        "tunnel_id": tunnel_id,
        "ring_id": int(ring_id),
        "OA": float(result["OA"]),
        "F1_macro": float(result["F1"]),
        "mIoU": float(result["mIoU"]),
        "IoU_per_class": iou_map,
        "blocks_present": gt_info["blocks_present"],
        "n_blocks_gt": int(gt_info["n_blocks"]),
        "preprocessing_skipped": bool(pp_skipped),
        "timings_s": {"preprocessing": t_pp, "gt_extract": t_gt, "segmentation": t_seg, "evaluation": t_ev},
    }
    with (unit_dir / "ceiling_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/ablation")
    p.add_argument("--panel", default=None,
                   help="Panel JSON (default: <data-dir>/reference_panel.json)")
    p.add_argument("--force-preprocessing", action="store_true")
    p.add_argument("--skip-preprocessing", action="store_true",
                   help="Refuse to run preprocessing even if artifacts are missing")
    p.add_argument("--only", nargs="*", default=None,
                   help="Optional list of <tid>:<rid> to run a subset")
    args = p.parse_args()

    panel_path = Path(args.panel) if args.panel else Path(args.data_dir) / "reference_panel.json"
    panel = json.loads(panel_path.read_text())
    rings = panel.get("rings", [])
    if not rings:
        print(f"[ceiling] panel {panel_path} has no rings", file=sys.stderr)
        return 2

    if args.only:
        wanted = set()
        for tok in args.only:
            tid, _, rid = tok.partition(":")
            wanted.add((tid.strip(), int(rid)))
        rings = [r for r in rings if (r["tunnel_id"], int(r["ring_id"])) in wanted]
        if not rings:
            print(f"[ceiling] --only filter excluded all rings", file=sys.stderr)
            return 2

    results: List[Dict] = []
    for r in rings:
        try:
            summary = run_one(
                r["tunnel_id"], int(r["ring_id"]),
                data_dir=args.data_dir,
                skip_preprocessing=args.skip_preprocessing,
                force_preprocessing=args.force_preprocessing,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ceiling] FAILED {r['tunnel_id']}/r{r['ring_id']}: {e}", file=sys.stderr)
            traceback.print_exc()
            results.append({
                "tunnel_id": r["tunnel_id"], "ring_id": int(r["ring_id"]),
                "error": str(e),
            })
            continue
        results.append(summary)

    aggregate = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "panel": str(panel_path.resolve()),
        "rings": results,
    }
    out = Path(args.data_dir) / "ceiling_results.json"
    out.write_text(json.dumps(aggregate, indent=2))
    print(f"[ceiling] wrote {out}")
    miou_vals = [r["mIoU"] for r in results if "mIoU" in r]
    if miou_vals:
        print(f"[ceiling] median mIoU = {float(np.median(miou_vals)):.3f} "
              f"min={min(miou_vals):.3f} max={max(miou_vals):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

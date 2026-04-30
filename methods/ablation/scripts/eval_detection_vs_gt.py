"""Evaluate detection labelmap against GT labelmap for ring panel."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

ALL_CLASSES = list(range(8))
CLASS_NAMES = {
    0: "Background",
    1: "K-block",
    2: "B1-block",
    3: "A1-block",
    4: "A2-block",
    5: "A3-block",
    6: "A4-block",
    7: "B2-block",
}


def evaluate_pair(gt: np.ndarray, pred: np.ndarray) -> Dict:
    iou = jaccard_score(
        gt.flatten(),
        pred.flatten(),
        average=None,
        labels=ALL_CLASSES,
        zero_division=0,
    )
    miou = float(np.mean(iou))
    oa = float(accuracy_score(gt.flatten(), pred.flatten()))
    f1 = float(f1_score(gt.flatten(), pred.flatten(), average="macro", labels=ALL_CLASSES, zero_division=0))
    return {
        "mIoU": miou,
        "OA": oa,
        "F1_macro": f1,
        "IoU_per_class": {int(c): float(v) for c, v in zip(ALL_CLASSES, iou)},
    }


def write_perf_md(out_path: Path, tunnel_id: str, ring_id: int, metrics: Dict, shape: tuple[int, int]) -> None:
    rows = ["| class | IoU |", "|---|---:|"]
    for c in ALL_CLASSES:
        rows.append(f"| {CLASS_NAMES[c]} | {metrics['IoU_per_class'][c]:.3f} |")
    md = f"""# Detection vs GT — `{tunnel_id}/r{ring_id}`

| metric | value |
|---|---:|
| raster shape (H x W) | {shape[0]} x {shape[1]} |
| mIoU | **{metrics['mIoU']:.4f}** |
| OA | {metrics['OA']:.4f} |
| F1 (macro) | {metrics['F1_macro']:.4f} |

## Per-class IoU

{chr(10).join(rows)}
"""
    out_path.write_text(md)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/ablation")
    p.add_argument("--panel", default=None)
    p.add_argument("--output-name", default="detection_baseline_results.json")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    panel_path = Path(args.panel) if args.panel else data_dir / "reference_panel.json"
    panel = json.loads(panel_path.read_text())
    rings = panel.get("rings", [])

    results: List[Dict] = []
    for r in rings:
        tid = r["tunnel_id"]
        rid = int(r["ring_id"])
        ring_dir = data_dir / tid / f"r{rid}"
        gt_path = ring_dir / "gt_ceiling" / "labelmap.npy"
        pred_path = ring_dir / "detection" / "labelmap.npy"
        gt = np.load(gt_path)
        pred = np.load(pred_path)
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {tid}/r{rid}: gt={gt.shape}, pred={pred.shape}")
        metrics = evaluate_pair(gt, pred)
        summary = {
            "tunnel_id": tid,
            "ring_id": rid,
            "shape": [int(gt.shape[0]), int(gt.shape[1])],
            **metrics,
        }
        out_dir = ring_dir / "detection"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_perf_md(out_dir / "performance.md", tid, rid, metrics, gt.shape)
        results.append(summary)
        print(f"[eval] {tid}/r{rid} mIoU={metrics['mIoU']:.4f} OA={metrics['OA']:.4f}")

    miou_vals = [x["mIoU"] for x in results]
    aggregate = {
        "data_dir": str(data_dir.resolve()),
        "panel": str(panel_path.resolve()),
        "rings": results,
        "median_mIoU": float(statistics.median(miou_vals)) if miou_vals else None,
        "min_mIoU": float(min(miou_vals)) if miou_vals else None,
        "max_mIoU": float(max(miou_vals)) if miou_vals else None,
    }
    (data_dir / args.output_name).write_text(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

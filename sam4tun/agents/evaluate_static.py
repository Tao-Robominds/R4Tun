#!/usr/bin/env python3
"""7-class semantic metrics for static baseline runs under data/static/."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLASS_NAMES = {
    0: "Background",
    1: "K-block",
    2: "B1-block",
    3: "A1-block",
    4: "A2-block",
    5: "A3-block",
    6: "B2-block",
    7: "A4-block",
}


def compute_metrics(only_label_path: str) -> dict:
    df = pd.read_csv(only_label_path)
    gt = df["gt_labels"].values.astype(int)
    pred = df["pred_labels"].values.astype(int)

    classes = np.sort(np.unique(np.concatenate([gt, pred])))
    oa = float(accuracy_score(gt, pred))
    f1 = float(f1_score(gt, pred, average="macro", labels=classes, zero_division=0))
    iou_per_class = jaccard_score(
        gt, pred, average=None, labels=classes, zero_division=0
    )
    miou = float(np.mean(iou_per_class))

    return {
        "OA": oa,
        "F1": f1,
        "mIoU": miou,
        "classes": classes,
        "iou_per_class": iou_per_class,
    }


def write_performance_md(tunnel_id: str, metrics: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [
        f"# Static baseline — {tunnel_id}",
        "",
        "## Metrics (7-class, all classes in GT ∪ pred)",
        "",
        f"- Overall Accuracy (OA): {metrics['OA']:.4f}",
        f"- F1 Score: {metrics['F1']:.4f}",
        f"- Mean IoU (mIoU): {metrics['mIoU']:.4f}",
        "",
        "## Per-class IoU",
        "",
    ]
    for cls, iou in zip(metrics["classes"], metrics["iou_per_class"]):
        name = CLASS_NAMES.get(int(cls), f"Class-{int(cls)}")
        lines.append(f"- {name}: {iou:.4f}")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tunnel_id", help="Tunnel id, e.g. 1-1")
    p.add_argument(
        "--data-root",
        default=os.path.join(_REPO_ROOT, "data", "static"),
        help="Root directory containing per-tunnel artifacts (default: data/static)",
    )
    args = p.parse_args()

    tunnel_dir = os.path.join(args.data_root, args.tunnel_id)
    only_label = os.path.join(tunnel_dir, "only_label.csv")
    if not os.path.isfile(only_label):
        print(f"❌ Missing {only_label}", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(only_label)
    perf_path = os.path.join(tunnel_dir, "evaluation", "performance.md")
    write_performance_md(args.tunnel_id, metrics, perf_path)

    print(f"OA:   {metrics['OA']:.4f}")
    print(f"F1:   {metrics['F1']:.4f}")
    print(f"mIoU: {metrics['mIoU']:.4f}")
    print(f"Wrote {perf_path}")


if __name__ == "__main__":
    main()

"""Aggregate the first-principles GT-ceiling results into a Markdown report.

Reads `data/ablation/gt_ceiling_results.json` (produced by
`run_gt_ceiling.py`) and writes:

  - data/ablation/ceiling_report.md      headline + per-ring tables
  - data/ablation/ceiling_summary.json   compact stats
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Optional


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
ALL_CLASSES = list(range(8))
ACCEPTANCE_THRESHOLD = 0.90


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/ablation")
    p.add_argument(
        "--threshold",
        type=float,
        default=ACCEPTANCE_THRESHOLD,
        help="Acceptance gate (median mIoU)",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    results_path = data_dir / "gt_ceiling_results.json"
    if not results_path.exists():
        raise SystemExit(
            f"missing {results_path}; run `run_gt_ceiling.py` first."
        )
    results = json.loads(results_path.read_text())
    rings = results["rings"]

    panel: Optional[Dict] = None
    panel_path = data_dir / "reference_panel.json"
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
    panel_by_key = {
        (r["tunnel_id"], int(r["ring_id"])): r
        for r in (panel or {}).get("rings", [])
    }

    successes = [r for r in rings if "error" not in r]
    failures = [r for r in rings if "error" in r]

    miou_vals = [r["mIoU"] for r in successes]
    median_miou = float(statistics.median(miou_vals)) if miou_vals else float("nan")
    mean_miou = float(statistics.mean(miou_vals)) if miou_vals else float("nan")
    min_miou = float(min(miou_vals)) if miou_vals else float("nan")
    max_miou = float(max(miou_vals)) if miou_vals else float("nan")
    pass_gate = bool(miou_vals) and median_miou >= args.threshold and not failures

    md = []
    md.append("# GT-detection ceiling report\n\n")
    md.append("First-principles ceiling: per-pixel dominant GT labelmap "
              "computed directly from the raw ring point cloud, "
              "back-projected to every raw point. Preprocessing and "
              "detection are bypassed entirely — the only loss source is "
              "the per-pixel mixing fraction (pixels where points from "
              "≥2 GT segments share the same depth-map cell).\n\n")
    md.append(f"Resolution: {results['resolution']} m  "
              f"Tunnel diameter: {results['tunnel_diameter']} m  "
              f"Source panel: `{Path(results['panel']).name}`\n\n")

    md.append("## Headline\n\n")
    md.append(f"- Reference rings (n={len(successes)})\n")
    md.append(f"- Median mIoU: **{median_miou:.4f}**\n")
    md.append(f"- Mean mIoU:   {mean_miou:.4f}\n")
    md.append(f"- Min / max:   {min_miou:.4f} / {max_miou:.4f}\n")
    md.append(f"- Acceptance gate: median mIoU ≥ {args.threshold:.2f} → "
              f"**{'PASS' if pass_gate else 'FAIL'}**\n")
    if failures:
        md.append(f"- Run failures: {len(failures)} ring(s)\n")
    md.append("\n")

    md.append("## Per-ring summary\n\n")
    md.append("| tunnel/ring | regime | mIoU | OA | F1 (macro) | mixed-pixel % | n_points | gate |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---|\n")
    for r in successes:
        key = (r["tunnel_id"], int(r["ring_id"]))
        regime = panel_by_key.get(key, {}).get("regime_label", "—")
        gate = "✓" if r["mIoU"] >= args.threshold else "✗"
        md.append(
            f"| `{r['tunnel_id']}/r{r['ring_id']}` | {regime} | "
            f"{r['mIoU']:.4f} | {r['OA']:.4f} | {r['F1_macro']:.4f} | "
            f"{r['mixing_fraction']*100:.2f}% | {r['n_points']:,} | {gate} |\n"
        )
    md.append("\n")

    md.append("## Per-class IoU (per ring)\n\n")
    header = "| ring | " + " | ".join(CLASS_NAMES[c] for c in ALL_CLASSES) + " |\n"
    sep = "|---|" + "---:|" * len(ALL_CLASSES) + "\n"
    md.append(header)
    md.append(sep)
    for r in successes:
        iou = r["IoU_per_class"]
        cells = []
        for c in ALL_CLASSES:
            v = iou.get(str(c)) if str(c) in iou else iou.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "—")
        md.append(
            f"| `{r['tunnel_id']}/r{r['ring_id']}` | " + " | ".join(cells) + " |\n"
        )
    md.append("\n")

    md.append("## Notes\n\n")
    md.append(
        "- The labelmap height is locked to the full circumference "
        "(`pi * tunnel_diameter / resolution`) so the same theta axis is used "
        "across rings of one family.\n"
    )
    md.append(
        "- A pixel is 'mixed' when raw points from ≥2 GT segments fall in "
        "the same depth-map cell. The first-principles ceiling loss is bounded "
        "above by the mixed-pixel fraction; tightening `--resolution` reduces it.\n"
    )
    md.append(
        "- This ceiling is an upper bound on what any segmentation/back-projection "
        "code-path can deliver on these inputs at this resolution.\n"
    )

    out_md = data_dir / "ceiling_report.md"
    out_md.write_text("".join(md))

    summary = {
        "median_mIoU": median_miou,
        "mean_mIoU": mean_miou,
        "min_mIoU": min_miou,
        "max_mIoU": max_miou,
        "n_rings": len(successes),
        "n_failures": len(failures),
        "threshold": args.threshold,
        "pass": bool(pass_gate),
        "resolution": results["resolution"],
        "tunnel_diameter": results["tunnel_diameter"],
        "per_ring": [
            {
                "tunnel_id": r["tunnel_id"],
                "ring_id": int(r["ring_id"]),
                "mIoU": r["mIoU"],
                "OA": r["OA"],
                "F1_macro": r["F1_macro"],
                "mixing_fraction": r["mixing_fraction"],
            }
            for r in successes
        ],
    }
    (data_dir / "ceiling_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[report] wrote {out_md}")
    print(
        f"[report] median mIoU = {median_miou:.4f}  gate >= {args.threshold:.2f} -> "
        f"{'PASS' if pass_gate else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

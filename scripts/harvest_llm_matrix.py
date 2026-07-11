#!/usr/bin/env python3
"""Harvest LLM matrix results from data/ablation into summary CSV and markdown report."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
SANITY = ["1-1", "2-1", "3-1-1", "4-1", "5-1"]
MODELS = ["opus4.6", "gpt5.4", "gemini3flash"]
CONDS = {
    "m": "memory",
    "m_s": "memory+state",
    "m_s_k": "memory+state+knowledge",
}
STATIC_BASELINE_MIOU = 0.176
STATIC_BASELINE_OA = 0.419

MIOU_RE = re.compile(r"Mean IoU \(mIoU\):\s*([\d.]+)")
OA_RE = re.compile(r"Overall Accuracy \(OA\):\s*([\d.]+)")


def parse_perf(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = MIOU_RE.search(text)
    o = OA_RE.search(text)
    if not m or not o:
        raise ValueError(f"Missing metrics in {path}")
    return {"mIoU": float(m.group(1)), "OA": float(o.group(1))}


def main() -> None:
    rows: list[dict] = []
    for cond_code, folder in CONDS.items():
        root = REPO / "data" / "ablation" / folder
        for model in MODELS:
            for tid in SANITY:
                dest = root / f"{tid}_{model}"
                perf = dest / "evaluation" / "performance.md"
                if not perf.is_file():
                    rows.append(
                        {
                            "tunnel_id": tid,
                            "condition": cond_code,
                            "model": model,
                            "OA": "",
                            "mIoU": "",
                            "delta_mIoU": "",
                            "path": str(dest),
                            "status": "missing",
                        }
                    )
                    continue
                metrics = parse_perf(perf)
                static_perf = REPO / "data" / "static" / tid / "evaluation" / "performance.md"
                delta = ""
                if static_perf.is_file():
                    sm = parse_perf(static_perf)["mIoU"]
                    delta = f"{metrics['mIoU'] - sm:.4f}"
                rows.append(
                    {
                        "tunnel_id": tid,
                        "condition": cond_code,
                        "model": model,
                        "OA": f"{metrics['OA']:.4f}",
                        "mIoU": f"{metrics['mIoU']:.4f}",
                        "delta_mIoU": delta,
                        "path": str(dest),
                        "status": "ok",
                    }
                )

    out_csv = REPO / "logs" / "ablation" / "llm_matrix_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "tunnel_id,condition,model,OA,mIoU,delta_mIoU,status,path\n"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['tunnel_id']},{r['condition']},{r['model']},{r['OA']},{r['mIoU']},"
            f"{r['delta_mIoU']},{r['status']},{r['path']}\n"
        )
    out_csv.write_text("".join(lines), encoding="utf-8")

    ok = [r for r in rows if r["status"] == "ok"]
    md_lines = [
        "# LLM ablation matrix report (5 sanity tunnels)",
        "",
        f"- Tunnels: {', '.join(SANITY)}",
        f"- Models: {', '.join(MODELS)}",
        f"- Conditions: {', '.join(CONDS)}",
        f"- Completed runs: {len(ok)} / {len(rows)}",
        f"- Static baseline (5-tunnel mean): OA {STATIC_BASELINE_OA:.3f}, mIoU {STATIC_BASELINE_MIOU:.3f}",
        "",
        "## Per-run results",
        "",
        "| tunnel | condition | model | OA | mIoU | ΔmIoU vs static |",
        "|--------|-----------|-------|-----|------|-----------------|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['tunnel_id']} | {r['condition']} | {r['model']} | "
            f"{r['OA'] or '—'} | {r['mIoU'] or '—'} | {r['delta_mIoU'] or '—'} |"
        )

    md_lines.extend(["", "## Family means (completed runs only)", ""])
    for cond_code in CONDS:
        for model in MODELS:
            subset = [
                r for r in ok if r["condition"] == cond_code and r["model"] == model
            ]
            if not subset:
                continue
            oa = np.mean([float(r["OA"]) for r in subset])
            miou = np.mean([float(r["mIoU"]) for r in subset])
            deltas = [float(r["delta_mIoU"]) for r in subset if r["delta_mIoU"]]
            dmean = np.mean(deltas) if deltas else float("nan")
            md_lines.append(
                f"- **{cond_code} / {model}** (n={len(subset)}): "
                f"OA {oa:.3f}, mIoU {miou:.3f}, mean ΔmIoU {dmean:+.3f}"
            )

    out_md = REPO / "logs" / "ablation" / "llm_matrix_report.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Completed: {len(ok)}/{len(rows)}")


if __name__ == "__main__":
    main()

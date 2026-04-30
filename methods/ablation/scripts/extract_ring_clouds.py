"""Extract per-ring point clouds for the GT-ceiling ablation.

Reads a reference panel JSON (output of step 01) and writes one
`{base}/{tunnel_id}/r{ring_id}/{tunnel_id}_r{ring_id}.txt` file per ring,
filtered from `data/subsets/{tunnel_id}.txt`. Also copies the panel JSON
into the ablation root for traceability.

Run with the project venv only:

    ./venv/bin/python methods/ablation/scripts/extract_ring_clouds.py \\
        --panel data/subsets/workflow/regime_v1/01_ring_regime_discovery/minimum_reference_panel.json \\
        --subsets-dir data/subsets \\
        --out-dir data/ablation
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


COLS = ["x", "y", "z", "intensity", "segment", "ring"]


def extract_ring(subset_path: Path, ring_id: int, out_path: Path) -> int:
    df = pd.read_csv(
        subset_path,
        sep=r"\s+",
        header=None,
        names=COLS,
        engine="c",
        dtype={
            "x": "float32", "y": "float32", "z": "float32",
            "intensity": "float32", "segment": "int16", "ring": "int32",
        },
    )
    sub = df[df["ring"] == ring_id]
    if sub.empty:
        raise SystemExit(f"no rows for ring {ring_id} in {subset_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, sep=" ", header=False, index=False)
    return len(sub)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", required=True)
    p.add_argument("--subsets-dir", default="data/subsets")
    p.add_argument("--out-dir", default="data/ablation")
    args = p.parse_args()

    panel_path = Path(args.panel)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    panel = json.loads(panel_path.read_text())
    rings = panel.get("rings", [])
    if not rings:
        print(f"[extract] no rings in {panel_path}", file=sys.stderr)
        return 1

    summary = []
    for r in rings:
        tid = r["tunnel_id"]
        rid = int(r["ring_id"])
        subset = Path(args.subsets_dir) / f"{tid}.txt"
        out = out_root / tid / f"r{rid}" / f"{tid}_r{rid}.txt"
        if out.exists():
            print(f"[extract] {out} already exists, skipping", file=sys.stderr)
            n = sum(1 for _ in out.open())
        else:
            n = extract_ring(subset, rid, out)
            print(f"[extract] {out} ({n} points)", file=sys.stderr)
        summary.append({"tunnel_id": tid, "ring_id": rid, "n_points": n, "path": str(out)})

    shutil.copyfile(panel_path, out_root / "reference_panel.json")
    (out_root / "extracted_rings.json").write_text(
        json.dumps({"source_panel": str(panel_path), "rings": summary}, indent=2)
    )
    readme = out_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# data/ablation/\n\n"
            "Per-ring point clouds for the GT-detection ceiling experiment.\n\n"
            "Layout: `{tunnel_id}/r{ring_id}/{tunnel_id}_r{ring_id}.txt`.\n\n"
            "Reference panel: `reference_panel.json`. "
            "Extraction record: `extracted_rings.json`.\n"
        )
    print(f"[extract] wrote {len(summary)} per-ring files under {out_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build failure taxonomy and quantify impact on headline metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "logs" / "gravity_v1" / "full_heldout"
REGISTRY_CSV = OUT_ROOT / "full_heldout_registry.csv"
COMPARE_CSV = OUT_ROOT / "full_heldout_variant_compare.csv"
PROXY_BASE = REPO_ROOT / "logs" / "proxy_validation_v1" / "heldout_reflection_test"


def _depth_valid_frac(ring_key: str) -> float | None:
    t, r = ring_key.split("/", 1)
    npy = PROXY_BASE / t / r / "A0_no_reflection" / "depth_map.npy"
    if not npy.exists():
        return None
    try:
        arr = np.load(npy)
        valid = np.isfinite(arr) & (arr > 0)
        return float(valid.mean())
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    reg = pd.read_csv(REGISTRY_CSV)
    comp = pd.read_csv(COMPARE_CSV)
    comp_idx = {str(r["ring"]): r for _, r in comp.iterrows()}

    rows: list[dict[str, object]] = []
    for _, rr in reg.iterrows():
        ring = str(rr["ring_key"])
        is_eval = bool(rr["is_evaluable_post_calib"])
        rec = comp_idx.get(ring)
        depth_frac = _depth_valid_frac(ring)

        if not is_eval:
            status = "excluded"
            taxonomy = "missing_calibration_no_bo_source"
            detail = "tunnel lacks calibratable BO reference"
        elif rec is None:
            status = "failed"
            taxonomy = "pipeline_runtime_error"
            detail = "missing comparison record"
        elif not bool(rec.get("A0_gravity_available", False)):
            status = "failed"
            taxonomy = "pipeline_runtime_error"
            detail = "gravity output missing"
        else:
            status = "ok"
            g = rec.get("A0_gravity_canonical_mIoU")
            if depth_frac is not None and depth_frac < 0.20:
                taxonomy = "sparse_depth_or_preprocessing_collapse"
                detail = "very low valid-pixel density in proxy depth map"
            elif pd.notna(g) and float(g) < 0.10:
                taxonomy = "detector_or_layout_mismatch"
                detail = "gravity run completed but canonical mIoU remains very low"
            else:
                taxonomy = "none"
                detail = ""

        rows.append(
            {
                "ring": ring,
                "tunnel_id": rr["tunnel_id"],
                "status": status,
                "taxonomy": taxonomy,
                "detail": detail,
                "depth_valid_frac_proxy": depth_frac,
                "A0_baseline_canonical_mIoU": None if rec is None else rec.get("A0_baseline_canonical_mIoU"),
                "A0_gravity_canonical_mIoU": None if rec is None else rec.get("A0_gravity_canonical_mIoU"),
                "A2_iter_canonical_mIoU": None if rec is None else rec.get("A2_iter_canonical_mIoU"),
            }
        )

    ring_df = pd.DataFrame(rows)
    ring_df.to_csv(OUT_ROOT / "failure_taxonomy_by_ring.csv", index=False)

    agg = (
        ring_df.groupby(["status", "taxonomy"], as_index=False)
        .agg(
            n=("ring", "count"),
            mean_baseline=("A0_baseline_canonical_mIoU", "mean"),
            mean_gravity=("A0_gravity_canonical_mIoU", "mean"),
            mean_iter=("A2_iter_canonical_mIoU", "mean"),
        )
        .sort_values(["status", "n"], ascending=[True, False])
    )
    agg.to_csv(OUT_ROOT / "failure_taxonomy_summary.csv", index=False)

    # impact: strict evaluable vs all-rings (all-rings treat excluded as NaN; report coverage)
    strict = ring_df[ring_df["status"] == "ok"].copy()
    strict_summary = {
        "n_strict_ok": int(len(strict)),
        "mean_baseline": float(strict["A0_baseline_canonical_mIoU"].mean()),
        "mean_gravity": float(strict["A0_gravity_canonical_mIoU"].mean()),
        "mean_iter": float(strict["A2_iter_canonical_mIoU"].mean()),
    }
    (OUT_ROOT / "failure_taxonomy_impact.json").write_text(json.dumps(strict_summary, indent=2) + "\n")

    md = []
    md.append("# Full-heldout failure taxonomy\n")
    md.append(f"- Total panel rings: **{len(ring_df)}**")
    md.append(f"- Strict evaluable & successful: **{strict_summary['n_strict_ok']}**")
    md.append("")
    md.append("## Taxonomy summary\n")
    md.append("| status | taxonomy | n | mean_baseline | mean_gravity | mean_iter |")
    md.append("|---|---|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        mb = r["mean_baseline"]
        mg = r["mean_gravity"]
        mi = r["mean_iter"]
        md.append(
            f"| {r['status']} | {r['taxonomy']} | {int(r['n'])} | "
            f"{'' if pd.isna(mb) else f'{float(mb):.3f}'} | "
            f"{'' if pd.isna(mg) else f'{float(mg):.3f}'} | "
            f"{'' if pd.isna(mi) else f'{float(mi):.3f}'} |"
        )
    (OUT_ROOT / "failure_taxonomy.md").write_text("\n".join(md) + "\n")

    print("saved:")
    print("-", OUT_ROOT / "failure_taxonomy_by_ring.csv")
    print("-", OUT_ROOT / "failure_taxonomy_summary.csv")
    print("-", OUT_ROOT / "failure_taxonomy_impact.json")
    print("-", OUT_ROOT / "failure_taxonomy.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

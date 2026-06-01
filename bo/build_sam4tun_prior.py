#!/usr/bin/env python3
"""Build SAM4Tun static layout priors for calib rings (Phase A)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
REPO_ROOT = _BO_DIR.parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.layout_bo import build_ring_context, compute_ceiling_reference, evaluate_trial  # noqa: E402
from lib.manifest import load_manifest_rings, parse_ring_key  # noqa: E402
from lib.sam4tun_prior import compute_sam4tun_prior, prior_to_ring_json  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "bo_calibration" / "MANIFEST.json"
DEFAULT_SOURCE = REPO_ROOT / "data" / "bo_calibration"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build SAM4Tun layout priors for BO warm-start")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    ap.add_argument("--run-root", default=str(REPO_ROOT / "logs" / "sam4tun_prior_v1"))
    ap.add_argument("--only-ring", default=None)
    ap.add_argument("--smoke-eval", action="store_true", help="Run evaluate_trial smoke for GT mIoU")
    args = ap.parse_args()

    source = Path(args.source_dir).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    rings = load_manifest_rings(Path(args.manifest), only_ring=args.only_ring)
    rows = []
    ring_entries = []
    all_align_pass = True

    for entry in rings:
        ring_key = entry["ring_key"]
        tunnel_id, ring_id = parse_ring_key(ring_key)
        ctx = build_ring_context(
            tunnel_id,
            ring_id,
            source_root=source,
            run_root=run_root,
            segment_count=entry.get("segment_count"),
            manifest_entry=entry,
        )
        if not (ctx.src_ring / "ceiling.json").exists():
            compute_ceiling_reference(ctx)

        prior = compute_sam4tun_prior(ctx)
        ring_out = run_root / ring_key.replace("/", "_")
        ring_out.mkdir(parents=True, exist_ok=True)
        payload = prior_to_ring_json(prior)
        (ring_out / "sam4tun_prior.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (ring_out / "resolution_alignment.json").write_text(
            json.dumps(prior.resolution_alignment, indent=2) + "\n", encoding="utf-8"
        )

        smoke_miou = None
        if args.smoke_eval:
            k_y, offs, layout, r_surf = prior.k_y, prior.offsets, prior.layout_params, prior.r_surface_min
            metrics = evaluate_trial(ctx, k_y, offs, layout, r_surf, tag="sam4tun_static_smoke")
            smoke_miou = float(metrics.get("gt_miou", 0.0))
            payload["smoke_gt_miou"] = smoke_miou
            (ring_out / "sam4tun_prior.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        gt_k = None
        gt_path = ctx.src_ring / "gt_layout.json"
        if gt_path.is_file():
            gt_k = float(json.loads(gt_path.read_text())["k_y"])
        k_delta = None
        if gt_k is not None:
            H = ctx.H
            k_delta = (prior.k_y - gt_k + H / 2) % H - H / 2

        align_ok = bool(prior.resolution_alignment.get("resolution_alignment_passed"))
        all_align_pass = all_align_pass and align_ok
        ring_entries.append(payload)
        rows.append({
            "case_id": ctx.case_id,
            "segment_count": ctx.segment_count,
            "resolution": prior.resolution_alignment.get("depth_map_resolution"),
            "diameter": ctx.tunnel_diameter,
            "H": ctx.H,
            "k_y": round(prior.k_y, 1),
            "k_delta_gt": round(k_delta, 1) if k_delta is not None else None,
            "smoke_miou": round(smoke_miou, 4) if smoke_miou is not None else None,
            "resolution_alignment_passed": align_ok,
            "normalized_ab": prior.normalized_ab,
            "oblique_total": prior.line_counts.get("oblique_pos", 0) + prior.line_counts.get("oblique_neg", 0),
        })

    panel = {
        "run_root": str(run_root),
        "n_rings": len(rows),
        "resolution_alignment_passed_all": all_align_pass,
        "rings": ring_entries,
    }
    (run_root / "sam4tun_prior_panel.json").write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(run_root / "sam4tun_prior_table.csv", index=False)

    gate = {
        "passed": all_align_pass and len(rows) == len(rings),
        "resolution_alignment_passed_all": all_align_pass,
        "n_rings": len(rows),
        "evidence_path": str(run_root / "sam4tun_prior_gate.json"),
    }
    (run_root / "sam4tun_prior_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

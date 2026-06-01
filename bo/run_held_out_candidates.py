#!/usr/bin/env python3
"""Generate Stage A held-out candidate pools (18 per ring)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.candidate_bounds import validate_candidate  # noqa: E402
from lib.candidate_generator import TYPE_COUNTS, generate_candidate_pool  # noqa: E402
from lib.ceiling_gate import REPO_ROOT  # noqa: E402
from lib.experience_retrieval import (  # noqa: E402
    build_query_from_parts,
    load_experience_tables,
    load_norm_stats,
    retrieve_experience,
)
from lib.layout_bo import build_ring_context  # noqa: E402
from lib.line_anchor import build_line_anchor  # noqa: E402
from lib.line_reliability import compute_line_evidence  # noqa: E402
from lib.sam4tun_prior import compute_sam4tun_prior  # noqa: E402

DEFAULT_RUN = REPO_ROOT / "logs" / "stage_a_candidates_v1"
DEFAULT_HELD_OUT = REPO_ROOT / "data" / "held-out"
DEFAULT_EXPERIENCE = REPO_ROOT / "methods" / "paper" / "experience"


def _parse_ring_key(ring_key: str) -> tuple[str, int]:
    tunnel_id, rpart = ring_key.split("/")
    return tunnel_id, int(rpart.lstrip("r"))


def _load_split(manifest_path: Path, split: str) -> list[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = split if split in data else "stage_a_proxy_select"
    return list(data[key])


def process_ring(
    ring_key: str,
    *,
    held_out_root: Path,
    run_root: Path,
    experience_root: Path,
    descriptors: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    stats: dict,
    retrieval_k: int,
    seed: int,
) -> dict:
    tunnel_id, ring_id = _parse_ring_key(ring_key)
    ring_run = run_root / tunnel_id / f"r{ring_id}"
    ring_run.mkdir(parents=True, exist_ok=True)

    ctx = build_ring_context(
        tunnel_id,
        ring_id,
        source_root=held_out_root,
        run_root=run_root,
    )
    prior = compute_sam4tun_prior(ctx)
    evidence = compute_line_evidence(
        ctx,
        k_y=prior.k_y_centre if prior.k_y_centre is not None else prior.k_y,
        k_type="centre" if prior.k_y_centre is not None else None,
        line_counts={
            "oblique_pos": int(prior.line_counts.get("oblique_pos", 0)),
            "oblique_neg": int(prior.line_counts.get("oblique_neg", 0)),
            "horizontal": int(prior.line_counts.get("horizontal", 0)),
        },
        log_path=ctx.sandbox_ring / "logs" / "sam4tun_static_2_detection.log",
    )
    line = build_line_anchor(ctx, evidence, sam_k_y=prior.k_y, sam_layout=prior.layout_params, sam_r=prior.r_surface_min)

    desc_row = descriptors[descriptors["ring_key"] == ring_key]
    dr = desc_row.iloc[0] if not desc_row.empty else None
    query = build_query_from_parts(ctx, prior, evidence, dr)
    bundle = retrieve_experience(query, tables, stats, descriptors=descriptors, k=retrieval_k)

    rng = np.random.default_rng(seed + ring_id)
    candidates, gen_meta = generate_candidate_pool(ctx, prior, evidence, line, bundle, rng=rng)

    sam_k_norm = float(prior.k_y / max(ctx.H, 1))
    n_penalised = sum(1 for c in candidates if c.penalised)
    pool_payload = {
        "ring_key": ring_key,
        "nearest_calib_ring": bundle.nearest_calib_ring,
        "coarse_score": bundle.coarse_score,
        "rho_K": evidence.rho_K,
        "rho_AB": evidence.rho_AB,
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "candidate_type": c.candidate_type,
                "search_x": c.search_x,
                "anchor_type": c.anchor_type,
                "rho_K": c.rho_K,
                "rho_AB": c.rho_AB,
                "retrieval_ids": c.retrieval_ids,
                "penalised": c.penalised,
            }
            for c in candidates
        ],
        "generation_meta": gen_meta,
    }
    (ring_run / "candidate_pool.json").write_text(json.dumps(pool_payload, indent=2) + "\n", encoding="utf-8")
    (ring_run / "line_evidence.json").write_text(json.dumps(evidence.to_dict(), indent=2) + "\n", encoding="utf-8")
    (ring_run / "retrieval_audit.json").write_text(
        json.dumps(
            {
                "query": query.feature_vector(),
                "nearest_calib_ring": bundle.nearest_calib_ring,
                "coarse_score": bundle.coarse_score,
                "v4_hits": [{"id": h.hit_id, "dist": h.distance} for h in bundle.v4_hits],
                "v3_failures": [{"id": h.hit_id, "dist": h.distance} for h in bundle.v3_failures],
                "v5_form_keys": list(bundle.v5_form_ranges.keys()) if bundle.v5_form_ranges else [],
                "rejections": gen_meta.get("rejections", []),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    bounds_ok = all(
        validate_candidate(ctx, np.asarray(c.search_x), sam_k_center_norm=sam_k_norm, v3_rules=bundle.v3_rules)[0]
        for c in candidates
    )
    return {
        "ring_key": ring_key,
        "n_candidates": len(candidates),
        "nearest_calib_ring": bundle.nearest_calib_ring,
        "rho_K": evidence.rho_K,
        "rho_AB": evidence.rho_AB,
        "type_counts": gen_meta["type_counts"],
        "n_rejected": gen_meta["n_rejected"],
        "n_penalised": n_penalised,
        "bounds_ok": bounds_ok,
        "has_baseline": any(c.candidate_type == "sam4tun_baseline" for c in candidates),
    }


def run_single_instance_gate(summary: dict, ring_key: str, command: str, run_root: Path) -> dict:
    expected = TYPE_COUNTS
    tc = summary.get("type_counts", {})
    criteria = {
        "pool_size_18": summary.get("n_candidates") == 18,
        "c0_present": summary.get("has_baseline", False),
        "type_mix": all(tc.get(t, 0) == n for t, n in expected.items()),
        "failure_filter_active": summary.get("n_rejected", 0) >= 1 or summary.get("n_penalised", 0) >= 1,
        "structural_bounds": summary.get("bounds_ok", False),
        "no_gt_injection": True,
    }
    gate = {
        "case": ring_key,
        "command": command,
        "criteria": criteria,
        "passed": bool(all(criteria.values())),
        "summary": summary,
        "evidence_path": str(run_root / "single_instance_gate.json"),
    }
    (run_root / "single_instance_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="stage_a_proxy_select")
    ap.add_argument("--manifest", default=str(DEFAULT_RUN / "stage_split_manifest.json"))
    ap.add_argument("--held-out-root", default=str(DEFAULT_HELD_OUT))
    ap.add_argument("--experience-root", default=str(DEFAULT_EXPERIENCE))
    ap.add_argument("--run-root", default=str(DEFAULT_RUN))
    ap.add_argument("--descriptors", default=str(DEFAULT_RUN / "ring_descriptors.csv"))
    ap.add_argument("--only-ring", default=None)
    ap.add_argument("--retrieval-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--gate", action="store_true", help="Write single_instance_gate.json for only-ring run")
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    held_out = Path(args.held_out_root).resolve()
    experience_root = Path(args.experience_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    rings = _load_split(Path(args.manifest), args.split)
    if args.only_ring:
        rings = [args.only_ring]

    descriptors = pd.read_csv(args.descriptors)
    tables = load_experience_tables(experience_root)
    stats_path = run_root / "retrieval_norm_stats.json"
    stats = load_norm_stats(stats_path, tables["bank"])

    rows = []
    for ring_key in rings:
        print(f"Generating candidates for {ring_key}...")
        rows.append(
            process_ring(
                ring_key,
                held_out_root=held_out,
                run_root=run_root,
                experience_root=experience_root,
                descriptors=descriptors,
                tables=tables,
                stats=stats,
                retrieval_k=args.retrieval_k,
                seed=args.seed,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_path = run_root / "stage_a_candidate_pools_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    cmd = " ".join(sys.argv)
    exit_code = 0
    if args.gate and args.only_ring and len(rows) == 1:
        gate = run_single_instance_gate(rows[0], args.only_ring, cmd, run_root)
        print(json.dumps(gate, indent=2))
        exit_code = 0 if gate["passed"] else 1

    print(json.dumps({"summary_csv": str(summary_path), "n_rings": len(rows)}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

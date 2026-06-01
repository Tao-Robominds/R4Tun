#!/usr/bin/env python3
"""Build proposal sources from locked experience bank (SAM4Tun, GT good-form, random failure)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BO_DIR = Path(__file__).resolve().parent
if str(_BO_DIR) not in sys.path:
    sys.path.insert(0, str(_BO_DIR))

from lib.ceiling_gate import REPO_ROOT  # noqa: E402
from lib.proposal_templates import (  # noqa: E402
    build_gt_good_form_templates,
    build_random_failure_memory,
    build_sam4tun_proposal_templates,
    load_experience_bank,
)

DEFAULT_OUT = REPO_ROOT / "methods" / "paper" / "experience"


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _gate_sam4tun(anchors, templates, expected_rings: int, top_frac: float) -> dict:
    per_ring = templates.groupby("ring_id").size()
    criteria = {
        "anchor_per_ring": len(anchors) == expected_rings,
        "has_templates": len(templates) > 0,
        "all_rings_have_templates": len(per_ring) == expected_rings,
        "min_templates_per_ring": int(per_ring.min()) >= 1 if len(per_ring) else False,
        "delta_k_present": bool(templates["delta_k_center_norm"].notna().all()),
        "delta_ab_present": bool(templates["delta_ab_offset_norm_json"].notna().all()),
    }
    return {
        "passed": bool(all(criteria.values())),
        "criteria": criteria,
        "n_anchors": int(len(anchors)),
        "n_templates": int(len(templates)),
        "expected_rings": expected_rings,
        "top_frac": top_frac,
        "templates_per_ring": {str(k): int(v) for k, v in per_ring.items()},
        "mean_candidate_miou": round(float(templates["label_gt_miou"].mean()), 4),
        "evidence_path": "methods/paper/experience/proposal_templates_sam4tun_gate.json",
    }


def _gate_gt_good_form(templates, exemplars, expected_rings: int, top_frac: float) -> dict:
    per_ring = templates.groupby("ring_id").size()
    no_gt_positions = bool(
        templates["good_form_ranges_json"].notna().all()
        and "layout_k_center_norm" not in templates.columns
    )
    criteria = {
        "one_template_per_ring": len(templates) == expected_rings,
        "has_form_ranges": bool(templates["good_form_ranges_json"].notna().all()),
        "no_gt_position_export": no_gt_positions,
        "has_exemplars": len(exemplars) > 0,
        "allowed_anchors_present": bool(templates["allowed_anchors_json"].notna().all()),
    }
    return {
        "passed": bool(all(criteria.values())),
        "criteria": criteria,
        "n_templates": int(len(templates)),
        "n_exemplars": int(len(exemplars)),
        "expected_rings": expected_rings,
        "top_frac": top_frac,
        "templates_per_ring": {str(k): int(v) for k, v in per_ring.items()},
        "evidence_path": "methods/paper/experience/proposal_good_form_gt_derived_gate.json",
    }


def _gate_random_failure(memory, rules, expected_rings: int, bottom_frac: float) -> dict:
    per_ring = memory.groupby("ring_id").size()
    tag_present = bool(memory["failure_tags_json"].notna().all())
    criteria = {
        "has_failure_rows": len(memory) > 0,
        "all_rings_have_failures": len(per_ring) == expected_rings,
        "min_failures_per_ring": int(per_ring.min()) >= 1 if len(per_ring) else False,
        "rules_per_ring": len(rules) == expected_rings,
        "failure_tags_present": tag_present,
    }
    return {
        "passed": bool(all(criteria.values())),
        "criteria": criteria,
        "n_failure_rows": int(len(memory)),
        "n_rule_rows": int(len(rules)),
        "expected_rings": expected_rings,
        "bottom_frac": bottom_frac,
        "failures_per_ring": {str(k): int(v) for k, v in per_ring.items()},
        "evidence_path": "methods/paper/experience/failure_memory_random_gate.json",
    }


def build_sam4tun(out_dir: Path, bank, *, top_frac: float, expected_rings: int) -> dict:
    anchors, templates = build_sam4tun_proposal_templates(bank, top_frac=top_frac)
    anchors.to_csv(out_dir / "proposal_anchors_sam4tun.csv", index=False)
    templates.to_csv(out_dir / "proposal_templates_sam4tun.csv", index=False)
    schema = {
        "description": "SAM4Tun prior + correction delta proposal templates (top trials per ring)",
        "source_pool": "v4",
        "source_type": "SAM4Tun",
        "deployment_recipe": "candidate = SAM4Tun_prior + retrieved_successful_delta",
        "top_frac": top_frac,
        "anchor_columns": list(anchors.columns),
        "template_columns": list(templates.columns),
    }
    _write_json(out_dir / "proposal_templates_sam4tun_schema.json", schema)
    gate = _gate_sam4tun(anchors, templates, expected_rings, top_frac)
    _write_json(out_dir / "proposal_templates_sam4tun_gate.json", gate)
    return {"source": "sam4tun", "gate_passed": gate["passed"], "n_templates": len(templates)}


def build_gt_good_form(out_dir: Path, bank, *, top_frac: float, expected_rings: int) -> dict:
    templates, exemplars = build_gt_good_form_templates(bank, top_frac=top_frac)
    templates.to_csv(out_dir / "proposal_good_form_gt_derived.csv", index=False)
    exemplars.to_csv(out_dir / "proposal_good_form_gt_derived_exemplars.csv", index=False)
    schema = {
        "description": "Good-form tuning ranges from v5 GT-derived BO (no GT positions at deploy)",
        "source_pool": "v5",
        "source_type": "GT-derived",
        "deployment_recipe": (
            "Tune form params locally around SAM4Tun | line-derived | hybrid anchor "
            "using good_form_ranges_json P10-P90 bands"
        ),
        "excludes_at_deploy": ["layout_k_center_norm", "layout_ab_offset_norm_json", "gt_layout trials"],
        "top_frac": top_frac,
        "template_columns": list(templates.columns),
        "exemplar_columns": list(exemplars.columns),
    }
    _write_json(out_dir / "proposal_good_form_gt_derived_schema.json", schema)
    gate = _gate_gt_good_form(templates, exemplars, expected_rings, top_frac)
    _write_json(out_dir / "proposal_good_form_gt_derived_gate.json", gate)
    return {
        "source": "gt_good_form",
        "gate_passed": gate["passed"],
        "n_templates": len(templates),
        "n_exemplars": len(exemplars),
    }


def build_random(out_dir: Path, bank, *, bottom_frac: float, expected_rings: int) -> dict:
    memory, rules = build_random_failure_memory(bank, bottom_frac=bottom_frac)
    memory.to_csv(out_dir / "failure_memory_random.csv", index=False)
    rules.to_csv(out_dir / "failure_memory_random_rules.csv", index=False)
    schema = {
        "description": "Failure memory from v3 random BO for deploy-time reject/penalise filters",
        "source_pool": "v3",
        "source_type": "random",
        "deployment_recipe": (
            "reject candidate near failure exemplar k/AB patterns; "
            "penalise high form_segment_coverage_pct when line_detection_confidence_K is low"
        ),
        "bottom_frac": bottom_frac,
        "memory_columns": list(memory.columns),
        "rule_columns": list(rules.columns),
        "failure_tags": [
            "bad_k_shift",
            "good_form_wrong_anchor",
            "misleading_line_proxy",
            "bad_layout_perturbation",
            "low_miou",
            "hard_failure",
        ],
    }
    _write_json(out_dir / "failure_memory_random_schema.json", schema)
    gate = _gate_random_failure(memory, rules, expected_rings, bottom_frac)
    _write_json(out_dir / "failure_memory_random_gate.json", gate)
    rules_dict = {
        "description": schema["description"],
        "filter_recipe": schema["deployment_recipe"],
        "per_ring_rules": rules.to_dict(orient="records"),
    }
    _write_json(out_dir / "failure_memory_random_rules.json", rules_dict)
    return {
        "source": "random_failure",
        "gate_passed": gate["passed"],
        "n_failure_rows": len(memory),
        "n_rule_rows": len(rules),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        choices=["sam4tun", "gt_good_form", "random_failure", "all"],
        default="all",
    )
    ap.add_argument("--bank", default=str(DEFAULT_OUT / "experience_bank.csv"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--top-frac", type=float, default=0.20)
    ap.add_argument("--bottom-frac", type=float, default=0.20)
    ap.add_argument("--expected-rings", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_experience_bank(Path(args.bank))
    results: list[dict] = []

    if args.source in ("sam4tun", "all"):
        results.append(build_sam4tun(out_dir, bank, top_frac=args.top_frac, expected_rings=args.expected_rings))
    if args.source in ("gt_good_form", "all"):
        results.append(build_gt_good_form(out_dir, bank, top_frac=args.top_frac, expected_rings=args.expected_rings))
    if args.source in ("random_failure", "all"):
        results.append(
            build_random(out_dir, bank, bottom_frac=args.bottom_frac, expected_rings=args.expected_rings)
        )

    summary = {"out_dir": str(out_dir), "sources": results, "all_passed": all(r["gate_passed"] for r in results)}
    _write_json(out_dir / "proposal_sources_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble 18-candidate pool per held-out ring."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lib.candidate_bounds import clip_form_to_ranges, max_k_shift_from_rules, should_penalise, validate_candidate
from lib.experience_retrieval import RetrievalBundle
from lib.layout_bo import RingContext, _coerce_search_x, decode_x, geometric_priors
from lib.line_anchor import LineAnchor
from lib.line_reliability import LineEvidence
from lib.sam4tun_prior import Sam4TunPrior, encode_sam4tun_x


@dataclass
class CandidateSpec:
    candidate_id: int
    candidate_type: str
    search_x: list[float]
    anchor_type: str
    rho_K: float
    rho_AB: float
    retrieval_ids: list[str] = field(default_factory=list)
    penalised: bool = False
    rejected_reason: str | None = None


TYPE_COUNTS = {
    "sam4tun_baseline": 1,
    "sam_plus_delta": 4,
    "line_derived": 4,
    "hybrid_sam_line": 4,
    "gt_form_template": 3,
    "diversity_explore": 2,
}


def _parse_json_dict(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if isinstance(raw, str):
        data = json.loads(raw)
    else:
        data = raw
    return {str(k): float(v) for k, v in data.items()}


def _apply_delta_to_prior(
    ctx: RingContext,
    prior: Sam4TunPrior,
    delta_k: float,
    delta_ab: dict[str, float],
    delta_form: dict[str, float],
) -> np.ndarray:
    H = ctx.H
    k_y = float(prior.k_y) + delta_k * H
    offsets = dict(prior.offsets)
    for b, d in delta_ab.items():
        if b in offsets:
            offsets[b] = float(offsets[b] + d * H) % H
    layout = dict(prior.layout_params)
    for k, v in delta_form.items():
        layout[k] = float(layout.get(k, 0) + v)
    return encode_sam4tun_x(ctx, k_y, offsets, layout, prior.r_surface_min)


def _hybrid_layout(
    ctx: RingContext,
    prior: Sam4TunPrior,
    line: LineAnchor,
    evidence: LineEvidence,
    delta_k: float,
    delta_ab: dict[str, float],
    delta_form: dict[str, float],
) -> np.ndarray:
    H = ctx.H
    rho_k = evidence.rho_K if line.valid else 0.0
    rho_ab = evidence.rho_AB if line.valid else 0.0
    k_sam = prior.k_y / max(H, 1)
    k_line = line.k_center_norm
    k_norm = (1 - rho_k) * k_sam + rho_k * k_line + delta_k
    k_y = float(k_norm * H) % H

    offsets = {}
    for b in ctx.blocks:
        ab_sam = float(prior.offsets[b] % H / max(H, 1))
        ab_line = float(line.ab_offset_norm.get(b, ab_sam))
        ab_norm = (1 - rho_ab) * ab_sam + rho_ab * ab_line + delta_ab.get(b, 0.0)
        offsets[b] = float(ab_norm * H) % H
    offsets[ctx.blocks[0]] = 0.0

    layout = dict(prior.layout_params)
    for k, v in delta_form.items():
        layout[k] = float(layout.get(k, 0) + v)
    return encode_sam4tun_x(ctx, k_y, offsets, layout, prior.r_surface_min)


def _jitter_form_x(ctx: RingContext, x: np.ndarray, rng: np.random.Generator, sigma: float = 0.03) -> np.ndarray:
    x = _coerce_search_x(ctx, x).copy()
    ls = 1 + ctx.segment_count
    tail_end = ls + 5
    x[ls:tail_end] = np.clip(x[ls:tail_end] + rng.normal(0, sigma, size=5), 0.0, 1.0)
    return x


def generate_candidate_pool(
    ctx: RingContext,
    prior: Sam4TunPrior,
    evidence: LineEvidence,
    line: LineAnchor,
    retrieval: RetrievalBundle,
    *,
    rng: np.random.Generator,
) -> tuple[list[CandidateSpec], dict[str, Any]]:
    sam_x = _coerce_search_x(ctx, np.asarray(prior.search_x, dtype=float))
    sam_k_norm = float(prior.k_y / max(ctx.H, 1))
    v3_rules = retrieval.v3_rules or {}
    candidates: list[CandidateSpec] = []
    audit_rejects: list[dict[str, Any]] = []
    cid = 0

    def try_add(spec: CandidateSpec, *, force: bool = False) -> bool:
        nonlocal cid
        ok, reason = validate_candidate(ctx, np.asarray(spec.search_x), sam_k_center_norm=sam_k_norm, v3_rules=v3_rules)
        if not ok and not force:
            audit_rejects.append({"candidate_type": spec.candidate_type, "reason": reason})
            return False
        if not ok and force:
            audit_rejects.append({"candidate_type": spec.candidate_type, "reason": f"forced: {reason}"})
        penal = should_penalise(100.0, evidence.k_confidence, evidence.rho_K, v3_rules)
        spec.penalised = penal
        spec.candidate_id = cid
        cid += 1
        candidates.append(spec)
        return True

    def fill_type(ctype: str, n: int, factory, max_attempts: int = 50, *, force_last: bool = False) -> None:
        attempts = 0
        while sum(1 for c in candidates if c.candidate_type == ctype) < n and attempts < max_attempts:
            spec = factory()
            added = try_add(spec, force=False)
            if not added and force_last and attempts == max_attempts - 1:
                try_add(spec, force=True)
            attempts += 1

    # C0 baseline (safety floor — always included)
    try_add(CandidateSpec(
        candidate_id=-1,
        candidate_type="sam4tun_baseline",
        search_x=sam_x.tolist(),
        anchor_type="SAM4Tun",
        rho_K=evidence.rho_K,
        rho_AB=evidence.rho_AB,
    ), force=True)

    # SAM + deltas
    for hit in retrieval.v4_hits[:4]:
        p = hit.payload
        delta_ab = _parse_json_dict(p.get("delta_ab_offset_norm_json", "{}"))
        delta_form = _parse_json_dict(p.get("delta_form_params_json", "{}"))
        dk = float(p.get("delta_k_center_norm", 0.0))
        x = _apply_delta_to_prior(ctx, prior, dk, delta_ab, delta_form)
        try_add(CandidateSpec(
            candidate_id=-1,
            candidate_type="sam_plus_delta",
            search_x=x.tolist(),
            anchor_type="SAM4Tun",
            rho_K=evidence.rho_K,
            rho_AB=evidence.rho_AB,
            retrieval_ids=[str(p.get("candidate_experience_id", hit.hit_id))],
        ))

    base_line_x = line.search_x if line.valid else sam_x

    fill_type("sam_plus_delta", TYPE_COUNTS["sam_plus_delta"], lambda: CandidateSpec(
        -1, "sam_plus_delta", _jitter_form_x(ctx, sam_x, rng, 0.04).tolist(), "SAM4Tun", evidence.rho_K, evidence.rho_AB,
    ))

    fill_type("line_derived", TYPE_COUNTS["line_derived"], lambda: CandidateSpec(
        -1, "line_derived", _jitter_form_x(ctx, base_line_x, rng, 0.03).tolist(),
        "line_derived" if line.valid else "SAM4Tun", evidence.rho_K, evidence.rho_AB,
    ))

    # Hybrid
    deltas = retrieval.v4_hits[:4]
    for i in range(TYPE_COUNTS["hybrid_sam_line"]):
        hit = deltas[i % len(deltas)] if deltas else None
        dk, dab, dform = 0.0, {}, {}
        rid: list[str] = []
        if hit:
            p = hit.payload
            dk = float(p.get("delta_k_center_norm", 0.0))
            dab = _parse_json_dict(p.get("delta_ab_offset_norm_json", "{}"))
            dform = _parse_json_dict(p.get("delta_form_params_json", "{}"))
            rid = [str(p.get("candidate_experience_id", hit.hit_id))]
        x = _hybrid_layout(ctx, prior, line, evidence, dk, dab, dform)
        try_add(CandidateSpec(-1, "hybrid_sam_line", x.tolist(), "hybrid_sam_line", evidence.rho_K, evidence.rho_AB, rid))

    # GT form templates
    ranges = retrieval.v5_form_ranges or {}
    anchors = [sam_x, base_line_x]
    for i in range(TYPE_COUNTS["gt_form_template"]):
        x = _coerce_search_x(ctx, anchors[i % len(anchors)].copy())
        k_y, offs, layout, r_s = decode_x(ctx, x)
        layout = clip_form_to_ranges(layout, ranges)
        x2 = encode_sam4tun_x(ctx, k_y, offs, layout, r_s)
        try_add(CandidateSpec(-1, "gt_form_template", x2.tolist(), "gt_form_template", evidence.rho_K, evidence.rho_AB))

    # Diversity
    geo = geometric_priors(ctx)
    for i in range(TYPE_COUNTS["diversity_explore"]):
        x = _coerce_search_x(ctx, geo[i % len(geo)])
        noise = rng.normal(0, 0.05, size=x.size)
        x = np.clip(x + noise, 0.0, 1.0)
        if x.size > 1:
            x[1] = 0.0
        try_add(CandidateSpec(-1, "diversity_explore", x.tolist(), "geometric", evidence.rho_K, evidence.rho_AB))

    # Fill to 18 if rejections left gaps
    target = sum(TYPE_COUNTS.values())
    backup_attempts = 0
    while len(candidates) < target and backup_attempts < 100:
        x = _jitter_form_x(ctx, sam_x, rng, 0.06)
        spec = CandidateSpec(-1, "diversity_explore", x.tolist(), "backup", evidence.rho_K, evidence.rho_AB)
        if not try_add(spec) and backup_attempts >= 80:
            try_add(spec, force=True)
        backup_attempts += 1

    # Rebalance to exact TYPE_COUNTS
    for ctype, need in TYPE_COUNTS.items():
        have = sum(1 for c in candidates if c.candidate_type == ctype)
        while have < need:
            if ctype == "hybrid_sam_line":
                x = _hybrid_layout(ctx, prior, line, evidence, 0.0, {}, {})
            elif ctype == "sam_plus_delta":
                x = _jitter_form_x(ctx, sam_x, rng, 0.04)
            elif ctype == "line_derived":
                x = _jitter_form_x(ctx, base_line_x, rng, 0.03)
            elif ctype == "gt_form_template":
                x = sam_x
            else:
                x = _jitter_form_x(ctx, sam_x, rng, 0.05)
            try_add(CandidateSpec(-1, ctype, x.tolist(), ctype, evidence.rho_K, evidence.rho_AB), force=True)
            have += 1

    # Trim to target, prefer dropping extra diversity/backup first
    while len(candidates) > target:
        drop_idx = next(
            (i for i in range(len(candidates) - 1, -1, -1) if candidates[i].candidate_type == "diversity_explore"),
            len(candidates) - 1,
        )
        if candidates[drop_idx].candidate_type == "sam4tun_baseline":
            break
        candidates.pop(drop_idx)
    for i, c in enumerate(candidates):
        c.candidate_id = i

    meta = {
        "n_candidates": len(candidates),
        "n_rejected": len(audit_rejects),
        "rejections": audit_rejects,
        "type_counts": {t: sum(1 for c in candidates if c.candidate_type == t) for t in TYPE_COUNTS},
    }
    return candidates, meta

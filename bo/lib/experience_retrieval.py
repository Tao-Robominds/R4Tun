"""Cross-ring experience retrieval from locked BO banks."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.ceiling_gate import REPO_ROOT
from lib.layout_bo import RingContext
from lib.line_reliability import LineEvidence
from lib.sam4tun_prior import Sam4TunPrior

DEFAULT_EXPERIENCE = REPO_ROOT / "methods" / "paper" / "experience"

DIVERSITY_SLOTS: dict[str, str] = {
    "dense_6": "1-5/r271",
    "medium_6": "1-1/r20",
    "sparse_6": "1-4/r206",
    "medium_7": "5-5/r258",
    "sparse_7": "4-6/r283",
    "partial_irregular_7": "4-1/r116",
}

BLOCK_WEIGHTS = {"depth": 0.25, "line": 0.35, "layout": 0.25, "form": 0.15}

DEPTH_FEATURES = [
    "ring_depth_finite_ratio",
    "ring_depth_row_nonempty_ratio",
    "ring_depth_coverage_ratio",
    "ring_depth_blank_band_ratio",
]
LINE_FEATURES = [
    "line_oblique_line_count",
    "line_oblique_line_strength_pos",
    "line_oblique_line_strength_neg",
    "line_oblique_angle_consistency",
    "line_horizontal_line_count",
    "line_horizontal_spacing_consistency",
]
LAYOUT_FEATURES = ["layout_k_center_norm", "layout_k_width_norm"]
FORM_FEATURES = ["form_segment_coverage_pct", "form_arc_width_entropy"]

QUERY_MAP = {
    "finite_ratio": "ring_depth_finite_ratio",
    "row_nonempty_ratio": "ring_depth_row_nonempty_ratio",
    "depth_coverage_ratio": "ring_depth_coverage_ratio",
    "blank_band_ratio": "ring_depth_blank_band_ratio",
    "oblique_line_count": "line_oblique_line_count",
    "oblique_strength_pos": "line_oblique_line_strength_pos",
    "oblique_strength_neg": "line_oblique_line_strength_neg",
    "oblique_angle_consistency": "line_oblique_angle_consistency",
    "horizontal_line_count": "line_horizontal_line_count",
    "horizontal_spacing_consistency": "line_horizontal_spacing_consistency",
    "k_center_norm": "layout_k_center_norm",
    "k_width_norm": "layout_k_width_norm",
    "segment_coverage_pct": "form_segment_coverage_pct",
    "arc_width_entropy": "form_arc_width_entropy",
}


@dataclass
class ExperienceQuery:
    ring_key: str
    segment_count: int
    diameter_m: float
    depth: dict[str, float]
    line: dict[str, float]
    layout: dict[str, float]
    form: dict[str, float]
    rho_K: float
    rho_AB: float

    def feature_vector(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.depth.items():
            out[QUERY_MAP.get(k, k)] = float(v)
        for k, v in self.line.items():
            out[QUERY_MAP.get(k, k)] = float(v)
        for k, v in self.layout.items():
            out[QUERY_MAP.get(k, k)] = float(v)
        for k, v in self.form.items():
            out[QUERY_MAP.get(k, k)] = float(v)
        return out


@dataclass
class RetrievalHit:
    hit_id: str
    ring_id: str
    distance: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalBundle:
    nearest_calib_ring: str
    coarse_score: float
    v4_hits: list[RetrievalHit]
    v5_form_ranges: dict[str, Any]
    v5_audit_exemplars: list[RetrievalHit]
    v3_failures: list[RetrievalHit]
    v3_rules: dict[str, Any]


def load_experience_tables(experience_root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = experience_root or DEFAULT_EXPERIENCE
    return {
        "bank": pd.read_csv(root / "experience_bank.csv"),
        "sam_templates": pd.read_csv(root / "proposal_templates_sam4tun.csv"),
        "gt_form": pd.read_csv(root / "proposal_good_form_gt_derived.csv"),
        "failures": pd.read_csv(root / "failure_memory_random.csv"),
        "failure_rules": pd.read_csv(root / "failure_memory_random_rules.csv"),
    }


def fit_norm_stats(bank: pd.DataFrame) -> dict[str, dict[str, float]]:
    all_feats = DEPTH_FEATURES + LINE_FEATURES + LAYOUT_FEATURES + FORM_FEATURES
    stats: dict[str, dict[str, float]] = {}
    for col in all_feats:
        if col not in bank.columns:
            continue
        s = pd.to_numeric(bank[col], errors="coerce").dropna()
        if s.empty:
            stats[col] = {"mu": 0.0, "sigma": 1.0}
        else:
            stats[col] = {"mu": float(s.mean()), "sigma": float(max(s.std(), 1e-6))}
    return stats


def save_norm_stats(stats: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def load_norm_stats(path: Path, bank: pd.DataFrame) -> dict[str, dict[str, float]]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    stats = fit_norm_stats(bank)
    save_norm_stats(stats, path)
    return stats


def _slot_centroids(bank: pd.DataFrame, descriptors: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    centroids: dict[str, dict[str, float]] = {}
    ring_level = bank.drop_duplicates("ring_id")
    for slot, ring_id in DIVERSITY_SLOTS.items():
        row = ring_level[ring_level["ring_id"] == ring_id]
        if row.empty:
            continue
        r = row.iloc[0]
        centroids[slot] = {
            "ring_id": ring_id,
            "segment_count": int(r["ring_segment_count"]),
            "diameter_m": float(r["ring_tunnel_diameter_m"]),
            "finite_ratio": float(r["ring_depth_finite_ratio"]),
            "density_score": 0.5,
            "k_span_score": 0.5,
        }
    if descriptors is not None:
        for slot, ring_id in DIVERSITY_SLOTS.items():
            drow = descriptors[descriptors["ring_key"] == ring_id]
            if not drow.empty and slot in centroids:
                dr = drow.iloc[0]
                centroids[slot]["density_score"] = float(dr.get("density_score", 0.5))
                centroids[slot]["k_span_score"] = float(dr.get("k_span_score", 0.5))
                centroids[slot]["diameter_m"] = float(dr.get("tunnel_diameter_m", centroids[slot]["diameter_m"]))
    return centroids


def nearest_calib_ring(
    q: ExperienceQuery,
    descriptors: pd.DataFrame | None,
    bank: pd.DataFrame,
) -> tuple[str, float]:
    centroids = _slot_centroids(bank, descriptors)
    desc_row = None
    if descriptors is not None:
        m = descriptors[descriptors["ring_key"] == q.ring_key]
        if not m.empty:
            desc_row = m.iloc[0]

    density = float(desc_row["density_score"]) if desc_row is not None else 0.5
    k_span = float(desc_row["k_span_score"]) if desc_row is not None else 0.5
    finite = float(q.depth.get("finite_ratio", q.depth.get("ring_depth_finite_ratio", 0.0)))

    best_ring = ""
    best_score = float("inf")
    for slot, c in centroids.items():
        if int(c["segment_count"]) != q.segment_count:
            continue
        score = (
            0.30 * abs(q.diameter_m - c["diameter_m"])
            + 0.25 * abs(density - c["density_score"])
            + 0.25 * abs(finite - c["finite_ratio"])
            + 0.20 * abs(k_span - c["k_span_score"])
        )
        if score < best_score:
            best_score = score
            best_ring = str(c["ring_id"])
    if not best_ring:
        pool = bank[bank["ring_segment_count"] == q.segment_count]["ring_id"].unique()
        best_ring = str(pool[0]) if len(pool) else "1-1/r20"
    return best_ring, float(best_score)


def _zdist(q_vec: dict[str, float], row: pd.Series, stats: dict[str, dict[str, float]]) -> float:
    total = 0.0
    for block, weight in BLOCK_WEIGHTS.items():
        feats = {
            "depth": DEPTH_FEATURES,
            "line": LINE_FEATURES,
            "layout": LAYOUT_FEATURES,
            "form": FORM_FEATURES,
        }[block]
        block_d = 0.0
        n = 0
        for col in feats:
            if col not in stats or col not in row.index:
                continue
            qv = q_vec.get(col)
            if qv is None:
                continue
            mu = stats[col]["mu"]
            sigma = stats[col]["sigma"]
            rv = float(row[col]) if pd.notna(row[col]) else mu
            block_d += ((float(qv) - rv) / sigma) ** 2
            n += 1
        if n:
            total += weight * (block_d / n)
    return float(np.sqrt(total))


def build_query(
    ctx: RingContext,
    prior: Sam4TunPrior,
    evidence: LineEvidence,
    *,
    descriptor_row: pd.Series | None = None,
) -> ExperienceQuery:
    H = ctx.H
    finite = evidence.finite_ratio
    row_ne = evidence.row_nonempty_ratio
    if descriptor_row is not None:
        finite = float(descriptor_row.get("finite_ratio", finite))
        row_ne = float(descriptor_row.get("row_nonempty_ratio", row_ne))

    ob_total = evidence.oblique_pos + evidence.oblique_neg
    return ExperienceQuery(
        ring_key=ctx.case_id,
        segment_count=ctx.segment_count,
        diameter_m=ctx.tunnel_diameter,
        depth={
            "finite_ratio": finite,
            "row_nonempty_ratio": row_ne,
            "depth_coverage_ratio": finite,
            "blank_band_ratio": float(descriptor_row["blank_band_ratio"]) if descriptor_row is not None else 0.0,
        },
        line={
            "oblique_line_count": float(ob_total),
            "oblique_strength_pos": float(evidence.oblique_pos),
            "oblique_strength_neg": float(evidence.oblique_neg),
            "oblique_angle_consistency": evidence.oblique_angle_consistency,
            "horizontal_line_count": float(evidence.horizontal),
            "horizontal_spacing_consistency": evidence.horizontal_spacing_consistency,
        },
        layout={
            "k_center_norm": float(prior.k_y / max(H, 1)),
            "k_width_norm": 0.07,
        },
        form={
            "segment_coverage_pct": 100.0,
            "arc_width_entropy": 1.7,
        },
        rho_K=evidence.rho_K,
        rho_AB=evidence.rho_AB,
    )


def _fix_query_layout(q: ExperienceQuery, prior: Sam4TunPrior, ctx: RingContext) -> None:
    from lib.layout_bo import offsets_to_arc_widths

    H = ctx.H
    q.layout["k_center_norm"] = float(prior.k_y / max(H, 1))
    widths = offsets_to_arc_widths(ctx.blocks, prior.offsets, H)
    q.layout["k_width_norm"] = float(widths[0] / max(H, 1))


def retrieve_experience(
    q: ExperienceQuery,
    tables: dict[str, pd.DataFrame],
    stats: dict[str, dict[str, float]],
    *,
    descriptors: pd.DataFrame | None = None,
    k: int = 8,
) -> RetrievalBundle:
    bank = tables["bank"]
    nearest, coarse = nearest_calib_ring(q, descriptors, bank)
    q_vec = q.feature_vector()

    v4_pool = bank[(bank["experience_pool"] == "v4") & (bank["trial_kind"] != "sam4tun_static")]
    v4_local = v4_pool[v4_pool["ring_id"] == nearest]
    if len(v4_local) < k:
        v4_local = v4_pool[v4_pool["ring_segment_count"] == q.segment_count]
    v4_local = v4_local.copy()
    v4_local["_dist"] = v4_local.apply(lambda r: _zdist(q_vec, r, stats), axis=1)
    v4_local = v4_local.sort_values(["_dist", "label_gt_miou"], ascending=[True, False])
    templates = tables["sam_templates"]
    v4_hits: list[RetrievalHit] = []
    for _, row in v4_local.head(k).iterrows():
        tpl = templates[templates["candidate_experience_id"] == row["experience_id"]]
        payload = tpl.iloc[0].to_dict() if not tpl.empty else row.to_dict()
        v4_hits.append(RetrievalHit(
            hit_id=str(row["experience_id"]),
            ring_id=str(row["ring_id"]),
            distance=float(row["_dist"]),
            payload=payload,
        ))

    gt_row = tables["gt_form"][tables["gt_form"]["ring_id"] == nearest]
    v5_ranges: dict[str, Any] = {}
    if not gt_row.empty:
        v5_ranges = json.loads(gt_row.iloc[0]["good_form_ranges_json"])

    v5_pool = bank[
        (bank["experience_pool"] == "v5")
        & (bank["ring_id"] == nearest)
        & (~bank["trial_kind"].astype(str).str.startswith("gt_layout"))
    ].copy()
    v5_pool["_dist"] = v5_pool.apply(lambda r: _zdist(q_vec, r, stats), axis=1)
    v5_pool = v5_pool.sort_values("_dist")
    v5_audit = [
        RetrievalHit(str(r["experience_id"]), str(r["ring_id"]), float(r["_dist"]), r.to_dict())
        for _, r in v5_pool.head(3).iterrows()
    ]

    fail_local = tables["failures"][tables["failures"]["ring_id"] == nearest].copy()
    if fail_local.empty:
        fail_local = tables["failures"].copy()
    fail_bank = bank.set_index("experience_id")
    v3_hits: list[RetrievalHit] = []
    for _, fr in fail_local.iterrows():
        eid = str(fr.get("experience_id", ""))
        dist = 0.5
        if eid in fail_bank.index:
            dist = _zdist(q_vec, fail_bank.loc[eid], stats)
        tags = json.loads(fr["failure_tags_json"]) if isinstance(fr["failure_tags_json"], str) else []
        boost = 0.0
        if q.rho_K < 0.5 and "bad_k_shift" in tags:
            boost -= 0.1
        if "good_form_wrong_anchor" in tags:
            boost -= 0.05
        v3_hits.append(RetrievalHit(
            hit_id=str(fr.get("failure_id", eid)),
            ring_id=str(fr["ring_id"]),
            distance=float(dist + boost),
            payload={**fr.to_dict(), "failure_tags": tags},
        ))
    v3_hits.sort(key=lambda h: h.distance)

    rules_row = tables["failure_rules"][tables["failure_rules"]["ring_id"] == nearest]
    v3_rules = rules_row.iloc[0].to_dict() if not rules_row.empty else {}

    return RetrievalBundle(
        nearest_calib_ring=nearest,
        coarse_score=coarse,
        v4_hits=v4_hits[:k],
        v5_form_ranges=v5_ranges,
        v5_audit_exemplars=v5_audit,
        v3_failures=v3_hits[:k],
        v3_rules=v3_rules,
    )


def build_query_from_parts(
    ctx: RingContext,
    prior: Sam4TunPrior,
    evidence: LineEvidence,
    descriptor_row: pd.Series | None,
) -> ExperienceQuery:
    q = build_query(ctx, prior, evidence, descriptor_row=descriptor_row)
    _fix_query_layout(q, prior, ctx)
    return q

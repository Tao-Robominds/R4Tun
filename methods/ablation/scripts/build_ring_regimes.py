"""Step 01 — Ring-regime discovery.

Build a multi-tunnel ring descriptor catalog from `data/subsets/*.txt`,
assign rule-based regime labels, select representative BO panels and
holdout, and write artifacts under
`data/subsets/workflow/{run_id}/01_ring_regime_discovery/`.

Run with the project venv only:

    ./venv/bin/python methods/ablation/scripts/build_ring_regimes.py \
        --subsets-dir data/subsets \
        --run regime_v1 \
        --families 4 5 \
        --regular-families 1 2 3 \
        --regular-ratio 0.20 \
        --panel-size 30 \
        --holdout-per-regime 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Constants
# =============================================================================

IRREGULAR_FAMILIES = {"4", "5"}
REGULAR_FAMILIES = {"1", "2", "3"}

# segment_id -> name; matches agents/3_segmentation/segmentation.py:57-58
IRREGULAR_SEG_NAMES = {0: "BG", 1: "K", 2: "B1", 3: "A1", 4: "A2", 5: "A3", 6: "A4", 7: "B2"}
REGULAR_SEG_NAMES = {0: "BG", 1: "S1", 2: "S2", 3: "S3", 4: "S4", 5: "S5", 6: "S6"}

# Domain rule: in a valid irregular ring, K must sit between B1 and B2 in the
# cyclic walking order (ignoring BG). Rings whose K has different non-BG
# neighbors are noisy / corrupt and are dropped from the catalog.
K_VALID_NEIGHBORS = frozenset({"B1", "B2"})

# Density bins from logs/4-1/balanced_30_rings_summary.md
DENSITY_BINS: List[Tuple[int, str]] = [
    (10_000, "sparse"),
    (50_000, "low"),
    (200_000, "medium"),
]
# >= 200_000 -> "dense"

ANGLE_GAP_FRAC_FULL = 0.02      # < 2% gap -> full
ANGLE_GAP_FRAC_PARTIAL = 0.10   # < 10% gap -> partial; else poor

# K span tier percentiles from balanced-ring docs (narrow / normal / wide)
K_SPAN_NARROW_PCTL = 20.0
K_SPAN_WIDE_PCTL = 80.0

# Canonical irregular cyclic block order (matches agents/2_detection/scripts/extract_intrinsics.py)
CANONICAL_K_ORDER = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]

ANGULAR_BIN_WIDTH_DEG = 1.0

# =============================================================================
# IO
# =============================================================================


def list_subset_files(subsets_dir: str, families: List[str]) -> List[Path]:
    out: List[Path] = []
    root = Path(subsets_dir)
    family_set = {str(f) for f in families}
    for path in sorted(root.glob("*.txt")):
        # Accept 2-part (e.g. "4-1.txt") and 3-part (e.g. "3-1-1.txt") tunnel ids.
        m = re.match(r"^(\d+)(?:-\d+)+\.txt$", path.name)
        if not m:
            continue
        if m.group(1) in family_set:
            out.append(path)
    return out


def load_subset_ring_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["x", "y", "z", "intensity", "segment", "ring"],
        engine="c",
        dtype={"x": "float32", "y": "float32", "z": "float32",
               "intensity": "float32", "segment": "int16", "ring": "int32"},
    )
    return df


# =============================================================================
# Geometry helpers
# =============================================================================


def ring_angle_deg(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Angle around the ring axis (Y-up cylinder) in degrees, [0, 360).

    Recenters on the per-ring (median x, median z) before computing
    `theta = -atan2(z - cz, x - cx)`. Centering matters: tunnel rings
    are not centered at the world origin, so a global atan2 inflates
    `angular_gap_frac` and scrambles the walking order. Validation
    against `data/rings/summary.json` (see write_descriptor_validation)
    confirms this convention places K between B1 and B2 in the cyclic
    walking order for canonical rings.
    """
    cx = float(np.median(x))
    cz = float(np.median(z))
    theta = -np.degrees(np.arctan2(z - cz, x - cx))
    return np.mod(theta, 360.0)


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    rad = np.radians(angles_deg)
    s = float(np.sin(rad).mean())
    c = float(np.cos(rad).mean())
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def angular_coverage_and_gap(angles_deg: np.ndarray, bin_width: float = ANGULAR_BIN_WIDTH_DEG) -> Tuple[float, float]:
    """Return (angular_coverage_deg, angular_gap_frac).

    Coverage is the number of occupied 1° bins. Gap is the largest empty
    contiguous run of bins divided by 360.
    """
    n_bins = int(round(360.0 / bin_width))
    if angles_deg.size == 0:
        return 0.0, 1.0
    occupied = np.zeros(n_bins, dtype=bool)
    idx = (np.floor(angles_deg / bin_width).astype(int)) % n_bins
    occupied[idx] = True
    coverage_deg = float(occupied.sum() * bin_width)
    if occupied.all():
        return 360.0, 0.0
    if not occupied.any():
        return 0.0, 1.0
    # largest run of empty bins (cyclic)
    empty = (~occupied).astype(np.int8)
    doubled = np.concatenate([empty, empty])
    longest = 0
    cur = 0
    for v in doubled:
        if v:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    longest = min(longest, n_bins)
    gap_frac = float(longest * bin_width / 360.0)
    return coverage_deg, gap_frac


def circular_span_deg(angles_deg: np.ndarray, bin_width: float = ANGULAR_BIN_WIDTH_DEG) -> float:
    """Span of a (possibly wrap-around) cluster as occupied 1° bins.

    Robust to bimodal noise: returns the count of bins covered,
    treating the cluster as the smallest cyclic arc that contains
    all occupied bins. For tightly packed segments this matches the
    intuitive "extent" used in the balanced-30 docs.
    """
    n_bins = int(round(360.0 / bin_width))
    if angles_deg.size == 0:
        return 0.0
    occupied = np.zeros(n_bins, dtype=bool)
    idx = (np.floor(angles_deg / bin_width).astype(int)) % n_bins
    occupied[idx] = True
    if occupied.all():
        return 360.0
    if not occupied.any():
        return 0.0
    # find largest empty cyclic run; span = 360 - empty_run
    empty = (~occupied).astype(np.int8)
    doubled = np.concatenate([empty, empty])
    longest = 0
    cur = 0
    for v in doubled:
        if v:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    longest = min(longest, n_bins)
    return float(360.0 - longest * bin_width)


def density_tier(n_points: int) -> str:
    for thr, label in DENSITY_BINS:
        if n_points < thr:
            return label
    return "dense"


def coverage_tier_label(angular_gap_frac: float) -> str:
    if angular_gap_frac < ANGLE_GAP_FRAC_FULL:
        return "full"
    if angular_gap_frac < ANGLE_GAP_FRAC_PARTIAL:
        return "partial"
    return "poor"


def k_quadrant(angle_deg: Optional[float]) -> str:
    if angle_deg is None or (isinstance(angle_deg, float) and math.isnan(angle_deg)):
        return "na"
    a = angle_deg % 360.0
    return f"q{int(a // 90.0)}"


# =============================================================================
# Per-ring descriptor
# =============================================================================


def _walking_order_blocks(angles_deg: np.ndarray, segs: np.ndarray, name_map: Dict[int, str], drop_bg: bool = False) -> List[str]:
    pairs: List[Tuple[float, str]] = []
    for s in np.unique(segs):
        name = name_map.get(int(s), f"S{int(s)}")
        if drop_bg and name == "BG":
            continue
        mask = segs == s
        if mask.sum() == 0:
            continue
        pairs.append((circular_mean_deg(angles_deg[mask]), name))
    pairs.sort(key=lambda t: t[0])
    return [p[1] for p in pairs]


def _k_neighbors_no_bg(walking_no_bg: List[str]) -> Optional[Tuple[str, str]]:
    if not walking_no_bg or "K" not in walking_no_bg:
        return None
    n = len(walking_no_bg)
    i = walking_no_bg.index("K")
    return walking_no_bg[(i - 1) % n], walking_no_bg[(i + 1) % n]


def _k_neighbors_are_b1_b2(walking_no_bg: List[str]) -> bool:
    pair = _k_neighbors_no_bg(walking_no_bg)
    if pair is None:
        return False
    return set(pair) == {"B1", "B2"}


def _is_canonical_rotation(order: List[str], canonical: List[str]) -> str:
    """Return 'canonical', 'reversed_canonical', or 'noncanonical'."""
    n = len(canonical)
    if len(order) != n:
        return "noncanonical"
    for rot in range(n):
        rotated = canonical[rot:] + canonical[:rot]
        if order == rotated:
            return "canonical"
    rev = list(reversed(canonical))
    for rot in range(n):
        rotated = rev[rot:] + rev[:rot]
        if order == rotated:
            return "reversed_canonical"
    return "noncanonical"


def _segment_balance_cv(segs: np.ndarray, name_map: Dict[int, str]) -> float:
    counts = []
    for s in np.unique(segs):
        name = name_map.get(int(s), str(int(s)))
        if name == "BG":
            continue
        counts.append(int((segs == s).sum()))
    if not counts:
        return float("nan")
    arr = np.asarray(counts, dtype=float)
    if arr.mean() == 0:
        return float("nan")
    return float(arr.std(ddof=0) / arr.mean())


def _ring_complexity_score(
    pattern_type: str,
    has_k: bool,
    coverage_tier: str,
    segment_balance_cv: float,
    k_span_tier: str,
    radius_iqr: float,
    radius_median: float,
) -> float:
    score = 0.0
    if pattern_type == "noncanonical":
        score += 1.0
    elif pattern_type == "reversed_canonical":
        score += 0.3
    if not has_k:
        score += 1.0
    if coverage_tier == "partial":
        score += 0.5
    elif coverage_tier == "poor":
        score += 1.0
    if not math.isnan(segment_balance_cv) and segment_balance_cv > 0.6:
        score += min((segment_balance_cv - 0.6) / 0.4, 1.0)
    if k_span_tier in ("narrow", "wide"):
        score += 0.3
    if radius_median and not math.isnan(radius_median) and radius_median > 0:
        rel_iqr = radius_iqr / radius_median
        if rel_iqr > 0.05:
            score += min((rel_iqr - 0.05) / 0.10, 1.0)
    return float(round(score, 3))


def compute_ring_descriptors(
    df: pd.DataFrame,
    tunnel_id: str,
    source_path: str,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Return (descriptors_df, dropped_rings).

    Irregular rings whose K is not surrounded by B1 and B2 in the cyclic
    walking order are excluded from descriptors and recorded in
    `dropped_rings` for the audit trail.
    """
    family = tunnel_id.split("-", 1)[0]
    is_irregular = family in IRREGULAR_FAMILIES
    name_map = IRREGULAR_SEG_NAMES if is_irregular else REGULAR_SEG_NAMES

    rows: List[Dict] = []
    dropped: List[Dict] = []
    grouped = df.groupby("ring", sort=True)
    for ring_id, gdf in grouped:
        x = gdf["x"].to_numpy()
        y = gdf["y"].to_numpy()
        z = gdf["z"].to_numpy()
        seg = gdf["segment"].to_numpy()

        n_points = int(len(gdf))
        theta = ring_angle_deg(x, z)

        coverage_deg, gap_frac = angular_coverage_and_gap(theta)
        cov_tier = coverage_tier_label(gap_frac)

        # radius around ring axis (Y-up assumption matches angle convention)
        r = np.sqrt(x.astype(np.float64) ** 2 + z.astype(np.float64) ** 2)
        radius_median = float(np.median(r)) if r.size else float("nan")
        radius_std = float(np.std(r)) if r.size else float("nan")
        radius_iqr = float(np.percentile(r, 75) - np.percentile(r, 25)) if r.size else float("nan")
        radius_range = float(r.max() - r.min()) if r.size else float("nan")

        seg_unique = [int(s) for s in np.unique(seg).tolist()]
        non_bg = [s for s in seg_unique if name_map.get(s) != "BG"]
        segment_count_non_bg = len(non_bg)

        walking_full = _walking_order_blocks(theta, seg, name_map, drop_bg=False)
        walking_no_bg = _walking_order_blocks(theta, seg, name_map, drop_bg=True)

        if is_irregular:
            has_k = 1 in seg_unique
            if not has_k:
                dropped.append({
                    "tunnel_id": tunnel_id,
                    "ring_id": int(ring_id),
                    "n_points": n_points,
                    "walking_order_no_bg": "-".join(walking_no_bg),
                    "k_neighbors_no_bg": "",
                    "reason": "no_k",
                })
                continue
            if not _k_neighbors_are_b1_b2(walking_no_bg):
                pair = _k_neighbors_no_bg(walking_no_bg)
                dropped.append({
                    "tunnel_id": tunnel_id,
                    "ring_id": int(ring_id),
                    "n_points": n_points,
                    "walking_order_no_bg": "-".join(walking_no_bg),
                    "k_neighbors_no_bg": "-".join(pair) if pair else "",
                    "reason": "k_neighbors_not_B1_B2",
                })
                continue
            k_mask = seg == 1
            k_angle = circular_mean_deg(theta[k_mask])
            k_span = circular_span_deg(theta[k_mask])
            pattern_type = _is_canonical_rotation(walking_no_bg, CANONICAL_K_ORDER)
            pair = _k_neighbors_no_bg(walking_no_bg)
            k_neighbors = f"{pair[0]}-{pair[1]}" if pair else None
        else:
            # Regular tunnels do not have a K block.
            has_k = False
            k_angle = float("nan")
            k_span = float("nan")
            pattern_type = "no_k"
            k_neighbors = None

        rows.append({
            "tunnel_id": tunnel_id,
            "family": family,
            "ring_id": int(ring_id),
            "source_path": source_path,
            "n_points": n_points,
            "density_tier": density_tier(n_points),
            "angular_coverage_deg": round(coverage_deg, 2),
            "angular_gap_frac": round(gap_frac, 4),
            "coverage_tier": cov_tier,
            "radius_median": round(radius_median, 4) if not math.isnan(radius_median) else None,
            "radius_std": round(radius_std, 4) if not math.isnan(radius_std) else None,
            "radius_iqr": round(radius_iqr, 4) if not math.isnan(radius_iqr) else None,
            "radius_range": round(radius_range, 4) if not math.isnan(radius_range) else None,
            "segment_count_non_bg": segment_count_non_bg,
            "has_k": bool(has_k),
            "walking_order": "-".join(walking_full) if walking_full else "",
            "walking_order_no_bg": "-".join(walking_no_bg) if walking_no_bg else "",
            "pattern_type": pattern_type,
            "k_angle_deg": round(k_angle, 2) if not math.isnan(k_angle) else None,
            "k_quadrant": k_quadrant(k_angle if not math.isnan(k_angle) else None),
            "k_span_deg": round(k_span, 2) if not math.isnan(k_span) else None,
            "k_neighbors": k_neighbors,
            "segment_balance_cv": (
                round(_segment_balance_cv(seg, name_map), 4)
                if not math.isnan(_segment_balance_cv(seg, name_map))
                else None
            ),
        })

    desc = pd.DataFrame(rows)
    return desc, dropped


# =============================================================================
# Regime assignment
# =============================================================================


def _assign_k_span_tier(desc: pd.DataFrame) -> pd.Series:
    out = pd.Series(["na"] * len(desc), index=desc.index, dtype=object)
    irreg = desc["family"].isin(IRREGULAR_FAMILIES) & desc["has_k"] & desc["k_span_deg"].notna()
    if irreg.sum() == 0:
        return out
    spans = desc.loc[irreg, "k_span_deg"].astype(float).to_numpy()
    p_low = float(np.percentile(spans, K_SPAN_NARROW_PCTL))
    p_high = float(np.percentile(spans, K_SPAN_WIDE_PCTL))
    for idx, span in zip(desc.loc[irreg].index, spans):
        if span < p_low:
            out.loc[idx] = "narrow"
        elif span >= p_high:
            out.loc[idx] = "wide"
        else:
            out.loc[idx] = "normal"
    return out


def assign_regime_labels(desc: pd.DataFrame) -> pd.DataFrame:
    out = desc.copy()
    out["k_span_tier"] = _assign_k_span_tier(out)

    pattern_type = out["pattern_type"].fillna("no_k")
    out["regime_label"] = (
        out["density_tier"].astype(str)
        + "_" + out["coverage_tier"].astype(str)
        + "_" + out["k_span_tier"].astype(str)
        + "_" + pattern_type.astype(str)
    )

    out["domain_role"] = out["family"].apply(
        lambda f: "target_irregular" if str(f) in IRREGULAR_FAMILIES else "sanity_regular"
    )

    complexity = []
    for _, row in out.iterrows():
        complexity.append(_ring_complexity_score(
            pattern_type=str(row.get("pattern_type", "no_k") or "no_k"),
            has_k=bool(row.get("has_k", False)),
            coverage_tier=str(row.get("coverage_tier", "poor")),
            segment_balance_cv=float(row.get("segment_balance_cv") or float("nan")),
            k_span_tier=str(row.get("k_span_tier", "na")),
            radius_iqr=float(row.get("radius_iqr") or 0.0),
            radius_median=float(row.get("radius_median") or 0.0),
        ))
    out["ring_complexity_score"] = complexity
    return out


# =============================================================================
# Panel selection
# =============================================================================


def _stratified_pick(rows: pd.DataFrame, n: int, axes: List[str], rng: random.Random) -> List[int]:
    """Greedy round-robin pick across (axis-value) buckets to spread coverage."""
    if rows.empty or n <= 0:
        return []
    n = min(n, len(rows))
    remaining = rows.copy()
    chosen: List[int] = []

    def diversity_key(row: pd.Series) -> Tuple:
        return tuple(row.get(a, "na") for a in axes)

    bucket_counts: Counter = Counter()

    while len(chosen) < n and not remaining.empty:
        scored = []
        for idx, row in remaining.iterrows():
            key = diversity_key(row)
            scored.append((bucket_counts[key], rng.random(), idx, key))
        scored.sort()
        _, _, idx, key = scored[0]
        chosen.append(int(idx))
        bucket_counts[key] += 1
        remaining = remaining.drop(idx)
    return chosen


def select_representative_panel(
    regimes: pd.DataFrame,
    panel_size: int,
    irregular_count: int,
    family4_count: int,
    family5_count: int,
    holdout_per_regime: int = 0,
    seed: int = 7,
) -> Dict:
    rng = random.Random(seed)

    irregular = regimes[regimes["domain_role"] == "target_irregular"].copy()
    regular = regimes[regimes["domain_role"] == "sanity_regular"].copy()

    family4 = irregular[irregular["family"] == "4"].copy()
    family5 = irregular[irregular["family"] == "5"].copy()

    # Reserve holdout first so it never overlaps the panel.
    holdout_ids: List[int] = []
    if holdout_per_regime > 0:
        for label, grp in irregular.groupby("regime_label"):
            picks = _stratified_pick(grp, holdout_per_regime, ["k_quadrant", "family"], rng)
            holdout_ids.extend(picks)
    holdout_set = set(holdout_ids)

    irregular = irregular.drop(index=[i for i in holdout_ids if i in irregular.index])
    family4 = family4.drop(index=[i for i in holdout_ids if i in family4.index])
    family5 = family5.drop(index=[i for i in holdout_ids if i in family5.index])

    # Prefer has_k=True for primary irregular calibration rings.
    fam4_with_k = family4[family4["has_k"]]
    fam5_with_k = family5[family5["has_k"]]

    diversity_axes = ["density_tier", "coverage_tier", "k_span_tier", "k_quadrant", "pattern_type"]

    fam4_picks = _stratified_pick(fam4_with_k if len(fam4_with_k) >= family4_count else family4,
                                   family4_count, diversity_axes, rng)
    fam5_picks = _stratified_pick(fam5_with_k if len(fam5_with_k) >= family5_count else family5,
                                   family5_count, diversity_axes, rng)
    reg_picks = _stratified_pick(regular, panel_size - irregular_count,
                                  ["density_tier", "coverage_tier", "family"], rng)

    irregular_panel_ids = list(fam4_picks) + list(fam5_picks)
    regular_panel_ids = list(reg_picks)

    panel_rows: List[Dict] = []
    for idx in irregular_panel_ids:
        r = regimes.loc[idx]
        panel_rows.append({
            "tunnel_id": r["tunnel_id"],
            "ring_id": int(r["ring_id"]),
            "family": r["family"],
            "domain_role": "target_irregular",
            "regime_label": r["regime_label"],
            "density_tier": r["density_tier"],
            "coverage_tier": r["coverage_tier"],
            "k_quadrant": r["k_quadrant"],
            "k_span_tier": r["k_span_tier"],
            "pattern_type": r["pattern_type"],
        })
    for idx in regular_panel_ids:
        r = regimes.loc[idx]
        panel_rows.append({
            "tunnel_id": r["tunnel_id"],
            "ring_id": int(r["ring_id"]),
            "family": r["family"],
            "domain_role": "sanity_regular",
            "regime_label": r["regime_label"],
            "density_tier": r["density_tier"],
            "coverage_tier": r["coverage_tier"],
            "k_quadrant": r["k_quadrant"],
            "k_span_tier": r["k_span_tier"],
            "pattern_type": r["pattern_type"],
        })

    holdout_rows: List[Dict] = []
    for idx in holdout_ids:
        r = regimes.loc[idx]
        holdout_rows.append({
            "tunnel_id": r["tunnel_id"],
            "ring_id": int(r["ring_id"]),
            "family": r["family"],
            "domain_role": r["domain_role"],
            "regime_label": r["regime_label"],
            "density_tier": r["density_tier"],
            "coverage_tier": r["coverage_tier"],
            "k_quadrant": r["k_quadrant"],
            "k_span_tier": r["k_span_tier"],
            "pattern_type": r["pattern_type"],
        })

    return {
        "panel_size": panel_size,
        "irregular_count": irregular_count,
        "regular_count": panel_size - irregular_count,
        "family4_count": family4_count,
        "family5_count": family5_count,
        "panel": panel_rows,
        "holdout": holdout_rows,
    }


# =============================================================================
# Validation against data/rings/summary.json
# =============================================================================


def validate_against_summary(regimes: pd.DataFrame, summary_path: str) -> Dict:
    p = Path(summary_path)
    if not p.exists():
        return {"available": False, "reason": f"{summary_path} not found"}
    with p.open() as f:
        summary = json.load(f)

    by_key: Dict[Tuple[str, int], Dict] = {}
    for s in summary.get("samples", []):
        by_key[(s["file"], int(s["ring_id"]))] = s

    if not by_key:
        return {"available": False, "reason": "summary samples empty"}

    matched: List[Dict] = []
    quadrant_match = 0
    quadrant_total = 0
    walk_match_any_rotation_or_reverse = 0
    walk_total = 0
    angle_deltas_deg: List[float] = []
    span_deltas_deg: List[float] = []
    gap_deltas: List[float] = []
    n_points_deltas: List[int] = []

    for _, row in regimes.iterrows():
        key = (row["tunnel_id"], int(row["ring_id"]))
        if key not in by_key:
            continue
        ref = by_key[key]
        n_diff = int(row["n_points"]) - int(ref.get("n_points") or 0)
        n_points_deltas.append(n_diff)

        gap_diff = float(row["angular_gap_frac"]) - float(ref.get("angular_gap_frac") or 0.0)
        gap_deltas.append(gap_diff)

        if ref.get("k_angle_deg") is not None and row["k_angle_deg"] is not None:
            quadrant_total += 1
            ref_q = int((float(ref["k_angle_deg"]) % 360.0) // 90)
            our_q = int((float(row["k_angle_deg"]) % 360.0) // 90)
            if ref_q == our_q:
                quadrant_match += 1
            d = (float(row["k_angle_deg"]) - float(ref["k_angle_deg"])) % 360.0
            if d > 180.0:
                d -= 360.0
            angle_deltas_deg.append(float(d))

        if ref.get("k_span_deg") is not None and row["k_span_deg"] is not None:
            span_deltas_deg.append(float(row["k_span_deg"]) - float(ref["k_span_deg"]))

        if ref.get("walking_order"):
            walk_total += 1
            target = ref["walking_order"]
            ours = row["walking_order"]
            ours_blocks = ours.split("-") if ours else []
            target_blocks = target.split("-") if target else []
            matches = False
            if len(ours_blocks) == len(target_blocks) and target_blocks:
                n = len(target_blocks)
                target_rev = list(reversed(target_blocks))
                for rot in range(n):
                    rotated = target_blocks[rot:] + target_blocks[:rot]
                    if ours_blocks == rotated:
                        matches = True
                        break
                if not matches:
                    for rot in range(n):
                        rotated = target_rev[rot:] + target_rev[:rot]
                        if ours_blocks == rotated:
                            matches = True
                            break
            if matches:
                walk_match_any_rotation_or_reverse += 1

        matched.append({
            "tunnel_id": row["tunnel_id"],
            "ring_id": int(row["ring_id"]),
            "n_points_ours": int(row["n_points"]),
            "n_points_ref": int(ref.get("n_points") or 0),
            "angular_gap_frac_ours": float(row["angular_gap_frac"]),
            "angular_gap_frac_ref": float(ref.get("angular_gap_frac") or 0.0),
            "k_angle_ours": row["k_angle_deg"],
            "k_angle_ref": ref.get("k_angle_deg"),
            "k_span_ours": row["k_span_deg"],
            "k_span_ref": ref.get("k_span_deg"),
            "walking_order_ours": row["walking_order"],
            "walking_order_ref": ref.get("walking_order"),
        })

    return {
        "available": True,
        "matched_rings": len(matched),
        "quadrant_match_rate": (quadrant_match / quadrant_total) if quadrant_total else None,
        "walking_order_match_rate": (walk_match_any_rotation_or_reverse / walk_total) if walk_total else None,
        "n_points_abs_max": int(max(map(abs, n_points_deltas))) if n_points_deltas else 0,
        "angular_gap_frac_mean_abs": (float(np.mean([abs(v) for v in gap_deltas])) if gap_deltas else None),
        "k_angle_mean_abs_deg": (float(np.mean([abs(v) for v in angle_deltas_deg])) if angle_deltas_deg else None),
        "k_span_mean_abs_deg": (float(np.mean([abs(v) for v in span_deltas_deg])) if span_deltas_deg else None),
        "samples_summary": matched[:30],
    }


# =============================================================================
# Reporting helpers
# =============================================================================


def write_distribution_csv(regimes: pd.DataFrame, path: Path) -> None:
    cols = ["family", "density_tier", "coverage_tier", "k_span_tier", "k_quadrant", "pattern_type"]
    counts = (
        regimes.groupby(cols)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    counts.to_csv(path, index=False)


def write_summary_md(
    regimes: pd.DataFrame,
    panel_20: Dict,
    panel_30: Dict,
    validation: Dict,
    out_path: Path,
    angle_convention_note: str,
    dropped_count: int = 0,
) -> None:
    lines: List[str] = []
    lines.append("# Step 01 — Ring Regime Discovery")
    lines.append("")
    lines.append("## Pool counts")
    lines.append("")
    lines.append(f"- Total rings cataloged: {len(regimes)}")
    if dropped_count:
        lines.append(
            f"- Irregular rings dropped (no K, or K not surrounded by B1/B2): {dropped_count}"
        )
    fam_counts = regimes.groupby(["family", "domain_role"]).size().to_dict()
    for (fam, role), n in sorted(fam_counts.items()):
        lines.append(f"  - family {fam} ({role}): {n}")
    lines.append("")

    irreg = regimes[regimes["domain_role"] == "target_irregular"]
    reg = regimes[regimes["domain_role"] == "sanity_regular"]
    lines.append(f"- Irregular rings (target): {len(irreg)}")
    lines.append(f"- Regular rings (sanity): {len(reg)}")
    lines.append("")

    lines.append("## Distribution (irregular only)")
    lines.append("")
    for col in ["density_tier", "coverage_tier", "k_quadrant", "k_span_tier", "pattern_type", "regime_label"]:
        c = irreg[col].value_counts().sort_index()
        items = ", ".join(f"{k}={v}" for k, v in c.items())
        lines.append(f"- **{col}**: {items}")
    lines.append("")

    def panel_section(title: str, panel: Dict) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- Size: {panel['panel_size']} (irregular={panel['irregular_count']}, regular={panel['regular_count']})")
        lines.append(f"- Family 4 picks: {panel['family4_count']}, Family 5 picks: {panel['family5_count']}")
        lines.append("")
        lines.append("| tunnel_id | ring_id | role | regime |")
        lines.append("|-----------|---------|------|--------|")
        for r in panel["panel"]:
            lines.append(f"| {r['tunnel_id']} | {r['ring_id']} | {r['domain_role']} | {r['regime_label']} |")
        lines.append("")
        if panel["holdout"]:
            lines.append("### Holdout")
            lines.append("")
            lines.append("| tunnel_id | ring_id | role | regime |")
            lines.append("|-----------|---------|------|--------|")
            for r in panel["holdout"]:
                lines.append(f"| {r['tunnel_id']} | {r['ring_id']} | {r['domain_role']} | {r['regime_label']} |")
            lines.append("")

    panel_section("panel_20", panel_20)
    panel_section("panel_30", panel_30)

    lines.append("## Validation against data/rings/summary.json")
    lines.append("")
    if validation.get("available"):
        lines.append(f"- Matched rings: {validation['matched_rings']}")
        lines.append(f"- K quadrant match rate: {validation['quadrant_match_rate']}")
        lines.append(f"- Walking-order match rate (any rotation or reversal): {validation['walking_order_match_rate']}")
        lines.append(f"- Mean |Δ angular_gap_frac|: {validation['angular_gap_frac_mean_abs']}")
        lines.append(f"- Mean |Δ k_angle_deg|: {validation['k_angle_mean_abs_deg']}")
        lines.append(f"- Mean |Δ k_span_deg|: {validation['k_span_mean_abs_deg']}")
        lines.append(f"- Max |Δ n_points|: {validation['n_points_abs_max']}")
    else:
        lines.append(f"- Validation unavailable: {validation.get('reason')}")
    lines.append("")
    lines.append("## Angle convention note")
    lines.append("")
    lines.append(angle_convention_note)
    lines.append("")
    out_path.write_text("\n".join(lines))


def write_descriptor_validation(validation: Dict, out_path: Path, angle_convention_note: str) -> None:
    lines: List[str] = []
    lines.append("# Descriptor validation against data/rings/summary.json")
    lines.append("")
    lines.append(angle_convention_note)
    lines.append("")
    if not validation.get("available"):
        lines.append(f"Not available: {validation.get('reason')}")
        out_path.write_text("\n".join(lines))
        return

    lines.append(f"- Matched rings: {validation['matched_rings']}")
    lines.append(f"- K quadrant match rate: {validation['quadrant_match_rate']}")
    lines.append(f"- Walking-order match rate (rotation or reversal): {validation['walking_order_match_rate']}")
    lines.append(f"- Mean |Δ angular_gap_frac|: {validation['angular_gap_frac_mean_abs']}")
    lines.append(f"- Mean |Δ k_angle_deg|: {validation['k_angle_mean_abs_deg']}")
    lines.append(f"- Mean |Δ k_span_deg|: {validation['k_span_mean_abs_deg']}")
    lines.append(f"- Max |Δ n_points|: {validation['n_points_abs_max']}")
    lines.append("")
    lines.append("## Sample rows")
    lines.append("")
    lines.append("| tunnel | ring | n_pts (ours/ref) | gap (ours/ref) | k_angle (ours/ref) | k_span (ours/ref) | walk_ours | walk_ref |")
    lines.append("|--------|------|------------------|----------------|--------------------|-------------------|-----------|----------|")
    for s in validation.get("samples_summary", []):
        lines.append(
            f"| {s['tunnel_id']} | {s['ring_id']} | {s['n_points_ours']}/{s['n_points_ref']} | "
            f"{s['angular_gap_frac_ours']:.3f}/{s['angular_gap_frac_ref']:.3f} | "
            f"{s['k_angle_ours']}/{s['k_angle_ref']} | {s['k_span_ours']}/{s['k_span_ref']} | "
            f"{s['walking_order_ours']} | {s['walking_order_ref']} |"
        )
    out_path.write_text("\n".join(lines))


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 01 ring regime discovery")
    p.add_argument("--subsets-dir", default="data/subsets")
    p.add_argument("--run", default="regime_v1")
    p.add_argument("--families", nargs="+", default=["4", "5"])
    p.add_argument("--regular-families", nargs="+", default=["1", "2", "3"])
    p.add_argument("--regular-ratio", type=float, default=0.20)
    p.add_argument("--panel-size", type=int, default=30)
    p.add_argument("--holdout-per-regime", type=int, default=1)
    p.add_argument("--summary-json", default="data/rings/summary.json")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-root", default="data/subsets/workflow")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out_root) / args.run / "01_ring_regime_discovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    families_all = sorted(set(args.families) | set(args.regular_families))
    paths = list_subset_files(args.subsets_dir, families_all)
    print(f"[step 01] {len(paths)} subset files to load", file=sys.stderr)

    descriptors_all: List[pd.DataFrame] = []
    dropped_all: List[Dict] = []
    for path in paths:
        tunnel_id = path.stem
        try:
            df = load_subset_ring_points(path)
        except Exception as e:
            print(f"[step 01] failed to load {path}: {e}", file=sys.stderr)
            continue
        desc, dropped = compute_ring_descriptors(df, tunnel_id, str(path))
        dropped_all.extend(dropped)
        if desc.empty:
            print(f"[step 01] no rings kept in {path} (dropped={len(dropped)})", file=sys.stderr)
            continue
        descriptors_all.append(desc)
        print(f"[step 01] {tunnel_id}: kept={len(desc)} dropped={len(dropped)}", file=sys.stderr)

    if not descriptors_all:
        print("[step 01] ERROR: no descriptors produced", file=sys.stderr)
        return 1

    descriptors = pd.concat([d for d in descriptors_all if not d.empty], ignore_index=True)
    descriptors.to_csv(out_dir / "ring_descriptors.csv", index=False)

    if dropped_all:
        pd.DataFrame(dropped_all).to_csv(out_dir / "dropped_rings.csv", index=False)
    drop_reason_counts = Counter(d["reason"] for d in dropped_all)
    print(
        f"[step 01] dropped {len(dropped_all)} rings "
        f"({dict(drop_reason_counts)})",
        file=sys.stderr,
    )

    regimes = assign_regime_labels(descriptors)
    regimes.to_csv(out_dir / "ring_regimes.csv", index=False)

    write_distribution_csv(regimes, out_dir / "regime_distribution.csv")

    panel20_irregular = 16
    panel20_fam4 = 9
    panel20_fam5 = 7
    panel_20 = select_representative_panel(
        regimes,
        panel_size=20,
        irregular_count=panel20_irregular,
        family4_count=panel20_fam4,
        family5_count=panel20_fam5,
        holdout_per_regime=0,
        seed=args.seed,
    )

    panel30_irregular = 24
    panel30_fam4 = 14
    panel30_fam5 = 10
    panel_30 = select_representative_panel(
        regimes,
        panel_size=args.panel_size,
        irregular_count=panel30_irregular,
        family4_count=panel30_fam4,
        family5_count=panel30_fam5,
        holdout_per_regime=args.holdout_per_regime,
        seed=args.seed + 1,
    )

    panel_payload = {
        "run": args.run,
        "subsets_dir": args.subsets_dir,
        "families_target": list(args.families),
        "families_sanity": list(args.regular_families),
        "regular_ratio": args.regular_ratio,
        "seed": args.seed,
        "panel_20": panel_20,
        "panel_30": panel_30,
    }
    (out_dir / "regime_sampling_panel.json").write_text(
        json.dumps(panel_payload, indent=2, sort_keys=False)
    )

    validation = validate_against_summary(regimes, args.summary_json)

    angle_convention_note = (
        "Angles are computed per ring as `theta = -atan2(z - cz, x - cx) mod 360` "
        "where `(cx, cz)` is the per-ring median (centroid). The segment-id -> "
        "block-name mapping follows `agents/3_segmentation/segmentation.py`: "
        "{0:BG, 1:K, 2:B1, 3:A1, 4:A2, 5:A3, 6:A4, 7:B2}. Irregular rings are "
        "dropped from the catalog (see `dropped_rings.csv`) when (a) K is "
        "missing or (b) K is not surrounded by B1 and B2 in the cyclic walking "
        "order; both cases indicate noisy / non-meaningful rings. The "
        "`walking_order` field in `data/rings/summary.json` uses a different "
        "(image-space) convention, so per-ring walking orders are not expected "
        "to match it literally; only K quadrant, n_points, and "
        "angular_gap_frac are directly comparable."
    )

    write_descriptor_validation(
        validation, out_dir / "descriptor_validation_against_data_rings.md",
        angle_convention_note,
    )

    write_summary_md(
        regimes, panel_20, panel_30, validation,
        out_dir / "regime_summary.md", angle_convention_note,
        dropped_count=len(dropped_all),
    )

    print(f"[step 01] wrote artifacts to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

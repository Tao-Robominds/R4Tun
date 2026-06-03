"""Forced perturbation candidates for experience-mode BO (Step 3 ablation).

Perturbations anchor on best warm-start vector or geometric_0 — never GT layout.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lib.layout_bo import RingContext

PRIOR_K_SMALL_7 = np.array([0.07, 0.15, 0.15, 0.15, 0.15, 0.15, 0.18])
PRIOR_K_SMALL_6 = np.array([0.07, 0.18, 0.18, 0.18, 0.18, 0.21])


def _clip_x(ctx: RingContext, x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float).ravel(), 0.0, 1.0).copy()
    stream = getattr(ctx, "experience_stream", "full")
    if stream == "layout" and x.size > 0:
        x[0] = 0.0
    elif stream not in ("k", "layout") and x.size > 1:
        x[1] = 0.0
    return x


def _layout_start(ctx: RingContext) -> int:
    return 1 + ctx.segment_count


def _k_small_widths(ctx: RingContext) -> np.ndarray:
    n = ctx.segment_count
    k_small = PRIOR_K_SMALL_7 if n == 7 else PRIOR_K_SMALL_6
    if len(k_small) != n:
        k_small = np.concatenate([k_small[: n - 1], [k_small[-1]]])[:n]
        k_small = k_small / k_small.sum()
    return k_small


def _r_surface_index(ctx: RingContext) -> int:
    return ctx.search_dim - 1


def forced_perturbation_candidates(
    ctx: RingContext,
    base_x: np.ndarray,
    *,
    rng: np.random.Generator,
    n_target: int,
) -> list[tuple[np.ndarray, str]]:
    """Generate controlled near-negative layouts from a reference vector (GT or best-so-far)."""
    from lib.layout_bo import _coerce_search_x, widths_to_offset_fracs

    if getattr(ctx, "experience_stream", "full") == "layout":
        return _forced_perturbation_layout_stream(ctx, base_x, rng=rng, n_target=n_target)
    if getattr(ctx, "experience_stream", "full") == "k":
        return _forced_perturbation_k_stream(ctx, base_x, rng=rng, n_target=n_target)

    base = _coerce_search_x(ctx, base_x)
    ls = _layout_start(ctx)
    n = ctx.segment_count
    pool: list[tuple[np.ndarray, str]] = []

    for sign in (1.0, -1.0):
        x = base.copy()
        x[0] = float(np.clip(x[0] + sign * rng.uniform(0.10, 0.18), 0.0, 1.0))
        pool.append((_clip_x(ctx, x), "perturb_wrong_k"))

    x = base.copy()
    for i in range(2, 1 + n):
        x[i] = float(np.clip(x[i] + rng.uniform(-0.12, 0.12), 0.0, 1.0))
    pool.append((_clip_x(ctx, x), "perturb_offset_shift"))

    k_small = _k_small_widths(ctx)
    off_fracs = widths_to_offset_fracs(ctx.blocks, k_small)
    x = np.concatenate([[base[0]], off_fracs, base[ls:]])
    pool.append((_clip_x(ctx, x), "perturb_small_segment"))

    x = base.copy()
    x[ls] = float(rng.uniform(0.85, 1.0))
    x[ls + 1] = float(rng.uniform(0.85, 1.0))
    pool.append((_clip_x(ctx, x), "perturb_weak_hough"))

    x = base.copy()
    x[ls + 2] = float(rng.uniform(0.85, 1.0))
    x[ls + 3] = float(rng.uniform(0.85, 1.0))
    pool.append((_clip_x(ctx, x), "perturb_ambiguous_lines"))

    x = base.copy()
    x[0] = float(np.clip(x[0] + rng.uniform(-0.08, 0.08), 0.0, 1.0))
    for i in range(2, 1 + n):
        x[i] = float(np.clip(x[i] + rng.uniform(-0.06, 0.06), 0.0, 1.0))
    pool.append((_clip_x(ctx, x), "perturb_misaligned"))

    x = base.copy()
    x[0] = float(rng.choice([0.02, 0.98]))
    for i in range(2, 1 + n):
        x[i] = float(rng.uniform(0.35, 0.65))
    pool.append((_clip_x(ctx, x), "perturb_guardrail_smoke"))

    ri = _r_surface_index(ctx)
    x = base.copy()
    x[ri] = 0.05
    pool.append((_clip_x(ctx, x), "perturb_wrong_r_low"))
    x = base.copy()
    x[ri] = 0.95
    pool.append((_clip_x(ctx, x), "perturb_wrong_r_high"))

    if n_target <= 0:
        return []

    if n_target <= len(pool):
        picks = rng.choice(len(pool), size=n_target, replace=False)
        return [pool[int(i)] for i in picks]

    out: list[tuple[np.ndarray, str]] = []
    order = rng.permutation(len(pool))
    while len(out) < n_target:
        for i in order:
            out.append(pool[int(i)])
            if len(out) >= n_target:
                break
    return out[:n_target]


def _forced_perturbation_k_stream(
    ctx: RingContext,
    base_x: np.ndarray,
    *,
    rng: np.random.Generator,
    n_target: int,
) -> list[tuple[np.ndarray, str]]:
    """K-position perturbations only (layout frozen)."""
    from lib.layout_bo import _coerce_search_x

    base = float(_coerce_search_x(ctx, base_x)[0])
    pool: list[tuple[np.ndarray, str]] = []

    for sign in (1.0, -1.0):
        k = float(np.clip(base + sign * rng.uniform(0.10, 0.18), 0.0, 1.0))
        pool.append((np.asarray([k], dtype=float), "perturb_k_shift"))
    for anchor in (0.02, 0.98):
        pool.append((np.asarray([anchor], dtype=float), "perturb_k_edge"))
    for _ in range(3):
        k = float(np.clip(base + rng.uniform(-0.08, 0.08), 0.0, 1.0))
        pool.append((np.asarray([k], dtype=float), "perturb_k_jitter"))
    k = float(rng.uniform(0.25, 0.75))
    pool.append((np.asarray([k], dtype=float), "perturb_k_mid"))

    if n_target <= 0:
        return []
    if n_target <= len(pool):
        picks = rng.choice(len(pool), size=n_target, replace=False)
        return [pool[int(i)] for i in picks]
    out: list[tuple[np.ndarray, str]] = []
    order = rng.permutation(len(pool))
    while len(out) < n_target:
        for i in order:
            out.append(pool[int(i)])
            if len(out) >= n_target:
                break
    return out[:n_target]


def _forced_perturbation_layout_stream(
    ctx: RingContext,
    base_x: np.ndarray,
    *,
    rng: np.random.Generator,
    n_target: int,
) -> list[tuple[np.ndarray, str]]:
    """Offset + layout-tail perturbations only (k_y frozen)."""
    from lib.layout_bo import _coerce_search_x, widths_to_offset_fracs

    base = _coerce_search_x(ctx, base_x)
    n = ctx.segment_count
    ls = 0
    pool: list[tuple[np.ndarray, str]] = []

    x = base.copy()
    for i in range(1, n):
        x[i] = float(np.clip(x[i] + rng.uniform(-0.12, 0.12), 0.0, 1.0))
    pool.append((_clip_x(ctx, x), "perturb_offset_shift"))

    k_small = _k_small_widths(ctx)
    off_fracs = widths_to_offset_fracs(ctx.blocks, k_small)
    x = np.concatenate([off_fracs, base[n:]])
    pool.append((_clip_x(ctx, x), "perturb_small_segment"))

    x = base.copy()
    x[ls] = float(rng.uniform(0.85, 1.0))
    x[ls + 1] = float(rng.uniform(0.85, 1.0))
    pool.append((_clip_x(ctx, x), "perturb_weak_hough"))

    x = base.copy()
    x[ls + 2] = float(rng.uniform(0.85, 1.0))
    x[ls + 3] = float(rng.uniform(0.85, 1.0))
    pool.append((_clip_x(ctx, x), "perturb_ambiguous_lines"))

    for i in range(1, n):
        x = base.copy()
        x[i] = float(np.clip(x[i] + rng.uniform(-0.06, 0.06), 0.0, 1.0))
        pool.append((_clip_x(ctx, x), "perturb_misaligned"))

    ri = _r_surface_index(ctx)
    x = base.copy()
    x[ri] = 0.05
    pool.append((_clip_x(ctx, x), "perturb_wrong_r_low"))
    x = base.copy()
    x[ri] = 0.95
    pool.append((_clip_x(ctx, x), "perturb_wrong_r_high"))

    if n_target <= 0:
        return []
    if n_target <= len(pool):
        picks = rng.choice(len(pool), size=n_target, replace=False)
        return [pool[int(i)] for i in picks]
    out: list[tuple[np.ndarray, str]] = []
    order = rng.permutation(len(pool))
    while len(out) < n_target:
        for i in order:
            out.append(pool[int(i)])
            if len(out) >= n_target:
                break
    return out[:n_target]


def experience_phase_budgets(n_evals: int) -> dict[str, int]:
    """20% warm-start / 20% forced perturbation / 60% GP-BO."""
    n_perturb = max(1, int(round(0.20 * n_evals)))
    n_warm = int(round(0.20 * n_evals))
    n_gp = max(0, n_evals - n_warm - n_perturb)
    return {"warm": n_warm, "perturb": n_perturb, "gp": n_gp}

"""Shared layout + r_surface_min BO core (GP-BO + EI).

Search vector x = [k_y_frac, off_frac[block_0], ..., off_frac[block_n-1], r_frac]
where off_frac[i] = per_ring_offsets[block_i] / H (K block fixed at 0).
Do NOT use cumulative arc-width encoding — it breaks GT round-trip on wrapped rings.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from lib.ceiling_gate import (
    REPO_ROOT,
    VENV_PY,
    DET_CLI,
    SEG_CLI,
    DET_DEFAULT,
    SEG_DEFAULT,
    best_ceiling_over_cutoff,
    blocks_for_segment_count,
    derive_gt_layout,
    detect_segment_count,
    otsu_threshold,
    run_agents_unfiltered,
    setup_sandbox,
)

EXTRACT_INTRINSICS = REPO_ROOT / "agents" / "2_detection" / "scripts" / "extract_intrinsics.py"

# 7-block K-small / AB-large prior (normalized arc widths)
PRIOR_K_SMALL_7 = np.array([0.07, 0.15, 0.15, 0.15, 0.15, 0.15, 0.18])
PRIOR_K_SMALL_6 = np.array([0.07, 0.18, 0.18, 0.18, 0.18, 0.21])


@dataclass
class RingContext:
    tunnel_id: str
    ring_id: int
    source_root: Path
    run_root: Path
    segment_count: int = 0
    blocks: list[str] = field(default_factory=list)
    H: int = 0
    r_lo: float = 0.0
    r_hi: float = 0.0
    r_otsu: float = float("nan")
    ceiling_miou: float = 0.0
    src_ring: Path = field(default_factory=Path)
    sandbox_data: Path = field(default_factory=Path)
    sandbox_ring: Path = field(default_factory=Path)
    out_dir: Path = field(default_factory=Path)

    @property
    def ring_key(self) -> str:
        return f"r{int(self.ring_id)}"

    @property
    def case_id(self) -> str:
        return f"{self.tunnel_id}/{self.ring_key}"

    @property
    def search_dim(self) -> int:
        return self.segment_count + 2


def _compute_miou(df: pd.DataFrame, max_class: int = 7) -> float | None:
    if "segment" not in df.columns or "pred" not in df.columns:
        return None
    tmp = df[["segment", "pred"]].dropna(subset=["segment", "pred"]).copy()
    if tmp.empty:
        return None
    gt = pd.to_numeric(tmp["segment"], errors="coerce").fillna(0).astype(int).to_numpy()
    pred = pd.to_numeric(tmp["pred"], errors="coerce").fillna(0).astype(int).to_numpy()
    valid = (gt >= 0) & (gt <= max_class) & (pred >= 0) & (pred <= max_class)
    gt, pred = gt[valid], pred[valid]
    if gt.size == 0:
        return None
    labels = sorted(set(gt.tolist()) | set(pred.tolist()))
    ious = []
    for cls in labels:
        g, p = gt == cls, pred == cls
        union = np.logical_or(g, p).sum()
        if union:
            ious.append(float(np.logical_and(g, p).sum() / union))
    return float(np.mean(ious)) if ious else None


def _import_extract_metrics():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    agents_scripts = REPO_ROOT / "agents" / "2_detection" / "scripts"
    if str(agents_scripts) not in sys.path:
        sys.path.insert(0, str(agents_scripts))
    from extract_intrinsics import extract_detection_metrics  # noqa: WPS433

    return extract_detection_metrics


def build_ring_context(
    tunnel_id: str,
    ring_id: int,
    *,
    source_root: Path,
    run_root: Path,
    segment_count: int | None = None,
) -> RingContext:
    src_ring = source_root / tunnel_id / f"r{int(ring_id)}"
    if not src_ring.is_dir():
        raise FileNotFoundError(f"No preprocessing at {src_ring}")

    seg_n = segment_count if segment_count is not None else detect_segment_count(src_ring)
    blocks = blocks_for_segment_count(seg_n)
    sandbox_data = run_root / "sandbox"
    sandbox_ring = sandbox_data / tunnel_id / f"r{int(ring_id)}"
    out_dir = run_root / tunnel_id / f"r{int(ring_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_sandbox(src_ring, sandbox_ring)
    H = int(np.load(sandbox_ring / "depth_map.npy").shape[0])
    enh = pd.read_csv(src_ring / "enhanced.csv")
    r_vals = enh["r"].to_numpy()
    r_lo = float(np.nanpercentile(r_vals, 1))
    r_hi = float(np.nanpercentile(r_vals, 60))
    r_otsu = float(otsu_threshold(r_vals))

    ctx = RingContext(
        tunnel_id=tunnel_id,
        ring_id=ring_id,
        source_root=source_root,
        run_root=run_root,
        segment_count=seg_n,
        blocks=blocks,
        H=H,
        r_lo=r_lo,
        r_hi=r_hi,
        r_otsu=r_otsu,
        src_ring=src_ring,
        sandbox_data=sandbox_data,
        sandbox_ring=sandbox_ring,
        out_dir=out_dir,
    )
    return ctx


def compute_ceiling_reference(ctx: RingContext) -> dict[str, Any]:
    gt = derive_gt_layout(ctx.src_ring, ctx.sandbox_ring, ctx.segment_count)
    (ctx.out_dir / "gt_layout.json").write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")

    final_df = run_agents_unfiltered(
        ctx.tunnel_id,
        ctx.ring_id,
        ctx.sandbox_data,
        ctx.sandbox_ring,
        gt["k_y"],
        gt["offsets"],
        ctx.segment_count,
        ctx.blocks,
        tag="ceiling_ref",
    )
    if final_df is None:
        ceiling = {"agents_gt_ceiling_miou": None, "failure_reason": "agent_error"}
    else:
        cap = best_ceiling_over_cutoff(final_df, max_class=ctx.segment_count)
        ceiling = {
            "case_id": ctx.case_id,
            "agents_gt_ceiling_miou": cap["ceiling_miou"],
            "ceiling_no_filter_miou": cap["ceiling_no_filter_miou"],
            "r_surface_min_selected": cap["r_surface_min"],
            "r_surface_min_otsu_intrinsic": round(ctx.r_otsu, 4),
            "per_class_iou": cap["per_class_iou"],
            "r_search_bounds": {"r_lo": ctx.r_lo, "r_hi": ctx.r_hi},
        }
        ctx.ceiling_miou = float(cap["ceiling_miou"])

    (ctx.out_dir / "ceiling.json").write_text(json.dumps(ceiling, indent=2) + "\n", encoding="utf-8")
    return ceiling


def widths_to_offset_fracs(blocks: list[str], widths: np.ndarray) -> np.ndarray:
    """Cumulative arc positions (as H-fractions) from normalized block widths."""
    w = np.clip(widths, 1e-3, None)
    w = w / w.sum()
    return np.concatenate([[0.0], np.cumsum(w)[:-1]])


def decode_x(ctx: RingContext, x: np.ndarray) -> tuple[float, dict[str, float], float]:
    x = np.asarray(x, dtype=float)
    k_y = float(x[0]) * ctx.H
    # Per-block boundary offset fractions (detection uses absolute offsets from K; K=0).
    off_fracs = x[1 : 1 + ctx.segment_count]
    offsets = {ctx.blocks[i]: float(off_fracs[i] % 1.0) * ctx.H for i in range(len(ctx.blocks))}
    offsets[ctx.blocks[0]] = 0.0
    r_frac = float(x[1 + ctx.segment_count])
    r_surface_min = ctx.r_lo + r_frac * (ctx.r_hi - ctx.r_lo)
    return k_y, offsets, r_surface_min


def arc_width_entropy(widths: np.ndarray) -> float:
    w = np.clip(widths, 1e-9, None)
    w = w / w.sum()
    return float(-np.sum(w * np.log(w)))


def offsets_to_arc_widths(blocks: list[str], offsets: dict[str, float], H: int) -> np.ndarray:
    ys = [float(offsets[b]) % H for b in blocks]
    widths = []
    for i in range(len(blocks)):
        w = (ys[(i + 1) % len(blocks)] - ys[i]) % H
        widths.append(w)
    return np.array(widths, dtype=float)


def encode_gt_layout_x(ctx: RingContext, gt_layout: dict[str, Any], r_surface_min: float) -> np.ndarray:
    """Encode GT k_y + per-block offset fractions + r_surface_min into search vector."""
    k_y_frac = float(gt_layout["k_y"]) % ctx.H / max(ctx.H, 1)
    off_fracs = np.array([float(gt_layout["offsets"][b]) % ctx.H / max(ctx.H, 1) for b in ctx.blocks])
    off_fracs[0] = 0.0
    r_span = max(ctx.r_hi - ctx.r_lo, 1e-9)
    r_frac = float(np.clip((r_surface_min - ctx.r_lo) / r_span, 0.0, 1.0))
    return np.concatenate([[k_y_frac], off_fracs, [r_frac]])


def r_surface_min_to_frac(ctx: RingContext, r_surface_min: float) -> float:
    r_span = max(ctx.r_hi - ctx.r_lo, 1e-9)
    return float(np.clip((r_surface_min - ctx.r_lo) / r_span, 0.0, 1.0))


def ceiling_push_priors(ctx: RingContext, gt_layout: dict[str, Any], ceiling: dict[str, Any]) -> list[tuple[np.ndarray, str]]:
    """GT-layout seeds + geometric priors for ceiling-push BO."""
    r_mid = ctx.r_lo + 0.5 * (ctx.r_hi - ctx.r_lo)
    r_ceil = ceiling.get("r_surface_min_selected")
    if r_ceil is None:
        r_ceil = r_mid
    priors: list[tuple[np.ndarray, str]] = [
        (encode_gt_layout_x(ctx, gt_layout, r_mid), "gt_layout"),
        (encode_gt_layout_x(ctx, gt_layout, float(r_ceil)), "gt_layout_ceiling_r"),
        (encode_gt_layout_x(ctx, gt_layout, float(ctx.r_otsu)), "gt_layout_otsu_r"),
    ]
    for i, x in enumerate(geometric_priors(ctx)):
        priors.append((x, f"geometric_{i}"))
    return priors


def load_trial_history(ctx: RingContext) -> tuple[list[np.ndarray], list[float], list[dict[str, Any]], int, float]:
    """Load existing bo_trials.csv for resume. Returns X, Y, rows, next_idx, best_y."""
    trials_path = ctx.out_dir / "bo_trials.csv"
    if not trials_path.exists():
        return [], [], [], 0, -1.0
    df = pd.read_csv(trials_path)
    if df.empty:
        return [], [], [], 0, -1.0
    X, Y, rows = [], [], []
    for _, row in df.iterrows():
        x = np.asarray(json.loads(row["search_x"]), dtype=float)
        y = float(row["gt_miou"])
        X.append(x)
        Y.append(y)
        rows.append(row.to_dict())
    next_idx = int(df["trial_id"].max()) + 1
    best_y = float(df["gt_miou"].max())
    return X, Y, rows, next_idx, best_y


def _regret_at_checkpoints(trials_df: pd.DataFrame, ceiling: float, checkpoints: list[int]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if trials_df.empty or not ceiling:
        return {str(c): None for c in checkpoints}
    for c in checkpoints:
        sub = trials_df[trials_df["trial_id"] < c]
        if sub.empty:
            out[str(c)] = None
        else:
            best = float(sub["gt_miou"].max())
            out[str(c)] = round(float(ceiling - best), 4)
    return out


def write_ceiling_push_report(
    ctx: RingContext,
    trials_df: pd.DataFrame,
    *,
    target_regret: float,
    stop_reason: str,
    n_iterations: int,
    order_branch: str,
    gt_layout_trial_miou: float | None = None,
) -> dict[str, Any]:
    best_bo = float(trials_df["gt_miou"].max()) if not trials_df.empty else 0.0
    target_miou = float(ctx.ceiling_miou) - float(target_regret) if ctx.ceiling_miou else None
    regret = float(ctx.ceiling_miou) - best_bo if ctx.ceiling_miou else None
    target_reached = target_miou is not None and best_bo >= target_miou

    checkpoints = [128, 256, 384, 512, 768, 1024]
    report = {
        "objective": "minimize_regret_vs_gt_ceiling",
        "case_id": ctx.case_id,
        "ceiling_miou_reference": ctx.ceiling_miou,
        "best_bo_miou": round(best_bo, 4),
        "regret_vs_ceiling": round(regret, 4) if regret is not None else None,
        "target_miou": round(target_miou, 4) if target_miou is not None else None,
        "target_regret": float(target_regret),
        "target_reached": bool(target_reached),
        "stop_reason": stop_reason,
        "total_evals": int(len(trials_df)),
        "n_iterations": int(n_iterations),
        "regret_at_checkpoints": _regret_at_checkpoints(trials_df, ctx.ceiling_miou, checkpoints),
        "gt_layout_trial_miou": round(gt_layout_trial_miou, 4) if gt_layout_trial_miou is not None else None,
        "improvement_over_gt_layout_seed": (
            round(best_bo - gt_layout_trial_miou, 4) if gt_layout_trial_miou is not None else None
        ),
        "order_branch": order_branch,
    }
    (ctx.out_dir / "ceiling_push_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def geometric_priors(ctx: RingContext) -> list[np.ndarray]:
    n = ctx.segment_count
    equal_w = np.full(n, 1.0 / n)
    k_small = PRIOR_K_SMALL_7 if n == 7 else PRIOR_K_SMALL_6
    if len(k_small) != n:
        k_small = np.concatenate([k_small[: n - 1], [k_small[-1]]])[:n]
        k_small = k_small / k_small.sum()
    r_mid = np.array([0.5])
    return [
        np.concatenate([[0.0], widths_to_offset_fracs(ctx.blocks, equal_w), r_mid]),
        np.concatenate([[0.0], widths_to_offset_fracs(ctx.blocks, k_small), r_mid]),
    ]


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def make_gp(seed: int, n_dims: int) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(n_dims),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=seed,
    )


def evaluate_trial(
    ctx: RingContext,
    k_y: float,
    offsets: dict[str, float],
    r_surface_min: float,
    tag: str,
    order_branch: str = "plus",
) -> dict[str, Any]:
    """Run det+seg with injected layout; return metrics dict."""
    H = ctx.H
    det = json.loads(DET_DEFAULT.read_text(encoding="utf-8"))
    seg = json.loads(SEG_DEFAULT.read_text(encoding="utf-8"))
    det["segment_count"] = ctx.segment_count
    det["enabled_blocks"] = ctx.blocks
    det["k_anchor_semantics"] = "boundary_start"
    det["per_ring_offsets"] = {"0": {b: float(offsets[b]) for b in ctx.blocks}}
    det["k_y_positions"] = [float(k_y) % H]
    branch = str(order_branch).lower()
    det["reverse_ring_order"] = branch == "minus"
    seg["segment_count"] = ctx.segment_count
    seg["r_surface_min"] = float(r_surface_min)

    ctx.sandbox_ring.mkdir(parents=True, exist_ok=True)
    (ctx.sandbox_ring / "parameters_detection.json").write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")
    (ctx.sandbox_ring / "parameters_segmentation.json").write_text(json.dumps(seg, indent=2) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    pred_before_filter = None

    for cli in (DET_CLI, SEG_CLI):
        log = ctx.sandbox_ring / "logs" / f"{tag}_{cli.parent.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                [str(VENV_PY), str(cli), ctx.tunnel_id, str(ctx.ring_id), "--data-dir", str(ctx.sandbox_data)],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=False,
            )
        if proc.returncode != 0:
            return {"gt_miou": 0.0, "agent_error": True}

    final_path = ctx.sandbox_ring / "final.csv"
    if not final_path.exists():
        return {"gt_miou": 0.0, "agent_error": True}

    final_df = pd.read_csv(final_path)
    gt_miou = _compute_miou(final_df, max_class=ctx.segment_count) or 0.0

    n_reclass = 0
    seg_log = ctx.sandbox_ring / "logs" / f"{tag}_3_segmentation.log"
    if seg_log.exists():
        for line in seg_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "Radial filter" in line and "background" in line:
                try:
                    part = line.split(":")[1].split("points")[0].strip().replace(",", "")
                    n_reclass = int(part)
                except (IndexError, ValueError):
                    pass
                break

    intrinsics: dict[str, Any] = {}
    try:
        extract_detection_metrics = _import_extract_metrics()
        raw = extract_detection_metrics(str(ctx.sandbox_ring))
        for k, v in raw.items():
            if k == "det_guardrail_violations":
                intrinsics[k] = json.dumps(v) if v else "[]"
            else:
                intrinsics[k] = v
    except Exception as exc:
        intrinsics["det_extract_error"] = str(exc)

    r_span = max(ctx.r_hi - ctx.r_lo, 1e-9)
    arc_widths = offsets_to_arc_widths(ctx.blocks, offsets, H)

    return {
        "gt_miou": float(gt_miou),
        "agent_error": False,
        "n_reclassified_by_r_filter": n_reclass,
        "r_surface_min_otsu_ref": round(ctx.r_otsu, 4),
        "r_surface_min_frac": float((r_surface_min - ctx.r_lo) / r_span),
        "k_y_frac": float((k_y % H) / max(H, 1)),
        "arc_width_entropy": arc_width_entropy(arc_widths),
        "order_branch": branch,
        "branch_is_minus": branch == "minus",
        **intrinsics,
    }


def run_gp_bo(
    ctx: RingContext,
    *,
    n_evals: int = 64,
    seed: int = 7,
    order_branch: str = "plus",
    resume: bool = False,
    gt_layout: dict[str, Any] | None = None,
    ceiling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GP-BO + EI over layout + r_surface_min. Appends n_evals new trials (resume skips priors)."""
    rng = np.random.default_rng(seed)
    dim = ctx.search_dim

    if resume:
        X, Y, prior_rows, idx, best_y = load_trial_history(ctx)
        best_x = None
        best_row: dict[str, Any] | None = None
        if prior_rows:
            best_idx = int(np.argmax(Y))
            best_x = X[best_idx]
            best_row = prior_rows[best_idx]
        trial_rows: list[dict[str, Any]] = []
    else:
        X, Y, trial_rows = [], [], []
        idx, best_y = 0, -1.0
        best_x, best_row = None, None

    end_idx = idx + n_evals
    consecutive_errors = 0

    def record(x: np.ndarray, trial_idx: int, kind: str, gp_meta: dict[str, Any] | None = None) -> bool:
        nonlocal best_y, best_x, best_row, consecutive_errors
        k_y, offs, r_smin = decode_x(ctx, x)
        tag = f"trial{trial_idx:03d}"
        metrics = evaluate_trial(ctx, k_y, offs, r_smin, tag=tag, order_branch=order_branch)
        y = float(metrics["gt_miou"])
        if metrics.get("agent_error"):
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        X.append(np.asarray(x, dtype=float))
        Y.append(y)
        if y > best_y:
            best_y, best_x = y, np.asarray(x, dtype=float)
            best_row = dict(metrics)

        row: dict[str, Any] = {
            "trial_id": trial_idx,
            "tunnel_id": ctx.tunnel_id,
            "ring_id": ctx.ring_id,
            "case_id": ctx.case_id,
            "kind": kind,
            "k_y": k_y,
            "per_ring_offsets": json.dumps({"0": offs}),
            "r_surface_min": r_smin,
            "gt_miou": y,
            "best_so_far": best_y,
            "regret_vs_ceiling": ctx.ceiling_miou - best_y if ctx.ceiling_miou else None,
            "search_x": json.dumps(x.tolist()),
        }
        row.update(metrics)
        if gp_meta:
            row.update(gp_meta)
        trial_rows.append(row)
        regret_s = f"{ctx.ceiling_miou - best_y:.4f}" if ctx.ceiling_miou else "n/a"
        print(f"  trial{trial_idx:03d} [{kind}] miou={y:.4f} best={best_y:.4f} regret={regret_s}")
        return consecutive_errors > 5

    if not resume:
        if gt_layout is None:
            gt_path = ctx.out_dir / "gt_layout.json"
            gt_layout = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
        if ceiling is None:
            ceil_path = ctx.out_dir / "ceiling.json"
            ceiling = json.loads(ceil_path.read_text(encoding="utf-8")) if ceil_path.exists() else {}

        if gt_layout:
            for x, kind in ceiling_push_priors(ctx, gt_layout, ceiling):
                if idx >= end_idx:
                    break
                if record(x, idx, kind):
                    break
                idx += 1

        n_random_target = min(32, max(0, (end_idx - idx) // 8), max(0, end_idx - idx))
        for _ in range(n_random_target):
            if idx >= end_idx:
                break
            if record(rng.random(dim), idx, "random"):
                break
            idx += 1

    while idx < end_idx:
        if len(X) < 2:
            if record(rng.random(dim), idx, "random"):
                break
            idx += 1
            continue
        gp = make_gp(seed, dim)
        X_arr = np.vstack(X)
        y_arr = np.asarray(Y)
        gp.fit(X_arr, y_arr)
        pool = rng.random((4096, dim))
        mu, std = gp.predict(pool, return_std=True)
        ei = expected_improvement(mu, std, y_best=float(np.max(y_arr)))
        j = int(np.argmax(ei))
        chosen = pool[j]
        gp_meta = {
            "bo_surrogate_mean": float(mu[j]),
            "bo_surrogate_std": float(std[j]),
            "ei_value": float(ei[j]),
        }
        if record(chosen, idx, "bo", gp_meta=gp_meta):
            break
        idx += 1

    if resume and (ctx.out_dir / "bo_trials.csv").exists():
        old_df = pd.read_csv(ctx.out_dir / "bo_trials.csv")
        new_df = pd.DataFrame(trial_rows)
        trials_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        trials_df = pd.DataFrame(trial_rows)

    trials_path = ctx.out_dir / "bo_trials.csv"
    trials_df.to_csv(trials_path, index=False)

    conv_cols = ["trial_id", "kind", "gt_miou", "best_so_far", "regret_vs_ceiling", "k_y", "r_surface_min"]
    trials_df[conv_cols].to_csv(ctx.out_dir / "convergence.csv", index=False)

    k_y_b, offs_b, r_b = decode_x(ctx, best_x) if best_x is not None else (0.0, {}, 0.0)
    best_payload = {
        "case_id": ctx.case_id,
        "best_bo_miou": best_y,
        "best_k_y": k_y_b,
        "best_offsets": offs_b,
        "best_r_surface_min": r_b,
        "n_evals": int(len(trials_df)),
        "ceiling_miou_reference": ctx.ceiling_miou,
        "order_branch": order_branch,
    }
    (ctx.out_dir / "best_bo_trial.json").write_text(json.dumps(best_payload, indent=2) + "\n", encoding="utf-8")

    gt_layout_miou = None
    if not trials_df.empty:
        gt_rows = trials_df[trials_df["kind"] == "gt_layout"]
        if not gt_rows.empty:
            gt_layout_miou = float(gt_rows.iloc[0]["gt_miou"])

    return {
        "trials_df": trials_df,
        "best_payload": best_payload,
        "best_row": best_row,
        "gt_layout_trial_miou": gt_layout_miou,
        "chunk_trials": len(trial_rows),
        "agent_error_stop": consecutive_errors > 5,
    }


def run_iterative_ceiling_push(
    tunnel_id: str,
    ring_id: int,
    *,
    source_root: Path,
    run_root: Path,
    segment_count: int | None = None,
    order_branch: str = "plus",
    eval_chunk: int = 128,
    max_total_evals: int = 1024,
    target_regret: float = 0.05,
    min_improvement: float = 0.005,
    seed: int = 7,
    skip_ceiling_if_exists: bool = True,
) -> dict[str, Any]:
    """Iteratively run BO chunks until target regret, ceiling_bind, or max budget."""
    ctx = build_ring_context(
        tunnel_id, ring_id, source_root=source_root, run_root=run_root, segment_count=segment_count
    )
    ceil_path = ctx.out_dir / "ceiling.json"
    if skip_ceiling_if_exists and ceil_path.exists():
        ceiling = json.loads(ceil_path.read_text(encoding="utf-8"))
        ctx.ceiling_miou = float(ceiling.get("agents_gt_ceiling_miou") or 0.0)
        print(f"== ceiling reference (cached): {ctx.case_id} = {ctx.ceiling_miou:.4f} ==")
    else:
        print(f"== ceiling reference: {ctx.case_id} ==")
        ceiling = compute_ceiling_reference(ctx)
        print(f"  ceiling mIoU (reference) = {ctx.ceiling_miou:.4f}")

    gt_path = ctx.out_dir / "gt_layout.json"
    gt_layout = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}

    target_miou = float(ctx.ceiling_miou) - float(target_regret)
    iteration_summaries: list[dict[str, Any]] = []
    stop_reason = "max_budget"
    n_iterations = 0
    gt_layout_trial_miou: float | None = None

    while True:
        existing = 0
        if (ctx.out_dir / "bo_trials.csv").exists():
            existing = len(pd.read_csv(ctx.out_dir / "bo_trials.csv"))
        if existing >= max_total_evals:
            stop_reason = "max_budget"
            break

        chunk = min(eval_chunk, max_total_evals - existing)
        if chunk <= 0:
            stop_reason = "max_budget"
            break

        n_iterations += 1
        resume = existing > 0
        print(f"== iteration {n_iterations}: +{chunk} evals (total cap {max_total_evals}, resume={resume}): {ctx.case_id} ==")
        prev_best = -1.0
        if resume:
            _, _, _, _, prev_best = load_trial_history(ctx)

        bo_result = run_gp_bo(
            ctx,
            n_evals=chunk,
            seed=seed + n_iterations,
            order_branch=order_branch,
            resume=resume,
            gt_layout=gt_layout if not resume else None,
            ceiling=ceiling if not resume else None,
        )

        if bo_result.get("gt_layout_trial_miou") is not None:
            gt_layout_trial_miou = bo_result["gt_layout_trial_miou"]

        trials_df = bo_result["trials_df"]
        best_bo = float(trials_df["gt_miou"].max())
        regret = float(ctx.ceiling_miou) - best_bo if ctx.ceiling_miou else None
        chunk_gain = best_bo - prev_best if prev_best >= 0 else best_bo

        iter_dir = ctx.out_dir / f"iteration_{n_iterations:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        chunk_df = pd.DataFrame(trials_df.tail(bo_result["chunk_trials"]))
        chunk_df.to_csv(iter_dir / "chunk_trials.csv", index=False)
        iter_summary = {
            "iteration": n_iterations,
            "chunk_evals": chunk,
            "total_evals": int(len(trials_df)),
            "best_bo_miou": round(best_bo, 4),
            "regret_vs_ceiling": round(regret, 4) if regret is not None else None,
            "chunk_gain": round(chunk_gain, 4),
            "target_miou": round(target_miou, 4),
        }
        (iter_dir / "iteration_summary.json").write_text(json.dumps(iter_summary, indent=2) + "\n", encoding="utf-8")
        iteration_summaries.append(iter_summary)

        if bo_result.get("agent_error_stop"):
            stop_reason = "agent_errors"
            break
        if best_bo >= target_miou:
            stop_reason = "target_reached"
            break
        if regret is not None and regret <= target_regret and chunk_gain < min_improvement:
            stop_reason = "ceiling_bind"
            break
        if len(trials_df) >= max_total_evals:
            stop_reason = "max_budget"
            break

    trials_df = pd.read_csv(ctx.out_dir / "bo_trials.csv") if (ctx.out_dir / "bo_trials.csv").exists() else pd.DataFrame()
    report = write_ceiling_push_report(
        ctx,
        trials_df,
        target_regret=target_regret,
        stop_reason=stop_reason,
        n_iterations=n_iterations,
        order_branch=order_branch,
        gt_layout_trial_miou=gt_layout_trial_miou,
    )
    return {
        "ctx": ctx,
        "report": report,
        "iteration_summaries": iteration_summaries,
        "trials_df": trials_df,
    }


def write_experience_gate(ctx: RingContext, trials_df: pd.DataFrame, bo_result: dict[str, Any]) -> dict[str, Any]:
    miou_std = float(trials_df["gt_miou"].std()) if len(trials_df) > 1 else 0.0
    random_phase = trials_df[trials_df["kind"].isin(["prior", "random", "gt_layout", "geometric_0", "geometric_1"])]
    intrinsic_cols = [c for c in trials_df.columns if c.startswith("det_") or c in (
        "k_y_frac", "arc_width_entropy", "n_reclassified_by_r_filter",
    )]
    has_intrinsic = any(trials_df[c].notna().any() for c in intrinsic_cols if c in trials_df.columns)

    best_bo = float(trials_df["gt_miou"].max())
    random_median = float(random_phase["gt_miou"].median()) if not random_phase.empty else 0.0

    gate = {
        "case_id": ctx.case_id,
        "objective": "GT mIoU experience collection (design-time)",
        "n_evals": int(len(trials_df)),
        "ceiling_miou_reference": ctx.ceiling_miou,
        "best_bo_miou": best_bo,
        "regret_vs_ceiling": ctx.ceiling_miou - best_bo if ctx.ceiling_miou else None,
        "miou_std": miou_std,
        "random_phase_median_miou": random_median,
        "bo_beats_random_median": bool(best_bo > random_median),
        "criteria": {
            "n_trials_complete": len(trials_df) >= 64,
            "trial_schema_ok": "gt_miou" in trials_df.columns and "r_surface_min" in trials_df.columns,
            "miou_spread_ok": miou_std > 0.02,
            "bo_improves_over_random": best_bo > random_median,
            "intrinsics_populated": has_intrinsic,
        },
        "passed": bool(
            len(trials_df) >= 64
            and miou_std > 0.02
            and best_bo > random_median
            and has_intrinsic
        ),
        "outputs": {
            "bo_trials": str((ctx.out_dir / "bo_trials.csv").relative_to(REPO_ROOT)),
            "best_bo_trial": str((ctx.out_dir / "best_bo_trial.json").relative_to(REPO_ROOT)),
            "ceiling": str((ctx.out_dir / "ceiling.json").relative_to(REPO_ROOT)),
        },
    }
    (ctx.out_dir / "experience_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate


def write_panel_ceiling_push_summary(run_root: Path, ring_results: list[dict[str, Any]]) -> None:
    """Write panel_summary.csv, iteration_log.json, merged bo_trials.csv, ceiling_push_summary.md."""
    run_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    iteration_log: dict[str, Any] = {}
    all_trials: list[pd.DataFrame] = []

    for res in ring_results:
        report = res["report"]
        ctx = res["ctx"]
        ring_key = ctx.case_id
        summaries.append({
            "ring_key": ring_key,
            "segment_count": ctx.segment_count,
            "order_branch": report.get("order_branch"),
            "ceiling_reference": report.get("ceiling_miou_reference"),
            "target_miou": report.get("target_miou"),
            "target_regret": report.get("target_regret"),
            "best_bo_miou": report.get("best_bo_miou"),
            "regret_vs_ceiling": report.get("regret_vs_ceiling"),
            "target_reached": report.get("target_reached"),
            "stop_reason": report.get("stop_reason"),
            "total_evals": report.get("total_evals"),
            "n_iterations": report.get("n_iterations"),
        })
        iteration_log[ring_key] = res.get("iteration_summaries", [])
        trials_path = ctx.out_dir / "bo_trials.csv"
        if trials_path.exists():
            all_trials.append(pd.read_csv(trials_path))

    panel_df = pd.DataFrame(summaries)
    panel_df.to_csv(run_root / "panel_summary.csv", index=False)
    (run_root / "iteration_log.json").write_text(json.dumps(iteration_log, indent=2) + "\n", encoding="utf-8")

    if all_trials:
        merged = pd.concat(all_trials, ignore_index=True)
        merged.to_csv(run_root / "bo_trials.csv", index=False)

    lines = [
        "# Ceiling-push BO summary",
        "",
        f"Run root: `{run_root.relative_to(REPO_ROOT)}`",
        "",
        "| Ring | Ceiling | Target | Best BO | Regret | Stop | Evals | Iters |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['ring_key']} | {s['ceiling_reference']} | {s['target_miou']} | {s['best_bo_miou']} | "
            f"{s['regret_vs_ceiling']} | {s['stop_reason']} | {s['total_evals']} | {s['n_iterations']} |"
        )
    lines.extend(["", "## Targets", "", "- Default success: regret ≤ 0.05 vs GT ceiling", ""])
    (run_root / "ceiling_push_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ring_bo(
    tunnel_id: str,
    ring_id: int,
    *,
    source_root: Path,
    run_root: Path,
    n_evals: int = 64,
    seed: int = 7,
    segment_count: int | None = None,
    order_branch: str = "plus",
) -> dict[str, Any]:
    ctx = build_ring_context(tunnel_id, ring_id, source_root=source_root, run_root=run_root, segment_count=segment_count)
    print(f"== ceiling reference: {ctx.case_id} ==")
    ceiling = compute_ceiling_reference(ctx)
    print(f"  ceiling mIoU (reference) = {ctx.ceiling_miou:.4f}")

    gt_path = ctx.out_dir / "gt_layout.json"
    gt_layout = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}

    print(f"== GP-BO ({n_evals} evals, branch={order_branch}): {ctx.case_id} ==")
    bo_result = run_gp_bo(
        ctx, n_evals=n_evals, seed=seed, order_branch=order_branch, gt_layout=gt_layout, ceiling=ceiling
    )
    gate = write_experience_gate(ctx, bo_result["trials_df"], bo_result)
    print(f"== experience gate: passed={gate['passed']} ==")
    return {"ctx": ctx, "gate": gate, "bo_result": bo_result}

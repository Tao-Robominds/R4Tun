from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import jaccard_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "venv" / "bin" / "python"
DET_CLI = REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
SEG_CLI = REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"

RUN_ROOT = REPO_ROOT / "logs" / "v5_finer_proxy_search_v1"
PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
SCOREBOARD = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t123_recovered.csv"
DEPTH_AUDIT = REPO_ROOT / "logs" / "v5_depth_contract_paper_audit_v1" / "all_50_depth_quality_audit.csv"
SRC_T123 = REPO_ROOT / "logs" / "v5_t123_depth_contract_v1"
SRC_T45 = REPO_ROOT / "logs" / "v5_t45_depth_contract_v1"

DET_DEFAULT = REPO_ROOT / "agents" / "2_detection" / "parameters" / "_default_irregular" / "parameters_detection.json"
SEG_DEFAULT = REPO_ROOT / "agents" / "3_segmentation" / "parameters" / "_default_irregular" / "parameters_segmentation.json"

PROTECTED_PREFIXES = (
    REPO_ROOT / "data" / "ablation",
    REPO_ROOT / "data" / "bo",
    REPO_ROOT / "data" / "baseline",
    REPO_ROOT / "data" / "preprocessing_qa",
    REPO_ROOT / "data" / "represents",
    REPO_ROOT / "logs" / "context_preprocessing_v1",
    REPO_ROOT / "r4tun" / "data",
    REPO_ROOT / "r4tun" / "references",
    REPO_ROOT / "methods" / "plans" / "output",
    REPO_ROOT / "stages" / "v4",
)

ROTATIONS = [0, 1, 4]
KB7_BLOCKS = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
KB7_OFFSETS = {"K": 0.0, "B1": 181.9, "A1": 727.5, "A2": 1273.2, "A3": -1636.9, "A4": -1091.3, "B2": -545.6}

CURRENT_PROXY = {
    "intercept": 0.015213,
    "struct_missing_ids_before_n": -0.031164,
    "depth_row_nonempty_ratio_audit": 0.196603,
    "geom_boundary_gap_cv": 0.081910,
}

PILOT_GROUPS = {
    "regression_vs_stabilised": ["1-1/r18", "1-2/r58", "1-4/r197", "1-4/r204", "3-1-2/r46", "3-1-3/r86"],
    "low_tail_final_miou": ["3-1-1/r36", "4-6/r276", "5-6/r285"],
    "search_oracle_diagnostic": ["4-7/r308", "4-4/r212", "5-3/r192", "4-8/r332"],
}

DEPTH_RISK_CONTROLS = {"4-6/r276", "5-6/r285"}


@dataclass(frozen=True)
class KCfg:
    anchor_frac: float
    low_frac: float
    high_frac: float
    low_parity: int

    @property
    def tag(self) -> str:
        return f"a{self.anchor_frac:.3f}_l{self.low_frac:.3f}_h{self.high_frac:.3f}_p{int(self.low_parity)}"


def _assert_writable(path: Path) -> None:
    resolved = path.resolve()
    logs_root = (REPO_ROOT / "logs").resolve()
    try:
        resolved.relative_to(logs_root)
    except ValueError as exc:
        raise ValueError(f"Output must be under logs/: {resolved}") from exc
    for pref in PROTECTED_PREFIXES:
        if not pref.exists():
            continue
        p = pref.resolve()
        if resolved == p:
            raise ValueError(f"Protected output path: {resolved}")
        try:
            resolved.relative_to(p)
            raise ValueError(f"Protected output path: {resolved}")
        except ValueError:
            pass


def _parse_ring_key(ring_key: str) -> tuple[str, int]:
    tid, rr = ring_key.split("/")
    return tid, int(rr.lstrip("r"))


def _ring_dir(root: Path, ring_key: str) -> Path:
    tid, rid = _parse_ring_key(ring_key)
    return root / tid / f"r{rid}"


def _run(cmd: list[str], log_path: Path | None = None, timeout_sec: float = 1800.0) -> None:
    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(proc.stdout or "", encoding="utf-8")
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _ensure_param(path: Path, default_path: Path) -> None:
    if path.exists():
        return
    if not default_path.exists():
        raise FileNotFoundError(f"Missing default parameter file: {default_path}")
    path.write_text(default_path.read_text(encoding="utf-8"), encoding="utf-8")


def _stage_ring(ring_key: str) -> Path:
    tid, _ = _parse_ring_key(ring_key)
    family = int(tid.split("-")[0])
    src_root = SRC_T123 if family in (1, 2, 3) else SRC_T45
    src = _ring_dir(src_root, ring_key)
    if not src.exists():
        raise FileNotFoundError(f"Missing source ring for pilot: {src}")
    dst = _ring_dir(RUN_ROOT, ring_key)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    _ensure_param(dst / "parameters_detection.json", DET_DEFAULT)
    _ensure_param(dst / "parameters_segmentation.json", SEG_DEFAULT)
    return dst


def _rotation_id_map(shift: int) -> dict[int, int]:
    ordered = [2, 3, 4, 5, 6, 7]
    shift = int(shift) % len(ordered)
    mp = {1: 1}
    for i, v in enumerate(ordered):
        mp[v] = ordered[(i + shift) % len(ordered)]
    return mp


def _compute_miou_from_df(df: pd.DataFrame) -> tuple[float | None, float | None]:
    if "segment" not in df.columns or "pred" not in df.columns:
        return None, None
    tmp = df[["segment", "pred"]].dropna(subset=["segment"])
    if tmp.empty:
        return None, None
    gt = tmp["segment"].astype(int).to_numpy()
    pred = tmp["pred"].astype(int).to_numpy()
    valid = (gt >= 0) & (gt <= 7) & (pred >= 0) & (pred <= 7)
    gt = gt[valid]
    pred = pred[valid]
    if gt.size == 0:
        return None, None
    labels = sorted(set(gt.tolist()) | set(pred.tolist()))
    miou = float(jaccard_score(gt, pred, average=None, labels=labels, zero_division=0).mean())
    oa = float((gt == pred).mean())
    return miou, oa


def _boundary_gap_cv(bnd_json: Path, ring_height: int) -> float:
    if not bnd_json.exists():
        return float("nan")
    data = json.loads(bnd_json.read_text(encoding="utf-8"))
    entries = data.get("0") or data.get(0) or []
    ys = sorted(float(e.get("y", 0.0)) for e in entries)
    if len(ys) < 2:
        return float("nan")
    diffs = np.array([(ys[(i + 1) % len(ys)] - ys[i]) % float(ring_height) for i in range(len(ys))], dtype=float)
    return float(np.std(diffs) / (np.mean(diffs) + 1e-9))


def _struct_missing_ids_n(meta_path: Path) -> float:
    if not meta_path.exists():
        return 0.0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cap = meta.get("completion_after_projection", {})
    miss = cap.get("missing_ids_before", [])
    if isinstance(miss, list):
        return float(len(miss))
    return 0.0


def _segment_balance(pred_values: pd.Series) -> tuple[float, float, float, float, float]:
    pred = pred_values[(pred_values >= 1) & (pred_values <= 7)]
    if pred.empty:
        return 0.0, 0.0, 1.0, 1.0, 0.0
    counts = pred.value_counts().reindex(range(1, 8), fill_value=0).astype(float).to_numpy()
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(7.0))
    present_ratio = float((counts > 0).mean())
    cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    max_share = float(counts.max() / (counts.sum() + 1e-9))
    balance_raw = 1.5 * present_ratio + entropy - 0.35 * cv - 0.5 * max(0.0, max_share - 0.45)
    balance_norm = float(np.clip(balance_raw / 2.5, 0.0, 1.0))
    return present_ratio, entropy, cv, max_share, balance_norm


def _score_current_proxy(row: dict[str, Any]) -> float:
    v = CURRENT_PROXY["intercept"]
    v += CURRENT_PROXY["struct_missing_ids_before_n"] * float(row["struct_missing_ids_before_n"])
    v += CURRENT_PROXY["depth_row_nonempty_ratio_audit"] * float(row["depth_row_nonempty_ratio_audit"])
    v += CURRENT_PROXY["geom_boundary_gap_cv"] * float(row["geom_boundary_gap_cv"])
    return float(v)


def _score_intrinsic_composite(row: dict[str, Any]) -> float:
    b = float(row["balance_norm"])
    d = float(np.clip(row["depth_row_nonempty_ratio_audit"], 0.0, 1.0))
    g = float(np.clip(1.0 - min(max(row["geom_boundary_gap_cv"], 0.0), 1.0), 0.0, 1.0))
    s = float(np.clip(1.0 - min(max(row["struct_missing_ids_before_n"] / 3.0, 0.0), 1.0), 0.0, 1.0))
    return float(0.40 * b + 0.25 * d + 0.20 * g + 0.15 * s)


def _build_cfg(anchor_frac: float, parity: int, half_window: float = 0.04) -> KCfg:
    a = float(np.clip(anchor_frac, 0.02, 0.98))
    low = float(max(0.01, a - half_window))
    high = float(min(0.99, a + half_window))
    if high <= low:
        high = min(0.99, low + 0.03)
    return KCfg(anchor_frac=a, low_frac=low, high_frac=high, low_parity=int(parity))


def _tier0_cfgs() -> list[KCfg]:
    anchors = [0.10, 0.26, 0.42, 0.58, 0.74, 0.90]
    return [_build_cfg(a, p, half_window=0.04) for a in anchors for p in (0, 1)]


def _tier1_cfgs(top_anchor_rows: pd.DataFrame) -> list[KCfg]:
    out: list[KCfg] = []
    for _, r in top_anchor_rows.iterrows():
        a = float(r["anchor_frac"])
        p = int(r["low_parity"])
        for d in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02):
            out.append(_build_cfg(a + d, p, half_window=0.03))
            out.append(_build_cfg(a + d, 1 - p, half_window=0.03))
    return out


def _tier2_cfgs() -> list[KCfg]:
    anchors = np.linspace(0.04, 0.96, 24)
    return [_build_cfg(float(a), p, half_window=0.03) for a in anchors for p in (0, 1)]


def _tier3_cfgs(best_row: pd.Series) -> list[KCfg]:
    a = float(best_row["anchor_frac"])
    p = int(best_row["low_parity"])
    out = []
    for d in (-0.01, -0.006, -0.003, 0.0, 0.003, 0.006, 0.01):
        out.append(_build_cfg(a + d, p, half_window=0.025))
        out.append(_build_cfg(a + d, 1 - p, half_window=0.025))
    return out


def _run_cfg_for_ring(
    ring_key: str,
    ring_dir: Path,
    cfg: KCfg,
    *,
    depth_row_nonempty_ratio_audit: float,
    tier: str,
) -> list[dict[str, Any]]:
    tid, rid = _parse_ring_key(ring_key)
    det_path = ring_dir / "parameters_detection.json"
    det = json.loads(det_path.read_text(encoding="utf-8"))
    det["detector_mode"] = "single_ring_regular_prior"
    det["k_anchor_semantics"] = "center"
    det["ring_topology"] = "k_bearing"
    det["segment_count"] = 7
    det["enabled_blocks"] = list(det.get("enabled_blocks", KB7_BLOCKS))
    det["per_ring_offsets"] = det.get("per_ring_offsets", {"0": dict(KB7_OFFSETS)})
    det["regular_k_prior_low_frac"] = float(cfg.low_frac)
    det["regular_k_prior_high_frac"] = float(cfg.high_frac)
    det["regular_k_prior_low_ring_parity"] = int(cfg.low_parity)
    det_path.write_text(json.dumps(det, indent=2) + "\n", encoding="utf-8")

    tag = cfg.tag
    _run([str(VENV_PY), str(DET_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"det_{tier}_{tag}.log")
    _run([str(VENV_PY), str(SEG_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"seg_{tier}_{tag}.log")

    src_plus_final = ring_dir / "final_plus.csv"
    src_minus_final = ring_dir / "final_minus.csv"
    src_plus_bnd = ring_dir / "boundaries_per_ring_direction_plus.json"
    src_minus_bnd = ring_dir / "boundaries_per_ring_direction_minus.json"
    if not src_plus_final.exists():
        src_plus_final = ring_dir / "final.csv"
    if not src_minus_final.exists():
        src_minus_final = src_plus_final
    if not src_plus_bnd.exists():
        src_plus_bnd = ring_dir / "boundaries_per_ring.json"
    if not src_minus_bnd.exists():
        src_minus_bnd = src_plus_bnd

    ring_height = int(np.load(ring_dir / "depth_map.npy").shape[0])
    struct_missing = _struct_missing_ids_n(ring_dir / "segment_completion_meta_segmentation.json")
    rows: list[dict[str, Any]] = []
    for direction, final_path, bnd_path in (("plus", src_plus_final, src_plus_bnd), ("minus", src_minus_final, src_minus_bnd)):
        base_df = pd.read_csv(final_path)
        bcv = _boundary_gap_cv(bnd_path, ring_height=ring_height)
        for shift in ROTATIONS:
            df = base_df.copy()
            if "pred" in df.columns:
                rot_map = _rotation_id_map(shift)
                df["pred"] = df["pred"].map(lambda x: rot_map.get(int(x), int(x)) if pd.notna(x) else x)
            miou, oa = _compute_miou_from_df(df)
            present_ratio, entropy, cv, max_share, balance_norm = _segment_balance(df["pred"].astype(int) if "pred" in df.columns else pd.Series(dtype=int))
            row = {
                "ring_key": ring_key,
                "tier": tier,
                "det_tag": tag,
                "anchor_frac": float(cfg.anchor_frac),
                "low_frac": float(cfg.low_frac),
                "high_frac": float(cfg.high_frac),
                "low_parity": int(cfg.low_parity),
                "branch": direction,
                "rotation_shift": int(shift),
                "branch_is_minus": 1.0 if direction == "minus" else 0.0,
                "rotation_shift_num": float(shift),
                "struct_missing_ids_before_n": float(struct_missing),
                "depth_row_nonempty_ratio_audit": float(depth_row_nonempty_ratio_audit),
                "geom_boundary_gap_cv": float(bcv),
                "present_ratio": float(present_ratio),
                "entropy": float(entropy),
                "cv": float(cv),
                "max_share": float(max_share),
                "balance_norm": float(balance_norm),
                "miou": miou,
                "oa": oa,
            }
            row["proxy_current"] = _score_current_proxy(row)
            row["proxy_intrinsic_composite"] = _score_intrinsic_composite(row)
            row["proxy_oracle_diagnostic"] = float(miou) if miou is not None else float("nan")
            rows.append(row)
    return rows


def _pick_top_anchor_rows(cands: pd.DataFrame, proxy_col: str, topk: int = 3) -> pd.DataFrame:
    top = (
        cands.sort_values([proxy_col, "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True])
        [["anchor_frac", "low_parity"]]
        .drop_duplicates()
        .head(topk)
    )
    return top.reset_index(drop=True)


def _build_hard_pilot() -> pd.DataFrame:
    panel = pd.read_csv(PANEL)[["ring_key", "tunnel_id", "ring_id", "family"]]
    sb = pd.read_csv(SCOREBOARD)[["ring_key", "stabilised_miou", "intrinsic_final_miou"]]
    df = panel.merge(sb, on="ring_key", how="left")
    rows = []
    for reason, rings in PILOT_GROUPS.items():
        for ring in rings:
            rec = df[df["ring_key"].eq(ring)]
            if rec.empty:
                raise RuntimeError(f"Missing ring in panel/scoreboard: {ring}")
            r = rec.iloc[0].to_dict()
            r["reason_group"] = reason
            r["is_depth_risk_control"] = ring in DEPTH_RISK_CONTROLS
            rows.append(r)
    out = pd.DataFrame(rows).drop_duplicates(subset=["ring_key"]).reset_index(drop=True)
    out["pilot_idx"] = range(1, len(out) + 1)
    out = out[["pilot_idx", "ring_key", "tunnel_id", "ring_id", "family", "stabilised_miou", "intrinsic_final_miou", "reason_group", "is_depth_risk_control"]]
    return out


def _select_by_proxy(cands: pd.DataFrame, proxy_col: str) -> pd.Series:
    valid = cands[cands[proxy_col].notna()].copy()
    if valid.empty:
        raise RuntimeError(f"No valid candidates for {proxy_col}")
    return valid.sort_values([proxy_col, "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).iloc[0]


def _fit_retrained_proxy(cands: pd.DataFrame) -> pd.DataFrame:
    out = cands.copy()
    feat_cols = [
        "struct_missing_ids_before_n",
        "depth_row_nonempty_ratio_audit",
        "geom_boundary_gap_cv",
        "present_ratio",
        "entropy",
        "cv",
        "max_share",
        "balance_norm",
        "anchor_frac",
        "low_parity",
        "branch_is_minus",
        "rotation_shift_num",
    ]
    train = out[pd.to_numeric(out["miou"], errors="coerce").notna()].copy()
    if len(train) < 20:
        out["proxy_ridge_retrained_hardpilot"] = np.nan
        return out
    X = train[feat_cols]
    y = pd.to_numeric(train["miou"], errors="coerce")
    model = Pipeline(
        steps=[
            ("prep", ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), feat_cols)])),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    model.fit(X, y)
    out["proxy_ridge_retrained_hardpilot"] = model.predict(out[feat_cols])
    return out


def _build_failure_mode(ring_key: str, cands: pd.DataFrame, selected_row: pd.Series, proxy_col: str, stabilised_miou: float) -> dict[str, Any]:
    ranked = cands.sort_values([proxy_col, "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
    top = float(ranked.iloc[0][proxy_col])
    second = float(ranked.iloc[1][proxy_col]) if len(ranked) > 1 else top
    margin = top - second
    oracle = float(pd.to_numeric(cands["miou"], errors="coerce").max())
    selected_miou = float(selected_row["miou"]) if pd.notna(selected_row["miou"]) else float("nan")
    categories: list[str] = []
    if margin < 0.01:
        categories.append("weak_proxy_margin")
    if np.isfinite(oracle) and np.isfinite(selected_miou):
        if oracle < 0.3:
            categories.append("search_coverage_low")
        elif oracle - selected_miou > 0.15:
            categories.append("proxy_ranking_failure")
    if np.isfinite(stabilised_miou) and np.isfinite(selected_miou) and selected_miou < stabilised_miou:
        categories.append("regression_below_stabilised")
    if not categories:
        categories.append("pass_or_no_clear_failure")
    return {
        "ring_key": ring_key,
        "proxy": proxy_col,
        "proxy_margin_top2": float(margin),
        "oracle_miou": oracle if np.isfinite(oracle) else None,
        "selected_miou": selected_miou if np.isfinite(selected_miou) else None,
        "categories": categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Hard-case finer search and proxy-variant comparison (no preprocessing rerun).")
    parser.add_argument("--max-rings", type=int, default=0, help="Debug cap; 0 means full hard pilot.")
    args = parser.parse_args()

    _assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    depth = pd.read_csv(DEPTH_AUDIT)[["ring_key", "row_nonempty_ratio", "depth_gate_pass", "depth_gate_reason"]]
    pilot = _build_hard_pilot().merge(depth, on="ring_key", how="left")
    if args.max_rings > 0:
        pilot = pilot.head(int(args.max_rings)).copy()
    pilot.to_csv(RUN_ROOT / "pilot_ring_list.csv", index=False)

    all_candidates: list[dict[str, Any]] = []
    proxy_score_rows: list[dict[str, Any]] = []
    failure_modes: list[dict[str, Any]] = []
    proxy_cols = ["proxy_current", "proxy_intrinsic_composite", "proxy_oracle_diagnostic", "proxy_ridge_retrained_hardpilot"]

    for rr in pilot.itertuples(index=False):
        ring_key = str(rr.ring_key)
        ring_dir = _stage_ring(ring_key)
        tried: set[tuple[float, float, float, int]] = set()
        ring_rows: list[dict[str, Any]] = []

        def run_cfgs(cfgs: list[KCfg], tier: str) -> pd.DataFrame:
            rows: list[dict[str, Any]] = []
            for cfg in cfgs:
                key = (round(cfg.anchor_frac, 6), round(cfg.low_frac, 6), round(cfg.high_frac, 6), int(cfg.low_parity))
                if key in tried:
                    continue
                tried.add(key)
                rows.extend(
                    _run_cfg_for_ring(
                        ring_key,
                        ring_dir,
                        cfg,
                        depth_row_nonempty_ratio_audit=float(rr.row_nonempty_ratio) if pd.notna(rr.row_nonempty_ratio) else 0.0,
                        tier=tier,
                    )
                )
            return pd.DataFrame(rows)

        tier0 = run_cfgs(_tier0_cfgs(), tier="tier0")
        if tier0.empty:
            raise RuntimeError(f"No tier0 candidates for {ring_key}")
        ring_rows.extend(tier0.to_dict(orient="records"))

        top3 = _pick_top_anchor_rows(tier0, "proxy_current", topk=3)
        tier1 = run_cfgs(_tier1_cfgs(top3), tier="tier1")
        if not tier1.empty:
            ring_rows.extend(tier1.to_dict(orient="records"))

        merged = pd.DataFrame(ring_rows)
        ranked_current = merged.sort_values(["proxy_current", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
        margin = float(ranked_current.iloc[0]["proxy_current"] - ranked_current.iloc[1]["proxy_current"]) if len(ranked_current) > 1 else 0.0
        weak_or_incomplete = margin < 0.01 or float(ranked_current.iloc[0]["struct_missing_ids_before_n"]) > 0
        if weak_or_incomplete:
            tier2 = run_cfgs(_tier2_cfgs(), tier="tier2")
            if not tier2.empty:
                ring_rows.extend(tier2.to_dict(orient="records"))

        merged2 = pd.DataFrame(ring_rows).sort_values(["proxy_current", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
        top_before = float(merged2.iloc[0]["proxy_current"])
        tier3 = run_cfgs(_tier3_cfgs(merged2.iloc[0]), tier="tier3")
        if not tier3.empty:
            candidate_t3 = pd.concat([merged2, tier3], ignore_index=True)
            candidate_t3 = candidate_t3.sort_values(["proxy_current", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
            top_after = float(candidate_t3.iloc[0]["proxy_current"])
            if top_after - top_before >= 0.005:
                ring_rows = candidate_t3.to_dict(orient="records")

        ring_df = pd.DataFrame(ring_rows)
        all_candidates.extend(ring_df.to_dict(orient="records"))
        shutil.rmtree(ring_dir, ignore_errors=True)

    cand_df = pd.DataFrame(all_candidates)
    cand_df = _fit_retrained_proxy(cand_df)
    cand_df.to_csv(RUN_ROOT / "candidate_scores.csv", index=False)

    scoreboard_rows: list[dict[str, Any]] = []
    for rr in pilot.itertuples(index=False):
        ring_key = str(rr.ring_key)
        ring_c = cand_df[cand_df["ring_key"].eq(ring_key)].copy()
        oracle = float(pd.to_numeric(ring_c["miou"], errors="coerce").max()) if not ring_c.empty else float("nan")
        for proxy_col in proxy_cols:
            if proxy_col == "proxy_ridge_retrained_hardpilot" and proxy_col not in ring_c.columns:
                continue
            sel = _select_by_proxy(ring_c, proxy_col)
            selected_miou = float(sel["miou"]) if pd.notna(sel["miou"]) else float("nan")
            stabilised = float(rr.stabilised_miou) if pd.notna(rr.stabilised_miou) else float("nan")
            production = float(rr.intrinsic_final_miou) if pd.notna(rr.intrinsic_final_miou) else float("nan")
            row = {
                "ring_key": ring_key,
                "tunnel_id": rr.tunnel_id,
                "family": int(rr.family),
                "reason_group": rr.reason_group,
                "is_depth_risk_control": bool(rr.is_depth_risk_control),
                "proxy_variant": proxy_col,
                "selected_proxy_score": float(sel[proxy_col]) if pd.notna(sel[proxy_col]) else float("nan"),
                "selected_miou": selected_miou if np.isfinite(selected_miou) else np.nan,
                "stabilised_miou": stabilised if np.isfinite(stabilised) else np.nan,
                "production_intrinsic_miou": production if np.isfinite(production) else np.nan,
                "oracle_miou": oracle if np.isfinite(oracle) else np.nan,
                "oracle_gap": (oracle - selected_miou) if np.isfinite(oracle) and np.isfinite(selected_miou) else np.nan,
                "lift_vs_stabilised": (selected_miou - stabilised) if np.isfinite(selected_miou) and np.isfinite(stabilised) else np.nan,
                "lift_vs_production_intrinsic": (selected_miou - production) if np.isfinite(selected_miou) and np.isfinite(production) else np.nan,
                "selected_det_tag": str(sel["det_tag"]),
                "selected_tier": str(sel["tier"]),
                "selected_branch": str(sel["branch"]),
                "selected_rotation_shift": int(sel["rotation_shift"]),
                "selected_anchor_frac": float(sel["anchor_frac"]),
                "selected_low_parity": int(sel["low_parity"]),
            }
            scoreboard_rows.append(row)
            failure_modes.append(
                _build_failure_mode(
                    ring_key=ring_key,
                    cands=ring_c,
                    selected_row=sel,
                    proxy_col=proxy_col,
                    stabilised_miou=stabilised,
                )
            )

    score_df = pd.DataFrame(scoreboard_rows).sort_values(["proxy_variant", "family", "ring_key"]).reset_index(drop=True)
    score_df.to_csv(RUN_ROOT / "pilot_scoreboard.csv", index=False)
    with (RUN_ROOT / "search_vs_proxy_failure_modes.jsonl").open("w", encoding="utf-8") as f:
        for rec in failure_modes:
            f.write(json.dumps(rec) + "\n")

    cmp_df = score_df[
        [
            "ring_key",
            "proxy_variant",
            "selected_miou",
            "oracle_miou",
            "oracle_gap",
            "stabilised_miou",
            "production_intrinsic_miou",
            "lift_vs_stabilised",
            "lift_vs_production_intrinsic",
            "is_depth_risk_control",
        ]
    ].copy()
    cmp_df["regression_vs_stabilised"] = cmp_df["lift_vs_stabilised"] < 0
    cmp_df["regression_vs_production"] = cmp_df["lift_vs_production_intrinsic"] < 0
    cmp_df.to_csv(RUN_ROOT / "proxy_comparison.csv", index=False)

    summary_rows = []
    for proxy, g in cmp_df.groupby("proxy_variant", sort=True):
        nonrisk = g[~g["is_depth_risk_control"].astype(bool)].copy()
        summary_rows.append(
            {
                "proxy_variant": proxy,
                "n_rows": int(len(g)),
                "mean_selected_miou": float(pd.to_numeric(g["selected_miou"], errors="coerce").mean()),
                "mean_oracle_miou": float(pd.to_numeric(g["oracle_miou"], errors="coerce").mean()),
                "mean_oracle_gap": float(pd.to_numeric(g["oracle_gap"], errors="coerce").mean()),
                "regressions_vs_stabilised": int((pd.to_numeric(g["lift_vs_stabilised"], errors="coerce") < 0).sum()),
                "regressions_vs_production": int((pd.to_numeric(g["lift_vs_production_intrinsic"], errors="coerce") < 0).sum()),
                "mean_selected_miou_non_depth_risk": float(pd.to_numeric(nonrisk["selected_miou"], errors="coerce").mean()) if len(nonrisk) > 0 else None,
                "mean_production_non_depth_risk": float(pd.to_numeric(nonrisk["production_intrinsic_miou"], errors="coerce").mean()) if len(nonrisk) > 0 else None,
            }
        )
    summary_tbl = pd.DataFrame(summary_rows).sort_values("mean_selected_miou", ascending=False).reset_index(drop=True)
    winner = summary_tbl.iloc[0].to_dict() if not summary_tbl.empty else {}
    scale_allowed = False
    if winner:
        scale_allowed = bool(
            float(winner.get("mean_selected_miou_non_depth_risk", -1.0)) > float(winner.get("mean_production_non_depth_risk", 1e9))
            and int(winner.get("regressions_vs_stabilised", 999)) < int(cmp_df[cmp_df["proxy_variant"].eq("proxy_current")]["regression_vs_stabilised"].sum())
        )
    summary = {
        "n_rings": int(pilot["ring_key"].nunique()),
        "candidate_rows": int(len(cand_df)),
        "proxy_summary": summary_tbl.to_dict(orient="records"),
        "winner_proxy": winner.get("proxy_variant"),
        "winner_stats": winner,
        "scale_decision": "scale_allowed" if scale_allowed else "stop_and_rework",
        "decision_reason": "allowed only if non-depth-risk mean beats production and regressions drop vs current proxy",
        "notes": [
            "proxy_oracle_diagnostic is non-deployable and for diagnosis only.",
            "proxy_ridge_retrained_hardpilot is in-sample exploratory unless later validated on held-out rings.",
        ],
    }
    (RUN_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


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
from sklearn.metrics import jaccard_score

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PY = REPO_ROOT / "venv" / "bin" / "python"
DET_CLI = REPO_ROOT / "agents" / "2_detection" / "2_detection.py"
SEG_CLI = REPO_ROOT / "agents" / "3_segmentation" / "segmentation.py"

RUN_ROOT = REPO_ROOT / "logs" / "v5_adaptive_proxy_pilot_v1"
PANEL = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
SCOREBOARD = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard.csv"
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

DEPTH_RISK_CONTROLS = {"4-6/r276", "5-6/r285"}
ROTATIONS = [0, 1, 4]
KB7_BLOCKS = ["K", "B1", "A1", "A2", "A3", "A4", "B2"]
KB7_OFFSETS = {"K": 0.0, "B1": 181.9, "A1": 727.5, "A2": 1273.2, "A3": -1636.9, "A4": -1091.3, "B2": -545.6}

# Latest chosen observable proxy.
PROXY_COEF = {
    "intercept": 0.015213,
    "struct_missing_ids_before_n": -0.031164,
    "depth_row_nonempty_ratio_audit": 0.196603,
    "geom_boundary_gap_cv": 0.081910,
}


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


def _run(cmd: list[str], log_path: Path, timeout_sec: float = 1800.0) -> None:
    env = dict(os.environ)
    env["INTRINSIC_PARAMS_BASE_DIR_ONLY"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
    if proc.returncode != 0:
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
    # K=1, rotating the six non-K classes: B1,A1,A2,A3,A4,B2 -> 2..7.
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


def _proxy_score(*, struct_missing_ids_before_n: float, depth_row_nonempty_ratio_audit: float, geom_boundary_gap_cv: float) -> float:
    v = PROXY_COEF["intercept"]
    v += PROXY_COEF["struct_missing_ids_before_n"] * float(struct_missing_ids_before_n)
    v += PROXY_COEF["depth_row_nonempty_ratio_audit"] * float(depth_row_nonempty_ratio_audit)
    v += PROXY_COEF["geom_boundary_gap_cv"] * float(geom_boundary_gap_cv)
    return float(v)


def _build_cfg(anchor_frac: float, parity: int) -> KCfg:
    a = float(np.clip(anchor_frac, 0.02, 0.98))
    low = float(max(0.01, a - 0.04))
    high = float(min(0.99, a + 0.04))
    if high <= low:
        high = min(0.99, low + 0.03)
    return KCfg(anchor_frac=a, low_frac=low, high_frac=high, low_parity=int(parity))


def _coarse_cfgs() -> list[KCfg]:
    anchors = [0.10, 0.26, 0.42, 0.58, 0.74, 0.90]
    out: list[KCfg] = []
    for a in anchors:
        for p in (0, 1):
            out.append(_build_cfg(a, p))
    return out


def _choose_action(cands: pd.DataFrame, depth_gate_pass: bool) -> str:
    ranked = cands.sort_values(["proxy_score", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
    top = float(ranked.iloc[0]["proxy_score"])
    second = float(ranked.iloc[1]["proxy_score"]) if len(ranked) > 1 else top
    margin = top - second
    if not bool(depth_gate_pass):
        return "flag_unstable"
    if margin >= 0.03 and top >= 0.18:
        return "refine_top1"
    if margin < 0.01 or top < 0.12:
        return "expand_global"
    return "refine_top3"


def _refine_cfgs(cands: pd.DataFrame, action: str) -> list[KCfg]:
    best = cands.sort_values(["proxy_score", "det_tag"], ascending=[False, True]).copy()
    if action == "expand_global":
        anchors = np.linspace(0.04, 0.96, 12)
        return [_build_cfg(float(a), p) for a in anchors for p in (0, 1)]
    if action == "refine_top1":
        a = float(best.iloc[0]["anchor_frac"])
        anchors = [a - 0.02, a - 0.01, a, a + 0.01, a + 0.02]
        return [_build_cfg(float(v), p) for v in anchors for p in (0, 1)]
    # refine_top3 or flag_unstable fallback to top3 neighborhood.
    top_cfg = best[["anchor_frac", "low_parity"]].drop_duplicates().head(3)
    out: list[KCfg] = []
    for _, r in top_cfg.iterrows():
        a = float(r["anchor_frac"])
        p = int(r["low_parity"])
        for d in (-0.015, 0.0, 0.015):
            out.append(_build_cfg(a + d, p))
            out.append(_build_cfg(a + d, 1 - p))
    return out


def _run_cfg_for_ring(
    ring_key: str,
    ring_dir: Path,
    cfg: KCfg,
    *,
    depth_row_nonempty_ratio_audit: float,
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
    _run([str(VENV_PY), str(DET_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"det_{tag}.log")
    _run([str(VENV_PY), str(SEG_CLI), tid, str(rid), "--data-dir", str(RUN_ROOT)], ring_dir / "logs" / f"seg_{tag}.log")

    # Snapshot outputs per cfg before next cfg overwrites them.
    file_map = {
        "plus_final": ring_dir / f"final_{tag}_plus.csv",
        "minus_final": ring_dir / f"final_{tag}_minus.csv",
        "plus_segments": ring_dir / f"all_segments_{tag}_plus.csv",
        "minus_segments": ring_dir / f"all_segments_{tag}_minus.csv",
        "plus_boundaries": ring_dir / f"boundaries_{tag}_plus.json",
        "minus_boundaries": ring_dir / f"boundaries_{tag}_minus.json",
    }
    src_plus_final = ring_dir / "final_plus.csv"
    src_minus_final = ring_dir / "final_minus.csv"
    src_plus_seg = ring_dir / "all_segments_direction_plus.csv"
    src_minus_seg = ring_dir / "all_segments_direction_minus.csv"
    src_plus_bnd = ring_dir / "boundaries_per_ring_direction_plus.json"
    src_minus_bnd = ring_dir / "boundaries_per_ring_direction_minus.json"
    if not src_plus_final.exists():
        src_plus_final = ring_dir / "final.csv"
    if not src_minus_final.exists():
        src_minus_final = src_plus_final
    if not src_plus_seg.exists():
        src_plus_seg = ring_dir / "all_segments.csv"
    if not src_minus_seg.exists():
        src_minus_seg = src_plus_seg
    if not src_plus_bnd.exists():
        src_plus_bnd = ring_dir / "boundaries_per_ring.json"
    if not src_minus_bnd.exists():
        src_minus_bnd = src_plus_bnd

    shutil.copy2(src_plus_final, file_map["plus_final"])
    shutil.copy2(src_minus_final, file_map["minus_final"])
    shutil.copy2(src_plus_seg, file_map["plus_segments"])
    shutil.copy2(src_minus_seg, file_map["minus_segments"])
    shutil.copy2(src_plus_bnd, file_map["plus_boundaries"])
    shutil.copy2(src_minus_bnd, file_map["minus_boundaries"])

    ring_height = int(np.load(ring_dir / "depth_map.npy").shape[0])
    struct_missing = _struct_missing_ids_n(ring_dir / "segment_completion_meta_segmentation.json")
    rows: list[dict[str, Any]] = []
    for direction, final_key, bnd_key in (
        ("plus", "plus_final", "plus_boundaries"),
        ("minus", "minus_final", "minus_boundaries"),
    ):
        base_df = pd.read_csv(file_map[final_key])
        bcv = _boundary_gap_cv(file_map[bnd_key], ring_height=ring_height)
        for shift in ROTATIONS:
            df = base_df.copy()
            if "pred" in df.columns:
                rot_map = _rotation_id_map(shift)
                df["pred"] = df["pred"].map(lambda x: rot_map.get(int(x), int(x)) if pd.notna(x) else x)
            miou, oa = _compute_miou_from_df(df)
            pscore = _proxy_score(
                struct_missing_ids_before_n=struct_missing,
                depth_row_nonempty_ratio_audit=depth_row_nonempty_ratio_audit,
                geom_boundary_gap_cv=bcv,
            )
            rows.append(
                {
                    "ring_key": ring_key,
                    "det_tag": tag,
                    "anchor_frac": float(cfg.anchor_frac),
                    "low_frac": float(cfg.low_frac),
                    "high_frac": float(cfg.high_frac),
                    "low_parity": int(cfg.low_parity),
                    "branch": direction,
                    "rotation_shift": int(shift),
                    "struct_missing_ids_before_n": float(struct_missing),
                    "depth_row_nonempty_ratio_audit": float(depth_row_nonempty_ratio_audit),
                    "geom_boundary_gap_cv": float(bcv),
                    "proxy_score": float(pscore),
                    "miou": miou,
                    "oa": oa,
                }
            )
    return rows


def _build_pilot_list() -> pd.DataFrame:
    panel = pd.read_csv(PANEL)
    score = pd.read_csv(SCOREBOARD)[["ring_key", "stabilised_miou"]]
    depth = pd.read_csv(DEPTH_AUDIT)[["ring_key", "depth_gate_pass", "row_nonempty_ratio", "depth_gate_reason"]]
    df = panel.merge(score, on="ring_key", how="left").merge(depth, on="ring_key", how="left")
    df["depth_pass"] = df["depth_gate_pass"].fillna(False).astype(bool)
    df["stabilised_miou"] = pd.to_numeric(df["stabilised_miou"], errors="coerce")
    df["stabilised_ge_05"] = df["stabilised_miou"].fillna(-1.0) >= 0.5
    df["is_depth_risk_control"] = df["ring_key"].astype(str).isin(DEPTH_RISK_CONTROLS)
    pick_rows: list[pd.Series] = []
    for tunnel_id, g in df.groupby("tunnel_id", sort=True):
        g2 = g.sort_values(
            ["depth_pass", "stabilised_ge_05", "stabilised_miou", "ring_id"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        pick_rows.append(g2.iloc[0])
    out = pd.DataFrame(pick_rows).sort_values(["family", "tunnel_id"]).reset_index(drop=True)
    out["pilot_idx"] = range(1, len(out) + 1)
    return out[
        [
            "pilot_idx",
            "ring_key",
            "tunnel_id",
            "ring_id",
            "family",
            "stabilised_miou",
            "depth_pass",
            "row_nonempty_ratio",
            "depth_gate_reason",
            "is_depth_risk_control",
        ]
    ]


def _select_final(cands: pd.DataFrame) -> pd.Series:
    valid = cands[cands["proxy_score"].notna()].copy()
    if valid.empty:
        raise RuntimeError("No valid proxy-scored candidates")
    return valid.sort_values(
        ["proxy_score", "det_tag", "branch", "rotation_shift"],
        ascending=[False, True, True, True],
    ).iloc[0]


def _failure_explanation(
    *,
    ring_row: pd.Series,
    candidates: pd.DataFrame,
    selected: pd.Series,
) -> dict[str, Any]:
    categories: list[str] = []
    ranked = candidates.sort_values(["proxy_score", "det_tag", "branch", "rotation_shift"], ascending=[False, True, True, True]).reset_index(drop=True)
    top = float(ranked.iloc[0]["proxy_score"])
    second = float(ranked.iloc[1]["proxy_score"]) if len(ranked) > 1 else top
    margin = top - second
    if not bool(ring_row["depth_pass"]):
        categories.append("depth_quality_failure")
    if margin < 0.01:
        categories.append("weak_proxy_margin")
    if float(selected.get("struct_missing_ids_before_n", 0.0)) > 0:
        categories.append("structural_incompleteness")
    if float(selected.get("geom_boundary_gap_cv", 0.0)) > 0.35:
        categories.append("boundary_spacing_ambiguity")
    top5 = ranked.head(min(5, len(ranked)))
    if top5["branch"].nunique() > 1 or top5["rotation_shift"].nunique() > 1:
        if margin < 0.02:
            categories.append("branch_or_rotation_ambiguity")
    stab = float(ring_row["stabilised_miou"]) if pd.notna(ring_row["stabilised_miou"]) else float("nan")
    final_miou = float(selected["miou"]) if pd.notna(selected["miou"]) else float("nan")
    proxy_failed = bool(np.isfinite(stab) and np.isfinite(final_miou) and final_miou < stab)
    if proxy_failed:
        categories.append("proxy_regression_below_stabilised")
    if not categories:
        categories.append("pass_or_no_clear_failure")
    return {
        "ring_key": str(ring_row["ring_key"]),
        "tunnel_id": str(ring_row["tunnel_id"]),
        "adaptive_action": str(ring_row.get("adaptive_action", "")),
        "proxy_margin_top2": float(margin),
        "categories": categories,
        "proxy_failed": proxy_failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive intrinsic proxy pilot (GT-free runtime, GT only for offline eval).")
    parser.add_argument("--max-rings", type=int, default=0, help="Optional cap for debugging; 0 means all pilot rings.")
    args = parser.parse_args()

    _assert_writable(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    pilot = _build_pilot_list()
    if args.max_rings > 0:
        pilot = pilot.head(int(args.max_rings)).copy()
    pilot.to_csv(RUN_ROOT / "pilot_ring_list.csv", index=False)

    all_candidates: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    explanations: list[dict[str, Any]] = []
    for rr in pilot.itertuples(index=False):
        ring_key = str(rr.ring_key)
        ring_dir = _stage_ring(ring_key)
        tried: set[tuple[float, float, float, int]] = set()

        def run_cfgs(cfgs: list[KCfg]) -> pd.DataFrame:
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
                    )
                )
            return pd.DataFrame(rows)

        coarse = run_cfgs(_coarse_cfgs())
        if coarse.empty:
            raise RuntimeError(f"No coarse candidates for {ring_key}")
        action = _choose_action(coarse, bool(rr.depth_pass))
        refine = run_cfgs(_refine_cfgs(coarse, action))
        cands = pd.concat([coarse, refine], ignore_index=True)
        cands["adaptive_action"] = action
        cands["tunnel_id"] = rr.tunnel_id
        cands["family"] = int(rr.family)
        all_candidates.extend(cands.to_dict(orient="records"))

        selected = _select_final(cands)
        final_miou = float(selected["miou"]) if pd.notna(selected["miou"]) else float("nan")
        stab = float(rr.stabilised_miou) if pd.notna(rr.stabilised_miou) else float("nan")
        proxy_failed = bool(np.isfinite(stab) and np.isfinite(final_miou) and final_miou < stab)
        exp = _failure_explanation(
            ring_row=pd.Series(
                {
                    "ring_key": rr.ring_key,
                    "tunnel_id": rr.tunnel_id,
                    "depth_pass": rr.depth_pass,
                    "stabilised_miou": rr.stabilised_miou,
                    "adaptive_action": action,
                }
            ),
            candidates=cands,
            selected=selected,
        )
        explanations.append(exp)
        score_rows.append(
            {
                "ring_key": ring_key,
                "tunnel_id": rr.tunnel_id,
                "family": int(rr.family),
                "is_depth_risk_control": bool(rr.is_depth_risk_control),
                "depth_gate_pass": bool(rr.depth_pass),
                "stabilised_miou": stab,
                "final_intrinsic_miou": final_miou,
                "proxy_score": float(selected["proxy_score"]),
                "selected_det_tag": str(selected["det_tag"]),
                "selected_branch": str(selected["branch"]),
                "selected_rotation_shift": int(selected["rotation_shift"]),
                "adaptive_action": action,
                "proxy_failed": proxy_failed,
                "failure_mode": ";".join(exp["categories"]),
            }
        )

    cand_df = pd.DataFrame(all_candidates)
    cand_df.to_csv(RUN_ROOT / "pilot_candidates.csv", index=False)
    score_df = pd.DataFrame(score_rows).sort_values(["family", "tunnel_id"]).reset_index(drop=True)
    score_df.to_csv(RUN_ROOT / "pilot_scoreboard.csv", index=False)

    with (RUN_ROOT / "failure_mode_explanations.jsonl").open("w", encoding="utf-8") as f:
        for rec in explanations:
            f.write(json.dumps(rec) + "\n")

    non_risk = score_df[~score_df["is_depth_risk_control"].astype(bool)].copy()
    pass_mask = (
        (non_risk["final_intrinsic_miou"] >= 0.5)
        | ((non_risk["stabilised_miou"] < 0.5) & (non_risk["final_intrinsic_miou"] >= non_risk["stabilised_miou"]))
    )
    summary = {
        "n_pilot_rings": int(len(score_df)),
        "n_non_depth_risk": int(len(non_risk)),
        "n_depth_risk_controls": int(score_df["is_depth_risk_control"].astype(bool).sum()),
        "proxy_failed_count": int(score_df["proxy_failed"].astype(bool).sum()),
        "mean_stabilised_miou": float(score_df["stabilised_miou"].mean()),
        "mean_final_intrinsic_miou": float(score_df["final_intrinsic_miou"].mean()),
        "pilot_gate_pass_non_depth_risk": bool(pass_mask.all()) if len(non_risk) > 0 else False,
        "pilot_gate_failed_rings": non_risk.loc[~pass_mask, "ring_key"].astype(str).tolist(),
        "depth_risk_control_rows": score_df[score_df["is_depth_risk_control"].astype(bool)][
            ["ring_key", "stabilised_miou", "final_intrinsic_miou", "failure_mode"]
        ].to_dict(orient="records"),
    }
    (RUN_ROOT / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

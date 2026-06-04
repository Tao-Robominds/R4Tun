"""Train Proxy4Tun axis Ridge proxies (L, K, L+K concat, L+K joint) on BO trial CSVs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from lib.ceiling_gate import REPO_ROOT
from lib.v5_relative_proxy import ring_selection_metrics

L_PROXY_ALLOWLIST = [
    "arc_width_entropy",
    "det_min_y_gap_px",
    "det_y_order_consistency",
    "det_y_coverage_pct",
    "hough_oblique_threshold",
    "merge_distance_threshold",
    "single_ring_visual_slot_snap_px",
    "slot_inset_y",
    "segmentation_slot_inset_y",
    "r_surface_min_frac",
    "n_reclassified_by_r_filter",
    "det_k_confidence_avg",
    "det_block_count_per_ring",
    "finite_ratio",
    "row_nonempty_ratio",
]

K_PROXY_ALLOWLIST = [
    "k_y_frac",
    "layout_k_center_norm",
    "k_anchor_dist_sam_frac",
    "k_anchor_dist_line_frac",
    "line_detection_confidence_K",
    "rho_K",
    "rho_AB",
    "det_k_confidence_avg",
    "det_k_count_match",
    "det_min_y_gap_px",
    "det_y_coverage_pct",
    "finite_ratio",
    "row_nonempty_ratio",
    "arc_width_entropy",
]

LK_PROXY_ALLOWLIST = sorted(set(L_PROXY_ALLOWLIST) | set(K_PROXY_ALLOWLIST))

V5_ENRICHED_ALLOWLIST = [
    "v5_depth_row_nonempty_ratio_audit",
    "v5_finite_ratio_audit",
    "v5_present_ratio",
    "v5_entropy",
    "v5_cv",
    "v5_max_share",
    "v5_balance_norm",
    "v5_struct_missing_ids_before_n",
    "v5_geom_boundary_gap_cv",
    "v5_geom_boundary_min_gap_frac",
    "v5_geom_boundary_max_gap_frac",
    "v5_geom_boundary_mean_gap_frac",
    "v5_n_boundaries",
    "v5_S_continuity",
    "v5_S_K",
    "v5_S_spacing",
    "v5_S_layout_coverage",
    "v5_S_boundary",
]

SEG_ENRICHED_ALLOWLIST = [
    "seg_segment_type_completeness",
    "seg_ring_completeness_avg",
    "seg_mask_coverage_pct",
    "seg_k_size_ratio",
    "seg_block_size_variance_ratio",
    "seg_groove_score",
]

INTRINSIC_ENRICHED_ALLOWLIST = [
    "feat_intrinsic_arc_width_entropy",
    "feat_intrinsic_n_reclassified_by_r_filter",
    "param_k_y_frac",
    "param_hough_oblique_threshold",
]

LK_ENRICHED_ALLOWLIST = sorted(
    set(LK_PROXY_ALLOWLIST)
    | set(V5_ENRICHED_ALLOWLIST)
    | set(SEG_ENRICHED_ALLOWLIST)
    | set(INTRINSIC_ENRICHED_ALLOWLIST)
)

SKIP_COLS = frozenset({
    "trial_id",
    "tunnel_id",
    "ring_id",
    "case_id",
    "kind",
    "experience_stream",
    "axis_source",
    "order_branch",
    "agent_error",
    "per_ring_offsets",
    "search_x",
    "gt_miou",
    "best_so_far",
    "regret_vs_ceiling",
    "log_path",
})


def load_panel_trials(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "agent_error" in df.columns:
        err = df["agent_error"].astype(str).str.lower().isin(("true", "1", "yes"))
        df = df.loc[~err].copy()
    df["gt_miou"] = pd.to_numeric(df["gt_miou"], errors="coerce")
    df = df.loc[df["gt_miou"].notna()].copy()
    return df


def available_features(df: pd.DataFrame, allowlist: list[str]) -> list[str]:
    cols: list[str] = []
    for c in allowlist:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 8 or s.nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols


def _spearman_pair(sub: pd.DataFrame, feat: str, target: str) -> float | None:
    d = sub[[feat, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 5 or d[feat].std() < 1e-12 or d[target].std() < 1e-12:
        return None
    r, _ = spearmanr(d[feat], d[target])
    return float(r) if np.isfinite(r) else None


def pick_axis_features(
    df: pd.DataFrame,
    allowlist: list[str],
    *,
    target: str = "gt_miou",
    ring_col: str = "case_id",
    top_k: int = 4,
    min_rings: int = 3,
    max_feature_corr: float = 0.9,
) -> dict[str, Any]:
    candidates = available_features(df, allowlist)
    rows: list[dict[str, Any]] = []
    for feat in candidates:
        rhos: list[float] = []
        rings_var = 0
        for _, g in df.groupby(ring_col):
            r = _spearman_pair(g, feat, target)
            if r is not None:
                rhos.append(abs(r))
                rings_var += 1
        pooled = _spearman_pair(df, feat, target)
        rows.append({
            "feature": feat,
            "mean_abs_spearman": float(np.mean(rhos)) if rhos else 0.0,
            "n_rings": rings_var,
            "pooled_abs_spearman": abs(pooled) if pooled is not None else 0.0,
        })
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return {"picked_features": [], "ranking": ranking, "n_rings": int(df[ring_col].nunique())}

    ranking = ranking.sort_values(
        ["mean_abs_spearman", "pooled_abs_spearman"],
        ascending=False,
    )
    picked: list[str] = []
    for feat in ranking["feature"]:
        if len(picked) >= top_k:
            break
        if int(ranking.loc[ranking["feature"] == feat, "n_rings"].iloc[0]) < min_rings:
            continue
        redundant = False
        for prev in picked:
            sub = df[[feat, prev]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) >= 8 and sub[feat].std() > 1e-12 and sub[prev].std() > 1e-12:
                c, _ = spearmanr(sub[feat], sub[prev])
                if np.isfinite(c) and abs(float(c)) >= max_feature_corr:
                    redundant = True
                    break
        if not redundant:
            picked.append(feat)

    n_rings_pass = 0
    for _, g in df.groupby(ring_col):
        if any(abs(_spearman_pair(g, f, target) or 0) >= 0.15 for f in picked):
            n_rings_pass += 1

    return {
        "picked_features": picked,
        "ranking": ranking,
        "n_rings": int(df[ring_col].nunique()),
        "n_rows": int(len(df)),
        "rings_with_picked_rho_ge_0_15": n_rings_pass,
        "top_k": top_k,
        "min_rings": min_rings,
    }


def build_lk_concat_records(stream_l: Path, stream_k: Path) -> pd.DataFrame:
    ldf = load_panel_trials(stream_l / "bo_trials.csv")
    kdf = load_panel_trials(stream_k / "bo_trials.csv")
    ldf["axis_source"] = "layout"
    kdf["axis_source"] = "k"
    return pd.concat([ldf, kdf], ignore_index=True)


def train_ridge_miou(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    alpha: float = 1.0,
    target: str = "gt_miou",
) -> dict[str, Any]:
    if not feature_cols:
        raise ValueError("No feature columns for Ridge fit")
    sub = df.copy()
    x = sub[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y = sub[target].to_numpy(float)
    sc = StandardScaler()
    xs = sc.fit_transform(x)
    reg = Ridge(alpha=alpha).fit(xs, y)
    pred = reg.predict(xs)
    rho, _ = spearmanr(pred, y)
    mae = float(np.mean(np.abs(pred - y)))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {
        "alpha": alpha,
        "target": target,
        "feature_columns": feature_cols,
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "coef": reg.coef_.tolist(),
        "intercept": float(reg.intercept_),
        "pooled_spearman": float(rho) if np.isfinite(rho) else None,
        "mae": mae,
        "r2": r2,
    }


def _feature_value(row: pd.Series | dict, name: str) -> float:
    try:
        v = float(row.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not np.isfinite(v) else v


def predict_ridge_miou(model: dict[str, Any], row: pd.Series | dict) -> float:
    if isinstance(row, dict):
        row = pd.Series(row)
    feats = model["feature_columns"]
    x = np.array([_feature_value(row, f) for f in feats], dtype=float)
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    xs = (x - mean) / np.where(scale == 0, 1.0, scale)
    coef = np.asarray(model["coef"], dtype=float)
    return float(np.dot(xs, coef) + model["intercept"])


def loro_train_and_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    alpha: float = 1.0,
    ring_col: str = "case_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cases = sorted(df[ring_col].unique())
    if len(cases) < 2:
        model = train_ridge_miou(df, feature_cols, alpha=alpha)
        out = df.copy()
        out["pred_gt_miou"] = out.apply(lambda r: predict_ridge_miou(model, r), axis=1)
        sel = ring_selection_metrics(out, score_col="pred_gt_miou")
        return out, {"model": model, "loro": sel, "oof_spearman": model.get("pooled_spearman")}

    parts: list[pd.DataFrame] = []
    for held in cases:
        tr = df.loc[df[ring_col] != held]
        te = df.loc[df[ring_col] == held].copy()
        model = train_ridge_miou(tr, feature_cols, alpha=alpha)
        te["pred_gt_miou"] = te.apply(lambda r: predict_ridge_miou(model, r), axis=1)
        parts.append(te)
    oof = pd.concat(parts, ignore_index=True)
    valid = oof["pred_gt_miou"].notna() & oof["gt_miou"].notna()
    rho, _ = spearmanr(oof.loc[valid, "pred_gt_miou"], oof.loc[valid, "gt_miou"])
    sel = ring_selection_metrics(oof, score_col="pred_gt_miou")
    pooled_model = train_ridge_miou(df, feature_cols, alpha=alpha)
    return oof, {
        "model": pooled_model,
        "loro": sel,
        "oof_spearman": float(rho) if np.isfinite(rho) else None,
    }


def train_axis_bundle(
    df: pd.DataFrame,
    allowlist: list[str],
    *,
    name: str,
    alpha: float = 1.0,
    top_k: int = 4,
    lineage: str,
) -> dict[str, Any]:
    pick = pick_axis_features(df, allowlist, top_k=top_k)
    features = pick["picked_features"]
    if not features:
        features = available_features(df, allowlist)[:top_k]
    oof, fit = loro_train_and_predict(df, features, alpha=alpha)
    sel = fit["loro"]
    gate_pass = bool(
        fit.get("oof_spearman") is not None
        and sel.get("n_rings", 0) >= 2
        and pick.get("rings_with_picked_rho_ge_0_15", 0) >= 3
    )
    thresholds = _thresholds_for_name(name)
    if fit.get("oof_spearman") is not None:
        gate_pass = gate_pass and fit["oof_spearman"] >= thresholds["min_oof_spearman"]
    if sel.get("mean_regret_vs_oracle") is not None:
        gate_pass = gate_pass and sel["mean_regret_vs_oracle"] <= thresholds["max_loro_regret"]

    return {
        "name": name,
        "lineage": lineage,
        "feature_pick": pick,
        "model": fit["model"],
        "oof_predictions": oof,
        "loro_selection": sel,
        "oof_spearman": fit.get("oof_spearman"),
        "gate_pass": gate_pass,
        "thresholds": thresholds,
    }


def _thresholds_for_name(name: str) -> dict[str, float]:
    base = name.split("_k")[0] if "_k" in name else name
    if base == "K":
        return {"min_oof_spearman": 0.35, "max_loro_regret": 0.20}
    if base in ("LK_concat", "LK_joint"):
        return {"min_oof_spearman": 0.35, "max_loro_regret": 0.20}
    return {"min_oof_spearman": 0.20, "max_loro_regret": 0.25}


def write_bundle_artifacts(bundle: dict[str, Any], out_dir: Path) -> None:
    name = bundle["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    pick = bundle["feature_pick"]
    ranking = pick.get("ranking", pd.DataFrame())
    pick_json = {k: v for k, v in pick.items() if k != "ranking"}
    (out_dir / f"feature_pick_{name}.json").write_text(
        json.dumps(pick_json, indent=2) + "\n",
        encoding="utf-8",
    )
    if not ranking.empty:
        ranking.to_csv(out_dir / f"feature_ranking_{name}.csv", index=False)

    model_path = out_dir / "models" / f"proxy_{name}.json"
    model_payload = dict(bundle["model"])
    model_payload["name"] = name
    model_payload["lineage"] = bundle["lineage"]
    model_path.write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")

    oof: pd.DataFrame = bundle["oof_predictions"]
    oof.to_csv(out_dir / f"proxy_calibration_predictions_{name}.csv", index=False)

    sel = bundle["loro_selection"]
    per_ring = sel.get("per_ring", [])
    if per_ring:
        pd.DataFrame(per_ring).to_csv(out_dir / f"proxy_ring_selection_{name}.csv", index=False)


def train_all_proxies(
    *,
    stream_l_root: Path,
    stream_k_root: Path,
    stream_full_root: Path | None,
    out_dir: Path,
    alpha: float = 1.0,
    top_k: int = 4,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ldf = load_panel_trials(stream_l_root / "bo_trials.csv")
    ldf.to_csv(out_dir / "records_L.csv", index=False)
    kdf = load_panel_trials(stream_k_root / "bo_trials.csv")
    kdf.to_csv(out_dir / "records_K.csv", index=False)

    lk_concat = build_lk_concat_records(stream_l_root, stream_k_root)
    lk_concat.to_csv(out_dir / "records_LK_concat.csv", index=False)

    bundles: list[dict[str, Any]] = []
    bundles.append(
        train_axis_bundle(
            ldf,
            L_PROXY_ALLOWLIST,
            name="L",
            alpha=alpha,
            top_k=top_k,
            lineage=str(stream_l_root.resolve().relative_to(REPO_ROOT.resolve())),
        )
    )
    bundles.append(
        train_axis_bundle(
            kdf,
            K_PROXY_ALLOWLIST,
            name="K",
            alpha=alpha,
            top_k=top_k,
            lineage=str(stream_k_root.resolve().relative_to(REPO_ROOT.resolve())),
        )
    )
    bundles.append(
        train_axis_bundle(
            lk_concat,
            LK_PROXY_ALLOWLIST,
            name="LK_concat",
            alpha=alpha,
            top_k=top_k,
            lineage=f"{stream_l_root.name}+{stream_k_root.name}",
        )
    )

    if stream_full_root is not None:
        full_path = stream_full_root / "bo_trials.csv"
        if full_path.is_file():
            jdf = load_panel_trials(full_path)
            jdf.to_csv(out_dir / "records_LK_joint.csv", index=False)
            bundles.append(
                train_axis_bundle(
                    jdf,
                    LK_PROXY_ALLOWLIST,
                    name="LK_joint",
                    alpha=alpha,
                    top_k=top_k,
                    lineage=str(stream_full_root.resolve().relative_to(REPO_ROOT.resolve())),
                )
            )

    for b in bundles:
        write_bundle_artifacts(b, out_dir)

    return build_training_gate(bundles, out_dir)


def build_training_gate(bundles: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for b in bundles:
        summary[b["name"]] = {
            "gate_pass": b["gate_pass"],
            "oof_spearman": b.get("oof_spearman"),
            "mean_regret_vs_oracle": b["loro_selection"].get("mean_regret_vs_oracle"),
            "mean_proxy_top1_miou": b["loro_selection"].get("mean_proxy_top1_miou"),
            "picked_features": b["feature_pick"].get("picked_features"),
            "rings_with_picked_rho_ge_0_15": b["feature_pick"].get("rings_with_picked_rho_ge_0_15"),
        }

    lk_concat = summary.get("LK_concat", {})
    lk_joint = summary.get("LK_joint", {})
    deploy_lk = "LK_concat"
    if lk_joint and lk_concat:
        c_reg = lk_concat.get("mean_regret_vs_oracle") or 999.0
        j_reg = lk_joint.get("mean_regret_vs_oracle") or 999.0
        c_rho = lk_concat.get("oof_spearman") or -1.0
        j_rho = lk_joint.get("oof_spearman") or -1.0
        if j_reg < c_reg or (j_reg == c_reg and j_rho > c_rho):
            deploy_lk = "LK_joint"
    elif lk_joint.get("gate_pass"):
        deploy_lk = "LK_joint"

    gate = {
        "passed": all(b["gate_pass"] for b in bundles if b["name"] in ("K", "LK_concat"))
        or any(b["gate_pass"] for b in bundles),
        "per_model": summary,
        "deploy_recommendation": {
            "L": "proxy_L",
            "K": "proxy_K",
            "LK": deploy_lk,
            "D": "direction_select",
        },
        "stream_d_note": "logs/proxy4tun/analysis/stream_d_proxy_gate.json",
    }
    (out_dir / "proxy_training_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "sandbox": str(out_dir.resolve().relative_to(REPO_ROOT.resolve())),
        "models": {b["name"]: f"models/proxy_{b['name']}.json" for b in bundles},
        "gate": gate,
    }
    (out_dir / "PROXY4TUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "proxy_loro_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    return gate


def train_lk_enriched_sweep(
    *,
    records_concat: Path,
    records_joint: Path,
    out_dir: Path,
    alpha: float = 1.0,
    top_k_values: tuple[int, ...] = (4, 8, 12),
    v1_gate_path: Path | None = None,
) -> dict[str, Any]:
    """Train LK_concat and LK_joint enriched proxies for each top-k."""
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_df = load_panel_trials(records_concat)
    joint_df = load_panel_trials(records_joint)

    bundles: list[dict[str, Any]] = []
    for top_k in top_k_values:
        for pool_name, df, lineage in (
            ("LK_concat", concat_df, str(records_concat)),
            ("LK_joint", joint_df, str(records_joint)),
        ):
            model_name = f"{pool_name}_k{top_k}"
            bundles.append(
                train_axis_bundle(
                    df,
                    LK_ENRICHED_ALLOWLIST,
                    name=model_name,
                    alpha=alpha,
                    top_k=top_k,
                    lineage=lineage,
                )
            )

    for b in bundles:
        write_bundle_artifacts(b, out_dir)

    v1_ref: dict[str, Any] = {}
    if v1_gate_path and v1_gate_path.is_file():
        v1_ref = json.loads(v1_gate_path.read_text(encoding="utf-8")).get("per_model", {})

    rows: list[dict[str, Any]] = []
    for b in bundles:
        name = b["name"]
        base = name.rsplit("_k", 1)[0]
        k = int(name.rsplit("_k", 1)[-1])
        v1_key = base if base in v1_ref else None
        v1m = v1_ref.get(v1_key, {}) if v1_key else {}
        rows.append({
            "model": name,
            "pool": base,
            "enriched": True,
            "top_k": k,
            "oof_spearman": b.get("oof_spearman"),
            "mean_regret_vs_oracle": b["loro_selection"].get("mean_regret_vs_oracle"),
            "mean_proxy_top1_miou": b["loro_selection"].get("mean_proxy_top1_miou"),
            "gate_pass": b["gate_pass"],
            "picked_features": "|".join(b["feature_pick"].get("picked_features", [])),
            "v1_oof_spearman": v1m.get("oof_spearman"),
            "v1_mean_regret": v1m.get("mean_regret_vs_oracle"),
            "delta_oof_spearman": (
                (b.get("oof_spearman") or 0) - (v1m.get("oof_spearman") or 0)
                if b.get("oof_spearman") is not None and v1m.get("oof_spearman") is not None
                else None
            ),
        })
    for v1_key in ("LK_concat", "LK_joint"):
        if v1_key in v1_ref:
            vm = v1_ref[v1_key]
            rows.append({
                "model": f"{v1_key}_v1_baseline",
                "pool": v1_key,
                "enriched": False,
                "top_k": 4,
                "oof_spearman": vm.get("oof_spearman"),
                "mean_regret_vs_oracle": vm.get("mean_regret_vs_oracle"),
                "mean_proxy_top1_miou": vm.get("mean_proxy_top1_miou"),
                "gate_pass": vm.get("gate_pass"),
                "picked_features": "|".join(vm.get("picked_features", [])),
                "v1_oof_spearman": vm.get("oof_spearman"),
                "v1_mean_regret": vm.get("mean_regret_vs_oracle"),
                "delta_oof_spearman": 0.0,
            })

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(out_dir / "topk_comparison.csv", index=False)

    enriched_only = [b for b in bundles if b["name"].startswith("LK_concat")]
    enriched_joint = [b for b in bundles if b["name"].startswith("LK_joint")]
    best_concat = max(enriched_only, key=lambda b: b.get("oof_spearman") or -999.0)
    best_joint = max(enriched_joint, key=lambda b: b.get("oof_spearman") or -999.0)

    manifest = {
        "sandbox": str(out_dir.resolve().relative_to(REPO_ROOT.resolve())),
        "enriched": True,
        "top_k_sweep": list(top_k_values),
        "models": {b["name"]: f"models/proxy_{b['name']}.json" for b in bundles},
        "deploy_recommendation": {
            "LK_concat": best_concat["name"],
            "LK_joint": best_joint["name"],
            "L": "proxy_L (v1)",
            "K": "proxy_K (v1)",
            "D": "direction_select",
        },
        "per_model": {b["name"]: {
            "gate_pass": b["gate_pass"],
            "oof_spearman": b.get("oof_spearman"),
            "mean_regret_vs_oracle": b["loro_selection"].get("mean_regret_vs_oracle"),
            "picked_features": b["feature_pick"].get("picked_features"),
        } for b in bundles},
    }
    (out_dir / "PROXY4TUN_V2_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "proxy_training_gate_v2.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


WEIGHTED_LK_META_FEATURES = ("pred_L_sub", "pred_K_sub")


def load_proxy_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def attach_subproxy_scores(
    df: pd.DataFrame,
    model_l: dict[str, Any],
    model_k: dict[str, Any],
) -> pd.DataFrame:
    out = df.copy()
    out["pred_L_sub"] = out.apply(lambda r: predict_ridge_miou(model_l, r), axis=1)
    out["pred_K_sub"] = out.apply(lambda r: predict_ridge_miou(model_k, r), axis=1)
    return out


def _fit_axis_model_for_subset(
    df: pd.DataFrame,
    allowlist: list[str],
    *,
    alpha: float,
    top_k: int,
) -> dict[str, Any]:
    pick = pick_axis_features(df, allowlist, top_k=top_k)
    features = pick["picked_features"]
    if not features:
        features = available_features(df, allowlist)[:top_k]
    return train_ridge_miou(df, features, alpha=alpha)


def loro_nested_weighted_lk(
    ldf: pd.DataFrame,
    kdf: pd.DataFrame,
    concat_df: pd.DataFrame,
    *,
    alpha: float = 1.0,
    top_k: int = 4,
    ring_col: str = "case_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """LORO with axis models and meta ridge re-fit per held-out ring."""
    cases = sorted(concat_df[ring_col].unique())
    if len(cases) < 2:
        m_l = _fit_axis_model_for_subset(ldf, L_PROXY_ALLOWLIST, alpha=alpha, top_k=top_k)
        m_k = _fit_axis_model_for_subset(kdf, K_PROXY_ALLOWLIST, alpha=alpha, top_k=top_k)
        scored = attach_subproxy_scores(concat_df, m_l, m_k)
        meta = train_ridge_miou(scored, list(WEIGHTED_LK_META_FEATURES), alpha=alpha)
        out = scored.copy()
        out["pred_gt_miou"] = out.apply(lambda r: predict_ridge_miou(meta, r), axis=1)
        sel = ring_selection_metrics(out, score_col="pred_gt_miou")
        rho, _ = spearmanr(out["pred_gt_miou"], out["gt_miou"])
        return out, {
            "model_l": m_l,
            "model_k": m_k,
            "model": meta,
            "loro": sel,
            "oof_spearman": float(rho) if np.isfinite(rho) else None,
            "loro_mode": "nested",
        }

    parts: list[pd.DataFrame] = []
    for held in cases:
        tr_l = ldf.loc[ldf[ring_col] != held]
        tr_k = kdf.loc[kdf[ring_col] != held]
        tr_c = concat_df.loc[concat_df[ring_col] != held]
        te = concat_df.loc[concat_df[ring_col] == held].copy()
        m_l = _fit_axis_model_for_subset(tr_l, L_PROXY_ALLOWLIST, alpha=alpha, top_k=top_k)
        m_k = _fit_axis_model_for_subset(tr_k, K_PROXY_ALLOWLIST, alpha=alpha, top_k=top_k)
        tr_scored = attach_subproxy_scores(tr_c, m_l, m_k)
        meta = train_ridge_miou(tr_scored, list(WEIGHTED_LK_META_FEATURES), alpha=alpha)
        te = attach_subproxy_scores(te, m_l, m_k)
        te["pred_gt_miou"] = te.apply(lambda r: predict_ridge_miou(meta, r), axis=1)
        parts.append(te)

    oof = pd.concat(parts, ignore_index=True)
    valid = oof["pred_gt_miou"].notna() & oof["gt_miou"].notna()
    rho, _ = spearmanr(oof.loc[valid, "pred_gt_miou"], oof.loc[valid, "gt_miou"])
    sel = ring_selection_metrics(oof, score_col="pred_gt_miou")
    m_l = _fit_axis_model_for_subset(ldf, L_PROXY_ALLOWLIST, alpha=alpha, top_k=top_k)
    m_k = _fit_axis_model_for_subset(kdf, K_PROXY_ALLOWLIST, alpha=alpha, top_k=top_k)
    pooled = attach_subproxy_scores(concat_df, m_l, m_k)
    meta = train_ridge_miou(pooled, list(WEIGHTED_LK_META_FEATURES), alpha=alpha)
    return oof, {
        "model_l": m_l,
        "model_k": m_k,
        "model": meta,
        "loro": sel,
        "oof_spearman": float(rho) if np.isfinite(rho) else None,
        "loro_mode": "nested",
    }


def attach_axis_gated_score(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row score from the axis that actually varied in that BO stream."""
    out = df.copy()
    is_k = out["axis_source"].astype(str).str.lower() == "k"
    out["pred_axis_gated"] = np.where(is_k, out["pred_K_sub"], out["pred_L_sub"])
    return out


def train_gated_weighted_lk_proxy(
    *,
    records_concat: Path,
    model_l_path: Path,
    model_k_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Deploy score = pred_K on k-rows, pred_L on layout-rows (no cross-axis blend)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_df = load_panel_trials(records_concat)
    model_l = load_proxy_model(model_l_path)
    model_k = load_proxy_model(model_k_path)
    scored = attach_axis_gated_score(attach_subproxy_scores(concat_df, model_l, model_k))
    scored.to_csv(out_dir / "records_LK_concat_scored.csv", index=False)

    oof = scored.copy()
    oof["pred_gt_miou"] = oof["pred_axis_gated"]
    valid = oof["pred_gt_miou"].notna() & oof["gt_miou"].notna()
    rho, _ = spearmanr(oof.loc[valid, "pred_gt_miou"], oof.loc[valid, "gt_miou"])
    sel = ring_selection_metrics(oof, score_col="pred_gt_miou")

    name = "LK_gated"
    model_path = out_dir / "models" / f"proxy_{name}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "lineage": f"gated|{model_l_path}|{model_k_path}",
        "subproxy_L": str(model_l_path),
        "subproxy_K": str(model_k_path),
        "rule": "pred = pred_K_sub if axis_source==k else pred_L_sub",
    }
    model_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    oof.to_csv(out_dir / f"proxy_calibration_predictions_{name}.csv", index=False)
    per_ring = sel.get("per_ring", [])
    if per_ring:
        pd.DataFrame(per_ring).to_csv(out_dir / f"proxy_ring_selection_{name}.csv", index=False)

    gate = {
        "passed": bool(rho is not None and rho >= 0.35 and sel.get("mean_regret_vs_oracle", 1) <= 0.20),
        "oof_spearman": float(rho) if np.isfinite(rho) else None,
        "mean_regret_vs_oracle": sel.get("mean_regret_vs_oracle"),
        "mean_proxy_top1_miou": sel.get("mean_proxy_top1_miou"),
        "loro_mode": "axis_gated",
        "baselines_on_same_pool": _weighted_lk_baselines(scored),
    }
    (out_dir / "proxy_training_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "sandbox": str(out_dir.resolve().relative_to(REPO_ROOT.resolve())),
        "model": f"models/proxy_{name}.json",
        "gate": gate,
    }
    (out_dir / "PROXY4TUN_WEIGHTED_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def loro_alpha_blend_weighted(
    scored: pd.DataFrame,
    *,
    ring_col: str = "case_id",
    alphas: tuple[float, ...] = tuple(i / 20 for i in range(21)),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """LORO: per held-out ring pick alpha maximizing Spearman on train, apply on test."""
    cases = sorted(scored[ring_col].unique())
    parts: list[pd.DataFrame] = []
    chosen: list[dict[str, Any]] = []
    for held in cases:
        tr = scored.loc[scored[ring_col] != held]
        te = scored.loc[scored[ring_col] == held].copy()
        best_a, best_rho = 0.5, -999.0
        for a in alphas:
            p = (1.0 - a) * tr["pred_L_sub"] + a * tr["pred_K_sub"]
            r, _ = spearmanr(p, tr["gt_miou"])
            if np.isfinite(r) and float(r) > best_rho:
                best_rho = float(r)
                best_a = a
        te["pred_gt_miou"] = (1.0 - best_a) * te["pred_L_sub"] + best_a * te["pred_K_sub"]
        te["blend_alpha_k"] = best_a
        parts.append(te)
        chosen.append({"held_ring": held, "alpha_k": best_a, "train_spearman": best_rho})
    oof = pd.concat(parts, ignore_index=True)
    valid = oof["pred_gt_miou"].notna() & oof["gt_miou"].notna()
    rho, _ = spearmanr(oof.loc[valid, "pred_gt_miou"], oof.loc[valid, "gt_miou"])
    sel = ring_selection_metrics(oof, score_col="pred_gt_miou")
    pooled_a = float(np.mean([c["alpha_k"] for c in chosen]))
    return oof, {
        "loro": sel,
        "oof_spearman": float(rho) if np.isfinite(rho) else None,
        "loro_mode": "alpha_blend",
        "per_ring_alpha": chosen,
        "pooled_alpha_k": pooled_a,
    }


def train_weighted_lk_proxy(
    *,
    records_concat: Path,
    model_l_path: Path,
    model_k_path: Path,
    out_dir: Path,
    alpha: float = 1.0,
    blend: str = "ridge",
    nested_loro: bool = False,
    records_l: Path | None = None,
    records_k: Path | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    """Meta Ridge: gt_miou ~ w_L * pred_L_sub + w_K * pred_K_sub (+ intercept)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_df = load_panel_trials(records_concat)
    concat_df.to_csv(out_dir / "records_LK_concat.csv", index=False)

    model_l = load_proxy_model(model_l_path)
    model_k = load_proxy_model(model_k_path)

    if nested_loro:
        if not records_l or not records_k:
            raise ValueError("nested_loro requires records_l and records_k")
        ldf = load_panel_trials(records_l)
        kdf = load_panel_trials(records_k)
        oof, fit = loro_nested_weighted_lk(
            ldf, kdf, concat_df, alpha=alpha, top_k=top_k
        )
        scored = attach_subproxy_scores(concat_df, fit["model_l"], fit["model_k"])
        deploy_meta = fit["model"]
        w_report = _weighted_lk_coef_report(deploy_meta)
    else:
        scored = attach_subproxy_scores(concat_df, model_l, model_k)
        if blend == "alpha":
            oof, fit = loro_alpha_blend_weighted(scored)
            deploy_scored = scored
            deploy_meta = {
                "blend": "alpha",
                "alpha_k": fit["pooled_alpha_k"],
                "formula": "pred = (1-alpha_k)*pred_L_sub + alpha_k*pred_K_sub",
            }
            w_report = {"alpha_k": fit["pooled_alpha_k"], "per_ring_alpha": fit["per_ring_alpha"]}
        else:
            oof, fit = loro_train_and_predict(
                scored,
                list(WEIGHTED_LK_META_FEATURES),
                alpha=alpha,
            )
            meta = fit["model"]
            deploy_scored = scored
            deploy_meta = train_ridge_miou(
                deploy_scored, list(WEIGHTED_LK_META_FEATURES), alpha=alpha
            )
            w_report = _weighted_lk_coef_report(deploy_meta)

    scored.to_csv(out_dir / "records_LK_concat_scored.csv", index=False)

    sel = fit["loro"]
    name = "LK_weighted" if blend != "alpha" else "LK_weighted_alpha"

    model_path = out_dir / "models" / f"proxy_{name}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(deploy_meta) if isinstance(deploy_meta, dict) else {}
    payload["name"] = name
    payload["lineage"] = f"{model_l_path}|{model_k_path}"
    payload["subproxy_L"] = str(model_l_path)
    payload["subproxy_K"] = str(model_k_path)
    payload["weight_report"] = w_report
    payload["blend"] = blend
    model_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    oof.to_csv(out_dir / f"proxy_calibration_predictions_{name}.csv", index=False)
    per_ring = sel.get("per_ring", [])
    if per_ring:
        pd.DataFrame(per_ring).to_csv(out_dir / f"proxy_ring_selection_{name}.csv", index=False)

    baselines = _weighted_lk_baselines(scored)
    gate = {
        "passed": fit.get("oof_spearman") is not None and fit["oof_spearman"] >= 0.35,
        "oof_spearman": fit.get("oof_spearman"),
        "mean_regret_vs_oracle": sel.get("mean_regret_vs_oracle"),
        "mean_proxy_top1_miou": sel.get("mean_proxy_top1_miou"),
        "weight_report": w_report,
        "loro_mode": fit.get("loro_mode", "frozen_subproxy"),
        "baselines_on_same_pool": baselines,
    }
    (out_dir / "proxy_training_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "sandbox": str(out_dir.resolve().relative_to(REPO_ROOT.resolve())),
        "model": f"models/proxy_{name}.json",
        "gate": gate,
    }
    (out_dir / "PROXY4TUN_WEIGHTED_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _weighted_lk_coef_report(meta: dict[str, Any]) -> dict[str, Any]:
    """Standardized Ridge coefs map to pred_L_sub / pred_K_sub weights."""
    feats = meta["feature_columns"]
    coef = np.asarray(meta["coef"], dtype=float)
    scale = np.asarray(meta["scaler_scale"], dtype=float)
    mean = np.asarray(meta["scaler_mean"], dtype=float)
    raw_effect = coef / np.where(scale == 0, 1.0, scale)
    return {
        "intercept": meta["intercept"],
        "feature_columns": feats,
        "standardized_coef": coef.tolist(),
        "raw_effect_per_unit": dict(zip(feats, raw_effect.tolist())),
        "note": "pred_miou = intercept + sum(raw_effect[f] * (x[f] - mean[f]))",
    }


def _weighted_lk_baselines(scored: pd.DataFrame) -> dict[str, Any]:
    """Compare meta proxy to K-only and max(L,K) sub-scores on pooled ranking."""
    y = scored["gt_miou"].to_numpy(float)

    def _rho(col: str) -> float | None:
        p = scored[col].to_numpy(float)
        r, _ = spearmanr(p, y)
        return float(r) if np.isfinite(r) else None

    return {
        "spearman_pred_K_sub": _rho("pred_K_sub"),
        "spearman_pred_L_sub": _rho("pred_L_sub"),
        "spearman_max_sub": float(
            spearmanr(
                scored[["pred_L_sub", "pred_K_sub"]].max(axis=1),
                y,
            )[0]
        )
        if len(scored) >= 3
        else None,
    }

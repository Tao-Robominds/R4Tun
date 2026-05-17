from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "logs" / "v5_balance_norm_proxy_v1"
PANEL_CSV = REPO_ROOT / "stages" / "v5" / "panels" / "v5_50ring_panel.csv"
SCOREBOARD_CSV = REPO_ROOT / "logs" / "v5_stage_validation_v1" / "v5_50ring_scoreboard_t123_recovered.csv"
POOL_CSV = REPO_ROOT / "logs" / "v5_proxy_improvement_v1" / "proxy_training_pool.csv"
FINER_CANDS_CSV = REPO_ROOT / "logs" / "v5_finer_proxy_search_v1" / "candidate_scores.csv"
DEPTH_AUDIT_CSV = REPO_ROOT / "logs" / "v5_depth_contract_paper_audit_v1" / "all_50_depth_quality_audit.csv"

FEATURES = [
    "struct_missing_ids_before_n",
    "depth_row_nonempty_ratio_audit",
    "geom_boundary_gap_cv",
    "balance_norm",
]

OLD_PROXY = {
    "intercept": 0.015213,
    "struct_missing_ids_before_n": -0.031164,
    "depth_row_nonempty_ratio_audit": 0.196603,
    "geom_boundary_gap_cv": 0.081910,
}


def _ensure_out() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _compute_miou_from_final(final_csv: Path) -> float | None:
    if not final_csv.exists():
        return None
    df = pd.read_csv(final_csv)
    if "segment" not in df.columns or "pred" not in df.columns:
        return None
    tmp = df[["segment", "pred"]].dropna(subset=["segment"])
    if tmp.empty:
        return None
    gt = tmp["segment"].astype(int).to_numpy()
    pred = tmp["pred"].astype(int).to_numpy()
    valid = (gt >= 0) & (gt <= 7) & (pred >= 0) & (pred <= 7)
    gt = gt[valid]
    pred = pred[valid]
    if gt.size == 0:
        return None
    labels = sorted(set(gt.tolist()) | set(pred.tolist()))
    ious = []
    for c in labels:
        g = gt == c
        p = pred == c
        u = np.logical_or(g, p).sum()
        if u == 0:
            continue
        i = np.logical_and(g, p).sum()
        ious.append(float(i / u))
    return float(np.mean(ious)) if ious else None


def _segment_balance(pred: pd.Series) -> float:
    x = pd.to_numeric(pred, errors="coerce").dropna().astype(int)
    x = x[(x >= 1) & (x <= 7)]
    if x.empty:
        return 0.0
    counts = x.value_counts().reindex(range(1, 8), fill_value=0).astype(float).to_numpy()
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum() / np.log(7.0))
    present_ratio = float((counts > 0).mean())
    cv = float(np.std(counts) / (np.mean(counts) + 1e-9))
    max_share = float(counts.max() / (counts.sum() + 1e-9))
    raw = 1.5 * present_ratio + entropy - 0.35 * cv - 0.5 * max(0.0, max_share - 0.45)
    return float(np.clip(raw / 2.5, 0.0, 1.0))


def _boundary_gap_cv(path: Path, ring_height: int) -> float:
    if not path.exists():
        return float("nan")
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("0") or data.get(0) or []
    ys = sorted(float(e.get("y", 0.0)) for e in entries)
    if len(ys) < 2:
        return float("nan")
    diffs = np.array([(ys[(i + 1) % len(ys)] - ys[i]) % float(ring_height) for i in range(len(ys))], dtype=float)
    return float(np.std(diffs) / (np.mean(diffs) + 1e-9))


def _load_base_pool() -> pd.DataFrame:
    usecols = set(
        [
            "ring_key",
            "miou",
            "family",
            "struct_missing_ids_before_n",
            "depth_row_nonempty_ratio_audit",
            "geom_boundary_gap_cv",
            "balance_norm",
        ]
    )
    df = pd.read_csv(POOL_CSV, low_memory=False)
    cols = [c for c in df.columns if c in usecols]
    out = df[cols].copy()
    # backfill balance_norm if not present
    if "balance_norm" not in out.columns:
        for c in ("feat_present_ratio", "feat_entropy", "feat_cv", "feat_max_share"):
            if c not in df.columns:
                df[c] = np.nan
        out["balance_norm"] = (
            1.5 * pd.to_numeric(df["feat_present_ratio"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["feat_entropy"], errors="coerce").fillna(0.0)
            - 0.35 * pd.to_numeric(df["feat_cv"], errors="coerce").fillna(1.0)
            - 0.5 * (pd.to_numeric(df["feat_max_share"], errors="coerce").fillna(1.0) - 0.45).clip(lower=0.0)
        ).clip(lower=0.0, upper=2.5) / 2.5
    out["candidate_source"] = "proxy_training_pool"
    return out


def _load_finer_candidates() -> pd.DataFrame:
    if not FINER_CANDS_CSV.exists():
        return pd.DataFrame(columns=["ring_key", "miou"] + FEATURES + ["family", "candidate_source"])
    df = pd.read_csv(FINER_CANDS_CSV)
    need = ["ring_key", "miou", "family"] + FEATURES
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    out = df[need].copy()
    out["candidate_source"] = "finer_search_hardpilot"
    return out


def _single_candidate_for_missing_ring(ring_key: str) -> pd.DataFrame:
    tid, rr = ring_key.split("/")
    rid = int(rr.lstrip("r"))
    ring_dir = REPO_ROOT / "logs" / "v5_t45_depth_contract_v1" / tid / f"r{rid}"
    final_csv = ring_dir / "final.csv"
    miou = _compute_miou_from_final(final_csv)
    depth = pd.read_csv(DEPTH_AUDIT_CSV)
    drow = depth.loc[depth["ring_key"].eq(ring_key), "row_nonempty_ratio"]
    depth_row = float(drow.iloc[0]) if not drow.empty else np.nan
    bcv = _boundary_gap_cv(ring_dir / "boundaries_per_ring.json", ring_height=int(np.load(ring_dir / "depth_map.npy").shape[0])) if (ring_dir / "depth_map.npy").exists() else np.nan
    bal = np.nan
    if final_csv.exists():
        fdf = pd.read_csv(final_csv)
        if "pred" in fdf.columns:
            bal = _segment_balance(fdf["pred"])
    return pd.DataFrame(
        [
            {
                "ring_key": ring_key,
                "miou": miou,
                "family": int(tid.split("-")[0]),
                "struct_missing_ids_before_n": 0.0,
                "depth_row_nonempty_ratio_audit": depth_row,
                "geom_boundary_gap_cv": bcv,
                "balance_norm": bal,
                "candidate_source": "single_fallback_current_final",
            }
        ]
    )


def _score_old_proxy(df: pd.DataFrame) -> pd.Series:
    s = OLD_PROXY["intercept"]
    s += OLD_PROXY["struct_missing_ids_before_n"] * pd.to_numeric(df["struct_missing_ids_before_n"], errors="coerce").fillna(0.0)
    s += OLD_PROXY["depth_row_nonempty_ratio_audit"] * pd.to_numeric(df["depth_row_nonempty_ratio_audit"], errors="coerce").fillna(0.0)
    s += OLD_PROXY["geom_boundary_gap_cv"] * pd.to_numeric(df["geom_boundary_gap_cv"], errors="coerce").fillna(0.0)
    return s


def main() -> int:
    _ensure_out()
    panel = pd.read_csv(PANEL_CSV)
    score = pd.read_csv(SCOREBOARD_CSV)[["ring_key", "stabilised_miou", "intrinsic_final_miou", "family"]]

    pool = pd.concat([_load_base_pool(), _load_finer_candidates()], ignore_index=True, sort=False)
    pool = pool[pool["ring_key"].isin(panel["ring_key"])].copy()
    pool["miou"] = pd.to_numeric(pool["miou"], errors="coerce")
    for c in FEATURES:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")

    missing = sorted(set(panel["ring_key"]) - set(pool["ring_key"]))
    if missing:
        extra_rows = []
        for rk in missing:
            extra_rows.append(_single_candidate_for_missing_ring(rk))
        pool = pd.concat([pool] + extra_rows, ignore_index=True, sort=False)

    # train new ridge on all available candidate rows (offline diagnostic)
    train_df = pool[pool["miou"].notna()].copy()
    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURES,
            )
        ]
    )
    best_model = None
    best_alpha = None
    best_mse = float("inf")
    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0]:
        model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=alpha, random_state=42))])
        model.fit(train_df[FEATURES], train_df["miou"])
        pred = model.predict(train_df[FEATURES])
        mse = float(np.mean((pred - train_df["miou"].to_numpy()) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_model = model

    assert best_model is not None
    pool["proxy_balance_ridge"] = best_model.predict(pool[FEATURES])
    pool["proxy_current"] = _score_old_proxy(pool)

    # per-ring selection
    rows = []
    for rk, g in pool.groupby("ring_key"):
        g_new = g.sort_values(["proxy_balance_ridge", "miou"], ascending=[False, False]).iloc[0]
        g_old = g.sort_values(["proxy_current", "miou"], ascending=[False, False]).iloc[0]
        oracle = pd.to_numeric(g["miou"], errors="coerce").max()
        sb = score[score["ring_key"].eq(rk)]
        stabilised = float(sb["stabilised_miou"].iloc[0]) if not sb.empty else np.nan
        production = float(sb["intrinsic_final_miou"].iloc[0]) if not sb.empty else np.nan
        fam = int(sb["family"].iloc[0]) if not sb.empty else int(str(rk).split("-")[0])
        rows.append(
            {
                "ring_key": rk,
                "family": fam,
                "selected_miou_new": float(g_new["miou"]) if pd.notna(g_new["miou"]) else np.nan,
                "selected_miou_old": float(g_old["miou"]) if pd.notna(g_old["miou"]) else np.nan,
                "oracle_miou": float(oracle) if pd.notna(oracle) else np.nan,
                "oracle_gap_new": float(oracle - g_new["miou"]) if pd.notna(oracle) and pd.notna(g_new["miou"]) else np.nan,
                "oracle_gap_old": float(oracle - g_old["miou"]) if pd.notna(oracle) and pd.notna(g_old["miou"]) else np.nan,
                "stabilised_miou": stabilised,
                "production_intrinsic_miou": production,
                "lift_new_vs_old": float(g_new["miou"] - g_old["miou"]) if pd.notna(g_new["miou"]) and pd.notna(g_old["miou"]) else np.nan,
                "lift_new_vs_production": float(g_new["miou"] - production) if pd.notna(g_new["miou"]) and pd.notna(production) else np.nan,
                "selected_source_new": str(g_new.get("candidate_source", "")),
                "selected_source_old": str(g_old.get("candidate_source", "")),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["family", "ring_key"]).reset_index(drop=True)
    out_df.to_csv(OUT_ROOT / "selected_scoreboard_50rings.csv", index=False)
    pool.to_csv(OUT_ROOT / "candidate_pool_used.csv", index=False)

    # formula
    ridge = best_model.named_steps["ridge"]
    # get transformed coef back in feature space approximation with standardized pipeline not straightforward.
    # store model params and feature list instead.
    formula_txt = "\n".join(
        [
            "Ridge proxy with features:",
            ", ".join(FEATURES),
            f"best_alpha={best_alpha}",
            f"train_rows={len(train_df)}",
            "Note: model uses median-impute + standardize before ridge.",
        ]
    )
    (OUT_ROOT / "proxy_balance_ridge_formula.txt").write_text(formula_txt + "\n", encoding="utf-8")

    summary = {
        "n_panel_rings": int(panel["ring_key"].nunique()),
        "n_pool_rings": int(pool["ring_key"].nunique()),
        "n_pool_rows": int(len(pool)),
        "missing_rings_filled_with_single_candidate": missing,
        "best_alpha": float(best_alpha),
        "train_mse": float(best_mse),
        "mean_selected_miou_new": float(pd.to_numeric(out_df["selected_miou_new"], errors="coerce").mean()),
        "mean_selected_miou_old": float(pd.to_numeric(out_df["selected_miou_old"], errors="coerce").mean()),
        "mean_production_intrinsic_miou": float(pd.to_numeric(out_df["production_intrinsic_miou"], errors="coerce").mean()),
        "mean_oracle_miou": float(pd.to_numeric(out_df["oracle_miou"], errors="coerce").mean()),
        "mean_oracle_gap_new": float(pd.to_numeric(out_df["oracle_gap_new"], errors="coerce").mean()),
        "mean_oracle_gap_old": float(pd.to_numeric(out_df["oracle_gap_old"], errors="coerce").mean()),
        "regressions_vs_production_new": int((pd.to_numeric(out_df["lift_new_vs_production"], errors="coerce") < 0).sum()),
        "improved_vs_old_count": int((pd.to_numeric(out_df["lift_new_vs_old"], errors="coerce") > 0).sum()),
        "worse_vs_old_count": int((pd.to_numeric(out_df["lift_new_vs_old"], errors="coerce") < 0).sum()),
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


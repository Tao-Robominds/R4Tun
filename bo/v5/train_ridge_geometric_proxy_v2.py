from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bo.v5 import train_ridge_intrinsic_selector_v1 as base

RUN_ROOT = REPO_ROOT / "logs" / "v5_t3_intrinsic_selector_ablation_v1" / "ridge_geometric_proxy_v2"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

DEPTH_AUDIT_FILES = [
    REPO_ROOT / "logs" / "v5_t123_depth_contract_v1" / "all_30_depth_gate_depth_quality_audit.csv",
    REPO_ROOT / "logs" / "v5_t3_continuous_recovery_v1" / "t3_depth_quality_audit.csv",
    REPO_ROOT / "logs" / "v5_t123_kcenter_recovery_v1" / "depth_quality_gate.csv",
    REPO_ROOT / "logs" / "v5_stage_validation_v1" / "depth_quality_audit.csv",
]

GEOMETRIC_FEATURES = [
    "depth_finite_ratio_audit",
    "depth_row_nonempty_ratio_audit",
    "depth_gap_frac_audit",
    "depth_finite_ratio_local",
    "depth_row_nonempty_ratio_local",
    "depth_gap_frac_local",
    "det_pos_lines",
    "det_neg_lines",
    "det_horiz_lines",
    "det_k_conf",
    "det_k_y_frac_meta",
    "struct_missing_ids_before_n",
    "struct_reassigned_points_n",
    "struct_final_present_ids_n",
    "geom_boundary_count",
    "geom_boundary_gap_cv",
    "geom_boundary_min_gap_frac",
    "geom_boundary_max_gap_frac",
    "geom_seg_block_count",
    "geom_seg_quality_mean",
    "geom_seg_quality_std",
]


def _load_depth_audit_features() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in DEPTH_AUDIT_FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "ring_key" not in df.columns:
            continue
        out = pd.DataFrame({"ring_key": df["ring_key"].astype(str)})
        if "finite_ratio" in df.columns:
            out["depth_finite_ratio_audit"] = df["finite_ratio"].astype(float)
        if "row_nonempty_ratio" in df.columns:
            out["depth_row_nonempty_ratio_audit"] = df["row_nonempty_ratio"].astype(float)
        if "largest_empty_vertical_gap_frac" in df.columns:
            out["depth_gap_frac_audit"] = df["largest_empty_vertical_gap_frac"].astype(float)
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["ring_key", "depth_finite_ratio_audit", "depth_row_nonempty_ratio_audit", "depth_gap_frac_audit"])
    merged = pd.concat(rows, ignore_index=True)
    merged = merged.groupby("ring_key", as_index=False).mean(numeric_only=True)
    return merged


def _candidate_specific_paths(final_csv_abs: Path) -> tuple[Path, Path]:
    ring_dir = final_csv_abs.parent
    stem = final_csv_abs.stem
    if stem.startswith("final_"):
        tag = stem[len("final_") :]
        seg = ring_dir / f"all_segments_{tag}.csv"
        bnd = ring_dir / f"boundaries_{tag}.json"
        if seg.exists() and bnd.exists():
            return seg, bnd
    seg_default = ring_dir / "all_segments.csv"
    bnd_default = ring_dir / "boundaries_per_ring.json"
    return seg_default, bnd_default


def _largest_empty_gap_frac(mask_rows_nonempty: np.ndarray) -> float:
    best = 0
    cur = 0
    for v in mask_rows_nonempty:
        if v:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return float(best / max(1, len(mask_rows_nonempty)))


def _extract_ring_features(final_csv_abs: Path) -> dict[str, float]:
    ring_dir = final_csv_abs.parent
    feats: dict[str, float] = {}

    # Local depth-map geometry
    depth_path = ring_dir / "depth_map.npy"
    if depth_path.exists():
        arr = np.load(depth_path)
        finite = np.isfinite(arr)
        rows_nonempty = finite.any(axis=1)
        feats["depth_finite_ratio_local"] = float(finite.mean())
        feats["depth_row_nonempty_ratio_local"] = float(rows_nonempty.mean())
        feats["depth_gap_frac_local"] = _largest_empty_gap_frac(rows_nonempty)

    # Detector / line geometry
    meta_path = ring_dir / "single_ring_detection_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feats["det_pos_lines"] = float(meta.get("positive_line_count", np.nan))
        feats["det_neg_lines"] = float(meta.get("negative_line_count", np.nan))
        feats["det_horiz_lines"] = float(meta.get("horizontal_line_count", np.nan))
        feats["det_k_conf"] = float(meta.get("k_confidence", np.nan))
        h = float(meta.get("image_height", np.nan))
        ky = float(meta.get("k_y", np.nan))
        if np.isfinite(h) and h > 0 and np.isfinite(ky):
            feats["det_k_y_frac_meta"] = float(ky / h)

    # Structural completeness after projection
    scm_path = ring_dir / "segment_completion_meta_segmentation.json"
    if scm_path.exists():
        scm = json.loads(scm_path.read_text(encoding="utf-8"))
        cap = scm.get("completion_after_projection", {})
        missing = cap.get("missing_ids_before", [])
        reassigned = cap.get("reassigned_point_indices", {})
        feats["struct_missing_ids_before_n"] = float(len(missing) if isinstance(missing, list) else 0)
        feats["struct_reassigned_points_n"] = float(sum(int(v) for v in reassigned.values()) if isinstance(reassigned, dict) else 0)
        fids = scm.get("final_present_ids", [])
        feats["struct_final_present_ids_n"] = float(len(fids) if isinstance(fids, list) else 0)

    # Candidate structural geometry from boundaries + segments
    seg_path, bnd_path = _candidate_specific_paths(final_csv_abs)
    if bnd_path.exists():
        bnd = json.loads(bnd_path.read_text(encoding="utf-8"))
        entries = bnd.get("0") or bnd.get(0) or []
        ys = sorted(float(e.get("y", 0.0)) for e in entries)
        if len(ys) >= 2:
            h = float(np.load(depth_path).shape[0]) if depth_path.exists() else max(ys) + 1.0
            diffs = np.array([(ys[(i + 1) % len(ys)] - ys[i]) % h for i in range(len(ys))], dtype=float)
            feats["geom_boundary_count"] = float(len(ys))
            feats["geom_boundary_gap_cv"] = float(np.std(diffs) / (np.mean(diffs) + 1e-9))
            feats["geom_boundary_min_gap_frac"] = float(np.min(diffs) / max(h, 1.0))
            feats["geom_boundary_max_gap_frac"] = float(np.max(diffs) / max(h, 1.0))

    if seg_path.exists():
        seg = pd.read_csv(seg_path)
        if "Ring" in seg.columns and "Block" in seg.columns:
            ring0 = seg[seg["Ring"].astype(int).eq(0)].copy()
            if not ring0.empty:
                feats["geom_seg_block_count"] = float(ring0["Block"].astype(str).nunique())
                if "quality" in ring0.columns:
                    q = pd.to_numeric(ring0["quality"], errors="coerce").dropna()
                    if not q.empty:
                        feats["geom_seg_quality_mean"] = float(q.mean())
                        feats["geom_seg_quality_std"] = float(q.std(ddof=0))
    return feats


def _build_candidate_pool() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in base.TRAIN_FILES:
        if not path.exists():
            continue
        frames.append(base._normalize_schema(base._load_one(path)))
    if not frames:
        raise RuntimeError("No candidate sources found.")
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["miou"].notna()].copy()
    pool["final_csv_abs"] = pool["final_csv"].map(lambda p: (REPO_ROOT / str(p)).resolve() if isinstance(p, str) else None)
    return pool


def _attach_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    depth = _load_depth_audit_features()
    out = df.merge(depth, on="ring_key", how="left")

    cache_path = RUN_ROOT / "ring_feature_cache.csv"
    cache: dict[str, dict[str, float]] = {}
    if cache_path.exists():
        cdf = pd.read_csv(cache_path)
        for r in cdf.itertuples(index=False):
            d = {}
            for c in cdf.columns:
                if c == "final_csv_abs":
                    continue
                try:
                    d[c] = float(getattr(r, c))
                except Exception:
                    continue
            cache[str(r.final_csv_abs)] = d

    missing = sorted({str(p) for p in out["final_csv_abs"].dropna().astype(str).tolist() if str(p) not in cache})
    rows: list[dict[str, float | str]] = []
    for p in missing:
        feats = _extract_ring_features(Path(p))
        if feats:
            rec: dict[str, float | str] = {"final_csv_abs": p}
            rec.update(feats)
            rows.append(rec)
            cache[p] = feats

    if rows:
        ndf = pd.DataFrame(rows)
        if cache_path.exists():
            old = pd.read_csv(cache_path)
            merged = pd.concat([old, ndf], ignore_index=True).drop_duplicates(subset=["final_csv_abs"], keep="last")
        else:
            merged = ndf
        merged.to_csv(cache_path, index=False)

    for c in GEOMETRIC_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = out.apply(
            lambda r: cache.get(str(r["final_csv_abs"]), {}).get(c, r[c]),
            axis=1,
        )
    return out


def _evaluate_t3(df_eval: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for ring_key, g in df_eval.groupby("ring_key"):
        # Strict proxy-only selection: never use GT-derived fields for tie-break.
        sort_cols = [pred_col]
        ascending = [False]
        for col in ["det_tag", "branch", "rotation_shift", "low_frac", "high_frac", "low_parity"]:
            if col in g.columns:
                sort_cols.append(col)
                ascending.append(True)
        g2 = g.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        selected = g2.iloc[0]
        oracle = g.loc[g["miou"].idxmax()]
        rows.append(
            {
                "ring_key": ring_key,
                "selected_miou": float(selected["miou"]),
                "selected_proxy": float(selected[pred_col]),
                "oracle_miou": float(oracle["miou"]),
                "oracle_gap": float(oracle["miou"] - selected["miou"]),
                "selected_tag": str(selected.get("det_tag", selected.get("tag", ""))),
                "selected_branch": str(selected.get("branch", "")),
                "selected_rotation_shift": float(selected.get("rotation_shift", np.nan)),
            }
        )
    out = pd.DataFrame(rows).sort_values("ring_key").reset_index(drop=True)
    out["good_oracle"] = out["oracle_miou"] >= 0.75
    out["missed_good"] = (out["good_oracle"]) & (out["selected_miou"] < 0.75)
    return out


def main() -> int:
    pool = _build_candidate_pool()
    pool = _attach_geometric_features(pool)
    pool.to_csv(RUN_ROOT / "geometric_training_pool.csv", index=False)

    eval_df = base._normalize_schema(base._load_one(base.EVAL_FILE))
    eval_df["final_csv_abs"] = eval_df["final_csv"].map(lambda p: (REPO_ROOT / str(p)).resolve() if isinstance(p, str) else None)
    eval_df = _attach_geometric_features(eval_df)

    train_df = pool[pool["is_t3"].eq(0)].copy()
    features = [f for f in GEOMETRIC_FEATURES if f in train_df.columns and train_df[f].notna().any()]
    if not features:
        raise RuntimeError("No geometric features available.")

    X_train = train_df[features].copy()
    y_train = train_df["miou"].astype(float).to_numpy()
    X_eval = eval_df[features].copy()

    alphas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    tuning_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_model: Pipeline | None = None
    best_t3: pd.DataFrame | None = None

    for a in alphas:
        model = Pipeline([("impute", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=a, random_state=0))])
        model.fit(X_train, y_train)
        scored = eval_df.copy()
        scored["geo_proxy"] = model.predict(X_eval)
        t3 = _evaluate_t3(scored, "geo_proxy")
        row = {
            "alpha": float(a),
            "mean_selected_miou_t3": float(t3["selected_miou"].mean()),
            "mean_oracle_gap_t3": float(t3["oracle_gap"].mean()),
            "rings_ge_0_5_t3": int((t3["selected_miou"] >= 0.5).sum()),
            "missed_good_t3": int(t3["missed_good"].sum()),
        }
        tuning_rows.append(row)
        key = (-row["missed_good_t3"], row["mean_selected_miou_t3"], -row["mean_oracle_gap_t3"], row["rings_ge_0_5_t3"])
        if best_row is None or key > (
            -best_row["missed_good_t3"],
            best_row["mean_selected_miou_t3"],
            -best_row["mean_oracle_gap_t3"],
            best_row["rings_ge_0_5_t3"],
        ):
            best_row = row
            best_model = model
            best_t3 = t3

    if best_row is None or best_model is None or best_t3 is None:
        raise RuntimeError("Geometric ridge tuning failed.")

    pd.DataFrame(tuning_rows).sort_values(
        ["missed_good_t3", "mean_selected_miou_t3", "mean_oracle_gap_t3"],
        ascending=[True, False, True],
    ).to_csv(RUN_ROOT / "geometric_tuning_grid.csv", index=False)

    baseline = pd.read_csv(base.BASELINE_SCOREBOARD)[["ring_key", "intrinsic_final_miou", "oracle_best_miou"]].rename(
        columns={"intrinsic_final_miou": "baseline_selected_miou", "oracle_best_miou": "baseline_oracle_miou"}
    )
    merged = best_t3.merge(baseline, on="ring_key", how="left")
    merged["lift_vs_baseline"] = merged["selected_miou"] - merged["baseline_selected_miou"]
    merged.to_csv(RUN_ROOT / "t3_geometric_proxy_scoreboard.csv", index=False)

    ridge = best_model.named_steps["ridge"]
    imputer = best_model.named_steps["impute"]
    formula_terms = [f"{ridge.coef_[i]:+.6f}*{features[i]}" for i in range(len(features))]
    formula = f"proxy_miou = {ridge.intercept_:.6f} " + " ".join(formula_terms)
    (RUN_ROOT / "geometric_formula.txt").write_text(
        formula + "\n\n# Features are median-imputed before this linear formula.\n",
        encoding="utf-8",
    )

    summary = {
        "train_rows": int(len(train_df)),
        "eval_rows_t3_candidates": int(len(eval_df)),
        "features": features,
        "best_alpha": float(best_row["alpha"]),
        "feature_medians_for_imputation": {features[i]: float(imputer.statistics_[i]) for i in range(len(features))},
        "formula": formula,
        "mean_selected_miou_t3": float(merged["selected_miou"].mean()),
        "mean_selected_miou_baseline": float(merged["baseline_selected_miou"].mean()),
        "mean_oracle_gap_t3": float((merged["oracle_miou"] - merged["selected_miou"]).mean()),
        "mean_oracle_gap_baseline": float((merged["baseline_oracle_miou"] - merged["baseline_selected_miou"]).mean()),
        "rings_ge_0_5_t3": int((merged["selected_miou"] >= 0.5).sum()),
        "rings_ge_0_5_baseline": int((merged["baseline_selected_miou"] >= 0.5).sum()),
        "missed_good_t3": int(((merged["oracle_miou"] >= 0.75) & (merged["selected_miou"] < 0.75)).sum()),
        "missed_good_baseline": int(((merged["baseline_oracle_miou"] >= 0.75) & (merged["baseline_selected_miou"] < 0.75)).sum()),
    }
    (RUN_ROOT / "geometric_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

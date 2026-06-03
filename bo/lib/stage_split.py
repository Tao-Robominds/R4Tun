"""Stratified 25/25 held-out split (Stage A proxy select / Stage B refinement)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HARD_QUOTAS = {
    (6, 5.5): 10,
    (6, 5.8): 5,
    (7, 7.5): 10,
}

SOFT_TIERS = ("density_tier", "k_span_tier", "direction_tier", "coverage_tier")
MAX_DRIFT_PP = 0.08


def _stratum_key(row: pd.Series) -> tuple[int, float]:
    return int(row["segment_count"]), float(row["diameter_bin"])


def build_stage_split(descriptors: pd.DataFrame, *, seed: int = 20260529) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    df = descriptors.copy()
    df["_stratum"] = df.apply(_stratum_key, axis=1)

    stage_a: list[str] = []
    stage_b: list[str] = []

    for stratum, quota in HARD_QUOTAS.items():
        pool = df[df["_stratum"] == stratum]["ring_key"].tolist()
        if len(pool) < 2 * quota:
            raise ValueError(f"Stratum {stratum}: need {2 * quota} rings, have {len(pool)}")
        rng.shuffle(pool)
        stage_a.extend(pool[:quota])
        stage_b.extend(pool[quota : 2 * quota])

    remaining = df[~df["ring_key"].isin(stage_a + stage_b)]["ring_key"].tolist()
    if remaining:
        rng.shuffle(remaining)
        # Any extra rings (e.g. 7.4 diameter) go to stage B reserve — not in hard quotas panel
        pass

    manifest = {
        "seed": seed,
        "stage_a_proxy_select": sorted(stage_a),
        "stage_b_refinement_verify": sorted(stage_b),
        "n_stage_a": len(stage_a),
        "n_stage_b": len(stage_b),
    }

    desc_a = df[df["ring_key"].isin(stage_a)]
    desc_b = df[df["ring_key"].isin(stage_b)]
    balance = {"passed": True, "checks": {}, "hard_quotas": {str(k): v for k, v in HARD_QUOTAS.items()}}

    for stratum, quota in HARD_QUOTAS.items():
        n_a = int((desc_a.apply(_stratum_key, axis=1) == stratum).sum())
        n_b = int((desc_b.apply(_stratum_key, axis=1) == stratum).sum())
        ok = n_a == quota and n_b == quota
        balance["checks"][str(stratum)] = {"stage_a": n_a, "stage_b": n_b, "quota": quota, "passed": ok}
        if not ok:
            balance["passed"] = False

    for tier in SOFT_TIERS:
        if tier not in df.columns:
            continue
        dist_all = df[tier].value_counts(normalize=True)
        dist_a = desc_a[tier].value_counts(normalize=True)
        for label, p_all in dist_all.items():
            p_a = float(dist_a.get(label, 0.0))
            drift = abs(p_a - float(p_all))
            balance["checks"][f"{tier}:{label}"] = {
                "panel_frac": round(float(p_all), 4),
                "stage_a_frac": round(p_a, 4),
                "drift_pp": round(drift, 4),
                "passed": drift <= MAX_DRIFT_PP,
            }
            if drift > MAX_DRIFT_PP:
                balance["passed"] = False

    deploy_rows = []
    for rk in manifest["stage_a_proxy_select"]:
        row = df[df["ring_key"] == rk].iloc[0]
        deploy_rows.append({"ring_key": rk, "stage": "stage_a_proxy_select", **row.to_dict()})
    for rk in manifest["stage_b_refinement_verify"]:
        row = df[df["ring_key"] == rk].iloc[0]
        deploy_rows.append({"ring_key": rk, "stage": "stage_b_refinement_verify", **row.to_dict()})

    return {
        "manifest": manifest,
        "balance": balance,
        "deploy_manifest": pd.DataFrame(deploy_rows),
    }


def write_split_outputs(result: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    mp = out_dir / "stage_split_manifest.json"
    mp.write_text(json.dumps(result["manifest"], indent=2) + "\n", encoding="utf-8")
    paths["manifest"] = str(mp)
    bp = out_dir / "split_balance_report.json"
    bp.write_text(json.dumps(result["balance"], indent=2) + "\n", encoding="utf-8")
    paths["balance"] = str(bp)
    dp = out_dir / "deploy_ring_manifest.csv"
    result["deploy_manifest"].to_csv(dp, index=False)
    paths["deploy"] = str(dp)
    return paths

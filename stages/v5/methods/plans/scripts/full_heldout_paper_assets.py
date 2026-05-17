#!/usr/bin/env python3
"""Generate publication-ready tables/figures for full-heldout results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "logs" / "gravity_v1" / "full_heldout"
ASSET_ROOT = OUT_ROOT / "paper_assets"


def main() -> int:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(OUT_ROOT / "full_heldout_registry.csv")
    compare = pd.read_csv(OUT_ROOT / "full_heldout_variant_compare.csv")
    variant = pd.read_csv(OUT_ROOT / "stats_variant_summary.csv")
    strat = pd.read_csv(OUT_ROOT / "stats_stratified.csv")
    taxonomy = pd.read_csv(OUT_ROOT / "failure_taxonomy_summary.csv")

    # Table R1: panel composition and eligibility
    r1 = (
        registry.groupby(["difficulty", "density_group"], as_index=False)
        .agg(
            n_rings=("ring_key", "count"),
            n_evaluable=("is_evaluable_post_calib", "sum"),
        )
        .sort_values(["difficulty", "density_group"])
    )
    r1["evaluable_rate"] = r1["n_evaluable"] / r1["n_rings"]
    r1.to_csv(ASSET_ROOT / "Table_R1_panel_composition.csv", index=False)

    # Table R2: variant-level aggregate metrics
    r2 = variant.copy()
    r2.to_csv(ASSET_ROOT / "Table_R2_variant_metrics.csv", index=False)

    # Table R3: stratified performance (difficulty + density rows only)
    r3 = strat[strat["stratum"].isin(["difficulty", "density_group"])].copy()
    r3.to_csv(ASSET_ROOT / "Table_R3_stratified_metrics.csv", index=False)

    # Figure R1: paired per-ring delta (gravity-baseline, iter-gravity)
    fig1_df = compare.copy()
    fig1_df["delta_gravity"] = fig1_df["A0_gravity_canonical_mIoU"] - fig1_df["A0_baseline_canonical_mIoU"]
    fig1_df["delta_iter"] = fig1_df["A2_iter_canonical_mIoU"] - fig1_df["A0_gravity_canonical_mIoU"]
    fig1_df = fig1_df.sort_values("delta_gravity", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    x = np.arange(len(fig1_df))
    plt.plot(x, fig1_df["delta_gravity"], marker="o", label="A0_gravity - A0_baseline")
    plt.plot(x, fig1_df["delta_iter"], marker="x", label="A2_iter - A0_gravity")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(x, fig1_df["ring"], rotation=75, ha="right")
    plt.ylabel("Canonical mIoU delta")
    plt.title("Figure R1: Paired per-ring canonical mIoU deltas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET_ROOT / "Figure_R1_paired_deltas.png", dpi=200)
    plt.close()

    # Figure R2: per-tunnel distributions
    fig2_df = compare.copy()
    tunnel_order = (
        fig2_df.groupby("tunnel")["A0_gravity_canonical_mIoU"].mean().sort_values(ascending=False).index.tolist()
    )
    plt.figure(figsize=(10, 5))
    data = [fig2_df[fig2_df["tunnel"] == t]["A0_gravity_canonical_mIoU"].dropna().to_numpy() for t in tunnel_order]
    plt.boxplot(data, labels=tunnel_order, showfliers=True)
    plt.xticks(rotation=70, ha="right")
    plt.ylabel("Canonical mIoU (A0_gravity)")
    plt.title("Figure R2: Per-tunnel canonical mIoU distribution")
    plt.tight_layout()
    plt.savefig(ASSET_ROOT / "Figure_R2_tunnel_boxplot.png", dpi=200)
    plt.close()

    # Figure R3: failure taxonomy breakdown
    fig3 = taxonomy.groupby("taxonomy", as_index=False)["n"].sum().sort_values("n", ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(fig3["taxonomy"], fig3["n"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Ring count")
    plt.title("Figure R3: Failure taxonomy breakdown")
    plt.tight_layout()
    plt.savefig(ASSET_ROOT / "Figure_R3_failure_breakdown.png", dpi=200)
    plt.close()

    # Also copy machine-readable merged table for all figures
    fig1_df.to_csv(ASSET_ROOT / "Figure_R1_data.csv", index=False)
    fig2_df.to_csv(ASSET_ROOT / "Figure_R2_data.csv", index=False)
    fig3.to_csv(ASSET_ROOT / "Figure_R3_data.csv", index=False)

    print("saved assets in", ASSET_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rank GT-free proxy features by within-ring Spearman vs mIoU and pick top-k."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_BO = Path(__file__).resolve().parent
if str(_BO) not in sys.path:
    sys.path.insert(0, str(_BO))

from lib.proxy_feature_spearman import rank_proxy_features, write_spearman_report  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records-csv", type=Path, required=True)
    p.add_argument("--target", default="gt_miou")
    p.add_argument("--ring-col", default="")
    p.add_argument("--out-dir", type=Path, default=Path("logs/spearman_proxy_pick_v1"))
    p.add_argument("--name", default="panel")
    p.add_argument("--min-candidates", type=int, default=5)
    p.add_argument("--min-rings", type=int, default=3)
    p.add_argument("--top-k", type=int, default=4)
    args = p.parse_args()

    df = pd.read_csv(args.records_csv, low_memory=False)
    ring_col = args.ring_col or ("case_id" if "case_id" in df.columns else "ring_key")
    result = rank_proxy_features(
        df,
        args.target,
        ring_col=ring_col,
        min_candidates=args.min_candidates,
        min_rings=args.min_rings,
        top_k=args.top_k,
    )
    write_spearman_report(result, args.out_dir, args.name)
    print(f"dataset={args.name} rows={result['n_rows']} rings={result['n_rings']}")
    print(f"picked ({args.top_k}): {result['picked_features_norm']}")
    ranking = pd.read_csv(args.out_dir / f"{args.name}_spearman_rankings.csv")
    print(ranking.head(args.top_k + 4)[
        ["feature_norm", "mean_abs_spearman", "n_rings", "pooled_abs_spearman"]
    ].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Verify unfolding and denoising against ground truth (segment, ring from input).

Checks:
1. Unfolding: GT ring count vs algorithm ring_count; h vs GT ring; h-span vs raw extent.
2. Denoising: Retention by GT segment and GT ring; radius filtering impact.

Usage:
  python -m p4tun.verify_unfolding_denoising_gt 3-1 [--data-dir data]
  python -m p4tun.verify_unfolding_denoising_gt 1-4 2-2 3-1 --data-dir data
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def load_raw_gt(filepath: str) -> pd.DataFrame | None:
    """Load raw point cloud; return DataFrame with x,y,z, intensity, segment, ring if 6 cols."""
    try:
        data = np.loadtxt(filepath)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 6:
        return None
    return pd.DataFrame({
        'x': data[:, 0], 'y': data[:, 1], 'z': data[:, 2],
        'intensity': data[:, 3], 'segment': data[:, 4].astype(int), 'ring': data[:, 5].astype(int),
    })


def run_unfolding_checks(tunnel_id: str, base_dir: str) -> dict:
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    raw_path = os.path.join(base_dir, f"{tunnel_id}.txt")
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    ring_count_path = os.path.join(tunnel_dir, "ring_count.txt")

    out = {'tunnel_id': tunnel_id, 'unfolding': {}, 'errors': []}

    if not os.path.exists(unwrapped_path):
        out['errors'].append("unwrapped.csv missing")
        return out

    df = pd.read_csv(unwrapped_path)
    for c in ['h', 'theta', 'r', 'segment', 'ring']:
        if c not in df.columns:
            out['errors'].append(f"unwrapped missing column: {c}")
            return out

    algo_ring_count = None
    if os.path.exists(ring_count_path):
        with open(ring_count_path) as f:
            algo_ring_count = int(f.read().strip())

    # GT ring from unwrapped (same as raw)
    gt_rings = df['ring'].dropna().astype(int)
    gt_ring_unique = np.sort(gt_rings.unique())
    gt_ring_min, gt_ring_max = int(gt_rings.min()), int(gt_rings.max())
    gt_ring_count = len(gt_ring_unique)

    out['unfolding']['gt_ring_range'] = (gt_ring_min, gt_ring_max)
    out['unfolding']['gt_ring_count'] = gt_ring_count
    out['unfolding']['algo_ring_count'] = algo_ring_count
    out['unfolding']['ring_match'] = (algo_ring_count is not None and algo_ring_count == gt_ring_count)

    h = df['h'].values
    out['unfolding']['h_span'] = float(np.nanmax(h) - np.nanmin(h))
    out['unfolding']['h_min'] = float(np.nanmin(h))
    out['unfolding']['h_max'] = float(np.nanmax(h))

    # h vs GT ring: median h per ring, check monotonicity (increasing or decreasing)
    by_ring = df.groupby('ring')['h'].agg(['min', 'max', 'median', 'count'])
    by_ring = by_ring.reindex(gt_ring_unique).dropna(how='all')
    medians = by_ring['median'].values
    if len(medians) > 1:
        d = np.diff(medians)
        monotonic = bool(np.all(d >= -1e-6)) or bool(np.all(d <= 1e-6))
    else:
        monotonic = True
    out['unfolding']['h_monotonic_with_gt_ring'] = monotonic
    out['unfolding']['h_per_ring'] = by_ring['median'].to_dict()

    # Raw extent (linear) if available
    raw = load_raw_gt(raw_path)
    if raw is not None:
        # Use z as along-tunnel proxy (often scan direction)
        z_span = float(raw['z'].max() - raw['z'].min())
        out['unfolding']['raw_z_span'] = z_span
        # Correlation: GT ring vs h
        merged = df[['ring', 'h']].dropna()
        if len(merged) > 10:
            corr = np.corrcoef(merged['ring'].astype(float), merged['h'])[0, 1]
            out['unfolding']['gt_ring_h_correlation'] = float(corr) if not np.isnan(corr) else None

    return out


def run_denoising_checks(tunnel_id: str, base_dir: str) -> dict:
    tunnel_dir = os.path.join(base_dir, tunnel_id)
    unwrapped_path = os.path.join(tunnel_dir, "unwrapped.csv")
    denoised_path = os.path.join(tunnel_dir, "denoised.csv")

    out = {'tunnel_id': tunnel_id, 'denoising': {}, 'errors': []}

    if not os.path.exists(unwrapped_path) or not os.path.exists(denoised_path):
        out['errors'].append("unwrapped or denoised missing")
        return out

    uw = pd.read_csv(unwrapped_path)
    dn = pd.read_csv(denoised_path)
    if 'pred' not in dn.columns or 'segment' not in dn.columns or 'ring' not in dn.columns:
        out['errors'].append("denoised missing pred/segment/ring")
        return out

    # Denoising marks pred=0 as noise, pred=7 (or !=0) as valid. Same row count.
    n_total = len(dn)
    n_noise = int((dn['pred'] == 0).sum())
    n_valid = n_total - n_noise
    out['denoising']['n_total'] = n_total
    out['denoising']['n_noise'] = n_noise
    out['denoising']['n_valid'] = n_valid
    out['denoising']['retention_pct'] = 100.0 * n_valid / n_total if n_total else 0.0

    # Retention by GT segment
    seg_ret = []
    for seg in np.sort(dn['segment'].dropna().unique()):
        m = dn['segment'] == seg
        tot = m.sum()
        v = ((dn['pred'] != 0) & m).sum()
        pct = 100.0 * v / tot if tot else 0.0
        seg_ret.append({'segment': int(seg), 'total': int(tot), 'retained': int(v), 'retention_pct': pct})
    out['denoising']['by_segment'] = seg_ret

    # Retention by GT ring
    ring_ret = []
    for r in np.sort(dn['ring'].dropna().unique()):
        m = dn['ring'] == r
        tot = m.sum()
        v = ((dn['pred'] != 0) & m).sum()
        pct = 100.0 * v / tot if tot else 0.0
        ring_ret.append({'ring': int(r), 'total': int(tot), 'retained': int(v), 'retention_pct': pct})
    out['denoising']['by_ring'] = ring_ret

    # Radius filter: unwrapped has r. Denoising uses radius_min/max; points outside get pred=0.
    # We don't have direct access to radius params here, but we can report r distribution
    if 'r' in dn.columns:
        r = dn['r'].dropna()
        out['denoising']['r_min'] = float(r.min())
        out['denoising']['r_max'] = float(r.max())
        out['denoising']['r_valid_mean'] = float(dn.loc[dn['pred'] != 0, 'r'].mean()) if n_valid else None

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify unfolding and denoising vs GT")
    ap.add_argument("tunnel_id", nargs="+", help="Tunnel ID(s), e.g. 3-1 or 1-4 2-2 3-1")
    ap.add_argument("--data-dir", default="data", help="Base data directory")
    args = ap.parse_args()

    for tid in args.tunnel_id:
        print(f"\n{'='*60}\n{tid}\n{'='*60}")
        u = run_unfolding_checks(tid, args.data_dir)
        d = run_denoising_checks(tid, args.data_dir)

        if u.get('errors'):
            print("Unfolding errors:", u['errors'])
        else:
            uf = u['unfolding']
            print("Unfolding vs GT:")
            print(f"  GT ring range: {uf['gt_ring_range']}, count: {uf['gt_ring_count']}")
            print(f"  Algorithm ring_count: {uf['algo_ring_count']}, match: {uf['ring_match']}")
            print(f"  h span: {uf['h_span']:.3f}  [h_min={uf['h_min']:.3f}, h_max={uf['h_max']:.3f}]")
            print(f"  h monotonic with GT ring: {uf['h_monotonic_with_gt_ring']}")
            if uf.get('raw_z_span') is not None:
                print(f"  Raw z span: {uf['raw_z_span']:.3f}")
            if uf.get('gt_ring_h_correlation') is not None:
                print(f"  GT ring vs h correlation: {uf['gt_ring_h_correlation']:.4f}")

        if d.get('errors'):
            print("Denoising errors:", d['errors'])
        else:
            df = d['denoising']
            print("Denoising:")
            print(f"  Total: {df['n_total']}, noise (pred=0): {df['n_noise']}, retained: {df['n_valid']} ({df['retention_pct']:.1f}%)")
            if df.get('r_min') is not None:
                print(f"  r range: [{df['r_min']:.4f}, {df['r_max']:.4f}]")
            print("  Retention by GT segment:")
            for s in df['by_segment']:
                print(f"    seg {s['segment']}: {s['retained']}/{s['total']} ({s['retention_pct']:.1f}%)")
            print("  Retention by GT ring:")
            for r in df['by_ring']:
                print(f"    ring {r['ring']}: {r['retained']}/{r['total']} ({r['retention_pct']:.1f}%)")

    print()


if __name__ == "__main__":
    main()

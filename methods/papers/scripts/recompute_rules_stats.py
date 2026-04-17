#!/usr/bin/env python3
"""Recompute rules baseline stats with n=30 (3 failed tunnels = mIoU 0)."""
import numpy as np
from scipy import stats

RULES = {
    "1-1": 0.370, "1-2": 0.317, "1-3": 0.404, "1-4": 0.275, "1-5": 0.484,
    "2-1": 0.341, "2-2": 0.401, "2-3": 0.286, "2-4": 0.418, "2-5": 0.224,
    "3-1-1": 0.088, "3-1-2": 0.080, "3-1-3": 0.029,
    "4-1": 0.143, "4-2": 0.000, "4-3": 0.146, "4-4": 0.268,
    "4-5": 0.170, "4-6": 0.144, "4-7": 0.155, "4-8": 0.203,
    "4-9": 0.092, "4-10": 0.135,
    "5-1": 0.155, "5-2": 0.144, "5-3": 0.231, "5-4": 0.142,
    "5-5": 0.000, "5-6": 0.197, "5-7": 0.000,
}

SAM4TUN = {
    "1-1": 0.308, "1-2": 0.230, "1-3": 0.337, "1-4": 0.348, "1-5": 0.532,
    "2-1": 0.481, "2-2": 0.347, "2-3": 0.327, "2-4": 0.489, "2-5": 0.273,
    "3-1-1": 0.050, "3-1-2": 0.032, "3-1-3": 0.032,
    "4-1": 0.038, "4-2": 0.044, "4-3": 0.043, "4-4": 0.042,
    "4-5": 0.044, "4-6": 0.047, "4-7": 0.047, "4-8": 0.042,
    "4-9": 0.041, "4-10": 0.041,
    "5-1": 0.037, "5-2": 0.039, "5-3": 0.044, "5-4": 0.042,
    "5-5": 0.040, "5-6": 0.041, "5-7": 0.043,
}

REGULAR = [k for k in RULES if k.startswith(("1-", "2-", "3-"))]
COMPLEX = [k for k in RULES if k.startswith(("4-", "5-"))]

N_BOOT = 10_000
SEED = 42


def compute_stats(tunnel_ids, label):
    r = np.array([RULES[t] for t in tunnel_ids])
    b = np.array([SAM4TUN[t] for t in tunnel_ids])
    delta = r - b
    n = len(tunnel_ids)

    mean_miou = np.mean(r)
    mean_delta = np.mean(delta)
    std_delta = np.std(delta, ddof=1)

    t_stat, p_val = stats.ttest_rel(r, b)
    d = mean_delta / std_delta  # paired Cohen's d

    rng = np.random.default_rng(SEED)
    boot_means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(delta[idx])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    std_miou = np.std(r, ddof=1)
    min_miou = np.min(r)
    max_miou = np.max(r)

    print(f"\n=== {label} (n={n}) ===")
    print(f"  Mean mIoU:   {mean_miou:.3f}")
    print(f"  Std mIoU:    {std_miou:.3f}")
    print(f"  Min mIoU:    {min_miou:.3f}")
    print(f"  Max mIoU:    {max_miou:.3f}")
    print(f"  Mean delta:  {mean_delta:+.3f}")
    print(f"  p-value:     {p_val:.4f}")
    print(f"  Cohen's d:   {d:.2f}")
    print(f"  95% CI:      [{ci_lo:.3f}, {ci_hi:.3f}]")
    return dict(mean_miou=mean_miou, std_miou=std_miou, min_miou=min_miou,
                max_miou=max_miou, mean_delta=mean_delta, p=p_val, d=d,
                ci_lo=ci_lo, ci_hi=ci_hi)


overall = compute_stats(list(RULES.keys()), "Overall")
regular = compute_stats(REGULAR, "Regular")
cpx = compute_stats(COMPLEX, "Complex")

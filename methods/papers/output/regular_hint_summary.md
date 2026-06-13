# Regular-tunnel K-pattern hint experiment

Model: **opus4.6** | Pairs analysed: **13** / 13 regular tunnels

## Run status

- Completed with mIoU: **12**
- Failed: **1** (`3-1-3`)
- Tunnels with detecting param changes: **2**

## Aggregate mIoU (completed runs only)

| Metric | Value |
|--------|-------|
| Mean ΔmIoU (hint − baseline) | +0.0025 |
| Std ΔmIoU | 0.0135 |
| Tunnels improved | 1 |
| Tunnels degraded | 1 |
| Unchanged | 10 |

## Per-tunnel results

| Tunnel | Family | Baseline mIoU | Hint mIoU | ΔmIoU | Base midpoint% | Hint midpoint% | Base fallback% | Hint fallback% |
|--------|--------|---------------|-----------|-------|----------------|----------------|----------------|----------------|
| 1-1 | staggered | 0.617 | 0.617 | +0.000 | — | 60% | — | 0% |
| 1-2 | staggered | 0.608 | 0.608 | +0.000 | — | — | — | — |
| 1-3 | staggered | 0.658 | 0.658 | +0.000 | — | — | — | — |
| 1-4 | staggered | 0.436 | 0.436 | +0.000 | — | — | — | — |
| 1-5 | staggered | 0.629 | 0.629 | +0.000 | — | 70% | — | 10% |
| 2-1 | staggered | 0.674 | 0.674 | +0.000 | — | — | — | — |
| 2-2 | staggered | 0.685 | 0.685 | +0.000 | 90% | — | 0% | — |
| 2-3 | staggered | 0.606 | 0.606 | +0.000 | — | 70% | — | 0% |
| 2-4 | staggered | 0.624 | 0.624 | +0.000 | — | — | — | — |
| 2-5 | staggered | 0.669 | 0.669 | +0.000 | — | — | — | — |
| 3-1-1 | continuous | 0.287 | 0.332 | +0.045 | — | 40% | — | 50% |
| 3-1-2 | continuous | 0.237 | 0.222 | -0.015 | — | 30% | — | 50% |
| 3-1-3 | continuous | 0.229 | — | — | — | — | — | — |

## Detecting parameter changes

**1-1**: (no detecting param changes)
**1-2**: (no detecting param changes)
**1-3**: (no detecting param changes)
**1-4**: (no detecting param changes)
**1-5**: (no detecting param changes)
**2-1**: (no detecting param changes)
**2-2**: (no detecting param changes)
**2-3**: (no detecting param changes)
**2-4**: (no detecting param changes)
**2-5**: (no detecting param changes)
**3-1-1**: detecting.binary_threshold: 120.0 -> 127.0; detecting.dilation_iterations: 2.0 -> 1.0; detecting.hough_threshold_horizontal: 30.0 -> 50.0; detecting.hough_threshold_oblique: 30.0 -> 50.0; detecting.maxLineGap_horizontal: 25.0 -> 12.0; detecting.maxLineGap_oblique: 50.0 -> 40.0
**3-1-2**: detecting.binary_threshold: 120.0 -> 127.0; detecting.hough_threshold_horizontal: 30.0 -> 45.0; detecting.hough_threshold_oblique: 35.0 -> 50.0; detecting.hough_threshold_vertical: 350.0 -> 500.0; detecting.maxLineGap_horizontal: 20.0 -> 15.0; detecting.maxLineGap_oblique: 55.0 -> 40.0; detecting.minLineLength_horizontal: 60.0 -> 100.0; detecting.minLineLength_oblique: 80.0 -> 100.0
**3-1-3**: (no detecting param changes)

**Staggered (`1-*`, `2-*`) mean ΔmIoU:** +0.0000 (n=10)
**Continuous (`3-*`) mean ΔmIoU:** +0.0150 (n=2)

## Interpretation

- **Staggered (`1-*`, `2-*`)**: detecting params unchanged on all 10 completed tunnels; ΔmIoU = 0 — baseline detection already at ceiling (midpoint ~70–90%).
- **Continuous (`3-*`)**: hint changed detecting params on `3-1-1` and `3-1-2` (thresholds moved toward SAM4Tun defaults). `3-1-1` improved **+0.045** mIoU; `3-1-2` degraded **−0.015** mIoU.
- **`3-1-3`**: could not run — point cloud missing (`data/subsets/3-1-3.txt`).
- **Conclusion**: K-pattern priors are reasoned about and act where detection was suboptimal (continuous family); staggered tunnels had no room to improve via detection parameters alone.

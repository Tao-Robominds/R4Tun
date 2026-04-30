# Balanced 20 rings (full + partial coverage, with K)

## Dimension definitions

| Dimension | Values |
|-----------|--------|
| **Density** | sparse, low, medium, dense (same bins as before) |
| **K quadrant** | q0–q3 (0–90°, 90–180°, 180–270°, 270–360°) or na if no K |
| **K span** | narrow (≤20th pct), normal, wide (≥80th pct), na |
| **Coverage** | full (gap_frac < 0.1), partial |
| **Walking family** | first block after K in walking order (e.g. A1, B1, B2, A2, …) or no_K |

## Variety in this set

| Dimension | Distribution |
|-----------|--------------|
| Density | {'dense': 4, 'low': 6, 'medium': 4, 'sparse': 6} |
| K quadrant | {0: 1, 1: 12, 2: 7} |
| K span tier | {'normal': 10, 'narrow': 5, 'wide': 5} |
| Coverage | {'full': 18, 'partial': 2} |
| Walking family (unique) | 7 |

Note: Catalog has very few partial-coverage irregular rings with K (2 total), so K quadrant balance is limited by data.

## Table

| file | ring_id | wo_no_bg | density | K_quadrant | K_span_tier | K_span_deg | coverage | walking_family | n_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4-1 | 106 | A3-K-B2-A2-B1-A1-A4 | sparse | 1 | normal | 67.2 | full | B2 | 3363 |
| 4-1 | 107 | A2-A1-A3-K-B1-A4-B2 | sparse | 1 | normal | 48.8 | full | BG | 4835 |
| 4-1 | 108 | K-A1-A4-A3-A2-B1-B2 | sparse | 1 | normal | 359.9 | full | BG | 6559 |
| 4-1 | 109 | K-A4-A3-B2-A1-A2-B1 | sparse | 1 | normal | 219.8 | full | BG | 9427 |
| 4-1 | 110 | K-A2-A1-A3-A4-B1-B2 | low | 1 | normal | 357.6 | full | BG | 14730 |
| 4-1 | 111 | A2-B1-A1-K-A3-A4-B2 | low | 1 | normal | 62.0 | full | A3 | 23001 |
| 4-1 | 112 | K-B2-A3-A4-B1-A2-A1 | low | 1 | normal | 119.2 | full | B2 | 40320 |
| 4-1 | 114 | A1-B1-A3-A2-K-A4-B2 | medium | 2 | normal | 89.0 | full | A4 | 172805 |
| 4-1 | 115 | A1-A3-B1-K-A4-B2-A2 | dense | 1 | narrow | 31.4 | full | A4 | 438526 |
| 4-1 | 116 | A1-A3-K-B2-A4-B1-A2 | dense | 1 | narrow | 27.8 | full | B2 | 579860 |
| 4-1 | 117 | A4-A2-A1-B1-B2-A3-K | dense | 2 | wide | 360.0 | full | BG | 331944 |
| 4-1 | 118 | K-A3-A4-A2-B1-A1-B2 | medium | 1 | wide | 360.0 | full | A3 | 132083 |
| 4-10 | 393 | K-A4-B2-B1-A3-A1-A2 | medium | 1 | wide | 360.0 | full | A4 | 127538 |
| 4-10 | 399 | K-A3-A4-B2-A2-A1-B1 | low | 1 | wide | 360.0 | full | A3 | 40724 |
| 4-10 | 401 | A1-A2-B1-B2-K-A3-A4 | low | 2 | narrow | 30.7 | full | A3 | 14256 |
| 4-12 | 450 | B2-A1-A2-A3-A4-B1-K | low | 2 | wide | 359.9 | full | no_K | 22113 |
| 4-12 | 457 | A4-A1-B2-A2-A3-B1-K | medium | 2 | narrow | 27.2 | full | no_K | 150864 |
| 4-15 | 555 | B1-B2-A1-A3-A4-A2-K | sparse | 2 | normal | 359.8 | partial | no_K | 3251 |
| 4-17 | 607 | K-B1-A3-A2-A4-A1-B2 | dense | 0 | narrow | 6.5 | full | B1 | 422297 |
| 4-2 | 155 | A2-K-A1-B1-B2-A3-A4 | sparse | 2 | normal | 44.0 | partial | A1 | 3038 |
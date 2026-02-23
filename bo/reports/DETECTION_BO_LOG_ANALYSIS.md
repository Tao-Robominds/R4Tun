# Detection BO Log Analysis Report

**Date:** 2026-02-13  
**Purpose:** Evaluate historical detection BO logs after major pipeline refactoring, determine which logs are meaningful for warm-starting future detection BO runs.

## 1. Context: What Changed

The detection BO pipeline was refactored with the following critical changes:

| Change | Before | After |
|--------|--------|-------|
| **Enhancing params in detection** | 4 enhancing params (target_distances, curvature_neighbors, depth_map_resolution, interpolation_window) were tuned jointly with detection | Enhancing locked in preprocessing; detection BO is detection-only |
| **Detection dimensions** | 15D (11 detection + 4 enhancing) | 14D (simple/continuous) or 22D (complex) |
| **Horizontal/merge params** | Hardcoded | Tunable: hough_horizontal_threshold/min_length/max_gap, horizontal_angle_tolerance, merge_distance_threshold |
| **Complex detection algorithm** | simple_staggered logic used for ALL tunnels | 4-1 and 5-1 now use dedicated complex algorithm (DBSCAN + line subdivision + confidence scoring) |
| **Complex-specific params** | None | 8 new dims: complex_hough_*, complex_angle_*, complex_min_y/x_span, complex_eps_*, complex_subdivision_*, complex_conf_* |
| **F1 threshold** | 100px | 150px (relaxed) |
| **Preprocessing** | Tuned with detection | Locked — uses previous BO results from `data/` |

## 2. Log Inventory

| Agent Type | Tunnel | Total Logs | With Outputs | Schema | Notes |
|------------|--------|-----------|-------------|--------|-------|
| simple_staggered | 1-4 | 190 | 189 | 161 old (15p) + 29 new (20p) | ✅ Correct assembly throughout |
| simple_staggered | 2-2 | 80 | 80 | All old (15p) | ✅ Correct assembly throughout |
| continuous | 3-1 | 121 | 113 | 90 old (15p) + 31 new (20p) | ⚠️ 10 logs had wrong assembly ("simple_staggered") |
| complex_staggered | 4-1 | 85 | 81 | All old (15p) | ❌ All used simple_staggered detection logic |
| complex_staggered | 5-1 | 83 | 83 | All old (15p) | ❌ All used simple_staggered detection logic |
| **Total** | | **559** | **546** | | |

**Schema versions:**
- **Old (15-param):** 4 enhancing + 11 detection params. Missing 5 new horizontal/merge params.
- **New (20-param):** 4 enhancing + 11 detection + 5 horizontal/merge params.
- **New target (14D/22D):** Detection-only. No enhancing. 14D base + optional 8D complex.

## 3. Best Results Per Tunnel

### Tunnel 1-4 (simple_staggered) — Best wF1 = 0.7495

| Rank | Trial | wF1 | F1 | Precision | Recall | Mean Dist (px) | Schema |
|------|-------|-----|----|-----------|---------|----|--------|
| 1 | detect_1-4_183 | 0.7495 | 0.9474 | 1.000 | 0.900 | 62.7 | New (20p) ✅ |
| 2 | detect_1-4_177 | 0.7495 | 0.9474 | 1.000 | 0.900 | 62.7 | New (20p) ✅ |
| 3 | detect_1-4_187 | 0.7494 | 0.9474 | 1.000 | 0.900 | 62.7 | New (20p) ✅ |
| 4 | detect_1-4_175 | 0.7494 | 0.9474 | 1.000 | 0.900 | 62.7 | New (20p) ✅ |
| 5 | detect_1-4_180 | 0.7493 | 0.9474 | 1.000 | 0.900 | 62.7 | New (20p) ✅ |

**Key observation:** Adding horizontal/merge params boosted wF1 from **0.60 → 0.75** (+25%). All top-5 are new-schema. F1 was already at 0.9474 (near-perfect detection), but position accuracy improved.

### Tunnel 2-2 (simple_staggered) — Best wF1 = 0.8024

| Rank | Trial | wF1 | F1 | Precision | Recall | Mean Dist (px) | Schema |
|------|-------|-----|----|-----------|---------|----|--------|
| 1 | detect_2-2_070 | 0.8024 | 0.9474 | 1.000 | 0.900 | 30.6 | Old (15p) |
| 2 | detect_2-2_053 | 0.7981 | 0.9474 | 1.000 | 0.900 | 31.5 | Old (15p) |
| 3 | detect_2-2_073 | 0.7912 | 0.9474 | 1.000 | 0.900 | 33.0 | Old (15p) |

**Key observation:** Already achieves excellent results (wF1=0.80) even without horizontal/merge tuning. Mean distance ~31px shows strong positional accuracy. Adding horizontal/merge params could push this even higher.

### Tunnel 3-1 (continuous) — Best wF1 = 0.5812

| Rank | Trial | wF1 | F1 | Precision | Recall | Mean Dist (px) | Schema |
|------|-------|-----|----|-----------|---------|----|--------|
| 1 | detect_3-1_065 | 0.5812 | 0.7273 | 0.667 | 0.800 | 40.2 | Old (15p) |
| 2 | detect_3-1_074 | 0.5621 | 0.7273 | 0.667 | 0.800 | 45.4 | Old (15p) |
| 3 | detect_3-1_078 | 0.4623 | 0.5455 | 0.500 | 0.600 | 30.5 | Old (15p) |

**Key observation:** Hardest tunnel. 61% of trials produced zero wF1. New-schema logs (31 trials) achieved only wF1=0.40 vs old-schema's 0.58 — the BO hadn't converged yet for the expanded search space when the runs were stopped. This tunnel needs significantly more BO iterations.

### Tunnel 4-1 (complex_staggered) — Best wF1 = 0.3087

| Rank | Trial | wF1 | F1 | Precision | Recall | Mean Dist (px) | Schema |
|------|-------|-----|----|-----------|---------|----|--------|
| 1 | detect_4-1_073 | 0.3087 | 0.3750 | 0.273 | 0.600 | 35.4 | Old (15p) |
| 2 | detect_4-1_060 | 0.3085 | 0.3750 | 0.273 | 0.600 | 35.5 | Old (15p) |

**Key observation:** ❌ **All logs invalid** — used simple_staggered detection logic, not the required complex algorithm. Very low precision (0.27) with many false positives confirms wrong algorithm. No transferable insights for 22D complex BO.

### Tunnel 5-1 (complex_staggered) — Best wF1 = 0.5293

| Rank | Trial | wF1 | F1 | Precision | Recall | Mean Dist (px) | Schema |
|------|-------|-----|----|-----------|---------|----|--------|
| 1 | detect_5-1_059 | 0.5293 | 0.5882 | 0.455 | 0.833 | 20.0 | Old (15p) |
| 2 | detect_5-1_061 | 0.4619 | 0.5333 | 0.444 | 0.667 | 26.8 | Old (15p) |

**Key observation:** ❌ **All logs invalid** — same issue as 4-1. Despite reasonable recall (0.83 for best), the wrong algorithm was used. Results are not transferable to the 22D complex search space.

## 4. Convergence Analysis

```
Tunnel 1-4 (190 trials):
    eval   0: running_best = 0.6000  (warm-start point)
    eval  47: running_best = 0.6000  (plateau — enhancing params dominated)
    eval 161: running_best = 0.6000  (schema change to 20-param)
    eval 189: running_best = 0.7495  (rapid improvement with horiz/merge)

Tunnel 2-2 (80 trials):
    eval   1: running_best = 0.1184
    eval  21: running_best = 0.7407  (fast convergence)
    eval  80: running_best = 0.8024  (still improving slowly)

Tunnel 3-1 (121 trials):
    eval  21: running_best = 0.3522
    eval  94: running_best = 0.5812  (late breakthrough)
    eval 120: running_best = 0.5812  (not yet converged)

Tunnel 4-1 (85 trials): ❌ Wrong algorithm — convergence data meaningless
Tunnel 5-1 (83 trials): ❌ Wrong algorithm — convergence data meaningless
```

**Key insight for 1-4:** The BO was stuck at wF1=0.60 for 161 iterations because enhancing params dominated the search space. Once horizontal/merge params were added (eval 161+), wF1 jumped to 0.75 in just 29 iterations. This strongly validates the decision to separate enhancing from detection BO.

## 5. Parameter Sensitivity (Top-10% Trials)

### Simple Staggered Tunnels — Stable Detection Param Ranges

| Parameter | 1-4 (top 15) | 2-2 (top 7) | Notes |
|-----------|-------------|------------|-------|
| binary_threshold | 70–110 (μ=95) | 109–162 (μ=144) | Tunnel-specific |
| dilation_kernel_size | 2–3 | 2 (fixed) | Small kernels preferred |
| dilation_iterations | 1–2 | 1 (fixed) | Minimal dilation |
| hough_oblique_threshold | 50–88 | 24–66 | Low thresholds for sensitivity |
| hough_oblique_min_length | 45–85 | 40–142 | Wide range |
| hough_oblique_max_gap | 34–60 | 20–28 | 2-2 prefers tight gaps |
| angle_positive_min | 5.1–6.3° | 4.5–6.6° | Both ~5° lower bound |
| angle_positive_max | 8.6–11.3° | 8.1–11.4° | Both ~10° upper bound |
| hough_vertical_threshold | 499–700 | 595–793 | High to suppress verticals |
| hough_horizontal_threshold | 35–67 | (not tuned) | New param, only in 1-4 |
| hough_horizontal_min_length | 71–95 | (not tuned) | New param, only in 1-4 |
| merge_distance_threshold | 2–8 | (not tuned) | New param, only in 1-4 |

### Continuous Tunnel 3-1 — Different Character

| Parameter | 3-1 (top 4) | vs Simple Staggered |
|-----------|------------|---------------------|
| binary_threshold | 150–172 | **Much higher** than 1-4/2-2 |
| dilation_kernel_size | 3 (fixed) | Larger kernel needed |
| hough_oblique_max_gap | 94–99 | **Very large** gap tolerance |
| hough_oblique_min_length | 121–149 | Longer lines needed |
| hough_vertical_threshold | 738–750 | Very high suppression |
| angle_positive_min | 3.1–4.3° | **Wider** angle range |

## 6. Meaningfulness Assessment

### ✅ Keep: Tunnel 1-4 New-Schema Logs (29 logs)
- `detect_1-4_161.json` through `detect_1-4_189.json`
- **Why:** Correct assembly type, has all 14 detection params (after removing enhancing keys). Best wF1=0.75. Can directly warm-start new 14D BO.
- **Caveat:** Enhancing params are embedded but should be ignored.

### ⚠️ Partial: Tunnel 1-4 Old-Schema Logs (161 logs)
- `detect_1-4_000.json` through `detect_1-4_160.json`
- **Why partial:** Has 10 of 14 detection params. Missing 5 horizontal/merge dims. Cannot directly warm-start 14D BO.
- **Recommendation:** Keep for reference but don't use for warm-starting. The 29 new-schema logs are sufficient.

### ⚠️ Partial: Tunnel 2-2 All Logs (80 logs)
- **Why partial:** All old-schema (15p). Has 10 of 14 detection params. Best wF1=0.80 shows strong convergence.
- **Recommendation:** Keep all — these are the only data for 2-2. Use top-performing detection params as initial guesses for new BO, with default values for the 5 missing horizontal/merge params.

### ⚠️ Partial: Tunnel 3-1 Logs (121 logs)
- 31 new-schema + 90 old-schema + 10 wrong-assembly
- **Why partial:** New-schema logs haven't converged (best wF1=0.40 vs old-schema's 0.58). High failure rate (61% zero wF1).
- **Recommendation:** Keep the 31 new-schema logs and top old-schema logs for reference. Discard the 10 wrong-assembly logs. This tunnel needs the most BO iterations.

### ❌ Discard: Tunnel 4-1 All Logs (85 logs)
- **Why:** All used simple_staggered detection logic. The new complex algorithm (DBSCAN + subdivision + confidence) is completely different. 22D search space shares only ~10 base params. 8+ complex-specific params have zero historical data.
- **Recommendation:** Delete all. Start fresh with 22D complex BO.

### ❌ Discard: Tunnel 5-1 All Logs (83 logs)
- **Why:** Same as 4-1 — wrong algorithm used throughout.
- **Recommendation:** Delete all. Start fresh with 22D complex BO.

## 7. Warm-Start Recommendations for New BO Runs

### Tunnel 1-4 (14D simple_staggered)
Best detection params from `detect_1-4_183`:
```json
{
  "binary_threshold": 104,
  "dilation_kernel_size": 3,
  "dilation_iterations": 1,
  "hough_oblique_threshold": 76,
  "hough_oblique_min_length": 68,
  "hough_oblique_max_gap": 48,
  "angle_positive_min": 5.48,
  "angle_positive_max": 9.61,
  "angle_negative_min": -9.61,
  "angle_negative_max": -5.48,
  "hough_vertical_threshold": 686,
  "hough_horizontal_threshold": 47,
  "hough_horizontal_min_length": 77,
  "hough_horizontal_max_gap": 6,
  "horizontal_angle_tolerance": 1.76,
  "merge_distance_threshold": 3
}
```

### Tunnel 2-2 (14D simple_staggered)
Best detection params from `detect_2-2_070` + defaults for new params:
```json
{
  "binary_threshold": 122,
  "dilation_kernel_size": 2,
  "dilation_iterations": 1,
  "hough_oblique_threshold": 51,
  "hough_oblique_min_length": 109,
  "hough_oblique_max_gap": 21,
  "angle_positive_min": 6.34,
  "angle_positive_max": 8.75,
  "angle_negative_min": -8.75,
  "angle_negative_max": -6.34,
  "hough_vertical_threshold": 777,
  "hough_horizontal_threshold": 50,
  "hough_horizontal_min_length": 80,
  "hough_horizontal_max_gap": 10,
  "horizontal_angle_tolerance": 1.5,
  "merge_distance_threshold": 5
}
```

### Tunnel 3-1 (14D continuous)
Best detection params from `detect_3-1_065` + defaults for new params:
```json
{
  "binary_threshold": 167,
  "dilation_kernel_size": 3,
  "dilation_iterations": 3,
  "hough_oblique_threshold": 75,
  "hough_oblique_min_length": 121,
  "hough_oblique_max_gap": 96,
  "angle_positive_min": 3.28,
  "angle_positive_max": 11.13,
  "angle_negative_min": -11.13,
  "angle_negative_max": -3.28,
  "hough_vertical_threshold": 750,
  "hough_horizontal_threshold": 50,
  "hough_horizontal_min_length": 80,
  "hough_horizontal_max_gap": 10,
  "horizontal_angle_tolerance": 1.5,
  "merge_distance_threshold": 5
}
```

### Tunnels 4-1 & 5-1 (22D complex_staggered)
**No warm-start available.** Start with default bounds. Use random initialization (n_initial_points ≥ 30) to explore the 22D space.

## 8. Ground Truth Reference (from .log files)

| Tunnel | GT K Positions | h range | θ range |
|--------|---------------|---------|---------|
| 3-1 | 5 | [573.8, 583.3] | [1.3, 16.3] |
| 4-1 | 5 | [25.5, 42.0] | [0.3, 23.2] |
| 5-1 | 6 | [29.3, 43.0] | [0.3, 23.3] |

**Note:** 4-1 and 5-1 have similar h and θ ranges (both short-h, wide-θ), explaining why they need similar detection logic. 3-1's h range is ~10× higher.

## 9. Best Enhancing Params from Joint BO (reference only — now locked)

These were the enhancing values the BO converged to when it had freedom to tune them jointly with detection. Useful as a sanity check against the locked preprocessing values.

| Parameter | 3-1 | 4-1 | 5-1 |
|-----------|-----|-----|-----|
| target_distances | [0.097, 0.049, 0.02] | [0.079, 0.040, 0.02] | [0.081, 0.040, 0.02] |
| curvature_neighbors | 21 | 15 | 9 |
| depth_map_resolution | 0.009904 | 0.008478 | 0.009212 |
| interpolation_window | 14 | 15 | 5 |

**Insight:** All tunnels converged to `depth_map_resolution` ≈ 0.008–0.010 and `target_distance_1` ≈ 0.08–0.10. If locked preprocessing values deviate significantly from these, detection performance may suffer.

## 10. Search Bound Clamping Warnings (from 4-1 & 5-1 .log files)

During warm-start, several parameters were clamped to search bounds — evidence the old bounds were too narrow:

**Tunnel 4-1:**
| Parameter | Warm-start value | Clamped to | Direction |
|-----------|-----------------|------------|-----------|
| binary_threshold | 164 | 129 | ↓ too high |
| dilation_kernel_size | 5 | 3 | ↓ too large |
| dilation_iterations | 1 | 2 | ↑ too low |
| hough_oblique_threshold | 101 | 57 | ↓ too high |
| hough_oblique_min_length | 150 | 134 | ↓ too long |
| angle_max | 11.84 | 11.76 | ↓ marginal |

**Tunnel 5-1:**
| Parameter | Warm-start value | Clamped to | Direction |
|-----------|-----------------|------------|-----------|
| binary_threshold | 117 | 103 | ↓ too high |
| hough_oblique_threshold | 79 | 62 | ↓ too high |
| hough_vertical_threshold | 760 | 625 | ↓ too high |

**Guardrail:** For the new 22D complex BO, ensure bounds are wide enough to avoid clamping. In particular, `binary_threshold` upper bound should be ≥170 and `hough_oblique_threshold` upper bound should be ≥100.

## 11. Failure Mode: FP Explosions

Certain parameter combinations caused extreme false positive counts:

| Tunnel | Eval | TP | FP | FN | Likely cause |
|--------|------|----|----|----|-------------|
| 4-1 | 22 | 0 | **171** | 5 | Very low binary_threshold + low hough thresholds → noise lines everywhere |
| 5-1 | 6 | 4 | **65** | 2 | High recall but garbage precision — too permissive |
| 5-1 | 4 | 1 | **50** | 5 | Same pattern |
| 3-1 | 58 | 1 | **19** | 4 | Moderate FP explosion |

**Guardrail for future BO:** If FP > 20, the trial is almost certainly in a degenerate region. Consider early-stopping or penalty in the objective function to avoid wasting BO budget exploring these regions.

## 12. Detailed Convergence Trajectories (from .log files)

### Tunnel 3-1 (80 evals, best wF1=0.5812)
```
Eval 19: wF1=0.3522  P=0.357 R=1.000 TP=5 FP=9  FN=0  dist=66.2px  ← high recall, low precision
Eval 72: wF1=0.4139  P=0.500 R=0.600 TP=3 FP=3  FN=2  dist=48.3px  ← precision improving
Eval 74: wF1=0.5812  P=0.667 R=0.800 TP=4 FP=2  FN=1  dist=40.2px  ← BEST (late breakthrough)
```
Pattern: BO traded recall for precision over time. Best result came at eval 74/80 — **barely converged**.

### Tunnel 4-1 (80 evals, best wF1=0.3087, ❌ wrong algorithm)
```
Eval 13: wF1=0.0000  P=0.000 R=0.000 TP=0 FP=46  FN=5              ← total failure
Eval 21: wF1=0.2114  P=0.182 R=0.400 TP=2 FP=9   FN=3  dist=30.9px
Eval 22:              P=0.000 R=0.000 TP=0 FP=171  FN=5              ← FP EXPLOSION
Eval 38: wF1=0.2988  P=0.273 R=0.600 TP=3 FP=8   FN=2  dist=40.7px
Eval 81: wF1=0.3087  P=0.273 R=0.600 TP=3 FP=8   FN=2  dist=35.4px ← BEST (marginal gain)
```
Pattern: Stuck at TP=3 ceiling. Wrong algorithm couldn't find remaining 2 K-blocks. Precision never exceeded 0.27.

### Tunnel 5-1 (80 evals, best wF1=0.5293, ❌ wrong algorithm)
```
Eval  4: wF1=0.0310  P=0.020 R=0.167 TP=1 FP=50  FN=5              ← FP explosion start
Eval 11: wF1=0.2108  P=0.167 R=0.333 TP=2 FP=10  FN=4  dist=10.3px ← low dist!
Eval 23: wF1=0.3666  P=0.333 R=0.500 TP=3 FP=6   FN=3  dist=16.7px
Eval 31: wF1=0.4332  P=0.444 R=0.667 TP=4 FP=5   FN=2  dist=37.5px
Eval 60: wF1=0.5293  P=0.455 R=0.833 TP=5 FP=6   FN=1  dist=20.0px ← BEST
```
Pattern: Steadily improving throughout — suggests more budget would help. Best result at eval 60/80 with 5/6 GT found. Even with wrong algorithm, 5-1 was more tractable than 4-1.

## 13. Action Items

1. **Delete 4-1 and 5-1 detection logs** — All used wrong algorithm.
2. **Delete 1-4 old-schema logs** (000–160) — Superseded by new-schema logs.
3. **Delete 3-1 wrong-assembly logs** — 10 logs where assembly_type was "simple_staggered" instead of "continuous".
4. **Keep 1-4 new-schema logs** (161–189) — 29 logs with full 14D params.
5. **Keep all 2-2 logs** (80 total) — Only available data, strong results.
6. **Keep 3-1 valid logs** (111 total after removing wrong-assembly) — Needs more iterations.
7. **Budget recommendation:** Allocate most BO iterations to tunnels 3-1, 4-1, 5-1 which have the most room for improvement.

## 14. Summary Table

| Tunnel | Old Best wF1 | Logs to Keep | Logs to Delete | Warm-Start? | Priority |
|--------|-------------|-------------|---------------|------------|----------|
| 1-4 | 0.7495 | 29 (new schema) | 161 (old schema) | ✅ Yes | Low (already good) |
| 2-2 | 0.8024 | 80 (all) | 0 | ✅ Yes (partial) | Low (already good) |
| 3-1 | 0.5812 | ~111 | ~10 (wrong assembly) | ⚠️ Partial | **High** |
| 4-1 | 0.3087 | 0 | 85 (all) | ❌ No | **Critical** |
| 5-1 | 0.5293 | 0 | 83 (all) | ❌ No | **Critical** |

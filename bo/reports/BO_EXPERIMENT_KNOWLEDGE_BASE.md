# Bayesian Optimization Experiment Knowledge Base
## Consolidated Insights from All Previous BO Experiments

**Date:** 2026-02-13  
**Scope:** All tunnels (1-4, 2-2, 3-1, 4-1, 5-1), all pipeline stages  
**Source:** 15+ reports, 559+ vanilla BO evaluations, 190+ no-GT BO evaluations, 34 surviving detection logs  
**Purpose:** Single reference for all BO knowledge — stage impact, best parameters, ranges, correlations, guardrails, tuning guidelines, and lessons learned

---

## 1. Pipeline Architecture Overview

```
Raw Point Cloud
    │
    ▼
┌────────────────────────────────────────┐
│  STAGE 1: PREPROCESSING (locked)       │
│  ├── 1a. Unfolding     (+0.0% mIoU)   │
│  ├── 1b. Denoising     (+0.1% mIoU)   │
│  └── 1c. Enhancing     (~0.0% mIoU)   │
│  Output: depth_map_outlier.npy         │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│  STAGE 2: DETECTION    (+6.3% mIoU)   │  ← HIGHEST IMPACT
│  Output: detected.csv (K positions)    │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│  STAGE 3: SAM          (+4–7% mIoU)   │  ← SECOND HIGHEST
│  Output: final.csv (segmented points)  │
└────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│  STAGE 4: EVALUATION                   │
│  Output: mIoU, OA, F1, per-class IoU  │
└────────────────────────────────────────┘
```

**Agent routing:**

| Tunnel | Pattern | Agent | Detection Algorithm | BO Dims (Detection) |
|--------|---------|-------|--------------------|--------------------|
| 1-4 | simple_staggered | simple_staggered | Hough lines + midpoint | 14D |
| 2-2 | simple_staggered | simple_staggered | Hough lines + midpoint | 14D |
| 3-1 | continuous | continuous | Hough lines + midpoint | 14D |
| 4-1 | complex_staggered | complex_staggered | DBSCAN + line subdivision + confidence | 22D |
| 5-1 | complex_staggered | complex_staggered | DBSCAN + line subdivision + confidence | 22D |

---

## 2. Stage-by-Stage Impact on mIoU

### 2.1 Measured Impact (Tunnel 2-2, 150+ iterations)

| Phase | Stage | Iterations | mIoU Before | mIoU After | Δ mIoU |
|-------|-------|-----------|-------------|------------|--------|
| 1 | SAM (initial) | 30 | 0.672 | 0.700 | **+4.2%** |
| 2 | **Detection** | 30 | 0.700 | 0.744 | **+6.3%** |
| 3 | SAM (expanded) | 30 | 0.744 | 0.768 | **+3.2%** |
| 4 | Preprocessing | 30 | 0.768 | 0.769 | +0.1% |
| 5 | Unfolding | 30 | 0.769 | 0.769 | +0.0% |

**Key insight:** Preprocessing (unfolding + denoising + enhancing) contributes **≤0.1%** to final mIoU when Detection and SAM are tuned. All future BO should focus on Detection and SAM.

### 2.2 Best Achieved Performance Per Tunnel

| Tunnel | Initial mIoU | Best mIoU | Improvement | Total BO Evals | Best OA |
|--------|-------------|-----------|-------------|---------------|---------|
| 1-4 | 0.511 | **0.807** | +57.9% | 80 | 0.846 |
| 2-2 | 0.672 | **0.765** | +13.8% | 150+ | 0.890 |
| 3-1 | 0.490 | **0.687** | +40.2% | 185 | 0.801 |
| 4-1 | 0.069 | **0.428** | +520% | 45 | 0.621 |
| 5-1 | 0.188 | **0.509** | +171% | 249 | — |

### 2.3 Detection BO Performance (K-block F1 @ 150px, weighted)

| Tunnel | Best wF1 | F1 | Precision | Recall | Mean Dist (px) | Logs Remaining |
|--------|---------|----|-----------|---------|----|------|
| 1-4 | **0.7495** | 0.9474 | 1.000 | 0.900 | 62.7 | 29 |
| 2-2 | **0.8024** | 0.9474 | 1.000 | 0.900 | 30.6 | 0 (old schema) |
| 3-1 | **0.5812** | 0.7273 | 0.667 | 0.800 | 40.2 | 5 |
| 4-1 | 0.3087 | 0.3750 | 0.273 | 0.600 | 35.4 | 0 (wrong algo) |
| 5-1 | 0.5293 | 0.5882 | 0.455 | 0.833 | 20.0 | 0 (wrong algo) |

---

## 3. Best Parameters Per Tunnel

### 3.1 Detection Parameters — Simple/Continuous (14D)

| Parameter | 1-4 (best) | 2-2 (best) | 3-1 (best old) | Typical Range | Sensitivity |
|-----------|-----------|-----------|---------------|---------------|-------------|
| `binary_threshold` | 104 | 122 | 167 | 70–180 | **HIGH** |
| `dilation_kernel_size` | 3 | 2 | 3 | 1–5 | LOW |
| `dilation_iterations` | 1 | 1 | 3 | 1–4 | LOW |
| `hough_oblique_threshold` | 76 | 51 | 75 | 24–100 | **HIGH** |
| `hough_oblique_min_length` | 68 | 109 | 121 | 40–150 | MEDIUM |
| `hough_oblique_max_gap` | 48 | 21 | 96 | 20–110 | MEDIUM |
| `angle_positive_min` | 5.48° | 6.34° | 3.28° | 3–8° | **VERY HIGH** |
| `angle_positive_max` | 9.61° | 8.75° | 11.13° | 7–12° | **VERY HIGH** |
| `angle_negative_min` | -9.61° | -8.75° | -11.13° | -12 to -7° | **VERY HIGH** |
| `angle_negative_max` | -5.48° | -6.34° | -3.28° | -8 to -3° | **VERY HIGH** |
| `hough_vertical_threshold` | 686 | 777 | 750 | 400–900 | MEDIUM |
| `hough_horizontal_threshold` | 47 | 50* | 50* | 30–80 | MEDIUM |
| `hough_horizontal_min_length` | 77 | 80* | 80* | 50–150 | MEDIUM |
| `hough_horizontal_max_gap` | 6 | 10* | 10* | 3–30 | LOW |
| `horizontal_angle_tolerance` | 1.76° | 1.5°* | 1.5°* | 0.5–3° | LOW |
| `merge_distance_threshold` | 3 | 5* | 5* | 2–15 | LOW |

*\* = default values (not yet tuned for this tunnel)*

#### Tunnel-Specific Patterns

**1-4 (simple_staggered, 10 rings):**
- Moderate `binary_threshold` (~104), narrow angle range (5.5–9.6°)
- Tight `hough_oblique_max_gap` (48) and low `merge_distance` (3)
- Already near-optimal: wF1=0.75, P=1.0, R=0.9

**2-2 (simple_staggered, 10 rings):**
- Higher `binary_threshold` (122), tighter angle range (6.3–8.8°)
- Very tight gaps (21) — cleaner depth map boundaries
- Already near-optimal: wF1=0.80

**3-1 (continuous, 6 rings):**
- Much higher `binary_threshold` (150–172), **widest angle range** (3.3–11.1°)
- Very large `hough_oblique_max_gap` (88–109) — needs to connect broken lines
- Higher `dilation_iterations` (3) — more aggressive morphological operations
- **Hardest tunnel:** Only 53% depth map row coverage, 6 rings, theta-seam clipping
- Needs significantly more BO iterations (currently best wF1=0.40 in new schema)

### 3.2 Detection Parameters — Complex Staggered (22D)

In addition to the 14 base detection parameters, complex tunnels add 8 parameters:

| Parameter | 4-1 Default | 5-1 Default | Range | Notes |
|-----------|------------|------------|-------|-------|
| `complex_hough_re_detect_threshold` | 30 | 30 | 20–50 | Re-detection for wider angles |
| `complex_hough_re_detect_min_length` | 50 | 50 | 30–100 | Minimum re-detected line length |
| `complex_hough_re_detect_max_gap` | 100 | 100 | 50–150 | Gap tolerance for re-detection |
| `complex_angle_positive_min` | 4.0° | 4.0° | 3–6° | Wider than base angles |
| `complex_angle_positive_max` | 12.0° | 12.0° | 10–15° | Wider than base angles |
| `complex_min_y_span` | 30 | 30 | 20–50 | Line extent filter |
| `complex_eps_scale` | 0.10 | 0.10 | 0.03–0.22 | DBSCAN clustering scale |
| `complex_subdivision_threshold` | 1.5 | 1.5 | 1.0–2.5 | Ring width multiplier for subdivision |

**No warm-start data available for complex tunnels** — all previous logs used the wrong (simple_staggered) algorithm. Start fresh with `n_initial_points ≥ 30`.

### 3.3 Preprocessing Parameters (8D, locked)

| Parameter | 1-4 | 2-2 | 3-1 | 4-1 | 5-1 | Role |
|-----------|-----|-----|-----|-----|-----|------|
| `gradient_threshold` | 0.462 | 0.500 | 0.482 | 0.457 | 0.448 | Noise sensitivity |
| `radius_min` | 2.612 | 2.613 | 2.569 | 3.515 | 3.526 | Inner radius bound |
| `radius_max` | 2.924 | 3.056 | 3.025 | 3.907 | 4.051 | Outer radius bound |
| `ring_spacing` | 1.313 | 1.400 | 1.170 | 1.256 | 1.400 | Ring width (m) |
| `target_distances` | [0.076,...] | [0.060,...] | [0.098,...] | [0.079,...] | [0.081,...] | Upsampling levels |
| `curvature_neighbors` | 11 | 20 | 28 | 15 | 9 | Surface smoothing |
| `depth_map_resolution` | 0.005 | 0.009 | 0.005 | 0.008 | 0.009 | mm/pixel |
| `interpolation_window` | 15 | 12 | 15 | 15 | 5 | Gap filling |

**These are locked and should NOT be re-tuned.** Preprocessing contributes ≤0.1% to mIoU.

### 3.4 SAM Parameters (Critical Reference)

| Parameter | Sensitivity | 1-4/2-2 Range | 3-1 Range | 4-1/5-1 Range |
|-----------|-------------|--------------|-----------|---------------|
| `segment_width` | **HIGH (complex)** | 1100–1250 mm | 1100–1250 mm | 1150–1350 mm (lower=better) |
| `k_height` | MEDIUM | 1000–1200 mm | 1000–1160 mm | 900–1200 mm |
| `ab_height` | **VERY HIGH** | 3100–3800 mm | 3100–3400 mm | 3000–3500 mm |
| `angle_deg` | **HIGH** | 6.5–8.5° | 6.5–8.5° | 6.0–9.0° |
| `k_block.height_neg` | **HIGH** | 460–600 mm | 550–680 mm | — |
| `k_mask_height` | **CRITICAL** | 580–680 mm | ≥650 mm (3-1) | — |
| `a_blocks.width` | MEDIUM | 550–700 mm | 550–700 mm | 550 mm |
| `a_blocks.height` | MEDIUM | 1500–1750 mm | 1500–1750 mm | ~1567 mm |

**⚠️ CRITICAL WARNING:** K-block parameters (`k_mask_height`, `angle_deg`) must be **protected** during SAM BO. BO can sacrifice K-block IoU to improve overall mIoU — this is a trap. Always monitor K-block IoU separately.

---

## 4. Correlations and Predictive Features

### 4.1 Simple Patterns (1-4, 2-2, 3-1)

**Intrinsic metrics that predict mIoU (no GT needed):**

| Metric | Spearman r | p-value | Role | Threshold |
|--------|-----------|---------|------|-----------|
| `det_midpoint_ratio` | **+0.87** | 0.000 | **Best predictor** | min 0.40 |
| `sam_mask_fill_rate` | **-0.82** | 0.001 | SAM quality signal | max 0.95 |
| `det_real_detection_ratio` | +0.69 | 0.001 | Detection reliability | min 0.50 |
| `det_k_count_match` | +0.52 | 0.019 | Count accuracy | min 0.80 |
| `det_x_spacing_cv` | +0.50 | 0.025 | Spacing regularity | **max 0.15** (critical) |

**Predictor model:** Ridge Regression, R²=0.72, Spearman=0.84, MAE=0.09

**Formula:**
```
mIoU ≈ 0.030·det_midpoint_ratio + 0.009·det_real_detection_ratio 
      - 0.008·det_x_spacing_cv + 0.004·sam_mask_fill_rate + 0.434
```

### 4.2 Complex Patterns (4-1, 5-1)

Detection intrinsic metrics are **NOT** predictive for complex patterns (Spearman=0.59).

**SAM geometry parameters dominate:**

| Parameter | Spearman r | p-value | Direction |
|-----------|-----------|---------|-----------|
| `segment_width` | **-0.789** | 0.000 | **Lower = better** (DOMINANT) |
| `ab_height` | -0.341 | 0.002 | Lower = better |
| `k_height` | -0.269 | 0.024 | Lower = better |
| `angle_deg` | -0.152 | 0.210 | Not significant |

**Predictor model:** Ridge Regression, Spearman=0.87, CV MAE=0.0125 (n=70)

### 4.3 Feature Importance (Ablation Results)

| Feature | Simple ΔMAE | Simple Importance | Complex Importance |
|---------|-------------|-------------------|-------------------|
| `det_midpoint_ratio` | +0.0031 | **CRITICAL** | LOW |
| `det_real_detection_ratio` | +0.0008 | MEDIUM | LOW |
| `det_x_spacing_cv` | +0.0002 | KEEP (guardrail) | LOW |
| `sam_mask_fill_rate` | +0.0000 | KEEP (corr strong) | N/A |
| `det_y_std` | -0.0012 | KEEP | LOW |
| `segment_width` | — | N/A | **DOMINANT** |

### 4.4 Key Empirical Finding

**`det_x_spacing_cv` guardrail saved 2-2:** When `det_x_spacing_cv` exceeded 0.10, true mIoU dropped from 0.690 to 0.476. Adding the guardrail `det_x_spacing_cv < 0.10` for 2-2 (0.15 for others) prevented this failure and recovered +45% mIoU.

---

## 5. Tuning Guidelines

### 5.1 Detection BO — Recommended Search Spaces

**Simple/Continuous (14D):**

| Parameter | Lower | Upper | Type | Priority |
|-----------|-------|-------|------|----------|
| `binary_threshold` | 50 | 200 | Integer | 1st |
| `dilation_kernel_size` | 1 | 5 | Integer | Low |
| `dilation_iterations` | 1 | 5 | Integer | Low |
| `hough_oblique_threshold` | 20 | 120 | Integer | 1st |
| `hough_oblique_min_length` | 30 | 180 | Integer | 2nd |
| `hough_oblique_max_gap` | 10 | 120 | Integer | 2nd |
| `angle_positive_min` | 2.0 | 8.0 | Real | **1st** |
| `angle_positive_max` | 7.0 | 14.0 | Real | **1st** |
| `angle_negative_min` | -14.0 | -7.0 | Real | **1st** |
| `angle_negative_max` | -8.0 | -2.0 | Real | **1st** |
| `hough_vertical_threshold` | 300 | 900 | Integer | 2nd |
| `hough_horizontal_threshold` | 20 | 100 | Integer | 2nd |
| `hough_horizontal_min_length` | 30 | 180 | Integer | 3rd |
| `hough_horizontal_max_gap` | 3 | 40 | Integer | 3rd |
| `horizontal_angle_tolerance` | 0.3 | 3.0 | Real | 3rd |
| `merge_distance_threshold` | 1 | 20 | Integer | 3rd |

**Complex Staggered (22D):** Add 8 complex-specific dims (see §3.2).

### 5.2 Detection BO — Recommended Order

```
PHASE 1: Tune angle parameters first
   ├── angle_positive_min/max
   └── angle_negative_min/max
   (These control WHICH lines are detected — highest sensitivity)

PHASE 2: Tune threshold parameters
   ├── binary_threshold (edge sensitivity)
   ├── hough_oblique_threshold (line confidence)
   └── hough_vertical_threshold (vertical line suppression)

PHASE 3: Tune length/gap parameters
   ├── hough_oblique_min_length/max_gap
   ├── hough_horizontal_threshold/min_length/max_gap
   └── merge_distance_threshold

PHASE 4: Fine-tune morphological parameters
   ├── dilation_kernel_size
   └── dilation_iterations
```

### 5.3 SAM BO — Recommended Guardrails

| Rule | Rationale |
|------|-----------|
| **Protect K-block params** in BO: constrain `k_mask_height` ≥ 550, `angle_deg` ≥ 6.0° | BO sacrifices K-block for overall mIoU |
| Monitor **K-block IoU** separately from mIoU | K-block is the anchor for all segments |
| **Never allow** `segment_geometry` params to change >15% from baseline | Catastrophic regressions (e.g. 0.765→0.673) |
| For complex: **Lower `segment_width` is better** (r=-0.79) | Dominant predictor |
| For 3-1: `k_mask_height` ≥ 650 mm | Lower values degrade K detection |

### 5.4 Warm-Start Recommendations

| Tunnel | Method | Source | Notes |
|--------|--------|--------|-------|
| 1-4 | ✅ Direct warm-start | 29 logs (new 14D schema) | Best wF1=0.75 |
| 2-2 | ⚠️ Partial warm-start | Old best + defaults for 5 new params | Best old wF1=0.80 |
| 3-1 | ⚠️ Partial warm-start | 5 new-schema logs + old best params | **Needs most iterations** |
| 4-1 | ❌ Cold start | No valid logs (wrong algorithm) | Use defaults, n_initial ≥ 30 |
| 5-1 | ❌ Cold start | No valid logs (wrong algorithm) | Use defaults, n_initial ≥ 30 |

### 5.5 Iteration Budget Recommendations

| Tunnel | Priority | Recommended Iterations | Why |
|--------|----------|----------------------|-----|
| 3-1 | **HIGH** | 100+ | Hardest tunnel, sparse depth map, needs convergence |
| 4-1 | **CRITICAL** | 80+ | New 22D space, no warm-start, complex algorithm |
| 5-1 | **CRITICAL** | 80+ | Same as 4-1 |
| 1-4 | LOW | 30–50 | Already at wF1=0.75, near-converged |
| 2-2 | LOW | 30–50 | Already at wF1=0.80, strongest tunnel |

---

## 6. Guardrails and Quality Checks

### 6.1 Preprocessing Guardrails (Fail-Fast)

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| `theta_coverage` | 98–102% | Flag wraparound/coverage issue |
| `point_retention_ratio` | > 90% | Flag aggressive denoising |
| `interpolation_coverage` | > 95% | Flag sparse depth map |

### 6.2 Detection Guardrails

| Pattern | Metric | Threshold | Severity |
|---------|--------|-----------|----------|
| Both | `det_k_count` | ∈ [expected ± 3] | **Tier 1 (must rerun)** |
| Simple | `det_x_spacing_cv` | < 0.15 (0.10 for 2-2) | **Tier 1** |
| Simple | `det_midpoint_ratio` | > 0.40 | Tier 2 |
| Simple | `det_real_detection_ratio` | > 0.50 | Tier 2 |
| Complex | `det_x_spacing_cv` | < 0.60 (0.50 for 4-1, 0.80 for 5-1) | Tier 2 |
| Complex | `det_y_range` | 200–1500 px (200–3500 for 5-1) | Tier 2 |
| Complex | `det_k_count` | 4–12 (7–12 for 4-1, 5–10 for 5-1) | Tier 2 |

### 6.3 SAM Guardrails

| Metric | Threshold | Action |
|--------|-----------|--------|
| `sam_mask_fill_rate` | < 0.10 | **Tier 1 (must rerun)** — SAM completely failed |
| `sam_mask_fill_rate` | > 0.95 | Tier 2 — over-segmentation likely |
| K-block IoU | < previous - 0.05 | Tier 2 — K params may be degraded |
| Background ratio | > 30% | Tier 3 — templates too small |

### 6.4 SAM Parameter Guardrails (Complex)

| Parameter | Min | Max | Optimal Direction |
|-----------|-----|-----|-------------------|
| `segment_width` | 1150 | 1350 | Lower is better |
| `k_height` | 900 | 1200 | Lower is better |
| `ab_height` | 3000 | 3500 | Lower is better |
| `angle_deg` | 6.0 | 9.0 | ~7.3° optimal |

### 6.5 Confidence Mapping

| Predicted mIoU | Confidence | Action |
|----------------|------------|--------|
| ≥ 0.60 | High | Accept |
| 0.45–0.60 | Medium | Accept with warning |
| 0.35–0.45 | Low | Manual review |
| < 0.35 | Very Low | Likely failed |

---

## 7. Known Tunnel-Specific Challenges

### 7.1 Tunnel 3-1 (Continuous) — The Hardest Simple Tunnel

| Challenge | Detail | Impact |
|-----------|--------|--------|
| **Sparse depth map** | Only 53% row coverage (vs 73–76% for 1-4/2-2) | Weak edges, poor detection |
| **Few rings** | 6 rings vs 10 | Fewer SAM prompts, coarser aggregation |
| **Short h-span** | 1.37m vs ~3.6m | Aspect ratio 1.96 vs ~1.1 |
| **Theta-seam clipping** | All blocks below K, stack crosses Y=0 | A2, A3, B2 systematically cut |
| **Low pattern confidence** | continuous confidence 0.25 vs 0.7 for simple_staggered | Ambiguous classification |
| **Wide angle range needed** | angle_positive_min as low as 3.3° | More false positive lines |
| **High failure rate** | 61% of detection trials produced wF1=0 | Unstable search space |

**Recommendation:** Consider selective wraparound for 3-1; improve depth-map coverage; allocate 100+ BO iterations.

### 7.2 Tunnel 4-1 (Complex Staggered) — Algorithm Mismatch Fixed

- All previous BO used simple_staggered algorithm → **results invalid**
- New complex algorithm adds DBSCAN clustering, line subdivision, confidence scoring
- **22D search space** with no warm-start data
- After BO: Detection score 0.865, mIoU 0.428, OA 0.621
- Key: Combined detection (Hough + gradient cross-validation) reduced error from 632 to 114 px

### 7.3 Tunnel 5-1 (Complex Staggered) — Non-Uniform Ring Spacing

- 7 rings (107–113), **ring 110 has no K-block**
- Non-uniform ring spacing (large X-gap at one ring)
- Detection BO: mean K error 204→110 px; one ring can still have 473 px error
- SAM: Ring 110 A3 has ~0% accuracy (structural limitation, no K anchor)
- Best mIoU 0.509 with correct GT alignment
- **GT alignment is foundational** — wrong ring mapping cost hundreds of pixels

---

## 8. Critical Lessons Learned

### 8.1 Foundational Rules

| # | Lesson | Evidence |
|---|--------|----------|
| 1 | **Detection is the highest-impact stage** (+6.3% single-stage mIoU) | Tunnel 2-2 stage ablation |
| 2 | **Preprocessing has minimal impact** (≤0.1% mIoU) | All tunnel BO results |
| 3 | **K-block parameters must be protected in SAM BO** | 3-1: K IoU dropped while mIoU improved |
| 4 | **GT alignment must be verified before any tuning** | 5-1: Wrong ring mapping wasted compute |
| 5 | **Depth map dimensions must be consistent across pipeline** | 3-1: Stale intermediates (3083×1471 vs 2925×1495) broke everything |
| 6 | **`det_x_spacing_cv` is the single most critical guardrail** | 2-2: Fixed 0.476→0.690 mIoU (+45%) |
| 7 | **Different pattern types need separate models** | Simple uses detection metrics; complex uses SAM geometry |
| 8 | **Per-ring alignment destroys spatial coherence** — never use it | Complete failure on 3-1 |

### 8.2 BO Configuration Lessons

| # | Lesson | Evidence |
|---|--------|----------|
| 1 | **Separating enhancing from detection BO is correct** | 1-4: Stuck at wF1=0.60 for 161 iters due to enhancing params; jumped to 0.75 in 29 iters after separation |
| 2 | **Adding horizontal/merge params boosted 1-4** by +25% wF1 | From 0.60→0.75 |
| 3 | **Old 4D preprocessing logs are useless** for new 8D search space | All 180 deleted |
| 4 | **Old 15-param detection logs** can partially warm-start (10 of 14 params match) but are inferior | New schema consistently outperforms |
| 5 | **Complex tunnels need 30+ initial random points** for 22D space | No prior data available |
| 6 | **3-1 needs significantly more iterations** than 1-4/2-2 | High failure rate (61% zero wF1) |
| 7 | **Relaxing F1 threshold from 100→150px** was necessary for realistic scoring | 100px was too strict |

### 8.3 What NOT to Do

| Anti-Pattern | What Happened | Correct Approach |
|-------------|---------------|------------------|
| Tune preprocessing before detection | Wasted 30+ iterations with 0% improvement | Tune detection first, then SAM |
| Force GT values as SAM template sizes | K-block IoU dropped (e.g. 750mm from GT vs 1028.5mm tuned) | Use GT to learn, not to override tuned values |
| Enlarge crops/templates blindly | mIoU 0.509→0.325 for 5-1 | Test one change at a time, revert if worse |
| Use evenly-spaced X for K positions | Systematic 46px error | Use actual (h_mean, θ_mean) centroids per ring |
| Debug code in hot path with format assumptions | `score:.4f` on numpy array skipped 1 segment | Safe logging, type-check |
| Trust stale intermediate files | Different depth map dims broke all tuning | Re-run full pipeline before comparing |
| Optimize with wrong algorithm | All 4-1/5-1 logs invalid | Verify algorithm matches tunnel type |

### 8.4 BO Convergence Patterns

| Tunnel | Pattern | Convergence Notes |
|--------|---------|-------------------|
| 1-4 | Fast improvement, slow plateau | wF1 plateau at 0.60 for 161 iters → 0.75 after schema change |
| 2-2 | Fast convergence | Best at eval 21; slow improvement to eval 80 |
| 3-1 | Late breakthrough | Best at eval 94; NOT yet converged |
| 4-1 | N/A (wrong algo) | Start fresh |
| 5-1 | N/A (wrong algo) | Start fresh |

---

## 9. Current System State (as of 2026-02-13)

### 9.1 Log Inventory

| Directory | Logs | Content | Usable? |
|-----------|------|---------|---------|
| simple_staggered/logs | 29 | 1-4 detection, new 14D (with horiz/merge) | ✅ Full warm-start |
| continuous/logs | 5 | 3-1 detection, new 14D (with horiz/merge) | ✅ Partial warm-start |
| complex_staggered/logs | 0 | Empty | ❌ Start fresh |

### 9.2 What Was Deleted and Why

| Category | Count | Reason |
|----------|-------|--------|
| All 4-1 detection logs | 85 | Wrong algorithm (simple_staggered instead of complex) |
| All 5-1 detection logs | 83 | Wrong algorithm |
| 1-4 old-schema detection logs (000–160) | 161 | Missing 5 horizontal/merge params |
| 3-1 wrong-assembly logs | 10 | Labeled "simple_staggered" instead of "continuous" |
| All preprocessing logs | 180 | Old 4D schema (missing 4 enhancing params) |
| Zero-wF1 logs | 73 | No useful optimization signal |
| Low-precision + low-wF1 logs | ~40 | Garbage detections, mostly false positives |
| Divergent enhancing params logs | ~30 | >20% different from locked enhancing → different depth map |

### 9.3 Data Protection

⚠️ **`data/bo/` must NEVER be touched** — contains original BO preprocessing results that are now the baseline for all experiments.

---

## 10. Reflection/Rerun System

### 10.1 Tier 1: Hard Fail (Always Rerun)

| Metric | Criteria | Action |
|--------|----------|--------|
| `det_k_count` | < expected−3 OR > expected+3 | Adjust `binary_threshold` and `hough_oblique_threshold` |
| `det_x_spacing_cv` | > 0.50 (simple) | Increase `hough_oblique_threshold`, narrow angle range |
| `sam_mask_fill_rate` | < 0.10 | Check template sizes, re-run SAM |

### 10.2 Tier 2: Soft Fail (Rerun if Budget Allows)

| Metric | Criteria | Action |
|--------|----------|--------|
| `predicted_mIoU` | < 0.40 | Rerun with adjusted params |
| `det_midpoint_ratio` | < 0.30 | Adjust angle params |
| `sam_mask_fill_rate` | > 0.95 | Shrink templates |

### 10.3 Retry Limits

| Stage | Max Retries | Escalation |
|-------|-------------|------------|
| Detection | 3 | Use historical best params |
| SAM | 3 | Flag for manual review |
| Overall | 5 | Output with low-confidence flag |

### 10.4 Rerun Heuristics

```python
# If too few detections:
if det_k_count < expected - 2:
    binary_threshold -= 10      # More sensitive edge detection
    hough_oblique_threshold -= 5 # Lower line confidence threshold

# If irregular spacing:
if det_x_spacing_cv > 0.15:
    hough_oblique_threshold += 10  # Reduce false positive lines
    angle_positive_max -= 0.5      # Narrower angle range

# If SAM over-segmentation:
if sam_mask_fill_rate > 0.90:
    template_width_factor *= 0.9   # Shrink templates

# If SAM under-segmentation:
if sam_mask_fill_rate < 0.30:
    template_width_factor *= 1.1   # Expand templates
```

---

## 11. Quick Reference Card

### Detection Tuning Priority (Simple/Continuous)
1. `angle_positive_min/max` + `angle_negative_min/max` → **MOST CRITICAL**
2. `binary_threshold` → Edge sensitivity
3. `hough_oblique_threshold` → Line detection confidence
4. `hough_vertical_threshold` → Vertical line suppression
5. Horizontal/merge params → Fine-tuning

### Detection Tuning Priority (Complex Staggered)
1. All base params above
2. `complex_hough_re_detect_threshold` → Re-detection sensitivity
3. `complex_angle_*` → Wider angle range
4. `complex_eps_scale` → DBSCAN clustering granularity
5. `complex_subdivision_threshold` → Cluster subdivision

### SAM Tuning Priority
1. `ab_height` → **VERY HIGH** sensitivity
2. `k_mask_height` / `k_block.height_neg` → **HIGH**, protect in BO
3. `angle_deg` → **HIGH**, protect in BO
4. Template mask dimensions → MEDIUM, safer to tune
5. `segment_geometry` → **DANGEROUS**, constrain tightly

### Metric Hierarchy
```
mIoU
 ├── Detection Quality (r=0.84 with mIoU for simple)
 │    ├── det_midpoint_ratio (r=+0.87)
 │    ├── det_real_detection_ratio (r=+0.69)
 │    └── det_x_spacing_cv (guardrail, saved 2-2)
 └── SAM Quality
      ├── sam_mask_fill_rate (r=-0.82 for simple)
      └── segment_width (r=-0.79 for complex, DOMINANT)
```

---

*Report compiled from: P4TUN_OPTIMIZATION_JOURNEY_2-2, P4TUN_OPTIMIZATION_JOURNEY_3-1_1-4, P4TUN_OPTIMIZATION_JOURNEY_3-1_GT_K_BLOCK, P4TUN_OPTIMIZATION_JOURNEY_4-1, P4TUN_5-1_COMPLEX_STAGGERED_JOURNEY, P4TUN_PARAMETERIZATION_JOURNEY, P4TUN_PER_RING_ALIGNMENT_LESSONS, NO_GT_BO_COMPREHENSIVE_REPORT, NUMERICAL_EVIDENCE_SUMMARY, INTRINSIC_METRICS_REPORT, CRITICAL_PARAMETERS_DETECTION_SAM, COMPLEX_STAGGERED_PARAMETERS, WHY_3-1_UNDERPERFORMS_1-4_2-2, DETECTION_BO_LOG_ANALYSIS, R4TUN_EXPLORATION_JOURNEY, full_ablation_results.json*

*Total evidence base: 559+ vanilla BO evals, 190+ no-GT BO evals, 80+ proxy experiments, 34 surviving detection logs*

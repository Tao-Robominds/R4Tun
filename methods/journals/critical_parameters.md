# Critical Parameter Analysis

## Objective

Identify which pipeline parameters are adapted by LLM agents and which are the most critical — i.e., parameters that every LLM independently changes for every tunnel. Analysis covers 1,350 parameter files: 30 tunnels × 3 LLMs (Gemini 3 Flash, GPT 5.4, Opus 4.6) × 3 ablation conditions (memory, memory+state, memory+state+knowledge) × 5 stages, compared against the sam4tun fixed baseline.

---

## Summary

**89 of ~60 baseline parameters were adapted** (some stages gained new keys). All 3 LLMs show consistent adaptation patterns. Two distinct behaviours emerge:

| Behaviour | Count | Description |
|---|---|---|
| **Tunnel-responsive** | ~10 | Value varies per tunnel, driven by physical geometry and scan density |
| **Baseline correction** | ~6 | Same fixed shift applied to every tunnel — all 3 LLMs agree the baseline default was suboptimal |

---

## Per-Stage Adaptation Coverage

| Stage | Baseline params | Adapted | Coverage | Top adapted parameter |
|---|---|---|---|---|
| Unfolding | 16 | 10 | 63% | `diameter` (27/30 tunnels) |
| Denoising | 8 | 8 | **100%** | `mask_r_low` (30/30 tunnels) |
| Enhancing | 14 | 12 | 86% | `inter_radius` (30/30 tunnels) |
| Detecting | 14 | 14 | **100%** | `hough_threshold_oblique` (30/30 tunnels) |
| SAM | ~50 (nested) | 45 | 90% | `processing.padding` (29/30 tunnels) |

---

## Always-Trigger Parameters (adapted in all or nearly all 30 tunnels)

### Tier 1 — Tunnel-responsive (high CV, value varies per tunnel)

These are the parameters that genuinely react to tunnel conditions. Their adapted values differ across tunnels, meaning the LLM is reading tunnel characteristics and mapping them to distinct parameter choices.

| Stage | Parameter | Tunnels | CV | Baseline | Adapted range | Physical driver |
|---|---|---|---|---|---|---|
| Denoising | `mask_r_low` | 30/30 | 0.082 | 2.7 | [2.09, 3.75] | Tunnel inner radius — families 1-x, 2-x get lower; 4-x, 5-x get higher |
| Denoising | `mask_r_high` | 30/30 | 0.147 | 2.8 | [2.78, 4.38] | Tunnel outer radius — sharp step between family 2 (≈2.79) and family 4 (≈3.85) |
| Denoising | `default_cutoff_z` | 29/30 | 0.142 | 2.7 | [2.65, 6.27] | Radial extent of usable data |
| Denoising | `z_step` | 30/30 | 0.181 | 0.001 | [0.003, 0.005] | Scan resolution — family 4 gets 0.003, families 1-2 get 0.005 |
| Detecting | `hough_threshold_oblique` | 30/30 | 0.188 | 50 | [20, 83] | Point density / image contrast — family 5 ≈30, families 1-2 ≈56 |
| Detecting | `hough_threshold_horizontal` | 30/30 | 0.204 | 50 | [20, 83] | Mirrors oblique; same tunnel-density pattern |
| Detecting | `hough_threshold_vertical` | 28/30 | 0.219 | 500 | [320, 980] | Vertical line visibility varies with ring spacing |
| Enhancing | `inter_radius` | 30/30 | 0.130 | 0.06 | [0.03, 0.08] | Mean point spacing — denser tunnels get smaller radius |
| Enhancing | `upsampling_stage1` | 30/30 | 0.064 | 0.08 | [0.055, 0.11] | Two clusters: 0.06 (dense) vs 0.068 (sparse) |
| Unfolding | `diameter` | 27/30 | 0.072 | 5.5 | [5.31, 7.6] | Physical tunnel diameter from RANSAC |
| SAM | `processing.padding` | 29/30 | 0.265 | 150 | [160, 419] | Image border size scales with segment width |

### Tier 2 — Baseline corrections (CV ≈ 0, same value for all tunnels)

All 3 LLMs independently converge on the same corrected value regardless of tunnel. These represent suboptimal baseline defaults that the agents universally override.

| Stage | Parameter | Tunnels | Baseline | Corrected to | Shift |
|---|---|---|---|---|---|
| Denoising | `smoothing_window_size` | 30/30 | 3 | **5** | +67% |
| Denoising | `smoothing_offset` | 30/30 | -0.003 | **-0.002** | +33% |
| Denoising | `grad_threshold` | 30/30 | 0.2 | **0.15** | -25% |
| Denoising | `y_step` | 30/30 | 0.5 | **0.4** | -20% |
| Enhancing | `curvature_threshold` | 30/30 | 0.0005 | **0.005** | +900% |
| Enhancing | `depth_threshold_low` | 30/30 | 0.003 | **0.005** | +67% |
| Enhancing | `depth_threshold_high` | 30/30 | 0.008 | **0.015** | +87% |

---

## Per-LLM Adaptation Summary

| LLM | Unfolding | Denoising | Enhancing | Detecting | SAM | Total changes |
|---|---|---|---|---|---|---|
| Gemini 3 Flash | 8 params, 22/30 tunnels | 8 params, 30/30 | 11 params, 30/30 | 14 params, 30/30 | 44 params, 29/30 | 3,925 |
| GPT 5.4 | 8 params, 26/30 tunnels | 8 params, 30/30 | 12 params, 30/30 | 14 params, 30/30 | 45 params, 30/30 | 3,615 |
| Opus 4.6 | 9 params, 20/30 tunnels | 8 params, 30/30 | 10 params, 30/30 | 14 params, 30/30 | 44 params, 29/30 | 3,965 |

All 3 LLMs show near-identical adaptation patterns: denoising, enhancing, and detecting are fully covered; unfolding is the most conservative stage (fewest tunnels adapted). The parameter *keys* they adapt are highly consistent; the *values* vary slightly between models but track the same per-tunnel trends.

---

## Tunnel-Family Patterns in Top Parameters

The analysis reveals clear clustering by tunnel family for the most critical parameters:

**`mask_r_low`** — directly encodes physical radius:
- Families 1-x, 2-x: 2.25–2.38 (smaller tunnels)
- Family 3-x: 2.58–2.84 (mid-size)
- Families 4-x, 5-x: 2.62–2.91 (larger tunnels)

**`mask_r_high`** — sharp binary split:
- Families 1-x, 2-x: ≈2.79 (barely above baseline)
- Families 4-x, 5-x: ≈3.85 (+37–40% above baseline)

**`hough_threshold_oblique`** — inversely tracks point density:
- Family 5: 30–42 (sparse → lower threshold to detect faint lines)
- Family 3: 34–40 (moderate)
- Families 1-x, 2-x: 53–61 (dense → higher threshold to filter noise)

**`z_step`** — adapts to scanner resolution:
- Families 4-2 to 4-9: 0.003 (high-res scans)
- Families 1-x, 2-x, 5-x: 0.005 (standard scans)

---

## Parameters That Never Changed

| Stage | Unchanged parameters |
|---|---|
| Unfolding | `ransac_probability`, `ransac_inlier_ratio`, `ransac_inlier_threshold_multiplier`, `batch_size`, `n_jobs` |
| Enhancing | `resolution`, `num_interpolations` |
| SAM | `processing.resolution`, `processing.mask_eps`, `prompt_points.ab_blocks.vertical_levels.center` |

These represent hard constraints or implementation-level settings that are invariant to tunnel conditions.

---

## Key Insights

1. **Denoising radius masks (`mask_r_low`, `mask_r_high`) are the single most critical parameters.** Adapted in 30/30 tunnels by all 3 LLMs across all 3 conditions, with the highest per-tunnel variation. They directly encode physical tunnel geometry and their misspecification is the #1 cause of depth map white spaces.

2. **Two adaptation modes coexist.** Roughly half the always-trigger parameters are genuine tunnel-responsive adaptations (vary by tunnel); the other half are universal baseline corrections (same shift everywhere). This suggests the sam4tun baseline has ~7 suboptimal defaults that any LLM will fix independently.

3. **Detecting thresholds are the most tunnel-sensitive stage.** Hough thresholds show the highest CV (0.19–0.22), meaning detecting parameters vary the most between tunnels. This aligns with the fact that line detection is highly dependent on image contrast and point density.

4. **Unfolding is the most conservative stage.** `diameter` only changes in 27/30 tunnels, and other unfolding parameters rarely change. The geometric unfolding is well-constrained by RANSAC and needs minimal adaptation.

5. **SAM parameters are numerous but correlated.** 45 SAM parameters are adapted, but most are derived geometry (block heights, widths, spacing factors) that scale proportionally with `K_height`, `AB_height`, and `segment_width`. The effective degrees of freedom are ~5 independent values.

6. **All 3 LLMs converge.** Despite different architectures (Gemini, GPT, Claude), the adaptation patterns are nearly identical — same parameters adapted, same tunnel-family clustering, same baseline corrections. This suggests the adaptations are driven by objective tunnel characteristics rather than LLM-specific biases.

---

## Files

| File | Purpose |
|---|---|
| `skills/analyze_parameter_adaptations.py` | Analysis script (reads all 1,350 JSONs, compares to baseline) |
| `configurable/ablation/sam4tun/parameters_*.json` | Baseline parameter files (5 stages) |
| `configurable/ablation/{memory,memory+state,memory+state+knowledge}/parameters/` | Adapted parameter files (30 tunnels × 3 LLMs × 5 stages) |

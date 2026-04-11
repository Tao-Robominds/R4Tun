# Critical Parameters Identification — Full Process

## Objective

Determine which of the ~60 baseline pipeline parameters are **actually adapted** by LLM agents, identify the **most critical** ones (adapted by every LLM for every tunnel), and classify them by adaptation behaviour. The output feeds BO search-space design: only critical parameters need optimisation; the rest can be fixed or inherited.

---

## Data scope

| Dimension | Values | Count |
|-----------|--------|-------|
| Tunnels | 1-1 … 5-7 | 30 |
| LLMs | Gemini 3 Flash, GPT 5.4, Opus 4.6 | 3 |
| Ablation conditions | memory, memory+state, memory+state+knowledge | 3 |
| Pipeline stages | unfolding, denoising, enhancing, detecting, SAM | 5 |
| **Parameter files analysed** | 30 × 3 × 3 × 5 | **1,350** |

Baseline: `agents/ablation/sam4tun/parameters_*.json` (one file per stage, shared across all tunnels).

Adapted files: `agents/ablation/{memory,memory+state,memory+state+knowledge}/parameters/{tunnel_id}/parameters_{stage}_{code}_{model}.json`.

---

## Discovery methodology

### Step 1 — Baseline comparison (which parameters change?)

**Script:** `skills/scripts/analyze_parameter_adaptations.py`

For every adapted parameter file, the script:
1. Flattens both the baseline and adapted JSONs into dot-separated leaf keys.
2. Compares every leaf value; a difference of >1e-9 (numeric) or `!=` (other) counts as an adaptation.
3. Aggregates across all 1,350 files: for each `(stage, parameter_key)` pair, records the number of unique tunnels, LLMs, and conditions where it was adapted.

**Output:** A ranked list of all parameters by tunnel-adaptation breadth (how many of the 30 tunnels triggered a change).

### Step 2 — Always-trigger filtering (which parameters are critical?)

Parameters adapted in **≥28 of 30 tunnels** across **all 3 LLMs** were flagged as "always-trigger" — these are the parameters that every model independently decides to change for (nearly) every tunnel.

**Result:** ~18 always-trigger parameters survived this filter.

### Step 3 — Tunnel-responsiveness vs baseline-correction classification

For each always-trigger parameter, the script computes per-tunnel mean values (averaged across LLMs and conditions) and the **coefficient of variation (CV)**:

- **High CV** (≥0.06): The adapted value varies across tunnels → parameter is **tunnel-responsive** (the LLM is reading tunnel characteristics and mapping them to distinct values).
- **CV ≈ 0**: Every tunnel gets the same corrected value → parameter is a **baseline correction** (the sam4tun default was suboptimal; all LLMs converge on the same fix).

The boundary was set empirically: tunnel-responsive parameters showed CV 0.06–0.27; baseline corrections had CV < 0.01 with identical adapted values for all 30 tunnels.

### Step 4 — Per-LLM consistency check

For each always-trigger parameter, per-LLM tunnel counts and value ranges were extracted (Part 2b of the script output). This confirmed that all three LLMs (Gemini, GPT, Claude) adapt the **same parameter keys** and produce the **same tunnel-family clustering** (e.g. families 1-x and 2-x cluster together for `mask_r_low`; families 4-x and 5-x form a separate cluster). The adapted *values* differ slightly between models, but the *patterns* are near-identical.

### Step 5 — Tunnel-family pattern extraction

The script's Part 4 prints per-tunnel adapted values sorted by magnitude, revealing clear tunnel-family grouping for the top parameters. These patterns were manually inspected and summarised (e.g. `mask_r_low` maps directly to physical tunnel radius; `hough_threshold_oblique` inversely tracks point density).

### Step 6 — Never-changed parameter identification

Parameters where the baseline value was retained in all 1,350 files (zero adaptations) were listed as invariant. These represent hard constraints (e.g. `batch_size`, `n_jobs`, `resolution`) or values that are already optimal across all tunnel conditions.

---

## Classification result

### Tier 1 — Tunnel-responsive (11 parameters)

Value varies per tunnel; driven by physical geometry and scan density.

| Stage | Parameter | Tunnels adapted | CV | Baseline | Adapted range | Physical driver |
|-------|-----------|-----------------|-----|----------|---------------|-----------------|
| Denoising | `mask_r_low` | 30/30 | 0.082 | 2.7 | [2.09, 3.75] | Inner radius — families 1-x, 2-x lower; 4-x, 5-x higher |
| Denoising | `mask_r_high` | 30/30 | 0.147 | 2.8 | [2.78, 4.38] | Outer radius — binary split: family 2 ≈2.79, family 4 ≈3.85 |
| Denoising | `default_cutoff_z` | 29/30 | 0.142 | 2.7 | [2.65, 6.27] | Radial extent of usable data |
| Denoising | `z_step` | 30/30 | 0.181 | 0.001 | [0.003, 0.005] | Scanner resolution — family 4 gets 0.003; families 1-2 get 0.005 |
| Detecting | `hough_threshold_oblique` | 30/30 | 0.188 | 50 | [20, 83] | Point density / image contrast — family 5 ≈30; families 1-2 ≈56 |
| Detecting | `hough_threshold_horizontal` | 30/30 | 0.204 | 50 | [20, 83] | Mirrors oblique; same density pattern |
| Detecting | `hough_threshold_vertical` | 28/30 | 0.219 | 500 | [320, 980] | Vertical line visibility varies with ring spacing |
| Enhancing | `inter_radius` | 30/30 | 0.130 | 0.06 | [0.03, 0.08] | Mean point spacing — denser tunnels get smaller radius |
| Enhancing | `upsampling_stage1` | 30/30 | 0.064 | 0.08 | [0.055, 0.11] | Two clusters: 0.06 (dense) vs 0.068 (sparse) |
| Unfolding | `diameter` | 27/30 | 0.072 | 5.5 | [5.31, 7.6] | Physical tunnel diameter from RANSAC |
| SAM | `processing.padding` | 29/30 | 0.265 | 150 | [160, 419] | Image border size scales with segment width |

### Tier 2 — Baseline corrections (7 parameters)

Same corrected value for all tunnels; CV ≈ 0. All 3 LLMs independently converge on the identical fix.

| Stage | Parameter | Tunnels | Baseline | Corrected to | Shift |
|-------|-----------|---------|----------|-------------|-------|
| Denoising | `smoothing_window_size` | 30/30 | 3 | 5 | +67% |
| Denoising | `smoothing_offset` | 30/30 | -0.003 | -0.002 | +33% |
| Denoising | `grad_threshold` | 30/30 | 0.2 | 0.15 | -25% |
| Denoising | `y_step` | 30/30 | 0.5 | 0.4 | -20% |
| Enhancing | `curvature_threshold` | 30/30 | 0.0005 | 0.005 | +900% |
| Enhancing | `depth_threshold_low` | 30/30 | 0.003 | 0.005 | +67% |
| Enhancing | `depth_threshold_high` | 30/30 | 0.008 | 0.015 | +87% |

### Never-changed parameters

| Stage | Parameters |
|-------|-----------|
| Unfolding | `ransac_probability`, `ransac_inlier_ratio`, `ransac_inlier_threshold_multiplier`, `batch_size`, `n_jobs` |
| Enhancing | `resolution`, `num_interpolations` |
| SAM | `processing.resolution`, `processing.mask_eps`, `prompt_points.ab_blocks.vertical_levels.center` |

---

## Per-LLM adaptation summary

| LLM | Unfolding | Denoising | Enhancing | Detecting | SAM | Total changes |
|-----|-----------|-----------|-----------|-----------|-----|---------------|
| Gemini 3 Flash | 8 params, 22/30 tunnels | 8 params, 30/30 | 11 params, 30/30 | 14 params, 30/30 | 44 params, 29/30 | 3,925 |
| GPT 5.4 | 8 params, 26/30 tunnels | 8 params, 30/30 | 12 params, 30/30 | 14 params, 30/30 | 45 params, 30/30 | 3,615 |
| Opus 4.6 | 9 params, 20/30 tunnels | 8 params, 30/30 | 10 params, 30/30 | 14 params, 30/30 | 44 params, 29/30 | 3,965 |

---

## Per-stage adaptation coverage

| Stage | Baseline params | Adapted | Coverage | Top adapted parameter |
|-------|----------------|---------|----------|----------------------|
| Unfolding | 16 | 10 | 63% | `diameter` (27/30 tunnels) |
| Denoising | 8 | 8 | **100%** | `mask_r_low` (30/30 tunnels) |
| Enhancing | 14 | 12 | 86% | `inter_radius` (30/30 tunnels) |
| Detecting | 14 | 14 | **100%** | `hough_threshold_oblique` (30/30 tunnels) |
| SAM | ~50 (nested) | 45 | 90% | `processing.padding` (29/30 tunnels) |

---

## Key insights

1. **Denoising radius masks (`mask_r_low`, `mask_r_high`) are the single most critical parameters.** Adapted in 30/30 tunnels by all 3 LLMs across all 3 conditions, with the highest per-tunnel variation. They directly encode physical tunnel geometry and their misspecification is the #1 cause of depth map white spaces.

2. **Two adaptation modes coexist.** ~11 always-trigger parameters are genuine tunnel-responsive adaptations (vary by tunnel); ~7 are universal baseline corrections (same shift everywhere). This means the sam4tun baseline has ~7 suboptimal defaults that any LLM will fix independently.

3. **Detecting thresholds are the most tunnel-sensitive stage.** Hough thresholds show the highest CV (0.19–0.22), meaning detecting parameters vary the most between tunnels. This aligns with line detection being highly dependent on image contrast and point density.

4. **Unfolding is the most conservative stage.** `diameter` only changes in 27/30 tunnels, and other unfolding parameters rarely change. Geometric unfolding is well-constrained by RANSAC and needs minimal adaptation.

5. **SAM parameters are numerous but correlated.** 45 SAM parameters are adapted, but most are derived geometry (block heights, widths, spacing factors) that scale proportionally with `K_height`, `AB_height`, and `segment_width`. The effective degrees of freedom are ~5 independent values.

6. **All 3 LLMs converge.** Despite different architectures, adaptation patterns are near-identical — same parameters adapted, same tunnel-family clustering, same baseline corrections. This suggests the adaptations are driven by objective tunnel characteristics rather than LLM-specific biases.

---

## Implication for BO search space

- **Tier 1 (tunnel-responsive):** These 11 parameters form the BO search space — their values depend on tunnel characteristics and need per-tunnel optimisation.
- **Tier 2 (baseline corrections):** Fix these at the LLM-corrected values in the BO baseline; no need to search over them since all models agree on one value.
- **Never-changed:** Lock at baseline; exclude from the search space entirely.

---

## Files

| File | Purpose |
|------|---------|
| `skills/scripts/analyze_parameter_adaptations.py` | Main analysis script (reads 1,350 JSONs, compares to baseline, computes CV, ranks by tunnel breadth) |
| `agents/ablation/sam4tun/parameters_*.json` | Baseline parameter files (5 stages) |
| `agents/ablation/{memory,memory+state,memory+state+knowledge}/parameters/` | Adapted parameter files (30 tunnels × 3 LLMs × 5 stages) |
| `methods/journals/critical_parameters.md` | Condensed results table (this file documents the full process) |

# On-Site Rules Baseline — Methodology and Supplementary Details

## Overview

The rules baseline is a deterministic, non-LLM parameter adaptation method. It represents what a practitioner could achieve using only **field-observable inputs** — information available before running any part of the pipeline — to adjust SAM4Tun's expert-tuned defaults for a new tunnel.

No characteriser output, pipeline intermediate state, domain-knowledge document, or LLM inference is used.

## Inputs

Four field-observable values per tunnel, stored in `agents/ablation/rules/site_inputs.json`:

| Input | Source | T1/T2 | T3 | T4/T5 |
| ----- | ------ | ----- | -- | ----- |
| `diameter_m` | Design drawings or tape | 5.5 | 5.5 | 7.5 |
| `family` | Visual inspection | T1_T2 | T3 | T4_T5 |
| `ring_length_m` | Design drawings | 1.2 | 1.2 | 1.8 |
| `segment_per_ring` | Counted visually | 6 | 6 | 7 |

All 30 tunnels fall into these three families. T1/T2 and T3 share geometry but differ in joint pattern (staggered vs continuous). T4/T5 have larger diameter, wider rings, and an extra segment (A4).

## Adaptation Rules

The script `agents/ablation/rules/rule_adapt.py` loads sam4tun defaults from `agents/ablation/sam4tun/parameters_{stage}.json`, applies the rules below, and writes per-tunnel parameter JSONs to `agents/ablation/rules/parameters/{tunnel_id}/`.

### Stage 1: Unfolding

| Parameter | sam4tun default | Rule | T1–T3 value | T4–T5 value |
| --------- | --------------- | ---- | ----------- | ----------- |
| `diameter` | 5.5 | Set to `diameter_m` | 5.5 (unchanged) | 7.5 |

All other unfolding parameters (RANSAC, polynomial degree, etc.) keep defaults.

### Stage 2: Denoising

| Parameter | sam4tun default | Rule | T1–T3 value | T4–T5 value |
| --------- | --------------- | ---- | ----------- | ----------- |
| `mask_r_low` | 2.70 | `diameter_m / 2 - 0.15` | 2.60 | 3.60 |
| `mask_r_high` | 2.80 | `diameter_m / 2 + 0.15` | 2.90 | 3.90 |
| `default_cutoff_z` | 2.70 | `mask_r_high + 0.05` | 2.95 | 3.95 |

Rationale: the radial mask must bracket the tunnel surface. A ±0.15 m margin around the nominal radius accommodates typical scan noise without requiring percentile estimates.

### Stage 3: Enhancing

| Parameter | sam4tun default | Rule | T1–T3 value | T4–T5 value |
| --------- | --------------- | ---- | ----------- | ----------- |
| `n_segment_end` | 5 | `segment_per_ring - 1` | 5 (unchanged) | 6 |

This parameter controls the angular sweep range for the depth-map enhancement. 6-segment tunnels use indices 0–5; 7-segment tunnels use 0–6.

### Stage 4: Detecting

| Parameter | sam4tun default | Rule | T1–T3 value | T4–T5 value |
| --------- | --------------- | ---- | ----------- | ----------- |
| `ring_spacing_constant` | 1.2 | Set to `ring_length_m` | 1.2 (unchanged) | 1.8 |

All Hough parameters, angle ranges, and morphological settings keep defaults. This is the conservative choice: the rules do not attempt to lower detection thresholds for complex tunnels (where sparser point clouds might warrant it), avoiding the false-positive risk that comes with over-aggressive detection.

### Stage 5: SAM Segmentation

| Parameter | sam4tun default | Rule | T1–T3 value | T4–T5 value |
| --------- | --------------- | ---- | ----------- | ----------- |
| `segment_per_ring` | 6 | Set to actual count | 6 (unchanged) | 7 |
| `segment_order` | [K, B1, A1, A2, A3, B2] | 6 or 7-seg layout | unchanged | [K, B1, A1, A2, A3, A4, B2] |
| `segment_width` | 1200 | Scale by `ring_length / 1.2` | 1200 (unchanged) | 1800 |
| `K_height` | 1079.92 | Scale by `diameter / 5.5` | unchanged | 1472.62 |
| `AB_height` | 3239.77 | Scale by `diameter / 5.5` | unchanged | 4417.87 |
| `processing.padding` | 150 | Scale by `diameter / 5.5` | unchanged | 205 |
| `processing.crop_margin` | 50 | Scale by `diameter / 5.5` | unchanged | 68 |
| `processing.y_bounds` | [4200, 13100] | Scale by `diameter / 5.5` | unchanged | [5727, 17864] |

Scaling is only applied to T4/T5 tunnels. T1–T3 tunnels match the reference geometry exactly, so all SAM parameters remain at defaults.

## Summary of Changes

| | T1/T2 (regular) | T3 (continuous) | T4/T5 (complex) |
| --- | --- | --- | --- |
| Parameters changed | 0 | 0 | 8+ |
| Unchanged from sam4tun | All ~80 | All ~80 | ~72 |
| Key change | — | — | diameter, radial masks, ring spacing, SAM template |

For T1/T2 and T3 tunnels, the rules produce parameters **identical** to sam4tun defaults, because the reference configuration was tuned for the 5.5 m family. The rules baseline only provides meaningful adaptation for T4/T5 complex tunnels.

## Execution

```bash
# Generate parameter JSONs for all 30 tunnels
python agents/ablation/rules/rule_adapt.py --all

# Run the pipeline with rules parameters
./run_agents.sh 4-4 --ablation rules
./run_agents.sh --all --ablation rules
```

The pipeline resolves parameters via `agents/pipeline_data.py` with `ablation_code="rules"`. Parameter files use the suffix `_rules` (model-independent, since no LLM is involved):
```
agents/ablation/rules/parameters/{tunnel_id}/parameters_{stage}_rules.json
```

Output data goes to `data/ablation/rules/{tunnel_id}/`.

## Failure Cases

Three complex tunnels (4-2, 5-5, 5-7) fail under rules and are scored as mIoU = 0. The failure occurs in the detecting stage: the default Hough thresholds produce a line-count or row-count mismatch when applied to the larger, sparser depth maps of these tunnels. Because rules do not adjust any detection thresholds (only `ring_spacing_constant`), the detector receives the same sensitivity settings as a 5.5 m tunnel applied to a 7.5 m depth map.

The LLM conditions avoid these failures because the state context (detected line counts, left/right coverage) allows the agent to diagnose and correct detection parameters. The rules baseline, by design, has no access to this feedback.

## Comparison with LLM Conditions

| What rules use | What LLMs additionally use |
| -------------- | ------------------------- |
| Diameter (drawings) | Raw characteristics (density, z-range, nn-distance) |
| Ring length (drawings) | Unfolded characteristics (r-percentiles, h-span, theta-span) |
| Segment count (visual) | Denoised characteristics (surface completeness, curvature) |
| Family (visual) | Enhanced characteristics (coverage uniformity, spacing) |
| | Detected characteristics (line counts, coverage balance) |
| | Domain knowledge (parameter semantics, diagnostic rules) |
| | Chain-of-thought reasoning over all of the above |

The rules baseline tests a specific hypothesis: how far can simple family-level adjustments from field-observable inputs take you? The answer: substantially on complex tunnels (mIoU 0.042 → 0.137, +0.095) where the geometry mismatch is gross, but not at all on regular tunnels where sam4tun defaults already match. The LLM conditions go further by adapting ~20–40 parameters per tunnel based on quantitative pipeline feedback.

## Files

| File | Purpose |
| ---- | ------- |
| `agents/ablation/rules/rule_adapt.py` | Parameter generation script |
| `agents/ablation/rules/site_inputs.json` | Field-observable inputs for 30 tunnels |
| `agents/ablation/rules/parameters/{tid}/` | Generated parameter JSONs (5 per tunnel) |
| `agents/ablation/sam4tun/parameters_{stage}.json` | Default parameters (starting point) |
| `agents/pipeline_data.py` | Ablation condition registry (rules entry) |
| `data/ablation/rules/{tid}/` | Pipeline output data |
| `methods/journals/comparison_rules.md` | Per-tunnel mIoU comparison: rules vs 3 LLMs |
| `methods/papers/scripts/recompute_rules_stats.py` | Statistical analysis script |
| `methods/papers/figs/rules_comparison.pdf` | Bar chart: rules vs ablation conditions |

## Appendix Material (for paper)

The following appendix sections in `methods/papers/appendices.tex` cover the rules baseline:

1. **Appendix E (§app:rules)** — Table `tab:rules-spec`: the 6 rule-driven parameters with their formulas
2. **Appendix F (§app:distribution)** — Table `tab:distribution`: performance distribution including rules column (mean 0.201, std 0.132, min 0.000, max 0.484)
3. **Appendix G (§app:perclass)** — Tables `tab:perclass-regular` and `tab:perclass-complex`: per-class IoU for rules vs all conditions
4. **Main text Section 3.1.1** — Prose description of the rules baseline design and rationale
5. **Main text Section 4.1** — Results analysis comparing rules to LLM conditions
6. **Main text Section 4.3** — Discussion of the role of LLM reasoning vs rules

## Pending Tasks (post-rerun follow-ups)

The rerun promotion (3 tunnels x 2 conditions x 3 models, 13 improved combos) changed Opus m+s and m+s+k per-tunnel mIoU values. Two downstream tables need refreshing.

### Task 1: Re-bootstrap Opus Cohen's d and 95% CI (Table 4)

**Target**: `methods/papers/r4tun_review_v4.tex`, `tab:main-results` (lines 333-368), Opus column for the m+s and m+s+k rows.

**Source**: per-tunnel deltas from `methods/journals/comparison_anthropic.md` (the updated post-rerun journal; `reviews/logs/comparison_anthropic.md` referenced in the original request resolves to this file — it is the only copy in the repo).

**Provisional values** (in place but unverified):
- m+s: Cohen's d = 1.77, 95% CI = [0.135, 0.205]
- m+s+k: Cohen's d = 1.96, 95% CI = [0.158, 0.227]

**Verified values** (10,000 bootstrap resamples, seed=42, percentile method):
- m+s: mean_delta=0.1698, std=0.0959, Cohen's d = 1.77, 95% CI = [0.137, 0.204]
- m+s+k: mean_delta=0.1920, std=0.0982, Cohen's d = 1.95, 95% CI = [0.158, 0.227]
- Status: DONE. All values patched in r4tun_review_v4.tex (m+s d=1.77, CI=[0.137,0.204]; m+s+k d=1.95, CI=[0.158,0.227]).

**Computation**:
1. Parse all 30 tunnel rows from `comparison_anthropic.md`
2. Compute paired deltas: delta_i = mIoU(cond, i) - mIoU(sam4tun, i) for n=30
3. Cohen's d = mean(delta) / std(delta, ddof=1)
4. Bootstrap 95% CI on mean(delta): 10,000 resamples, seed=42, percentile method
5. Compare against provisional values; confirm or replace

**Files updated** (all patched):
- `r4tun_review_v4.tex` tab:main-results — Opus m+s and m+s+k rows (d, CI, mIoU, Δ), Gemini m+s+k row (d, CI, mIoU, Δ), GPT m+s+k row (d, CI)
- `r4tun_review_v4.tex` tab:cross-model — Opus (Δ, d, CI), GPT (d, CI), Gemini (Δ, d, CI)
- `r4tun_review_v4.tex` tab:cumulative_miou — Opus m→m+s and m+s→m+s+k, GPT m→m+s and m+s→m+s+k, Gemini m+s→m+s+k
- Inline narrative: abstract (0.316--0.342, 0.178--0.194), Section 4.1 (d=1.51--1.95, d=1.4--2.4, 0.178--0.194, d=2.0--2.4, 0.136--0.152), Section 4.2 (m+s+k complex 0.184; m+s d=1.33--1.77; knowledge +0.014--+0.022, d=0.13--0.43, CI/binomial updated), Discussion (+0.103--+0.176), Limitations (0.178--0.194), Conclusions (0.316--0.342)

### Task 2: Regenerate per-class IoU Tables 16 and 17

**Target**: `methods/papers/appendices.tex`, `tab:perclass-regular` (lines 250-265) and `tab:perclass-complex` (lines 267-283). Currently hold v5 numbers; user audit flagged a stray 0.155 near line 254.

**Source**: parse `## Per-class IoU` block from each `data/ablation_anthropic/{cond}/{tid}/evaluation/performance.md` for cond in {sam4tun, memory, memory+state, memory+state+knowledge}. Rules column from `data/ablation/rules/{tid}/evaluation/performance.md` (rules values are unchanged).

**Aggregation**:
- Table 16 (Regular, n=13): 7 classes (Background, K, B1, A1, A2, A3, B2), tunnels 1-1..2-5 + 3-1-1..3-1-3
- Table 17 (Complex, n=17): 8 classes (adds A4), tunnels 4-1..4-10 + 5-1..5-7; rules column includes 3 failed tunnels scored as zero
- Mean per-class IoU across tunnels in each family, rounded to 3 decimal places

**Files to update**:
- `appendices.tex` lines 256-262 (Table 16 body rows) and 273-280 (Table 17 body rows)

**Verified values** (computed from `data/ablation_anthropic/` and `data/ablation/rules/`):

Table 16 — Regular (n=13), mean per-class IoU:

| Class | sam4tun | memory | m+s   | m+s+k | rules |
|-------|---------|--------|-------|-------|-------|
| Bg    | 0.640   | 0.559  | 0.741 | 0.751 | 0.597 |
| K     | 0.192   | 0.129  | 0.373 | 0.386 | 0.222 |
| B1    | 0.261   | 0.206  | 0.527 | 0.540 | 0.250 |
| A1    | 0.253   | 0.276  | 0.537 | 0.542 | 0.263 |
| A2    | 0.150   | 0.159  | 0.380 | 0.420 | 0.144 |
| A3    | 0.286   | 0.259  | 0.519 | 0.550 | 0.289 |
| B2    | 0.256   | 0.232  | 0.534 | 0.558 | 0.236 |

Table 17 — Complex (n=17), mean per-class IoU (rules: 3 failed tunnels scored 0):

| Class | sam4tun | memory | m+s   | m+s+k | rules |
|-------|---------|--------|-------|-------|-------|
| Bg    | 0.337   | 0.358  | 0.561 | 0.581 | 0.533 |
| K     | 0.000   | 0.034  | 0.146 | 0.142 | 0.000 |
| B1    | 0.000   | 0.001  | 0.095 | 0.142 | 0.041 |
| A1    | 0.000   | 0.005  | 0.139 | 0.167 | 0.072 |
| A2    | 0.000   | 0.006  | 0.132 | 0.131 | 0.083 |
| A3    | 0.000   | 0.017  | 0.122 | 0.150 | 0.149 |
| B2    | 0.000   | 0.000  | 0.034 | 0.097 | 0.099 |
| A4    | 0.000   | 0.019  | 0.128 | 0.142 | 0.116 |

LaTeX rows for Table 16 (Regular):
```
        Bg   & 0.640 & 0.559 & 0.741 & 0.751 & 0.597 \\
        K    & 0.192 & 0.129 & 0.373 & 0.386 & 0.222 \\
        B1   & 0.261 & 0.206 & 0.527 & 0.540 & 0.250 \\
        A1   & 0.253 & 0.276 & 0.537 & 0.542 & 0.263 \\
        A2   & 0.150 & 0.159 & 0.380 & 0.420 & 0.144 \\
        A3   & 0.286 & 0.259 & 0.519 & 0.550 & 0.289 \\
        B2   & 0.256 & 0.232 & 0.534 & 0.558 & 0.236 \\
```

LaTeX rows for Table 17 (Complex):
```
        Bg   & 0.337 & 0.358 & 0.561 & 0.581 & 0.533 \\
        K    & 0.000 & 0.034 & 0.146 & 0.142 & 0.000 \\
        B1   & 0.000 & 0.001 & 0.095 & 0.142 & 0.041 \\
        A1   & 0.000 & 0.005 & 0.139 & 0.167 & 0.072 \\
        A2   & 0.000 & 0.006 & 0.132 & 0.131 & 0.083 \\
        A3   & 0.000 & 0.017 & 0.122 & 0.150 & 0.149 \\
        B2   & 0.000 & 0.000 & 0.034 & 0.097 & 0.099 \\
        A4   & 0.000 & 0.019 & 0.128 & 0.142 & 0.116 \\
```

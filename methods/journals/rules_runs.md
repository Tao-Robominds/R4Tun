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

# Daily Journal — 30 April 2026 (Step 01 execution)

## Objective

Execute step 01 ring-regime discovery against `data/subsets/` per the sampling decision in `journal_2026-04-30_ring_regime_sampling.md` and the plan in `methods/plans/steps/01_ring_regime_discovery.md`. Produce the descriptor catalog, regime labels, BO panels, and audit trail.

---

## Implementation

`methods/ablation/scripts/build_ring_regimes.py` (run via `./venv/bin/python`).

Run command:

```bash
./venv/bin/python methods/ablation/scripts/build_ring_regimes.py \
  --subsets-dir data/subsets \
  --run regime_v1 \
  --families 4 5 \
  --regular-families 1 2 3 \
  --regular-ratio 0.20 \
  --panel-size 30 \
  --holdout-per-regime 1
```

Outputs land under `data/subsets/workflow/regime_v1/01_ring_regime_discovery/`.

### Conventions used

- Per-ring centroid for the angle: `theta = -atan2(z - cz, x - cx) mod 360`. Centering matters; without it `angular_gap_frac` is meaningless and the walking order scrambles.
- Segment-id → block-name mapping from `agents/3_segmentation/segmentation.py`: `{0:BG, 1:K, 2:B1, 3:A1, 4:A2, 5:A3, 6:A4, 7:B2}`.
- Density bins from `logs/4-1/balanced_30_rings_summary.md`: sparse (<10k) / low (<50k) / medium (<200k) / dense (≥200k).
- Coverage tier: full (gap < 2%) / partial (< 10%) / poor (≥ 10%).
- K span tier from per-catalog percentiles of irregular rings: narrow (<P20), normal (P20–P80), wide (≥P80).
- Regime label = `{density}_{coverage}_{k_span}_{pattern_type}`. K quadrant kept as a balancing axis, not part of the label.

### Drop rules (applied during cataloging)

Irregular rings are dropped when:

1. K segment is missing (`no_k`).
2. K is present but its non-BG cyclic neighbors are not `{B1, B2}` (`k_neighbors_not_B1_B2`).

Both cases indicate noisy / non-meaningful rings and are recorded in `dropped_rings.csv` for traceability.

---

## Final state

`data/subsets/workflow/regime_v1/01_ring_regime_discovery/`

- `ring_descriptors.csv`, `ring_regimes.csv`, `regime_sampling_panel.json`, `regime_summary.md`, `descriptor_validation_against_data_rings.md`, `regime_distribution.csv`, `dropped_rings.csv`.

### Catalog

- 567 rings cataloged.
- 227 irregular target (family 4: 158, family 5: 69).
- 340 regular sanity (families 1/2/3).

### Drops (12 irregular rings)

| reason | count | rings |
|--------|------:|-------|
| `no_k` | 3 | `4-1/113`, `4-10/394`, `4-3/174` |
| `k_neighbors_not_B1_B2` | 9 | all from `4-12` |

`4-12` appears to use a different segment-id encoding ring-to-ring; its rings are excluded until that's understood.

### Distribution (irregular only)

- All retained irregular rings are `canonical` (109) or `reversed_canonical` (118) by construction.
- coverage_tier: full=221, partial=6.
- k_quadrant: q0=66, q1=73, q2=45, q3=43.
- k_span_tier: narrow=38, normal=140, wide=49.
- density_tier: dense=71, low=70, medium=84, sparse=2.

### Panels

| panel | family-4 | family-5 | regular sanity | total |
|-------|---------:|---------:|---------------:|------:|
| `panel_20` | 9 | 7 | 4 | 20 |
| `panel_30` | 14 | 10 | 6 | 30 |

Holdout: 1 ring per irregular regime, reserved before panel selection so it is disjoint from `panel_30` by construction.

---

## Validation against `data/rings/summary.json`

- Matched rings (overlap with curated summary): ~99.
- `n_points`: max |Δ| = 0 (exact match).
- `angular_gap_frac`: mean |Δ| ≈ 0.004.
- K quadrant match rate: ~32%; K angle and span differ noticeably.
- Walking-order match rate: 0%.

The summary's `walking_order` and `k_angle_deg` use a different (image-space) convention. Plan risk acknowledged: per-ring walking orders should not be cross-mapped one-to-one between the two sources. Only `n_points`, `angular_gap_frac`, and (with caveats) K quadrant are directly comparable.

---

## Repo changes touching this step

- `.cursor/rules/intrinsic-project.mdc` — venv rule tightened: `./venv/bin/python` is mandatory; if `./venv/bin/python` is missing or broken, repair in place with `python3.12 -m venv ./venv --upgrade-deps`. No fallback to `python3` / pyenv shims / system Python / `--user` / `.deps`.
- `methods/ablation/scripts/build_ring_regimes.py` — sole new code; CLI matches the plan.

---

## Regime characteristics (irregular families 4 and 5, n = 227)

These tables are the methodology-level evidence for the regime axes. Each row is one regime label (`{density}_{coverage}_{k_span}_{pattern}`); k_quadrant is reported as a balancing axis, not as part of the label. All quartiles are p25 / median / p75. Point counts are integers; angles and radii are degrees and metres respectively.

### Why these axes

- **Density tier** (`n_points`): captures scan-quality regime — sparse (<10k) vs low (<50k) vs medium (<200k) vs dense (≥200k). It sets how much evidence per ring is available to BO and to the proxy predictor.
- **Coverage tier** (`angular_gap_frac`): captures occlusion regime — full (<2% gap) vs partial (<10%) vs poor (≥10%). Partial coverage breaks the canonical block tile and stresses K detection.
- **K span tier** (per-catalog percentiles of `k_span_deg`): narrow (<P20) vs normal (P20–P80) vs wide (≥P80). K span drives template width and groove alignment difficulty.
- **Pattern type** (cyclic walking order under `{1:K, 2:B1, 3:A1, 4:A2, 5:A3, 6:A4, 7:B2}`): canonical vs reversed_canonical (rings outside these are filtered out by the K-neighbor rule).
- **K quadrant** (kept off the label, reported as a counter): q0..q3 of K's centroid angle; relevant for BO seeding diversity but not regime identity.

### Axis-level descriptive stats

#### density_tier

| density_tier | n | n_points (med) | gap_frac (med) | k_span (med) | seg_cv (med) | complexity (med) |
|---|---|---|---|---|---|---|
| dense  | 71 | 434,780 | 0.000 | 27.0 | 0.44 | 1.30 |
| low    | 70 |  32,655 | 0.000 | 26.5 | 0.51 | 1.30 |
| medium | 84 |  98,065 | 0.000 | 29.0 | 0.40 | 1.30 |
| sparse |  2 |   8,943 | 0.001 | 47.5 | 0.58 | 1.05 |

#### coverage_tier

| coverage_tier | n | n_points (med) | gap_frac (med) | k_span (med) | seg_cv (med) | complexity (med) |
|---|---|---|---|---|---|---|
| full    | 221 |  87,340 | 0.000 | 28.0 | 0.45 | 1.30 |
| partial |   6 | 610,746 | 0.026 | 22.5 | 0.48 | 1.80 |

#### k_span_tier

| k_span_tier | n | n_points (med) | gap_frac (med) | k_span (med) | seg_cv (med) | complexity (med) |
|---|---|---|---|---|---|---|
| narrow |  38 |  61,146 | 0.007 | 19.0 | 0.50 | 1.60 |
| normal | 140 | 132,747 | 0.000 | 27.0 | 0.44 | 1.30 |
| wide   |  49 |  78,164 | 0.000 | 55.0 | 0.42 | 1.30 |

#### pattern_type

| pattern_type | n | n_points (med) | gap_frac (med) | k_span (med) | seg_cv (med) | complexity (med) |
|---|---|---|---|---|---|---|
| canonical          | 109 | 97,333 | 0.000 | 30.0 | 0.45 | 1.00 |
| reversed_canonical | 118 | 95,172 | 0.000 | 25.5 | 0.45 | 1.30 |

### Per-regime: scan condition and geometry

| regime | n | fams | tunnels | n_points (p25/med/p75) | gap_frac (med) | radius (med) | radius_iqr (med) |
|---|---|---|---|---|---|---|---|
| dense_full_narrow_canonical | 2 | 4=1, 5=1 | 2 | 424,904 / 499,288 / 573,673 | 0.001 | 3.69 | 3.40 |
| dense_full_narrow_reversed_canonical | 6 | 4=4, 5=2 | 4 | 352,090 / 417,414 / 434,469 | 0.001 | 3.31 | 3.48 |
| dense_full_normal_canonical | 23 | 4=19, 5=4 | 11 | 349,672 / 428,382 / 639,698 | 0.000 | 3.42 | 3.36 |
| dense_full_normal_reversed_canonical | 21 | 4=12, 5=9 | 11 | 353,514 / 478,081 / 643,058 | 0.000 | 3.36 | 3.40 |
| dense_full_wide_canonical | 9 | 4=4, 5=5 | 5 | 371,429 / 428,181 / 544,152 | 0.000 | 2.90 | 3.27 |
| dense_full_wide_reversed_canonical | 5 | 4=3, 5=2 | 3 | 239,272 / 440,114 / 453,257 | 0.000 | 3.19 | 3.19 |
| dense_partial_narrow_reversed_canonical | 1 | 4=1 | 1 | 642,983 / 642,983 / 642,983 | 0.028 | 4.47 | 2.79 |
| dense_partial_normal_canonical | 1 | 4=1 | 1 | 647,549 / 647,549 / 647,549 | 0.025 | 4.49 | 2.78 |
| dense_partial_normal_reversed_canonical | 3 | 4=3 | 3 | 576,052 / 579,860 / 610,746 | 0.025 | 3.79 | 3.09 |
| low_full_narrow_canonical | 3 | 4=1, 5=2 | 2 | 31,872 / 34,259 / 37,728 | 0.011 | 5.08 | 2.52 |
| low_full_narrow_reversed_canonical | 15 | 4=12, 5=3 | 6 | 22,732 / 34,026 / 38,646 | 0.006 | 5.26 | 2.69 |
| low_full_normal_canonical | 23 | 4=18, 5=5 | 11 | 20,016 / 23,201 / 35,185 | 0.000 | 6.87 | 3.26 |
| low_full_normal_reversed_canonical | 15 | 4=10, 5=5 | 9 | 19,432 / 28,034 / 35,554 | 0.000 | 6.88 | 3.22 |
| low_full_wide_canonical | 6 | 4=5, 5=1 | 4 | 29,032 / 34,883 / 36,324 | 0.000 | 7.81 | 2.36 |
| low_full_wide_reversed_canonical | 8 | 4=4, 5=4 | 6 | 26,078 / 33,726 / 42,920 | 0.000 | 7.32 | 2.79 |
| medium_full_narrow_canonical | 4 | 4=2, 5=2 | 2 | 66,440 / 68,998 / 72,920 | 0.013 | 4.91 | 2.54 |
| medium_full_narrow_reversed_canonical | 7 | 4=6, 5=1 | 4 | 64,240 / 72,214 / 146,960 | 0.011 | 4.73 | 2.73 |
| medium_full_normal_canonical | 23 | 4=21, 5=2 | 9 | 73,180 / 134,690 / 153,856 | 0.000 | 4.90 | 3.04 |
| medium_full_normal_reversed_canonical | 30 | 4=18, 5=12 | 10 | 73,024 / 101,941 / 156,528 | 0.004 | 4.94 | 3.10 |
| medium_full_wide_canonical | 13 | 4=7, 5=6 | 7 | 66,135 / 74,363 / 118,215 | 0.000 | 5.79 | 2.38 |
| medium_full_wide_reversed_canonical | 6 | 4=4, 5=2 | 3 | 83,322 / 99,380 / 118,867 | 0.000 | 5.65 | 2.60 |
| medium_partial_normal_reversed_canonical | 1 | 4=1 | 1 | 128,460 / 128,460 / 128,460 | 0.028 | 4.59 | 3.10 |
| sparse_full_wide_canonical | 2 | 4=1, 5=1 | 2 | 8,440 / 8,942 / 9,445 | 0.001 | 12.07 | 2.46 |

### Per-regime: K characteristics and complexity

| regime | n | k_angle (med) | k_quadrant mix | k_span (p25/med/p75) | seg_cv (med) | complexity (med) |
|---|---|---|---|---|---|---|
| dense_full_narrow_canonical | 2 | 223.2 | q1=1, q3=1 | 6.5 / 11.0 / 15.5 | 0.50 | 1.36 |
| dense_full_narrow_reversed_canonical | 6 | 268.8 | q0=1, q1=1, q2=1, q3=3 | 15.2 / 16.5 / 18.5 | 0.42 | 1.60 |
| dense_full_normal_canonical | 23 | 216.7 | q0=6, q1=5, q2=6, q3=6 | 25.0 / 27.0 / 33.5 | 0.41 | 1.00 |
| dense_full_normal_reversed_canonical | 21 | 279.3 | q0=1, q1=3, q2=6, q3=11 | 23.0 / 26.0 / 32.0 | 0.46 | 1.30 |
| dense_full_wide_canonical | 9 | 63.7 | q0=6, q1=1, q2=1, q3=1 | 55.0 / 80.0 / 92.0 | 0.45 | 1.30 |
| dense_full_wide_reversed_canonical | 5 | 159.6 | q0=1, q1=2, q2=1, q3=1 | 39.0 / 48.0 / 58.0 | 0.40 | 1.60 |
| dense_partial_narrow_reversed_canonical | 1 | 50.0 | q0=1 | 21.0 / 21.0 / 21.0 | 0.48 | 2.10 |
| dense_partial_normal_canonical | 1 | 130.9 | q1=1 | 22.0 / 22.0 / 22.0 | 0.48 | 1.50 |
| dense_partial_normal_reversed_canonical | 3 | 205.5 | q2=3 | 22.5 / 23.0 / 27.0 | 0.47 | 1.80 |
| low_full_narrow_canonical | 3 | 41.0 | q0=2, q1=1 | 18.5 / 19.0 / 20.0 | 0.52 | 1.30 |
| low_full_narrow_reversed_canonical | 15 | 60.6 | q0=10, q1=5 | 18.5 / 19.0 / 20.0 | 0.52 | 1.60 |
| low_full_normal_canonical | 23 | 146.3 | q0=6, q1=10, q2=4, q3=3 | 25.0 / 28.0 / 32.0 | 0.53 | 1.00 |
| low_full_normal_reversed_canonical | 15 | 79.5 | q0=8, q1=5, q2=2 | 23.5 / 25.0 / 28.5 | 0.51 | 1.30 |
| low_full_wide_canonical | 6 | 153.8 | q0=2, q1=2, q2=1, q3=1 | 43.8 / 46.5 / 50.0 | 0.46 | 1.30 |
| low_full_wide_reversed_canonical | 8 | 206.9 | q1=3, q2=4, q3=1 | 45.0 / 58.0 / 71.5 | 0.47 | 1.60 |
| medium_full_narrow_canonical | 4 | 90.9 | q0=2, q1=2 | 14.8 / 19.5 / 20.2 | 0.47 | 1.30 |
| medium_full_narrow_reversed_canonical | 7 | 141.9 | q0=1, q1=5, q3=1 | 18.0 / 19.0 / 20.5 | 0.45 | 1.60 |
| medium_full_normal_canonical | 23 | 127.8 | q0=8, q1=9, q2=3, q3=3 | 25.5 / 28.0 / 30.5 | 0.37 | 1.00 |
| medium_full_normal_reversed_canonical | 30 | 162.9 | q0=5, q1=14, q2=4, q3=7 | 25.0 / 27.5 / 31.0 | 0.38 | 1.30 |
| medium_full_wide_canonical | 13 | 234.6 | q0=2, q1=2, q2=6, q3=3 | 53.0 / 67.0 / 83.0 | 0.40 | 1.30 |
| medium_full_wide_reversed_canonical | 6 | 123.4 | q0=3, q1=1, q2=1, q3=1 | 39.8 / 42.5 / 106.8 | 0.37 | 1.60 |
| medium_partial_normal_reversed_canonical | 1 | 223.8 | q2=1 | 30.0 / 30.0 / 30.0 | 0.31 | 1.80 |
| sparse_full_wide_canonical | 2 | 141.3 | q0=1, q2=1 | 46.2 / 47.5 / 48.8 | 0.58 | 1.05 |

### Reading the tables

- **`fams` and `tunnels`** quantify diversity: e.g. `dense_full_normal_canonical` (n=23) spans 11 distinct tunnels and both irregular families, so it is robust enough to be a primary BO regime; `medium_partial_normal_reversed_canonical` (n=1) is rare and should be drawn from holdout, not BO seeds.
- **`gap_frac`** is the dominant signal that separates `full` from `partial` rings; partial regimes also tend to have noticeably higher complexity (1.8–2.1 vs ~1.0–1.6), confirming that occlusion is a real-difficulty axis.
- **`k_span` p25/med/p75** show the within-tier dispersion. `wide` regimes span 40–100°, `narrow` regimes 15–22° — the band gap justifies treating K span as a regime axis rather than a continuous covariate.
- **`k_quadrant` mix** is reported per regime to confirm K position is not collinear with regime label. For example `dense_full_normal_canonical` covers all four quadrants almost evenly (q0=6/q1=5/q2=6/q3=6), so per-regime BO does not implicitly fix K position.
- **`seg_cv`** (segment balance) is structurally insensitive to density (~0.4–0.5 across density tiers) but does flag rare high-imbalance rings; it complements `k_span` rather than replacing it.
- **`ring_complexity_score`** rises with `partial` coverage, `narrow` K, and `reversed_canonical` order — consistent with the qualitative expectation that those rings stress K detection and groove alignment.

## Minimum reference panel (6 rings)

If we have to pick the **smallest possible** ring set for BO that still spans every primary diversity axis (so it can serve as the long-term reference panel for proxy fitting and threshold calibration), greedy axis-cover on the irregular catalog gives a clean 6-ring answer.

### Method

For each candidate ring, count how many still-uncovered (axis, level) pairs it would add across `density_tier`, `coverage_tier`, `k_span_tier`, `pattern_type`, `family`, and `k_quadrant`. At each step pick the ring with the largest gain. Tie-break by:

1. Largest regime population (prefer rings from stable, well-populated regimes so the exemplar is representative, not an outlier).
2. Lowest in-regime deviation from medians of `n_points`, `k_span_deg`, and `segment_balance_cv` (exemplar closest to the regime medoid).
3. Anchor preference for `canonical` / `full` / `normal` when still tied.

### Result

Saved to `data/subsets/workflow/regime_v1/01_ring_regime_discovery/minimum_reference_panel.json`.

| # | tunnel | ring | family | regime | density | coverage | k_span | pattern | k_quad | n_points | k_span_deg | regime n | new axes covered |
|---|--------|------|--------|--------|---------|----------|--------|---------|--------|---------:|-----------:|---------:|------------------|
| 1 | 5-5  | 258 | 5 | medium_full_normal_reversed_canonical   | medium | full    | normal | reversed_canonical | q1 |  91,547 | 27.0 | 30 | medium, full, normal, reversed_canonical, family 5, q1 |
| 2 | 4-9  | 366 | 4 | dense_full_wide_canonical               | dense  | full    | wide   | canonical          | q0 | 396,950 | 56.0 |  9 | dense, wide, canonical, family 4, q0 |
| 3 | 5-3  | 190 | 5 | low_full_normal_canonical               | low    | full    | normal | canonical          | q2 |  20,512 | 23.0 | 23 | low, q2 |
| 4 | 4-8  | 337 | 4 | medium_full_narrow_reversed_canonical   | medium | full    | narrow | reversed_canonical | q3 | 157,262 | 21.0 |  7 | narrow, q3 |
| 5 | 4-1  | 116 | 4 | dense_partial_normal_reversed_canonical | dense  | partial | normal | reversed_canonical | q2 | 579,860 | 22.0 |  3 | partial |
| 6 | 4-6  | 283 | 4 | sparse_full_wide_canonical              | sparse | full    | wide   | canonical          | q2 |   7,938 | 45.0 |  2 | sparse |

### Coverage check

| axis | levels covered | levels missing |
|------|---------------|----------------|
| density_tier  | sparse, low, medium, dense | – |
| coverage_tier | full, partial              | – |
| k_span_tier   | narrow, normal, wide       | – |
| pattern_type  | canonical, reversed_canonical | – |
| family        | 4, 5                       | – |
| k_quadrant    | q0, q1, q2, q3             | – |

Family 4 / 5 split = 4 / 2; q-mix = q0×1, q1×1, q2×3, q3×1 (q2 over-represented because two of the rare-axis picks — partial and sparse — happen to land in q2).

### Caveats

- Rings 5 and 6 come from very small regimes (n = 3 and n = 2). Those axis-levels (`partial` coverage, `sparse` density) are genuinely rare in the catalog — there is no way to cover them from a large regime. Treat them as "edge probes", not as reference rings for the modal pipeline.
- Rings 1–4 come from regimes with n ≥ 7 (and ring 1 from the largest regime, n = 30); these are the "stable" reference rings that should drive the main BO objective and proxy fit.
- If 8 rings are acceptable instead of 6, two improvements are recommended:
  1. Add `4-x` and `5-x` rings in `low_full_narrow_reversed_canonical` (n = 15) and `medium_full_normal_canonical` (n = 23) to balance family 5 and add q-coverage redundancy on populous regimes.
  2. Replace ring 5 (`4-1/116`, partial coverage from a 3-regime) with another `partial` exemplar only if BO needs more than one partial-coverage data point.

### Suggested usage

- **BO seeds:** rings 1–4 (the four stable rings). Run the main BO loop here so the reference posterior is dominated by populous regimes.
- **Stress / out-of-distribution probes:** rings 5–6. Run BO on them too but treat their results as a robustness check, not as inputs to the proxy fit on its own.
- **Threshold calibration:** use rings 1–4 only.
- **Reference panel for step 07 generalisation:** keep all 6 reserved.

## Next step

Step 02 (`methods/plans/steps/02_bo_calibration.md`) consumes `regime_sampling_panel.json` (`panel_20` for debug, `panel_30` for the main BO calibration). Holdout is reserved for step 07 generalisation testing.

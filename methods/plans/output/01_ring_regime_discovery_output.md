# 01 Ring-Regime Discovery Output

## Objective
Document ring descriptors, regime assignment quality, and the representative BO sampling panel built from the cleaned `data/subsets/` pool only.

## Run
- `run_id`: `regime_v1`
- Subset pool: 30 sub-tunnels (1×5, 2×5, 3×3, 4×10, 5×7) under `data/subsets/`.
- Script: `methods/ablation/scripts/build_ring_regimes.py`

## Artifacts
- `data/subsets/workflow/regime_v1/01_ring_regime_discovery/ring_descriptors.csv`
- `data/subsets/workflow/regime_v1/01_ring_regime_discovery/ring_regimes.csv`
- `data/subsets/workflow/regime_v1/01_ring_regime_discovery/dropped_rings.csv`
- `data/subsets/workflow/regime_v1/01_ring_regime_discovery/regime_distribution.csv`
- `data/subsets/workflow/regime_v1/01_ring_regime_discovery/regime_sampling_panel.json`
- `data/subsets/workflow/regime_v1/01_ring_regime_discovery/regime_summary.md`
- `data/subsets/workflow/regime_v1/01_ring_regime_discovery/descriptor_validation_against_data_rings.md`

## Pool counts
- Total rings cataloged: 296 (irregular = 166, regular = 130)
- Dropped irregular rings (no K, or K not surrounded by B1/B2): 3
  - Family 1 / 2 / 3 (sanity_regular): 50 / 50 / 30
  - Family 4 / 5 (target_irregular): 97 / 69

## Distribution (irregular only)
- `density_tier`: dense=52, low=50, medium=62, sparse=2
- `coverage_tier`: full=163, partial=3
- `k_quadrant`: q0=46, q1=55, q2=30, q3=35
- `k_span_tier`: narrow=31, normal=98, wide=37
- `pattern_type`: canonical=79, reversed_canonical=87

## Panels
- `panel_20`: 20 rings (irregular=16, regular=4); family 4=9, family 5=7.
- `panel_30`: 30 rings (irregular=24, regular=6); family 4=14, family 5=10.
- `holdout`: 22 irregular rings, one per non-empty regime.

Full per-ring tables are in `regime_summary.md`.

## Required evidence
- Descriptor coverage across all rings in the cleaned subset pool.
- Regime distribution summary.
- Justification for representative ring selection (axis-cover with regime-population tie-break).

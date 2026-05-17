# Single-Instance Validation Proof

- Case id: `4-6/r275`
- Sandbox path: `logs/v6_agents_parameterized_smoke_v1/4-6/r275/`
- Data lineage: reused verified preprocessing inputs from `data/v6/4-6/r275/`
- Parameter lineage:
  - `logs/v6_deterministic_baseline_v1/4-6/r275/parameters_detection.json`
  - `logs/v6_deterministic_baseline_v1/4-6/r275/parameters_segmentation.json`
  - `data/v6/4-6/r275/parameters_preprocessing.json`

## Command

`./venv/bin/python bo/v6/run_agents_parameterized_smoke_v1.py --tunnel-id 4-6 --ring-id 275`

## Requested pass/fail checks

- `depth_map.npy` exists: **PASS**
- `pixel_to_point.pkl` exists: **PASS**
- `all_segments.csv` exists: **PASS**
- `boundaries_per_ring.json` exists: **PASS**
- `final.csv` exists: **PASS**
- Evaluation metrics file exists (`evaluation/performance.md`): **PASS**

## Metrics (from evaluation report)

- OA: `0.477`
- F1 macro: `0.310`
- mIoU: `0.228`

## Evidence files

- `logs/v6_agents_parameterized_smoke_v1/smoke_summary.json`
- `logs/v6_agents_parameterized_smoke_v1/4-6/r275/logs/stage2_detection.log`
- `logs/v6_agents_parameterized_smoke_v1/4-6/r275/logs/stage3_segmentation.log`
- `logs/v6_agents_parameterized_smoke_v1/4-6/r275/logs/stage4_evaluation.log`

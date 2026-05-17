# LLM-Free Runtime Audit (V6 Smoke)

Date: 2026-05-17  
Scope: paper-facing runtime path + smoke runner

## Runtime classification

- Runtime entrypoints:
  - `agents/1_preprocessing/1_preprocessing.py`
  - `agents/2_detection/2_detection.py`
  - `agents/3_segmentation/segmentation.py`
  - `agents/evaluation.py`
  - `bo/v6/run_agents_parameterized_smoke_v1.py`
- Docs:
  - `agents/1_preprocessing/knowledge.md` (rewritten as deterministic QA/retuning note)
- Historical/supporting utilities (non-entrypoint):
  - `agents/1_preprocessing/scripts/*`
  - `agents/2_detection/scripts/*`
  - `agents/3_segmentation/scripts/*`

## Keyword scan (strict)

Searched runtime entrypoints for:

- `LLM`, `prompt`, `reflection`, `reflect`, `openai`, `anthropic`, `chat`, `completion`, `gpt`, `claude`

Result:

- No matches for any LLM/prompt/reflection/API-provider keywords in runtime entrypoints.
- `completion` appears only inside internal status strings such as `segment_completion_failed` in detection/segmentation code, not model-completion logic.

## Parameterization evidence

- `agents/1_preprocessing/1_preprocessing.py` loads `parameters_preprocessing.json`, accepts `--data-dir`, and resolves per-ring JSON via `load_parameters(...)`.
- `agents/2_detection/2_detection.py` loads `parameters_detection.json` and preprocessing JSON, accepts `--data-dir`, and supports `INTRINSIC_PARAMS_BASE_DIR_ONLY=1` for explicit sandbox resolution.
- `agents/3_segmentation/segmentation.py` loads `parameters_segmentation.json` (per-ring/default) and accepts `--data-dir`.
- `agents/evaluation.py` loads detection parameters from JSON and accepts `--data-dir`.
- `bo/v6/run_agents_parameterized_smoke_v1.py`:
  - reads verified inputs from `data/v6/<tunnel>/r<ring>/`,
  - copies explicit detection/segmentation parameter JSON from `logs/v6_deterministic_baseline_v1/<tunnel>/r<ring>/`,
  - runs all CLIs with `--data-dir logs/v6_agents_parameterized_smoke_v1`,
  - sets `INTRINSIC_PARAMS_BASE_DIR_ONLY=1` while executing downstream stages.

## Hidden dependency checks

- No runtime references to `knowledge.md` in `agents/`.
- No runtime dependency on prompt templates, chat logs, or external model APIs in audited entrypoints.
- Smoke run evidence exists under:
  - `logs/v6_agents_parameterized_smoke_v1/smoke_summary.json`
  - `logs/v6_agents_parameterized_smoke_v1/4-6/r275/single_instance_validation.md`

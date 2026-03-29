# Memory ablation: paths, parameter generation, E2E, validation

Concise runbook for the **memory-only** analyst ablation (`-m`): raw characteristics only, no CoT/knowledge in prompts. **Every full E2E run must use an explicit output mode** (flag or pre-exported `R4TUN_PIPELINE_OUT_PREFIX`); do not rely on an undocumented default.

## 1. Canonical paths and consistency

**`configurable/` is configuration only** — parameter JSON the stages read, `configurable_*.py` / `evaluation.py` entrypoints, sample defaults, and ablation prompt/archive trees (`configurable/ablation/...`). It does **not** store pipeline run outputs. **All E2E artefacts** (CSVs, NPYs, PNGs, `final.csv`, evaluation dirs) live under **`data/`** via `R4TUN_PIPELINE_OUT_PREFIX` (typically `data/ablation/<condition>`). Do **not** point that prefix at anything under `configurable/`.

| Role | Path |
|------|------|
| Subset point clouds (canonical input) | `data/subsets/<tunnel_id>.txt` |
| Reference sample characteristics | `data/sample/characteristics/raw_characteristics.json` |
| Per-tunnel raw (memory ablation) | `data/ablation/memory/<tunnel_id>/characteristics/raw_characteristics.json` |
| **Executable parameters (only path the pipeline reads)** | `configurable/<tunnel_id>/parameters_{unfolding,denoising,enhancing,detecting,sam}.json` — `configurable_unfolding.py` et al. open **these** filenames only |
| **LLM reference + inference workspace** | `configurable/ablation/memory/parameters/<tunnel_id>/` — **Reference for inference:** `parameters_*.json` (starting JSON the model must stay shape-compatible with) and `parameters_*.md` (exported prompt context from [`skills/scripts/export_llm_parameter_context.py`](../../../skills/scripts/export_llm_parameter_context.py)). **Default** when this folder has no `parameters_<stage>.json`: use [`configurable/sample/parameters_*.json`](../../sample/). **Outputs:** `parameters_*_m_opus4.6.json` (paste model JSON here). Wiring for loading reference JSON + strict output contract: [`memory_ablation_context.py`](agents/memory_ablation_context.py) (used by memory ablation **analyst** code paths, not by `run_agents.sh`). |
| Sample parameter baseline | `configurable/sample/parameters_*.json` — must match `expected_keys` / schemas in each stage script under `configurable/` (e.g. [`configurable_unfolding.py`](../../configurable_unfolding.py) `expected_keys`) |

### What [`memory_ablation_context.py`](agents/memory_ablation_context.py) is for

It is **not** used by the configurable pipeline (`configurable_unfolding.py`, `run_agents.sh`, etc.). It supports **memory ablation analyst agents** (under `configurable/ablation/memory/agents/`) when building LLM prompts:

- **`load_raw_characteristics_pair`** — loads pretty-printed sample + tunnel `raw_characteristics.json` (tunnel path via `tunnel_characteristics_dir`).
- **`load_stage_parameters_pretty`** — loads the **reference** `parameters_<stage>.json` for prompts: from `configurable/ablation/memory/parameters/<tunnel_id>/` if present, else **`configurable/sample/`** (same rule as the table above).
- **`parameter_json_schema_contract_table` / `strict_output_instructions`** — build the leaf-path type table and the “single `json` fence, same keys as reference” contract so model output can be saved as `parameters_*_m_opus4.6.json` / archive JSON without schema drift.
- **`pipeline_tunnel_data_dir`** — resolves where pipeline artefacts live for prompt text when `R4TUN_PIPELINE_OUT_PREFIX` is set (defaults to `data/ablation/memory` for IDE runs).

### Archive vs executable (mandatory before E2E)

The two directories are **not** interchangeable:

- **`configurable/ablation/memory/parameters/<id>/`** — LLM workspace: `parameters_*.md` is the human/model-facing context; `parameters_*.json` there is the **reference** JSON for that stage (fallback: `configurable/sample/`). Inference results go to `parameters_*_m_opus4.6.json`.
- **`configurable/<id>/`** — **only** these `parameters_*.json` files are loaded at runtime.

**Required before every E2E run:** overwrite **all five** executables from the current inference files (`*_m_opus4.6.json`). Do not start `./run_agents.sh` until this step is done — there is no supported workflow that runs the pipeline on stale `configurable/<id>/` JSON without re-copying from the archive after inference.

| Inference (source) | Executable (destination, overwritten) |
|--------------------|----------------------------------------|
| `.../parameters/<id>/parameters_unfolding_m_opus4.6.json` | `configurable/<id>/parameters_unfolding.json` |
| `.../parameters/<id>/parameters_denoising_m_opus4.6.json` | `configurable/<id>/parameters_denoising.json` |
| `.../parameters/<id>/parameters_enhancing_m_opus4.6.json` | `configurable/<id>/parameters_enhancing.json` |
| `.../parameters/<id>/parameters_detecting_m_opus4.6.json` | `configurable/<id>/parameters_detecting.json` |
| `.../parameters/<id>/parameters_sam_m_opus4.6.json` | `configurable/<id>/parameters_sam.json` |

Example (repo root, `TID=1-1`):

```bash
TID=1-1
A="configurable/ablation/memory/parameters/${TID}"
C="configurable/${TID}"
cp "${A}/parameters_unfolding_m_opus4.6.json"   "${C}/parameters_unfolding.json"
cp "${A}/parameters_denoising_m_opus4.6.json"   "${C}/parameters_denoising.json"
cp "${A}/parameters_enhancing_m_opus4.6.json"   "${C}/parameters_enhancing.json"
cp "${A}/parameters_detecting_m_opus4.6.json"   "${C}/parameters_detecting.json"
cp "${A}/parameters_sam_m_opus4.6.json"         "${C}/parameters_sam.json"
```

**Verify** before E2E (every stage must print `OK`; do not run `./run_agents.sh` until `bad=0`):

```bash
TID=1-1
bad=0
for s in unfolding denoising enhancing detecting sam; do
  if cmp -s "configurable/ablation/memory/parameters/${TID}/parameters_${s}_m_opus4.6.json" \
         "configurable/${TID}/parameters_${s}.json"; then
    echo "${s}: OK"
  else
    echo "${s}: MISMATCH — fix copy before E2E"
    bad=1
  fi
done
exit "$bad"   # in a script, or check bad=0 before starting the pipeline manually
```

**Pre-E2E sync for every tunnel:** copy inference → executable with [`skills/scripts/sync_inference_to_executable.py`](../../../skills/scripts/sync_inference_to_executable.py) (works for **all** layout families — any `<id>` with the five `*_m_opus4.6.json` files under `parameters/<id>/`), e.g. `./venv/bin/python skills/scripts/sync_inference_to_executable.py --tunnel-id 4-1` or `--all` before a batch. The manual `cp` / `cmp` blocks above are equivalent.

### E2E output mode → artefact prefix

Artefacts (CSVs, NPYs, PNGs, `final.csv`, evaluation dirs) live under `{prefix}/{tunnel_id}/` with **`prefix` under `data/`** (not under `configurable/`). Set it **either** with a flag **or** by exporting `R4TUN_PIPELINE_OUT_PREFIX` before `run_agents.sh`.

| Mode | How | `R4TUN_PIPELINE_OUT_PREFIX` |
|------|-----|-----------------------------|
| Memory ablation | `./run_agents.sh <id> --memory-ablation ...` | `data/ablation/memory` |
| Sam4tun / baseline ablation slot | `./run_agents.sh <id> --sam4tun-ablation ...` | `data/ablation/sam4tun` |
| Custom | `export R4TUN_PIPELINE_OUT_PREFIX=<path>` then `./run_agents.sh <id> ...` | e.g. `data/ablation/my_condition` — must stay under **`data/`** |

[`configurable/pipeline_data.py`](../../pipeline_data.py) and [`sam4tun/plugins/paths.py`](../../../sam4tun/plugins/paths.py) (`tunnel_pipeline_dir`) use the **same** env var so plugin characterisers and the configurable pipeline agree on where `unwrapped.csv` and friends live.

**Direct Python** (e.g. `configurable_unfolding.py`, characteriser plugins): export `R4TUN_PIPELINE_OUT_PREFIX` first; there is no silent `data/<id>/` fallback.

### Scripts (actual paths)

- Export LLM parameter bundles: `./venv/bin/python skills/scripts/export_llm_parameter_context.py <tunnel_id>` (repo root; script adds repo root to `sys.path`; requires existing `configurable/ablation/memory/parameters/<tunnel_id>/`).
- **Inference → executable (all tunnels, mandatory before E2E):** `./venv/bin/python skills/scripts/sync_inference_to_executable.py --tunnel-id <id>` or `--all` (every archive subdir that has all five `*_m_opus4.6.json`). Optional check: `--verify-only` (no copy; exit 1 on mismatch).

## 2. End-to-end procedure

1. **Raw characteristics** for all subset tunnels: `./venv/bin/python methods/plans/steps/run_raw_characteristics_ablation.py` (writes `data/ablation/memory/<stem>/characteristics/raw_characteristics.json`).
2. **Bootstrap archive (if needed):** for a new tunnel, copy `configurable/sample/parameters_*.json` → `configurable/ablation/memory/parameters/<id>/` only. That gives the memory-ablation **reference** JSON on disk (what analysts/LLM use alongside `parameters_*.md`).
3. **Export LLM context**: `skills/scripts/export_llm_parameter_context.py <id>` for each tunnel.
4. **LLM inference**: produce `parameters_*_m_opus4.6.json` per stage under `configurable/ablation/memory/parameters/<id>/`.
5. **Copy inference → executable (mandatory)**: `./venv/bin/python skills/scripts/sync_inference_to_executable.py --tunnel-id <id>` (or `--all`), or the manual `cp` / `cmp` gate in **Archive vs executable**. Applies to **every** tunnel before E2E, regardless of layout family.
6. **Pre-flight**: confirm `configurable/<id>/parameters_unfolding.json` contains every key in `expected_keys` in [`configurable_unfolding.py`](../../configurable_unfolding.py) (and analogous checks for other stages if you add them).
7. **E2E**: `./run_agents.sh <id> --memory-ablation --schema auto` (or `--sam4tun-ablation` for the baseline tree). Stages 1–5 run `configurable/configurable_*.py`; step 6 runs **[`configurable/evaluation.py`](../../evaluation.py)** (not the analyst copy under `ablation/memory/agents/`). Success: log shows pipeline finished; under the chosen prefix, expect `final.csv`, `only_label.csv`, and `evaluation/` (schema in `performance.md`; `--schema both` → `performance_6.md` / `performance_7.md`). Evaluation runs only if `only_label.csv` exists under `${R4TUN_PIPELINE_OUT_PREFIX}/${id}/`.
8. **Drift–adaptation validation**: `./venv/bin/python methods/plans/scripts/validate_memory_ablation_adaptation.py --tunnel-ids 1-4 4-1` (space-separated), or `--tunnel-ids-file path.txt`, or `--discover` (all ids under `data/ablation/memory/` with `raw_characteristics.json`). See script `--help`.

### Batch reruns

```bash
LOG="logs/run_agents_memory_ablation_batch_$(date +%Y%m%d_%H%M%S).log"
for id in 1-4 4-1; do
  ./venv/bin/python skills/scripts/sync_inference_to_executable.py --tunnel-id "$id" || exit 1
  ./run_agents.sh "$id" --memory-ablation --schema auto || exit 1
done 2>&1 | tee "$LOG"
```

Adjust the ID list for **any** tunnels (3-x, 4-x, 5-x, …). Each iteration syncs inference → executable for that id, verifies, then runs E2E. To refresh executables for **every** archive that has complete inference files in one go: `./venv/bin/python skills/scripts/sync_inference_to_executable.py --all` before the loop (then optionally `--verify-only --all`). Keep `--memory-ablation` so outputs stay under `data/ablation/memory/<id>/`.

### Copying another ablation run into memory

`--save-ablation-memory` triggers an end-of-run rsync **only when `--memory-ablation` is not set** (`run_agents.sh`: copy runs if `SAVE_ABLATION_MEMORY=1` and `MEMORY_ABLATION!=1`). It rsyncs **from** `${R4TUN_PIPELINE_OUT_PREFIX}/${tunnel_id}/` **to** `data/ablation/memory/<tunnel_id>/`. Use with e.g. `--sam4tun-ablation` or a custom `R4TUN_PIPELINE_OUT_PREFIX` when you want a duplicate under the memory tree; with `--memory-ablation`, outputs already land there so the extra copy is skipped.

## 3. Post–E2E validation (drift vs adaptation)

[`methods/plans/scripts/validate_memory_ablation_adaptation.py`](../../../methods/plans/scripts/validate_memory_ablation_adaptation.py) compares:

- Tunnel `raw_characteristics.json` vs sample → **drift score** (numeric geometry / density / count fields).
- `configurable/<id>/parameters_*.json` vs `configurable/sample/` → per-stage equality / leaf diffs.

**FAIL (exit 1):** drift above threshold **and** all five stage JSONs match sample (no adaptation when characteristics differ).

**WARN:** high drift but only trivial numeric noise in diffs (reported on stderr).

**PASS:** drift below threshold **or** at least one stage differs from sample beyond trivial epsilon.

Run after each batch; `tee` the report into `logs/` if you want an audit trail.

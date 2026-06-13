# Key characteristics for BO / observation space (m_s_k)

## Discovery methodology

### Goal

Identify which fields from the five per-stage characteristic JSONs (raw, unfolded, denoised, enhanced, detected) actually drive LLM parameter adaptation, so that the Bayesian Optimisation observation space is compact yet complete.

### Step 1 -- Quantitative (Spearman rank correlation)

**Script:** `skills/scripts/correlate_characteristics_params.py`

For each pipeline stage, every numeric leaf in the characteristics JSONs visible to that stage was paired with every numeric output parameter across 30 tunnels. Spearman's rho was computed for each (characteristic, parameter) pair; fields with |rho| >= 0.5 and p < 0.05 were flagged as significant.

- **Primary run:** `--chars-root data/ablation_gpt/memory+state+knowledge --models gpt5.4` -- this data root has **30/30 tunnel coverage** for all five characteristic files (raw + 4 state), giving N=30 for every field.
- **Validation run:** `--chars-root data/ablation_anthropic/memory+state+knowledge --models opus4.6` -- partial state coverage (5-22/30 depending on file) but confirms that the same raw fields rank similarly.

**Data note:** The initial analysis used only `data/ablation_anthropic/memory+state+knowledge`, which had just 5/30 unfolded and 8/30 denoised characteristics (pipeline stages were skipped when parameters matched across models). Switching to the GPT data root resolved the coverage gap.

### Step 2 -- Qualitative (text-mining LLM reasoning traces)

**Script:** `skills/scripts/audit_characteristics_usage.py`

LLM inference was run on tunnels 1-1 and 4-1 with three models (Claude Opus 4.6, GPT-5.4, Gemini 3 Flash) under the `memory+state+knowledge` condition. Each model produced 5 reasoning traces per tunnel (one per stage), saved as `{stage}_reasoning_{model}.md` under `data/ablation/memory+state+knowledge/{tunnel_id}/analysis/`.

The audit script flattens characteristics JSONs and checks whether each leaf key name (or its numeric value) appears in the corresponding stage's reasoning text. A hit-rate matrix (field x stage x model x tunnel) was aggregated:

- **ALWAYS** = referenced in every audited trace where the field was in scope.
- **SOMETIMES** = referenced in some but not all.
- **NEVER** = not found in any trace.

### Step 3 -- Cross-reference and curation

Fields were classified into three tiers:

| Tier | Criterion | Interpretation |
|------|-----------|----------------|
| **KEY** | \|rho\| >= 0.8 **and** text hit rate >= 30% | Strongest evidence: statistically drives parameters *and* LLMs explicitly use it. |
| **CORR-only** | \|rho\| >= 0.85, text hit rate < 30% | Strong statistical signal; LLMs may use it implicitly or via derived quantities. |
| **TEXT-only** | Text hit rate >= 50%, \|rho\| < 0.8 | LLMs consistently reference it (qualitative reasoning) even without strong linear correlation. |

Fields that fell below all three thresholds were dropped.
`total_points` (raw) was dropped: \|rho\| ~ 0.55 and only 6/30 text hits -- redundant with NND and geometry fields.

### Result

38 fields total, growing cumulatively per stage (8 -> 17 -> 26 -> 33 -> 38). Full list with per-field evidence is in the tables below.

---

JSON paths are relative to each `*_characteristics.json` root (use dotted paths into the file).

Curated from **Spearman rank correlation** (N=30 tunnels) plus **text-mining** of reasoning traces (opus / gpt / gemini on tunnels 1-1 and 4-1).

**How to read “why”:** *Corr* = strong association across tunnels between that number and chosen parameters; *Text* = fraction of (stage × model × tunnel) traces where the field name or its numeric value appeared; *Role* = what the pipeline stage uses that quantity for.

---

## Raw (`raw_characteristics.json`) — 8 fields

| Path | Why it’s key |
|------|----------------|
| `tunnel_geometry.estimated_diameter` (same idea as `tunnel_geometry.diameter_estimation.estimated_diameter`) | **Corr:** \|ρ\| ≈ 0.91 vs. unfolding/denoising/enhancing/detecting/SAM numeric params (diameter, spacing, filters). **Role:** Sets physical scale for the whole pipeline. |
| `point_density.mean_nearest_neighbor_distance` | **Corr:** \|ρ\| ≈ 0.87 vs. `vertical_filter_window`, `slice_spacing_factor`, and downstream spacing-like params. **Role:** Global sampling density before any stage-specific cleaning. |
| `point_density.median_nearest_neighbor_distance` | **Corr:** \|ρ\| ≈ 0.86 (similar to mean NND). **Text:** High hit rate (e.g. 24/30 stage–model–tunnel checks in the audit). **Role:** Robust density summary (less tail-sensitive than mean). |
| `tunnel_geometry.dimensions.tunnel_length` | **Corr:** \|ρ\| ≈ 0.86 with `slice_spacing_factor`, `delta`, and extent-related params. **Role:** Along-tunnel extent drives slice count and step sizes. |
| `tunnel_geometry.dimensions.tunnel_height` | **Corr:** \|ρ\| ≈ 0.87–0.88 with `vertical_filter_window` and geometry-related params. **Role:** Vertical extent of the scan. |
| `basic_statistics.coordinate_ranges.z_range[0]` | **Corr:** \|ρ\| ≈ 0.86 with unfolding spacing and filter params. **Role:** Bounding the tunnel in Z (world frame). |
| `basic_statistics.coordinate_ranges.z_range[1]` | **Corr:** \|ρ\| ≈ 0.87 (often moves with z_min as a pair). **Role:** Same as z_min; together define vertical span. |
| `point_density.min_nearest_neighbor_distance` | **Text:** Referenced in **all** audited traces where the field was visible (strong name/value overlap). **Corr:** Weaker than mean/median NND (\|ρ\| ≈ 0.69) but **LLMs consistently cite it** when arguing “sparse vs dense”. **Role:** Outlier/sparse-tail cue for denoising conservatism. |

**Dropped example:** `total_points` — modest \|ρ\| and low text hits vs. other raw fields; redundant with NND/geometry for the chosen BO observation space.

---

## Unfolded (`unfolded_characteristics.json`) — 9 fields

| Path | Why it’s key |
|------|----------------|
| `cylindrical_coordinates.r_percentiles.p10` | **Corr:** \|ρ\| = **1.0** vs. `mask_r_low` (N=30). **Role:** Direct statistic for the inner radial bound of the depth-map cylinder mask. |
| `cylindrical_coordinates.r_percentiles.p99` | **Corr:** \|ρ\| = **1.0** vs. `mask_r_high` (and very high vs. related denoising scalars). **Role:** Outer radial bound of the mask; ties tightly to depth-map generation. |
| `cylindrical_coordinates.h_span` | **Corr:** \|ρ\| ≈ 0.97 vs. denoising mask and step parameters. **Role:** Axial span in unfolded (h) coordinates — scales cutoff and mask extent along the tunnel. |
| `cylindrical_coordinates.theta_span` **or** `cylindrical_coordinates.theta_coverage_degrees` | **Corr:** \|ρ\| ≈ 0.88 vs. many params through denoising→SAM (angular coverage of the unwrapped view). **Role:** How much of the ring is present in the depth map; redundant with each other — keep **one** for BO to avoid double-counting. |
| `point_density.median_nn_distance` | **Corr:** \|ρ\| ≈ 0.82 unfolded-state vs. downstream params. **Text:** Frequent in traces when discussing unfolded sampling. **Role:** Local spacing after unfolding (feeds enhancing upsampling targets). |
| `point_density.std_nn_distance` | **Corr:** \|ρ\| ≈ 0.86 unfolded-state vs. default_cutoff_z, mask bounds, n_segment_end, etc. **Role:** Heterogeneity of spacing (irregular scans). |
| `intensity_analysis.median` | **Text:** ~63% of visible stage–model–tunnel checks (unfolded-visible stages). **Corr:** Weaker linear signal alone. **Role:** Depth-map intensity baseline; LLMs use it when adjusting `y_step` / contrast-related choices. |
| `intensity_analysis.min` | **Text:** ~46% hit rate. **Role:** Dark-floor / noise floor in the unfolded intensity image. |
| `cylindrical_coordinates.theta_range[0]` | **Text:** Referenced whenever that stage’s trace was scored (e.g. 24/24 in the audit grid for visible cells). **Role:** Angular bound (one edge of the θ domain used for the map). |

---

## Denoised (`denoised_characteristics.json`) — 9 fields

| Path | Why it’s key |
|------|----------------|
| `point_density_analysis.median_nn_distance` | **Corr:** \|ρ\| ≈ 0.88 vs. enhancing `n_segment_end`, upsampling target distances, detecting geometry. **Text:** Moderate hits. **Role:** Post-filter local spacing — what enhancing “sees”. |
| `point_density_analysis.std_nn_distance` | **Corr:** \|ρ\| ≈ 0.86 (same cluster as median NND). **Role:** Variability after denoising; informs how aggressive upsampling can be. |
| `point_density_analysis.mean_nn_distance` | **Corr:** \|ρ\| ≈ 0.89 vs. detecting/enhancing/SAM-related scalars. **Role:** Average spacing on the cleaned cloud. |
| `geometry_characteristics.estimated_diameter` | **Corr:** \|ρ\| ≈ 0.88 vs. downstream segment and angle-range families. **Role:** Cylinder geometry after noise removal (closer to “true” tunnel cross-section for line detection). |
| `geometry_characteristics.tunnel_length` | **Corr:** \|ρ\| ≈ 0.88 with segment-end and upsampling-stage distances. **Role:** Extent of valid structure along the tunnel after denoising. |
| `denoising_summary.surface_completeness` | **Corr:** \|ρ\| ≈ 0.86 vs. inter_radius, n_segment_end, upsampling targets. **Role:** How much surface survived masking — gates enhancement strength. |
| `geometry_characteristics.surface_regularity` | **Text:** ~72% when denoised chars were in scope. **Corr:** Weaker linear correlation. **Role:** Qualitative smoothness; LLMs lean on it when arguing stable vs. aggressive parameters. |
| `geometry_characteristics.average_curvature_estimate` | **Text:** ~50% hit rate. **Corr:** Moderate (\|ρ\| ≈ 0.57) but links to **detecting** Hough thresholds in the full correlation matrix. **Role:** Bends in the ring — affects line/oblique detection difficulty. |
| `geometry_characteristics.section_curvatures` | **Text:** ~56% hit rate; **Corr:** \|ρ\| up to ~0.66 with some detecting params. **Role:** Per-section curvature vector — local deviation from straight rings; use as vector or summaries (mean/max). |

---

## Enhanced (`enhanced_characteristics.json`) — 7 fields

| Path | Why it’s key |
|------|----------------|
| `upsampling_quality.coverage_uniformity` | **Corr:** \|ρ\| ≈ 0.85 vs. detecting angle ranges and SAM-related processing params. **Text:** **Every** audited trace where enhanced chars applied (12/12). **Role:** Evenness of upsampling — LLMs treat it as a go/no-go for aggressive detection/SAM. |
| `enhanced_density.final_nn_distances.median` | **Corr:** \|ρ\| ≈ 0.88 vs. `angle_range_oblique_*` and related detecting params. **Role:** Achieved target spacing after Algorithm 3 — directly tied to line spacing and template pitch. |
| `enhanced_density.final_nn_distances.mean` | **Corr:** \|ρ\| ≈ 0.88 (cluster with median). **Role:** Average post-enhancement spacing (complements median under skewed distributions). |
| `segmentation_readiness.template_spacing_suitability` | **Corr:** \|ρ\| ≈ 0.88 vs. same detecting/SAM cluster as final NN stats. **Role:** Engineered readiness score for “is spacing SAM/detect friendly?”. |
| `enhanced_density.total_points_after` | **Corr:** \|ρ\| ≈ 0.86 vs. detecting and SAM scalars. **Role:** Scale of the enhanced cloud (memory + local density for prompts). |
| `segmentation_readiness.current_median_spacing` | **Corr:** \|ρ\| ≈ 0.88 with detecting angle ranges. **Role:** Spacing the pipeline believes it achieved before detection — aligns with Hough/merge logic. |
| `segmentation_readiness.reference_template_spacing_m` | **Text:** ~50% hit rate. **Role:** Explicit link to nominal ring/template spacing (LLMs quote it when setting detection/SAM spacing). |

---

## Detected (`detected_characteristics.json`) — 5 fields

| Path | Why it’s key |
|------|----------------|
| `prompt_distribution.sam_template_distribution.prompt_density` | **Corr:** \|ρ\| ≈ 0.83 vs. `K_height`, `AB_height`, `processing.crop_margin`, and many `prompt_points.*` numerics. **Text:** **Always** referenced in audited SAM-stage traces. **Role:** How crowded the SAM prompt layout is — primary driver of template and crop geometry. |
| `prompt_effectiveness.prompt_to_target_ratio` | **Text:** **Always** in audited SAM traces. **Role:** Prompts per target point — scarcity signal for segment_width / per-ring settings. |
| `prompt_effectiveness.sam_coverage_analysis.estimated_template_coverage` | **Text:** ~50% hit rate. **Role:** Fraction of tunnel covered by templates — LLMs use it for padding/crop/margin decisions. |
| `prompt_distribution.prompt_spacing_analysis.potential_template_overlap` | **Text:** ~67% hit rate. **Corr:** Moderate links to SAM layout params. **Role:** Risk of overlapping masks — pushes conservative spacing and crop margins. |
| `prompt_distribution.sam_template_distribution.coverage_area` | **Corr:** \|ρ\| ≈ 0.83 vs. the same SAM geometry cluster as `prompt_density`. **Role:** Absolute spatial extent of template coverage (pairs with density). |

---

## Per-stage observation counts (cumulative)

| Stage | Cumulative count |
|------|------------------|
| Unfolding | 8 (raw only) |
| Denoising | 17 |
| Enhancing | 26 |
| Detecting | 33 |
| SAM | 38 |

---

## Related scripts

- Correlation: `skills/scripts/correlate_characteristics_params.py` (e.g. `--chars-root data/ablation_gpt/memory+state+knowledge --models gpt5.4`)
- Text audit: `skills/scripts/audit_characteristics_usage.py` (multi-tunnel, `--models`, `--chars-root`)

---

## E2E validation — key-only characterisers vs segmentation (2026-04-11)

### Objective

Check whether **replacing full** `*_characteristics.json` **with BO key-only** outputs (see `bo/characteriser/` and `skills/key_characteristics_observation_space.md`) **changes final segmentation quality** when stage parameters are already fixed.

### Experiment

| Item | Detail |
|------|--------|
| Tunnel | `1-1` |
| Ablation / params | `memory+state+knowledge`, model tag `opus4.6` (existing `agents/ablation/memory+state+knowledge/parameters/1-1/parameters_*_m_s_k_opus4.6.json`) |
| Characteristics | All five JSONs regenerated with `bo/characteriser/{raw,unfolded,denoised,enhanced,detected}_characteriser.py`; output under `data/ablation/memory+state+knowledge/1-1/characteristics/` |
| Env for characterisers | `PYTHONPATH=<repo root>`, `R4TUN_PIPELINE_OUT_PREFIX=data/ablation/memory+state+knowledge`, `R4TUN_ABLATION_TUNNEL_SUBROOT=memory+state+knowledge`; raw input `data/subsets/1-1.txt` |
| Pipeline | `./run_agents.sh 1-1 --ablation m_s_k --model opus4.6 --schema auto` (full six steps including SAM and `agents/evaluation.py`) |

### Metrics (6-class, auto schema)

| When | OA | F1 | mIoU |
|------|-----|-----|------|
| `evaluation/performance.md` **before** this full rerun | 0.776 | 0.731 | **0.585** |
| **After** rerun (same path, overwritten) | 0.767 | 0.707 | **0.559** |

### Key insight

The five numeric stage scripts (`agents/unfolding.py` … `sam.py`) **do not load** tunnel `*_characteristics.json`; they consume **parameter JSONs** and CSV / point-cloud artefacts. Therefore, for a **fixed-parameter** `./run_agents.sh` run, swapping fat vs key-only characteristic files **does not alter the deterministic part of the pipeline**. Any **mIoU change after a full rerun** is **not evidence** that key-only JSONs hurt quality; it is consistent with **re-executing SAM** (and GPU / implementation variability) producing a new `only_label.csv`. To isolate JSON effects one would need to **re-infer parameters** with full vs key-only context, or hold `only_label.csv` fixed and only change characteristics.

### Files touched (this run)

- `data/ablation/memory+state+knowledge/1-1/characteristics/*.json` (key-only regeneration)
- Full pipeline outputs under `data/ablation/memory+state+knowledge/1-1/` (including `only_label.csv`, `evaluation/performance.md`) — overwritten by the E2E run.

---

## Caveats

1. **Per-model state:** Unfolded/denoised JSONs depend on upstream parameters; correlation used **one model’s** chars vs. **that model’s** parameter files (`gpt5.4` for the primary N=30 table).
2. **Perfect ρ for r-percentiles:** Indicates the adapted `mask_r_low` / `mask_r_high` track those statistics almost deterministically across tunnels — strong BO signal but also near-collinear; consider constraining or using one derived feature.
3. **Text-mining** is literal (names + numbers in markdown); it undercounts paraphrases and overcounts boilerplate if numbers repeat.

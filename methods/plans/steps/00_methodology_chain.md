# Methodology chain: subset baselines → configurable ablation → paired statistics

Single overview for the **ablation study** comparing a **fixed engineer-designed `sam4tun` pipeline** (no per-tunnel adaptation) against the **`configurable` + `agents` pipeline** with **staged analyst context** (memory → memory+state → memory+state+knowledge). 

Read this before running plugins, baselines, ablation runs, evaluation, and statistical comparison.

---

## Tunnel taxonomy (from `data/subsets` filenames)

All files under `data/subsets/` are **raw point clouds** (`.txt`). Classify by **filename prefix**:

| Prefix pattern | Layout class        | Semantic segments (GT / eval schema) |
|----------------|---------------------|--------------------------------------|
| `1_*`, `2_*`   | regular | **6-class** (`agents/evaluation.py --schema 6`) |
| `3_*`          | continuous          | **6-class** |
| `4_*`, `5_*`   | complex   | **7-class** (`--schema 7`) |

Assign each file a **tunnel_id** (e.g. stem without `.txt`). Materialise working directories **`data/{tunnel_id}/`** as required by `sam4tun` and `configurable` (inputs, intermediates, `only_label.csv`, evaluation outputs).

---

## Ordered chain (high level)

1. **Characteristics (plugins)** — Run **`sam4tun/plugins`** on **`data/sample.txt`** (reference tunnel) and on **every** subset point cloud, and save JSON features under a stable tree (see below).
2. **Baseline mIoU (sam4tun)** — Run **one fixed** `sam4tun` script sequence and parameters **for all** subset tunnels (same engineer defaults; no BO, no per-tunnel retuning). Evaluate mIoU per tunnel with the correct **6 vs 7** schema. This quantifies **lack of adaptability** of the hand-tuned stack.
3. **Ablation pipeline (configurable + agents)** — For each subset tunnel, run the **full** end-to-end process **three times** with agentic adaptation (same raw point cloud each time), differing by **analyst context** (`m`, `m_s`, `m_s_k`). Each condition stores inferred parameter JSONs under its own `configurable/ablation/{condition}/parameters/{tunnel_id}/`. Run: `./run_agents.sh <tunnel_id> --ablation <code>` (or `--all`). Pipeline stages load parameters **directly** from the ablation folder — no sync/copy. Outputs persist under `data/ablation/{condition}/{tunnel_id}/`. 
4. **Evaluation** — `configurable/evaluation.py` with **`--ablation <code> --schema 6`** or **`--schema 7`** by family (or **`--schema auto`**); metrics under `data/ablation/{condition}/{tunnel_id}/evaluation/` (`performance.md`; **`--schema both`** adds `performance_6.md` / `performance_7.md` and suffixed plots).
5. **Statistics** — **Paired** comparison per subset: **mIoU_agents − mIoU_sam4tun**. Report **mean and std** of the paired differences **per layout family** separately, plus a **p-value** (paired test) for whether agents/configurable beats the fixed sam4tun baseline on that family.

---

## Step 1 — Plugins: point-cloud characteristics

**Purpose:** Same structured descriptors for the **reference sample** and **each subset**, for analysts and for traceability.

**Scripts (under `sam4tun/plugins/`):** run the characterisers appropriate to each stage output as your project defines (e.g. unfolded / denoised / enhanced / detected), consistent with existing plugin contracts.

**Raw point cloud (pre-pipeline) characteristics:** from the repo root run  
`python methods/plans/steps/run_raw_characteristics_ablation.py`  
which reads `data/sample.txt` and `data/subsets/*.txt` and writes `raw_characteristics.json` under **`data/sample/characteristics/`** for the reference sample and **`data/ablation/memory/{tunnel_id}/characteristics/`** for each subset stem (layout knob: `ABLATION_TUNNEL_SUBROOT` in `sam4tun/plugins/paths.py`).

**Outputs (convention):**

- Reference sample (universal baseline): **`data/sample/characteristics/`** (JSON artefacts per plugin; folder name **`characteristics`**, not `charateristics`).
- **`data/ablation/memory/{tunnel_id}/characteristics/`** — **raw / pre-pipeline** characteristics (e.g. `raw_characteristics.json`). **Level 1** (`-m`) **full pipeline** outputs use the **same tunnel root** **`data/ablation/memory/{tunnel_id}/`**: CSVs and other artefacts sit next to the existing `characteristics/` subfolder (so raw JSON and the memory-only run share one tree). **Levels 2–3** use **separate roots** (Step 3) with **no** duplicate raw batch there unless you copy for traceability.

If a plugin currently assumes paths like `data/{tunnel_id}/`, either run it after copying/symlinking the subset into `data/{tunnel_id}/` or adapt invocations so tunnel_id resolves correctly; the methodology requires **one clear row in the journal** listing exact commands and paths.

**Staged characterisers (unfolded → denoised → enhanced → detected)** for the **reference** tunnel only: after `data/sample/` contains `unwrapped.csv`, `denoised.csv`, `enhanced.csv`, and `detected.csv`, run from repo root with `PYTHONPATH=.` (so `sam4tun.plugins` imports resolve):

```bash
export PYTHONPATH=.
python sam4tun/plugins/1-unfolded_characteriser.py sample
python sam4tun/plugins/2-denoised_characteriser.py sample
python sam4tun/plugins/3-enhanced_characteriser.py sample
python sam4tun/plugins/4-detected_characteriser.py sample
```

All four write under **`data/sample/characteristics/`** when `tunnel_id` is **`sample`**. For a **subset** tunnel with materialised `data/{tunnel_id}/` CSVs, pass that `tunnel_id` instead; JSON then lands under **`data/ablation/memory/{tunnel_id}/characteristics/`** (unless `ABLATION_TUNNEL_SUBROOT` is changed).

---

## Step 2 — Baseline: fixed sam4tun on all subsets

- **Single** frozen configuration (scripts, checkpoints, default JSON params) for **every** subset.
- Run the full sam4tun pipeline that produces **`only_label.csv`** (or equivalent) per `data/{tunnel_id}/`.
- **Evaluate** with `agents/evaluation.py` using **6-class for `1_*`,`2_*`,`3_*`** and **7-class for `4_*`,`5_*`**.
- Log commands, git commit hash, and paths under **`data/logs/{tunnel_id}/`** (e.g. `baseline_sam4tun.json`, copied `performance.md`).

---

## Step 3 — Ablation: configurable end-to-end + analyst levels

**Goal:** For each subset **`tunnel_id`**, same **raw** input as in **`data/subsets/{tunnel_id}.txt`** (materialised under **`data/{tunnel_id}/`** for the active run), produce **three** agentic-condition output trees comparable to **`data/sample/`** (plus the shared **`sam4tun`** baseline).

### 3a — Full pipeline outputs per condition (mirror `data/sample/`)

**Ablation code ↔ parameter source ↔ output root:**

| Code | Condition | Parameter source | Output root |
|------|-----------|------------------|-------------|
| `sam4tun` | Baseline (fixed) | `configurable/ablation/sam4tun/parameters_{stage}.json` (shared) | `data/ablation/sam4tun/{tunnel_id}/` |
| `m` | Memory only | `configurable/ablation/memory/parameters/{tunnel_id}/parameters_{stage}_m_opus4.6.json` | `data/ablation/memory/{tunnel_id}/` |
| `m_s` | Memory + state | `configurable/ablation/memory+state/parameters/{tunnel_id}/parameters_{stage}_m_s.json` | `data/ablation/memory+state/{tunnel_id}/` |
| `m_s_k` | + Knowledge | `configurable/ablation/memory+state+knowledge/parameters/{tunnel_id}/parameters_{stage}_m_s_k.json` | `data/ablation/memory+state+knowledge/{tunnel_id}/` |

Each run populates the usual artefacts (`unwrapped.csv`, `denoised.csv`, `enhanced.csv`, `detected.csv`, `final.csv`, `only_label.csv`, `evaluation/`). **`data/sample/`** remains the **single universal reference**.

**Shell note:** folder names contain **`+`**; `run_agents.sh` handles quoting internally.

### 3b — Analyst context (three levels)

| Level | Name (concept) | What the analyst sees |
|-------|----------------|------------------------|
| **1** | Memory only (`m`) | Sample tunnel characteristics only |
| **2** | Memory + state (`m+s`) | Sample **+** new tunnel characteristics |
| **3** | + Knowledge (`m+s+k`) | As level 2 **+** `agents/denoising/knowledge.md` |

### 3c — Running a condition

Each condition's parameters live in their own folder under `configurable/ablation/{condition}/parameters/{tunnel_id}/`. The pipeline loads them **directly** via `configurable/pipeline_data.py::resolve_ablation_param_file()` — there is no separate "active" copy under `configurable/{tunnel_id}/`.

```bash
# Single tunnel
./run_agents.sh 1-4 --ablation m --schema auto

# All tunnels for a condition
./run_agents.sh --all --ablation m --schema auto
```

The mapping from `--ablation <code>` to parameter paths and output prefixes is defined in `configurable/pipeline_data.py::ABLATION_CONDITIONS`.

**Stopping rule:** **No BO.** Per subset, **one** final mIoU after the **`m_s_k`** condition; earlier conditions attribute gains.

---

## Step 4 — Evaluation artefacts

- Use **`configurable/evaluation.py`** with **`--ablation <code> --schema 6`** or **`--schema 7`** per family (or **`auto` / `both`**).
- For ablation runs, evaluate from each condition's **`only_label.csv`** under **`data/ablation/{condition}/{tunnel_id}/`** (the `--ablation` flag resolves the correct output directory).
- Keep **baseline**, **per-condition ablation**, and **main** `data/{tunnel_id}/` results distinguishable in **`data/logs/{tunnel_id}/`** (copies or manifests; include **condition** / suffix in filenames where helpful).

---

## Step 5 — Statistics (paired, per family)

For each **layout family** (`regularly_staggered`, `continuous`, `complex_staggered`):

1. For each subset *i* in that family, form **paired** values **(mIoU_sam4tun_i, mIoU_agents_i)** on the **same** tunnel_id and evaluation schema.
2. Difference **Δ_i = mIoU_agents_i − mIoU_sam4tun_i**.
3. Report **mean(Δ)**, **std(Δ)** (or SE), and **n**.
4. **P-value:** **paired** test on the Δ_i (e.g. **paired t-test** if normality is plausible; else **Wilcoxon signed-rank** on paired observations). State the test and significance level (e.g. α = 0.05).

Do **not** pool all subsets into one global p-value without a clear hierarchical plan; **primary reporting is per family** as above.

---

## Data flow (mermaid)

```mermaid
flowchart TB
  subgraph plugins [Step 1 Plugins]
    P[sam4tun/plugins]
    S[data/sample/characteristics]
    U[data/ablation/memory/tunnel_id/characteristics]
    P --> S
    P --> U
  end
  subgraph base [Step 2 Baseline]
    B[Fixed sam4tun all subsets]
    E1[configurable/evaluation.py]
    B --> E1
  end
  subgraph abl [Step 3 Ablation]
    C[configurable full chain]
    PA[configurable/ablation/condition/parameters/tunnel_id]
    D1[data/ablation/memory/tunnel_id]
    D2[data/ablation/memory+state/tunnel_id]
    D3[data/ablation/memory+state+knowledge/tunnel_id]
    PA -->|"--ablation code"| C
    C --> D1
    C --> D2
    C --> D3
  end
  subgraph stats [Step 5 Stats]
    T[Paired delta mIoU per family]
    T --> M[mean std p-value]
  end
  E1 --> T
  D3 --> E2[configurable/evaluation.py]
  E2 --> T
```

---

## Dependencies and scope

- **GT** is used **only** for **mIoU** evaluation and statistics (design-time / study metric), not for runtime pipeline decisions in this chain.
- Steps **01–05** in `methods/plans/steps/` may still inform *documentation* of assumptions; **this chain replaces BO-centric steps 06–09** for the ablation study. Do not require BO, proxy, or Platt calibration for this workflow.
- **Reproducibility:** every run leaves a trace under **`data/logs/{tunnel_id}/`** (commands, commit, parameter paths, evaluation paths).

---

## Files and entrypoints (reference)

| Piece | Location |
|-------|----------|
| Subset point clouds | `data/subsets/*.txt` |
| Reference sample | `data/sample.txt` |
| Raw / pre-pipeline characteristics | `data/sample/characteristics/` · `data/ablation/memory/{tunnel_id}/characteristics/` only |
| Full pipeline outputs (ablation) | `data/ablation/{condition}/{tunnel_id}/` (mirror `data/sample/`) |
| Ablation condition registry | `configurable/pipeline_data.py::ABLATION_CONDITIONS` |
| Parameter resolution | `configurable/pipeline_data.py::resolve_ablation_param_file()` |
| Configurable stages | `configurable/configurable_unfolding.py`, `configurable_denoising.py`, `configurable_enhancing.py`, `configurable_detecting.py`, `configurable_sam.py` |
| Shared pipeline utilities | `configurable/pipeline_data.py` (arg parsing, path resolution, output dirs) |
| E2E runner | `./run_agents.sh <tunnel_id> --ablation <code> [--schema auto]` |
| mIoU evaluation | `configurable/evaluation.py` (`--ablation <code> --schema auto` / `6` / `7` / `both`) |
| Cross-condition comparison | `skills/scripts/compare_ablation_conditions.py` |
| Denoising analyst | `agents/denoising/analyst.py` |
| Denoising knowledge (level 3+) | `agents/denoising/knowledge.md` |
| Raw characteristics batch | `methods/plans/steps/run_raw_characteristics_ablation.py` |
| Run logs | `data/logs/{tunnel_id}/` |

---

## Checklist (operator)

- [ ] Map each `data/subsets/*.txt` to **tunnel_id** and **family** (1–2 / 3 / 4–5).
- [ ] Materialise `data/{tunnel_id}/` inputs as required.
- [ ] Run plugins → reference under `data/sample/characteristics/`, each subset under `data/ablation/memory/{tunnel_id}/characteristics/`.
- [ ] Run **fixed** sam4tun baseline → evaluate → copy/summarise to `data/logs/{tunnel_id}/`.
- [ ] For each condition: `./run_agents.sh <tunnel_id|--all> --ablation <code> --schema auto`. Pipeline loads parameters directly from `configurable/ablation/{condition}/parameters/{tunnel_id}/`; outputs go to `data/ablation/{condition}/{tunnel_id}/`.
- [ ] Evaluate ablation mIoU (correct schema per family).
- [ ] Compare across conditions: `python skills/scripts/compare_ablation_conditions.py`.
- [ ] Compute **paired** Δ per subset; **per family**: mean, std, p-value; document test choice.

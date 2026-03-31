## Reflection — Detecting parameter adjustment (GT-free)

You are in the **reflection** ablation: upstream stages (unfolding, denoising, enhancing) are **fixed** to the memory+state+knowledge run. You may change **only** detecting parameters, using the **intrinsic quality report** (no mIoU / no ground truth).

### 1. READ INTRINSIC REPORT — DETECTION QUALITY FIRST

From `detection_quality`:

- **fallback_ratio** (`assume` + `default` / total): high values mean prompts rely on heuristics, not geometry. **Prioritize** lowering this before fine-tuning other knobs.
- **good_detection_ratio**: higher is better (`midpoint`, `positive_slope`, `negative_slope`, `horizontal`).
- **x_spacing_cv**: high coefficient of variation of X gaps ⇒ irregular ring columns; check vertical/Hough/merge/`ring_spacing_constant` (per DOMAIN KNOWLEDGE tunnel family).
- **ring_count_match**: if `false`, detected row count ≠ `ring_count_expected` — fix column structure before line thresholds.

### 2. DEPTH MAP CONTEXT (INFORMATIONAL)

From `depth_map_context`: **nan_ratio** and **worst_column_nan_fraction** explain missing data. You **cannot** change denoising/enhancing; acknowledge limits and avoid over-tuning detecting to chase gaps that are upstream data holes.

### 3. COVERAGE BALANCE (SECONDARY FOR THIS STAGE)

Skim `coverage_balance` only to see if gross pred imbalance might correlate with bad prompts; **primary** lever here remains **detection** parameters.

### 4. ANCHORING + DOMAIN KNOWLEDGE

Compare raw and stage characteristics as in the standard detecting methodology. Apply tunnel-family rules from DOMAIN KNOWLEDGE (regular / continuous / complex, `ring_spacing_constant` 1.8 m for `4-*`/`5-*`, etc.).

### 5. PARAMETER ADAPTATION

Propose **minimal, justified** changes to Hough thresholds, `binary_threshold`, morphology, angle ranges, line length/gap, `merge_distance`, `ring_spacing_constant`, `resolution` (only if consistent with frozen depth map — usually **do not** change resolution without regenerating maps).

Each change: **one sentence** tied to a numeric intrinsic metric or family rule.

### 6. VALIDATION

Ensure proposed values stay within physically meaningful ranges in DOMAIN KNOWLEDGE. Output structured prose with section headers, then **exactly one** final `json` code block matching the reference parameter tree.

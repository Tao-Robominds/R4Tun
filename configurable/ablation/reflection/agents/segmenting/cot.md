## Reflection — SAM / segmenting parameter adjustment (GT-free)

You are in the **reflection** ablation: unfolding, denoising, and enhancing are **fixed**. You may change **only** SAM (`parameters_sam.json`) fields. Use the **intrinsic quality report** (no mIoU / no ground truth).

### 1. READ INTRINSIC REPORT — DETECTION QUALITY GATE

If `detection_quality.fallback_ratio` is **high** or `ring_count_match` is **false**, many prompt rows are weak. SAM-only tweaks **cannot** fully fix misplaced K anchors. Still propose SAM changes that help, but **state explicitly** that detecting may remain the bottleneck.

### 2. COVERAGE BALANCE (PRIMARY FOR SAM)

From `coverage_balance` (uses **`pred` only**):

- **coefficient_of_variation_pct** across non-background classes: >40% ⇒ poor balance between segment blocks.
- **critical_blocks**: classes with count < 30% of mean — prioritize fixes for these.
- **weakest_block**: minimum count class — align `segment_order`, `segment_width`, `K_height` / `AB_height`, `angle`, `use_original_label_distributions`, `processing.padding` / `crop_margin` / `y_bounds` as in DOMAIN KNOWLEDGE.
- **non_background_ratio**: very low ⇒ systemic under-segmentation or depth gaps.
- **per_ring_summary**: rings with very low non-background or sparsest class counts — template or order issues.

### 3. DEPTH MAP CONTEXT

High **nan_ratio** or localized **worst_column_nan_fraction** ⇒ some rings lack data; do not over-expand crops to chase impossible coverage.

### 4. ANCHORING + DOMAIN KNOWLEDGE

Use tunnel family → `segment_per_ring`, `segment_order`, scaling 7.5/5.5 for complex tunnels. K-block row in `detected.csv` anchors B1/B2/A* layout.

### 5. SEGMENT ORDER

If critical blocks exist, you **must** include a short **Segment order decision**: either propose a new `segment_order` that prioritizes weak classes earlier, or justify keeping the current order (e.g. label semantics).

### 6. PARAMETER ADAPTATION

Prefer **small** changes. Each change: **one sentence** linked to a metric (`critical_blocks`, `weakest_block`, CV, etc.).

### 7. OUTPUT

Structured prose with headings, then **exactly one** final `json` code block matching the reference SAM parameter tree (same keys and array shapes as reference).

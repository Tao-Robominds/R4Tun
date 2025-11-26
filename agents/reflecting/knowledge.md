## SAM Reflecting / Evolution Knowledge

### 1. What the reflecting stage sees
- **Inputs**:
  - `data/{tunnel_id}/final.csv` – per‑point predictions with `pred` labels.
  - `data/{tunnel_id}/characteristics/algorithm4_characteristics.json` – geometry & layout metadata.
  - `data/{tunnel_id}/detected.csv` – prompt point locations & types.
  - `configurable/{tunnel_id}/parameters_sam.json` – current SAM prompting configuration.
- **Outputs**:
  - Updated `configurable/{tunnel_id}/parameters_sam.json` tuned to this tunnel.
  - A markdown analysis file explaining what changed and why.

The reflecting stage does **not** change the core algorithm; it only tunes parameters and, critically, the **segment processing order**.

### 2. Coverage metrics
- For each non‑background block (K, B1, A1, A2, A3, B2), track:
  - `count` – number of predicted points.
  - `percentage` – share of total points.
- Global statistics:
  - `average_points_per_block` – mean of non‑background counts.
  - `coefficient_of_variation` – \(100 * \sigma / \mu\) over non‑background counts.
  - `critical_threshold` – typically 30% of the average.
  - `critical_blocks` – blocks with coverage `< critical_threshold`.
  - `weakest_block` – block with minimum coverage.

Heuristics:
- If **any block is CRITICAL**, the configuration is **unbalanced** and needs targeted action.
- CV \< 20% → **excellent** balance, 20–40% → **good**, > 40% → **poor**.

### 3. Parameters you may change
Work on these knobs, starting from the **smallest impactful change**:

- **segment_order**:
  - Ordering of logical blocks, e.g. `["K", "B1", "A1", "A2", "A3", "B2"]`.
  - Earlier blocks “claim” pixels first; later blocks must fit into the leftovers.
  - For a CRITICAL block, consider moving it **earlier** so it is segmented on a cleaner canvas.
  - Constraint: `len(segment_order) == segment_per_ring`.

- **segment_width**:
  - Horizontal span of each segment crop in the depth map (in pixels or mm, depending on config convention).
  - Too small → under‑coverage, especially on large‑diameter tunnels.
  - Too large → excessive overlap between adjacent blocks.

- **K_height / AB_height / angle**:
  - Control vertical extent and skew of template masks.
  - Mis‑tuned values can clip important parts of a block (under‑coverage) or leak into neighbours (overlap).

- **use_original_label_distributions**:
  - Controls balance of positive/negative prompt points.
  - Turning this off typically **increases positives**, which can help under‑covered blocks but may blur boundaries.

- **processing block**:
  - `resolution` – usually fixed at 0.005; change only with strong justification.
  - `padding`, `crop_margin`, `y_bounds`, `mask_eps` – context and numerical stability; small adjustments can help.

### 4. Weakest‑block and CRITICAL‑block reasoning
When reasoning about coverage:
- Always **name** the weakest block and show its count vs. average.
- For each CRITICAL block, answer:
  1. Is this mainly a **prompt geometry** issue (segment_width, heights, angle)?
  2. A **prompt density / label distribution** issue?
  3. A **segment_order** / mask‑overlap issue?

Actions:
- Geometry issue → adjust `segment_width`, heights, or angle slightly.
- Prompt density issue → toggle `use_original_label_distributions` or document why it stays unchanged.
- Order issue → propose a new `segment_order` that gives the CRITICAL block earlier priority, or **explicitly justify** why the existing order must be preserved (e.g. downstream class‑ID constraints).

### 5. Required behaviour for `segment_order`
When one or more blocks are CRITICAL:
- You **must** add a dedicated section in your analysis:
  - `### Segment order decision`
  - In that section:
    - Either propose a new `segment_order` and explain why it helps the weakest block(s),
    - Or explain clearly why `segment_order` is kept unchanged (e.g. other parameters fully explain the issue, or order is tied to fixed semantics).
- The final JSON **must** include an explicit `segment_order` array, even if unchanged.

### 6. Output contract
For reflecting:
- The **analyst** produces:
  - A structured markdown analysis with:
    - Coverage summary.
    - Diagnosis per weak / CRITICAL block.
    - A `### Segment order decision` section.
    - Clear, concrete recommendations (e.g. “segment_width: 1300”).
- The **coder/evolver** then:
  - Reads the analysis and the current `parameters_sam.json`.
  - Produces a full, self‑contained updated `parameters_sam.json`.
  - Ensures all fields present in the original config still exist, with updated values where justified.



## SAM Prompting Overview

Algorithm 4‑2 consumes the detection results and `configurable/*/parameters_sam.json` to build template prompts for SAM. Every tunnel type shares the same logical layout (K block + A/B blocks), but the constants differ slightly between 5.5 m and 7.5 m tunnels. Keep the following reference handy when creating new parameter files.

## Parameter Reference (Segmenting Stage)

### Global Layout
- **segment_per_ring** – number of concrete segments per ring. Most historical datasets use **6** (T1–T3); large diameter tunnels switch to **7**.
- **segment_order** – ordered list that maps template names (K, B1, A1…) to class IDs. It must contain `segment_per_ring` entries; otherwise SAM label projection will misalign.
- **segment_width (px)** – horizontal crop width for each segment ROI in the depth map. Values around **1 200 px** cover a 1.2 m ring when the map resolution is 5 mm/px.
- **K_height / AB_height (px)** – vertical crop heights for K-blocks vs A/B blocks. Expect **1 050–1 100 px** for K blocks and **3 200–3 400 px** for the longer A/B sections. Increase these when the depth map is generated at finer resolution.
- **angle (deg)** – nominal skew between the scanner axis and the tunnel axis used to offset crops. Typically **6–8°**; adjust if the scanner is heavily tilted in the dataset.
- **use_original_label_distributions** – keep `true` to reuse the canonical class IDs. Set `false` only when experimenting with custom ordering.

### Processing Block
- **processing.resolution (m/px)** – must match the depth-map projection step. All released data uses **0.005**; changing it requires re-exporting depth maps and retraining prompt templates.
- **processing.padding (px)** – horizontal padding applied around each crop. **150–300 px** gives enough context without wasting memory.
- **processing.crop_margin (px)** – additional vertical padding; **50 px** works for most scenes.
- **processing.mask_eps** – numerical epsilon used when building template logits; keep it in the **1e‑3** range so the sigmoid is well behaved.
- **processing.y_bounds (px)** – allowable Y-range in the depth map to clamp prompt points. For 5.5 m tunnels, **[4200, 13100]** covers the entire wall; adjust proportionally if the projection resolution changes.

### Prompt Point Templates
Each entry inside `prompt_points` encodes a family of control points (in pixels) relative to the crop centre:

- **prompt_points.k_block** – radii for the inner/middle/outer rings and vertical spacing factors (`k_block_spacing`, `vertical_spacing`). Values around **700/500/350 px** and spacings of **310–730 px** align with 5.5 m tunnels; scale proportionally for larger diameters (multiply by 7.5 / 5.5).
- **prompt_points.ab_blocks** – geometry for A/B segments, including `outer_ring`, `middle_ring`, `inner_ring`, `center_ring`, `fine_spacing`, `ultra_fine`, `edge_ring`, and `edge_spacing`. Observed ranges: **325–700 px** for ring radii, **162–350 px** for fine spacings. Vertical levels (`level_1` … `level_7`) trace bolt rows between **270–1 720 px**; `special_levels` capture additional offsets used for bolt holes. Retain symmetry between positive/negative heights.
- **prompt_points.template_mask** – half-widths/heights used to rasterize the initial mask logits. Widths stay around **625 px**; heights vary per segment (e.g., **460–1 700 px**). Adjust them if the templates consistently miss the segment edges.

### Practical Ranges
- Horizontal parameters (segment_width, outer_ring, padding, edge_spacing, etc.) should scale linearly with the ring length projected into pixels. When the resolution stays at 0.005 m/px, multiplying 5.5 m physical dimensions by ~182 gives the pixel count; for 7.5 m tunnels multiply by ~240.
- Vertical parameters (K_height, AB_height, vertical_spacing arrays) are tied to the projected radius (≈ diameter / 2 translated into pixels). Expect 5.5 m tunnels to use 600–3 200 px, and 7.5 m tunnels to add ~30 %.
- Keep `segment_order`, `prompt_points.k_block`, and template masks synchronized: if you remove a segment type from `segment_order`, also remove its template configuration to avoid mismatched logits.


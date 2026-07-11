## SAM Prompting Overview

Algorithm 4-2 consumes the detection results and `agents/ablation/{condition}/parameters/<tunnel_id>/parameters_sam.json` to build template prompts for SAM. Every tunnel type shares the same logical layout (K block + A/B blocks), but the constants differ slightly between 5.5 m and 7.5 m tunnels. Keep the following reference handy when creating new parameter files.

## Parameter Reference (Segmenting Stage)

### Global Layout
- **segment_per_ring** – number of concrete segments per ring. Most historical datasets use **6** (T1–T3); large diameter tunnels switch to **7**.
- **segment_order** – ordered list that maps template names (K, B1, A1…) to class IDs. It must contain `segment_per_ring` entries; otherwise SAM label projection will misalign.
- **segment_width (px)** – horizontal crop width for each segment ROI in the depth map. Values around **1 200 px** cover a 1.2 m ring when the map resolution is 5 mm/px.
- **K_height / AB_height (px)** – vertical crop heights for K-blocks vs A/B blocks. Expect **1 050–1 100 px** for K blocks and **3 200–3 400 px** for the longer A/B sections. Increase these when the depth map is generated at finer resolution.
- **angle (deg)** – nominal skew between the scanner axis and the tunnel axis used to offset crops. Typically **6–8°**; adjust if the scanner is heavily tilted in the dataset.
- **use_original_label_distributions** – keep `true` to reuse the canonical class IDs. Set `false` only when experimenting with custom ordering.

### Processing Block
- **processing.resolution (m/px)** – must match the depth-map projection step. All released data uses **0.005**; changing it requires re-exporting depth maps and retraining prompt templates.
- **processing.padding (px)** – horizontal padding applied around each crop. **150–300 px** gives enough context without wasting memory.
- **processing.crop_margin (px)** – additional vertical padding; **50 px** works for most scenes.
- **processing.mask_eps** – numerical epsilon used when building template logits; keep it in the **1e-3** range so the sigmoid is well behaved.
- **processing.y_bounds (px)** – allowable Y-range in the depth map to clamp prompt points. For 5.5 m tunnels, **[4200, 13100]** covers the entire wall; adjust proportionally if the projection resolution changes.

### Prompt Point Templates
Each entry inside `prompt_points` encodes a family of control points (in pixels) relative to the crop centre:

- **prompt_points.k_block** – radii for the inner/middle/outer rings and vertical spacing factors (`k_block_spacing`, `vertical_spacing`). Values around **700/500/350 px** and spacings of **310–730 px** align with 5.5 m tunnels; scale proportionally for larger diameters (multiply by 7.5 / 5.5).
- **prompt_points.ab_blocks** – geometry for A/B segments, including `outer_ring`, `middle_ring`, `inner_ring`, `center_ring`, `fine_spacing`, `ultra_fine`, `edge_ring`, and `edge_spacing`. Observed ranges: **325–700 px** for ring radii, **162–350 px** for fine spacings. Vertical levels (`level_1` … `level_7`) trace bolt rows between **270–1 720 px**; `special_levels` capture additional offsets used for bolt holes. Retain symmetry between positive/negative heights.
- **prompt_points.template_mask** – half-widths/heights used to rasterize the initial mask logits. Widths stay around **625 px**; heights vary per segment (e.g., **460–1 700 px**). Adjust them if the templates consistently miss the segment edges.

### Practical Ranges
- Horizontal parameters (segment_width, outer_ring, padding, edge_spacing, etc.) should scale linearly with the ring length projected into pixels. When the resolution stays at 0.005 m/px, multiplying 5.5 m physical dimensions by ~182 gives the pixel count; for 7.5 m tunnels multiply by ~240.
- Vertical parameters (K_height, AB_height, vertical_spacing arrays) are tied to the projected radius (≈ diameter / 2 translated into pixels). Expect 5.5 m tunnels to use 600–3 200 px, and 7.5 m tunnels to add ~30 %.
- Keep `segment_order`, `prompt_points.k_block`, and template masks synchronized: if you remove a segment type from `segment_order`, also remove its template configuration to avoid mismatched logits.

---

## Tunnel family → SAM configuration

### Regular and continuous (`1-*`, `2-*`, `3-*`)
- **segment_per_ring = 6**
- **segment_order**: `["K","B1","A1","A2","A3","B2"]`
- **Geometry**: 5.5 m class — **K_height** ~1050–1100 px, **AB_height** ~3200–3400 px at **resolution 0.005** m/px (scale `prompt_points` from the 5.5 m reference).

### Complex (`4-*`, `5-*`)
- **segment_per_ring = 7**
- **segment_order**: `["K","B1","A1","A2","A3","A4","B2"]`
- **Geometry**: 7.5 m, 1.8 m rings — scale horizontal prompt radii and template widths by **7.5/5.5** vs the 5.5 m defaults; vertical spans ~**+30%** vs 5.5 m tunnels.

### K-block as anchor
Each row of `detected.csv` is the **K-block centre** for that ring: **B1** crops upward, **B2** downward, **A*** blocks fill the column between B1 and B2. **Y error in detection dominates** downstream SAM quality; **X** is already well aligned to ring columns.

### Continuous (`3-*`) — K template uniformity

| Topic | Guidance |
|-------|----------|
| Prior | One K detection → global **Y\*** + fixed K trapezoid geometry on every ring |
| Y input | All rings share identical K centre Y from detecting uniform snap; only X varies |
| Template | `prompt_points` / `K_height` scale with diameter but **not** per-ring |
| Bolt holes | Adjust `y_bounds` / `prompt_points` **once** from state evidence — not ring-by-ring |
| Tune levers | `K_height`, `angle`, `crop_margin`, `y_bounds` when K IoU low despite uniform Y |
| Oversize K mask | K orange mask taller than depth-map joints → apply **K_HEIGHT_OVERSIZE** (cot.md): reduce `K_height` toward sample/band (**1050–1100 px**); **do not** set from GT span measurement |
| `K_height` anchor | Sample reference **~1080 px**; 3-1-1 GLM prior **1137 px** is above band — step down conservatively if oversize |
| `template_mask` JSON | `prompt_points.template_mask.k_block` heights are **not consumed** by `sam.py` for K (hardcoded mm vertices in `generate_template_mask`); tuning JSON template_mask alone will not shrink K mask |
| Anti-pattern | Per-ring Y tweaks; geometric fallback (reserved for `4-*`/`5-*`); GT-derived `K_height` |
| Success | K-block IoU **> 0.65** on 3-1-1 with `Y_std < 10 px` on prompts |

See **K_TEMPLATE_UNIFORM** in segmenting cot.md.

### y_bounds for taller depth maps
7.5 m tunnels produce taller unwrapped maps — extend **processing.y_bounds** so the full wall height used by SAM crops stays inside the clamp (proportional to image height vs the 5.5 m reference).

### Geometric fallback (4-\*, 5-\*, `segment_per_ring=7`)
`sam4tun/agents/sam.py` sets `_use_geometric_fallback` when the tunnel id starts with `4-` or `5-` and `segment_per_ring==7`. In that mode **SAM ignores `prompt_points` / `template_mask`** and builds crops from detecting ring Y plus scaled geometry:
- `segment_width × (ring_length / 1.2)` (ring_length from detecting, typically 1.8 m on T4/T5)
- `K_height`, `AB_height`, `padding`, `crop_margin`, `y_bounds` × `(diameter / 5.5)`
- Quality is dominated by **detecting Y accuracy**; tune detecting before SAM prompt geometry.
- For T1/T2 (6-seg), prompt-based SAM is active — retain sample `prompt_points` unless state shows systematic misalignment.

### Rules baseline (minimum viable adaptation)
Before percentile fine-tuning on T4/T5, apply rules formulas from `agents/ablation/rules/rule_adapt.py` (diameter 7.5, mask `r±0.15`, `ring_spacing_constant=1.8`, geometric SAM scaling). Rules lifted 4-1 mIoU ~+0.12 vs static.

## Chain of Thought Instructions for Segmenting Parameter Recommendations

Follow this structured analysis process when evaluating tunnel characteristics for SAM segmenting parameter recommendations:

### 0. CONSERVATIVE DEFAULT PRINCIPLE (read first, applies to every parameter)

When uncertain whether a parameter should deviate from the SAM4Tun default,
keep the default. Only change a parameter when you have clear evidence from
the tunnel characteristics that the default would cause a specific problem.

**Physical constants** — set from the tunnel type (always justified):
- segment_per_ring: 6 (1-\*, 2-\*, 3-\*) or 7 (4-\*, 5-\*)
- segment_order: match the tunnel's actual segment layout

**Template and prompt geometry** — keep SAM4Tun defaults unless the state
context (e.g., depth map dimensions, detection output) shows concrete evidence
that the default template geometry produces poor masks. Scaling template
dimensions proportionally with diameter is justified; arbitrary changes to
spacing factors or ring counts are not.

### 1. ANCHORING
Compare key tunnel characteristics against the sample baseline:
- Enhanced point cloud density and distribution
- Ring structure and segment count requirements
- Surface geometry complexity and segmentation challenges

### 2. CLASSIFICATION
Classify the tunnel based on the comparison:
- **SIMILAR**: <25% difference in key metrics → minimal changes needed
- **HIGH-DENSITY**: Dense enhancement results → may need finer segmentation
- **COMPLEX-GEOMETRY**: Irregular surface features → may need robust settings
- **LARGE-SCALE**: Different tunnel dimensions → may need parameter scaling

### 3. PARAMETER ADAPTATION
Adapt parameters based on classification:
- **segment_per_ring**: **Physical constant** — 6 for `1-*`/`2-*`/`3-*`, 7 for `4-*`/`5-*`. Never scale with "complexity".
- **4-* / 5-* geometric fallback** (`sam4tun/agents/sam.py`): when `segment_per_ring==7`, SAM **bypasses prompt_points** and uses geometric crops from detecting Y. Tune **detecting** ring Y accuracy; scale `K_height`, `AB_height`, `padding`, `crop_margin`, `y_bounds` by **diameter/5.5**; scale `segment_width` by **ring_length/1.2**. Leave `prompt_points` / `template_mask` near sample unless state shows crop misalignment.
- **segment_width/height**: For T1/T2 retain sample values; for T4/T5 use rules scaling above.
- **angle**: Adjust only for validated scanner tilt in state.
- **ring_spacing**: 1.2 m (T1–T3) vs 1.8 m (T4/T5) via detecting `ring_spacing_constant`.

### 0b. K_TEMPLATE_UNIFORM (mandatory for `3-*` before final JSON)

Continuous tunnels: **one K joint line** → K prompt centre **Y is identical** on every ring; only X shifts per `initial_points.csv` row.

**Strategy:**
1. **Uniform Y input:** detecting snaps all rings to `Y*`; SAM must not re-tune Y per ring.
2. **Fixed K geometry:** K trapezoid (`template_mask`) and `prompt_points` offsets are design-constant — tune **once**, apply all rings.
3. **Downstream chain:** B1…B2 `map_y` derives from each ring's K Y — uniform Y stabilizes the entire column.
4. **When K IoU low despite `Y_std < 10`:** tune `K_height`, `angle`, `crop_margin`, `y_bounds` (bolt-hole band) — **not** per-ring Y.
5. **Do not** enable geometric fallback for `3-*` (complex `4-*`/`5-*` only).

Document expected uniform K Y and which SAM levers you change before the JSON fence.

#### K_HEIGHT_OVERSIZE (within `0b`, for `3-*` when K mask is vertically too tall)

**Trigger:** `Y_std < 10` on prompts, K-block IoU still below target, and/or state/visual shows the K mask **extends past depth-map joint lines** vertically.

**Forbidden:** setting `K_height` from measured GT pixel span, centroid regression, or any auto-fit from `diagnose_k_centroids.py` output.

**Allowed levers (tune once, all rings):**
1. **`K_height`** — primary; for 5.5 m T3 stay in **1050–1100 px**; if current value **> 1100** and mask oversize, step down toward sample (~1080) in **one conservative step** (~30–60 px), not a large jump.
2. **`crop_margin`** — secondary; reduce if crop window still tall after `K_height` step.
3. **`y_bounds`** — only if prompt clipping asymmetry is suspected (bolt-hole band).
4. **`angle`** — do **not** flip sign (prior pipeline crash); adjust magnitude only with validated scanner-tilt evidence.

**Keep fixed:** per-ring Y, `segment_per_ring`, `segment_order`, and the `3-*` joint K X-mirror code path in `sam.py`.

Document which lever changed and the qualitative evidence (visual oversize, param vs knowledge band) before the JSON fence.

### Parameter Guidelines:
- **Always provide EXACT numerical values** - Never use ranges like "4-8"
- **Choose the most appropriate single value** from any range you consider
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**
- **Provide clear justification** for each parameter change
- **Output flowing analysis with section headers and final JSON parameter block**

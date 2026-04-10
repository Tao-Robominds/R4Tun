## Tunnel Variations and Adaptations (T1-T5)

### Base Tunnels (T1 & T2)
- **Geometry**: 5.5 m inner diameter metro tunnels
- **Ring structure**: 1.2 m ring length with six segments per ring
- **Joint type**: Staggered joints arrangement
- **Scanning**: Leica C10 scanner, single-station acquisitions in Wuxi
- **SAM4Tun approach**: Default parameter set, base case for template and threshold development
- **Test sets**: T1-4, T2-2 among others used for validation

### Tunnel T3 Variations
- **Geometry**: 5.5 m nominal diameter (similar to T1/T2)
- **Key difference**: Continuous joints instead of staggered
- **Data formation**: Multi-station registration creating more uniform density distribution
- **SAM4Tun adaptations**:
  - **No upsampling** applied due to uniform density
  - **Global outlier threshold**: 0.01 m
  - **Detection logic**: Reuses T1/T2 approach with fallback for horizontal-segment detection failures
  - **Template customization**: Adjusted prompts for T3's specific segment dimensions and bolt-hole locations
- **Evaluation scope**: First 50 rings spanning two stations

### Large Tunnels (T4 & T5)
- **Geometry**: 7.5 m inner diameter (larger scale)
- **Ring structure**: 1.8 m ring length with seven segments per ring
- **Joint type**: Complex interleaved K-block arrangement
- **Scanning**: Leica C10 scanner in Fuzhou with offset scanner center from tunnel axis
- **SAM4Tun adaptations**:
  - **Density zone detection**: Identifies high-density zones (seven rings nearest scanner)
  - **Adaptive outlier thresholds**: 10 mm for high-density areas vs 4 mm for low-density areas
  - **Parameter consistency**: Maintains T1/T2 setup for other parameters

### Cross-Tunnel Variation Summary
The five tunnels span:
- **Two diameters**: 5.5 m vs 7.5 m
- **Two ring lengths**: 1.2 m vs 1.8 m  
- **Two segment counts**: 6 vs 7 segments per ring
- **Two joint assemblies**: Staggered vs continuous joints
- **Scanning configurations**: Single-station TLS (T1/T2/T4/T5) vs multi-station registration (T3)

### Parameter Adaptation Strategy
- **Uniform-density tunnels** (T3): Drop upsampling, adjust prompts
- **Larger-diameter tunnels** (T4/T5): Split scenes into density regions with distinct thresholds
- **Core pipeline**: Downstream processing remains largely unchanged across variations
- **Customization focus**: Pre-processing choices adapt to geometric and scanning differences

## Success Criteria
- **Line detection accuracy**: Successful identification of tunnel joint patterns
- **Prompt point generation**: Valid intersection points for SAM guidance
- **Geometric consistency**: Detected patterns match expected tunnel structure
- **Processing robustness**: Graceful handling of missing or weak line features
- **Cross-tunnel adaptability**: Parameters adjust appropriately for T1-T5 variations

## Critical Parameter Distinctions

### segment_per_ring vs ring_count
- **segment_per_ring**: Fixed property of tunnel TYPE (T1/T2 = 6, T4/T5 = 7)
- **ring_count**: Variable number of rings in the dataset (e.g., 2-2 dataset has 10 rings)
- **segment_order**: Must match segment_per_ring length, NOT ring_count

### Tunnel Type Identification
- **T1/T2-type**: 5.5m diameter → segment_per_ring = 6
- **T4/T5-type**: 7.5m diameter → segment_per_ring = 7
- **Dataset naming**: "2-2" means tunnel type 2, dataset 2 (still 6 segments per ring)

## Parameter Reference (Detecting Stage)

`agents/ablation/{condition}/parameters/<tunnel_id>/parameters_detecting.json` controls the depth-map processing that locates joint lines before prompt generation. Key parameters:

- **binary_threshold (0–255)** – grayscale cutoff for binarizing the depth map. Cleaner scans accept **115–130**; lower the value when the wall contrast is weak.
- **morphological_kernel_size (px) & dilation_iterations** – shape and repetition of the structuring element used to thicken detected cracks. Kernels of **3×3 to 5×5** with **1–2** iterations cover all tunnels; increase only if Hough detection misses lines.
- **hough_threshold_oblique / horizontal / vertical** – accumulator thresholds for `cv2.HoughLines(P)`; larger tunnels require more votes (vertical thresholds up to **700**). Start around **40–80** for oblique lines and tune per dataset.
- **minLineLength_oblique / maxLineGap_oblique (px)** – minimum segment length and allowed gaps when fitting slanted joints. 5.5 m tunnels use **100–150 px / 40–50 px**; larger tunnels can push lengths to **300+ px**.
- **angle_range_oblique_positive / negative (deg)** – allowable slope windows for slanted joints (±6–10°). Tight ranges reduce false positives; widen them if the tunnel has noticeable skew.
- **merge_distance (px)** – distance threshold when consolidating vertical lines. Small tunnels use **2–3 px**; noisy datasets may require up to **10 px**.
- **ring_spacing_constant (m)** – expected spacing between ring centers in the unwrapped map. **Regular/continuous (`1-*`, `2-*`, `3-*`)**: 1.2 m rings ≈ **1.2–1.3**. **Complex (`4-*`, `5-*`)**: physical ring length is **1.8 m** — set **`ring_spacing_constant` to 1.8** (not 1.2). Using 1.2 on complex tunnels mis-calibrates vertical-line merging and ring-column spacing vs `ring_count`. This is a **physical constant** tied to construction, not a free hyperparameter to tune away from 1.8.
- **resolution (m/px)** – projection resolution used across stages (default **0.005**). Changing it requires regenerating all intermediate depth maps to keep the SAM templates aligned.

---

## Evidence-Based Y Detection (memory+state ablation, 30 tunnels)

### X positions (ring columns)
The evenly-spaced fallback `X_i ≈ (i + 0.5) * (W / ring_count)` matches measured `detected.csv` X spacing with **zero variance** across all evaluated tunnels. **Tuning focus is Y only** — locating the K-block center within each ring column.

### Dataset ID → family (same as `FAMILY_MAP` in `skills/scripts/compare_ablation_conditions.py`)
- **`1-*`, `2-*`**: regular staggered (6 segments / ring physically)
- **`3-*`**: continuous joints (6 segments / ring)
- **`4-*`, `5-*`**: complex staggered (7 segments / ring)

### Regular / staggered (`1-*`, `2-*`)
- **Physical expectation**: two-level alternating Y (staggered K-blocks).
- **Evidence**: `2-1`, `2-2` — 100% midpoint, Y ~1210 vs ~1640 (~430 px gap, ≈ `K_height_pixel/2` at resolution 0.005). `2-3`–`2-5` mostly follow this with some slope/fallback rows. `1-*` noisier (outliers at tunnel ends).
- **Goal**: maximize **midpoint** rate; tight oblique angle ranges (≈6–9°) usually help.

### Continuous (`3-*`)
- **Physical expectation**: K at **approximately the same Y** every ring (no stagger).
- **Evidence**: current pipeline often shows 30–40% **assume**/**default**; Y spread can be large when detection fails. Built-in **assume** logic was designed for staggered tunnels — improve horizontal/oblique detection instead of relying on assume.
- **Goal**: lower `hough_threshold_horizontal`, relax horizontal angle band; oblique lines may be weak.

### Complex (`4-*`, `5-*`)
- **`ring_spacing_constant`**: MUST be **1.8** (meters), matching 1.8 m ring length. Values such as 1.2 or 1.16 are wrong for this family and break spacing checks in detecting.
- **Physical layout**: 7 segments, interleaved K-blocks; scanner offset → variable density along X.
- **Y is not a simple global pattern**: measured **midpoint** Y can span ~400–4300 px with **no** consistent alternation, monotonic trend, or single constant. **Default** rows sit on image-center Y and are usually wrong.
- **Aggregate fallback** (default+assume): ~37% continuous, ~48% complex-4, ~36% complex-5 (memory+state runs).
- **Goal**: aggressive Hough tuning (lower thresholds, more dilation, longer `minLineLength` for 1.8 m rings); minimize **default** count.

#### Complex sparse-side recovery (left/right imbalance)
- If one side has many `default` rows while the other side has valid oblique/horizontal detections, prioritize **recall** first:
  - lower `hough_threshold_oblique` and `hough_threshold_horizontal` into ~20-30,
  - increase `dilation_iterations` to 2,
  - widen oblique angle windows to about ±(4-12) deg,
  - reduce `minLineLength_*` and increase `maxLineGap_*` to bridge fragmented joints.
- Keep this strategy **complex-only** (`4-*`, `5-*`). Do not apply to `1-*`, `2-*`, `3-*` by default because it can over-detect and harm already-stable cases.

### Detection type quality (Y)
| Type | Meaning |
|------|---------|
| **midpoint** | Best — both oblique families intersect |
| **positive_slope** / **negative_slope** | Adequate — single oblique + offset |
| **horizontal** | Usable when obliques missing |
| **default** | Worst — no geometry; Y = image center |
| **assume** | Copied from prior ring; propagates errors |

Target: keep **default** + **assume** as low as possible (<10% if achievable).

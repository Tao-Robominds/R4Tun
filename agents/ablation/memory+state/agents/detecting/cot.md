## Chain of Thought Instructions for Detecting Parameter Recommendations

Follow this structured analysis process when evaluating tunnel characteristics for detecting parameter recommendations:

### 0. CONSERVATIVE DEFAULT PRINCIPLE (read first, applies to every parameter)

When you are uncertain whether a parameter should deviate from the SAM4Tun
default, keep the default. The rules baseline beats LLM adaptation on several
complex tunnels precisely because it never strays from defaults it cannot
justify.

Reference SAM4Tun defaults for the detecting stage:
- binary_threshold = 127
- morphological_kernel_size = [3, 3]
- dilation_iterations = 1
- hough_threshold_oblique = 50
- hough_threshold_horizontal = 50
- hough_threshold_vertical = 500
- minLineLength_oblique = 100
- maxLineGap_oblique = 40
- minLineLength_horizontal = 100
- maxLineGap_horizontal = 10
- angle_range_oblique_positive = [6, 9]
- angle_range_oblique_negative = [-9, -6]
- merge_distance = 3
- ring_spacing_constant = 1.2 (regular) / 1.8 (complex 4-*, 5-*)
- resolution = 0.005

Do NOT lower hough_threshold_oblique or hough_threshold_horizontal below 50,
do NOT raise dilation_iterations above 1, and do NOT enlarge
morphological_kernel_size beyond 3x3 unless ALL of the following hold:
1. The current state context shows direct evidence of detection failure
   (default+assume rate >= 30% on at least one side, or measurable left/right
   imbalance in detected.csv).
2. The change is the smallest one that could plausibly fix that evidence.
3. You can name the specific symptom the change targets in your justification.

Background IoU is the single largest mIoU contributor. Over-aggressive
detection inflates false-positive lines in background regions and collapses
overall mIoU. Bias toward defaults; only deviate with evidence.

**Exception — continuous T3 (`3-*`) only:** when `K_ALIGNMENT_FAILURE` is documented (see below), you MAY lower `hough_threshold_horizontal` to **35–45** and set `maxLineGap_horizontal` to **20–30** to surface the horizontal K joint line. Do not apply this exception to `1-*`, `2-*`, `sample`, `4-*`, or `5-*`.

### 0b. K_UNIFORM_GATE / K_ALIGNMENT_GATE (mandatory for `3-*` before final JSON)

Continuous tunnels place all K-blocks on **one near-horizontal joint line** — same Y every ring (±40 px at `resolution=0.005`). Staggered T1/T2 alternates Y by ~430 px; that pattern must **not** drive `3-*` tuning.

**One-K-knows-all:** one reliable anchor defines **Y\*** and K block geometry for **all** rings; only X varies per ring column.

**Diagnostics from state** (`detected_characteristics`, or `initial_points.csv` if available):
- `Y_std` across ring prompt points (target **< 10 px** post-snap)
- `max_abs(Y − Y*)` after propagation (target **0 px**)
- `assume` + `default` rate in `type_distribution`
- `y_range` width in spatial bounds

**K_UNIFORM_FAILURE** (alias **K_ALIGNMENT_FAILURE** pre-snap) on `3-*` when:
- Pre-snap: `Y_std > 80` OR `assume + default > 30%`
- Post-snap: `Y_std > 10` OR `max_abs(Y − Y*) > 40`

**Strategy:**
1. **Anchor-and-uniform-snap:** compute `Y* = median(anchor Y)` from `midpoint`, `horizontal`, `positive_slope`, `negative_slope`. Pipeline sets **every ring** to `(X_i, Y*)` — anchors included; relabel rows `propagated`.
2. **Outlier anchors:** if one anchor deviates >40 px from median (e.g. spurious `midpoint`), exclude from median or still snap all rings to `Y*`.
3. **Param priority:** `hough_threshold_horizontal`, `minLineLength_horizontal`, `maxLineGap_horizontal`, `binary_threshold` — horizontal K seam is the primary cue; obliques often weak.
4. **Do not** tune `hough_threshold_vertical` for K Y (X columns use ring-spacing fallback).

Document `Y_std`, `max_abs(Y − Y*)`, anchor count, and planned horizontal Hough changes before the JSON fence.

### 1. ANCHORING
Compare key tunnel characteristics against the sample baseline:
- Point density changes and distribution patterns
- Tunnel diameter and scale differences
- Coordinate ranges and image resolution considerations

### 2. CLASSIFICATION
Classify the tunnel based on the comparison:
- **SIMILAR**: <25% difference in key metrics → minimal changes needed
- **DENSE**: Higher point density → may need threshold adjustments
- **LARGE-SCALE**: Significant size differences → may need parameter scaling
- **LOW-QUALITY**: Poor image clarity → may need sensitivity adjustments

### 3. PARAMETER ADAPTATION
Adapt parameters based on classification:
- **binary_threshold**: Adjust for image clarity and contrast
- **hough_threshold_oblique/horizontal/vertical**: Adapt for density and noise
- **minLineLength/maxLineGap**: Scale with tunnel dimensions
- **resolution**: Keep aligned with point density
- **angle_ranges, merge_distance, ring_spacing**: Generally stable

### Parameter Guidelines:
- **Always provide EXACT numerical values** - Never use ranges like "50-70"
- **Choose the most appropriate single value** from any range you consider
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**
- **Provide clear justification** for each parameter change
- **Output flowing analysis with section headers and final JSON parameter block**
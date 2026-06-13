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

**Exception — continuous tunnels (`3-1-*`) with anchor failure:** when step 2b
diagnostics show continuous-anchor failure (fallback-dominated
`type_distribution` and/or `y_range` spread > 150 px in detected
characteristics), you MAY lower `hough_threshold_horizontal` to **40–45** and
`hough_threshold_oblique` to **40–48** (not below 40), and raise
`maxLineGap_horizontal` to **12–15** px, provided you cite the state evidence.
This exception targets missing K-joint lines on continuous tunnels only; do not
apply it to staggered (`1-*`, `2-*`) or complex (`4-*`, `5-*`) tunnels.

Background IoU is the single largest mIoU contributor. Over-aggressive
detection inflates false-positive lines in background regions and collapses
overall mIoU. Bias toward defaults; only deviate with evidence.

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

### 2b. K-PATTERN PRIOR (mandatory for regular tunnels `1-*`, `2-*`, `3-*`)

Before adapting any parameter, state the **expected per-ring K-Y pattern** for this tunnel family:

| Family | Tunnel IDs | Expected K-Y pattern |
|--------|------------|----------------------|
| **Staggered** | `1-*`, `2-*` | Two-level alternation: odd rings at one Y band, even rings at another, gap ≈430 px |
| **Continuous** | `3-*` | Single constant Y across all rings (no alternation) |

Then:
1. Inspect the **state context** (`detected_characteristics.json` summaries) — do **not** use ground-truth labels. Key fields:
   - `type_distribution`: count of **midpoint**, **positive_slope**, **negative_slope**, **horizontal** vs **assume**/**default**
   - `spatial_bounds.y_range` (or equivalent Y min/max): spread of detected K-Y positions
2. **Continuous-anchor failure** (mandatory check for `3-*`): flag when **any** of:
   - `assume` + `default` ≥ **30%** of detection rows in `type_distribution`
   - `midpoint` + `assume` + `default` ≥ **50%** (weak geometry overall)
   - Y-range spread > **150 px** (continuous tunnels should be < 50 px when detection succeeds)
3. If continuous-anchor failure is flagged, prioritise recovering **joint-derived** rows (`midpoint`, `horizontal`, `positive_slope`, `negative_slope`) with the smallest threshold changes (see section 0 exception). Goal: every ring gets a slope/horizontal-derived Y within one narrow band.
4. For staggered tunnels, explain how each change increases **two-level alternation**; for continuous, explain how each change **collapses Y scatter** toward one band.
5. If the tunnel is SIMILAR to reference **and** state shows the K-Y pattern is already correct (continuous: Y spread < 50 px and fallback < 10%), **keep SAM4Tun defaults**.

### 3. PARAMETER ADAPTATION
Adapt parameters based on classification **and** the K-pattern prior from step 2b:
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
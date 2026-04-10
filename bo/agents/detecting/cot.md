## Chain of Thought — Detecting (Critical Parameters Only)

Follow this 3-step process. Eleven tunnel-responsive parameters plus 2 physical constants are adaptable; only `morphological_kernel_size` and `resolution` stay at baseline.

### 1. ANCHORING
Compare the target tunnel's enhanced characteristics against the sample:
- Point density and image contrast (affects binary_threshold)
- Tunnel family and diameter (determines Hough sensitivity, angle ranges, merge behavior)
- Ring structure (1.2 m vs 1.8 m rings — directly determines ring_spacing_constant and merge_distance)

### 2. PARAMETER ADAPTATION
Adapt ALL parameters from DOMAIN KNOWLEDGE based on family:

**Hough detection:**
- `hough_threshold_oblique/horizontal`: [20, 83] — **aggressively lower for T4/T5** (20–35)
- `hough_threshold_vertical`: [320, 980]
- `minLineLength_oblique/horizontal`: [60, 240] / [60, 220] — variable, can go shorter for fragmented joints
- `maxLineGap_oblique/horizontal`: [30, 100] / [12, 70] — widen for T4/T5

**Image & morphology (critical for T4/T5 ring count):**
- `binary_threshold`: [115, 127] — lower for T4/T5
- `merge_distance`: [3, 8] — **increase to 4–8 for T4/T5** (prevents ring over-counting)
- `angle_range_oblique_positive`: [4, 12] as [low, high] — widen for T4/T5
- `angle_range_oblique_negative`: [-12, -4] as [low, high] — mirror of positive

**Physical constants:**
- `ring_spacing_constant`: 1.2 for T1-T3, 1.8 for T4-T5
- `dilation_iterations`: 1 for regular, 2 for complex

### 3. OUTPUT
Output the full JSON with all adapted values.

### Parameter Guidelines:
- **Always provide EXACT numerical values** — Never use ranges
- For T4/T5: `merge_distance`, `angle_range_oblique_*`, and `binary_threshold` are just as critical as Hough thresholds
- Output flowing analysis with section headers and final JSON parameter block

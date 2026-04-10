## Tunnel Families

- **T1/T2 (1-x, 2-x):** 5.5 m inner diameter, 1.2 m rings, 6 segments/ring, staggered joints
- **T3 (3-x):** 5.5 m diameter, continuous joints, multi-station registration
- **T4/T5 (4-x, 5-x):** 7.5 m inner diameter, 1.8 m rings, 7 segments/ring, complex interleaved K-blocks

## Critical Parameters (Detecting Stage)

Eleven parameters are tunnel-responsive (must adapt per tunnel). Two parameters are physical constants set by family. The rest are locked.

### Tunnel-Responsive Parameters — Hough Detection

- **hough_threshold_oblique** — Accumulator threshold for oblique Hough lines. Empirical range: **[20, 83]**, baseline 50. T1/T2 ≈ 55–60 (dense, higher threshold); T3 ≈ 30–40; T4/T5 ≈ 20–35 (sparse, must lower significantly for line recovery). Adapted in **30/30** tunnels.

- **hough_threshold_horizontal** — Accumulator threshold for horizontal Hough lines. Empirical range: **[20, 83]**, baseline 50. Mirrors oblique pattern. T4/T5 ≈ 20–35. Adapted in **30/30** tunnels.

- **hough_threshold_vertical** — Accumulator threshold for vertical Hough lines. Empirical range: **[320, 980]**, baseline 500. Adapted in **28/30** tunnels.

- **maxLineGap_oblique (px)** — Max allowed gap when fitting oblique joints. Empirical range: **[30, 100]**, baseline 40. T4/T5 need 55–90 to bridge fragmented joints in larger rings. Adapted in **25/30** tunnels.

- **maxLineGap_horizontal (px)** — Max allowed gap for horizontal joints. Empirical range: **[12, 70]**, baseline 10. T4/T5 need 15–70; T3 need 18–25. Adapted in **20/30** tunnels.

- **minLineLength_oblique (px)** — Min segment length for oblique joints. Empirical range: **[60, 240]**, baseline 100. Highly variable for T4/T5 (60–240); some tunnels need shorter lines (60–90) for fragmented joint recovery, others longer. Adapted in **22/30** tunnels.

- **minLineLength_horizontal (px)** — Min segment length for horizontal joints. Empirical range: **[60, 220]**, baseline 100. Same variability as oblique. Adapted in **22/30** tunnels.

### Tunnel-Responsive Parameters — Image & Morphology

- **binary_threshold** — Depth map binarization threshold. Empirical range: **[115, 127]**, baseline 127. T4/T5 depth maps have lower contrast → lower threshold (115–124) improves joint visibility. T1/T2 keep 127. Adapted in **27/90** m_s_k files (primarily T3/T4/T5).

- **merge_distance (px)** — Max distance for merging nearby detected lines into one. Empirical range: **[3, 8]**, baseline 3. T4/T5 with 1.8 m rings produce wider joint signatures → need merge_distance 4–8 to avoid fragmenting single joints into multiple detections. **Critical for correct ring_count.** T1/T2 keep 3. Adapted in **23/90** m_s_k files.

- **angle_range_oblique_positive (deg)** — Accepted angle range for positive-slope oblique joints. Empirical range: **[4, 12]** (as [low, high]), baseline [6, 9]. T4/T5 have wider joint angles → widen to [4, 11] or [4, 12]. T1/T2 keep [6, 9]. Adapted in **54/90** m_s_k files.

- **angle_range_oblique_negative (deg)** — Accepted angle range for negative-slope oblique joints. Empirical range: **[-12, -4]** (as [low, high]), baseline [-9, -6]. Mirror of positive. T4/T5 → [-12, -4] or [-11, -4]. Adapted in **54/90** m_s_k files.

### Physical Constants (set by family, not tuned)

- **ring_spacing_constant (m)** — **1.2** for T1/T2/T3; **1.8** for T4/T5. Construction constant.

- **dilation_iterations** — Range **[1, 3]**, baseline 1. T4/T5 benefit from **2** for thicker crack visibility.

- **morphological_kernel_size** — Dilation kernel size. Baseline **[3, 3]**. T4/T5 may benefit from **[5, 5]** for thicker edge structures in larger-diameter tunnels. Adapted in 9/90 m_s_k files (primarily T4). Keep [3, 3] for T1/T2/T3.

### Locked Parameters (keep baseline)

| Parameter | Baseline |
|---|---|
| resolution | 0.005 |

### Adaptation Rules by Family

**Regular (1-x, 2-x):** Thresholds near baseline (oblique/horizontal 50–60); minLineLength 100; maxLineGap_oblique 40, maxLineGap_horizontal 10; binary_threshold 127; merge_distance 3; angle ranges [6,9]/[-9,-6]. Minimal changes needed.

**Continuous (3-x):** Lower thresholds (oblique 30–40, horizontal 25–35) for weaker joints; widen maxLineGap_horizontal to 18–25; may widen angle ranges to [5,10]/[-10,-5]; binary_threshold 120–127.

**Complex (4-x, 5-x):** Aggressively lower thresholds (oblique/horizontal **20–35**); widen maxLineGap (oblique 55–90, horizontal 15–70); increase merge_distance to **4–8** (critical for correct ring count with 1.8 m rings); widen angle ranges to **[4,12]/[-12,-4]**; lower binary_threshold to **115–124**; dilation_iterations = 2; ring_spacing_constant = 1.8. minLineLength is highly variable — can go lower (60–90) for fragmented joints or higher depending on image clarity.

## Council Comparison Report (Oblique Line Detection Parameters)

### 1. Parameter Changes
- **Gemini 3**: Moderately relaxed oblique detection vs. original (minLineLength 70, maxLineGap 60, hough_threshold_oblique 35), keeps small morphology (`[3,3]`, 1→2 dilations) and slightly widens angle windows to \([5,11]\) / \([-11,-5]\). Horizontal/vertical thresholds and lengths are left relatively strict.
- **GPT 5.2**: Further strengthens morphology with a `5x5` kernel and 2 dilations, and is slightly more permissive on gaps (`maxLineGap_oblique` 80) and angles \([5,12]\) / \([-12,-5]\). It also tightens horizontal/vertical detection (higher thresholds and min lengths) to counter extra noise from stronger dilation.
- **Opus 4.5**: Sits between Gemini and GPT on morphology and angles (`5x5`, 2 dilations; \([5,10]\) / \([-10,-5]\)), reducing `minLineLength_oblique` to 60 and raising `maxLineGap_oblique` to 70. It keeps horizontal parameters closer to the original (lower thresholds and lengths than GPT), accepting slightly more horizontal clutter.
- **Group (combined)**: Largely adopts GPT’s more aggressive connectivity and thresholds (`5x5`, 2 dilations, `hough_threshold_oblique` 35, `maxLineGap_oblique` 80) while pushing `minLineLength_oblique` down further to 60. Horizontal settings follow GPT (stricter) and vertical stays close to Gemini/original, forming a compromise that maximizes oblique recall with conservative axis-aligned detection.

### 2. Reasoning Style
- **Gemini 3**: Focuses on a qualitative explanation of right-side bias (denser, better-connected features) and argues mainly in terms of “weaker vs. stronger” features. The changes are framed as small, targeted relaxations to help the sparse left side without overhauling preprocessing.
- **GPT 5.2**: Gives the most detailed systems-level explanation, explicitly tying the binary mask, Hough voting mechanics, fragmentation, and angle windows together. It reasones about trade-offs between recall and false positives, justifying stronger morphology and stricter horizontal/vertical parameters to offset added noise.
- **Opus 4.5**: Presents a clear, concise diagnosis centered on four specific failure causes (length, gap, threshold, limited dilation) and directly maps each to a parameter adjustment. Its reasoning is pragmatic and optimization-oriented, accepting controlled false positives on the right to materially improve left-side recall.
- **Group (combined)**: Synthesizes prior ideas: it adopts GPT’s “understand the full pipeline” framing (Hough on a validity mask) and Opus’s focus on aggressive recall for obliques, while retaining Gemini’s caution on angle range and vertical lines. The result is a balanced, ensemble-style parameter set that encodes consensus where all models agree (lower oblique thresholds, more dilation) and compromises where they differ (horizontal/vertical strictness, exact angle span).



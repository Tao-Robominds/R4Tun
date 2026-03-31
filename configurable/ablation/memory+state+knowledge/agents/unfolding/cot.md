## Chain of Thought Instructions for Parameter Recommendations

Follow this structured 5-step analysis process when evaluating tunnel characteristics and making parameter recommendations. Use **DOMAIN KNOWLEDGE** for all numeric thresholds, regime definitions, parameter ranges, and per-parameter adaptation rules.

### 1. ANCHORING
Compare the current tunnel's point cloud characteristics against the fixed sample memories from the open source SAM4Tun implementation to establish a baseline of differences.

**Key metrics to compare:**
- Total points and point density (mean/median nearest neighbor distance)
- Tunnel dimensions (length, diameter, height)
- Coordinate ranges and intensity distributions

**Calculation process:**
- Calculate percentage differences: Δ = (new_value - sample_value) / sample_value × 100%
- Show exact numerical comparisons for all key metrics
- Document both absolute and relative differences

### 2. CLASSIFICATION
Group the tunnel into broad regimes based on the anchoring comparison. Apply the **classification criteria and logic** defined in DOMAIN KNOWLEDGE (SIMILAR, DENSE, SPARSE, UNBALANCED, LARGE-DIAMETER, CHALLENGING, VERY-SPARSE).

### 3. DIAGNOSTIC INSPECTION
Examine stage-specific cues to identify concrete challenges based on the classification. Follow the **diagnostic rules** in DOMAIN KNOWLEDGE for each regime (slice population checks, RANSAC considerations, vertical window coverage, etc.).

### 4. PARAMETER ADAPTATION
Consult DOMAIN KNOWLEDGE for parameter ranges, interdependencies, and **parameter-specific adaptation logic**. Apply **adaptation principles** below; use knowledge for all concrete numbers.

**Adaptation principles:**
- Apply adjustments ONLY when justified by clear evidence from steps 1-3
- Preserve robustness of original settings when tunnels are classified as SIMILAR
- For similar tunnels: explicitly recommend NO CHANGE to maintain proven robustness
- When characteristics show moderate differences (<30%), lean towards SIMILAR classification to maintain robustness
- Higher point density often improves rather than degrades processing — verify actual performance impact before adjusting
- Document specific reasoning for each parameter decision with evidence

**Evidence requirements:**
- Each parameter change must be supported by specific evidence from diagnostic inspection
- Quantify the problem being solved (e.g., "delta=0.005 creates 0.01m slices, but mean NND=0.0033m suggests 0.003 delta more appropriate")
- Consider parameter interdependencies and downstream impact

### 5. VALIDATION
Check that proposed changes resolve identified issues without undermining later stages:

**Consistency checks:**
- Ensure parameter changes align with tunnel regime classification
- Verify that modifications address the specific issues identified in diagnostic inspection
- Confirm that adaptations follow the evidence-based reasoning from anchoring analysis

**Downstream compatibility:**
- Check that slice thickness changes don't affect subsequent processing stages
- Ensure vertical_filter_window changes maintain adequate edge detection capability
- Verify RANSAC parameter modifications preserve fitting robustness

**Final validation criteria:**
- Parameters should resolve identified challenges while preserving proven robustness
- For SIMILAR tunnels: changes should be minimal or absent
- All modifications should have clear quantitative justification
- Output flowing analysis with section headers and final JSON parameter block

### Parameter Guidelines:
- **Always provide EXACT numerical values** — Never use ranges like "0.01-0.015" or "6-7"
- **Choose the most appropriate single value** from any range DOMAIN KNOWLEDGE suggests
- **Be specific and decisive** in your recommendations
- **If uncertain, choose the middle value** from a potential range and explain your reasoning
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**

Example of CORRECT recommendations:
- "Keep delta at 0.005 (no change needed - tunnel characteristics are similar to sample)"
- "Set delta to 0.012" (not "0.01-0.015")
- "Use ransac_sample_size of 6" (not "6-7")

Remember: The system requires exact values for implementation — ranges cannot be processed.

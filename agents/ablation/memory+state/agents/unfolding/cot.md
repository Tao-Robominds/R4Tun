## Chain of Thought Instructions for Parameter Recommendations

Follow this structured 5-step analysis process when evaluating tunnel characteristics and making parameter recommendations:

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
Group the tunnel into broad regimes based on the anchoring comparison:

**Classification criteria:**
- **SIMILAR**: <25% difference in key density metrics AND <25% in geometric metrics → minimal/no changes needed
- **DENSE**: Extremely higher point density (>35% decrease in nearest neighbor distance AND evidence of slice overcrowding) → reduce sampling parameters  
- **SPARSE**: Extremely lower point density (>35% increase in nearest neighbor distance AND evidence of slice underpopulation) → increase sampling tolerance
- **UNBALANCED**: Irregular density distribution across tunnel sections → adjust filtering parameters
- **LARGE-DIAMETER**: Significant tunnel size difference (>35% diameter/height change) → scale geometric parameters
- **CHALLENGING**: Combined indicators of difficult data conditions (high sparsity + large scale + noise indicators) → more robust parameter settings
- **VERY-SPARSE**: Extremely sparse data with significant quality challenges (>40% NND increase + large scale + post-processing quality indicators) → aggressive parameter adaptations

**Classification logic:**
- Primary classification based on density changes (nearest neighbor distance)
- Secondary classification based on geometric differences
- Multiple classifications possible (e.g., DENSE + LARGE-DIAMETER)

### 3. DIAGNOSTIC INSPECTION
Examine stage-specific cues to identify concrete challenges based on the classification:

**For DENSE tunnels:**
- Verify slice overcrowding: check if current slice thickness leads to excessive points per slice (>50% more than sample)
- Only reduce delta if clear evidence of processing bottlenecks or degraded fit quality
- Consider that higher point density may actually improve RANSAC robustness

**For SPARSE tunnels:**
- Verify slice underpopulation: check if current slice thickness leads to insufficient points per slice (<50% of sample)
- Only increase delta if clear evidence of poor fitting due to insufficient data
- Consider that moderate density reduction may still provide adequate sampling

**For CHALLENGING tunnels:**
- Assess combination of factors: sparsity, scale, and data quality indicators
- Check if default RANSAC threshold (0.5) is adequate for noisier, sparser data
- Consider if larger scale requires more robust fitting parameters
- Evaluate if multiple challenging factors compound the fitting difficulty

**For VERY-SPARSE tunnels:**
- Assess extreme sparsity requiring significant slice thickness increase (delta adjustment)
- Check if RANSAC sample size needs reduction due to insufficient points per slice
- Evaluate if RANSAC threshold needs substantial increase (1.5) for very noisy conditions
- Consider if multiple extreme factors require coordinated parameter increases

**For LARGE-DIAMETER tunnels:**
- Assess if vertical_filter_window covers appropriate tunnel height percentage
- Check if slice_spacing_factor scales appropriately with tunnel length

**General diagnostic checks:**
- **Density compatibility**: Ensure slice thickness (2×delta) relative to nearest neighbor distance
- **Geometric coverage**: Verify vertical_filter_window covers adequate tunnel height (target ~80-90%)
- **RANSAC robustness**: Consider if noise levels or point distribution affects fitting reliability

### 4. PARAMETER ADAPTATION
Consult encoded knowledge of parameter ranges and interdependencies to make specific parameter decisions:

**Adaptation principles:**
- Apply adjustments ONLY when justified by clear evidence from steps 1-3
- Preserve robustness of original settings when tunnels are classified as SIMILAR
- For similar tunnels: explicitly recommend NO CHANGE to maintain proven robustness
- When characteristics show moderate differences (<30%), lean towards SIMILAR classification to maintain robustness
- Higher point density often improves rather than degrades processing - verify actual performance impact before adjusting
- Document specific reasoning for each parameter decision with evidence

**Parameter-specific adaptation logic:**
- **delta**: Adjust based on density - decrease for DENSE, increase for SPARSE (typical range: 0.003-0.008), increase significantly for VERY-SPARSE (up to 0.01)
- **slice_spacing_factor**: Generally stable, adjust only for extreme length differences
- **vertical_filter_window**: Scale with tunnel height for LARGE-DIAMETER (target 80-90% coverage)
- **ransac_threshold**: Generally stable (0.5), but increase to 1.5 for CHALLENGING tunnels with combined sparsity + large scale + noise indicators
- **ransac_probability, ransac_inlier_ratio**: Preserve unless specific robustness issues
- **ransac_sample_size**: Generally stable (6), but reduce to 5 for VERY-SPARSE tunnels with insufficient points per slice
- **polynomial_degree**: Keep at 3 unless tunnel complexity requires change
- **num_samples_factor**: Generally stable, proven robust across tunnel types

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
- **Always provide EXACT numerical values** - Never use ranges like "0.01-0.015" or "6-7"
- **Choose the most appropriate single value** from any range you might consider
- **Be specific and decisive** in your recommendations
- **If uncertain, choose the middle value** from a potential range and explain your reasoning
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**

Example of CORRECT recommendations:
- "Keep delta at 0.005 (no change needed - tunnel characteristics are similar to sample)"
- "Set delta to 0.012" (not "0.01-0.015") 
- "Use ransac_sample_size of 6" (not "6-7")

Remember: The system requires exact values for implementation - ranges cannot be processed.
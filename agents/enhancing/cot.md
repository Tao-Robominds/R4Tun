## Chain of Thought Instructions for Enhancing Parameter Recommendations

Follow this structured 5-step analysis process when evaluating denoised tunnel characteristics and making enhancing parameter recommendations:

### 1. ANCHORING
Compare the current tunnel's denoised point cloud characteristics against the sample baseline to establish differences that affect enhancing performance.

**Key metrics to compare:**
- Point density after denoising (mean/median nearest neighbor distance)
- Data retention rate and point distribution patterns
- Surface geometry complexity and curvature distribution
- Spatial coverage and gap patterns that need enhancement

**Calculation process:**
- Calculate percentage differences: Δ = (new_value - sample_value) / sample_value × 100%
- Focus on denoised characteristics that impact enhancement effectiveness
- Document density and geometric variations that affect upsampling needs

### 2. CLASSIFICATION
Group the tunnel into enhancing regimes based on the anchoring comparison:

**Classification criteria:**
- **SIMILAR**: <25% difference in density metrics AND <150% curvature change → minimal/no changes needed
- **SPARSE**: Lower point density after denoising (>30% difference) → may need more aggressive upsampling
- **DENSE**: Higher point density after denoising (>30% difference) → may need less aggressive upsampling
- **LOW-QUALITY**: Poor denoising retention rate (>35% difference) → may need adjusted thresholds
- **COMPLEX-GEOMETRY**: Extreme curvature patterns (>300% change AND validated as significant) → may need sensitivity adjustments
- **LARGE-SCALE**: Significantly larger tunnel dimensions requiring scaled enhancement parameters → increase target distances and interpolation radii
- **CRITICAL-SPARSE**: Extremely sparse post-denoising with complex requirements (>80% data loss + large scale + high curvature) → specialized parameter combinations

**Classification logic:**
- Primary classification based on denoised point density changes
- Secondary classification based on data quality and geometric complexity
- Multiple classifications possible (e.g., SPARSE + COMPLEX-GEOMETRY)

### 3. DIAGNOSTIC INSPECTION
Examine enhancing-specific challenges based on the classification:

**For SPARSE tunnels:**
- Check if upsampling target distances need reduction for denser enhancement
- Assess if curvature_threshold needs relaxation for fewer feature points

**For DENSE tunnels:**
- Evaluate if upsampling target distances need increase to avoid over-densification
- Consider if duplicate_threshold needs adjustment for higher density

**For LOW-QUALITY tunnels:**
- Assess if depth_threshold_low/high need adjustment for gap filling
- Check if inter_radius needs modification for interpolation effectiveness

**For LARGE-SCALE tunnels:**
- Scale upsampling target distances proportionally (e.g., stage1: 0.10, stage2: 0.05, stage3: 0.025 for larger tunnels vs default 0.06/0.03/0.015)
- Increase inter_radius for larger interpolation range (e.g., 0.08 vs default 0.03)
- Adjust n_segment range to match tunnel ring structure (e.g., 10-21 for 20-ring tunnels vs 0-9 for 10-ring)

**For CRITICAL-SPARSE tunnels:**
- Use larger target distances despite sparsity to accommodate scale (0.10/0.05/0.025)
- Dramatically reduce curvature_threshold for aggressive feature capture (e.g., 0.0003 vs default 0.005)
- Increase inter_radius significantly (0.08) for wide gap interpolation
- Extend n_segment range for full tunnel coverage (e.g., 0-13 for 16-ring tunnels)

**General diagnostic checks:**
- **Upsampling compatibility**: Ensure target distances match point density patterns
- **Geometry sensitivity**: Verify curvature and depth thresholds suit surface complexity
- **Interpolation effectiveness**: Consider if neighbor counts and resolution match data quality

### 4. PARAMETER ADAPTATION
Consult encoded knowledge of enhancing parameter ranges and interdependencies:

**Adaptation principles:**
- Apply adjustments ONLY when justified by clear evidence from steps 1-3
- Preserve robustness of original settings when tunnels are classified as SIMILAR
- For similar tunnels: explicitly recommend NO CHANGE to maintain proven robustness
- Large curvature differences (>100%) often reflect processing variations rather than true geometry - verify significance
- Moderate density differences (<25%) and curvature changes (<150%) should lean towards SIMILAR classification
- Original curvature_threshold (0.005) is robust across diverse tunnel types - prefer keeping unless extreme validated differences
- Document specific reasoning for each parameter decision with evidence

**Parameter-specific adaptation logic:**
- **upsampling_stage1/2/3_target_distance**: Adjust based on density and scale - decrease for SPARSE, increase for DENSE or LARGE-SCALE (ranges: 0.06-0.10, 0.03-0.05, 0.015-0.025)
- **curvature_threshold**: Adjust based on geometry complexity - decrease for COMPLEX-GEOMETRY or CRITICAL-SPARSE (0.0003 for aggressive feature capture), increase for simpler surfaces (range: 0.0003-0.008)
- **depth_threshold_low/high**: Adjust based on data quality - relax for LOW-QUALITY (typical ranges: 0.002-0.005, 0.006-0.012)
- **inter_radius**: Generally stable, but increase for LARGE-SCALE tunnels or extreme density differences (typical range: 0.03-0.08)
- **duplicate_threshold**: Adjust based on final density needs (typical range: 0.015-0.03)
- **num_neighbors, num_interpolations**: Generally stable, proven robust across tunnel types
- **resolution, window_size**: Generally stable unless extreme quality issues

**Evidence requirements:**
- Each parameter change must be supported by specific evidence from diagnostic inspection
- Quantify the problem being solved (e.g., "upsampling_stage1_target_distance=0.08 too coarse for NN distance=0.044")
- Consider parameter interdependencies and enhancement quality

### 5. VALIDATION
Check that proposed changes resolve identified issues without undermining enhancement effectiveness:

**Consistency checks:**
- Ensure parameter changes align with tunnel regime classification
- Verify that modifications address the specific issues identified in diagnostic inspection
- Confirm that adaptations follow the evidence-based reasoning from anchoring analysis

**Enhancement effectiveness:**
- Check that upsampling parameters produce appropriate density enhancement
- Ensure geometry thresholds maintain surface feature detection
- Verify interpolation parameters preserve surface quality

**Final validation criteria:**
- Parameters should resolve identified challenges while preserving proven robustness
- For SIMILAR tunnels: changes should be minimal or absent
- All modifications should have clear quantitative justification
- Output flowing analysis with section headers and final JSON parameter block

### Parameter Guidelines:
- **Always provide EXACT numerical values** - Never use ranges like "0.03-0.05" or "15-25"
- **Choose the most appropriate single value** from any range you might consider
- **Be specific and decisive** in your recommendations
- **If uncertain, choose the middle value** from a potential range and explain your reasoning
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**

Example of CORRECT recommendations:
- "Keep upsampling_stage1_target_distance at 0.08 (no change needed - tunnel characteristics are similar to sample)"
- "Set curvature_threshold to 0.0004" (not "0.0003-0.0005")
- "Use inter_radius of 0.05" (not "0.04-0.06")

Remember: The system requires exact values for implementation - ranges cannot be processed.
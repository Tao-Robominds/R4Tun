## Chain of Thought Instructions for Enhancing Parameter Recommendations

Follow this structured analysis process when evaluating denoised tunnel characteristics and making enhancing parameter recommendations:

### 0. CONSERVATIVE DEFAULT PRINCIPLE (read first, applies to every parameter)

When uncertain whether a parameter should deviate from the SAM4Tun reference
default, keep the reference default. Only change a parameter when you have
clear evidence from the tunnel characteristics that the default would cause a
specific problem.

SAM4Tun reference defaults for enhancing:
- upsampling_stage1/2/3_target_distance = 0.08 / 0.04 / 0.02
- curvature_threshold = 0.0005
- depth_threshold_low / depth_threshold_high = 0.003 / 0.008
- inter_radius = 0.06
- duplicate_threshold = 0.02
- n_segment_start = 0, n_segment_end = 6 (regular) or 7+ (complex)
- num_neighbors = 20, num_interpolations = 2
- resolution = 0.005, window_size = 9

For complex tunnels (4-\*, 5-\*) where geometry differs significantly from the
sample tunnel, prefer SAM4Tun reference defaults as the safe starting point.
The "proven robust defaults" listed later in this document were calibrated on
regular/continuous tunnel types and may not generalise to complex tunnels.

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
- **LARGE-SCALE**: Tunnel diameter >50% larger than sample (~8.25m+) requiring scaled enhancement parameters → increase target distances
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

**For LARGE-SCALE tunnels (>50% diameter increase only):**
- Scale upsampling target distances only for genuinely large tunnels (e.g., stage1: 0.08, stage2: 0.04, stage3: 0.02 for >8m diameter). For modest diameter increases (<30%), keep defaults 0.06/0.03/0.015.
- Increase inter_radius to 0.05 only for genuinely large tunnels with wide gaps. Default 0.03 is sufficient for most cases.
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

**CRITICAL**: The REFERENCE PARAMETERS section in this prompt shows the baseline (sam4tun) starting-point defaults. These are NOT the proven optimal values. The PROVEN ROBUST DEFAULTS listed below supersede the reference parameters. For SIMILAR tunnels, always use the proven defaults, not the reference values. In particular:
- Reference shows target distances 0.08/0.04/0.02 → proven defaults are **0.06/0.03/0.015** (denser, gap-free)
- Reference shows inter_radius 0.06 → proven default is **0.03** (tighter, surface-preserving)
- Reference shows depth_threshold_high 0.008 → proven default is **0.015** (detects more outliers for gap filling)

**Adaptation principles:**
- Apply adjustments ONLY when justified by clear evidence from steps 1-3
- For SIMILAR tunnels: use the proven robust defaults below (NOT the reference parameters)
- Large curvature differences (>100%) often reflect processing variations rather than true geometry - verify significance
- Moderate density differences (<25%) and curvature changes (<150%) should lean towards SIMILAR classification
- Original curvature_threshold (0.005) is robust across diverse tunnel types - prefer keeping unless extreme validated differences
- Document specific reasoning for each parameter decision with evidence

**Proven robust defaults (use these unless strong evidence requires change):**
- **upsampling_stage1/2/3_target_distance** = 0.06 / 0.03 / 0.015 (produces dense, gap-free depth maps; proven on diverse tunnels)
- **curvature_threshold** = 0.005 (robust feature detection threshold)
- **depth_threshold_low / depth_threshold_high** = 0.005 / 0.015 (detects sufficient outliers for interpolation)
- **inter_radius** = 0.03 (tight interpolation radius preserves surface accuracy)
- **duplicate_threshold** = 0.02
- **num_neighbors** = 20, **num_interpolations** = 2, **resolution** = 0.005, **window_size** = 9

**Parameter-specific adaptation logic:**
- **upsampling_stage1/2/3_target_distance**: Default 0.06/0.03/0.015. Only increase to 0.08/0.04/0.02 for CRITICAL-SPARSE tunnels. Only increase to 0.10/0.05/0.025 for extreme LARGE-SCALE (>50% diameter increase). For SIMILAR tunnels, always use the defaults.
- **curvature_threshold**: Default 0.005. Decrease to 0.0003-0.001 only for CRITICAL-SPARSE with aggressive feature capture needs. Increase slightly (0.006-0.008) only for very simple tunnel geometry.
- **depth_threshold_low/high**: Default 0.005/0.015. Relax for LOW-QUALITY. The high threshold controls how many outlier points get interpolated — higher values detect more outliers and fill more gaps.
- **inter_radius**: Default 0.03. Only increase for CRITICAL-SPARSE (0.05-0.08). Larger inter_radius can blur surface detail — avoid increasing without evidence.
- **duplicate_threshold**: Default 0.02. Adjust based on final density needs (range: 0.015-0.03).
- **num_neighbors, num_interpolations**: Default 20, 2. Generally stable, proven robust across tunnel types.
- **resolution, window_size**: Default 0.005, 9. Generally stable unless extreme quality issues.

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
- "Keep upsampling_stage1_target_distance at 0.06 (proven default — tunnel characteristics are similar to sample)"
- "Set curvature_threshold to 0.005" (not "0.003-0.005")
- "Use inter_radius of 0.03" (not "0.03-0.06")

Remember: The system requires exact values for implementation - ranges cannot be processed.
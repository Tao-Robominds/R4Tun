## Chain of Thought Instructions for Denoising Parameter Recommendations

Follow this structured 5-step analysis process when evaluating unfolded tunnel characteristics and making denoising parameter recommendations:

### 1. ANCHORING
Compare the current tunnel's unfolded point cloud characteristics against the sample baseline to establish differences that affect denoising performance.

**Key metrics to compare:**
- Point density in cylindrical coordinates (mean/median nearest neighbor distance)
- Radial distribution and tunnel diameter characteristics
- Angular coverage and theta span characteristics  
- Ring structure and height distribution patterns

**Calculation process:**
- Calculate percentage differences: Δ = (new_value - sample_value) / sample_value × 100%
- Focus on cylindrical coordinate differences that impact denoising effectiveness
- Document density and geometric variations that affect noise patterns

### 2. CLASSIFICATION
Group the tunnel into denoising regimes based on the anchoring comparison:

**Classification criteria:**
- **SIMILAR**: <25% difference in key density metrics AND radial span <200% change → minimal/no changes needed
- **DENSE**: Significantly higher point density (>30% decrease in NN distance) → may need finer grid parameters
- **SPARSE**: Significantly lower point density (>30% increase in NN distance) → may need coarser grid parameters  
- **THICK-RING**: Extreme radial span difference (>300% change AND validated as real geometric difference) → adjust radial masking parameters
- **ANGULAR-DENSE**: Significant theta coverage difference (>25% change) → adjust angular grid parameters
- **LARGE-DIAMETER**: Tunnel diameter significantly larger than sample (>30% increase) → scale radial mask parameters appropriately
- **EXTREME-RANGE**: Very wide radial distribution requiring extensive mask expansion (>60% radial span increase OR >50% diameter increase + irregular distribution patterns) → aggressive radial parameter adaptation

**Classification logic:**
- Primary classification based on cylindrical density changes
- Secondary classification based on radial span and angular coverage differences
- Multiple classifications possible (e.g., DENSE + THICK-RING)

### 3. DIAGNOSTIC INSPECTION
Examine denoising-specific challenges based on the classification:

**For DENSE tunnels:**
- Check if y_step (angular grid) needs reduction for finer sampling
- Assess if grad_threshold needs adjustment for noise detection sensitivity

**For SPARSE tunnels:**
- Evaluate if y_step needs increase to capture sufficient points per grid cell
- Consider if grad_threshold needs relaxation for sparse data

**For THICK-RING tunnels:**
- Assess if mask_r_low/mask_r_high need adjustment for different radial spans
- Check if z_step needs modification for radial density variations

**For LARGE-DIAMETER tunnels:**
- Scale mask_r_low/mask_r_high proportionally to tunnel diameter (e.g., for ~7.2m diameter: 3.5-3.8 vs default 2.8-3.0 for ~5.5m)
- Consider if y_step needs increase for coarser angular sampling due to larger circumference
- Evaluate if z_step needs decrease for finer radial resolution in larger tunnels

**For EXTREME-RANGE tunnels:**
- Expand radial mask dramatically to capture full range (e.g., 2.2-3.9 for very wide distribution vs standard 2.8-3.0)
- Assess if z_step needs significant decrease for finer radial sampling across wide range
- Consider if irregular tunnel shape requires non-standard mask positioning

**General diagnostic checks:**
- **Grid compatibility**: Ensure y_step captures appropriate angular resolution
- **Radial masking**: Verify mask_r_low/mask_r_high match tunnel diameter characteristics
- **Noise sensitivity**: Consider if grad_threshold suits the density pattern

### 4. PARAMETER ADAPTATION
Consult encoded knowledge of denoising parameter ranges and interdependencies:

**Adaptation principles:**
- Apply adjustments ONLY when justified by clear evidence from steps 1-3
- Preserve robustness of original settings when tunnels are classified as SIMILAR
- For similar tunnels: explicitly recommend NO CHANGE to maintain proven robustness
- Large radial span differences (>100%) are often measurement or preprocessing artifacts - require validation before adapting radial masks
- Moderate density differences (<25%) should lean towards SIMILAR classification for robustness
- Original mask parameters [2.8, 3.0] are robust across diverse tunnel geometries - prefer keeping unless extreme validated differences
- Document specific reasoning for each parameter decision with evidence

**Parameter-specific adaptation logic:**
- **mask_r_low/mask_r_high**: Adjust based on tunnel characteristics - proportional scaling for LARGE-DIAMETER ([3.5, 3.8] for ~7.2m), or extreme expansion for EXTREME-RANGE ([2.2, 3.9] for wide distribution)
- **y_step**: Adjust based on angular density - decrease for DENSE, increase for SPARSE or LARGE-DIAMETER (typical range: 0.3-0.8)
- **z_step**: Generally stable unless extreme radial density variations, LARGE-DIAMETER (0.002-0.004), or EXTREME-RANGE tunnels (0.002 for wide range sampling)
- **grad_threshold**: Adjust based on noise sensitivity - decrease for DENSE, increase for SPARSE (typical range: 0.15-0.3)
- **smoothing_window_size**: Generally stable, proven robust across tunnel types
- **smoothing_offset**: Generally stable unless specific calibration issues

**Evidence requirements:**
- Each parameter change must be supported by specific evidence from diagnostic inspection
- Quantify the problem being solved (e.g., "y_step=0.5 creates too coarse angular sampling for theta_span=989°")
- Consider parameter interdependencies and noise removal effectiveness

### 5. VALIDATION
Check that proposed changes resolve identified issues without undermining denoising effectiveness:

**Consistency checks:**
- Ensure parameter changes align with tunnel regime classification
- Verify that modifications address the specific issues identified in diagnostic inspection
- Confirm that adaptations follow the evidence-based reasoning from anchoring analysis

**Denoising effectiveness:**
- Check that grid parameters capture appropriate point distributions
- Ensure radial masking parameters match tunnel geometry
- Verify gradient threshold maintains noise detection sensitivity

**Final validation criteria:**
- Parameters should resolve identified challenges while preserving proven robustness
- For SIMILAR tunnels: changes should be minimal or absent
- All modifications should have clear quantitative justification
- Output flowing analysis with section headers and final JSON parameter block

### Parameter Guidelines:
- **Always provide EXACT numerical values** - Never use ranges like "0.25-0.3" or "2.6-2.9"
- **Choose the most appropriate single value** from any range you might consider
- **Be specific and decisive** in your recommendations
- **If uncertain, choose the middle value** from a potential range and explain your reasoning
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**

Example of CORRECT recommendations:
- "Keep mask_r_low at 2.7 (no change needed - tunnel characteristics are similar to sample)"
- "Set grad_threshold to 0.27" (not "0.25-0.3")
- "Use y_step of 0.35" (not "0.3-0.4")

Remember: The system requires exact values for implementation - ranges cannot be processed.
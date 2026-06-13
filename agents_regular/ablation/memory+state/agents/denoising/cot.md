## Chain of Thought Instructions for Denoising Parameter Recommendations

Follow this structured analysis process when evaluating unfolded tunnel characteristics and making denoising parameter recommendations:

### 0. CONSERVATIVE DEFAULT PRINCIPLE (read first, applies to every parameter)

When uncertain whether a parameter should deviate from the SAM4Tun reference
default, keep the reference default. Only change a parameter when you have
clear evidence from the tunnel characteristics that the default would cause a
specific problem.

SAM4Tun reference defaults for denoising:
- mask_r_low / mask_r_high = derived from tunnel radius (set via r_percentiles)
- y_step = 0.5
- z_step = 0.001
- grad_threshold = 0.2
- smoothing_window_size = 3
- smoothing_offset = -0.003
- default_cutoff_z = set to just above the tunnel wall radius

For complex tunnels (4-\*, 5-\*) where geometry differs significantly from the
sample tunnel, prefer SAM4Tun reference defaults as the safe starting point.
The "proven robust defaults" listed later in this document were calibrated on
regular/continuous tunnel types and may not generalise to complex tunnels.

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

**For LARGE-DIAMETER tunnels (>30% diameter increase, i.e. >7.2m):**
- Scale mask_r_low/mask_r_high proportionally to tunnel median radius (e.g., for ~7.2m diameter: [3.5, 3.8] vs default [2.8, 3.0] for ~5.5m). For modest diameter increases (<30%, e.g. 5.5m→5.8m), keep default [2.8, 3.0].
- Keep y_step at 0.4 (finer resolution is always better for surface detection). Do NOT increase y_step for larger tunnels.
- Keep z_step at 0.005 unless extreme radial variations.

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

**CRITICAL**: The REFERENCE PARAMETERS section in this prompt shows the baseline (sam4tun) starting-point defaults. These are NOT the proven optimal values. The PROVEN ROBUST DEFAULTS listed below supersede the reference parameters. For SIMILAR tunnels, always use the proven defaults, not the reference values. In particular:
- Reference shows y_step 0.5 → proven default is **0.4** (finer angular resolution)
- Reference shows z_step 0.001 → proven default is **0.005** (robust radial resolution)
- Reference shows grad_threshold 0.2 → proven default is **0.15** (better noise sensitivity)
- Reference shows smoothing_window_size 3 → proven default is **5** (smoother cutoff surface)
- Reference shows smoothing_offset -0.003 → proven default is **-0.002**

**Adaptation principles:**
- For SIMILAR tunnels: use the proven robust defaults below (NOT the reference parameters)
- Large radial span differences (>100%) are often measurement or preprocessing artifacts - require validation before adapting radial masks
- Moderate density differences (<25%) should lean towards SIMILAR classification for robustness
- Mask parameters must be derived from r_percentiles: mask_r_low at p10, mask_r_high at p99 for each tunnel
- Document specific reasoning for each parameter decision with evidence

**Proven robust defaults (use these unless strong evidence requires change):**
- **y_step** = 0.4 (finer angular grid produces better surface detection; proven on diverse tunnels)
- **z_step** = 0.005 (robust radial grid resolution)
- **grad_threshold** = 0.15 (sensitive gradient detection catches more noise boundaries)
- **smoothing_window_size** = 5 (larger window produces smoother, more accurate cutoff surfaces)
- **smoothing_offset** = -0.002 (standard radial cutoff offset)
- **mask_r_low/mask_r_high**: Set based on the **r_percentiles** from unfolded characteristics. mask_r_low should be at or below **p10** of r (to retain edge points where tunnel radius is locally smaller). mask_r_high MUST cover at least the **p99** of r to retain 99%+ of tunnel wall points. If p99 > 3.0, you MUST increase mask_r_high accordingly — otherwise large areas of the depth map will be white (missing data).

**Parameter-specific adaptation logic:**
- **mask_r_low/mask_r_high**: Use **r_percentiles** from unfolded characteristics: set mask_r_low at or below p10, mask_r_high to at least p99 of r. Example: if p10=2.77, p99=3.06, use [2.77, 3.06]. The mask MUST capture at least 99% of tunnel wall points — insufficient mask width is the #1 cause of large white areas in depth maps.
- **y_step**: Default 0.4. Only increase to 0.5-0.6 for VERY-SPARSE tunnels where angular bins have too few points. NEVER increase beyond 0.6.
- **z_step**: Default 0.005. For EXTREME-RANGE tunnels with wide radial spans: decrease to 0.002-0.003.
- **grad_threshold**: Default 0.15. Only increase to 0.2-0.25 for genuinely SPARSE data with high noise levels. Lower values produce cleaner denoising.
- **smoothing_window_size**: Default 5. Only reduce to 3 for very short tunnels with few angular bins.
- **smoothing_offset**: Default -0.002. Only change for specific calibration needs.

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
- "Set mask_r_low to 2.77 (at p10), mask_r_high to 3.06 (at p99) — captures 99%+ of wall points"
- "Set grad_threshold to 0.15" (proven default; not "0.15-0.25")
- "Use y_step of 0.4" (proven default for quality surface detection; not "0.3-0.5")

Remember: The system requires exact values for implementation - ranges cannot be processed.
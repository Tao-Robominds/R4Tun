## Chain of Thought Instructions for Enhancing Parameter Recommendations

Follow this structured 5-step analysis process when evaluating denoised tunnel characteristics and making enhancing parameter recommendations. Use **DOMAIN KNOWLEDGE** for all numeric thresholds, regime definitions, proven defaults, and per-parameter adaptation logic.

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
Group the tunnel into enhancing regimes based on the anchoring comparison. Apply **classification criteria and logic** from DOMAIN KNOWLEDGE (SIMILAR, SPARSE, DENSE, LOW-QUALITY, COMPLEX-GEOMETRY, LARGE-SCALE, CRITICAL-SPARSE).

### 3. DIAGNOSTIC INSPECTION
Examine enhancing-specific challenges based on the classification. Follow **diagnostic rules** in DOMAIN KNOWLEDGE for each regime (upsampling distances, curvature, depth thresholds, inter_radius, n_segment ranges, CRITICAL-SPARSE combinations).

### 4. PARAMETER ADAPTATION
Consult DOMAIN KNOWLEDGE for reference vs proven defaults, proven robust defaults, and parameter-specific adaptation logic.

**Adaptation principles:**
- Apply adjustments ONLY when justified by clear evidence from steps 1-3
- For SIMILAR tunnels: use proven robust defaults in DOMAIN KNOWLEDGE (NOT reference parameters alone)
- Large curvature differences (>100%) often reflect processing variations — verify significance
- Moderate density differences (<25%) and curvature changes (<150%) should lean towards SIMILAR classification
- curvature_threshold 0.005 is robust across tunnel types — prefer keeping unless extreme validated differences
- Document specific reasoning for each parameter decision with evidence

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
- **Always provide EXACT numerical values** — Never use ranges like "0.03-0.05" or "15-25"
- **Choose the most appropriate single value** from any range DOMAIN KNOWLEDGE suggests
- **Be specific and decisive** in your recommendations
- **If uncertain, choose the middle value** from a potential range and explain your reasoning
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**

Example of CORRECT recommendations:
- "Keep upsampling_stage1_target_distance at 0.06 (proven default — tunnel characteristics are similar to sample)"
- "Set curvature_threshold to 0.005" (not "0.003-0.005")
- "Use inter_radius of 0.03" (not "0.03-0.06")

Remember: The system requires exact values for implementation — ranges cannot be processed.

## Chain of Thought Instructions for Denoising Parameter Recommendations

Follow this structured analysis process when evaluating unfolded tunnel characteristics and making denoising parameter recommendations. Use **DOMAIN KNOWLEDGE** for all numeric thresholds, regime definitions, proven defaults, r_percentile rules, and per-parameter adaptation logic.

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

**T1/T2 SIMILAR_TO_SAMPLE (1-\*, 2-\*):** start with frozen sample mask
`mask_r_low=2.7`, `mask_r_high=2.8` and grid `y_step=0.5`, `z_step=0.001`.
Switch to rules formula (`mask_r_low = d/2 − 0.15`, `mask_r_high = d/2 + 0.15`)
**only** when unfolded `r_percentiles` prove sample mask excludes ≥1% of wall
points. Rules widening helped 1-1 but hurt 2-1 — this must be evidence-driven,
not default.

For T3/T4/T5 use rules formula from diameter, not percentile p10/p99 defaults alone.

### MASK_RETENTION_GATE (mandatory before final JSON)

From unfolded `r_percentiles` or denoised state, compute:

```
wall_pct = % points with mask_r_low <= r <= mask_r_high
```

**MASK_FAILURE** if `wall_pct < 15%` OR `p50(r) > mask_r_high` OR `p10(r) < mask_r_low` OR **`mask_r_high < p99`**.

On MASK_FAILURE:
- Do **not** keep sample `[2.7, 2.8]`.
- **T3 / continuous-joint (`3-*`):** if `p50(r) > d/2 + 0.15` (wall sits above rules high), rules `[2.6, 2.9]` is too narrow — set `mask_r_low = p10 − 0.02`, `mask_r_high = p99 + 0.02`.
- Otherwise widen to rules `[d/2 − 0.15, d/2 + 0.15]` and re-check `wall_pct`.
- Target **`wall_pct ≥ 50%`** before optional `mask_r_low` trim (never trim `mask_r_high` below `p99 + 0.02`).

Document `wall_pct`, p5/p50/p99, and chosen mask in prose before the JSON fence.

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
Group the tunnel into denoising regimes based on the anchoring comparison. Apply **classification criteria and logic** from DOMAIN KNOWLEDGE (SIMILAR, DENSE, SPARSE, THICK-RING, ANGULAR-DENSE, LARGE-DIAMETER, EXTREME-RANGE).

### 3. DIAGNOSTIC INSPECTION
Examine denoising-specific challenges based on the classification. Follow **diagnostic rules** in DOMAIN KNOWLEDGE for each regime (grid, radial mask, grad_threshold, LARGE-DIAMETER >7.2m rules, EXTREME-RANGE mask expansion, etc.).

### 4. PARAMETER ADAPTATION
Consult DOMAIN KNOWLEDGE for: reference vs proven defaults, proven robust defaults, r_percentile mask rules, and parameter-specific adaptation logic.

**Adaptation principles:**
- For SIMILAR tunnels: use the proven robust defaults in DOMAIN KNOWLEDGE (NOT the reference parameters block alone)
- Large radial span differences (>100%) are often measurement or preprocessing artifacts — validate before adapting radial masks
- Moderate density differences (<25%) should lean towards SIMILAR classification for robustness
- **For SIMILAR / T1/T2 tunnels:** retain **frozen sam4tun sample** mask `[2.7, 2.8]` only when `wall_pct ≥ 15%` — do **not** substitute legacy "proven defaults" when retention fails.
- **For T3 (`3-*`):** never default to sample mask; derive mask from `r_percentiles` per MASK_RETENTION_GATE and T3_CONTINUOUS tight-bracket rule.
- Mask parameters for T4/T5: rules formula `mask_r_low = d/2 − 0.15`, `mask_r_high = d/2 + 0.15`.
- Document specific reasoning for each parameter decision with evidence

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
- **Always provide EXACT numerical values** — Never use ranges like "0.25-0.3" or "2.6-2.9"
- **Choose the most appropriate single value** from any range DOMAIN KNOWLEDGE suggests
- **Be specific and decisive** in your recommendations
- **If uncertain, choose the middle value** from a potential range and explain your reasoning
- **For SIMILAR tunnels: explicitly recommend keeping original parameters**

Example of CORRECT recommendations:
- "Set mask_r_low to 2.77 (at p10), mask_r_high to 3.06 (at p99) — captures 99%+ of wall points"
- "Set grad_threshold to 0.15" (proven default; not "0.15-0.25")
- "Use y_step of 0.4" (proven default for quality surface detection; not "0.3-0.5")

Remember: The system requires exact values for implementation — ranges cannot be processed.

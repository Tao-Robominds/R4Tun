## Chain of Thought — Unfolding (Critical Parameters Only)

Follow this 3-step process. Four parameters may be adapted; all others must remain at baseline.

### 1. ANCHORING
Compare the target tunnel's raw characteristics against the sample:
- Tunnel diameter (RANSAC estimate vs baseline 5.5 m)
- Tunnel family (T1/T2 vs T3 vs T4/T5)
- Ring spacing (1.2 m vs 1.8 m)
- Registration method (single-station T1/T2/T4/T5 vs multi-station T3)

### 2. PARAMETER ADAPTATION
From DOMAIN KNOWLEDGE:
- `diameter`: RANSAC estimate; range [5.31, 7.6]
- `slice_spacing_factor`: 1.2 for T1/T2/T3; 1.8 for T4/T5 (construction constant)
- `vertical_filter_window`: 4.5 for T1/T2; 5.0–5.5 for T3; 6.5–6.9 for T4/T5
- `delta`: 0.005 for all except T3 → 0.006–0.01

### 3. OUTPUT
Keep ALL other parameters at baseline. Output the full JSON.

### Parameter Guidelines:
- **Always provide EXACT numerical values**
- `slice_spacing_factor` is a construction constant: 1.2 or 1.8, nothing in between
- `delta` only changes for T3 (multi-station registration)
- Output flowing analysis with section headers and final JSON parameter block

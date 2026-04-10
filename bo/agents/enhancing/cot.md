## Chain of Thought — Enhancing (Critical Parameters Only)

Follow this 3-step process. Only 5 tunnel-responsive parameters may be adapted; 3 others use proven defaults; the rest stay at baseline.

### 1. ANCHORING
Compare the target tunnel's denoised characteristics against the sample:
- Point density after denoising (nearest neighbor distance)
- Data retention rate
- Tunnel family and diameter
- Ring count for n_segment scaling

### 2. PARAMETER ADAPTATION
Adapt 5 tunnel-responsive parameters from DOMAIN KNOWLEDGE:
- `upsampling_stage1/2/3_target_distance`: maintain 2:1 ratio; range [0.055, 0.111] / [0.028, 0.056] / [0.014, 0.028]
- `inter_radius`: 0.03 for dense, 0.038–0.043 for large/sparse; range [0.03, 0.08]
- `n_segment_end`: scale with ring count; range [5, 21]

Hard-code these proven defaults:
- `curvature_threshold` = 0.005
- `depth_threshold_low` = 0.005
- `depth_threshold_high` = 0.015

### 3. OUTPUT
Output the full JSON with tunnel-responsive values adapted, proven defaults applied, and all locked parameters unchanged.

### Parameter Guidelines:
- **Always provide EXACT numerical values**
- Maintain proportional ratios between upsampling stages
- Output flowing analysis with section headers and final JSON parameter block

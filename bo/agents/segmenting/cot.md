## Chain of Thought — SAM Segmenting (Critical Parameters Only)

Follow this 3-step process. Adapt key driving parameters and scale all derived geometry proportionally.

### 1. ANCHORING
Compare the target tunnel against the sample:
- Tunnel family (determines segment_per_ring, segment_order)
- Diameter ratio vs baseline 5.5 m → pixel scaling factor
- Ring length ratio vs baseline 1.2 m → horizontal scaling factor
- Image dimensions from enhanced characteristics

### 2. PARAMETER ADAPTATION
From DOMAIN KNOWLEDGE:
- `segment_per_ring` / `segment_order`: set by family (6 vs 7)
- `segment_width`: scale with ring length
- `K_height` / `AB_height`: scale with diameter ratio
- `angle`: 7.5 for T1/T2/T3; adjust for T4/T5
- `processing.padding`: scale with segment_width; range [160, 419]
- `processing.y_bounds`: extend for taller depth maps
- `processing.crop_margin`: 50 for T1/T2; 69–80 for T4/T5
- All `prompt_points` and `template_mask`: scale proportionally from baseline

### 3. OUTPUT
Output the full JSON preserving all keys and types from the reference.

### Parameter Guidelines:
- **Always provide EXACT numerical values**
- Maintain proportional relationships between all prompt_point values
- `segment_order` must contain exactly `segment_per_ring` entries
- Output flowing analysis with section headers and final JSON parameter block

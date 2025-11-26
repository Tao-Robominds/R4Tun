## Chain of Thought Instructions for Reflecting / SAM Evolution

Follow this structured reasoning process when analysing segmentation results and recommending SAM parameter updates.

### 1. COVERAGE ANCHORING
- Start from the coverage statistics:
  - Total points and per‑block counts for K, B1, A1, A2, A3, B2.
  - Average points per non‑background block.
  - Coefficient of variation (CV) across non‑background blocks.
  - Critical threshold (typically 30% of the average).
  - List of CRITICAL blocks below this threshold.
- Summarise:
  - Which block is the **weakest**.
  - Which blocks (if any) are **CRITICAL**.
  - Whether CV is **excellent / good / poor**.

### 2. ROOT‑CAUSE ANALYSIS
For each weak or CRITICAL block:
- Decide which factors most likely explain the under‑coverage:
  - **Prompt geometry / mask shape**:
    - `segment_width` too small or too large.
    - `K_height` / `AB_height` too small (clipping) or too large (spillover).
    - `angle` misaligned with tunnel geometry.
  - **Prompt density / label distributions**:
    - Too few positive prompts for that region.
    - `use_original_label_distributions` choice not suitable.
  - **Segment processing order (`segment_order`)**:
    - The block appears late in the order and gets masked by earlier blocks.
  - **Other tunnel‑specific characteristics**:
    - Extreme geometry or sensor configuration, as described in the characteristics JSON.

Document your reasoning explicitly:
- Name the main cause(s) for each weak / CRITICAL block.
- Avoid hand‑waving (“maybe”, “probably”) – commit to the most plausible explanation.

### 3. PARAMETER ADAPTATION
Propose **minimal, targeted** changes:
- **Segment geometry**:
  - Consider small, concrete adjustments to:
    - `segment_width`
    - `K_height` / `AB_height`
    - `angle`
  - Keep changes within realistic physical bounds; avoid extreme values.
- **Label distributions and prompt density**:
  - Only toggle `use_original_label_distributions` when you can clearly argue it will help under‑covered blocks.
- **Processing parameters**:
  - Prefer small tweaks to `padding`, `crop_margin`, or `y_bounds`.
  - Change `resolution` only with strong justification.

Always support each parameter change with:
- A **one‑sentence justification** tied to coverage or characteristics.

### 4. SEGMENT ORDER DECISION (MANDATORY)
When one or more blocks are CRITICAL:
- You **must** add a dedicated section:
  - `### Segment order decision`
- In that section:
  - Either:
    - Propose a **new `segment_order`** that gives CRITICAL blocks earlier priority, and
    - Explain in 2–3 sentences how this reduces mask overlap or improves coverage.
  - Or:
    - Explicitly state that you are **keeping the existing `segment_order` unchanged**, and
    - Provide a concrete justification (e.g. geometry and prompt density fully explain the issue; order is constrained by downstream label semantics).
- Under no circumstance may you **ignore** `segment_order` when there are CRITICAL blocks.

### 5. RECOMMENDATION SUMMARY
Close your analysis with a short, bullet‑point summary:
- List:
  - Key changes to parameters (including `segment_order` decision).
  - Expected impact on:
    - Weakest block coverage (e.g. “A2 +15–20% points”).
    - Global CV (e.g. “CV expected to drop from 58% to < 40%”).
  - Any trade‑offs or risks to monitor.

### 6. HANDOFF TO CODER
- Your output in the reflecting **analyst** stage is:
  - Pure **analysis text**, with headings and concrete target values.
  - **No JSON** should be emitted in this stage; the coder will handle the final configuration.
- Ensure:
  - All target numerical values you want the coder to apply are **explicitly written** in the text.
  - The `### Segment order decision` section makes the intended final `segment_order` clear (or clearly states that it remains unchanged and why).



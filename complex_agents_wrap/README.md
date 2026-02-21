# complex_agents_wrap

Copy of `complex_agents` for **wrap-around experiments only**. Do not integrate changes here back into `complex_agents` until one or more approaches are validated.

**Planned experiments (see `docs/journal_2026-02-21.md`):**

- **A.** Periodic Y crops — crop each wrap block with θ-periodic Y so the block is contiguous in the crop.
- **B.** Double-height depth map — duplicate the depth map in Y and run SAM on the doubled image for wrap blocks.
- **C.** Per-ring unfolding — cut each ring at groove boundaries and unfold so no block wraps; reuse angular-boundary logic.

Run preprocessing/detection/SAM from this folder when testing wrap-around handling. Data under `data/<tunnel_id>/` is shared with `complex_agents`.

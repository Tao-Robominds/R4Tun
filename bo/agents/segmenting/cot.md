# Parameter reference — SAM segmenting (BO)

**Structure:** segment_per_ring [6, 7]; segment_order must match length (templates in `knowledge.md`).

**Pixels:** segment_width [1100, 2600], K_height [1080, 2290], AB_height [3240, 6868], angle [7.5, 14.0], processing.padding [160, 419], processing.y_bounds as `[y_min, y_max]` with y_min in [3500, 5500], y_max in [11000, 15000], processing.crop_margin [50, 80].

**Derived:** rescale `prompt_points` and `template_mask` proportionally to the drivers vs baseline JSON.

**Locked:** use_original_label_distributions, processing.resolution, processing.mask_eps — see `knowledge.md`.

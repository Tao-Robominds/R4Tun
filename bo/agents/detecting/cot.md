# Parameter reference — Detecting (BO)

**Hough (ranges):** hough_threshold_oblique [20, 83], hough_threshold_horizontal [20, 83], hough_threshold_vertical [320, 980], maxLineGap_oblique [30, 100], maxLineGap_horizontal [12, 70], minLineLength_oblique [60, 240], minLineLength_horizontal [60, 220].

**Image/morph:** binary_threshold [115, 127], merge_distance [3, 8], angle_range_oblique_positive as `[low, high]` with values in [4, 12], angle_range_oblique_negative as `[low, high]` with values in [-12, -4].

**Priors:** ring_spacing_constant [1.2, 1.8], dilation_iterations [1, 3], morphological_kernel_size typically `[3,3]` or `[5,5]`.

**Locked:** resolution — see `knowledge.md`.

# Challenge map — sam4tun on 4-1 and 5-1

**Depends on:** `output/01_assumptions_output.md`
**GT:** `data/4-1.txt`, `data/5-1.txt`.

---

## Challenge table

| ID | Assumption | Status | Challenge | Evidence (4-1 / 5-1) | Class | Failure mode | Response |
|----|-----------|--------|-----------|----------------------|-------|-------------|----------|
| A1 | Diameter 5.5 m | broken | Real diameter is 7.5 m | Both tunnels use `tunnel_diameter=7.5` in irregular params | structural | All cylindrical coords wrong; r shifted by ~1 m | Use measured or configurable diameter |
| A2 | Ring spacing ~1.2 m | broken | Real spacing is 1.816 m | Both tunnels use `ring_spacing=1.816` in irregular params | parameter | Wrong ring count; wrong grid step | Use measured spacing |
| A3 | Cross-section is elliptical | stable | Holds for both tunnels | Ellipse fit succeeds on sliced clouds | — | — | — |
| A4 | Axis = MBR short edge | stable | Holds for both tunnels | MBR correctly identifies tunnel axis | — | — | — |
| A6 | Ring count = number of slices | broken | Physical ring count differs from needed slicing planes | `4-1`: 6 rings, 7 planes; `5-1`: 7 rings, 9 planes | structural | Downstream grids and spacing distorted | Separate ring count from slicing plane count |
| A7 | Physical ring count = grid count | broken | Same as A6 | Same as A6 | structural | Same as A6 | Same as A6 |
| B1 | Surface band r ∈ [2.7, 2.8] | broken | GT r range is 2.4–4.4, not 2.7–2.8 | `4-1` GT r: 2.438–4.448, old band keeps 0.04%; `5-1` GT r: 3.542–3.937, old band keeps 0%. Tuned [3.526, 4.051] keeps 96–100% | parameter | All valid surface deleted before denoising | Derive radius_min/radius_max from tunnel geometry |
| B2 | Gradient threshold 0.2 | broken | Tuned irregular value is 10.0 | Both tunnels use `gradient_threshold=10.0` | parameter | Wrong surface boundary | Tune per tunnel |
| D1 | Vertical lines = ring boundaries | broken | Uneven K spacing breaks regular vertical assumption | K centroid x jumps: `4-1` 1.07→1.25→−5.87→−1.09→−6.93→−2.01 | structural | Wrong ring centres; wrong prompt columns | Per-ring boundary detection |
| D2 | Oblique lines ±6–9° | broken | Block edge angles are tunnel-specific | Irregular geometry means edges may not be in ±6–9° band | parameter | Prompt centres move off-block | Tune or infer angle priors per tunnel |
| D4 | K height = 1079.92 mm | broken | K size varies strongly by ring | K pts/ring: `4-1` 579–13,438; `5-1` 1,010–38,926 | parameter | Prompt/mask too small or too large | Per-ring or adaptive K size |
| D5 | A/B height = 3239.77 mm | broken | A/B size varies strongly by ring | B2 pts/ring: `4-1` 5,059–33,446; `5-1` 1,026–93,906 | parameter | Templates miss real extents | Per-block-type per-ring sizes |
| D6 | Ring spacing 1.2 m in detection | broken | Real spacing 1.816 m | Both tunnels | parameter | Vertical line extrapolation wrong | Use real spacing |
| D8 | K positions are evenly spaced | broken | No single spacing phase across rings | K centroid x per ring: not monotonic, large jumps | structural | Fixed-gap detection drifts | Detect K per ring |
| E1 | segment_per_ring = 6 | broken | GT has 7 block types including A4 | Both tunnels have segment IDs 1–7 | structural | A4 never modeled; labels collapse | Expand to 7 blocks |
| E2 | Segment width = 1200 mm | broken | Width varies by ring and block | Strong per-ring point-count spread | parameter | Wrong crop/centre | Infer from data |
| E4 | Walk order K→B1→A1→A2→A3→B2 | broken | GT has A4; order differs per ring | 7-block layout; A2/A3 side relative to K varies | structural | Later prompts cascade from wrong centre | Per-ring explicit layout |
| E5 | One global walk order | broken | Order/side of A blocks changes by ring | A2/A3 side instability across rings | structural | Walk order places prompts on wrong side | Per-ring segment order |
| E6 | Fixed template vertices | broken | One polygon per type cannot cover ring-to-ring shape variation | K and B2 size spread already proves mismatch | structural | Masks under-cover or overrun | Per-ring/per-instance geometry |
| E7 | One fixed template size per family | broken | Same as E6 | Same as E6 | structural | Same as E6 | Same as E6 |
| E8 | One group offset for all rings | broken | Offsets are ring-specific | `4-1`/`5-1` irregular pattern contradicts shared offsets | structural | A2/A3/A4 centres misplaced | Per-ring offsets or per-instance centres |
| E9 | SAM can segment depth maps | broken | SAM achieves only 0.19 mIoU with GT positions | Prior experiments: geometric methods 0.77–0.99 vs SAM 0.19 | structural | SAM vision encoder cannot distinguish blocks on grayscale depth | Use geometric segmentation |
| E11 | Only pred=7 points updated | broken | Earlier failures leave block points outside update set | Denoising/detection drops support unevenly by ring | bug | True block points remain unlabeled | Broader update set and mapping recovery |
| C5 | n_segment=[10,21] for high-density band | broken | High-density ring range differs per tunnel | Ring density varies `29k-231k` (4-1), `34k-681k` (5-1) | parameter | Wrong density thresholds | Derive from data |

**Stable assumptions:** A3 (elliptical cross-section), A4 (MBR axis), A5 (degree-3 centre line), A8 (δ=0.005), C1 (support=pred≠0), C6 (resolution=0.005), C7 (pixel_to_point excludes pred=8), E10 (template mask as mask_input).

# Why Complex Staggered Tunnels Are Intractable for Manual Tuning

This document explains why segment placement in complex staggered tunnels cannot be solved by manual parameter tuning, even by a domain specialist. It motivates the automated strategy: accurate K detection + template expansion + SAM drift correction.

## The Task

Given a depth map from preprocessing, place 7 segment centres (K, B1, B2, A1-A4) per ring around the tunnel circumference. The depth map is a 2D image where the Y axis wraps around (modular arithmetic on img_height). The goal is to get each segment centre close enough to its true position that downstream SAM segmentation can find the correct boundary.

## Why a Human Engineer Cannot Solve This

### 1. Block order is not fixed across rings

In tunnel 5-1, the circumferential block order varies per ring:

| Ring | Block order (sorted by Y position) |
|------|-------------------------------------|
| 0 | A4 > B2 > K > B1 > A1 > A3 > A2 |
| 1 | K > B1 > A1 > B2 > A2 > A3 > A4 |
| 3 | A1 > A2 > A3 > A4 > B2 > K > B1 |
| 4 | A2 > A1 > B1 > K > B2 > A4 > A3 |

A human cannot predict which order a new ring will have. The order depends on the stagger rotation, which varies per ring and per tunnel.

### 2. Angular gaps between blocks are wildly irregular

In 5-1 ground truth, the gaps between adjacent blocks (in pixels) within a single tunnel:

- Minimum gap: **19px** (Ring 2, between B1 and A1)
- Maximum gap: **1864px** (Ring 4, wrap-around gap)
- Expected uniform gap: 4712 / 7 = **673px**

The actual gaps range from 3% to 277% of the expected uniform value. No single "typical spacing" exists that a human could set as a default.

### 3. The stagger phase flips with anomalies

The B1/B2 offset sign pattern shows a phase structure:

- Rings 0-3: B1 positive (ahead of K), B2 negative (behind K)
- Rings 4-6: B1 negative, B2 positive (flipped)
- Ring 1 is anomalous: B2 is also positive (1390px ahead of K), breaking the phase pattern

The flip point, phase length, and anomalies are different for each tunnel. A human would need to inspect the depth map for each ring individually to determine the correct phase -- but block boundaries are not visually distinct in the sparse depth map.

### 4. 42 coupled parameters with wrap-around

For a 7-ring tunnel with 7 segments per ring, per-ring offset tuning requires 42 continuous parameters (6 non-K offsets x 7 rings). These parameters:

- Range from -2400 to +2400 pixels each
- Interact through wrap-around arithmetic: Y = (K_Y + offset) % img_height
- Must satisfy spacing constraints (no two blocks closer than ~147px)
- Have no closed-form solution

No human can mentally optimize a 42-dimensional space with modular arithmetic constraints.

### 5. The depth map provides no visual guidance

The depth map (`depth_map_outlier.npy`) is a sparse, noisy image where:

- Only ~15k-20k pixels out of ~4.7M are valid (non-NaN)
- Block boundaries appear as faint, irregular depth discontinuities
- K-block joints (oblique lines) are detectable by Hough transforms but AB-block boundaries are not reliably visible
- A human looking at the depth map cannot tell where "A3 should be relative to K" on a given ring

### 6. Every tunnel is different

| Property | Tunnel 4-1 | Tunnel 5-1 |
|----------|-----------|-----------|
| Segments per ring | 6 | 7 |
| Ring count | 10 | 7 |
| Image height | varies | 4712px |
| K height (mm) | 1079.92 | 1079.92 |
| Stagger pattern | different | different |
| Block order per ring | different | different |

Parameters tuned for one tunnel do not transfer to another. Even tunnels of the same type (complex staggered) have different ring counts, image dimensions, and stagger arrangements.

## The Automated Solution

The intractability of manual tuning motivates a three-stage automated decomposition:

1. **K detection (geometric method):** Hough line detection finds K-block joints. This is the only stage that needs high accuracy. It is ring-count-independent (works for any number of rings) and already achieves ~56px drift. Line detection parameters are BO-tunable and transferable across tunnels of the same type.

2. **Template expansion:** Places non-K blocks at physically-derived angular positions relative to K. Uses block dimensions (k_height, ab_height) and auto-rotates per ring via intersection scoring. Ring-count-independent. Produces ~500-800px drift, which is intentionally imprecise.

3. **SAM drift correction:** SAM crops a large window around each estimated block position (half-height ~780px for K, ~1860px for A/B blocks) and finds the actual segment boundary from image content. As long as the true block falls inside the crop window, SAM corrects the template drift. The final mask position is image-driven, not parameter-driven.

This decomposition replaces an intractable 42-dimensional manual tuning problem with:
- A well-posed line detection problem (BO-tunable, ~14 params, ring-count-independent)
- A physics-based template (3 params: k_height, ab_height, stagger_shift)
- An image-driven correction step (SAM, no parameters to tune for placement)

Each stage handles what it is good at: Hough transforms for line geometry, physics for block spacing, and SAM for pixel-level boundaries.

## Appendix: Experimental Validation

*(To be added after running template vs offset comparison from the 0.805 geometric baseline.)*

"""GT-free tunnel axis orientation using ring metadata."""

import numpy as np
from scipy.stats import spearmanr


def orient_centers_by_ring(
    points_xyz,
    ring,
    center1,
    center2,
    min_points_per_ring=10,
):
    """
    Orient center1 -> center2 along increasing ring numbers (shield-machine forward).

    Ring ids are construction metadata, not segmentation ground truth. Projects each
    ring's XY centroid onto the candidate axis and swaps endpoints when Spearman
    correlation is negative.

    Returns:
        center1, center2, rho, swapped
    """
    center1 = np.asarray(center1, dtype=float)
    center2 = np.asarray(center2, dtype=float)
    direction = center2 - center1
    norm = np.linalg.norm(direction)
    if norm == 0:
        return center1, center2, float("nan"), False

    unit = direction / norm
    ring_ids = []
    projs = []
    for r in np.unique(ring):
        mask = ring == r
        if mask.sum() < min_points_per_ring:
            continue
        centroid = points_xyz[mask, :2].mean(axis=0)
        ring_ids.append(r)
        projs.append(np.dot(centroid - center1, unit))

    if len(ring_ids) < 2:
        return center1, center2, float("nan"), False

    rho, _ = spearmanr(ring_ids, projs)
    swapped = rho < 0
    if swapped:
        center1, center2 = center2.copy(), center1.copy()
    return center1, center2, rho, swapped

"""Extract a 2-D footprint polygon from predicted building point clouds.

The primary method is a convex hull of near-ground points.  Convex hulls are
always correct for convex footprint types (rectangle, irregular); they
over-approximate concave shapes (L, T, courtyard).  All decisions are recorded
in the returned ``decisions`` dict so the caller can report them honestly.
"""

from __future__ import annotations

import numpy as np

# Reuse the monotone-chain convex hull already in the project.
from synthetic_city.core.mesh import _convex_hull  # noqa: PLC2701


def extract_footprint_convex(
    xyz: np.ndarray,
    base_frac: float = 0.15,
    min_pts: int = 3,
) -> tuple[list[tuple[float, float]], str]:
    """Return the convex hull of near-ground XY points and a source tag.

    Args:
        xyz:        (N, 3) float32 building points (in world coordinates).
        base_frac:  fraction of the height range considered "near-ground".
                    Points with z ≤ z_min + base_frac * (z_max - z_min) are used.
        min_pts:    minimum number of hull vertices to accept; if the hull is
                    smaller the whole-cloud hull is tried; if still too small
                    an empty list is returned and the caller should use the
                    metadata footprint.

    Returns:
        (hull_xy, source_tag)
        hull_xy:    list of (x, y) tuples in CCW order, may be empty.
        source_tag: "points_base_hull", "points_full_hull", or "empty".
    """
    if len(xyz) < min_pts:
        return [], "empty"

    z_arr = xyz[:, 2]
    z_min = float(z_arr.min())
    z_max = float(z_arr.max())
    z_thresh = z_min + base_frac * max(z_max - z_min, 1e-6)

    base_mask = z_arr <= z_thresh
    base_pts = xyz[base_mask]

    def _hull(pts: np.ndarray) -> list[tuple[float, float]]:
        return _convex_hull([(float(p[0]), float(p[1])) for p in pts])

    hull = _hull(base_pts)
    if len(hull) >= min_pts:
        return hull, "points_base_hull"

    hull = _hull(xyz)
    if len(hull) >= min_pts:
        return hull, "points_full_hull"

    return [], "empty"

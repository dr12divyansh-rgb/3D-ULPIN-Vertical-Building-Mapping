"""Deterministic LOD2 SurfaceMesh builders for flat, hip, and shed roofs.

All functions return a ``SurfaceMesh`` with properly tagged surfaces
(WallSurface, RoofSurface, GroundSurface).  Every mesh produced here is
watertight when the input polygon is valid.

Design notes
------------
* Flat roof  — works for any convex N-gon footprint.
* Hip roof   — requires exactly 4 vertices (rectangle, possibly rotated).
               The ridge is inset from each short end by W/2 where W is the
               shorter dimension, consistent with the dataset's
               ``roof_slope = rise / min(width, length)`` convention.
* Shed roof  — requires exactly 4 vertices.  The high long side is inferred
               from the mean z of building points near each long side; if no
               points are provided it defaults to the first long side being
               the high side.
* Hip/shed face counts match the ground-truth LOD2 for dataset buildings:
    hip  → 8 wall + 6 roof + 2 ground = 16 total
    shed → 12 wall + 2 roof + 2 ground = 16 total
"""

from __future__ import annotations

import math

import numpy as np

from synthetic_city.core.mesh import SurfaceMesh


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _side_lengths(poly4: list) -> list[float]:
    return [
        math.sqrt((poly4[i][0] - poly4[(i + 1) % 4][0]) ** 2
                  + (poly4[i][1] - poly4[(i + 1) % 4][1]) ** 2)
        for i in range(4)
    ]


def _short_long_sides(poly4: list):
    """Return (short_side_indices, long_side_indices) for a 4-vertex polygon.

    Each 'side' is represented as a pair (start_vertex_idx, end_vertex_idx).
    short_side_indices contains the two shorter opposite sides;
    long_side_indices contains the two longer opposite sides.
    """
    sl = _side_lengths(poly4)
    avg_02 = (sl[0] + sl[2]) / 2
    avg_13 = (sl[1] + sl[3]) / 2
    if avg_02 <= avg_13:
        return [(0, 1), (2, 3)], [(1, 2), (3, 0)]
    return [(1, 2), (3, 0)], [(0, 1), (2, 3)]


def _add_quad(mesh: SurfaceMesh, surface: str, p0, p1, p2, p3) -> None:
    """Fan-triangulate a quad (p0,p1,p2,p3) into two triangles."""
    mesh.add_triangle(surface, p0, p1, p2)
    mesh.add_triangle(surface, p0, p2, p3)


# --------------------------------------------------------------------------- #
# Public builders                                                              #
# --------------------------------------------------------------------------- #

def build_flat_roof_mesh(
    footprint_xy: list[tuple[float, float]],
    base_z: float,
    top_z: float,
) -> SurfaceMesh:
    """Build a flat-roofed solid for an arbitrary convex N-gon footprint.

    The resulting mesh is watertight (closed 2-manifold) for any valid convex
    polygon: N wall quads + 1 ground polygon + 1 roof polygon.
    """
    mesh = SurfaceMesh()
    n = len(footprint_xy)
    if n < 3:
        return mesh

    mesh.add_polygon("GroundSurface", [(x, y, base_z) for x, y in footprint_xy])
    mesh.add_polygon("RoofSurface",   [(x, y, top_z)  for x, y in footprint_xy])

    for i in range(n):
        a = footprint_xy[i]
        b = footprint_xy[(i + 1) % n]
        _add_quad(mesh, "WallSurface",
                  (a[0], a[1], base_z), (b[0], b[1], base_z),
                  (b[0], b[1], top_z),  (a[0], a[1], top_z))

    return mesh


def build_hip_roof_mesh(
    footprint_xy: list[tuple[float, float]],
    base_z: float,
    eave_z: float,
    peak_z: float,
) -> SurfaceMesh:
    """Build a hip-roofed solid for a 4-vertex rectangle footprint.

    The hip ridge runs parallel to the longer axis, inset from each short end
    by W/2 (where W = length of the shorter rectangle side).

    Face count: 8 wall + 6 roof + 2 ground = 16 (matches GT).
    Falls back to ``build_flat_roof_mesh(..., top_z=peak_z)`` if the polygon
    is not exactly 4 vertices.
    """
    n = len(footprint_xy)
    if n != 4:
        return build_flat_roof_mesh(footprint_xy, base_z, peak_z)

    poly = list(footprint_xy)
    mesh = SurfaceMesh()

    # Ground (2 triangles for a quad)
    mesh.add_polygon("GroundSurface", [(x, y, base_z) for x, y in poly])

    # Walls: all four sides from base_z to eave_z
    for i in range(4):
        a = poly[i]; b = poly[(i + 1) % 4]
        _add_quad(mesh, "WallSurface",
                  (a[0], a[1], base_z), (b[0], b[1], base_z),
                  (b[0], b[1], eave_z), (a[0], a[1], eave_z))

    # Identify short/long sides
    short_sides, long_sides = _short_long_sides(poly)
    # short_sides[0] = (i, j) → vertices poly[i] and poly[j] form one short end
    # The midpoints of the two short ends define the long axis.
    s0i, s0j = short_sides[0]
    s1i, s1j = short_sides[1]
    m0 = ((poly[s0i][0] + poly[s0j][0]) / 2, (poly[s0i][1] + poly[s0j][1]) / 2)
    m1 = ((poly[s1i][0] + poly[s1j][0]) / 2, (poly[s1i][1] + poly[s1j][1]) / 2)

    dx, dy = m1[0] - m0[0], m1[1] - m0[1]
    long_len = math.sqrt(dx * dx + dy * dy)
    if long_len < 1e-9:
        return build_flat_roof_mesh(footprint_xy, base_z, peak_z)
    lx, ly = dx / long_len, dy / long_len

    sl = _side_lengths(poly)
    short_w = (sl[short_sides[0][0]] + sl[short_sides[1][0]]) / 2  # avg of the two short sides
    # Actually each element of short_sides is (i, j) where i is the vertex index,
    # so the side length is sl[i] where i = short_sides[k][0].
    short_w = (sl[short_sides[0][0]] + sl[short_sides[1][0]]) / 2

    inset = short_w / 2.0
    r0 = (m0[0] + inset * lx, m0[1] + inset * ly, peak_z)  # ridge end near short side 0
    r1 = (m1[0] - inset * lx, m1[1] - inset * ly, peak_z)  # ridge end near short side 1

    # Short-end triangular roof faces
    a0 = (poly[s0i][0], poly[s0i][1], eave_z)
    b0 = (poly[s0j][0], poly[s0j][1], eave_z)
    mesh.add_triangle("RoofSurface", a0, b0, r0)

    a1 = (poly[s1i][0], poly[s1i][1], eave_z)
    b1 = (poly[s1j][0], poly[s1j][1], eave_z)
    mesh.add_triangle("RoofSurface", a1, b1, r1)

    # Long trapezoidal roof faces
    l0i, l0j = long_sides[0]
    l1i, l1j = long_sides[1]
    la0 = (poly[l0i][0], poly[l0i][1], eave_z)
    lb0 = (poly[l0j][0], poly[l0j][1], eave_z)
    la1 = (poly[l1i][0], poly[l1i][1], eave_z)
    lb1 = (poly[l1j][0], poly[l1j][1], eave_z)

    # Each long side: trapezoid eave → eave → ridge_end → ridge_end
    # Vertex order: la0, lb0, and the two corresponding ridge ends.
    # The ridge ends matching each long side depend on adjacency:
    # long_side 0 runs from vertex l0i → l0j; those vertices are adjacent to
    # short sides whose ridge ends are r0 and r1.
    # We need to determine which ridge end is "near" each long side endpoint.
    # l0i is adjacent to one short side; check which short side shares that vertex.
    if l0i == s0i or l0i == s0j:
        r_near_l0i = r0
    else:
        r_near_l0i = r1
    if l0j == s0i or l0j == s0j:
        r_near_l0j = r0
    else:
        r_near_l0j = r1

    _add_quad(mesh, "RoofSurface", la0, lb0, r_near_l0j, r_near_l0i)

    if l1i == s0i or l1i == s0j:
        r_near_l1i = r0
    else:
        r_near_l1i = r1
    if l1j == s0i or l1j == s0j:
        r_near_l1j = r0
    else:
        r_near_l1j = r1

    _add_quad(mesh, "RoofSurface", la1, lb1, r_near_l1j, r_near_l1i)

    return mesh


def build_shed_roof_mesh(
    footprint_xy: list[tuple[float, float]],
    base_z: float,
    eave_z: float,
    peak_z: float,
    xyz_building: "np.ndarray | None" = None,
) -> SurfaceMesh:
    """Build a shed-roofed solid for a 4-vertex rectangle footprint.

    The shed slope rises from one long side (low eave = eave_z) to the
    opposite long side (high eave = peak_z).  Which long side is high is
    determined by comparing the mean z of nearby building points; if no points
    are available the first long side (side index 1 or 2 depending on aspect
    ratio) is treated as the high side.

    Face count: 12 wall + 2 roof + 2 ground = 16 (matches GT).
    Falls back to ``build_flat_roof_mesh(..., top_z=peak_z)`` for non-quad.

    Wall breakdown:
        - low long side: 2 tri (rectangle base → eave_z)
        - high long side lower: 2 tri (base → eave_z)
        - high long side upper: 2 tri (eave_z → peak_z)
        - short side 1: 2 tri (rectangle) + 1 tri (gable)
        - short side 2: 2 tri (rectangle) + 1 tri (gable)
        = 12 wall faces
    """
    n = len(footprint_xy)
    if n != 4:
        return build_flat_roof_mesh(footprint_xy, base_z, peak_z)

    poly = list(footprint_xy)
    mesh = SurfaceMesh()

    short_sides, long_sides = _short_long_sides(poly)

    # Determine which long side is "high" using point z distribution
    l0i, l0j = long_sides[0]
    l1i, l1j = long_sides[1]
    m_l0 = ((poly[l0i][0] + poly[l0j][0]) / 2, (poly[l0i][1] + poly[l0j][1]) / 2)
    m_l1 = ((poly[l1i][0] + poly[l1j][0]) / 2, (poly[l1i][1] + poly[l1j][1]) / 2)

    high_is_l0 = True  # default
    if xyz_building is not None and len(xyz_building) >= 2:
        pts_xy = xyz_building[:, :2].astype(np.float64)
        arr_l0 = np.array(m_l0, dtype=np.float64)
        arr_l1 = np.array(m_l1, dtype=np.float64)
        d0 = np.sum((pts_xy - arr_l0) ** 2, axis=1)
        d1 = np.sum((pts_xy - arr_l1) ** 2, axis=1)
        near_l0 = xyz_building[d0 < d1]
        near_l1 = xyz_building[d1 <= d0]
        if len(near_l0) > 0 and len(near_l1) > 0:
            high_is_l0 = float(near_l0[:, 2].mean()) > float(near_l1[:, 2].mean())

    if high_is_l0:
        hi_side, lo_side = long_sides[0], long_sides[1]
    else:
        hi_side, lo_side = long_sides[1], long_sides[0]

    hi_i, hi_j = hi_side   # vertex indices of the high long side
    lo_i, lo_j = lo_side   # vertex indices of the low long side

    # Identify the two short sides by their vertex indices
    # short_sides[0] = (sa, sb), short_sides[1] = (sc, sd)
    # We label them relative to hi/lo sides for gable construction.
    # Each short side connects one vertex of the high side to one of the low side.
    # Find which vertex of each short side is on the high long side.
    hi_set = {hi_i, hi_j}
    ss0 = short_sides[0]; ss1 = short_sides[1]
    # For each short side: the vertex in hi_set is the "high corner"
    ss0_hi = ss0[0] if ss0[0] in hi_set else ss0[1]
    ss0_lo = ss0[1] if ss0[0] in hi_set else ss0[0]
    ss1_hi = ss1[0] if ss1[0] in hi_set else ss1[1]
    ss1_lo = ss1[1] if ss1[0] in hi_set else ss1[0]

    # --- Ground -------------------------------------------------------
    mesh.add_polygon("GroundSurface", [(x, y, base_z) for x, y in poly])

    # --- Low long side wall (base → eave_z) ---------------------------
    a = poly[lo_i]; b = poly[lo_j]
    _add_quad(mesh, "WallSurface",
              (a[0], a[1], base_z), (b[0], b[1], base_z),
              (b[0], b[1], eave_z), (a[0], a[1], eave_z))

    # --- High long side: lower (base → eave_z) + upper (eave_z → peak_z) --
    c = poly[hi_i]; d = poly[hi_j]
    _add_quad(mesh, "WallSurface",
              (c[0], c[1], base_z), (d[0], d[1], base_z),
              (d[0], d[1], eave_z), (c[0], c[1], eave_z))
    _add_quad(mesh, "WallSurface",
              (c[0], c[1], eave_z), (d[0], d[1], eave_z),
              (d[0], d[1], peak_z), (c[0], c[1], peak_z))

    # --- Short side 1: rectangle (base → eave) + gable triangle ------
    p_lo0 = poly[ss0_lo]; p_hi0 = poly[ss0_hi]
    _add_quad(mesh, "WallSurface",
              (p_lo0[0], p_lo0[1], base_z), (p_hi0[0], p_hi0[1], base_z),
              (p_hi0[0], p_hi0[1], eave_z), (p_lo0[0], p_lo0[1], eave_z))
    # Gable: low-corner@eave, hi-corner@eave → hi-corner@peak
    mesh.add_triangle("WallSurface",
                      (p_lo0[0], p_lo0[1], eave_z),
                      (p_hi0[0], p_hi0[1], eave_z),
                      (p_hi0[0], p_hi0[1], peak_z))

    # --- Short side 2: rectangle (base → eave) + gable triangle ------
    p_lo1 = poly[ss1_lo]; p_hi1 = poly[ss1_hi]
    _add_quad(mesh, "WallSurface",
              (p_lo1[0], p_lo1[1], base_z), (p_hi1[0], p_hi1[1], base_z),
              (p_hi1[0], p_hi1[1], eave_z), (p_lo1[0], p_lo1[1], eave_z))
    mesh.add_triangle("WallSurface",
                      (p_lo1[0], p_lo1[1], eave_z),
                      (p_hi1[0], p_hi1[1], eave_z),
                      (p_hi1[0], p_hi1[1], peak_z))

    # --- Roof: single inclined quad -----------------------------------
    # Low eave: lo_i@eave, lo_j@eave; high eave: hi_i@peak, hi_j@peak.
    # Winding: lo_i adj ss0_lo corner → hi_i is ss0_hi (or ss1_hi).
    # Match corners: lo_i is adjacent to which hi corner?
    # lo_i is a vertex shared with one short side; that short side's hi corner is hi_i or hi_j.
    if ss0_lo == lo_i:
        hi_near_lo_i = ss0_hi
        hi_near_lo_j = ss1_hi
    else:
        hi_near_lo_i = ss1_hi
        hi_near_lo_j = ss0_hi

    r_lo_i = (poly[lo_i][0], poly[lo_i][1], eave_z)
    r_lo_j = (poly[lo_j][0], poly[lo_j][1], eave_z)
    r_hi_i = (poly[hi_near_lo_i][0], poly[hi_near_lo_i][1], peak_z)
    r_hi_j = (poly[hi_near_lo_j][0], poly[hi_near_lo_j][1], peak_z)

    _add_quad(mesh, "RoofSurface", r_lo_i, r_lo_j, r_hi_j, r_hi_i)

    return mesh

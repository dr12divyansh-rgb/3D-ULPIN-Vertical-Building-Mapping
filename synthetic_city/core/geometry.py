"""Procedural footprint + roof geometry (pure Python, no bpy).

This module contains the *mathematical* definition of every footprint shape
and roof type. It is the deterministic core of the procedural building
generator: given a seeded ``random.Random`` instance it always produces the
same polygons.

All footprint polygons are defined in local coordinates, centred on the
origin, with the building "width" along the X axis and "length" along the Y
axis (length >= width for rectangle footprints so that gable/hip ridges are
always oriented along Y).
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- #
# 2D helpers
# --------------------------------------------------------------------------- #


def _signed_area2(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a


def polygon_area(pts):
    return abs(_signed_area2(pts)) / 2.0


def polygon_centroid(pts):
    """Centroid of a simple polygon via the shoelace formula."""
    n = len(pts)
    a2 = _signed_area2(pts)
    if abs(a2) < 1e-12:
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        f = x1 * y2 - x2 * y1
        cx += (x1 + x2) * f
        cy += (y1 + y2) * f
    return (cx / (3 * a2), cy / (3 * a2))


def _point_in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def triangulate_polygon(points):
    """Ear-clip triangulation of a *simple* polygon (concave allowed, no holes).

    ``points`` is a list of (x, y) in boundary order. Returns a list of
    (i, j, k) index triples referencing ``points`` directly.
    """
    n = len(points)
    if n < 3:
        return []
    pts = [(float(x), float(y)) for (x, y) in points]
    if _signed_area2(pts) >= 0:
        sign = 1.0
    else:
        sign = -1.0

    remaining = list(range(n))
    tris = []
    guard = 0
    while len(remaining) > 3 and guard < n * 100:
        guard += 1
        progressed = False
        m = len(remaining)
        for k in range(m):
            i = remaining[(k - 1) % m]
            j = remaining[k]
            l = remaining[(k + 1) % m]
            ax, ay = pts[i]
            bx, by = pts[j]
            cx, cy = pts[l]
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if cross * sign <= 1e-12:
                continue
            if any(
                _point_in_triangle(pts[p], (ax, ay), (bx, by), (cx, cy))
                for p in remaining
                if p != i and p != j and p != l
            ):
                continue
            tris.append((i, j, l))
            remaining.pop(k)
            progressed = True
            break
        if not progressed:
            break
    if len(remaining) == 3:
        tris.append((remaining[0], remaining[1], remaining[2]))
    return tris


def triangulate_planar_polygon(pts3d):
    """Fan triangulation for a convex planar 3D polygon."""
    out = []
    for i in range(1, len(pts3d) - 1):
        out.append((pts3d[0], pts3d[i], pts3d[i + 1]))
    return out


# --------------------------------------------------------------------------- #
# Footprint generators
# --------------------------------------------------------------------------- #


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _recenter(points):
    cx, cy = polygon_centroid(points)
    return [(x - cx, y - cy) for (x, y) in points]


def _rect_ring(w, l):
    hw, hl = w / 2.0, l / 2.0
    return [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]


def generate_footprint(rng, ftype, width, length):
    """Generate a footprint dict for the requested type.

    Returns::
        {
          "type": str,
          "width": float,   # X extent of the local polygon
          "length": float,  # Y extent of the local polygon
          "outer": [(x, y), ...],   # simple polygon ring (local, centred)
          "hole": [(x, y), ...] | None,
          "rects": [(x0, y0, x1, y1), ...],  # rectangle decomposition (courtyard)
        }

    For every type except ``courtyard``, meshing uses ``outer`` directly
    (``rects`` is empty). For ``courtyard``, meshing uses ``rects`` (four bars
    forming a ring); ``outer``/``hole`` are kept as the ground-truth outline.
    """
    w = float(width)
    l = float(length)

    if ftype == "rectangle":
        if w > l:
            w, l = l, w
        outer = _rect_ring(w, l)
        return {"type": ftype, "width": w, "length": l, "outer": outer,
                "hole": None, "rects": []}

    if ftype == "L":
        leg_x = w * rng.uniform(0.35, 0.60)
        leg_y = l * rng.uniform(0.35, 0.60)
        hw, hl = w / 2.0, l / 2.0
        outer = [
            (-hw, -hl), (hw, -hl), (hw, -hl + leg_y),
            (hw - leg_x, -hl + leg_y), (hw - leg_x, hl), (-hw, hl),
        ]
        outer = _recenter(outer)
        mnx, mny, mxx, mxy = _bbox(outer)
        return {"type": ftype, "width": mxx - mnx, "length": mxy - mny,
                "outer": outer, "hole": None, "rects": []}

    if ftype == "T":
        bar_h = l * rng.uniform(0.25, 0.40)
        stem_w = w * rng.uniform(0.30, 0.50)
        hw, hl = w / 2.0, l / 2.0
        hs = stem_w / 2.0
        outer = [
            (-hs, -hl), (hs, -hl), (hs, hl - bar_h), (hw, hl - bar_h),
            (hw, hl), (-hw, hl), (-hw, hl - bar_h), (-hs, hl - bar_h),
        ]
        outer = _recenter(outer)
        mnx, mny, mxx, mxy = _bbox(outer)
        return {"type": ftype, "width": mxx - mnx, "length": mxy - mny,
                "outer": outer, "hole": None, "rects": []}

    if ftype == "courtyard":
        hole_w = w * rng.uniform(0.40, 0.60)
        hole_l = l * rng.uniform(0.40, 0.60)
        outer = _rect_ring(w, l)
        hw2, hl2 = hole_w / 2.0, hole_l / 2.0
        hole = [(-hw2, -hl2), (hw2, -hl2), (hw2, hl2), (-hw2, hl2)]
        hw, hl = w / 2.0, l / 2.0
        rects = [
            (-hw, hl2, hw, hl),     # north bar
            (-hw, -hl, hw, -hl2),   # south bar
            (-hw, -hl2, -hw2, hl2),  # west bar
            (hw2, -hl2, hw, hl2),   # east bar
        ]
        return {"type": ftype, "width": w, "length": l, "outer": outer,
                "hole": hole, "rects": rects}

    if ftype == "irregular":
        n = rng.randint(5, 8)
        pts = []
        for i in range(n):
            ang = (2 * math.pi * i / n) + rng.uniform(-0.3, 0.3)
            rx = rng.uniform(0.55, 1.0)
            ry = rng.uniform(0.55, 1.0)
            pts.append((math.cos(ang) * (w / 2.0) * rx,
                        math.sin(ang) * (l / 2.0) * ry))
        pts = sorted(pts, key=lambda p: math.atan2(p[1], p[0]))
        outer = _recenter(pts)
        mnx, mny, mxx, mxy = _bbox(outer)
        return {"type": ftype, "width": mxx - mnx, "length": mxy - mny,
                "outer": outer, "hole": None, "rects": []}

    raise ValueError(f"unknown footprint type: {ftype}")


# --------------------------------------------------------------------------- #
# Roof generators (rectangle footprints only)
# --------------------------------------------------------------------------- #


def roof_polygons(roof_type, w, l, wall_height, rise):
    """Return tagged roof polygons for a rectangle roof.

    The building walls reach ``wall_height`` (H). The roof sits on top and
    rises by ``rise`` (R) above the eave line, except for ``flat`` where
    ``rise`` is ignored (the extrusion already closes the top).

    Returns a list of ``(polygon, surface_type)`` tuples where ``polygon`` is a
    list of (x, y, z) triples and ``surface_type`` is ``"RoofSurface"`` (the
    sloped/planar roof planes) or ``"WallSurface"`` (vertical gable-end walls).
    Rectangle is centred on the origin, width along X, length along Y
    (length >= width).
    """
    hw, hl = w / 2.0, l / 2.0
    H = float(wall_height)
    R = float(rise)
    if R < 1e-9:
        return []

    if roof_type == "flat":
        return []

    if roof_type == "gable":
        return [
            ([(hw, -hl, H), (0, -hl, H + R), (0, hl, H + R), (hw, hl, H)], "RoofSurface"),
            ([(-hw, -hl, H), (-hw, hl, H), (0, hl, H + R), (0, -hl, H + R)], "RoofSurface"),
            ([(hw, hl, H), (-hw, hl, H), (0, hl, H + R)], "WallSurface"),
            ([(-hw, -hl, H), (hw, -hl, H), (0, -hl, H + R)], "WallSurface"),
        ]

    if roof_type == "hip":
        h = min(w, l) * 0.5 * 0.6  # hip run; ridge shorter than footprint
        A = (-hw, -hl, H)
        B = (hw, -hl, H)
        C = (hw, hl, H)
        D = (-hw, hl, H)
        R1 = (0.0, -hl + h, H + R)
        R2 = (0.0, hl - h, H + R)
        return [
            ([A, B, R1], "RoofSurface"),
            ([C, D, R2], "RoofSurface"),
            ([D, A, R1, R2], "RoofSurface"),
            ([B, C, R2, R1], "RoofSurface"),
        ]

    if roof_type == "shed":
        # mono-pitch rising toward +X: a single slope, two gable-end triangles,
        # and a vertical high-side wall (from eave H up to ridge H+R).
        return [
            ([(hw, -hl, H + R), (hw, hl, H + R), (-hw, hl, H), (-hw, -hl, H)], "RoofSurface"),
            ([(hw, hl, H + R), (hw, hl, H), (-hw, hl, H)], "WallSurface"),
            ([(hw, -hl, H + R), (-hw, -hl, H), (hw, -hl, H)], "WallSurface"),
            ([(hw, -hl, H), (hw, hl, H), (hw, hl, H + R), (hw, -hl, H + R)], "WallSurface"),
        ]

    raise ValueError(f"unknown roof type: {roof_type}")

"""Structured surface mesh model (pure Python, no bpy).

A ``SurfaceMesh`` stores an indexed triangle mesh in which every face is
assigned to exactly one *semantic surface type* (``WallSurface``,
``RoofSurface``, ``GroundSurface``). This is the canonical structured
representation of LOD1 / LOD2 building geometry — it is not an anonymous
triangle soup — and it round-trips through a group-tagged OBJ file.
"""

from __future__ import annotations

import math
from pathlib import Path

from .surfaces import SURFACE_TYPE_NAMES


def _tri_area3(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def _tri_area2_xy(a, b, c):
    """2x signed area of a triangle projected to the XY plane."""
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


class SurfaceMesh:
    def __init__(self):
        self.vertices = []  # list of [x, y, z]
        self.surfaces = {}  # surface type name -> list of [i, j, k]
        self._vidx = {}

    # ---- construction -------------------------------------------------- #

    def add_triangle(self, surface_type, a, b, c):
        ia = self._vertex(a)
        ib = self._vertex(b)
        ic = self._vertex(c)
        self.surfaces.setdefault(surface_type, []).append([ia, ib, ic])
        return ia, ib, ic

    def _vertex(self, p):
        key = (round(p[0], 6), round(p[1], 6), round(p[2], 6))
        idx = self._vidx.get(key)
        if idx is None:
            idx = len(self.vertices)
            self._vidx[key] = idx
            self.vertices.append([round(p[0], 6), round(p[1], 6), round(p[2], 6)])
        return idx

    def add_polygon(self, surface_type, pts):
        """Fan-triangulate a (convex) polygon and add it as one surface group."""
        for i in range(1, len(pts) - 1):
            self.add_triangle(surface_type, pts[0], pts[i], pts[i + 1])

    # ---- queries ------------------------------------------------------- #

    def all_faces(self):
        out = []
        for faces in self.surfaces.values():
            out.extend(faces)
        return out

    def num_faces(self):
        return sum(len(f) for f in self.surfaces.values())

    def num_vertices(self):
        return len(self.vertices)

    def triangles(self):
        """All faces as 3D coordinate triples."""
        out = []
        for faces in self.surfaces.values():
            for (i, j, k) in faces:
                out.append((tuple(self.vertices[i]), tuple(self.vertices[j]),
                            tuple(self.vertices[k])))
        return out

    def bbox(self):
        if not self.vertices:
            return None
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def height(self):
        bb = self.bbox()
        return 0.0 if bb is None else bb[5]

    def area_by_type(self):
        areas = {}
        for name, faces in self.surfaces.items():
            a = 0.0
            for (i, j, k) in faces:
                a += _tri_area3(self.vertices[i], self.vertices[j], self.vertices[k])
            areas[name] = a
        return areas

    def total_area(self):
        return sum(self.area_by_type().values())

    def footprint_area(self):
        """Area of the ground-contact surface (projected to XY).

        Falls back to the 2D convex-hull area of all vertices if the mesh has
        no ``GroundSurface`` (e.g. a predicted mesh without an explicit floor).
        """
        if "GroundSurface" in self.surfaces:
            a = 0.0
            for (i, j, k) in self.surfaces["GroundSurface"]:
                a += 0.5 * _tri_area2_xy(self.vertices[i], self.vertices[j],
                                         self.vertices[k])
            return a
        pts = [(v[0], v[1]) for v in self.vertices]
        return _convex_hull_area(pts)

    def footprint_polygon(self):
        """2D convex hull of the XY-projected vertices (for IoU rasterisation)."""
        pts = [(v[0], v[1]) for v in self.vertices]
        return _convex_hull(pts)

    def zero_area_face_count(self, eps=1e-9):
        n = 0
        for faces in self.surfaces.values():
            for (i, j, k) in faces:
                if _tri_area3(self.vertices[i], self.vertices[j], self.vertices[k]) <= eps:
                    n += 1
        return n

    def has_finite_coords(self):
        for v in self.vertices:
            if not all(math.isfinite(c) for c in v):
                return False
        return True

    def is_watertight(self):
        """Closed 2-manifold check: every undirected edge shared by exactly 2 faces."""
        edge_count = {}
        for faces in self.surfaces.values():
            for (i, j, k) in faces:
                for a, b in ((i, j), (j, k), (k, i)):
                    e = (a, b) if a < b else (b, a)
                    edge_count[e] = edge_count.get(e, 0) + 1
        return len(edge_count) > 0 and all(c == 2 for c in edge_count.values())

    def non_manifold_edge_count(self):
        edge_count = {}
        for faces in self.surfaces.values():
            for (i, j, k) in faces:
                for a, b in ((i, j), (j, k), (k, i)):
                    e = (a, b) if a < b else (b, a)
                    edge_count[e] = edge_count.get(e, 0) + 1
        return sum(1 for c in edge_count.values() if c != 2)

    # ---- serialisation ------------------------------------------------- #

    def to_dict(self):
        return {"vertices": self.vertices, "surfaces": self.surfaces}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.vertices = [list(v) for v in d["vertices"]]
        m.surfaces = {k: [list(f) for f in v] for k, v in d["surfaces"].items()}
        m._vidx = {tuple(v): i for i, v in enumerate(m.vertices)}
        return m

    def to_obj(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for v in self.vertices:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        for name in SURFACE_TYPE_NAMES:
            if name not in self.surfaces:
                continue
            lines.append(f"g {name}")
            for (i, j, k) in self.surfaces[name]:
                lines.append(f"f {i + 1} {j + 1} {k + 1}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    @classmethod
    def from_obj(cls, path):
        m = cls()
        verts = []
        current_group = "WallSurface"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "v":
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "g":
                    current_group = parts[1]
                elif parts[0] == "f":
                    idx = []
                    for tok in parts[1:]:
                        tok = tok.split("/")[0]
                        idx.append(int(tok))
                    # resolve negative (relative) indices
                    idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                    for i in range(1, len(idx) - 1):
                        m.surfaces.setdefault(current_group, []).append(
                            [idx[0], idx[i], idx[i + 1]])
        m.vertices = verts
        m._vidx = {tuple(v): i for i, v in enumerate(verts)}
        return m


# --------------------------------------------------------------------------- #
# 2D convex hull (monotone chain) — used for footprint fallback + IoU
# --------------------------------------------------------------------------- #

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points):
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) <= 2:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _convex_hull_area(points):
    hull = _convex_hull(points)
    if len(hull) < 3:
        return 0.0
    a = 0.0
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0

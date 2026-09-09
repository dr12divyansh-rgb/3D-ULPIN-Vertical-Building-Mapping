"""Small bmesh utility functions (Blender-only; requires bpy/bmesh/mathutils)."""

from __future__ import annotations

import math

import bmesh
from mathutils import Matrix

from ..core.surfaces import SEMANTIC_LAYER_NAME


def new_bmesh():
    return bmesh.new()


def new_tagged_bmesh():
    """New bmesh with a face int layer for surface semantics."""
    bm = bmesh.new()
    bm.faces.layers.int.new(SEMANTIC_LAYER_NAME)
    return bm


def semantic_layer(bm):
    return bm.faces.layers.int.get(SEMANTIC_LAYER_NAME)


def transform_bmesh(bm, x, y, rot_deg):
    """Rotate around Z by rot_deg degrees then translate by (x, y, 0)."""
    rot = Matrix.Rotation(math.radians(rot_deg), 4, "Z")
    trans = Matrix.Translation((x, y, 0.0))
    bmesh.ops.transform(bm, matrix=trans @ rot, verts=bm.verts[:])


def translate_bmesh(bm, x, y, z=0.0):
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=(x, y, z))


def add_box(bm, cx, cy, cz, sx, sy, sz):
    """Add an axis-aligned box centered at (cx, cy, cz) with sizes (sx, sy, sz)."""
    ret = bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Identity(4), calc_uvs=False)
    for v in ret["verts"]:
        v.co.x = v.co.x * sx + cx
        v.co.y = v.co.y * sy + cy
        v.co.z = v.co.z * sz + cz
    return ret["verts"]


def add_ellipsoid(bm, cx, cy, cz, rx, ry, rz):
    """Add an icosphere (radius 1) scaled to radii (rx, ry, rz) centered at (cx, cy, cz)."""
    ret = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=1.0,
                                     matrix=Matrix.Identity(4), calc_uvs=False)
    for v in ret["verts"]:
        v.co.x = v.co.x * rx + cx
        v.co.y = v.co.y * ry + cy
        v.co.z = v.co.z * rz + cz
    return ret["verts"]


def extract_triangles(bm):
    """Triangulate and return a list of 3D triangles, each a tuple of 3 (x,y,z)."""
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    tris = []
    for f in bm.faces:
        v = f.verts
        n = len(v)
        for i in range(1, n - 1):
            tris.append((tuple(v[0].co), tuple(v[i].co), tuple(v[i + 1].co)))
    return tris


def extract_tagged_triangles(bm, sem):
    """Fan-triangulate faces and return ``(a, b, c, tag)`` per triangle."""
    tris = []
    for f in bm.faces:
        tag = f[sem]
        v = f.verts
        n = len(v)
        for i in range(1, n - 1):
            tris.append((tuple(v[0].co), tuple(v[i].co), tuple(v[i + 1].co), int(tag)))
    return tris


def mesh_bounds(bm):
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

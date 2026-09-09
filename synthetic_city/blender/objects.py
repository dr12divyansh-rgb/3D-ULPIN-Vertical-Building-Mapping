"""Non-building scene objects (ground, roads, trees, vehicles, streetlights)."""

from __future__ import annotations

import random

from . import meshutil as mu


def ground_triangles(ground_size):
    """A flat ground plane centred on the origin (class: ground)."""
    gx, gy = ground_size
    hx, hy = gx / 2.0, gy / 2.0
    return [
        ((-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0)),
        ((-hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0)),
    ]


def road_triangles(ground_size, road_width):
    """Two crossing roads (class: road)."""
    gx, gy = ground_size
    hx, hy = gx / 2.0, gy / 2.0
    rw = road_width / 2.0
    z = 0.01
    tris = []
    # horizontal road along X (strip between y=-rw and y=+rw)
    tris.extend(_rect((-hx, -rw, z), (hx, -rw, z), (hx, rw, z), (-hx, rw, z)))
    # vertical road along Y (strip between x=-rw and x=+rw)
    tris.extend(_rect((-rw, -hy, z), (rw, -hy, z), (rw, hy, z), (-rw, hy, z)))
    return tris


def _rect(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def tree_triangles(position, rng):
    """Trunk + ellipsoid canopy (class: vegetation)."""
    trunk_h = rng.uniform(1.5, 3.0)
    trunk_r = rng.uniform(0.12, 0.25)
    canopy_r = rng.uniform(1.5, 3.0)
    canopy_z = trunk_h + canopy_r * 0.55
    bm = mu.new_bmesh()
    mu.add_box(bm, 0.0, 0.0, trunk_h / 2.0, trunk_r * 2.0, trunk_r * 2.0, trunk_h)
    mu.add_ellipsoid(bm, 0.0, 0.0, canopy_z, canopy_r, canopy_r, canopy_r * rng.uniform(1.0, 1.3))
    mu.translate_bmesh(bm, position[0], position[1], 0.0)
    tris = mu.extract_triangles(bm)
    bm.free()
    return tris


def vehicle_triangles(position, rotation_deg, rng):
    """Body + cabin box (class: vehicle)."""
    length = rng.uniform(3.6, 5.0)
    width = rng.uniform(1.6, 2.0)
    body_h = rng.uniform(0.7, 0.9)
    bm = mu.new_bmesh()
    mu.add_box(bm, 0.0, 0.0, body_h / 2.0, length, width, body_h)
    mu.add_box(bm, 0.0, 0.0, body_h + 0.4, length * 0.5, width * 0.85, 0.8)
    mu.transform_bmesh(bm, position[0], position[1], rotation_deg)
    tris = mu.extract_triangles(bm)
    bm.free()
    return tris


def streetlight_triangles(position, rng):
    """Pole + head box (class: other)."""
    pole_h = rng.uniform(5.0, 7.0)
    bm = mu.new_bmesh()
    mu.add_box(bm, 0.0, 0.0, pole_h / 2.0, 0.12, 0.12, pole_h)
    mu.add_box(bm, 0.4, 0.0, pole_h + 0.05, 0.9, 0.25, 0.2)
    mu.translate_bmesh(bm, position[0], position[1], 0.0)
    tris = mu.extract_triangles(bm)
    bm.free()
    return tris

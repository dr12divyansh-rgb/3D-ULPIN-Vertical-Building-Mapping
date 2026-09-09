"""Synthetic LiDAR point sampling + imperfections (pure Python, no bpy).

This module turns triangle soups into LiDAR-like point clouds. It is a
*geometric* sampler, not a physical ray tracer: points are drawn uniformly
(area-weighted) over the surface of each object, which gives correct XYZ and
semantic labels with full reproducibility.

A physical sensor simulation can be layered on later by replacing the sampling
strategy while keeping the same public interface (``generate_lidar``).
"""

from __future__ import annotations

import bisect
import math

from .labels import NON_BUILDING_ID


def triangle_area(a, b, c):
    """Area of a 3D triangle via the cross product."""
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)


def _sample_triangle(a, b, c, rng):
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1.0:
        r1 = 1.0 - r1
        r2 = 1.0 - r2
    x = a[0] + r1 * (b[0] - a[0]) + r2 * (c[0] - a[0])
    y = a[1] + r1 * (b[1] - a[1]) + r2 * (c[1] - a[1])
    z = a[2] + r1 * (b[2] - a[2]) + r2 * (c[2] - a[2])
    return (x, y, z)


def sample_mesh_surface(triangles, target_points, rng):
    """Uniform area-weighted surface sampling of a triangle soup.

    ``triangles`` is a sequence of 3D triangles, each a tuple of three
    (x, y, z) triples. Returns a list of (x, y, z).
    """
    if not triangles or target_points <= 0:
        return []
    areas = [triangle_area(a, b, c) for a, b, c in triangles]
    total = sum(areas)
    if total <= 0:
        return []
    cum = []
    acc = 0.0
    for ar in areas:
        acc += ar
        cum.append(acc)
    out = []
    for _ in range(target_points):
        r = rng.random() * total
        chosen = bisect.bisect_left(cum, r)
        if chosen >= len(triangles):
            chosen = len(triangles) - 1
        a, b, c = triangles[chosen]
        out.append(_sample_triangle(a, b, c, rng))
    return out


def apply_imperfections(points, rng, noise_std=0.0, dropout_rate=0.0):
    """Apply Gaussian XYZ noise and random point dropout (in place)."""
    if noise_std > 0:
        for i in range(len(points)):
            x, y, z = points[i]
            points[i] = (x + rng.gauss(0.0, noise_std),
                         y + rng.gauss(0.0, noise_std),
                         z + rng.gauss(0.0, noise_std))
    if dropout_rate > 0:
        keep = [p for p in points if rng.random() >= dropout_rate]
        return keep
    return points


def synthetic_intensity(z, normal_z=None, rng=None):
    """A clearly synthetic 'intensity' scalar used only for schema parity.

    It is a crude, deterministic stand-in (a function of height and surface
    orientation) and is NOT physically meaningful.
    """
    base = 40.0
    if normal_z is not None:
        base += 180.0 * max(0.0, min(1.0, abs(normal_z)))
    else:
        base += 60.0
    base += max(0.0, min(30.0, z * 2.0))
    if rng is not None:
        base += rng.uniform(-8.0, 8.0)
    return int(max(0, min(255, round(base))))

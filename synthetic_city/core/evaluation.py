"""LOD2 reconstruction evaluation metrics (pure Python + numpy).

Compares a *predicted* LOD2 mesh against the *ground-truth* LOD2 mesh. This is
intended for the future reconstruction step; today it is exercised as a
sanity check (ground truth vs. itself) which must report (near) zero error.

Metrics:
  * height error (max-z difference)
  * footprint area error (absolute + relative)
  * footprint IoU (rasterised convex-hull projection)
  * chamfer surface distance (sampled)
  * topology: watertight, non-manifold edges, zero-area faces
"""

from __future__ import annotations

import math
import random

import numpy as np

from . import lidar
from .mesh import SurfaceMesh


def sample_surface_points(mesh: SurfaceMesh, n: int, rng: random.Random):
    tris = mesh.triangles()
    return lidar.sample_mesh_surface(tris, n, rng)


def _chamfer_one_way(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    diff = a[:, None, :] - b[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    return float(d.min(axis=1).mean())


def chamfer_distance(pred: SurfaceMesh, gt: SurfaceMesh, n=2000, seed=0) -> float:
    # Two independent RNGs with the *same* seed: identical meshes yield
    # identical samples -> chamfer of exactly 0 (desired for the sanity test).
    pa = np.asarray(sample_surface_points(pred, n, random.Random(seed)), dtype=np.float64)
    pb = np.asarray(sample_surface_points(gt, n, random.Random(seed)), dtype=np.float64)
    if pa.shape[0] == 0 or pb.shape[0] == 0:
        return float("inf")
    return (_chamfer_one_way(pa, pb) + _chamfer_one_way(pb, pa)) / 2.0


def _point_in_polygon(x, y, poly) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def footprint_iou(pred: SurfaceMesh, gt: SurfaceMesh, resolution=128) -> float:
    pp = pred.footprint_polygon()
    gp = gt.footprint_polygon()
    if len(pp) < 3 or len(gp) < 3:
        return 0.0
    allx = [p[0] for p in pp] + [p[0] for p in gp]
    ally = [p[1] for p in pp] + [p[1] for p in gp]
    minx, maxx = min(allx), max(allx)
    miny, maxy = min(ally), max(ally)
    if maxx - minx < 1e-9 or maxy - miny < 1e-9:
        return 1.0
    xs = np.linspace(minx, maxx, resolution)
    ys = np.linspace(miny, maxy, resolution)
    cnt_p = cnt_g = cnt_i = 0
    for y in ys:
        for x in xs:
            ip = _point_in_polygon(float(x), float(y), pp)
            ig = _point_in_polygon(float(x), float(y), gp)
            if ip and ig:
                cnt_i += 1
            if ip:
                cnt_p += 1
            if ig:
                cnt_g += 1
    union = cnt_p + cnt_g - cnt_i
    if union == 0:
        return 1.0 if cnt_p == cnt_g else 0.0
    return cnt_i / union


def evaluate_lod2(pred: SurfaceMesh, gt: SurfaceMesh, n_samples=2000, seed=0) -> dict:
    h_pred = pred.height()
    h_gt = gt.height()
    a_pred = pred.footprint_area()
    a_gt = gt.footprint_area()
    area_abs_err = abs(a_pred - a_gt)
    area_rel_err = area_abs_err / a_gt if a_gt > 0 else float("inf")
    return {
        "geometry": {
            "height_pred": h_pred,
            "height_gt": h_gt,
            "height_error": abs(h_pred - h_gt),
            "footprint_area_pred": a_pred,
            "footprint_area_gt": a_gt,
            "footprint_area_error": area_abs_err,
            "footprint_area_error_relative": area_rel_err,
            "footprint_iou": footprint_iou(pred, gt),
            "chamfer_distance": chamfer_distance(pred, gt, n_samples, seed),
        },
        "topology": {
            "watertight_pred": pred.is_watertight(),
            "watertight_gt": gt.is_watertight(),
            "non_manifold_edges_pred": pred.non_manifold_edge_count(),
            "non_manifold_edges_gt": gt.non_manifold_edge_count(),
            "zero_area_faces_pred": pred.zero_area_face_count(),
            "zero_area_faces_gt": gt.zero_area_face_count(),
        },
        "counts": {
            "vertices_pred": pred.num_vertices(),
            "vertices_gt": gt.num_vertices(),
            "faces_pred": pred.num_faces(),
            "faces_gt": gt.num_faces(),
        },
    }


def is_sane(results: dict, height_tol=1e-6, area_tol=1e-6, chamfer_tol=1e-6) -> bool:
    """Sanity check: a valid reconstruction should be near-identical to GT."""
    g = results["geometry"]
    return (g["height_error"] <= height_tol
            and g["footprint_area_error"] <= area_tol
            and abs(g["footprint_iou"] - 1.0) <= 1e-3
            and g["chamfer_distance"] <= chamfer_tol)

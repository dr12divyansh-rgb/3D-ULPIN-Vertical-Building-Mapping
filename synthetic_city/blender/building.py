"""Floor-aware procedural building generation (Blender bmesh backend).

Each building is generated from exact, recorded parameters and produces three
representations (the "original Blender geometry" is the source of truth):

    original Blender geometry (bmesh)
        ├── LOD1              (flat-top footprint extrusion, watertight)
        ├── ground-truth LOD2 (detailed external geometry, semantic surfaces)
        └── synthetic LiDAR   (surface-sampled points; excludes the hidden
                               ground-contact face)

Surface semantics (``WallSurface`` / ``RoofSurface`` / ``GroundSurface``) are
tagged onto bmesh faces so that LOD1/LOD2 remain structured, watertight meshes
rather than anonymous triangle soups.

Design notes
------------
* LOD1 is the footprint extruded to the *eave* height (no roof, no parapet).
* LOD2 adds the roof geometry (and a ground-contact face); it is the clean
  reconstruction target. Flat-roof LOD2 is therefore geometrically similar to
  LOD1 (height == eave height), which is intentional.
* Floor "marker" slabs (optional, see ``building.floor_markers``) are *not*
  part of the LOD2 mesh — they are facade detail emitted only into the LiDAR
  sampling geometry, so the point cloud can show floor lines without polluting
  the watertight LOD2 ground truth.
* Courtyard footprints are built as a true annulus (outer + inner walls, plus
  an annular top/bottom) so they remain watertight.
"""

from __future__ import annotations

import math
import random

import bmesh

from ..core import geometry as geo
from ..core import surfaces as surf
from ..core.mesh import SurfaceMesh
from . import meshutil as mu


# --------------------------------------------------------------------------- #
# tagged bmesh primitives
# --------------------------------------------------------------------------- #

def _add_walls(bm, pts_xy, z0, z1, sem, tag):
    n = len(pts_xy)
    bottom = [bm.verts.new((x, y, z0)) for (x, y) in pts_xy]
    top = [bm.verts.new((x, y, z1)) for (x, y) in pts_xy]
    for i in range(n):
        j = (i + 1) % n
        f = bm.faces.new([bottom[i], bottom[j], top[j], top[i]])
        f[sem] = tag


def _add_flat_face(bm, pts_xy, z, sem, tag):
    verts = [bm.verts.new((x, y, z)) for (x, y) in pts_xy]
    for (i, j, k) in geo.triangulate_polygon(list(pts_xy)):
        f = bm.faces.new([verts[i], verts[j], verts[k]])
        f[sem] = tag


def _add_roof(bm, tagged_polygons, sem):
    for poly, sname in tagged_polygons:
        tag = surf.SURFACE_NAMES[sname]
        verts = [bm.verts.new((x, y, z)) for (x, y, z) in poly]
        for i in range(1, len(verts) - 1):
            f = bm.faces.new([verts[0], verts[i], verts[i + 1]])
            f[sem] = tag


def _add_thin_band(bm, ax, ay, bx, by, nx, ny, z, thickness, protrusion):
    z0 = z - thickness / 2.0
    z1 = z + thickness / 2.0
    px, py = protrusion * nx, protrusion * ny
    v = [
        bm.verts.new((ax, ay, z0)),
        bm.verts.new((bx, by, z0)),
        bm.verts.new((bx + px, by + py, z0)),
        bm.verts.new((ax + px, ay + py, z0)),
        bm.verts.new((ax, ay, z1)),
        bm.verts.new((bx, by, z1)),
        bm.verts.new((bx + px, by + py, z1)),
        bm.verts.new((ax + px, ay + py, z1)),
    ]
    for face in ([0, 3, 2, 1], [4, 5, 6, 7], [3, 2, 6, 7], [0, 1, 5, 4],
                 [0, 4, 7, 3], [1, 2, 6, 5]):
        bm.faces.new([v[i] for i in face])


def _add_floor_markers(bm, rings, floor_zs, protrusion, thickness):
    for ring in rings:
        if len(ring) < 3:
            continue
        c = geo.polygon_centroid(ring)
        n = len(ring)
        for i in range(n):
            ax, ay = ring[i]
            bx, by = ring[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            if math.hypot(dx, dy) < 1e-9:
                continue
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            ox, oy = mx - c[0], my - c[1]
            ol = math.hypot(ox, oy)
            if ol < 1e-9:
                continue
            for z in floor_zs:
                _add_thin_band(bm, ax, ay, bx, by, ox / ol, oy / ol, z,
                               thickness, protrusion)


# --------------------------------------------------------------------------- #
# parameter sampling
# --------------------------------------------------------------------------- #

def sample_building_params(rng: random.Random, cfg: dict) -> dict:
    b = cfg["building"]

    ftype = rng.choice(b["footprint_types"])
    width = rng.uniform(*b["width_range"])
    length = rng.uniform(*b["length_range"])
    fp = geo.generate_footprint(rng, ftype, width, length)

    nfloors = rng.randint(*b["floors_range"])
    if b.get("per_floor_variation"):
        floor_heights = [rng.uniform(*b["floor_height_range"]) for _ in range(nfloors)]
    else:
        fh = rng.uniform(*b["floor_height_range"])
        floor_heights = [fh] * nfloors
    wall_height = sum(floor_heights)
    floor_levels = [0.0]
    acc = 0.0
    for fh in floor_heights:
        acc += fh
        floor_levels.append(acc)

    allowed_roofs = b["roof_types"] if ftype == "rectangle" else ["flat"]
    roof_type = rng.choice(allowed_roofs)
    roof_slope = None
    if roof_type == "flat":
        roof_rise = 0.0
    else:
        roof_slope = rng.uniform(*b["roof_slope_range"])
        roof_rise = roof_slope * min(fp["width"], fp["length"])
    total_height = wall_height + roof_rise

    return {
        "ftype": ftype,
        "width": fp["width"],
        "length": fp["length"],
        "fp": fp,
        "nfloors": nfloors,
        "floor_heights": floor_heights,
        "floor_levels": floor_levels,
        "wall_height": wall_height,
        "roof_type": roof_type,
        "roof_slope": roof_slope,
        "roof_rise": roof_rise,
        "total_height": total_height,
    }


def _check_bounds(bm, width, length, total_height, tol):
    mnx, mny, mnz, mxx, mxy, mxz = mu.mesh_bounds(bm)
    x_ext = mxx - mnx
    y_ext = mxy - mny
    z_ext = mxz - mnz
    if abs(x_ext - width) > tol or abs(y_ext - length) > tol or abs(z_ext - total_height) > tol:
        raise AssertionError(
            f"building geometry mismatch: expected width={width:.3f} length={length:.3f} "
            f"height={total_height:.3f}, got x_ext={x_ext:.3f} y_ext={y_ext:.3f} z_ext={z_ext:.3f}"
        )


def _world_pt(lx, ly, x, y, rot_rad):
    ca, sa = math.cos(rot_rad), math.sin(rot_rad)
    return (lx * ca - ly * sa + x, lx * sa + ly * ca + y)


def _has_hole(fp):
    return fp.get("hole") is not None


def _add_prism_walls(bm, fp, z0, z1, sem, tag):
    """Extrude the footprint outline to vertical walls (courtyard-aware)."""
    if _has_hole(fp):
        _add_walls(bm, fp["outer"], z0, z1, sem, tag)
        _add_walls(bm, fp["hole"], z0, z1, sem, tag)
    else:
        _add_walls(bm, fp["outer"], z0, z1, sem, tag)


def _add_horizontal_faces(bm, fp, z, sem, tag):
    """Add a horizontal face at height z (courtyard: annulus via 4 quads)."""
    if _has_hole(fp):
        for quad in _annulus_quads(fp["outer"], fp["hole"]):
            _add_flat_face(bm, quad, z, sem, tag)
    else:
        _add_flat_face(bm, fp["outer"], z, sem, tag)


def _annulus_quads(outer, hole):
    """Quadrangulate a rectangular annulus (outer ring + rectangular hole).

    Assumes both rings have 4 vertices and are ordered consistently (the
    courtyard footprint guarantees this). The four corner-to-corner bridges
    A-a, B-b, C-c, D-d divide the annulus into four trapezoids with no
    T-junctions, so the result is watertight.
    """
    A, B, C, D = outer[0], outer[1], outer[2], outer[3]
    a, b, c, d = hole[0], hole[1], hole[2], hole[3]
    return [
        [A, B, b, a],
        [B, C, c, b],
        [C, D, d, c],
        [D, A, a, d],
    ]


# --------------------------------------------------------------------------- #
# LOD1 / LOD2 / marker builders
# --------------------------------------------------------------------------- #

def _build_lod1_bmesh(fp, wall_height):
    bm = mu.new_tagged_bmesh()
    sem = mu.semantic_layer(bm)
    _add_prism_walls(bm, fp, 0.0, wall_height, sem, surf.WALL)
    _add_horizontal_faces(bm, fp, wall_height, sem, surf.ROOF)
    _add_horizontal_faces(bm, fp, 0.0, sem, surf.GROUND)
    return bm


def _build_lod2_bmesh(params, cfg):
    fp = params["fp"]
    wall_height = params["wall_height"]
    roof_type = params["roof_type"]
    roof_rise = params["roof_rise"]

    bm = mu.new_tagged_bmesh()
    sem = mu.semantic_layer(bm)
    _add_prism_walls(bm, fp, 0.0, wall_height, sem, surf.WALL)
    if roof_type == "flat":
        _add_horizontal_faces(bm, fp, wall_height, sem, surf.ROOF)
    _add_horizontal_faces(bm, fp, 0.0, sem, surf.GROUND)

    if roof_type != "flat":
        _add_roof(bm, geo.roof_polygons(roof_type, fp["width"], fp["length"],
                                        wall_height, roof_rise), sem)
    return bm


def _build_marker_bmesh(params, cfg):
    """Floor marker slabs (facade detail for the LiDAR sampling geometry only)."""
    b = cfg["building"]
    fp = params["fp"]
    bm = mu.new_bmesh()
    rings = [fp["outer"]]
    if _has_hole(fp):
        rings.append(fp["hole"])
    _add_floor_markers(bm, rings, params["floor_levels"][1:-1],
                       b.get("floor_marker_protrusion", 0.12),
                       b.get("floor_marker_thickness", 0.06))
    return bm


def _surfaces_summary(mesh: SurfaceMesh):
    areas = mesh.area_by_type()
    summary = {}
    for name in surf.SURFACE_TYPE_NAMES:
        if name not in mesh.surfaces:
            continue
        summary[name] = {
            "face_count": len(mesh.surfaces[name]),
            "area": round(areas.get(name, 0.0), 4),
        }
    return summary


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #

def build_building(params: dict, cfg: dict, building_id: int,
                   position, rotation_deg: float) -> dict:
    """Build LOD1 + ground-truth LOD2 + LiDAR triangles for one building."""
    b = cfg["building"]
    fp = params["fp"]
    ftype = params["ftype"]
    roof_type = params["roof_type"]
    wall_height = params["wall_height"]
    roof_rise = params["roof_rise"]
    total_height = params["total_height"]

    lod1_bm = _build_lod1_bmesh(fp, wall_height)
    lod2_bm = _build_lod2_bmesh(params, cfg)

    _check_bounds(lod1_bm, fp["width"], fp["length"], wall_height, 0.05)
    _check_bounds(lod2_bm, fp["width"], fp["length"], total_height, 0.05)

    mu.transform_bmesh(lod1_bm, position[0], position[1], rotation_deg)
    mu.transform_bmesh(lod2_bm, position[0], position[1], rotation_deg)

    lod1_tagged = mu.extract_tagged_triangles(lod1_bm, mu.semantic_layer(lod1_bm))
    lod2_tagged = mu.extract_tagged_triangles(lod2_bm, mu.semantic_layer(lod2_bm))
    lod1_bm.free()
    lod2_bm.free()

    lod1_mesh = SurfaceMesh()
    lod2_mesh = SurfaceMesh()
    lidar_tris = []
    for (a, b_, c, tag) in lod1_tagged:
        lod1_mesh.add_triangle(surf.SURFACE_CODES[tag], a, b_, c)
    for (a, b_, c, tag) in lod2_tagged:
        name = surf.SURFACE_CODES[tag]
        lod2_mesh.add_triangle(name, a, b_, c)
        if tag != surf.GROUND:
            lidar_tris.append((a, b_, c))

    if b.get("floor_markers", True):
        marker_bm = _build_marker_bmesh(params, cfg)
        mu.transform_bmesh(marker_bm, position[0], position[1], rotation_deg)
        lidar_tris.extend(mu.extract_triangles(marker_bm))
        marker_bm.free()

    rot_rad = math.radians(rotation_deg)
    x, y = position
    footprint_world = [_world_pt(px, py, x, y, rot_rad) for (px, py) in fp["outer"]]
    hole_world = ([_world_pt(px, py, x, y, rot_rad) for (px, py) in fp["hole"]]
                  if fp["hole"] else None)

    footprint_area = round(lod2_mesh.footprint_area(), 4)

    full_metadata = {
        "building_id": int(building_id),
        "footprint_type": ftype,
        "roof_type": roof_type,
        "width": round(float(fp["width"]), 4),
        "length": round(float(fp["length"]), 4),
        "footprint_area": footprint_area,
        "footprint_local": [[round(px, 4), round(py, 4)] for (px, py) in fp["outer"]],
        "footprint_world": [[round(px, 4), round(py, 4)] for (px, py) in footprint_world],
        "hole_local": ([[round(px, 4), round(py, 4)] for (px, py) in fp["hole"]]
                       if fp["hole"] else None),
        "hole_world": ([[round(px, 4), round(py, 4)] for (px, py) in hole_world]
                       if hole_world else None),
        "position": [round(x, 4), round(y, 4)],
        "rotation_deg": round(float(rotation_deg), 4),
        "floor_count": int(params["nfloors"]),
        "floor_heights": [round(fh, 4) for fh in params["floor_heights"]],
        "floor_levels": [round(z, 4) for z in params["floor_levels"]],
        "wall_height": round(wall_height, 4),
        "roof_height": round(roof_rise, 4),
        "total_height": round(total_height, 4),
        "roof_slope": None if params["roof_slope"] is None else round(params["roof_slope"], 4),
    }

    lod1_meta = {
        "building_id": int(building_id),
        "lod": "LOD1",
        "footprint": [[round(px, 4), round(py, 4)] for (px, py) in footprint_world],
        "hole": ([[round(px, 4), round(py, 4)] for (px, py) in hole_world]
                 if hole_world else None),
        "footprint_type": ftype,
        "ground_elevation": 0.0,
        "base_z": 0.0,
        "height": round(wall_height, 4),
        "top_z": round(wall_height, 4),
        "footprint_area": round(lod1_mesh.footprint_area(), 4),
        "num_vertices": lod1_mesh.num_vertices(),
        "num_faces": lod1_mesh.num_faces(),
        "surfaces": _surfaces_summary(lod1_mesh),
        "obj": f"building_{building_id:06d}.obj",
    }

    lod2_meta = {
        "building_id": int(building_id),
        "lod": "LOD2",
        "footprint": [[round(px, 4), round(py, 4)] for (px, py) in footprint_world],
        "hole": ([[round(px, 4), round(py, 4)] for (px, py) in hole_world]
                 if hole_world else None),
        "footprint_type": ftype,
        "roof_type": roof_type,
        "base_z": 0.0,
        "top_z": round(total_height, 4),
        "height": round(total_height, 4),
        "footprint_area": round(lod2_mesh.footprint_area(), 4),
        "num_vertices": lod2_mesh.num_vertices(),
        "num_faces": lod2_mesh.num_faces(),
        "surfaces": _surfaces_summary(lod2_mesh),
        "lod_comparison": {
            "footprint_area_lod1": round(lod1_mesh.footprint_area(), 4),
            "footprint_area_lod2": round(lod2_mesh.footprint_area(), 4),
            "height_lod1": round(wall_height, 4),
            "height_lod2": round(total_height, 4),
            "roof_type": roof_type,
        },
        "obj": f"building_{building_id:06d}.obj",
    }

    return {
        "lod1_mesh": lod1_mesh,
        "lod2_mesh": lod2_mesh,
        "lidar_tris": lidar_tris,
        "building": full_metadata,
        "lod1_meta": lod1_meta,
        "lod2_meta": lod2_meta,
    }

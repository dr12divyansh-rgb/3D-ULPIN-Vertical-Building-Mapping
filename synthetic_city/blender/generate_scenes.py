"""Blender entry point: generate the synthetic dataset scenes headlessly.

Run by the orchestrator as:

    blender --background --factory-startup \\
        --python synthetic_city/blender/generate_scenes.py \\
        -- --job <job.json> --out <output_root>

The job JSON contains the resolved config and the scene manifest. This script
is fully deterministic: every scene uses its own ``random.Random(seed)``.

Output layout (per scene):

    scene_XXXXX/
    ├── lod1/                 building_XXXXXX.obj + metadata.json
    ├── lidar/                pointcloud.ply + metadata.json
    ├── ground_truth/
    │   ├── lod2/             building_XXXXXX.obj + metadata.json
    │   └── building.json
    └── scene.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bpy  # noqa: E402  (initializes the Blender Python environment)
from mathutils.bvhtree import BVHTree  # noqa: E402

from synthetic_city.core import lidar  # noqa: E402
from synthetic_city.core import ply_io  # noqa: E402
from synthetic_city.core.labels import CLASS_IDS, NON_BUILDING_ID  # noqa: E402
from synthetic_city.blender import building as bld  # noqa: E402
from synthetic_city.blender import objects as objs  # noqa: E402


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Scene assembly + placement
# --------------------------------------------------------------------------- #

def _radius(w, l):
    return math.hypot(w / 2.0, l / 2.0)


def _collides(x, y, r, placed, spacing):
    for (px, py, pr) in placed:
        d = math.hypot(x - px, y - py)
        if d < r + pr + spacing:
            return True
    return False


def _on_road(cx, cy, r, rw):
    dy = max(0.0, abs(cy) - rw)
    dx = max(0.0, abs(cx) - rw)
    return (dy < r) or (dx < r)


def _sample_in_bounds(rng, hx, hy, margin):
    return (rng.uniform(-hx + margin, hx - margin),
            rng.uniform(-hy + margin, hy - margin))


def build_scene(scene, cfg, rng):
    """Assemble all scene elements plus per-building LOD1/LOD2 data."""
    scfg = cfg["scene"]
    ground_size = tuple(scfg["ground_size"])
    gx, gy = ground_size
    hx, hy = gx / 2.0, gy / 2.0
    margin = 2.0
    urban = scene["mode"] == "urban"
    road_half = scfg["road_width"] / 2.0 if urban else None
    spacing = scfg["building_spacing"]

    elements = []
    buildings_meta = []
    lod1_list = []  # (bid, SurfaceMesh, lod1_meta)
    lod2_list = []  # (bid, SurfaceMesh, lod2_meta)

    elements.append({"class": "ground", "building_id": NON_BUILDING_ID,
                     "tris": objs.ground_triangles(ground_size)})
    if urban:
        elements.append({"class": "road", "building_id": NON_BUILDING_ID,
                         "tris": objs.road_triangles(ground_size, scfg["road_width"])})

    # ---- buildings ----
    placed = []  # (x, y, radius)
    for i in range(scene["num_buildings"]):
        bid = scene["building_id_base"] + i
        params = bld.sample_building_params(rng, cfg)
        r = _radius(params["width"], params["length"])
        pos = None
        for _ in range(400):
            cand = _sample_in_bounds(rng, hx, hy, margin + r)
            if _collides(cand[0], cand[1], r, placed, spacing):
                continue
            if urban and _on_road(cand[0], cand[1], r, road_half):
                continue
            pos = cand
            break
        if pos is None:
            pos = _sample_in_bounds(rng, hx, hy, margin + r)
        rot = rng.uniform(*cfg["building"]["rotation_range"])
        result = bld.build_building(params, cfg, bid, pos, rot)
        elements.append({"class": "building", "building_id": bid,
                         "tris": result["lidar_tris"]})
        buildings_meta.append(result["building"])
        lod1_list.append((bid, result["lod1_mesh"], result["lod1_meta"]))
        lod2_list.append((bid, result["lod2_mesh"], result["lod2_meta"]))
        placed.append((pos[0], pos[1], r))

    counts = {"buildings": len(buildings_meta), "trees": 0, "vehicles": 0, "streetlights": 0}

    if urban:
        n_trees = rng.randint(*scfg["tree_count_range"])
        for _ in range(n_trees):
            for _ in range(200):
                cand = _sample_in_bounds(rng, hx, hy, margin + 1.5)
                if not _collides(cand[0], cand[1], 1.5, placed, spacing):
                    break
            elements.append({"class": "vegetation", "building_id": NON_BUILDING_ID,
                             "tris": objs.tree_triangles(cand, rng)})
            counts["trees"] += 1

        n_veh = rng.randint(*scfg["vehicle_count_range"])
        for _ in range(n_veh):
            for _ in range(200):
                if rng.random() < 0.5:
                    cand = (rng.uniform(-hx + 2.0, hx - 2.0),
                            rng.uniform(-road_half + 0.8, road_half - 0.8))
                else:
                    cand = (rng.uniform(-road_half + 0.8, road_half - 0.8),
                            rng.uniform(-hy + 2.0, hy - 2.0))
                if not _collides(cand[0], cand[1], 2.5, placed, spacing):
                    break
            rot = rng.uniform(0.0, 360.0)
            elements.append({"class": "vehicle", "building_id": NON_BUILDING_ID,
                             "tris": objs.vehicle_triangles(cand, rot, rng)})
            counts["vehicles"] += 1

        n_lights = rng.randint(0, 3)
        for _ in range(n_lights):
            for _ in range(200):
                cand = _sample_in_bounds(rng, hx, hy, margin + 0.5)
                if not _collides(cand[0], cand[1], 0.5, placed, spacing):
                    break
            elements.append({"class": "other", "building_id": NON_BUILDING_ID,
                             "tris": objs.streetlight_triangles(cand, rng)})
            counts["streetlights"] += 1

    return elements, buildings_meta, lod1_list, lod2_list, counts


# --------------------------------------------------------------------------- #
# LiDAR sampling + imperfections
# --------------------------------------------------------------------------- #

def _class_id(name):
    return CLASS_IDS[name]


def build_bvh(elements):
    verts = []
    polys = []
    vi = 0
    for el in elements:
        for tri in el["tris"]:
            for p in tri:
                verts.append(p)
            polys.append((vi, vi + 1, vi + 2))
            vi += 3
    return BVHTree.FromPolygons(verts, polys)


def visible_indices(points, sensor, tree):
    keep = []
    for i, p in enumerate(points):
        dx, dy, dz = p[0] - sensor[0], p[1] - sensor[1], p[2] - sensor[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-9:
            keep.append(i)
            continue
        loc, _, _, hd = tree.ray_cast(sensor, (dx / dist, dy / dist, dz / dist))
        if loc is None or hd >= dist - 0.05:
            keep.append(i)
    return keep


# --------------------------------------------------------------------------- #
# output writing
# --------------------------------------------------------------------------- #

def _write_scene_outputs(scene, cfg, scene_dir, elements, buildings_meta,
                         lod1_list, lod2_list, counts, points, cls, bids,
                         intensity, return_number, point_stats):
    lod1_dir = scene_dir / "lod1"
    lod2_dir = scene_dir / "ground_truth" / "lod2"
    lidar_dir = scene_dir / "lidar"
    ground_truth_dir = scene_dir / "ground_truth"
    for d in (lod1_dir, lod2_dir, lidar_dir, ground_truth_dir):
        d.mkdir(parents=True, exist_ok=True)

    # LiDAR
    ply_io.write_pointcloud(lidar_dir / "pointcloud.ply", points, cls, bids,
                            intensity, return_number)

    # LOD1 / LOD2 geometry + metadata
    lod1_meta_list = []
    lod2_meta_list = []
    for bid, mesh, meta in lod1_list:
        mesh.to_obj(lod1_dir / meta["obj"])
        lod1_meta_list.append(meta)
    for bid, mesh, meta in lod2_list:
        mesh.to_obj(lod2_dir / meta["obj"])
        lod2_meta_list.append(meta)

    (lod1_dir / "metadata.json").write_text(
        json.dumps({"lod": "LOD1", "buildings": lod1_meta_list}, indent=2),
        encoding="utf-8")
    (lod2_dir / "metadata.json").write_text(
        json.dumps({"lod": "LOD2", "buildings": lod2_meta_list}, indent=2),
        encoding="utf-8")
    (ground_truth_dir / "building.json").write_text(
        json.dumps({"buildings": buildings_meta}, indent=2), encoding="utf-8")

    lcfg = cfg["lidar"]
    lidar_meta = {
        "class_map": CLASS_IDS,
        "base_points_per_sqm": float(lcfg["base_points_per_sqm"]),
        "noise_std_m": float(lcfg["noise_std_m"]),
        "point_dropout_rate": float(lcfg["point_dropout_rate"]),
        "variable_density": bool(lcfg.get("variable_density")),
        "occlusion": bool(lcfg.get("sensor", {}).get("occlusion")),
        "sensor_position": lcfg.get("sensor", {}).get("position"),
        "ground_size": list(cfg["scene"]["ground_size"]),
        "point_counts": point_stats,
        "num_points": len(points),
    }
    (lidar_dir / "metadata.json").write_text(
        json.dumps(lidar_meta, indent=2), encoding="utf-8")

    scene_json = {
        "scene_id": scene["scene_id"],
        "split": scene["split"],
        "seed": scene["seed"],
        "mode": scene["mode"],
        "class_map": CLASS_IDS,
        "ground_size": list(cfg["scene"]["ground_size"]),
        "objects": counts,
        "num_buildings": len(buildings_meta),
        "building_ids": [b["building_id"] for b in buildings_meta],
    }
    (scene_dir / "scene.json").write_text(
        json.dumps(scene_json, indent=2), encoding="utf-8")


def generate_scene(scene, cfg, out_root):
    rng = random.Random(scene["seed"])
    lcfg = cfg["lidar"]
    elements, buildings_meta, lod1_list, lod2_list, counts = build_scene(scene, cfg, rng)

    # ---- surface sampling ----
    base = float(lcfg["base_points_per_sqm"])
    mult = lcfg["class_multipliers"]
    points = []
    cls = []
    bids = []
    per_class_sampled = {}
    for el in elements:
        area = sum(lidar.triangle_area(*t) for t in el["tris"])
        factor = (rng.uniform(*lcfg["variable_density_range"])
                  if lcfg.get("variable_density") else 1.0)
        n = int(round(area * base * mult.get(el["class"], 1.0) * factor))
        pts = lidar.sample_mesh_surface(el["tris"], n, rng)
        cid = _class_id(el["class"])
        points.extend(pts)
        cls.extend([cid] * len(pts))
        bids.extend([el["building_id"]] * len(pts))
        per_class_sampled[el["class"]] = per_class_sampled.get(el["class"], 0) + len(pts)

    sampled_total = len(points)
    max_pts = int(lcfg["max_points_per_scene"])
    if sampled_total > max_pts:
        keep = set(rng.sample(range(sampled_total), max_pts))
        points = [p for i, p in enumerate(points) if i in keep]
        cls = [c for i, c in enumerate(cls) if i in keep]
        bids = [b for i, b in enumerate(bids) if i in keep]

    after_cap = len(points)
    after_occlusion = after_cap

    if lcfg.get("sensor", {}).get("occlusion"):
        tree = build_bvh(elements)
        sensor = tuple(lcfg["sensor"]["position"])
        keep = set(visible_indices(points, sensor, tree))
        points = [p for i, p in enumerate(points) if i in keep]
        cls = [c for i, c in enumerate(cls) if i in keep]
        bids = [b for i, b in enumerate(bids) if i in keep]
        after_occlusion = len(points)

    noise_std = float(lcfg["noise_std_m"])
    dropout = float(lcfg["point_dropout_rate"])
    if noise_std > 0:
        for i in range(len(points)):
            x, y, z = points[i]
            points[i] = (x + rng.gauss(0.0, noise_std),
                         y + rng.gauss(0.0, noise_std),
                         z + rng.gauss(0.0, noise_std))
    if dropout > 0:
        pts2, cls2, bids2 = [], [], []
        for i in range(len(points)):
            if rng.random() >= dropout:
                pts2.append(points[i])
                cls2.append(cls[i])
                bids2.append(bids[i])
        points, cls, bids = pts2, cls2, bids2

    intensity = [lidar.synthetic_intensity(p[2], None, rng) for p in points]
    return_number = [1] * len(points)

    final_per_class = {}
    for cid in cls:
        name = list(CLASS_IDS.keys())[list(CLASS_IDS.values()).index(cid)]
        final_per_class[name] = final_per_class.get(name, 0) + 1

    point_stats = {
        "sampled": sampled_total,
        "after_cap": after_cap,
        "after_occlusion": after_occlusion,
        "final": len(points),
        "per_class_sampled": per_class_sampled,
        "per_class_final": final_per_class,
    }

    split_dir = "metadata/debug" if scene["split"] == "debug" else scene["split"]
    scene_dir = out_root / split_dir / scene["scene_id"]
    scene_dir.mkdir(parents=True, exist_ok=True)

    _write_scene_outputs(scene, cfg, scene_dir, elements, buildings_meta,
                         lod1_list, lod2_list, counts, points, cls, bids,
                         intensity, return_number, point_stats)

    return {
        "scene_id": scene["scene_id"],
        "split": scene["split"],
        "seed": scene["seed"],
        "num_points": len(points),
        "num_buildings": len(buildings_meta),
        "path": str(scene_dir),
    }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    cfg = job["config"]
    manifest = job["manifest"]
    out_root = Path(args.out)

    results = []
    total = len(manifest)
    for idx, scene in enumerate(manifest, 1):
        log(f"[{idx}/{total}] generating {scene['split']}/{scene['scene_id']} "
            f"(seed={scene['seed']}, buildings={scene['num_buildings']})")
        results.append(generate_scene(scene, cfg, out_root))

    print("SUMMARY_JSON_START")
    print(json.dumps(results))
    print("SUMMARY_JSON_END")


if __name__ == "__main__":
    main()

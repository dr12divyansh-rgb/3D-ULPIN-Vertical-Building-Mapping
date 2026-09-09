"""Validate a generated synthetic dataset (LOD1 / LiDAR / LOD2 architecture).

Per-scene checks:
  * required files exist (scene.json, lidar/pointcloud.ply, ground_truth/building.json,
    lod1/metadata.json, ground_truth/lod2/metadata.json)
  * point cloud is non-empty, finite, no NaN; class ids and building ids valid
  * every building has valid full metadata (height > 0, floors >= 1, valid
    footprint and floor levels)
  * point cloud matches building metadata (height / width / length, local frame)
  * LOD1 meshes: valid footprint, positive height, watertight, finite coords,
    no duplicate/zero-area faces, consistent surface types
  * LOD2 meshes: same, plus surface types are consistent

Dataset-wide:
  * no building id appears in more than one split (no leakage)
  * scene counts match the config (when --config is provided)

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from synthetic_city.core import ply_io  # noqa: E402
from synthetic_city.core.labels import CLASS_NAMES, NON_BUILDING_ID  # noqa: E402
from synthetic_city.core.mesh import SurfaceMesh  # noqa: E402
from synthetic_city.core.surfaces import SURFACE_TYPE_NAMES  # noqa: E402

REQUIRED_BUILDING_FIELDS = [
    "building_id", "footprint_type", "roof_type", "width", "length",
    "footprint_local", "position", "rotation_deg", "floor_count",
    "floor_heights", "floor_levels", "wall_height", "roof_height",
    "total_height",
]


def _fail(errors, msg):
    errors.append(msg)


def _validate_building(b, errors):
    tag = f"building {b.get('building_id', '?')}"
    for field in REQUIRED_BUILDING_FIELDS:
        if field not in b or b[field] is None:
            _fail(errors, f"{tag}: missing field '{field}'")
            return
    if not (b["total_height"] > 0 and b["wall_height"] > 0):
        _fail(errors, f"{tag}: height must be > 0")
    if b["floor_count"] < 1:
        _fail(errors, f"{tag}: floor_count must be >= 1")
    fl = b["floor_levels"]
    if len(fl) != b["floor_count"] + 1:
        _fail(errors, f"{tag}: floor_levels length {len(fl)} != floor_count+1 "
                      f"({b['floor_count'] + 1})")
    elif any(fl[i] > fl[i + 1] + 1e-9 for i in range(len(fl) - 1)):
        _fail(errors, f"{tag}: floor_levels not non-decreasing")
    elif abs(fl[0]) > 1e-6:
        _fail(errors, f"{tag}: floor_levels[0] must be 0.0")
    if len(b["footprint_local"]) < 3:
        _fail(errors, f"{tag}: footprint must have >= 3 vertices")
    if b["footprint_type"] not in ("rectangle", "L", "T", "courtyard", "irregular"):
        _fail(errors, f"{tag}: unknown footprint_type {b['footprint_type']!r}")
    if b["roof_type"] not in ("flat", "gable", "hip", "shed"):
        _fail(errors, f"{tag}: unknown roof_type {b['roof_type']!r}")


def _validate_lod_mesh(mesh, meta, lod, errors):
    bid = meta.get("building_id", "?")
    tag = f"{lod} building {bid}"
    if mesh is None:
        _fail(errors, f"{tag}: could not load mesh")
        return
    if mesh.num_vertices() == 0 or mesh.num_faces() == 0:
        _fail(errors, f"{tag}: empty mesh")
        return
    if not mesh.has_finite_coords():
        _fail(errors, f"{tag}: non-finite coordinates")
    bad_types = [n for n in mesh.surfaces if n not in SURFACE_TYPE_NAMES]
    if bad_types:
        _fail(errors, f"{tag}: unknown surface types {bad_types}")
    n_zero = mesh.zero_area_face_count()
    if n_zero:
        _fail(errors, f"{tag}: {n_zero} zero-area faces")
    if not mesh.is_watertight():
        _fail(errors, f"{tag}: not watertight ({mesh.non_manifold_edge_count()} "
                      f"non-manifold edges)")
    if meta.get("height", 0.0) <= 0:
        _fail(errors, f"{tag}: height must be > 0")
    if len(meta.get("footprint", [])) < 3:
        _fail(errors, f"{tag}: invalid footprint (< 3 vertices)")
    if abs(mesh.height() - meta.get("top_z", 0.0)) > 1e-3:
        _fail(errors, f"{tag}: mesh height {mesh.height():.4f} != top_z "
                      f"{meta.get('top_z')}")


def _validate_scene(scene_dir, errors):
    tag = scene_dir.name
    pc_path = scene_dir / "lidar" / "pointcloud.ply"
    building_path = scene_dir / "ground_truth" / "building.json"
    lod1_meta_path = scene_dir / "lod1" / "metadata.json"
    lod2_meta_path = scene_dir / "ground_truth" / "lod2" / "metadata.json"

    missing = [str(p) for p in (pc_path, building_path, lod1_meta_path, lod2_meta_path)
               if not p.exists()]
    if missing:
        _fail(errors, f"{tag}: missing files {missing}")
        return None, None

    with open(building_path, "r", encoding="utf-8") as f:
        buildings = json.load(f).get("buildings", [])
    lod1_meta = json.loads(lod1_meta_path.read_text(encoding="utf-8")).get("buildings", [])
    lod2_meta = json.loads(lod2_meta_path.read_text(encoding="utf-8")).get("buildings", [])

    data, _ = ply_io.read_pointcloud(pc_path)
    xyz = np.column_stack([np.asarray(data["x"], dtype=np.float64),
                           np.asarray(data["y"], dtype=np.float64),
                           np.asarray(data["z"], dtype=np.float64)])
    class_id = np.asarray(data["class_id"], dtype=np.int32)
    building_id = np.asarray(data["building_id"], dtype=np.int32)

    if xyz.shape[0] == 0:
        _fail(errors, f"{tag}: empty point cloud")
    if not np.all(np.isfinite(xyz)):
        _fail(errors, f"{tag}: non-finite coordinates present")
    bad_cls = np.where(~np.isin(class_id, list(range(len(CLASS_NAMES)))))[0]
    if len(bad_cls):
        _fail(errors, f"{tag}: {len(bad_cls)} points with invalid class_id")

    known_bids = {b["building_id"] for b in buildings}
    bid_mask = building_id != NON_BUILDING_ID
    unknown = np.where(bid_mask & ~np.isin(building_id, list(known_bids)))[0]
    if len(unknown):
        _fail(errors, f"{tag}: {len(unknown)} points reference unknown building ids")

    noise_std = float(json.loads((scene_dir / "lidar" / "metadata.json")
                                 .read_text(encoding="utf-8")).get("noise_std_m", 0.03))
    for b in buildings:
        _validate_building(b, errors)

    # ---- point cloud vs full metadata (local frame) ----
    tol_z = 3 * noise_std + 0.2
    tol_xy = 3 * noise_std + 0.5
    for b in buildings:
        bid = b["building_id"]
        m = building_id == bid
        if not m.any():
            _fail(errors, f"building {bid}: no points in point cloud")
            continue
        pts = xyz[m]
        zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
        if abs(zmax - b["total_height"]) > tol_z:
            _fail(errors, f"building {bid}: point zmax {zmax:.3f} vs metadata "
                          f"total_height {b['total_height']:.3f}")
        if abs(zmin) > tol_z:
            _fail(errors, f"building {bid}: point zmin {zmin:.3f} not at ground")
        cx, cy = b["position"]
        a = -math.radians(b["rotation_deg"])
        ca, sa = math.cos(a), math.sin(a)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        if abs(lx.max() - lx.min() - b["width"]) > tol_xy:
            _fail(errors, f"building {bid}: x extent {lx.max() - lx.min():.3f} vs "
                          f"metadata width {b['width']:.3f}")
        if abs(ly.max() - ly.min() - b["length"]) > tol_xy:
            _fail(errors, f"building {bid}: y extent {ly.max() - ly.min():.3f} vs "
                          f"metadata length {b['length']:.3f}")

    # ---- LOD1 / LOD2 geometry validation ----
    lod1_by_id = {m["building_id"]: m for m in lod1_meta}
    lod2_by_id = {m["building_id"]: m for m in lod2_meta}
    for b in buildings:
        bid = b["building_id"]
        if bid not in lod1_by_id:
            _fail(errors, f"building {bid}: missing LOD1 metadata")
            continue
        m1 = lod1_by_id[bid]
        obj1 = scene_dir / "lod1" / m1["obj"]
        if not obj1.exists():
            _fail(errors, f"building {bid}: missing LOD1 obj {obj1.name}")
        else:
            _validate_lod_mesh(SurfaceMesh.from_obj(obj1), m1, "LOD1", errors)

        if bid not in lod2_by_id:
            _fail(errors, f"building {bid}: missing LOD2 metadata")
            continue
        m2 = lod2_by_id[bid]
        obj2 = scene_dir / "ground_truth" / "lod2" / m2["obj"]
        if not obj2.exists():
            _fail(errors, f"building {bid}: missing LOD2 obj {obj2.name}")
        else:
            _validate_lod_mesh(SurfaceMesh.from_obj(obj2), m2, "LOD2", errors)

        if "lod_comparison" not in m2:
            _fail(errors, f"building {bid}: LOD2 missing lod_comparison")

    return buildings, building_id


def main():
    ap = argparse.ArgumentParser(description="Validate a synthetic dataset.")
    ap.add_argument("--data", default=None, help="dataset root (default: config output_dir)")
    ap.add_argument("--config", default=None, help="config file (for scene-count checks)")
    args = ap.parse_args()

    data_root = Path(args.data) if args.data else None
    cfg = None
    if args.config:
        import yaml
        from synthetic_city.utils import config as config_util
        cfg = config_util.load_config(args.config)
        if data_root is None:
            data_root = ROOT / cfg["dataset"]["output_dir"]
    if data_root is None:
        data_root = ROOT / "ml" / "data" / "synthetic"
    data_root = Path(data_root)

    errors = []
    split_ids = {}
    split_scene_counts = {}

    for split in ("train", "validation", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            _fail(errors, f"missing split directory: {split_dir}")
            continue
        ids = set()
        n_scenes = 0
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            n_scenes += 1
            buildings, _ = _validate_scene(scene_dir, errors)
            if buildings is not None:
                ids.update(b["building_id"] for b in buildings)
        split_ids[split] = ids
        split_scene_counts[split] = n_scenes

    splits = list(split_ids.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_ids[splits[i]] & split_ids[splits[j]]
            if overlap:
                _fail(errors, f"building id overlap between {splits[i]} and {splits[j]}: "
                              f"{sorted(overlap)[:20]}")

    if cfg is not None:
        expected = {
            "train": cfg["dataset"]["train_scenes"],
            "validation": cfg["dataset"]["validation_scenes"],
            "test": cfg["dataset"]["test_scenes"],
        }
        for split, exp in expected.items():
            got = split_scene_counts.get(split, 0)
            if got != exp:
                _fail(errors, f"split '{split}': expected {exp} scenes, found {got}")

    total_ids = sum(len(v) for v in split_ids.values())
    print("=" * 60)
    print("Dataset validation report")
    print(f"  data root: {data_root}")
    for split in ("train", "validation", "test"):
        print(f"  {split:>10}: {split_scene_counts.get(split, 0)} scenes, "
              f"{len(split_ids.get(split, ()))} buildings")
    print(f"  total buildings: {total_ids}")
    print("-" * 60)
    if errors:
        print(f"FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        print("=" * 60)
        sys.exit(1)
    print("PASS: all checks succeeded.")
    print("=" * 60)


if __name__ == "__main__":
    main()

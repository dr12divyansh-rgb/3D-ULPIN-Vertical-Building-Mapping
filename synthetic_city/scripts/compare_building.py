"""Visual comparison of LOD1 / LOD2 / LiDAR for a single building.

Usage:
    python synthetic_city/scripts/compare_building.py --scene <scene_dir|scene_id>
        [--data <dataset_root>] [--building <id>]
        [--view lod1|lod2|lidar|overlay|all] [--save out.png]

Colors LOD2 surfaces by semantic type (wall / roof / ground), shows LOD1 as a
wireframe or solid block, and LiDAR points colored by height.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from synthetic_city.core import ply_io  # noqa: E402
from synthetic_city.core.mesh import SurfaceMesh  # noqa: E402

SURFACE_COLORS = {
    "WallSurface": (0.85, 0.55, 0.30),
    "RoofSurface": (0.85, 0.20, 0.20),
    "GroundSurface": (0.45, 0.45, 0.45),
}
LOD1_COLOR = (0.35, 0.55, 0.85)


def _find_scene_dir(scene_arg, data_root):
    p = Path(scene_arg)
    if p.is_dir():
        return p
    for split in ("train", "validation", "test", "metadata/debug"):
        cand = Path(data_root) / split / scene_arg
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"scene not found: {scene_arg}")


def _surface_mesh_to_o3d(mesh, color_map):
    m = o3d.geometry.TriangleMesh()
    verts, tris, vcols = [], [], []
    vid = 0
    for name, faces in mesh.surfaces.items():
        col = color_map.get(name, (0.5, 0.5, 0.5))
        for (i, j, k) in faces:
            verts.append(mesh.vertices[i])
            verts.append(mesh.vertices[j])
            verts.append(mesh.vertices[k])
            tris.append([vid, vid + 1, vid + 2])
            vcols.extend([col, col, col])
            vid += 3
    if verts:
        m.vertices = o3d.utility.Vector3dVector(verts)
        m.triangles = o3d.utility.Vector3iVector(tris)
        m.vertex_colors = o3d.utility.Vector3dVector(vcols)
    return m


def _mesh_edges(mesh):
    edges = set()
    for faces in mesh.surfaces.values():
        for (i, j, k) in faces:
            for a, b in ((i, j), (j, k), (k, i)):
                edges.add((a, b) if a < b else (b, a))
    return [list(e) for e in edges]


def _mesh_to_lineset(mesh, color):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(mesh.vertices)
    ls.lines = o3d.utility.Vector2iVector(_mesh_edges(mesh))
    n = len(ls.lines)
    ls.colors = o3d.utility.Vector3dVector([color] * n)
    return ls


def _height_colors(xyz):
    zmin, zmax = xyz[:, 2].min(), xyz[:, 2].max()
    t = (xyz[:, 2] - zmin) / (zmax - zmin + 1e-9)
    return np.stack([t, 0.3 + 0.5 * (1 - t), 1 - t], axis=1)


def _build_geometries(scene_dir, building_id):
    lod1_meta = json.loads((scene_dir / "lod1" / "metadata.json").read_text(encoding="utf-8"))
    lod2_meta = json.loads((scene_dir / "ground_truth" / "lod2" / "metadata.json")
                           .read_text(encoding="utf-8"))
    m1 = {b["building_id"]: b for b in lod1_meta["buildings"]}
    m2 = {b["building_id"]: b for b in lod2_meta["buildings"]}

    if building_id is None:
        building_id = next(iter(m1))
    if building_id not in m1 or building_id not in m2:
        raise ValueError(f"building {building_id} not found in scene")

    lod1 = SurfaceMesh.from_obj(scene_dir / "lod1" / m1[building_id]["obj"])
    lod2 = SurfaceMesh.from_obj(scene_dir / "ground_truth" / "lod2" / m2[building_id]["obj"])

    data, _ = ply_io.read_pointcloud(scene_dir / "lidar" / "pointcloud.ply")
    xyz = np.column_stack([np.asarray(data["x"], np.float64),
                           np.asarray(data["y"], np.float64),
                           np.asarray(data["z"], np.float64)])
    bid = np.asarray(data["building_id"], np.int32)
    pts = xyz[bid == building_id]

    return building_id, lod1, lod2, pts, m2[building_id]


def main():
    ap = argparse.ArgumentParser(description="Compare LOD1/LOD2/LiDAR for a building.")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", default=None, help="dataset root")
    ap.add_argument("--building", type=int, default=None)
    ap.add_argument("--view", choices=["lod1", "lod2", "lidar", "overlay", "all"],
                    default="all")
    ap.add_argument("--save", default=None)
    ap.add_argument("--point-size", type=float, default=2.0)
    args = ap.parse_args()

    data_root = Path(args.data) if args.data else (ROOT / "ml" / "data" / "synthetic")
    scene_dir = _find_scene_dir(args.scene, data_root)
    building_id, lod1, lod2, pts, lod2_meta = _build_geometries(scene_dir, args.building)

    geoms = []
    if args.view in ("lod1", "all"):
        geoms.append(_surface_mesh_to_o3d(lod1, {"WallSurface": LOD1_COLOR,
                                                 "RoofSurface": LOD1_COLOR,
                                                 "GroundSurface": LOD1_COLOR}))
    if args.view in ("lod2", "overlay", "all"):
        geoms.append(_surface_mesh_to_o3d(lod2, SURFACE_COLORS))
    if args.view in ("lidar", "overlay", "all"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(_height_colors(pts))
        geoms.append(pcd)
    if args.view == "all":
        geoms.append(_mesh_to_lineset(lod1, (0.2, 0.2, 0.2)))

    if not geoms:
        print("no geometry to show", file=sys.stderr)
        sys.exit(1)

    print(f"scene={scene_dir.name} building={building_id}")
    print(f"  LOD1: verts={lod1.num_vertices()} faces={lod1.num_faces()} "
          f"height={lod1.height():.2f}")
    print(f"  LOD2: verts={lod2.num_vertices()} faces={lod2.num_faces()} "
          f"height={lod2.height():.2f} roof={lod2_meta['roof_type']}")
    print(f"  LiDAR building points: {len(pts)}")

    if args.save:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for g in geoms:
            vis.add_geometry(g)
        vis.get_render_option().point_size = args.point_size
        vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save, do_render=True)
        vis.destroy_window()
        print(f"saved {args.save}")
        return

    o3d.visualization.draw_geometries(geoms, window_name=f"building {building_id}",
                                      point_size=args.point_size)


if __name__ == "__main__":
    main()

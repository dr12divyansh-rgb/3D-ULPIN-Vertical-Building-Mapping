"""Open3D-based point cloud viewer (debugging / inspection).

Usage:
    python synthetic_city/scripts/view_pointcloud.py --input <file.ply|scene_dir>
        [--color semantic|height] [--save out.png] [--point-size 1.5]

If ``--input`` points at a scene directory, ``pointcloud.ply`` and
``ground_truth.json`` are used (for the semantic legend). ``--save`` renders
an offscreen image instead of opening a window (useful on headless machines).
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
from synthetic_city.core.labels import CLASS_NAMES, CLASS_COLORS  # noqa: E402


def _resolve_input(path):
    p = Path(path)
    if p.is_dir():
        # Step 2 layout: <scene>/lidar/pointcloud.ply (fallback to legacy root)
        for cand in (p / "lidar" / "pointcloud.ply", p / "pointcloud.ply"):
            if cand.exists():
                return cand, None
        return p / "lidar" / "pointcloud.ply", None
    return p, None


def main():
    ap = argparse.ArgumentParser(description="View a synthetic point cloud with Open3D.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--color", choices=["semantic", "height"], default="semantic")
    ap.add_argument("--save", default=None, help="render to PNG instead of opening a window")
    ap.add_argument("--point-size", type=float, default=1.5)
    args = ap.parse_args()

    pc_path, gt_path = _resolve_input(args.input)
    if not pc_path.exists():
        print(f"point cloud not found: {pc_path}", file=sys.stderr)
        sys.exit(1)

    data, _ = ply_io.read_pointcloud(pc_path)
    xyz = np.column_stack([np.asarray(data["x"], dtype=np.float64),
                           np.asarray(data["y"], dtype=np.float64),
                           np.asarray(data["z"], dtype=np.float64)])
    class_id = np.asarray(data["class_id"], dtype=np.int32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if args.color == "semantic":
        palette = np.array([CLASS_COLORS[n] for n in CLASS_NAMES], dtype=np.float64) / 255.0
        colors = palette[class_id]
    else:
        zmin, zmax = xyz[:, 2].min(), xyz[:, 2].max()
        t = (xyz[:, 2] - zmin) / (zmax - zmin + 1e-9)
        colors = np.stack([t, 1.0 - t, 0.3 * np.ones_like(t)], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if args.save:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = args.point_size
        opt.background_color = np.array([0.05, 0.05, 0.05])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save, do_render=True)
        vis.destroy_window()
        print(f"saved {args.save}")
        return

    print("Semantic legend:")
    for i, name in enumerate(CLASS_NAMES):
        r, g, b = CLASS_COLORS[name]
        print(f"  {i} {name:<10} rgb=({r},{g},{b})")
    print(f"points: {len(xyz)}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="synthetic_city viewer")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

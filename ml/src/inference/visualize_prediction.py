"""Visualise semantic segmentation predictions with Open3D.

Usage:
    python ml/src/inference/visualize_prediction.py --scene <scene_dir> \
        [--checkpoint <best_model.pth>] [--view gt|pred|errors] [--save out.png]

Coloring:
    gt     : ground-truth semantic classes
    pred   : predicted semantic classes
    errors : grey = correct, red = misclassified

If no ``prediction.ply`` exists next to the scene and a checkpoint is given,
predictions are generated on the fly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

from synthetic_city.core import ply_io  # noqa: E402
from synthetic_city.core.labels import CLASS_NAMES, CLASS_COLORS  # noqa: E402
from ml.src.inference.predict import build_model_from_checkpoint, predict_on_cloud, save_prediction_ply  # noqa: E402
from ml.src.data.dataset import load_scene_pointcloud  # noqa: E402
from ml.src.utils import select_device  # noqa: E402


def _class_colors(num_classes):
    palette = np.zeros((num_classes, 3), dtype=np.float64)
    for i, name in enumerate(CLASS_NAMES):
        palette[i] = np.array(CLASS_COLORS[name]) / 255.0
    return palette


def main():
    ap = argparse.ArgumentParser(description="Visualise segmentation predictions.")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--view", choices=["gt", "pred", "errors"], default="errors")
    ap.add_argument("--save", default=None)
    ap.add_argument("--point-size", type=float, default=1.5)
    args = ap.parse_args()

    scene = Path(args.scene)
    if not scene.is_dir():
        print(f"scene not found: {scene}", file=sys.stderr)
        sys.exit(1)

    xyz, gt, _ = load_scene_pointcloud(scene)
    pred_path = scene / "prediction.ply"

    if args.view in ("pred", "errors") and not pred_path.exists():
        if not args.checkpoint:
            print("prediction.ply not found and no --checkpoint given; "
                  "showing ground truth instead.", file=sys.stderr)
            args.view = "gt"
        else:
            device = select_device("auto")
            model, ckpt = build_model_from_checkpoint(args.checkpoint, device)
            num_points = int(ckpt["config"]["dataset"]["num_points"])
            pred, conf = predict_on_cloud(model, xyz, device, num_points, 8)
            save_prediction_ply(pred_path, xyz, pred, conf)
            print(f"predictions saved to {pred_path}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    num_classes = len(CLASS_NAMES)
    if args.view == "gt":
        colors = _class_colors(num_classes)[gt]
    elif args.view == "pred":
        data, _ = ply_io.read_pointcloud(pred_path)
        pred = np.asarray(data["predicted_class"], dtype=np.int64)
        colors = _class_colors(num_classes)[pred]
    else:  # errors
        data, _ = ply_io.read_pointcloud(pred_path)
        pred = np.asarray(data["predicted_class"], dtype=np.int64)
        colors = np.full((len(gt), 3), 0.25, dtype=np.float64)   # grey = correct
        err = pred != gt
        colors[err] = (1.0, 0.1, 0.1)                            # red = error
        n_err = int(err.sum())
        print(f"misclassified points: {n_err} / {len(gt)} ({100 * n_err / len(gt):.2f}%)")

    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if args.view in ("gt", "pred"):
        print("legend:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {i} {name}")

    if args.save:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = args.point_size
        vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save, do_render=True)
        vis.destroy_window()
        print(f"saved {args.save}")
        return

    o3d.visualization.draw_geometries([pcd], window_name=f"{scene.name} ({args.view})",
                                      point_size=args.point_size)


if __name__ == "__main__":
    main()

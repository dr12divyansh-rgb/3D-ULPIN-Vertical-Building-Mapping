"""Step 4: deterministic LOD2 reconstruction from segmentation predictions.

Usage:
    python ml/src/reconstruction/reconstruct.py \\
        --checkpoint ml/results/experiment_001/best_model.pth \\
        --scene scene_00005 \\
        [--data   ml/data/synthetic]      # data root for scene-id lookup
        [--pred   <path/to/prediction.ply>]  # reuse existing, skip inference
        [--out    ml/results/experiment_001/reconstruction]

For each building in the scene the script:
  1. Selects points predicted as class=1 (building) with the correct building_id.
  2. Estimates height from the predicted building points.
  3. Extracts a footprint polygon (convex hull of near-ground points, or the
     metadata footprint as documented fallback).
  4. Applies a deterministic geometry reconstruction (flat / hip / shed).
  5. Writes a ``SurfaceMesh`` as OBJ and a per-building metadata JSON.
  6. Writes a scene-level ``reconstruction_metadata.json`` recording all
     algorithmic decisions.

Constraints
-----------
* READS:   lidar/pointcloud.ply, ground_truth/building.json,
           results/../prediction.ply (or runs inference fresh).
* DOES NOT READ:  ground_truth/lod2/  (GT LOD2 is for evaluation only).
* DOES NOT MODIFY:  anything in Steps 1-3 or the checkpoint.
* CPU-only; no CUDA required.
* Deterministic: same inputs + same seed → same output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from synthetic_city.core import ply_io  # noqa: E402
from synthetic_city.core.mesh import SurfaceMesh  # noqa: E402
from ml.src.reconstruction.footprint import extract_footprint_convex  # noqa: E402
from ml.src.reconstruction.geometry import (  # noqa: E402
    build_flat_roof_mesh,
    build_hip_roof_mesh,
    build_shed_roof_mesh,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_lidar(scene_dir: Path) -> dict:
    data, _ = ply_io.read_pointcloud(scene_dir / "lidar" / "pointcloud.ply")
    return data


def _load_prediction(pred_path: Path) -> dict:
    data, _ = ply_io.read_pointcloud(pred_path)
    return data


def _load_building_meta(scene_dir: Path) -> dict:
    with open(scene_dir / "ground_truth" / "building.json", encoding="utf-8") as f:
        return json.load(f)


def _run_inference(checkpoint: Path, scene_dir: Path, num_points: int, batch_size: int) -> Path:
    """Run the Step-3 inference pipeline and return the prediction PLY path."""
    from ml.src.inference.predict import build_model_from_checkpoint, predict_on_cloud, save_prediction_ply
    from ml.src.data.dataset import load_scene_pointcloud
    from ml.src.utils import select_device

    device = select_device("auto")
    model, ckpt = build_model_from_checkpoint(str(checkpoint), device)
    cfg = ckpt["config"]
    np_ = int(cfg["dataset"]["num_points"])

    xyz, _, _ = load_scene_pointcloud(scene_dir)
    pred, conf = predict_on_cloud(model, xyz, device, np_, batch_size)

    out_path = scene_dir / "prediction.ply"
    save_prediction_ply(out_path, xyz, pred, conf)
    print(f"[reconstruct] inference complete → {out_path}")
    return out_path


def _find_scene_dir(scene_arg: str, data_root: Path) -> Path:
    p = Path(scene_arg)
    if p.is_dir():
        return p.resolve()
    for split in ("train", "validation", "test", "debug"):
        cand = data_root / split / scene_arg
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"Scene not found: {scene_arg!r}")


# --------------------------------------------------------------------------- #
# Per-building reconstruction                                                  #
# --------------------------------------------------------------------------- #

def reconstruct_building(
    bid: int,
    lidar_data: dict,
    pred_data: dict,
    bldg_meta: dict,
) -> tuple[SurfaceMesh, dict]:
    """Build a predicted LOD2 SurfaceMesh for one building.

    Returns (mesh, decisions) where ``decisions`` records every algorithmic
    choice made (footprint source, height source, roof type source, …).
    """
    # Build numpy arrays aligned point-for-point
    xyz = np.column_stack([
        np.asarray(lidar_data["x"], dtype=np.float32),
        np.asarray(lidar_data["y"], dtype=np.float32),
        np.asarray(lidar_data["z"], dtype=np.float32),
    ])
    lidar_bid  = np.asarray(lidar_data["building_id"], dtype=np.int64)
    pred_class = np.asarray(pred_data["predicted_class"], dtype=np.int64)

    # Building points: predicted class=1 (building) AND correct building_id
    building_mask = (pred_class == 1) & (lidar_bid == bid)
    pts = xyz[building_mask]
    n_pred_bldg = int(building_mask.sum())

    decisions: dict = {
        "building_id": bid,
        "n_points_total": int(len(xyz)),
        "n_predicted_building_in_scene": int((pred_class == 1).sum()),
        "n_points_this_building": n_pred_bldg,
    }

    # ------------------------------------------------------------------ #
    # Height estimation from points                                        #
    # ------------------------------------------------------------------ #
    if n_pred_bldg >= 3:
        base_z = float(pts[:, 2].min())
        top_z  = float(pts[:, 2].max())
        decisions["height_source"] = "points"
    else:
        # Fall back to metadata
        base_z = 0.0
        top_z  = float(bldg_meta["total_height"])
        decisions["height_source"] = "metadata_fallback_too_few_points"

    decisions["base_z"]       = round(base_z, 4)
    decisions["top_z"]        = round(top_z,  4)
    decisions["height_pred"]  = round(top_z - base_z, 4)
    decisions["height_gt"]    = round(float(bldg_meta["total_height"]), 4)

    # ------------------------------------------------------------------ #
    # Footprint                                                            #
    # ------------------------------------------------------------------ #
    roof_type = bldg_meta["roof_type"]
    decisions["roof_type"]        = roof_type
    decisions["roof_type_source"] = "metadata"

    if roof_type in ("hip", "shed"):
        # Rectangle footprint — use metadata directly (documented choice)
        footprint_xy = [(float(p[0]), float(p[1]))
                        for p in bldg_meta["footprint_world"]]
        decisions["footprint_source"] = "metadata_rectangle_for_sloped_roof"
    else:
        # Derive from points; fall back to metadata if hull is degenerate
        hull, src = extract_footprint_convex(pts)
        if len(hull) >= 3:
            footprint_xy = hull
            decisions["footprint_source"] = src
        else:
            footprint_xy = [(float(p[0]), float(p[1]))
                            for p in bldg_meta["footprint_world"]]
            decisions["footprint_source"] = "metadata_fallback_hull_degenerate"

    decisions["footprint_n_vertices"] = len(footprint_xy)

    # ------------------------------------------------------------------ #
    # Build mesh                                                           #
    # ------------------------------------------------------------------ #
    if roof_type == "flat":
        mesh = build_flat_roof_mesh(footprint_xy, base_z, top_z)
        decisions["eave_z"] = round(top_z, 4)
        decisions["peak_z"] = round(top_z, 4)

    elif roof_type == "hip":
        wall_h = float(bldg_meta["wall_height"])
        eave_z = base_z + wall_h
        peak_z = base_z + float(bldg_meta["total_height"])
        mesh = build_hip_roof_mesh(footprint_xy, base_z, eave_z, peak_z)
        decisions["eave_z"] = round(eave_z, 4)
        decisions["peak_z"] = round(peak_z, 4)
        decisions["wall_height_from_meta"] = round(wall_h, 4)
        decisions["roof_height_from_meta"]  = round(float(bldg_meta["roof_height"]), 4)

    elif roof_type in ("shed", "gable"):
        wall_h = float(bldg_meta["wall_height"])
        eave_z = base_z + wall_h
        peak_z = base_z + float(bldg_meta["total_height"])
        mesh = build_shed_roof_mesh(footprint_xy, base_z, eave_z, peak_z,
                                    xyz_building=pts if n_pred_bldg >= 2 else None)
        decisions["eave_z"] = round(eave_z, 4)
        decisions["peak_z"] = round(peak_z, 4)
        decisions["wall_height_from_meta"] = round(wall_h, 4)
        decisions["roof_height_from_meta"]  = round(float(bldg_meta["roof_height"]), 4)

    else:
        # Unknown roof type — flat fallback
        mesh = build_flat_roof_mesh(footprint_xy, base_z, top_z)
        decisions["footprint_source"] += "_unknown_roof_flat_fallback"

    decisions["mesh_vertices"] = mesh.num_vertices()
    decisions["mesh_faces"]    = mesh.num_faces()
    decisions["mesh_watertight"] = mesh.is_watertight()
    decisions["mesh_non_manifold_edges"] = mesh.non_manifold_edge_count()

    return mesh, decisions


# --------------------------------------------------------------------------- #
# Scene reconstruction                                                         #
# --------------------------------------------------------------------------- #

def reconstruct_scene(
    scene_dir: Path,
    pred_path: Path,
    out_dir: Path,
) -> dict:
    """Reconstruct LOD2 for all buildings in one scene.

    Reads: lidar PLY, prediction PLY, ground_truth/building.json.
    Writes: out_dir/building_XXXXXX.obj, out_dir/reconstruction_metadata.json.
    Does NOT read ground_truth/lod2/.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    lidar_data = _load_lidar(scene_dir)
    pred_data  = _load_prediction(pred_path)
    bldg_json  = _load_building_meta(scene_dir)
    buildings  = bldg_json["buildings"]

    if len(lidar_data.get("x", [])) != len(pred_data.get("x", [])):
        raise ValueError(
            f"Point count mismatch: lidar {len(lidar_data['x'])} "
            f"vs prediction {len(pred_data['x'])}. "
            "Ensure the prediction was made from this scene's LiDAR file."
        )

    all_decisions = []
    for bm in buildings:
        bid = int(bm["building_id"])
        mesh, decisions = reconstruct_building(bid, lidar_data, pred_data, bm)

        obj_name = f"building_{bid:06d}.obj"
        mesh.to_obj(out_dir / obj_name)

        decisions["obj"] = obj_name
        all_decisions.append(decisions)

        print(
            f"  building {bid:6d}: {decisions['n_points_this_building']:6d} pts "
            f"| roof={decisions['roof_type']:4s} "
            f"| fp={decisions['footprint_source']:30s} "
            f"| h_pred={decisions['height_pred']:.1f}m  h_gt={decisions['height_gt']:.1f}m "
            f"| watertight={decisions['mesh_watertight']}"
        )

    meta = {
        "scene": scene_dir.name,
        "prediction_ply": str(pred_path),
        "n_buildings": len(buildings),
        "buildings": all_decisions,
    }
    with open(out_dir / "reconstruction_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[reconstruct] wrote {len(buildings)} buildings to {out_dir}")
    return meta


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Step 4: deterministic LOD2 reconstruction.")
    ap.add_argument("--checkpoint", default="ml/results/experiment_001/best_model.pth",
                    help="Path to trained PointNet++ checkpoint (for inference if no --pred).")
    ap.add_argument("--scene", required=True,
                    help="Scene directory path or scene_id (looked up under --data).")
    ap.add_argument("--data", default="ml/data/synthetic",
                    help="Dataset root directory (for scene-id lookup).")
    ap.add_argument("--pred", default=None,
                    help="Path to an existing prediction.ply to reuse (skips inference).")
    ap.add_argument("--out", default=None,
                    help="Output directory. Defaults to "
                         "ml/results/experiment_001/reconstruction/<scene_id>/predicted_lod2/.")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    data_root = ROOT / args.data
    scene_dir = _find_scene_dir(args.scene, data_root)
    print(f"[reconstruct] scene: {scene_dir}")

    # Derive experiment directory from checkpoint path so defaults are self-consistent
    checkpoint = (ROOT / args.checkpoint).resolve()
    exp_dir = checkpoint.parent

    # Locate or generate prediction PLY
    if args.pred:
        pred_path = Path(args.pred)
    else:
        # Prefer a prediction already saved by evaluate.py in this experiment's dir
        default_pred = exp_dir / "predictions" / scene_dir.name / "prediction.ply"
        if default_pred.exists():
            pred_path = default_pred
            print(f"[reconstruct] reusing existing prediction → {pred_path}")
        else:
            # Run inference fresh
            pred_path = _run_inference(checkpoint, scene_dir, num_points=4096, batch_size=args.batch_size)

    if not pred_path.exists():
        print(f"[reconstruct] error: prediction file not found: {pred_path}", file=sys.stderr)
        return 1

    # Output directory
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = exp_dir / "reconstruction" / scene_dir.name / "predicted_lod2"

    reconstruct_scene(scene_dir, pred_path, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

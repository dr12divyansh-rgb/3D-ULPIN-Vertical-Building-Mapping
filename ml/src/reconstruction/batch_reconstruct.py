"""
Batch inference + reconstruction across all synthetic scenes.

Runs the experiment_001 checkpoint on every scene that lacks a prediction,
then reconstructs LOD2 buildings from every scene's prediction, and
finally merges all individual building OBJ files into one combined city OBJ.

Usage:
    python ml/src/reconstruction/batch_reconstruct.py \
        --data    ml/data/synthetic \
        --checkpoint ml/results/experiment_001/best_model.pth \
        --out     ml/results/city_reconstruction/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np


def _scene_dirs(data_root: Path) -> list[Path]:
    """Return all scene directories across train/validation/test splits."""
    scenes = []
    for split in ("train", "validation", "test"):
        split_dir = data_root / split
        if split_dir.exists():
            scenes += sorted(split_dir.iterdir())
    return [s for s in scenes if s.is_dir() and (s / "lidar" / "pointcloud.ply").exists()]


def run_inference(checkpoint: Path, scene_dir: Path, pred_out: Path) -> bool:
    """Run PointNet++ inference on one scene. Returns True on success."""
    from ml.src.inference.predict import build_model_from_checkpoint, predict_on_cloud, save_prediction_ply
    from ml.src.data.dataset import load_scene_pointcloud
    from ml.src.utils import select_device

    try:
        device   = select_device("auto")
        model, ckpt = build_model_from_checkpoint(str(checkpoint), device)
        cfg      = ckpt["config"]
        n_pts    = int(cfg["dataset"]["num_points"])
        xyz, _, _ = load_scene_pointcloud(scene_dir)
        pred, conf = predict_on_cloud(model, xyz, device, n_pts, batch_size=8)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        save_prediction_ply(pred_out, xyz, pred, conf)
        return True
    except Exception as e:
        print(f"    ERROR during inference: {e}")
        return False


def reconstruct_scene(pred_ply: Path, out_dir: Path, cell_size: float, min_pts: int) -> list[dict]:
    """
    Run the reconstruction algorithm on one scene's prediction PLY.
    Returns list of per-building summary dicts (empty on failure).
    """
    from ml.src.reconstruction.reconstruct_from_prediction import (
        load_prediction, separate_instances, analyze_building,
        generate_mesh, write_obj,
    )

    try:
        xyz, pred_class = load_prediction(pred_ply)
        bldg_xyz = xyz[pred_class == 1]
        if len(bldg_xyz) < min_pts:
            return []

        instances = separate_instances(bldg_xyz, cell_size, min_pts)
        rows = []
        for i, idx in enumerate(instances):
            pts  = bldg_xyz[idx]
            info = analyze_building(pts)
            if len(info["footprint"]) < 3 or info["height"] < 1.0:
                continue
            mb = generate_mesh(info)
            if not mb.verts:
                continue

            name    = f"{pred_ply.parent.name}_b{i+1:03d}"
            obj_out = out_dir / "buildings" / f"{name}.obj"
            write_obj(obj_out, mb, group=name)
            fp  = info["footprint"]
            cx  = round(sum(p[0] for p in fp) / len(fp), 3)
            cy  = round(sum(p[1] for p in fp) / len(fp), 3)
            rows.append({
                "name":         name,
                "scene":        pred_ply.parent.name,
                "n_points":     info["n_points"],
                "height_m":     round(info["height"], 2),
                "area_m2":      round(info["footprint_area"], 1),
                "roof_type":    "gable" if info["is_pitched"] else "flat",
                "z_ground":     round(info["z_ground"], 3),
                "z_peak":       round(info["z_peak"], 3),
                "centroid_x":   cx,
                "centroid_y":   cy,
                "footprint_verts": [[round(p[0],2), round(p[1],2)] for p in fp],
                "n_verts":      len(mb.verts),
                "n_faces":      len(mb.faces),
                "_mb":          mb,
                "_name":        name,
            })
        return rows
    except Exception as e:
        print(f"    ERROR during reconstruction: {e}")
        return []


def merge_city(rows: list[dict], out_dir: Path) -> Path:
    """Merge all per-building OBJ data into one city scene.obj + scene.mtl."""
    city_obj = out_dir / "city_500_buildings.obj"
    city_mtl = out_dir / "city_500_buildings.mtl"

    rng = np.random.default_rng(26011)
    mtl_lines = ["# SIH 26011 - city building materials\n\n"]
    obj_lines  = [
        "# SIH 26011 - 3D ULPIN City Reconstruction\n",
        f"# {len(rows)} buildings reconstructed from AI segmentation\n\n",
        "mtllib city_500_buildings.mtl\n\n",
    ]

    v_offset = 0
    for row in rows:
        mb   = row["_mb"]
        name = row["_name"]

        r, g, b = rng.uniform(0.25, 0.95, 3)
        mtl_lines += [f"newmtl {name}\n", f"Kd {r:.3f} {g:.3f} {b:.3f}\nKa 0.1 0.1 0.1\n\n"]

        obj_lines.append(f"g {name}\nusemtl {name}\n")
        for x, y, z in mb.verts:
            obj_lines.append(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for a, b_, c in mb.faces:
            obj_lines.append(f"f {a+v_offset} {b_+v_offset} {c+v_offset}\n")
        v_offset += len(mb.verts)
        obj_lines.append("\n")

    city_obj.write_text("".join(obj_lines), encoding="utf-8")
    city_mtl.write_text("".join(mtl_lines), encoding="utf-8")
    return city_obj


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch inference + LOD2 reconstruction across all synthetic scenes.")
    ap.add_argument("--data",       default="ml/data/synthetic",
                    help="Synthetic dataset root (default: ml/data/synthetic)")
    ap.add_argument("--checkpoint", default="ml/results/experiment_001/best_model.pth",
                    help="Trained checkpoint for inference")
    ap.add_argument("--pred-root",  default="ml/results/experiment_001/predictions",
                    help="Directory where per-scene prediction PLYs live / are written")
    ap.add_argument("--out",        default="ml/results/city_reconstruction",
                    help="Output directory")
    ap.add_argument("--cell-size",  type=float, default=0.8,
                    help="Grid cell size (m) for instance separation (default 0.8)")
    ap.add_argument("--min-points", type=int,   default=40,
                    help="Min points per building instance (default 40)")
    ap.add_argument("--skip-inference", action="store_true",
                    help="Skip inference — only reconstruct from existing predictions")
    args = ap.parse_args()

    data_root  = ROOT / args.data
    checkpoint = ROOT / args.checkpoint
    pred_root  = ROOT / args.pred_root
    out_dir    = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "buildings").mkdir(exist_ok=True)

    scenes = _scene_dirs(data_root)
    print(f"\nSIH 26011 - Batch City Reconstruction")
    print(f"  {len(scenes)} scenes found in {data_root}")
    print(f"  checkpoint : {checkpoint}")
    print(f"  output     : {out_dir}\n")

    all_rows: list[dict] = []
    t0 = time.time()

    for i, scene_dir in enumerate(scenes):
        scene_name = scene_dir.name
        pred_ply   = pred_root / scene_name / "prediction.ply"

        print(f"[{i+1}/{len(scenes)}] {scene_name}")

        # ── Inference ────────────────────────────────────────────────────
        if not pred_ply.exists():
            if args.skip_inference:
                print(f"  skip (no prediction + --skip-inference)")
                continue
            print(f"  running inference...", end=" ", flush=True)
            t_inf = time.time()
            ok = run_inference(checkpoint, scene_dir, pred_ply)
            if not ok:
                continue
            print(f"done ({time.time()-t_inf:.0f}s)")
        else:
            print(f"  reusing existing prediction")

        # ── Reconstruction ────────────────────────────────────────────────
        print(f"  reconstructing...", end=" ", flush=True)
        rows = reconstruct_scene(pred_ply, out_dir, args.cell_size, args.min_points)
        all_rows.extend(rows)
        n_flat   = sum(1 for r in rows if r["roof_type"] == "flat")
        n_gable  = sum(1 for r in rows if r["roof_type"] == "gable")
        print(f"{len(rows)} buildings  ({n_flat} flat, {n_gable} gable)  "
              f"heights: {min((r['height_m'] for r in rows), default=0):.0f}–"
              f"{max((r['height_m'] for r in rows), default=0):.0f}m")

    # ── Merge city ────────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo buildings reconstructed.")
        return 1

    print(f"\nMerging {len(all_rows)} buildings into city OBJ...", end=" ", flush=True)
    city_obj = merge_city(all_rows, out_dir)
    print("done")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "total_buildings":  len(all_rows),
        "n_flat":           sum(1 for r in all_rows if r["roof_type"] == "flat"),
        "n_gable":          sum(1 for r in all_rows if r["roof_type"] == "gable"),
        "min_height_m":     round(min(r["height_m"] for r in all_rows), 2),
        "max_height_m":     round(max(r["height_m"] for r in all_rows), 2),
        "total_time_s":     round(time.time() - t0, 1),
        "scenes_processed": len(scenes),
        "buildings":        [{k: v for k, v in r.items() if not k.startswith("_")}
                             for r in all_rows],
    }
    (out_dir / "city_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"  Total buildings : {summary['total_buildings']}")
    print(f"  Flat roofs      : {summary['n_flat']}")
    print(f"  Gable roofs     : {summary['n_gable']}")
    print(f"  Height range    : {summary['min_height_m']}m – {summary['max_height_m']}m")
    print(f"  Time elapsed    : {summary['total_time_s']:.0f}s")
    print(f"  City OBJ        : {city_obj}")
    print(f"{'='*55}")
    print(f"\nOpen {city_obj} in CloudCompare or Blender.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

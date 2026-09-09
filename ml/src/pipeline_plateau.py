"""PLATEAU-equivalent end-to-end pipeline for SIH 26011.

One command: raw PLY in → segmentation → LOD2 reconstruction → CityGML out.

Usage (synthetic scene, best quality):
    python ml/src/pipeline_plateau.py \\
        --scene scene_00005 \\
        --checkpoint ml/results/experiment_001/best_model.pth \\
        --out ml/results/plateau_demo

Usage (any raw PLY, real-world):
    python ml/src/pipeline_plateau.py \\
        --input path/to/cloud.ply \\
        --checkpoint ml/results/experiment_005/best_model.pth \\
        --out ml/results/plateau_output

Outputs written to --out/:
    city_model.gml          CityGML 2.0 city model (PLATEAU-equivalent)
    ulpin_registry.csv      ULPIN ↔ building attribute table
    building_XXXXXX.obj     Reconstructed LOD2 meshes (if --scene used)
    reconstruction_metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.src.export.citygml import export_scene
from ml.src.export.ulpin import write_registry


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_scene_dir(scene_arg: str, data_root: Path) -> Path:
    p = Path(scene_arg)
    if p.is_dir():
        return p.resolve()
    for split in ("train", "validation", "test", "debug"):
        cand = data_root / split / scene_arg
        if cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"Scene not found: {scene_arg!r}")


def _run_inference_on_scene(checkpoint: Path, scene_dir: Path) -> Path:
    from ml.src.inference.predict import (
        build_model_from_checkpoint, predict_on_cloud, save_prediction_ply,
    )
    from ml.src.data.dataset import load_scene_pointcloud
    from ml.src.utils import select_device

    device = select_device("auto")
    model, ckpt = build_model_from_checkpoint(str(checkpoint), device)
    num_points = int(ckpt["config"]["dataset"]["num_points"])

    xyz, _, _ = load_scene_pointcloud(scene_dir)
    pred, conf = predict_on_cloud(model, xyz, device, num_points, batch_size=4)

    out = scene_dir / "prediction.ply"
    save_prediction_ply(out, xyz, pred, conf)
    return out


def _run_inference_on_ply(checkpoint: Path, ply_path: Path) -> Path:
    from ml.src.inference.predict_raw import read_raw_ply, predict_raw, save_colored_ply
    from ml.src.inference.predict import build_model_from_checkpoint
    from ml.src.utils import select_device

    device = select_device("auto")
    model, ckpt = build_model_from_checkpoint(str(checkpoint), device)
    cfg = ckpt["config"]
    input_dim = 6 if cfg["dataset"].get("input_mode", "xyz") == "xyzrgb" else 3
    num_points = int(cfg["dataset"]["num_points"])

    xyz, rgb = read_raw_ply(ply_path)
    pred, conf = predict_raw(model, xyz, rgb, input_dim, device, num_points)

    out = ply_path.parent / (ply_path.stem + "_prediction.ply")
    save_colored_ply(out, xyz, pred, conf)
    return out


def _reconstruct_scene(scene_dir: Path, pred_path: Path, out_dir: Path) -> dict:
    from ml.src.reconstruction.reconstruct import reconstruct_scene
    return reconstruct_scene(scene_dir, pred_path, out_dir)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="PLATEAU-equivalent pipeline: PLY/scene → CityGML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="Scene ID or path (e.g. scene_00005) — uses full metadata")
    src.add_argument("--input", help="Raw PLY file path (any point cloud)")
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    ap.add_argument("--data", default="ml/data/synthetic", help="Synthetic dataset root")
    ap.add_argument("--pred", default=None,
                    help="Pre-existing prediction PLY (skip inference)")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  SIH 26011 — PLATEAU-Equivalent Pipeline")
    print("═" * 60)
    print(f"  checkpoint : {checkpoint}")
    print(f"  output     : {out_dir}")

    if args.scene:
        # ── Synthetic scene path ──────────────────────────────────────────
        data_root = (ROOT / args.data).resolve()
        scene_dir = _find_scene_dir(args.scene, data_root)
        print(f"  scene      : {scene_dir}")

        # 1) Inference
        if args.pred:
            pred_path = Path(args.pred)
            print(f"\n[1/3] Reusing prediction → {pred_path}")
        else:
            default_pred = (
                ROOT / "ml" / "results" / checkpoint.parent.name /
                "predictions" / scene_dir.name / "prediction.ply"
            )
            if default_pred.exists():
                pred_path = default_pred
                print(f"\n[1/3] Reusing existing prediction → {pred_path}")
            else:
                print("\n[1/3] Running inference …")
                pred_path = _run_inference_on_scene(checkpoint, scene_dir)
                print(f"      → {pred_path}")

        # 2) LOD2 Reconstruction
        lod2_dir = out_dir / "lod2"
        print("\n[2/3] Running LOD2 reconstruction …")
        meta = _reconstruct_scene(scene_dir, pred_path, lod2_dir)
        meta_json = lod2_dir / "reconstruction_metadata.json"

    else:
        # ── Raw PLY path ──────────────────────────────────────────────────
        ply_path = Path(args.input)
        if not ply_path.exists():
            print(f"ERROR: input file not found: {ply_path}", file=sys.stderr)
            return 1
        print(f"  input      : {ply_path}")
        print("\n[1/3] Running inference on raw PLY …")
        pred_path = _run_inference_on_ply(checkpoint, ply_path)
        print(f"      → {pred_path}")

        # No building.json metadata for raw PLY → skip reconstruction, export prediction only
        print("\n[2/3] Skipping LOD2 reconstruction (no scene metadata for raw PLY).")
        print("      Use --scene for full CityGML export with LOD2 geometry.")
        print("\n[3/3] No CityGML export possible without scene metadata.")
        print("\n      Prediction saved to:", pred_path)
        print("      Open in CloudCompare or MeshLab to visualize.")
        print("\n" + "═" * 60)
        return 0

    # 3) CityGML Export
    print("\n[3/3] Exporting CityGML 2.0 …")
    gml_path = out_dir / "city_model.gml"
    registry = export_scene(meta_json, lod2_dir, gml_path)

    # Write ULPIN registry
    csv_path = out_dir / "ulpin_registry.csv"
    write_registry(registry, csv_path)

    # Summary
    print("\n" + "═" * 60)
    print(f"  ✓ CityGML    → {gml_path}")
    print(f"  ✓ Registry   → {csv_path}")
    print(f"  ✓ LOD2 OBJs  → {lod2_dir}/")
    print(f"  ✓ Buildings  : {len(registry)}")
    print()
    for r in registry[:5]:
        print(f"    {r['ulpin']}  h={r['height_pred_m']:.1f}m  roof={r['roof_type']}")
    if len(registry) > 5:
        print(f"    … and {len(registry) - 5} more")
    print()
    print("  Next steps:")
    print("    1. python viewer/build_index.py")
    print("    2. python -m http.server 8000  (from project root)")
    print("    3. Open http://localhost:8000/viewer/ in your browser")
    print("    4. Load city_model.gml in FZKViewer (free) for GIS validation")
    print("═" * 60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

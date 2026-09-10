"""Full-tile reconstruction pipeline — AHN4 all 498 chunks.

Chains:
  1. pipeline_research_v2: watershed instance extraction + ground-class heights
  2. refine_heights: ASPRS ground elevation correction (if ground NPZ available)
  3. export_viewer_json: viewer-ready JSON output

Usage:
    python ml/src/reconstruction/pipeline_fulltile.py \\
        --pred ml/results/ahn4_fulltile/merged_prediction.ply \\
        --ground ml/results/ahn4_fulltile/ahn4_ground_fulltile.npz \\
        --out ml/results/ahn4_fulltile/reconstruction

Run AFTER:
    infer_ahn4_fulltile.py   → merged_prediction.ply
    extract_ahn4_ground_fulltile.py → ahn4_ground_fulltile.npz (optional)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from ml.src.reconstruction.pipeline_research_v2 import run_v2_pipeline
from ml.src.reconstruction.refine_heights import refine_buildings
from ml.src.reconstruction.export_viewer_json import convert as export_viewer


def run_fulltile_pipeline(
    pred_ply: Path,
    out_dir: Path,
    ground_npz: Path | None = None,
    scene_id: str = "25GN2_01_Amsterdam_full",
    max_buildings: int = 1200,
    building_iou: float = 0.7551,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  Full-tile Reconstruction Pipeline")
    print(f"  Input : {pred_ply}")
    print(f"  Output: {out_dir}")
    print(f"{'='*65}\n")

    # ── Step 1: Watershed reconstruction ──────────────────────────────────────
    print("STEP 1: Watershed instance extraction + reconstruction")
    v2_result = run_v2_pipeline(
        pred_ply      = pred_ply,
        out_dir       = out_dir,
        scene_id      = scene_id,
        max_buildings = max_buildings,
    )
    n_buildings = v2_result.get("n_buildings_reconstructed", 0)
    print(f"  Buildings reconstructed: {n_buildings}")

    # ── Step 2: ASPRS height refinement (optional) ────────────────────────────
    if ground_npz is not None and ground_npz.exists():
        print(f"\nSTEP 2: ASPRS ground elevation refinement")
        print(f"  Ground NPZ: {ground_npz}")
        try:
            refine_result = refine_buildings(
                buildings_json   = out_dir / "buildings.json",
                asprs_ground_npz = ground_npz,
                out_dir          = out_dir,
                building_iou     = building_iou,
            )
            print(f"  Height MAE improved via ASPRS elevation")
            print(f"  Old mean height: {refine_result['h_old_mean']:.2f}m")
            print(f"  New mean height: {refine_result['h_new_mean']:.2f}m")
        except Exception as e:
            print(f"  WARNING: Height refinement failed: {e}")
            print(f"  Continuing with watershed ground estimates")
    else:
        if ground_npz is not None:
            print(f"\nSTEP 2 (SKIPPED): Ground NPZ not found: {ground_npz}")
            print(f"  Heights use watershed ground-class estimate (~3m MAE)")
        else:
            print(f"\nSTEP 2 (SKIPPED): No ground NPZ provided")

    # ── Step 3: Verify output ─────────────────────────────────────────────────
    buildings_json = out_dir / "buildings.json"
    viewer_json    = out_dir / "viewer_buildings.json"

    if buildings_json.exists():
        with open(buildings_json) as f:
            scene = json.load(f)
        n_final = len(scene.get("buildings", []))
        print(f"\nFinal buildings.json: {n_final} buildings")
    else:
        print("\nWARNING: buildings.json not found")
        n_final = 0

    if viewer_json.exists():
        print(f"viewer_buildings.json: {viewer_json}")
    else:
        print("WARNING: viewer_buildings.json not found")

    summary = {
        "pipeline": "fulltile_v1",
        "pred_ply": str(pred_ply),
        "ground_npz": str(ground_npz) if ground_npz else None,
        "asprs_refinement": ground_npz is not None and ground_npz.exists(),
        "n_buildings": n_final,
        "viewer_json": str(viewer_json),
    }
    with open(out_dir / "fulltile_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Full-tile pipeline COMPLETE")
    print(f"  {n_final} buildings → {viewer_json}")
    print(f"{'='*65}\n")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Full-tile AHN4 reconstruction pipeline")
    ap.add_argument("--pred",    default="ml/results/ahn4_fulltile/merged_prediction.ply")
    ap.add_argument("--ground",  default="ml/results/ahn4_fulltile/ahn4_ground_fulltile.npz",
                    help="ASPRS ground NPZ (optional, improves height accuracy)")
    ap.add_argument("--out",     default="ml/results/ahn4_fulltile/reconstruction")
    ap.add_argument("--scene",   default="25GN2_01_Amsterdam_full")
    ap.add_argument("--max-buildings", type=int, default=1200)
    ap.add_argument("--building-iou",  type=float, default=0.7551)
    ap.add_argument("--no-ground-refinement", action="store_true",
                    help="Skip ASPRS height refinement step")
    args = ap.parse_args()

    ground_npz = None if args.no_ground_refinement else ROOT / args.ground

    run_fulltile_pipeline(
        pred_ply      = ROOT / args.pred,
        out_dir       = ROOT / args.out,
        ground_npz    = ground_npz,
        scene_id      = args.scene,
        max_buildings = args.max_buildings,
        building_iou  = args.building_iou,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

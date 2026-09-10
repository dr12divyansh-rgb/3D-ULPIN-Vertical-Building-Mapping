"""Reconstruction Pipeline v2 — SIH 26011.

Dataset-agnostic building instance extraction, floor decomposition,
roof analysis, LOD1/LOD2-like reconstruction, and structured JSON output.

Usage:
    python ml/src/reconstruction/pipeline_v2.py \\
        --pred ml/results/dales2_500k_prediction.ply \\
        --out  ml/results/reconstruction_v2 \\
        --dataset DALES \\
        --scene  dales2_500k \\
        --model  "PointNet++ Exp006 (DALES fine-tuned, Building IoU=0.763)"

    python ml/src/reconstruction/pipeline_v2.py \\
        --pred ml/results/stpls3d_real/RealWorldData__OCCC_points/prediction_world.ply \\
        --out  ml/results/reconstruction_v2_occc \\
        --dataset STPLS3D \\
        --scene  RealWorldData__OCCC_points \\
        --model  "PointNet++ Exp005 (STPLS3D XYZRGB, Building IoU=0.659)"

Output (all in --out directory):
    buildings.json              -- full schema for all buildings
    buildings/BLDG_ID.obj       -- per-building LOD2-like mesh
    scene.obj                   -- all buildings combined
    prediction_meta.json        -- provenance of input PLY
    run_summary.json            -- statistics

This pipeline DOES NOT modify:
    - ml/results/stpls3d_real/
    - ml/results/experiment_005/ or experiment_006/
    - ml/results/reconstructed_buildings_*/
    - viewer/
    - Any other existing result directories
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# ── Core v2 modules ────────────────────────────────────────────────────────────
from ml.src.reconstruction.prediction_contract import (  # noqa: E402
    load_prediction_ply, save_prediction_metadata, BUILDING_CLASS,
)
from ml.src.reconstruction.instance_extraction import (  # noqa: E402
    extract_building_instances_adaptive,
    evaluate_instance_extraction,
    RECOMMENDED_CELL_SIZE, RECOMMENDED_MIN_POINTS,
)
from ml.src.reconstruction.floor_decomposition import (  # noqa: E402
    decompose_floors, DEFAULT_FLOOR_HEIGHT_M,
)
from ml.src.reconstruction.building_schema import (  # noqa: E402
    building_record, scene_record, save_buildings_json,
)
from ml.src.reconstruction.footprint import (  # noqa: E402
    extract_footprint_best, _polygon_area,
)
from ml.src.reconstruction.roof_analysis import (  # noqa: E402
    analyze_roof, roof_result_to_dict,
)

# ── Reuse existing geometry (mesh building only — no algorithm duplication) ──
from ml.src.reconstruction.reconstruct_from_prediction import (  # noqa: E402
    analyze_building, generate_mesh,
    write_obj, write_scene_obj, write_mtl,
    MeshBuilder,
)


# ── Footprint helpers ──────────────────────────────────────────────────────────

def _footprint_from_pts(
    pts: np.ndarray,
    source_dataset: str = "unknown",
) -> tuple[list[list[float]], float, int, str, str]:
    """Return (footprint_xy_list, area_m2, n_vertices, method, source).

    Uses raster boundary method (handles L/T/concave) with convex-hull fallback.
    """
    # Choose cell size based on dataset density
    cell_size = {"STPLS3D": 0.5, "DALES": 1.0, "AHN4": 0.75}.get(source_dataset, 0.5)
    poly, meta = extract_footprint_best(pts, cell_size=cell_size)
    if poly and len(poly) >= 3:
        area = meta.get("area_m2") or _polygon_area(poly)
        return [[x, y] for x, y in poly], float(area), len(poly), meta["method"], meta.get("source", "unknown")
    return [], 0.0, 0, "failed", "no_footprint"


# ── Elevation helpers ──────────────────────────────────────────────────────────

def _eave_z(pts: np.ndarray, z_ground: float, z_peak: float) -> float:
    """80th-percentile Z clamped to [55%, 92%] of building height."""
    z = pts[:, 2]
    height = z_peak - z_ground
    raw = float(np.percentile(z, 80))
    lo  = z_ground + 0.55 * height
    hi  = z_ground + 0.92 * height
    return float(np.clip(raw, lo, hi))


# ── Mesh builder wrapper ───────────────────────────────────────────────────────

def _build_mesh(pts: np.ndarray) -> tuple[MeshBuilder, dict]:
    """
    Build LOD2-like mesh for one building.

    Uses analyze_building for geometry (footprint, eave, peak, roof type).
    Returns (mesh, info_dict).
    """
    info = analyze_building(pts)
    mb   = generate_mesh(info)
    return mb, info


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    pred_ply:      Path,
    out_dir:       Path,
    dataset:       str = "unknown",
    scene_id:      str | None = None,
    ai_model:      str = "PointNet++",
    cell_size:     float | None = None,
    min_points:    int  | None = None,
    max_buildings: int  = 300,
    floor_height:  float = DEFAULT_FLOOR_HEIGHT_M,
    building_class: int = BUILDING_CLASS,
) -> dict:
    """
    Full v2 reconstruction pipeline.

    Args:
        pred_ply:       Path to prediction PLY (must have x, y, z, predicted_class).
        out_dir:        Output directory (will be created if absent).
        dataset:        Dataset name ("DALES", "STPLS3D", or "unknown").
        scene_id:       Scene identifier (defaults to PLY file stem).
        ai_model:       Description of the model used for segmentation.
        cell_size:      BFS grid cell size for instance separation (metres).
                        If None, uses dataset-appropriate default.
        min_points:     Minimum points per building instance.
                        If None, uses dataset-appropriate default.
        max_buildings:  Maximum buildings to reconstruct.
        floor_height:   Assumed floor-to-floor height for floor decomposition.
        building_class: Integer class label for buildings (default 1).

    Returns:
        Summary dict with counts and paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bldg_dir = out_dir / "buildings"
    bldg_dir.mkdir(exist_ok=True)

    scene_id = scene_id or pred_ply.stem

    # ── Apply dataset-appropriate defaults ────────────────────────────────
    if cell_size is None:
        cell_size = RECOMMENDED_CELL_SIZE.get(dataset, 1.5)
    if min_points is None:
        min_points = RECOMMENDED_MIN_POINTS.get(dataset, 30)

    print(f"\nSIH 26011 — Reconstruction Pipeline v2")
    print(f"  dataset     : {dataset}")
    print(f"  scene       : {scene_id}")
    print(f"  pred PLY    : {pred_ply}")
    print(f"  output      : {out_dir}")
    print(f"  cell_size   : {cell_size} m")
    print(f"  min_points  : {min_points}")
    print(f"  floor_height: {floor_height} m")
    print()

    # ── Phase 2: Load prediction (common contract) ────────────────────────
    print("Loading prediction PLY...")
    record = load_prediction_ply(pred_ply, scene_id=scene_id, source_dataset=dataset)
    print(f"  {record.n_points:,} points loaded")
    print(f"  class summary: {record.class_summary()}")

    save_prediction_metadata(record, out_dir / "prediction_meta.json")

    n_building_pts = int(record.building_mask(building_class).sum())
    print(f"  {n_building_pts:,} building-class points ({100*n_building_pts/record.n_points:.1f}%)")

    if n_building_pts == 0:
        print("\nERROR: No building-class points found. Check --building-class argument.")
        return {"error": "no_building_points"}

    building_xyz = record.building_xyz(building_class)

    # ── Phase 3: Building Instance Extraction (adaptive: DBSCAN → grid BFS) ─
    print(f"\nBuilding Instance Extraction (adaptive — DBSCAN preferred)...")
    instances, instance_method = extract_building_instances_adaptive(
        building_xyz,
        scene_id       = scene_id,
        source_dataset = dataset,
        cell_size      = cell_size,
        min_points     = min_points,
        max_instances  = max_buildings,
        prefer_dbscan  = True,
    )
    print(f"  {len(instances)} instances found (method: {instance_method})")
    inst_stats = evaluate_instance_extraction(instances, building_xyz, instance_method)
    print(f"  noise rejected: {inst_stats['noise_pct']:.1f}%  "
          f"min/max pts: {inst_stats['min_pts']}/{inst_stats['max_pts']}")
    if not instances:
        print("  No instances found. Try --cell-size 2.0 or --min-points 15")
        return {"error": "no_instances_found"}

    # ── Reconstruction loop ───────────────────────────────────────────────
    print(f"\nReconstructing {len(instances)} buildings...")
    print(f"  {'ID':<28} {'pts':>6}  {'h_m':>6}  {'floors':>6}  {'roof':<10}  {'area_m2':>8}  {'fp':>8}")
    print(f"  {'-'*82}")

    buildings_out: list[dict]  = []
    scene_meshes:  list        = []
    n_flat = n_gable = n_skip = 0

    for inst in instances:
        pts = building_xyz[inst.point_indices]
        bid = inst.building_id

        if len(pts) < 3:
            n_skip += 1
            continue

        # ── Footprint (Phase 2 improved — raster boundary) ───────────────
        fp_list, fp_area, fp_nverts, fp_method, fp_source = _footprint_from_pts(pts, dataset)
        if len(fp_list) < 3:
            n_skip += 1
            continue

        # ── Heights — use existing analyze_building for geometry ──────────
        info = analyze_building(pts)
        z_ground_geom = info["z_ground"]   # min Z for mesh
        z_peak_geom   = info["z_peak"]     # max Z for mesh
        eave_z_geom   = info["z_eave"]     # 80th pct clamped for mesh

        if info["height"] < 1.0:
            n_skip += 1
            continue

        # ── Heights — robust percentile for schema output (Phase 4) ───────
        ground_z, roof_z, height, floor_count, floors = decompose_floors(
            pts, bid, floor_height_m=floor_height,
        )

        # ── Roof analysis (Phase 7 — Open3D RANSAC) ───────────────────────
        roof_result = analyze_roof(pts, z_ground_geom, z_peak_geom, eave_z_geom)
        roof_type  = roof_result.roof_type
        is_pitched = roof_result.is_pitched

        # ── Mesh — LOD2-like (Phase 8) ────────────────────────────────────
        mb, _ = _build_mesh(pts)
        if not mb.verts:
            n_skip += 1
            continue

        obj_name    = f"{bid}.obj"
        obj_rel     = f"buildings/{obj_name}"
        write_obj(bldg_dir / obj_name, mb, group=bid)
        scene_meshes.append((bid, mb))

        if is_pitched:
            n_gable += 1
        else:
            n_flat += 1

        # ── Structured output (Phase 10) ───────────────────────────────────
        brec = building_record(
            building_id            = bid,
            source_dataset         = dataset,
            scene_id               = scene_id,
            instance_id            = inst.instance_id,
            footprint              = fp_list,
            footprint_area_m2      = fp_area,
            footprint_n_vertices   = fp_nverts,
            point_count            = inst.n_points,
            ground_z               = ground_z,
            roof_z                 = roof_z,
            height                 = height,
            eave_z                 = eave_z_geom,
            floor_count            = floor_count,
            floor_height_m_assumed = floor_height,
            floors                 = [f.to_dict() for f in floors],
            roof_type              = roof_type,
            is_pitched             = is_pitched,
            lod2_obj_file          = obj_rel,
            ai_model               = ai_model,
        )
        # Enrich provenance with improved method details
        brec["footprint_source"]  = f"LiDAR-derived-{fp_method}"
        brec["footprint_note"]    = (
            f"Method: {fp_method} (source: {fp_source}). "
            "Raster boundary tracing preserves concave shapes (L, T, courtyard). "
            "Convex hull used as fallback."
        )
        brec["provenance"]["instance_separation"] = f"LiDAR-derived-{instance_method}"
        brec["provenance"]["footprint"] = f"LiDAR-derived-{fp_method}"
        brec["provenance"]["roof_type"] = f"LiDAR-derived-{roof_result.method}"
        # Merge roof detail fields
        brec.update(roof_result_to_dict(roof_result))

        buildings_out.append(brec)

        print(f"  {bid:<28} {inst.n_points:>6}  {height:>6.1f}  {floor_count:>6}  "
              f"{roof_type:<10}  {fp_area:>8.0f}  {fp_method}")

    # ── Scene OBJ ─────────────────────────────────────────────────────────
    print(f"\nWriting scene OBJ...")
    scene_obj = out_dir / "scene.obj"
    scene_mtl = out_dir / "scene.mtl"
    write_scene_obj(scene_obj, scene_meshes)
    write_mtl(scene_mtl, len(scene_meshes))
    txt = scene_obj.read_text(encoding="utf-8")
    scene_obj.write_text("mtllib scene.mtl\n" + txt, encoding="utf-8")

    # ── Save buildings.json ───────────────────────────────────────────────
    run_params = {
        "cell_size_m":         cell_size,
        "min_points":          min_points,
        "max_buildings":       max_buildings,
        "floor_height_m":      floor_height,
        "building_class":      building_class,
        "instance_method":     instance_method,
        "footprint_method":    "raster-boundary-with-convex-fallback",
        "height_method":       "LiDAR-percentile-5th-99th",
        "roof_method":         "RANSAC-open3d-with-Zstd-fallback",
        "instance_stats":      inst_stats,
    }
    scene_doc = scene_record(
        scene_id       = scene_id,
        source_dataset = dataset,
        source_ply     = str(pred_ply.resolve()),
        ai_model       = ai_model,
        buildings      = buildings_out,
        run_params     = run_params,
    )
    save_buildings_json(scene_doc, out_dir / "buildings.json")

    # ── Run summary ───────────────────────────────────────────────────────
    summary = {
        "pipeline_version":         "v2.1",
        "pred_ply":                 str(pred_ply),
        "scene_id":                 scene_id,
        "dataset":                  dataset,
        "ai_model":                 ai_model,
        "total_input_points":       record.n_points,
        "building_class_points":    n_building_pts,
        "n_instances_extracted":    len(instances),
        "instance_method":          instance_method,
        "instance_noise_pct":       inst_stats["noise_pct"],
        "n_buildings_reconstructed": len(buildings_out),
        "n_flat_roof":              n_flat,
        "n_gable_roof":             n_gable,
        "n_skipped":                n_skip,
        "outputs": {
            "buildings_json": str(out_dir / "buildings.json"),
            "scene_obj":      str(scene_obj),
            "buildings_dir":  str(bldg_dir),
        },
        "floor_height_m_assumed": floor_height,
        "floor_estimation_note":  (
            "All floor counts are estimates from total height ÷ assumed floor height. "
            "Not directly observed from LiDAR."
        ),
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Buildings reconstructed : {len(buildings_out)}")
    print(f"  Flat roof               : {n_flat}")
    print(f"  Gable roof              : {n_gable}")
    print(f"  Skipped                 : {n_skip}")
    print(f"  buildings.json          : {out_dir / 'buildings.json'}")
    print(f"  scene.obj               : {scene_obj}")
    print(f"{'='*60}\n")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="SIH 26011 Reconstruction Pipeline v2 — dataset-agnostic.")
    ap.add_argument("--pred",       required=True,
                    help="Prediction PLY (x, y, z, predicted_class fields).")
    ap.add_argument("--out",        required=True,
                    help="Output directory.")
    ap.add_argument("--dataset",    default="unknown",
                    help="Dataset name: STPLS3D, DALES, or unknown (default).")
    ap.add_argument("--scene",      default=None,
                    help="Scene identifier (default: PLY file stem).")
    ap.add_argument("--model",      default="PointNet++",
                    help="Model description for provenance.")
    ap.add_argument("--cell-size",  type=float, default=None,
                    help="BFS grid cell size in metres (default: dataset-adaptive).")
    ap.add_argument("--min-points", type=int,   default=None,
                    help="Minimum points per building (default: dataset-adaptive).")
    ap.add_argument("--max-buildings", type=int, default=300)
    ap.add_argument("--floor-height",  type=float, default=DEFAULT_FLOOR_HEIGHT_M,
                    help=f"Assumed floor height m (default {DEFAULT_FLOOR_HEIGHT_M}).")
    ap.add_argument("--building-class", type=int, default=BUILDING_CLASS,
                    help="Integer class for buildings (default 1).")
    args = ap.parse_args()

    result = run_pipeline(
        pred_ply       = Path(args.pred),
        out_dir        = Path(args.out),
        dataset        = args.dataset,
        scene_id       = args.scene,
        ai_model       = args.model,
        cell_size      = args.cell_size,
        min_points     = args.min_points,
        max_buildings  = args.max_buildings,
        floor_height   = args.floor_height,
        building_class = args.building_class,
    )

    if "error" in result:
        print(f"\nPipeline error: {result['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

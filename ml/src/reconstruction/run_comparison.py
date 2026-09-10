"""Instance extraction and footprint method comparison — SIH 26011.

Runs systematic comparisons on REAL DALES and STPLS3D predictions to choose
the best methods for each dataset. Results inform pipeline_v3 defaults.

Usage:
    python ml/src/reconstruction/run_comparison.py \\
        --out ml/results/reconstruction_v3_future

Reads:
    ml/results/dales2_500k_prediction.ply
    ml/results/stpls3d_real/RealWorldData__RA_points/prediction_world.ply

Writes:
    <out>/instance_comparison_report.json
    <out>/footprint_comparison_report.json

Does NOT modify:
    viewer/  ml/src/training/  ml/src/models/  ml/src/inference/  ml/src/data/
    model checkpoints  Session 1 files  Session 3 files
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

from ml.src.reconstruction.prediction_contract import load_prediction_ply, BUILDING_CLASS
from ml.src.reconstruction.instance_extraction import (
    extract_building_instances,
    extract_building_instances_dbscan,
    evaluate_instance_extraction,
    compare_extraction_methods,
    RECOMMENDED_CELL_SIZE, RECOMMENDED_MIN_POINTS,
    RECOMMENDED_DBSCAN_EPS, RECOMMENDED_DBSCAN_MIN_PTS,
    HAS_OPEN3D,
)
from ml.src.reconstruction.footprint import (
    extract_footprint_convex, extract_footprint_raster,
    extract_footprint_best, _polygon_area, footprint_quality,
)

# ── Dataset prediction paths ───────────────────────────────────────────────────

DATASETS = {
    "DALES": {
        "ply": "ml/results/dales2_500k_prediction.ply",
        "scene_id": "dales2_500k",
    },
    "STPLS3D": {
        "ply": "ml/results/stpls3d_real/RealWorldData__RA_points/prediction_world.ply",
        "scene_id": "RealWorldData__RA_points",
    },
}


# ── Instance extraction comparison ────────────────────────────────────────────

def compare_instances_on_dataset(
    dataset_name: str,
    ply_path: Path,
    scene_id: str,
) -> dict:
    """Run multiple instance extraction methods on real prediction data.

    Compares:
    - Grid BFS (recommended cell sizes)
    - DBSCAN (default + tight + loose eps)
    - Adaptive (auto-select)

    Returns a dict of method → statistics.
    """
    print(f"\n  Loading {dataset_name} from {ply_path.name}...")
    rec = load_prediction_ply(ply_path, scene_id=scene_id, source_dataset=dataset_name)
    building_xyz = rec.building_xyz(BUILDING_CLASS)
    n_bldg = len(building_xyz)
    print(f"  {rec.n_points:,} total points, {n_bldg:,} building-class points")

    if n_bldg == 0:
        return {"error": "no_building_points"}

    results: dict = {
        "dataset":         dataset_name,
        "n_total_pts":     rec.n_points,
        "n_building_pts":  n_bldg,
        "open3d_available": HAS_OPEN3D,
        "methods":         {},
    }

    # ── Grid BFS variants ──────────────────────────────────────────────────────
    bfs_configs = [
        ("grid_bfs_default",
         RECOMMENDED_CELL_SIZE.get(dataset_name, 1.5),
         RECOMMENDED_MIN_POINTS.get(dataset_name, 30)),
        ("grid_bfs_fine",
         max(0.5, RECOMMENDED_CELL_SIZE.get(dataset_name, 1.5) * 0.5),
         RECOMMENDED_MIN_POINTS.get(dataset_name, 30)),
        ("grid_bfs_coarse",
         RECOMMENDED_CELL_SIZE.get(dataset_name, 1.5) * 2.0,
         RECOMMENDED_MIN_POINTS.get(dataset_name, 30)),
    ]

    for name, cell_size, min_pts in bfs_configs:
        t0 = time.perf_counter()
        insts = extract_building_instances(
            building_xyz, scene_id, dataset_name,
            cell_size=cell_size, min_points=min_pts,
        )
        elapsed = time.perf_counter() - t0
        stats = evaluate_instance_extraction(insts, building_xyz, name)
        stats["cell_size_m"] = cell_size
        stats["min_points"] = min_pts
        stats["time_s"] = round(elapsed, 3)
        results["methods"][name] = stats
        print(f"    {name:25s}: {stats['n_instances']:4d} instances  "
              f"noise={stats['noise_pct']:5.1f}%  "
              f"min/max={stats['min_pts']}/{stats['max_pts']}  "
              f"t={elapsed:.2f}s")

    # ── DBSCAN variants ────────────────────────────────────────────────────────
    if HAS_OPEN3D:
        default_eps = RECOMMENDED_DBSCAN_EPS.get(dataset_name, 2.0)
        default_min = RECOMMENDED_DBSCAN_MIN_PTS.get(dataset_name, 15)

        dbscan_configs = [
            ("dbscan_default", default_eps, default_min),
            ("dbscan_tight",   default_eps * 0.6, default_min),
            ("dbscan_loose",   default_eps * 1.5, default_min),
            ("dbscan_lowmin",  default_eps, max(5, default_min // 2)),
        ]

        for name, eps, min_pts in dbscan_configs:
            t0 = time.perf_counter()
            insts = extract_building_instances_dbscan(
                building_xyz, scene_id, dataset_name,
                eps=eps, min_points=min_pts,
            )
            elapsed = time.perf_counter() - t0
            stats = evaluate_instance_extraction(insts, building_xyz, name)
            stats["eps_m"] = eps
            stats["min_points"] = min_pts
            stats["time_s"] = round(elapsed, 3)
            results["methods"][name] = stats
            print(f"    {name:25s}: {stats['n_instances']:4d} instances  "
                  f"noise={stats['noise_pct']:5.1f}%  "
                  f"min/max={stats['min_pts']}/{stats['max_pts']}  "
                  f"t={elapsed:.2f}s")
    else:
        results["methods"]["dbscan"] = {"available": False, "reason": "Open3D not installed"}
        print("    DBSCAN: Open3D not available")

    # ── Winner selection ───────────────────────────────────────────────────────
    # Prefer DBSCAN with noise_pct closest to 30-50% (typical building coverage)
    # and reasonable instance count (not too many tiny fragments)
    best = _select_best_instance_method(results["methods"], dataset_name)
    results["recommended_method"] = best
    results["selection_rationale"] = _instance_selection_rationale(
        results["methods"], best, dataset_name)

    return results


def _select_best_instance_method(methods: dict, dataset_name: str) -> str:
    """Heuristic: pick method with lowest noise_pct AND reasonable instance count."""
    valid = {k: v for k, v in methods.items()
             if isinstance(v, dict) and "n_instances" in v and v.get("n_instances", 0) > 0}
    if not valid:
        return "grid_bfs_default"

    # Score: low noise is good, but too many tiny instances are bad
    def score(stats: dict) -> float:
        n = stats.get("n_instances", 0)
        noise = stats.get("noise_pct", 100.0)
        size_ratio = stats.get("size_ratio", 99.0)
        # Penalty for very high noise or fragmentation
        return -noise - min(size_ratio, 50) * 0.5

    best = max(valid, key=lambda k: score(valid[k]))
    return best


def _instance_selection_rationale(methods: dict, best: str, dataset_name: str) -> str:
    m = methods.get(best, {})
    return (
        f"Method '{best}' selected for {dataset_name}. "
        f"Noise={m.get('noise_pct','?')}%, "
        f"instances={m.get('n_instances','?')}, "
        f"min/max pts={m.get('min_pts','?')}/{m.get('max_pts','?')}."
    )


# ── Footprint comparison ───────────────────────────────────────────────────────

def compare_footprints_on_dataset(
    dataset_name: str,
    ply_path: Path,
    scene_id: str,
    n_sample: int = 20,
) -> dict:
    """Compare convex hull vs raster boundary on sample buildings.

    Uses the default instance extraction to get buildings, then compares
    footprint methods on each.

    n_sample: number of buildings to sample (take the largest n_sample).
    """
    print(f"\n  Footprint comparison for {dataset_name}...")
    rec = load_prediction_ply(ply_path, scene_id=scene_id, source_dataset=dataset_name)
    building_xyz = rec.building_xyz(BUILDING_CLASS)

    if len(building_xyz) == 0:
        return {"error": "no_building_points"}

    # Extract instances (use DBSCAN if available, else grid BFS)
    if HAS_OPEN3D:
        insts = extract_building_instances_dbscan(
            building_xyz, scene_id, dataset_name)
    else:
        insts = extract_building_instances(
            building_xyz, scene_id, dataset_name,
            cell_size=RECOMMENDED_CELL_SIZE.get(dataset_name, 1.5),
            min_points=RECOMMENDED_MIN_POINTS.get(dataset_name, 30),
        )

    if not insts:
        return {"error": "no_instances"}

    # Sample the top-n by point count
    sample = sorted(insts, key=lambda i: -i.n_points)[:n_sample]
    cell_size = RECOMMENDED_CELL_SIZE.get(dataset_name, 1.5)

    convex_quality = []
    raster_quality = []
    best_quality = []
    n_concave = 0  # raster found more vertices than convex → concave shape

    for inst in sample:
        pts = building_xyz[inst.point_indices]

        # Convex hull
        c_poly, c_meta = extract_footprint_convex(pts)
        # Raster boundary
        r_poly, r_meta = extract_footprint_raster(pts, cell_size=cell_size)
        # Auto-best
        b_poly, b_meta = extract_footprint_best(pts, cell_size=cell_size)

        if c_poly and len(c_poly) >= 3:
            q = footprint_quality(c_poly, pts)
            q["area_m2"] = c_meta.get("area_m2", 0)
            q["n_verts"] = len(c_poly)
            convex_quality.append(q)

        if r_poly and len(r_poly) >= 3:
            q = footprint_quality(r_poly, pts)
            q["area_m2"] = r_meta.get("area_m2", 0)
            q["n_verts"] = len(r_poly)
            raster_quality.append(q)
            if c_poly and len(r_poly) > len(c_poly) * 1.5:
                n_concave += 1  # raster detected extra shape detail vs convex

        if b_poly and len(b_poly) >= 3:
            q = footprint_quality(b_poly, pts)
            q["area_m2"] = b_meta.get("area_m2", 0)
            q["n_verts"] = len(b_poly)
            q["method_used"] = b_meta.get("method", "unknown")
            best_quality.append(q)

    def agg(rows: list[dict], keys: list[str]) -> dict:
        if not rows:
            return {}
        return {k: round(float(np.mean([r[k] for r in rows if k in r])), 4)
                for k in keys}

    keys = ["coverage", "compactness", "n_verts", "area_m2"]
    result = {
        "dataset":          dataset_name,
        "n_buildings_sampled": len(sample),
        "n_with_concave_detail": n_concave,
        "cell_size_m": cell_size,
        "convex_hull": {
            "n_success": len(convex_quality),
            "mean": agg(convex_quality, keys),
            "note": "Over-approximates concave shapes (L, T, courtyard)",
        },
        "raster_boundary": {
            "n_success": len(raster_quality),
            "mean": agg(raster_quality, keys),
            "note": "Preserves concave shapes via grid edge tracing",
        },
        "auto_best": {
            "n_success": len(best_quality),
            "mean": agg(best_quality, keys),
            "methods_used": list({r.get("method_used", "?") for r in best_quality}),
        },
        "recommendation": (
            "raster_boundary"
            if len(raster_quality) >= len(convex_quality) * 0.9
            else "convex_hull"
        ),
        "recommendation_rationale": (
            f"Raster detected concave shape detail in {n_concave}/{len(sample)} buildings "
            f"({100*n_concave//max(1,len(sample))}%). Raster preferred when it succeeds."
        ),
    }

    print(f"    Convex:  coverage={result['convex_hull']['mean'].get('coverage','?'):.3f}  "
          f"verts={result['convex_hull']['mean'].get('n_verts','?'):.1f}")
    print(f"    Raster:  coverage={result['raster_boundary']['mean'].get('coverage','?'):.3f}  "
          f"verts={result['raster_boundary']['mean'].get('n_verts','?'):.1f}")
    print(f"    Concave shapes detected: {n_concave}/{len(sample)}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Instance + footprint comparison on real predictions")
    ap.add_argument("--out", default="ml/results/reconstruction_v3_future",
                    help="Output directory for comparison reports")
    ap.add_argument("--n-sample", type=int, default=20,
                    help="Number of buildings to sample for footprint comparison")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_report: dict = {
        "title": "Instance Extraction Method Comparison",
        "note": ("Comparison on real DALES (IoU=0.763) and STPLS3D (IoU=0.659) predictions. "
                 "This is NOT trained instance segmentation — these are spatial clustering heuristics."),
        "datasets": {},
    }

    footprint_report: dict = {
        "title": "Footprint Method Comparison",
        "note": ("Comparison of convex hull vs raster boundary on real predictions. "
                 "Raster boundary preserves L/T/concave shapes."),
        "datasets": {},
    }

    for ds_name, ds_cfg in DATASETS.items():
        ply_path = ROOT / ds_cfg["ply"]
        if not ply_path.exists():
            print(f"  SKIP {ds_name}: {ply_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {ds_name} Instance Comparison")
        print(f"{'='*60}")
        inst_result = compare_instances_on_dataset(
            ds_name, ply_path, ds_cfg["scene_id"]
        )
        instance_report["datasets"][ds_name] = inst_result

        print(f"\n{'='*60}")
        print(f"  {ds_name} Footprint Comparison")
        print(f"{'='*60}")
        fp_result = compare_footprints_on_dataset(
            ds_name, ply_path, ds_cfg["scene_id"],
            n_sample=args.n_sample,
        )
        footprint_report["datasets"][ds_name] = fp_result

    # Save reports
    inst_path = out_dir / "instance_comparison_report.json"
    fp_path   = out_dir / "footprint_comparison_report.json"

    with open(inst_path, "w", encoding="utf-8") as f:
        json.dump(instance_report, f, indent=2)
    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(footprint_report, f, indent=2)

    print(f"\n\nReports saved:")
    print(f"  {inst_path}")
    print(f"  {fp_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

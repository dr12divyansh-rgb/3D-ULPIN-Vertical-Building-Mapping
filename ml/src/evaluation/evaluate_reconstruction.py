"""Step 4: evaluate predicted LOD2 against ground-truth LOD2.

Usage:
    python ml/src/evaluation/evaluate_reconstruction.py \\
        --predicted ml/results/experiment_001/reconstruction/scene_00005/predicted_lod2 \\
        --gt        ml/data/synthetic/test/scene_00005

The script:
  * Loads each predicted OBJ from ``--predicted``.
  * Loads the matching GT LOD2 OBJ from ``--gt/ground_truth/lod2/``.
  * Runs the existing ``evaluate_lod2`` function per building.
  * Also runs the GT-vs-self sanity check (must be ~0 error).
  * Prints per-building results and an aggregate summary.
  * Writes results to ``--predicted/../evaluation.json`` (alongside the OBJs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from synthetic_city.core.evaluation import evaluate_lod2, is_sane  # noqa: E402
from synthetic_city.core.mesh import SurfaceMesh  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_mesh(path: Path) -> SurfaceMesh:
    return SurfaceMesh.from_obj(path)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


# --------------------------------------------------------------------------- #
# Per-building evaluation                                                      #
# --------------------------------------------------------------------------- #

def evaluate_building(pred_obj: Path, gt_obj: Path) -> dict:
    pred = _load_mesh(pred_obj)
    gt   = _load_mesh(gt_obj)
    result = evaluate_lod2(pred, gt, n_samples=2000, seed=0)
    result["_pred_obj"] = str(pred_obj)
    result["_gt_obj"]   = str(gt_obj)
    return result


def evaluate_gt_self(gt_obj: Path) -> dict:
    """GT vs itself: should yield (near) zero error on all geometry metrics."""
    gt = _load_mesh(gt_obj)
    result = evaluate_lod2(gt, gt, n_samples=2000, seed=0)
    result["passes_sanity"] = is_sane(result)
    return result


# --------------------------------------------------------------------------- #
# Scene evaluation                                                             #
# --------------------------------------------------------------------------- #

def evaluate_scene(predicted_dir: Path, gt_scene_dir: Path) -> dict:
    gt_lod2_dir = gt_scene_dir / "ground_truth" / "lod2"
    gt_meta_path = gt_lod2_dir / "metadata.json"
    if not gt_meta_path.exists():
        raise FileNotFoundError(f"GT LOD2 metadata not found: {gt_meta_path}")

    with open(gt_meta_path, encoding="utf-8") as f:
        gt_meta = json.load(f)

    pred_objs = sorted(predicted_dir.glob("building_*.obj"))
    if not pred_objs:
        raise FileNotFoundError(f"No predicted OBJs found in {predicted_dir}")

    # Build a map: building_id (from filename) → GT OBJ path
    gt_obj_map: dict[int, Path] = {}
    for b in gt_meta["buildings"]:
        bid = int(b["building_id"])
        gt_obj_path = gt_lod2_dir / b["obj"]
        if gt_obj_path.exists():
            gt_obj_map[bid] = gt_obj_path

    per_building: list[dict] = []
    height_errors: list[float] = []
    area_errors:   list[float] = []
    area_rel_errs: list[float] = []
    fp_ious:       list[float] = []
    chamfers:      list[float] = []
    watertight_pred: list[bool] = []
    nm_edges:      list[int] = []
    zero_faces:    list[int] = []

    sanity_results: list[dict] = []

    print("\n" + "=" * 90)
    print(f"{'BID':>7} {'h_pred':>8} {'h_gt':>8} {'h_err':>7} {'fp_iou':>7} {'chamfer':>9} "
          f"{'area_rel':>9} {'wt':>5} {'nm':>4}")
    print("-" * 90)

    for pred_obj in pred_objs:
        # Extract building id from filename (building_XXXXXX.obj)
        stem = pred_obj.stem  # "building_040001"
        try:
            bid = int(stem.split("_")[-1])
        except ValueError:
            print(f"  [warn] cannot parse building id from {pred_obj.name}", file=sys.stderr)
            continue

        if bid not in gt_obj_map:
            print(f"  [warn] no GT OBJ for building {bid}", file=sys.stderr)
            continue

        gt_obj = gt_obj_map[bid]

        # Per-building evaluation
        res = evaluate_building(pred_obj, gt_obj)
        res["building_id"] = bid
        per_building.append(res)

        g = res["geometry"]
        t = res["topology"]

        height_errors.append(g["height_error"])
        area_errors.append(g["footprint_area_error"])
        area_rel_errs.append(g["footprint_area_error_relative"])
        fp_ious.append(g["footprint_iou"])
        chamfers.append(g["chamfer_distance"])
        watertight_pred.append(t["watertight_pred"])
        nm_edges.append(t["non_manifold_edges_pred"])
        zero_faces.append(t["zero_area_faces_pred"])

        print(
            f"{bid:>7d} {g['height_pred']:>8.2f} {g['height_gt']:>8.2f} "
            f"{g['height_error']:>7.3f} {g['footprint_iou']:>7.4f} "
            f"{g['chamfer_distance']:>9.4f} {g['footprint_area_error_relative']:>9.4f} "
            f"{'Y' if t['watertight_pred'] else 'N':>5} {t['non_manifold_edges_pred']:>4d}"
        )

        # GT-vs-self sanity check for this building
        sr = evaluate_gt_self(gt_obj)
        sr["building_id"] = bid
        sanity_results.append(sr)

    print("-" * 90)

    # Aggregate
    n = len(per_building)
    aggregate = {
        "n_buildings": n,
        "mean_height_error_m": round(_mean(height_errors), 4),
        "mean_footprint_area_error_m2": round(_mean(area_errors), 4),
        "mean_footprint_area_error_relative": round(_mean(area_rel_errs), 4),
        "mean_footprint_iou": round(_mean(fp_ious), 4),
        "mean_chamfer_distance_m": round(_mean(chamfers), 4),
        "n_watertight": sum(watertight_pred),
        "n_non_watertight": n - sum(watertight_pred),
        "total_non_manifold_edges": sum(nm_edges),
        "total_zero_area_faces": sum(zero_faces),
    }

    print(f"\nAggregate over {n} buildings:")
    for k, v in aggregate.items():
        print(f"  {k:45s}: {_fmt(v)}")

    # Sanity check summary
    all_sanity_pass = all(s["passes_sanity"] for s in sanity_results)
    print(f"\nGT-vs-self sanity check (all buildings): {'PASS' if all_sanity_pass else 'FAIL'}")
    if not all_sanity_pass:
        for s in sanity_results:
            if not s["passes_sanity"]:
                print(f"  building {s['building_id']} FAILED sanity: "
                      f"h_err={s['geometry']['height_error']:.6g} "
                      f"area_err={s['geometry']['footprint_area_error']:.6g} "
                      f"chamfer={s['geometry']['chamfer_distance']:.6g}")

    result = {
        "scene": gt_scene_dir.name,
        "predicted_dir": str(predicted_dir),
        "aggregate": aggregate,
        "per_building": per_building,
        "gt_self_sanity": {
            "all_pass": all_sanity_pass,
            "results": sanity_results,
        },
    }

    out_path = predicted_dir.parent / "evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[evaluate] results written → {out_path}")
    return result


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Step 4: evaluate predicted LOD2 vs GT LOD2.")
    ap.add_argument("--predicted", required=True,
                    help="Directory containing predicted building_XXXXXX.obj files.")
    ap.add_argument("--gt", required=True,
                    help="Scene directory containing ground_truth/lod2/ (e.g. ml/data/synthetic/test/scene_00005).")
    args = ap.parse_args()

    predicted_dir = Path(args.predicted)
    gt_scene_dir  = Path(args.gt)

    if not predicted_dir.is_dir():
        print(f"error: predicted directory not found: {predicted_dir}", file=sys.stderr)
        return 1
    if not gt_scene_dir.is_dir():
        print(f"error: GT scene directory not found: {gt_scene_dir}", file=sys.stderr)
        return 1

    evaluate_scene(predicted_dir, gt_scene_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

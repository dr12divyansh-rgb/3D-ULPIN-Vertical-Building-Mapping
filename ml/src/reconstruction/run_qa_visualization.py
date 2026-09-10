"""QA visualization — save per-building reconstruction examples.

Reads existing reconstruction_v3 outputs and saves detailed QA records
for the top buildings from each dataset. Used for inspection and debugging.

Output: ml/results/reconstruction_v3_future/qa/<dataset>/building_XX.json

Each JSON file records all geometry, provenance, and reconstruction details
for one building so a human reviewer can verify correctness.

Usage:
    python ml/src/reconstruction/run_qa_visualization.py \\
        --out ml/results/reconstruction_v3_future/qa \\
        --n 8

Does NOT modify: viewer/, ml/src/training/, checkpoints, Session 1/3 files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

# ── Source reconstruction_v3 outputs ──────────────────────────────────────────

V3_SOURCES = {
    "dales": {
        "buildings_json":  "ml/results/reconstruction_v3/dales/buildings.json",
        "dataset_iou":     "0.763",
        "ai_model":        "PointNet++ Exp006",
    },
    "stpls3d_ra": {
        "buildings_json":  "ml/results/reconstruction_v3/stpls3d_ra/buildings.json",
        "dataset_iou":     "0.659",
        "ai_model":        "PointNet++ Exp005",
    },
    "ahn4_zeroreg": {
        "buildings_json":  "ml/results/reconstruction_v3_future/ahn4_zeroreg/buildings.json",
        "dataset_iou":     "0.699",
        "ai_model":        "PointNet++ Exp006 (zero-RGB workaround on AHN4)",
    },
}


# ── QA record builder ─────────────────────────────────────────────────────────

def build_qa_record(b: dict, source_key: str, source_info: dict, rank: int) -> dict:
    """Build a QA record for one building.

    Checks:
    - footprint validity (min 3 vertices, non-zero area)
    - height plausibility (>= 1m, <= 150m)
    - floor count >= 1
    - roof type is a known value
    - CRS present
    - provenance present
    - OBJ file referenced
    """
    checks: dict[str, bool | str] = {}
    warnings: list[str] = []

    # Footprint
    fp = b.get("footprint", [])
    checks["footprint_has_3plus_vertices"] = len(fp) >= 3
    area = b.get("footprint_area_m2", 0)
    checks["footprint_area_positive"] = area > 0
    if len(fp) < 3:
        warnings.append(f"Footprint has only {len(fp)} vertices — invalid polygon")
    if area <= 0:
        warnings.append("Zero or negative footprint area")

    # Height
    h = b.get("height", 0)
    checks["height_gte_1m"]   = h >= 1.0
    checks["height_lte_150m"] = h <= 150.0
    if h < 1.0:
        warnings.append(f"Very low building height: {h:.1f}m")
    if h > 150.0:
        warnings.append(f"Suspiciously tall building: {h:.1f}m")

    # Height sources
    checks["ground_z_present"]   = b.get("ground_z") is not None
    checks["roof_z_above_ground"] = (b.get("roof_z", 0) > b.get("ground_z", 0))

    # Floors
    floor_count = b.get("floor_count", 0)
    checks["floor_count_gte_1"] = floor_count >= 1
    floors_list = b.get("floors", [])
    checks["floors_list_present"] = len(floors_list) > 0
    checks["floors_count_matches"] = len(floors_list) == floor_count
    if floor_count < 1:
        warnings.append("Floor count < 1")

    # Floor source
    all_estimated = all(fl.get("source") == "estimated" for fl in floors_list)
    checks["floors_labelled_estimated"] = all_estimated
    if not all_estimated:
        warnings.append("Some floors not labelled 'estimated'")

    # Roof
    roof_type = b.get("roof_type", "")
    known_roof = {"flat", "gable", "shed", "hip", "unknown"}
    checks["roof_type_known"] = roof_type in known_roof
    if roof_type not in known_roof:
        warnings.append(f"Unknown roof type: {roof_type!r}")

    # OBJ file
    lod2 = b.get("lod2_like_geometry", {})
    obj_ref = lod2.get("obj_file", "")
    checks["lod2_obj_referenced"] = bool(obj_ref)

    # CRS
    checks["crs_present"] = bool(b.get("crs"))

    # Prototype ULPIN note
    checks["prototype_ulpin_note_present"] = bool(b.get("prototype_ulpin_note"))

    # Provenance
    prov = b.get("provenance", {})
    required_prov = {"segmentation", "instance_separation", "footprint", "height", "floors"}
    missing_prov = required_prov - set(prov.keys())
    checks["provenance_complete"] = len(missing_prov) == 0
    if missing_prov:
        warnings.append(f"Missing provenance fields: {missing_prov}")

    # No fabricated confidence in floors
    checks["no_fabricated_floor_confidence"] = all(
        fl.get("confidence") is None for fl in floors_list
    )

    passed = [k for k, v in checks.items() if v is True]
    failed = [k for k, v in checks.items() if v is False]

    return {
        "rank":           rank,
        "building_id":    b.get("building_id", ""),
        "dataset":        source_key,
        "dataset_iou":    source_info["dataset_iou"],
        "ai_model":       source_info["ai_model"],
        "qa_checks": {
            "total":   len(checks),
            "passed":  len(passed),
            "failed":  len(failed),
            "status":  "PASS" if not failed else "FAIL",
            "passed_list": passed,
            "failed_list": failed,
            "warnings": warnings,
        },
        "geometry_summary": {
            "height_m":        round(b.get("height", 0), 2),
            "ground_z":        round(b.get("ground_z", 0), 3),
            "roof_z":          round(b.get("roof_z", 0), 3),
            "footprint_area_m2": round(area, 1),
            "footprint_n_verts": len(fp),
            "floor_count":     floor_count,
            "roof_type":       roof_type,
            "point_count":     b.get("point_count", 0),
        },
        "floor_detail": [
            {"floor_id":    fl.get("floor_id"),
             "floor_number": fl.get("floor_number"),
             "z_min":       fl.get("z_min"),
             "z_max":       fl.get("z_max"),
             "height":      fl.get("height"),
             "source":      fl.get("source"),
             "confidence":  fl.get("confidence")}
            for fl in floors_list
        ],
        "provenance": prov,
        "crs": b.get("crs", "not set"),
        "lod2_obj_file": obj_ref,
        "prototype_ulpin_note": b.get("prototype_ulpin_note", ""),
    }


def run_qa_for_source(
    source_key: str,
    source_info: dict,
    out_dir: Path,
    n_buildings: int = 8,
) -> dict:
    """Load buildings.json, pick top-n by point count, save QA records."""
    buildings_path = ROOT / source_info["buildings_json"]
    if not buildings_path.exists():
        print(f"  [{source_key}] SKIP — {buildings_path} not found")
        return {"error": "not_found"}

    with open(buildings_path, encoding="utf-8") as f:
        data = json.load(f)

    buildings = data.get("buildings", [])
    if not buildings:
        return {"error": "no_buildings"}

    # Sort by point count (more points = more data to verify)
    ranked = sorted(buildings, key=lambda b: -b.get("point_count", 0))[:n_buildings]

    src_dir = out_dir / source_key
    src_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    n_pass = 0

    for rank, b in enumerate(ranked, start=1):
        qa_rec = build_qa_record(b, source_key, source_info, rank)
        bid = b.get("building_id", f"building_{rank:04d}").replace("/", "_")
        out_path = src_dir / f"{bid}_qa.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(qa_rec, f, indent=2)

        status = qa_rec["qa_checks"]["status"]
        if status == "PASS":
            n_pass += 1
        warnings = qa_rec["qa_checks"]["warnings"]
        print(f"  [{source_key}] rank={rank}  {bid}  "
              f"h={qa_rec['geometry_summary']['height_m']:.1f}m  "
              f"fl={qa_rec['geometry_summary']['floor_count']}  "
              f"pts={qa_rec['geometry_summary']['point_count']}  "
              f"QA={status}" + (f"  WARN:{warnings[0]}" if warnings else ""))

        summary_rows.append({
            "rank": rank,
            "building_id": b.get("building_id"),
            "status": status,
            "passed": qa_rec["qa_checks"]["passed"],
            "failed": qa_rec["qa_checks"]["failed"],
            "height_m": qa_rec["geometry_summary"]["height_m"],
            "floor_count": qa_rec["geometry_summary"]["floor_count"],
            "roof_type": qa_rec["geometry_summary"]["roof_type"],
            "point_count": qa_rec["geometry_summary"]["point_count"],
            "warnings": ", ".join(qa_rec["qa_checks"]["warnings"]),
        })

    summary = {
        "source_key":    source_key,
        "buildings_json": str(buildings_path),
        "n_sampled":     len(ranked),
        "n_pass":        n_pass,
        "n_fail":        len(ranked) - n_pass,
        "pass_rate":     f"{100*n_pass//max(1,len(ranked))}%",
        "buildings":     summary_rows,
    }

    with open(src_dir / "qa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="QA visualization for reconstruction_v3")
    ap.add_argument("--out", default="ml/results/reconstruction_v3_future/qa")
    ap.add_argument("--n", type=int, default=8, help="Buildings to sample per dataset")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: dict = {}

    for source_key, source_info in V3_SOURCES.items():
        print(f"\n{'='*60}")
        print(f"  QA: {source_key}")
        print(f"{'='*60}")
        summary = run_qa_for_source(source_key, source_info, out_dir, args.n)
        all_summaries[source_key] = summary

    # Save overall summary
    overall_path = out_dir / "overall_qa.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump({
            "title":    "Reconstruction v3 Geometry QA",
            "n_per_ds": args.n,
            "sources":  all_summaries,
        }, f, indent=2)

    print(f"\nQA complete. Outputs in {out_dir}")
    print(f"Overall QA report: {overall_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

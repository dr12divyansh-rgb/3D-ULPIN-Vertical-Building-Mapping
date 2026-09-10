"""Export reconstruction output to the viewer contract format.

Converts pipeline_v3 buildings.json -> viewer_buildings.json
matching Session 3's INTEGRATION_NOTES.md contract exactly.

Usage:
    python ml/src/reconstruction/export_viewer_json.py \\
        --input  ml/results/reconstruction_v3/ahn4/buildings.json \\
        --output ml/results/reconstruction_v3/ahn4/viewer_buildings.json \\
        --dataset AHN4 \\
        --model_id exp006 \\
        --building_iou 0.053
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(
    input_path: str | Path,
    output_path: str | Path,
    dataset: str,
    model_id: str,
    building_iou: float,
    crs: str = "unknown",
) -> int:
    """Convert a pipeline_v3 buildings.json to the Session 3 viewer contract.

    Session 3's INTEGRATION_NOTES.md specifies these per-building fields:
        id, source_dataset, scene_id,
        geometry.type, geometry.footprint, geometry.height_m, geometry.height_source,
        ai.model, ai.status, ai.class_label, ai.class_id,
        floors.count, floors.source, floors.levels[],
        roof.type, roof.source,
        footprint_area_m2, centroid_lon, centroid_lat (or centroid_x, centroid_y)

    Returns:
        Number of buildings exported.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    buildings_in = data.get("buildings", [])
    crs_str = data.get("run_parameters", {}).get("crs", crs)

    buildings_out = []
    for b in buildings_in:
        fp = b.get("footprint", [])
        cx = sum(p[0] for p in fp) / len(fp) if fp else 0.0
        cy = sum(p[1] for p in fp) / len(fp) if fp else 0.0

        # Floor levels from floor records
        floor_levels = None
        floors_list = b.get("floors", [])
        if floors_list:
            floor_levels = [f["z_min"] for f in floors_list] + [floors_list[-1]["z_max"]]

        floor_count = b.get("floor_count", 1)
        floor_src   = b.get("floor_count_source", "estimated-height-based")
        height      = b.get("height", 0.0)
        height_src  = b.get("height_source", "LiDAR-derived")
        roof_type   = b.get("roof_type", "unknown")
        roof_src    = b.get("roof_type_source", "LiDAR-derived-heuristic")

        # Map roof source to viewer-friendly string
        if "RANSAC" in str(roof_src):
            roof_src_label = "LiDAR-RANSAC-plane-fitting"
        else:
            roof_src_label = "LiDAR-Zstd-heuristic"

        out_bldg = {
            "id":             b.get("building_id", ""),
            "source_dataset": b.get("source_dataset", dataset),
            "scene_id":       b.get("scene_id", ""),
            "instance_id":    b.get("instance_id", 0),

            "geometry": {
                "type":           "lod1_extrusion",
                "footprint":      fp,
                "height_m":       round(height, 3),
                "height_source":  height_src,
                "lod1_obj_file":  b.get("lod2_like_geometry", {}).get("obj_file"),
            },

            "ai": {
                "model":       b.get("ai_model", ""),
                "status":      "predicted",
                "class_label": "building",
                "class_id":    1,
            },

            "floors": {
                "count":  floor_count,
                "source": floor_src,
                "levels": floor_levels,
            },

            "roof": {
                "type":       roof_type,
                "source":     roof_src_label,
                "slope_deg":  b.get("roof_slope_deg"),
                "n_planes":   b.get("roof_n_planes"),
            },

            "footprint_area_m2":    round(b.get("footprint_area_m2", 0.0), 1),
            "point_count":          b.get("point_count", 0),
            "ground_z":             b.get("ground_z"),
            "roof_z":               b.get("roof_z"),

            # CRS-dependent centroid — use x/y since many are local coordinates
            "centroid_x": round(cx, 3),
            "centroid_y": round(cy, 3),
            "crs":        crs_str,

            "provenance":           b.get("provenance", {}),
            "confidence_mean":      b.get("confidence_mean"),
            "density_pts_per_m2":   b.get("density_pts_per_m2"),

            # Prototype ULPIN note
            "prototype_ulpin_note": (
                "Prototype identifier only. Not an official ULPIN. "
                "Official ULPIN requires integration with government cadastral registry."
            ),
        }
        buildings_out.append(out_bldg)

    viewer_doc = {
        "dataset":       dataset,
        "model_id":      model_id,
        "building_iou":  building_iou,
        "crs":           crs_str,
        "n_buildings":   len(buildings_out),
        "schema_version": "viewer_v1",
        "buildings":     buildings_out,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(viewer_doc, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(buildings_out)} buildings -> {out}")
    return len(buildings_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",        required=True)
    ap.add_argument("--output",       required=True)
    ap.add_argument("--dataset",      required=True)
    ap.add_argument("--model_id",     required=True)
    ap.add_argument("--building_iou", type=float, required=True)
    ap.add_argument("--crs",          default="unknown")
    args = ap.parse_args()

    convert(
        input_path   = args.input,
        output_path  = args.output,
        dataset      = args.dataset,
        model_id     = args.model_id,
        building_iou = args.building_iou,
        crs          = args.crs,
    )


if __name__ == "__main__":
    main()

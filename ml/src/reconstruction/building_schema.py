"""Unified building output schema for SIH 26011 reconstruction v2.

Every field carries explicit provenance. Possible provenance values:
  "AI-derived"              -- from PointNet++ semantic segmentation
  "LiDAR-derived"           -- computed from the 3D point geometry
  "LiDAR-derived-heuristic" -- geometric heuristic on LiDAR data
  "estimated"               -- inferred, not directly observed
  "prototype"               -- not from official registry
  "unavailable"             -- field not computable from available data

DO NOT:
  - claim official ULPIN values
  - invent confidence values
  - use placeholder geometry
  - call simple extrusion "LOD2"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Per-floor record ───────────────────────────────────────────────────────────

def floor_record_schema() -> dict[str, Any]:
    """Return an example empty floor record for documentation."""
    return {
        "floor_id":     "DATASET_scene_0001/F01",
        "building_id":  "DATASET_scene_0001",
        "floor_number": 1,
        "z_min":        0.0,
        "z_max":        3.2,
        "height":       3.2,
        "source":       "estimated",
        "confidence":   None,
    }


# ── Per-building record ────────────────────────────────────────────────────────

def building_record(
    building_id: str,
    source_dataset: str,
    scene_id: str,
    instance_id: int,
    footprint: list[list[float]],
    footprint_area_m2: float,
    footprint_n_vertices: int,
    point_count: int,
    ground_z: float,
    roof_z: float,
    height: float,
    eave_z: float,
    floor_count: int,
    floor_height_m_assumed: float,
    floors: list[dict],
    roof_type: str,
    is_pitched: bool,
    lod2_obj_file: str,
    ai_model: str,
    confidence_note: str = (
        "No per-building confidence is claimed. "
        "Per-point confidence exists in source PLY where the model output contains it."
    ),
) -> dict[str, Any]:
    """
    Build the canonical per-building output record.

    All elevations are in the same CRS as the source LiDAR.
    Footprint vertices are (x, y) pairs in local LiDAR coordinates.

    LOD1 is an extruded footprint to a constant roof_z.
    LOD2-like adds detected roof shape (flat slab or gable); it is not
    certified against any official LOD2 standard.
    """
    lod2_description = (
        f"Gable-roof extrusion (footprint walls to eave_z={eave_z:.2f}m, "
        f"then pitched roof to roof_z={roof_z:.2f}m)"
        if is_pitched else
        f"Flat-roof extrusion (footprint extruded to roof_z={roof_z:.2f}m)"
    )

    return {
        # ── Identity ──────────────────────────────────────────────────────
        "building_id":        building_id,
        "prototype_ulpin":    building_id,
        "prototype_ulpin_note": (
            "Prototype identifier only. Not an official ULPIN. "
            "Official ULPIN requires government cadastral registry integration."
        ),

        # ── Source provenance ─────────────────────────────────────────────
        "source_dataset": source_dataset,
        "scene_id":       scene_id,
        "instance_id":    instance_id,

        # ── Footprint ─────────────────────────────────────────────────────
        "footprint":          footprint,
        "footprint_area_m2":  round(footprint_area_m2, 1),
        "footprint_n_vertices": footprint_n_vertices,
        "footprint_source":   "LiDAR-derived-convex-hull",
        "footprint_note":     (
            "Convex hull of near-ground building points. "
            "Concave shapes (L, T, courtyard) are over-approximated."
        ),

        # ── Point data ────────────────────────────────────────────────────
        "point_count": point_count,

        # ── Elevations (all LiDAR-derived) ────────────────────────────────
        "ground_z":        round(ground_z, 3),
        "ground_z_source": "LiDAR-derived-5th-percentile",
        "roof_z":          round(roof_z, 3),
        "roof_z_source":   "LiDAR-derived-99th-percentile",
        "eave_z":          round(eave_z, 3),
        "eave_z_source":   "LiDAR-derived-80th-percentile-clamped",
        "height":          round(height, 3),
        "height_source":   "LiDAR-derived",

        # ── Floors (estimated, NOT directly observed) ─────────────────────
        "floor_count":             floor_count,
        "floor_count_source":      "estimated-height-based",
        "floor_height_m_assumed":  floor_height_m_assumed,
        "floor_estimation_note": (
            "Floor count = round(height / floor_height_m). "
            "Aerial LiDAR observes roof and ground, not individual floor slabs. "
            "These floor boundaries are estimates."
        ),
        "floors": floors,

        # ── Roof ──────────────────────────────────────────────────────────
        "roof_type":        roof_type,
        "roof_is_pitched":  is_pitched,
        "roof_type_source": "LiDAR-derived-heuristic",
        "roof_type_note": (
            "Flat vs gable classification: Z-std > 0.5m in roof band "
            "and roof height > 0.5m → pitched. Otherwise flat."
        ),

        # ── LOD1 geometry (footprint + constant height, no roof shape) ────
        "lod1_geometry": {
            "type":        "extruded_footprint",
            "z_bottom":    round(ground_z, 3),
            "z_top":       round(roof_z, 3),
            "description": (
                "Constant-height extrusion of footprint polygon to roof_z. "
                "No roof shape information."
            ),
        },

        # ── LOD2-like geometry (footprint + detected roof) ─────────────────
        "lod2_like_geometry": {
            "type":        "gable_roof_extrusion" if is_pitched else "flat_roof_extrusion",
            "obj_file":    lod2_obj_file,
            "description": lod2_description,
            "lod2_note": (
                "This is a LOD2-like mesh produced from LiDAR geometry. "
                "It is NOT certified against CityGML LOD2 specifications."
            ),
        },

        # ── Model and provenance ──────────────────────────────────────────
        "ai_model": ai_model,
        "provenance": {
            "segmentation":        "AI-derived",
            "instance_separation": "LiDAR-derived-grid-BFS",
            "footprint":           "LiDAR-derived-convex-hull",
            "height":              "LiDAR-derived",
            "floors":              "estimated",
            "roof_type":           "LiDAR-derived-heuristic",
            "building_id":         "prototype",
        },
        "confidence_note": confidence_note,
    }


# ── Scene-level output ────────────────────────────────────────────────────────

def scene_record(
    scene_id: str,
    source_dataset: str,
    source_ply: str,
    ai_model: str,
    buildings: list[dict],
    run_params: dict,
) -> dict[str, Any]:
    """Build the top-level scene output document."""
    return {
        "schema_version":  "reconstruction_v2",
        "scene_id":        scene_id,
        "source_dataset":  source_dataset,
        "source_ply":      source_ply,
        "ai_model":        ai_model,
        "n_buildings":     len(buildings),
        "run_parameters":  run_params,
        "buildings":       buildings,
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def save_buildings_json(scene: dict, out_path: "str | Path") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)


def load_buildings_json(path: "str | Path") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

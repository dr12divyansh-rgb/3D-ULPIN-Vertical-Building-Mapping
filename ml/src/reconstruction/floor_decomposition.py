"""Height-based floor decomposition for SIH 26011.

THIS IS AN ESTIMATE. Aerial LiDAR typically observes roof and ground, but
does NOT directly observe individual floor slabs inside a building. Floor
count and boundaries here are inferred from total building height only.

Method: HEIGHT-BASED FLOOR DECOMPOSITION
  1. Determine ground_z (LiDAR-derived min percentile of building points).
  2. Determine roof_z   (LiDAR-derived max Z of building points).
  3. height = roof_z - ground_z
  4. floor_count = max(1, round(height / floor_height_m))
  5. Divide [ground_z, roof_z] into floor_count equal vertical bands.

Limitations (must be acknowledged in all outputs):
  - Floor slab positions are estimated, not observed.
  - Aerial LiDAR cannot confirm actual number of floors for tall buildings
    when only roof is sampled.
  - Dense terrestrial LiDAR (STPLS3D) may show façade detail that could
    partially corroborate floor count, but this module does not exploit it.
  - Results should be labelled source="estimated" in all downstream outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_FLOOR_HEIGHT_M = 3.2   # metres — configurable
GROUND_PERCENTILE      = 5.0   # percentile used for ground_z estimation
ROOF_PERCENTILE        = 99.0  # percentile used for roof_z (filters outliers)


# ── Floor record ───────────────────────────────────────────────────────────────

@dataclass
class FloorRecord:
    floor_id: str          # e.g. "DALES_scene_0001/F01"
    building_id: str
    floor_number: int      # 1-based
    z_min: float
    z_max: float
    height: float          # z_max - z_min (may differ from floor_height_m at top)
    source: str            # always "estimated" for this module
    confidence: None = None  # this module does not produce confidence values

    def to_dict(self) -> dict:
        return {
            "floor_id":     self.floor_id,
            "building_id":  self.building_id,
            "floor_number": self.floor_number,
            "z_min":        round(self.z_min, 3),
            "z_max":        round(self.z_max, 3),
            "height":       round(self.height, 3),
            "source":       self.source,
            "confidence":   self.confidence,
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def decompose_floors(
    building_xyz: np.ndarray,
    building_id: str,
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
) -> tuple[float, float, float, int, list[FloorRecord]]:
    """
    Estimate floor decomposition from a building's point cloud.

    Height-based floor decomposition. Call it exactly that — not AI floor
    segmentation.

    Args:
        building_xyz:   (N, 3) float32 — points belonging to this building.
        building_id:    Prototype building ID (used to form floor_id).
        floor_height_m: Assumed floor-to-floor height in metres.
                        Configurable; default 3.2 m.

    Returns:
        (ground_z, roof_z, height, floor_count, floors)
        All elevation values are LiDAR-derived.
    """
    if len(building_xyz) == 0:
        return 0.0, 0.0, 0.0, 0, []

    z = building_xyz[:, 2]

    # Ground z: use low percentile to be robust to isolated noise below building
    ground_z = float(np.percentile(z, GROUND_PERCENTILE))
    # Roof z: use high percentile to filter outlier noise above peak
    roof_z   = float(np.percentile(z, ROOF_PERCENTILE))

    # Safety clamp
    if roof_z <= ground_z:
        roof_z = float(z.max())
    if roof_z <= ground_z:
        return ground_z, ground_z, 0.0, 1, _single_floor(building_id, ground_z, ground_z)

    height = roof_z - ground_z
    floor_count = max(1, round(height / floor_height_m))

    # Build equal vertical bands
    band_h = height / floor_count
    floors: list[FloorRecord] = []
    for fn in range(1, floor_count + 1):
        z_min = ground_z + (fn - 1) * band_h
        z_max = ground_z + fn * band_h
        z_max = min(z_max, roof_z)   # clamp top floor to actual roof
        floors.append(FloorRecord(
            floor_id     = f"{building_id}/F{fn:02d}",
            building_id  = building_id,
            floor_number = fn,
            z_min        = z_min,
            z_max        = z_max,
            height       = z_max - z_min,
            source       = "estimated",
            confidence   = None,
        ))

    return ground_z, roof_z, height, floor_count, floors


def _single_floor(building_id: str, z_min: float, z_max: float) -> list[FloorRecord]:
    return [FloorRecord(
        floor_id     = f"{building_id}/F01",
        building_id  = building_id,
        floor_number = 1,
        z_min        = z_min,
        z_max        = z_max,
        height       = z_max - z_min,
        source       = "estimated",
        confidence   = None,
    )]

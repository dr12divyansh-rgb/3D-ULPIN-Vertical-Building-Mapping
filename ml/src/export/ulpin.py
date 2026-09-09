"""ULPIN identifier generator for SIH 26011.

A ULPIN (Unique Land Parcel Identification Number) is deterministically derived
from the project seed (26011), the scene ID, and the building_id.

Format:  ULPIN-26011-S{scene_num:04d}-B{building_id:06d}
Example: ULPIN-26011-S0005-B040001
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


def ulpin_from_ids(scene_id: str, building_id: int) -> str:
    """Return the canonical ULPIN string for a building.

    Args:
        scene_id: e.g. "scene_00005" or "00005" or "5"
        building_id: integer building_id from reconstruction metadata
    """
    m = re.search(r"(\d+)", str(scene_id))
    num = int(m.group(1)) if m else 0
    return f"ULPIN-26011-S{num:04d}-B{building_id:06d}"


def write_registry(records: list[dict], out_path: Path) -> None:
    """Write ULPIN registry CSV.

    Each record must have: ulpin, scene_id, building_id, roof_type,
    height_pred, footprint_vertices, watertight.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ulpin", "scene_id", "building_id", "roof_type",
              "height_pred_m", "footprint_vertices", "watertight"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  ULPIN registry → {out_path}  ({len(records)} buildings)")

"""Post-process research_v2 buildings with accurate AHN4 ground elevation.

Uses ASPRS class-2 ground points from the original AHN4 LAZ file instead
of Exp007-predicted ground-class points. This eliminates the -2.1m ground-
elevation bias caused by AI prediction errors on the ground class.

Also applies a gable-roof height correction for floor estimation:
  flat roof: floor_count = round(height / 3.2)
  gable/shed: floor_count = round((height - 0.8) / 3.2)

Writes updated buildings.json + viewer_buildings.json to the same directory.
Does NOT modify any other session's files.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from ml.src.reconstruction.floor_decomposition import DEFAULT_FLOOR_HEIGHT_M
from ml.src.reconstruction.export_viewer_json import convert as export_viewer


def poly_centroid(poly: list) -> tuple[float, float]:
    if not poly: return (0.0, 0.0)
    return (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))


def estimate_ground_z_from_asprs(
    gnd_x: np.ndarray,
    gnd_y: np.ndarray,
    gnd_z: np.ndarray,
    footprint: list,
    buffer_m: float = 3.0,
) -> tuple[float, str]:
    """Find ASPRS ground points near the building footprint and return median Z.

    Returns (ground_z, method_used).
    Falls back to a global default if no nearby ground points exist.
    """
    if not footprint:
        return 1.0, "global-default"

    fp = [(p[0], p[1]) for p in footprint]
    xs = [p[0] for p in fp]; ys = [p[1] for p in fp]
    xmin = min(xs) - buffer_m; xmax = max(xs) + buffer_m
    ymin = min(ys) - buffer_m; ymax = max(ys) + buffer_m

    bbox = (gnd_x >= xmin) & (gnd_x <= xmax) & (gnd_y >= ymin) & (gnd_y <= ymax)
    local_z = gnd_z[bbox]

    if len(local_z) >= 20:
        return float(np.median(local_z)), "asprs-class2-dense"
    elif len(local_z) >= 5:
        return float(np.median(local_z)), "asprs-class2-sparse"
    else:
        return 1.0, "asprs-fallback-global"


def estimate_floors(height: float, roof_type: str, floor_h: float = 3.2) -> int:
    if height < 2.0: return 1
    # Gable correction: ridge adds ~0.8m apparent height
    adj = height - 0.8 if roof_type in ("gable", "shed") else height
    return max(1, round(adj / floor_h))


def refine_buildings(
    buildings_json: Path,
    asprs_ground_npz: Path,
    out_dir: Path,
    crs: str = "EPSG:28992 (Amersfoort / RD New)",
    building_iou: float = 0.7551,
) -> dict:
    """Recompute heights and floor counts using ASPRS ground elevation."""
    print("\nLoading buildings...")
    with open(buildings_json) as f:
        scene = json.load(f)
    buildings = scene["buildings"]
    print(f"  Buildings: {len(buildings)}")

    print("Loading ASPRS ground points...")
    d = np.load(asprs_ground_npz)
    gnd_x = d["x"]; gnd_y = d["y"]; gnd_z = d["z"]
    print(f"  Ground points: {len(gnd_x):,}  mean Z={gnd_z.mean():.3f}m")

    # Rebuild each building's height and floor count
    method_counts: dict[str, int] = {}
    h_old = []; h_new = []

    for b in buildings:
        fp = b.get("footprint", [])
        roof_type = b.get("roof_type", "flat")
        roof_z = b.get("roof_z", 5.0)

        # New ground elevation from ASPRS
        new_gz, method = estimate_ground_z_from_asprs(gnd_x, gnd_y, gnd_z, fp)
        new_height = max(0.0, float(roof_z) - new_gz)
        new_floors  = estimate_floors(new_height, roof_type)

        method_counts[method] = method_counts.get(method, 0) + 1
        h_old.append(b.get("height", 0.0))
        h_new.append(new_height)

        # Update record
        b["ground_z"]    = round(new_gz, 3)
        b["height"]      = round(new_height, 2)
        b["floor_count"] = new_floors
        b["height_method"] = method
        b["height_source"] = "ASPRS-class2-ground" if "asprs" in method else b.get("height_source", "LiDAR-derived")
        b["ground_z_source"] = method

        # Update floor bands (simple equal-height bands)
        bands = []
        for fl in range(new_floors):
            fz_lo = new_gz + fl * 3.2
            fz_hi = new_gz + (fl + 1) * 3.2
            bands.append({
                "floor_id":     f"{b['building_id']}/F{fl+1:02d}",
                "building_id":  b["building_id"],
                "floor_number": fl + 1,
                "z_min":        round(fz_lo, 3),
                "z_max":        round(min(fz_hi, float(roof_z)), 3),
                "height":       round(min(3.2, float(roof_z) - fz_lo), 2),
                "source":       "estimated",
                "confidence":   None,
            })
        b["floors"]      = bands
        b["floor_height_m_assumed"] = 3.2

    h_old = np.array(h_old); h_new = np.array(h_new)
    print(f"\nHeight update:")
    print(f"  Old: mean={h_old.mean():.2f}m  median={np.median(h_old):.2f}m")
    print(f"  New: mean={h_new.mean():.2f}m  median={np.median(h_new):.2f}m")
    print(f"  Change: mean={h_new.mean()-h_old.mean():+.2f}m  median={np.median(h_new-h_old):+.2f}m")
    print(f"\nGround estimation methods:")
    for m, n in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {m}: {n} ({100*n/len(buildings):.0f}%)")

    # Save updated buildings.json
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "buildings.json"
    with open(out_json, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Viewer export
    try:
        export_viewer(
            input_path   = out_json,
            output_path  = out_dir / "viewer_buildings.json",
            dataset      = "AHN4",
            model_id     = "exp007",
            building_iou = building_iou,
            crs          = crs,
        )
        print(f"  Viewer JSON updated")
    except Exception as e:
        print(f"  Viewer export warning: {e}")

    return {"n_buildings": len(buildings), "h_old_mean": float(h_old.mean()), "h_new_mean": float(h_new.mean())}


def main():
    bldg_json = ROOT / "ml/results/final_ahn4_research_v2/buildings.json"
    ground_npz = ROOT / "ml/results/final_ahn4_research_v2/ahn4_ground_pts.npz"
    out_dir    = ROOT / "ml/results/final_ahn4_research_v2"

    refine_buildings(bldg_json, ground_npz, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

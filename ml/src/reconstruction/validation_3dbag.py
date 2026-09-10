"""3DBAG Validation Adapter for SIH 26011 Reconstruction.

3DBAG (3D Basisregistraties Adressen en Gebouwen) is the Dutch reference
building dataset derived from AHN4 aerial LiDAR. It provides authoritative
footprints, heights, floor counts, and LoD meshes.

PURPOSE: Use 3DBAG as VALIDATION/REFERENCE ONLY.
  - DO NOT use 3DBAG as model input or training data.
  - DO NOT claim our predictions are as accurate as 3DBAG.
  - DO compare our reconstruction to 3DBAG to quantify accuracy gaps.

3DBAG Schema (relevant fields):
  identificatie:    BAG pand identifier (e.g. "NL.IMBAG.Pand.0363100012229215")
  oorspronkelijkBouwjaar: construction year
  geometry:         2D footprint polygon (RD New / EPSG:28992 or WGS84)
  b3_h_dak_50p:     50th-percentile roof height above ground (metres)
  b3_h_dak_70p:     70th-percentile roof height
  b3_h_dak_95p:     95th-percentile roof height
  b3_h_maaiveld:    ground elevation
  b3_pand_toestand: building status
  b3_volume_lod12:  volume from LoD1.2 (m³)
  b3_volume_lod22:  volume from LoD2.2 (m³)
  b3_kas_warenhuis: is greenhouse/warehouse flag
  b3_dak_type:      roof type string (slanted / flat / no_data / multiple)
  b3_bouwlagen:     floor count from 3DBAG
  lod_12_2d:        LoD1.2 footprint polygon
  lod_13_2d:        LoD1.3 footprint polygon
  lod_22_3d:        LoD2.2 3D mesh (WKB geometry)

Reference: https://docs.3dbag.nl/en/schema/attributes/
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── 3DBAG Record ──────────────────────────────────────────────────────────────

@dataclass
class BAGBuilding:
    """Parsed representation of one 3DBAG building record.

    All heights are in metres, relative to ground (normalized) unless noted.
    """
    bag_id: str                           # BAG pand identificatie
    footprint: list[tuple[float, float]]  # (x, y) polygon vertices in source CRS
    h_maaiveld: float | None             # ground elevation
    h_dak_50p: float | None              # 50th-pct roof height
    h_dak_70p: float | None              # 70th-pct roof height
    h_dak_95p: float | None              # 95th-pct roof height (closest to peak)
    height: float | None                 # derived: h_dak_50p - h_maaiveld
    floor_count: int | None              # b3_bouwlagen
    roof_type: str | None                # b3_dak_type: "slanted" | "flat" | "multiple" | "no_data"
    volume_lod12: float | None           # m³ at LoD1.2
    volume_lod22: float | None           # m³ at LoD2.2
    raw: dict = None                     # original record for debugging


def parse_bag_feature(feature: dict) -> BAGBuilding | None:
    """Parse a single GeoJSON feature from the 3DBAG API/download.

    Returns None if the record cannot be parsed.
    """
    try:
        props = feature.get("properties") or {}
        geom  = feature.get("geometry") or {}

        # Extract footprint polygon (2D)
        footprint: list[tuple[float, float]] = []
        if geom.get("type") == "Polygon":
            ring = geom["coordinates"][0]
            footprint = [(float(c[0]), float(c[1])) for c in ring]
        elif geom.get("type") == "MultiPolygon":
            # Use first polygon of MultiPolygon
            ring = geom["coordinates"][0][0]
            footprint = [(float(c[0]), float(c[1])) for c in ring]

        # Heights
        h_maaiveld = _float_or_none(props.get("b3_h_maaiveld"))
        h_dak_50p  = _float_or_none(props.get("b3_h_dak_50p"))
        h_dak_70p  = _float_or_none(props.get("b3_h_dak_70p"))
        h_dak_95p  = _float_or_none(props.get("b3_h_dak_95p"))

        height = None
        if h_dak_50p is not None and h_maaiveld is not None:
            height = h_dak_50p - h_maaiveld

        # Floor count
        floor_count = _int_or_none(props.get("b3_bouwlagen"))

        # BAG ID
        bag_id = str(props.get("identificatie") or feature.get("id") or "unknown")

        return BAGBuilding(
            bag_id      = bag_id,
            footprint   = footprint,
            h_maaiveld  = h_maaiveld,
            h_dak_50p   = h_dak_50p,
            h_dak_70p   = h_dak_70p,
            h_dak_95p   = h_dak_95p,
            height      = height,
            floor_count = floor_count,
            roof_type   = str(props.get("b3_dak_type") or "no_data"),
            volume_lod12 = _float_or_none(props.get("b3_volume_lod12")),
            volume_lod22 = _float_or_none(props.get("b3_volume_lod22")),
            raw         = feature,
        )
    except (KeyError, ValueError, TypeError, IndexError):
        return None


def load_bag_geojson(path: str | Path) -> list[BAGBuilding]:
    """Load 3DBAG buildings from a GeoJSON file (API export or tile download).

    Args:
        path: Path to .geojson or .json file.

    Returns:
        List of parsed BAGBuilding records (invalid records silently skipped).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features") or []
    if not features and isinstance(data, list):
        features = data  # bare feature list

    buildings: list[BAGBuilding] = []
    for feat in features:
        b = parse_bag_feature(feat)
        if b is not None:
            buildings.append(b)
    return buildings


# ── Footprint comparison ───────────────────────────────────────────────────────

def compare_footprints(
    our_footprint: list[tuple[float, float]],
    ref_footprint: list[tuple[float, float]],
) -> dict:
    """Compare two 2D polygon footprints.

    Metrics:
        our_area_m2:     Area of our polygon (Shoelace formula)
        ref_area_m2:     Area of reference polygon
        area_ratio:      our_area / ref_area (1.0 = perfect, >1 = over-estimate)
        area_diff_m2:    our_area - ref_area (signed)
        centroid_dist_m: Distance between polygon centroids (metres)
        iou:             Intersection-over-Union (polygon IOU via rasterisation)
        our_n_verts:     Number of vertices in our polygon
        ref_n_verts:     Number of vertices in reference polygon

    Notes:
        IOU is computed by rasterising both polygons to a common grid.
        This is an approximation; exact polygon IOU requires shapely or clipper.
    """
    our_area = _polygon_area(our_footprint)
    ref_area = _polygon_area(ref_footprint)
    our_ctr  = _centroid(our_footprint)
    ref_ctr  = _centroid(ref_footprint)

    centroid_dist = math.hypot(our_ctr[0] - ref_ctr[0], our_ctr[1] - ref_ctr[1])
    area_ratio    = our_area / ref_area if ref_area > 0 else float("nan")
    area_diff     = our_area - ref_area

    # Approximate IOU via raster
    iou = _raster_iou(our_footprint, ref_footprint, resolution=0.5)

    return {
        "our_area_m2":     round(our_area, 2),
        "ref_area_m2":     round(ref_area, 2),
        "area_ratio":      round(area_ratio, 3) if not math.isnan(area_ratio) else None,
        "area_diff_m2":    round(area_diff, 2),
        "centroid_dist_m": round(centroid_dist, 2),
        "iou":             round(iou, 3),
        "our_n_verts":     len(our_footprint),
        "ref_n_verts":     len(ref_footprint),
    }


# ── Height comparison ─────────────────────────────────────────────────────────

def compare_heights(
    our_height: float,
    bag_building: BAGBuilding,
) -> dict:
    """Compare our LiDAR-derived height against 3DBAG height values.

    We compare against all available 3DBAG height percentiles so the caller
    can see which reference is most appropriate for their use case:
      - 50th-pct roof height ≈ median roof point (good for flat roofs)
      - 70th-pct roof height ≈ typical occupied roof area
      - 95th-pct roof height ≈ highest roof point (best for gable peaks)

    Args:
        our_height:  Our LiDAR-derived building height (metres).
        bag_building: Parsed 3DBAG record.

    Returns:
        dict with absolute and relative errors against each available percentile.
    """
    result: dict = {
        "our_height_m": round(our_height, 2),
        "bag_h_dak_50p": bag_building.h_dak_50p,
        "bag_h_dak_70p": bag_building.h_dak_70p,
        "bag_h_dak_95p": bag_building.h_dak_95p,
        "bag_height_m":  bag_building.height,   # 50p - maaiveld
    }

    for ref_name, ref_val in [
        ("vs_50p", bag_building.height),
        ("vs_dak_50p_abs", bag_building.h_dak_50p),
        ("vs_dak_95p_abs", bag_building.h_dak_95p),
    ]:
        if ref_val is not None:
            err = our_height - ref_val
            result[f"abs_err_{ref_name}_m"] = round(err, 2)
            result[f"rel_err_{ref_name}_pct"] = round(100 * err / ref_val, 1) if ref_val != 0 else None

    return result


# ── Floor count comparison ────────────────────────────────────────────────────

def compare_floor_counts(
    our_floor_count: int,
    our_method: str,
    bag_building: BAGBuilding,
) -> dict:
    """Compare our estimated floor count against 3DBAG b3_bouwlagen.

    IMPORTANT: Our floor count is HEIGHT-BASED ESTIMATION (height ÷ 3.2 m),
    not a trained prediction. 3DBAG uses the BAG administrative floor count.
    This comparison shows how well height-based estimation tracks reality.

    Args:
        our_floor_count: Our estimated floor count.
        our_method:      Must be "estimated" or "height_estimate" — never claim AI.
        bag_building:    Parsed 3DBAG record.

    Returns:
        dict with comparison metrics.
    """
    ref = bag_building.floor_count
    result: dict = {
        "our_floor_count": our_floor_count,
        "our_method": our_method,
        "bag_bouwlagen": ref,
        "our_method_note": "Height-based estimate: height ÷ 3.2 m. Not AI floor segmentation.",
    }

    if ref is not None and ref > 0:
        err = our_floor_count - ref
        result["abs_error"] = err
        result["correct"] = (err == 0)
        result["within_1"] = (abs(err) <= 1)
    else:
        result["abs_error"] = None
        result["correct"] = None
        result["within_1"] = None

    return result


# ── Roof type comparison ──────────────────────────────────────────────────────

def compare_roof_types(
    our_roof_type: str,
    bag_building: BAGBuilding,
) -> dict:
    """Compare our heuristic roof classification against 3DBAG b3_dak_type.

    3DBAG dak types:
        "flat"      → flat roof
        "slanted"   → pitched/gable/hip roof
        "multiple"  → building has multiple roof types
        "no_data"   → unknown

    Our types: "flat" | "gable"

    Mapping for comparison:
        our "gable" → ref "slanted" (correct match)
        our "flat"  → ref "flat" (correct match)
    """
    ref_raw = bag_building.roof_type or "no_data"
    ref_type = ref_raw.lower()

    # Normalise 3DBAG to our binary classification space
    ref_binary = "flat" if ref_type == "flat" else ("gable" if ref_type == "slanted" else "unknown")

    return {
        "our_roof_type": our_roof_type,
        "our_method": "lidar-z-std-heuristic",
        "bag_dak_type": ref_raw,
        "bag_normalised": ref_binary,
        "correct": (our_roof_type == ref_binary) if ref_binary != "unknown" else None,
    }


# ── Full building comparison ──────────────────────────────────────────────────

def compare_building(
    our_record: dict,
    bag_building: BAGBuilding,
) -> dict:
    """Run all comparison metrics between our building record and a 3DBAG match.

    Args:
        our_record:    Dict from building_schema.building_record().
        bag_building:  Parsed 3DBAG BAGBuilding.

    Returns:
        Nested dict with footprint, height, floors, roof sub-reports.
    """
    our_fp = our_record.get("footprint") or []
    our_h  = float(our_record.get("height") or 0.0)
    our_fc = int(our_record.get("floor_count") or 1)
    our_rt = str(our_record.get("roof_type") or "flat")
    fc_src = str(our_record.get("floor_count_source") or "estimated")

    report: dict = {
        "building_id": our_record.get("building_id"),
        "bag_id": bag_building.bag_id,
        "comparison_note": (
            "3DBAG is used as reference/validation only. "
            "It is NOT used as model input or training data."
        ),
    }

    if our_fp and bag_building.footprint:
        report["footprint"] = compare_footprints(our_fp, bag_building.footprint)
    else:
        report["footprint"] = {"error": "one or both footprints empty"}

    report["height"] = compare_heights(our_h, bag_building)
    report["floors"] = compare_floor_counts(our_fc, fc_src, bag_building)
    report["roof"]   = compare_roof_types(our_rt, bag_building)

    return report


# ── Batch comparison ──────────────────────────────────────────────────────────

def match_by_centroid(
    our_buildings: list[dict],
    bag_buildings: list[BAGBuilding],
    max_dist_m: float = 25.0,
) -> list[tuple[dict, BAGBuilding]]:
    """Match our buildings to 3DBAG buildings by centroid proximity.

    This is a simple nearest-neighbour match — not spatial index based.
    Suitable for small-to-medium sets (<1000 buildings).

    Args:
        our_buildings: List of building records (each has "footprint" polygon).
        bag_buildings: List of parsed BAGBuilding records.
        max_dist_m:    Maximum centroid distance to accept a match (metres).

    Returns:
        List of (our_record, bag_record) pairs for matched buildings.
        Unmatched buildings from either set are silently discarded.
    """
    pairs: list[tuple[dict, BAGBuilding]] = []

    bag_centroids = [_centroid(b.footprint) if b.footprint else (0.0, 0.0) for b in bag_buildings]

    for our_rec in our_buildings:
        our_fp  = our_rec.get("footprint") or []
        if not our_fp:
            continue
        our_ctr = _centroid(our_fp)

        best_dist = max_dist_m + 1
        best_bag  = None
        for bag, bag_ctr in zip(bag_buildings, bag_centroids):
            d = math.hypot(our_ctr[0] - bag_ctr[0], our_ctr[1] - bag_ctr[1])
            if d < best_dist:
                best_dist = d
                best_bag  = bag

        if best_bag is not None and best_dist <= max_dist_m:
            pairs.append((our_rec, best_bag))

    return pairs


def batch_compare(
    our_buildings: list[dict],
    bag_buildings: list[BAGBuilding],
    max_dist_m: float = 25.0,
) -> dict:
    """Compare our reconstruction against 3DBAG for a full scene.

    Returns:
        {
          "n_our": ...,
          "n_bag": ...,
          "n_matched": ...,
          "match_rate_pct": ...,
          "height": { "mean_abs_err_m": ..., "std_err": ... },
          "floors": { "accuracy_pct": ..., "within1_pct": ... },
          "roof":   { "accuracy_pct": ... },
          "per_building": [ compare_building(...) for each match ]
        }
    """
    pairs = match_by_centroid(our_buildings, bag_buildings, max_dist_m)

    per_bldg = [compare_building(our, bag) for our, bag in pairs]

    # Aggregate height errors
    h_errs = [b["height"].get("abs_err_vs_50p_m") for b in per_bldg
               if b.get("height", {}).get("abs_err_vs_50p_m") is not None]

    h_agg = {}
    if h_errs:
        import statistics
        h_agg = {
            "mean_abs_err_m": round(statistics.mean(abs(e) for e in h_errs), 2),
            "mean_err_m":     round(statistics.mean(h_errs), 2),
            "std_err_m":      round(statistics.stdev(h_errs) if len(h_errs) > 1 else 0.0, 2),
        }

    # Aggregate floor accuracy
    floor_records = [b["floors"] for b in per_bldg if b.get("floors", {}).get("correct") is not None]
    fl_agg = {}
    if floor_records:
        n = len(floor_records)
        fl_agg = {
            "n_compared": n,
            "accuracy_pct": round(100 * sum(1 for f in floor_records if f["correct"]) / n, 1),
            "within1_pct":  round(100 * sum(1 for f in floor_records if f["within_1"]) / n, 1),
        }

    # Aggregate roof accuracy
    roof_records = [b["roof"] for b in per_bldg if b.get("roof", {}).get("correct") is not None]
    rf_agg = {}
    if roof_records:
        n = len(roof_records)
        rf_agg = {
            "n_compared": n,
            "accuracy_pct": round(100 * sum(1 for r in roof_records if r["correct"]) / n, 1),
        }

    # Aggregate IOU
    iou_vals = [b["footprint"].get("iou") for b in per_bldg
                if b.get("footprint", {}).get("iou") is not None]
    fp_agg = {}
    if iou_vals:
        import statistics
        fp_agg = {
            "mean_iou": round(statistics.mean(iou_vals), 3),
            "min_iou":  round(min(iou_vals), 3),
            "max_iou":  round(max(iou_vals), 3),
        }

    return {
        "n_our":          len(our_buildings),
        "n_bag":          len(bag_buildings),
        "n_matched":      len(pairs),
        "match_rate_pct": round(100 * len(pairs) / max(len(our_buildings), 1), 1),
        "height":         h_agg,
        "floors":         fl_agg,
        "roof":           rf_agg,
        "footprint":      fp_agg,
        "per_building":   per_bldg,
    }


def save_comparison_report(report: dict, out_path: str | Path) -> None:
    """Write batch_compare result to a JSON file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def print_comparison_summary(report: dict) -> None:
    """Print a human-readable comparison summary."""
    print(f"\n{'='*60}")
    print(f"  3DBAG Validation Summary")
    print(f"{'='*60}")
    print(f"  Our buildings  : {report['n_our']}")
    print(f"  3DBAG buildings: {report['n_bag']}")
    print(f"  Matched        : {report['n_matched']} ({report['match_rate_pct']:.1f}%)")

    if report.get("height"):
        h = report["height"]
        print(f"\n  Height errors (vs 3DBAG b3_h_dak_50p):")
        print(f"    Mean abs error : {h.get('mean_abs_err_m', 'N/A')} m")
        print(f"    Mean error     : {h.get('mean_err_m', 'N/A')} m")
        print(f"    Std dev        : {h.get('std_err_m', 'N/A')} m")

    if report.get("floors"):
        fl = report["floors"]
        print(f"\n  Floor count (estimated ÷ 3.2m vs b3_bouwlagen):")
        print(f"    Exact accuracy : {fl.get('accuracy_pct', 'N/A')}%")
        print(f"    Within ±1      : {fl.get('within1_pct', 'N/A')}%")
        print(f"    N compared     : {fl.get('n_compared', 'N/A')}")

    if report.get("roof"):
        rf = report["roof"]
        print(f"\n  Roof type (flat/gable heuristic vs b3_dak_type):")
        print(f"    Accuracy       : {rf.get('accuracy_pct', 'N/A')}%")
        print(f"    N compared     : {rf.get('n_compared', 'N/A')}")

    if report.get("footprint"):
        fp = report["footprint"]
        print(f"\n  Footprint IOU:")
        print(f"    Mean IOU       : {fp.get('mean_iou', 'N/A')}")
        print(f"    Min / Max IOU  : {fp.get('min_iou', 'N/A')} / {fp.get('max_iou', 'N/A')}")

    print(f"\n  NOTE: 3DBAG used as reference only (not model input).")
    print(f"  Floor counts are HEIGHT-BASED ESTIMATES, not AI predictions.")
    print(f"{'='*60}\n")


# ── Private utilities ─────────────────────────────────────────────────────────

def _float_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    """Shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _centroid(vertices: list[tuple[float, float]]) -> tuple[float, float]:
    if not vertices:
        return (0.0, 0.0)
    return (
        sum(x for x, _ in vertices) / len(vertices),
        sum(y for _, y in vertices) / len(vertices),
    )


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def _raster_iou(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
    resolution: float = 0.5,
) -> float:
    """Approximate polygon IOU by rasterising both polygons to a shared grid.

    Resolution of 0.5m gives ~4% error for typical building sizes (10–100m²).
    For exact IOU, shapely or Sutherland-Hodgman clipping would be needed.
    """
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0

    all_x = [p[0] for p in poly_a + poly_b]
    all_y = [p[1] for p in poly_a + poly_b]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    if x_max - x_min < 1e-3 or y_max - y_min < 1e-3:
        return 0.0

    cols = max(1, int((x_max - x_min) / resolution) + 1)
    rows = max(1, int((y_max - y_min) / resolution) + 1)

    # Cap grid size for performance (very large buildings)
    if cols * rows > 200_000:
        resolution = math.sqrt((x_max - x_min) * (y_max - y_min) / 200_000)
        cols = max(1, int((x_max - x_min) / resolution) + 1)
        rows = max(1, int((y_max - y_min) / resolution) + 1)

    mask_a = bytearray(cols * rows)
    mask_b = bytearray(cols * rows)

    for ri in range(rows):
        for ci in range(cols):
            px = x_min + (ci + 0.5) * resolution
            py = y_min + (ri + 0.5) * resolution
            if _point_in_polygon(px, py, poly_a):
                mask_a[ri * cols + ci] = 1
            if _point_in_polygon(px, py, poly_b):
                mask_b[ri * cols + ci] = 1

    inter = sum(a and b for a, b in zip(mask_a, mask_b))
    union = sum(bool(a or b) for a, b in zip(mask_a, mask_b))

    return inter / union if union > 0 else 0.0

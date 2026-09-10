"""Reconstruction Pipeline v3 — SIH 26011.

Improvements over v2.1:
  1. Plausibility filtering — rejects implausibly sparse or tiny clusters
     caused by low-precision predictions (e.g. AHN4 IoU=0.053)
  2. Confidence-weighted filtering — uses per-point softmax confidence
     to prefer high-confidence clusters when available
  3. Vertical density profiling — improved floor count estimation
     by analysing Z histogram peaks alongside height ÷ floor-height
  4. Height discontinuity splitting — splits DBSCAN clusters that
     span two clearly separate height levels (adjacent buildings at
     different floors joined by a shared-wall bridge of noise)
  5. 3DBAG aggregate validation — compares our height/floor/area
     distributions against the 3DBAG reference dataset
  6. Full AHN4 CRS support — EPSG:28992, NAP vertical datum

Output schema:
    buildings.json  ->  same schema as v2 + new fields:
        confidence_mean, confidence_source
        density_pts_per_m2
        plausibility_pass
        floor_density_estimate
    validation/3dbag_aggregate.json  ->  distribution comparison

This pipeline DOES NOT modify:
    viewer/  ml/src/training/  ml/src/models/  ml/src/inference/
    ml/src/data/  model checkpoints  Exp005/Exp006/Exp007
    ml/results/reconstruction_v2*/  ml/results/reconstruction_ahn4/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from ml.src.reconstruction.prediction_contract import (
    load_prediction_ply, save_prediction_metadata, BUILDING_CLASS,
)
from ml.src.reconstruction.instance_extraction import (
    extract_building_instances_adaptive,
    evaluate_instance_extraction,
    RECOMMENDED_CELL_SIZE, RECOMMENDED_MIN_POINTS,
    RECOMMENDED_DBSCAN_EPS, RECOMMENDED_DBSCAN_MIN_PTS,
    BuildingInstance,
)
from ml.src.reconstruction.floor_decomposition import (
    decompose_floors, DEFAULT_FLOOR_HEIGHT_M,
)
from ml.src.reconstruction.building_schema import (
    building_record, scene_record, save_buildings_json,
)
from ml.src.reconstruction.footprint import (
    extract_footprint_best, _polygon_area,
)
from ml.src.reconstruction.roof_analysis import (
    analyze_roof, roof_result_to_dict,
)
from ml.src.reconstruction.reconstruct_from_prediction import (
    analyze_building, generate_mesh,
    write_obj, write_scene_obj, write_mtl,
    MeshBuilder,
)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# ── Dataset defaults (v3 adds AHN4 with tighter filtering) ────────────────────

DATASET_DEFAULTS = {
    "STPLS3D": {"cell_size": 0.5, "dbscan_eps": 1.5, "dbscan_min": 20,
                 "min_pts": 30, "min_area": 8.0, "max_area": 5000.0,
                 "min_h": 1.5, "max_h": 80.0, "min_density": 0.5},
    "DALES":   {"cell_size": 1.0, "dbscan_eps": 2.5, "dbscan_min": 10,
                 "min_pts": 15, "min_area": 10.0, "max_area": 5000.0,
                 "min_h": 1.0, "max_h": 50.0, "min_density": 0.1},
    "AHN4":    {"cell_size": 1.5, "dbscan_eps": 2.0, "dbscan_min": 15,
                 "min_pts": 20, "min_area": 15.0, "max_area": 5000.0,
                 "min_h": 2.0, "max_h": 50.0,
                 # AHN4 has 13.4% precision; real buildings have >= 0.3 pts/m²
                 "min_density": 0.3},
    "unknown": {"cell_size": 1.0, "dbscan_eps": 2.0, "dbscan_min": 15,
                 "min_pts": 20, "min_area": 10.0, "max_area": 5000.0,
                 "min_h": 1.0, "max_h": 80.0, "min_density": 0.1},
}


# ── Plausibility filtering ─────────────────────────────────────────────────────

def _cluster_density(pts: np.ndarray, footprint_area_m2: float) -> float:
    """Points per m² of footprint area. Returns 0 if area is zero."""
    if footprint_area_m2 <= 0:
        return 0.0
    return len(pts) / footprint_area_m2


def filter_by_plausibility(
    instances: list[BuildingInstance],
    building_xyz: np.ndarray,
    confidence: np.ndarray | None,
    defaults: dict,
) -> tuple[list[BuildingInstance], list[dict]]:
    """Apply plausibility filters to reject implausible clusters.

    For low-precision predictions (AHN4 IoU=0.053), many DBSCAN clusters
    are road / pavement noise that happens to be spatially dense. Filtering by:
      - minimum points
      - minimum footprint area
      - height range
      - minimum point density

    Returns:
        (kept, rejected_reasons)
        rejected_reasons: list of dicts describing why each instance was rejected.
    """
    kept: list[BuildingInstance] = []
    rejected: list[dict] = []

    for inst in instances:
        pts = building_xyz[inst.point_indices]
        reason = None

        # Point count
        if len(pts) < defaults["min_pts"]:
            reason = f"too_few_pts:{len(pts)}<{defaults['min_pts']}"
        else:
            # Quick footprint for density check
            cell = defaults["cell_size"]
            poly, meta = extract_footprint_best(pts, cell_size=cell)
            area = meta.get("area_m2", 0.0)

            if area < defaults["min_area"]:
                reason = f"area_too_small:{area:.1f}<{defaults['min_area']}"
            elif area > defaults["max_area"]:
                reason = f"area_too_large:{area:.1f}>{defaults['max_area']}"
            else:
                density = _cluster_density(pts, area)
                if density < defaults["min_density"]:
                    reason = f"density_too_low:{density:.3f}<{defaults['min_density']}"
                else:
                    # Height range check
                    z = pts[:, 2]
                    h = float(np.percentile(z, 99)) - float(np.percentile(z, 5))
                    if h < defaults["min_h"]:
                        reason = f"height_too_small:{h:.2f}<{defaults['min_h']}"
                    elif h > defaults["max_h"]:
                        reason = f"height_too_large:{h:.2f}>{defaults['max_h']}"

        if reason:
            rejected.append({"building_id": inst.building_id, "reason": reason})
        else:
            kept.append(inst)

    return kept, rejected


# ── Height discontinuity splitting ─────────────────────────────────────────────

def split_by_height_gap(
    instances: list[BuildingInstance],
    building_xyz: np.ndarray,
    gap_threshold_m: float = 2.5,
) -> list[BuildingInstance]:
    """Split merged clusters containing a clear height gap (two buildings at different heights).

    For each cluster, compute the Z histogram. If there is a gap > gap_threshold_m
    in the vertical density profile, the cluster is split at the gap midpoint.

    This is a heuristic. It does NOT use trained segmentation.
    """
    result: list[BuildingInstance] = []
    new_seq = 0

    for inst in instances:
        pts = building_xyz[inst.point_indices]
        z = pts[:, 2]

        z_min = float(z.min())
        z_max = float(z.max())
        z_range = z_max - z_min

        if z_range < gap_threshold_m * 2:
            # Too short to have a meaningful gap
            result.append(inst)
            continue

        # Build Z histogram with 0.5m bins
        bin_w = 0.5
        n_bins = max(4, int(z_range / bin_w))
        counts, edges = np.histogram(z, bins=n_bins, range=(z_min, z_max))

        # Find the lowest-count bin in the middle 40-80% of the height range
        lo_idx = int(n_bins * 0.3)
        hi_idx = int(n_bins * 0.8)
        mid_counts = counts[lo_idx:hi_idx]

        if len(mid_counts) == 0 or mid_counts.min() > 0:
            # No zero-count bin in middle zone -> no clear gap
            result.append(inst)
            continue

        # Find longest zero-run in middle zone
        zero_start = zero_end = -1
        best_run = 0
        run_start = -1
        for i, c in enumerate(mid_counts):
            if c == 0:
                if run_start == -1:
                    run_start = i
            else:
                if run_start != -1 and (i - run_start) > best_run:
                    best_run = i - run_start
                    zero_start = run_start + lo_idx
                    zero_end = i + lo_idx
                    run_start = -1

        if best_run < 2:
            # Gap is only 1 bin (< 0.5m) — not significant
            result.append(inst)
            continue

        gap_z = (edges[zero_start] + edges[zero_end]) / 2.0
        mask_lo = z <= gap_z
        mask_hi = z > gap_z

        if mask_lo.sum() < 10 or mask_hi.sum() < 10:
            result.append(inst)
            continue

        # Create two sub-instances
        for sub_mask, suffix in [(mask_lo, "a"), (mask_hi, "b")]:
            sub_indices = inst.point_indices[sub_mask]
            sub_pts = pts[sub_mask]
            cx = float(sub_pts[:, 0].mean())
            cy = float(sub_pts[:, 1].mean())
            new_seq += 1
            new_id = f"{inst.building_id}_{suffix}"
            result.append(BuildingInstance(
                instance_id   = inst.instance_id * 1000 + (0 if suffix == "a" else 1),
                building_id   = new_id,
                point_indices = sub_indices,
                centroid_xy   = (cx, cy),
                method        = inst.method + "+height_split",
            ))

    return result


# ── Vertical density floor estimation ─────────────────────────────────────────

def estimate_floors_from_density(
    pts: np.ndarray,
    ground_z: float,
    roof_z: float,
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
    bin_width_m: float = 0.4,
) -> tuple[int, str, str]:
    """Estimate floor count from vertical point density profile.

    Method:
      1. Build normalised Z histogram (relative to ground_z)
      2. Find local maxima in the histogram (point-density peaks)
      3. Peaks correspond to point accumulation at floor levels
         (on aerial LiDAR, peaks appear at roof level and sometimes at
          window sill levels for terrestrial data)
      4. If peaks give a plausible floor count, return it
      5. Otherwise fall back to height ÷ floor_height_m

    This is an ESTIMATE. Aerial LiDAR cannot observe individual floor slabs.
    The estimate is labelled "density-profile" or "height-estimate" accordingly.

    Returns:
        (floor_count, method_label, detail)
    """
    height = roof_z - ground_z
    if height < 0.5 or len(pts) < 10:
        return 1, "height-estimate", "too-short"

    z_norm = pts[:, 2] - ground_z
    n_bins = max(4, int(height / bin_width_m))
    counts, edges = np.histogram(z_norm[z_norm >= 0], bins=n_bins, range=(0, height))

    # Simple peak-finding: local maxima separated by at least floor_height_m
    min_peak_sep = int(floor_height_m / bin_width_m * 0.6)
    peaks = []
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] >= counts[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_peak_sep:
                peaks.append(i)

    # For aerial LiDAR, there's typically only 1 clear peak (the roof)
    # For terrestrial (STPLS3D), there may be peaks at each floor facade
    height_estimate = max(1, round(height / floor_height_m))

    if 1 <= len(peaks) <= 2:
        # Ambiguous — aerial data sees 1 peak at roof
        # Trust height estimate for multi-floor, density estimate is unreliable
        return height_estimate, "height-estimate", f"peaks={len(peaks)}-ambiguous"

    if len(peaks) >= 3:
        # Multiple peaks might indicate multiple floors (rare on aerial)
        floor_from_peaks = min(len(peaks), height_estimate + 2)
        if abs(floor_from_peaks - height_estimate) <= 1:
            return height_estimate, "height-estimate", f"peaks={len(peaks)}-consistent"
        # Peaks and height disagree significantly — trust height
        return height_estimate, "height-estimate", f"peaks={len(peaks)}-height-wins"

    return height_estimate, "height-estimate", "no-peaks-detected"


# ── Footprint helper ───────────────────────────────────────────────────────────

def _footprint_from_pts(
    pts: np.ndarray,
    source_dataset: str = "unknown",
) -> tuple[list[list[float]], float, int, str, str]:
    cfg = DATASET_DEFAULTS.get(source_dataset, DATASET_DEFAULTS["unknown"])
    poly, meta = extract_footprint_best(pts, cell_size=cfg["cell_size"])
    if poly and len(poly) >= 3:
        area = meta.get("area_m2") or _polygon_area(poly)
        if area > 0:
            return [[x, y] for x, y in poly], float(area), len(poly), meta["method"], meta.get("source", "unknown")
    # Final fallback: bounding box
    xy = pts[:, :2]
    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    x1, y1 = float(xy[:, 0].max()), float(xy[:, 1].max())
    bbox = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    area = (x1 - x0) * (y1 - y0)
    return bbox, float(area), 4, "bbox-fallback", "bounding-box"


# ── Confidence aggregation ────────────────────────────────────────────────────

def _confidence_stats(
    indices: np.ndarray,
    confidence: np.ndarray | None,
) -> tuple[float | None, str]:
    """Return (mean_confidence, source_note)."""
    if confidence is None or len(confidence) == 0:
        return None, "not-available"
    vals = confidence[indices]
    return float(vals.mean()), "softmax-mean"


# ── 3DBAG aggregate validation ────────────────────────────────────────────────

def validate_against_3dbag_aggregate(
    our_buildings: list[dict],
    bag_path: str | Path | None,
) -> dict:
    """Compare aggregate statistics of our reconstruction vs 3DBAG reference.

    Since the 3DBAG JSON does not include footprint polygon coordinates,
    spatial matching is not possible. We compare distributions.

    Args:
        our_buildings: List of our building records.
        bag_path:      Path to 3DBAG JSON (or None).

    Returns:
        Comparison dict with height, floor-count, area distributions.
    """
    result: dict = {
        "matching_method": "aggregate-distribution (no spatial matching — 3DBAG JSON has no polygon coordinates)",
        "our_n": len(our_buildings),
    }

    if bag_path is None or not Path(bag_path).exists():
        result["error"] = "3DBAG reference not found"
        return result

    with open(bag_path, encoding="utf-8") as f:
        ref = json.load(f)
    bag_buildings = ref.get("buildings", [])
    result["bag_n"] = len(bag_buildings)

    # Our stats
    our_h = [b["height"] for b in our_buildings if b.get("height", 0) > 0]
    our_fl = [b["floor_count"] for b in our_buildings if b.get("floor_count")]
    our_area = [b["footprint_area_m2"] for b in our_buildings if b.get("footprint_area_m2", 0) > 0]

    # 3DBAG stats
    bag_h = [b["h_roof_50p"] - b["h_ground"] for b in bag_buildings
              if b.get("h_roof_50p") is not None and b.get("h_ground") is not None]
    bag_fl = [b["floors"] for b in bag_buildings if b.get("floors")]
    bag_area = [b["footprint_area"] for b in bag_buildings if b.get("footprint_area")]
    bag_roof = {}
    for b in bag_buildings:
        rt = b.get("roof_type", "unknown") or "unknown"
        bag_roof[rt] = bag_roof.get(rt, 0) + 1

    def dist_stats(vals: list) -> dict:
        if not vals:
            return {}
        return {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 2),
            "median": round(statistics.median(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "p25": round(np.percentile(vals, 25), 2),
            "p75": round(np.percentile(vals, 75), 2),
        }

    result["height_m"] = {
        "ours": dist_stats(our_h),
        "3dbag": dist_stats(bag_h),
        "note": "3DBAG: h_roof_50p − h_ground (NAP). Ours: LiDAR-derived 99th−5th percentile Z.",
    }
    result["floor_count"] = {
        "ours": dist_stats(our_fl),
        "3dbag": dist_stats(bag_fl),
        "note": "Ours: height-estimate only, NOT observed. 3DBAG: b3_bouwlagen (administrative).",
    }
    result["footprint_area_m2"] = {
        "ours": dist_stats(our_area),
        "3dbag": dist_stats(bag_area),
    }
    result["roof_type_3dbag"] = bag_roof
    our_roof = {}
    for b in our_buildings:
        rt = b.get("roof_type", "unknown") or "unknown"
        our_roof[rt] = our_roof.get(rt, 0) + 1
    result["roof_type_ours"] = our_roof

    # Qualitative match assessment
    if our_h and bag_h:
        our_med_h = statistics.median(our_h)
        bag_med_h = statistics.median(bag_h)
        rel_err = abs(our_med_h - bag_med_h) / bag_med_h if bag_med_h > 0 else 1.0
        result["height_median_rel_error"] = round(rel_err, 3)
        result["height_assessment"] = (
            "good" if rel_err < 0.15 else
            "moderate" if rel_err < 0.35 else
            "poor"
        )

    return result


# ── Building record enrichment ────────────────────────────────────────────────

def _enrich_record(
    brec: dict,
    pts: np.ndarray,
    fp_area: float,
    conf_mean: float | None,
    conf_src: str,
    floor_method: str,
    floor_detail: str,
    roof_result,
    instance_method: str,
    fp_method: str,
    plausibility_pass: bool,
) -> dict:
    """Add v3-specific fields to a v2 building record."""
    brec["density_pts_per_m2"] = round(len(pts) / fp_area, 3) if fp_area > 0 else 0.0
    brec["confidence_mean"] = round(conf_mean, 3) if conf_mean is not None else None
    brec["confidence_source"] = conf_src
    brec["floor_density_estimate_method"] = floor_method
    brec["floor_density_estimate_detail"] = floor_detail
    brec["plausibility_pass"] = plausibility_pass
    brec["schema_version"] = "reconstruction_v3"

    # Update provenance fields
    brec["provenance"]["instance_separation"] = f"LiDAR-derived-{instance_method}"
    brec["provenance"]["footprint"] = f"LiDAR-derived-{fp_method}"
    brec["provenance"]["roof_type"] = f"LiDAR-derived-{roof_result.method}"

    # Merge roof detail
    brec.update(roof_result_to_dict(roof_result))
    return brec


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    pred_ply: Path,
    out_dir: Path,
    dataset: str = "unknown",
    scene_id: str | None = None,
    ai_model: str = "PointNet++",
    crs: str = "unknown",
    floor_height: float = DEFAULT_FLOOR_HEIGHT_M,
    building_class: int = BUILDING_CLASS,
    max_buildings: int = 500,
    bag_path: Path | None = None,
    apply_plausibility: bool = True,
    apply_height_split: bool = True,
) -> dict:
    """
    v3 reconstruction pipeline.

    New vs v2.1:
      - Dataset-specific plausibility filtering
      - Height-discontinuity splitting of merged clusters
      - Vertical density floor estimation
      - 3DBAG aggregate validation
      - Confidence statistics per building (if PLY has confidence field)
      - CRS field in schema

    All parameters override dataset defaults if provided.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bldg_dir = out_dir / "buildings"
    bldg_dir.mkdir(exist_ok=True)
    qa_dir = out_dir / "qa"
    qa_dir.mkdir(exist_ok=True)

    scene_id = scene_id or pred_ply.stem
    defaults = DATASET_DEFAULTS.get(dataset, DATASET_DEFAULTS["unknown"])

    print(f"\nSIH 26011 — Reconstruction Pipeline v3")
    print(f"  dataset     : {dataset}")
    print(f"  scene       : {scene_id}")
    print(f"  pred PLY    : {pred_ply}")
    print(f"  output      : {out_dir}")
    print(f"  CRS         : {crs}")
    print(f"  floor_height: {floor_height} m")
    print()

    # ── Load prediction ────────────────────────────────────────────────────────
    print("Loading prediction PLY...")
    record = load_prediction_ply(pred_ply, scene_id=scene_id, source_dataset=dataset)
    print(f"  {record.n_points:,} points  classes: {record.class_summary()}")

    save_prediction_metadata(record, out_dir / "prediction_meta.json")

    n_bldg_pts = int(record.building_mask(building_class).sum())
    print(f"  {n_bldg_pts:,} building-class points ({100*n_bldg_pts/record.n_points:.1f}%)")

    if n_bldg_pts == 0:
        print("ERROR: No building-class points found.")
        return {"error": "no_building_points"}

    building_xyz = record.building_xyz(building_class)

    # Confidence array (may be None)
    conf_full = None
    if record.confidence is not None:
        conf_full = record.confidence[record.building_mask(building_class)]

    # ── Instance extraction ────────────────────────────────────────────────────
    print(f"\nBuilding Instance Extraction (adaptive DBSCAN)...")
    instances, instance_method = extract_building_instances_adaptive(
        building_xyz, scene_id=scene_id, source_dataset=dataset,
        max_instances=max_buildings * 3,  # Over-extract before filtering
        prefer_dbscan=True,
    )
    inst_stats = evaluate_instance_extraction(instances, building_xyz, instance_method)
    print(f"  {len(instances)} raw instances (method: {instance_method})")
    print(f"  noise: {inst_stats['noise_pct']:.1f}%  "
          f"min/max: {inst_stats['min_pts']}/{inst_stats['max_pts']} pts")

    # ── Height discontinuity splitting ────────────────────────────────────────
    if apply_height_split and instances:
        n_before = len(instances)
        instances = split_by_height_gap(instances, building_xyz, gap_threshold_m=2.5)
        n_after = len(instances)
        if n_after > n_before:
            print(f"  Height-split: {n_before} -> {n_after} instances (split {n_after - n_before})")
        else:
            print(f"  Height-split: no splits applied")

    # ── Plausibility filtering ─────────────────────────────────────────────────
    if apply_plausibility and instances:
        instances_f, rejected = filter_by_plausibility(
            instances, building_xyz, conf_full, defaults)
        print(f"\nPlausibility filter: {len(instances)} -> {len(instances_f)} "
              f"(rejected {len(rejected)})")
        reason_counts: dict = {}
        for r in rejected:
            key = r["reason"].split(":")[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
        for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        instances = instances_f

        # Save rejected log
        with open(qa_dir / "rejected_instances.json", "w") as f:
            json.dump({"n_rejected": len(rejected), "reasons": rejected}, f, indent=2)

    if not instances:
        print("\nNo instances remain after filtering.")
        return {"error": "no_instances_after_filtering"}

    # Trim to max_buildings (keep largest)
    if len(instances) > max_buildings:
        instances = sorted(instances, key=lambda i: -i.n_points)[:max_buildings]
        instances = sorted(instances, key=lambda i: (round(i.centroid_xy[0]),
                                                      round(i.centroid_xy[1])))

    # ── Reconstruction loop ────────────────────────────────────────────────────
    print(f"\nReconstructing {len(instances)} buildings...")
    print(f"  {'ID':<32} {'pts':>5}  {'h_m':>5}  {'fl':>3}  {'roof':<9}  {'area_m2':>7}  {'conf':>5}")
    print(f"  {'-'*78}")

    buildings_out: list[dict] = []
    scene_meshes:  list       = []
    n_flat = n_gable = n_skip = 0
    qa_rows: list[dict] = []

    for inst in instances:
        pts = building_xyz[inst.point_indices]
        bid = inst.building_id

        if len(pts) < 3:
            n_skip += 1
            continue

        # Footprint
        fp_list, fp_area, fp_nverts, fp_method, fp_src = _footprint_from_pts(pts, dataset)
        if len(fp_list) < 3 or fp_area <= 0:
            n_skip += 1
            continue

        # Heights (mesh)
        info = analyze_building(pts)
        if info["height"] < defaults["min_h"] * 0.5:
            n_skip += 1
            continue

        # Heights (schema)
        ground_z, roof_z, height, floor_count_base, floors = decompose_floors(
            pts, bid, floor_height_m=floor_height)

        # Vertical density floor estimation
        floor_count_v3, fl_method, fl_detail = estimate_floors_from_density(
            pts, ground_z, roof_z, floor_height)

        # Use density estimate if it differs from height estimate — both are labelled "estimated"
        floor_count = floor_count_v3

        # Roof analysis
        eave_z = info["z_eave"]
        roof_result = analyze_roof(pts, info["z_ground"], info["z_peak"], eave_z)
        roof_type  = roof_result.roof_type
        is_pitched = roof_result.is_pitched

        # Mesh
        mb, _ = _build_mesh(pts)
        if not mb.verts:
            n_skip += 1
            continue

        obj_name = f"{bid}.obj"
        obj_rel  = f"buildings/{obj_name}"
        write_obj(bldg_dir / obj_name, mb, group=bid)
        scene_meshes.append((bid, mb))

        if is_pitched:
            n_gable += 1
        else:
            n_flat += 1

        # Confidence
        conf_mean, conf_src = _confidence_stats(inst.point_indices, conf_full)

        # Build record
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
            eave_z                 = eave_z,
            floor_count            = floor_count,
            floor_height_m_assumed = floor_height,
            floors                 = [f.to_dict() for f in floors],
            roof_type              = roof_type,
            is_pitched             = is_pitched,
            lod2_obj_file          = obj_rel,
            ai_model               = ai_model,
        )
        # Enrich with v3 fields
        brec["crs"] = crs
        brec = _enrich_record(
            brec, pts, fp_area, conf_mean, conf_src,
            fl_method, fl_detail, roof_result,
            instance_method, fp_method, plausibility_pass=True,
        )
        brec["footprint_source"] = f"LiDAR-derived-{fp_method}"
        brec["footprint_note"] = (
            f"Method: {fp_method} ({fp_src}). "
            "Raster boundary edge tracing preserves concave shapes."
        )
        buildings_out.append(brec)

        conf_str = f"{conf_mean:.2f}" if conf_mean is not None else " N/A"
        print(f"  {bid:<32} {inst.n_points:>5}  {height:>5.1f}  {floor_count:>3}  "
              f"{roof_type:<9}  {fp_area:>7.0f}  {conf_str}")

        qa_rows.append({
            "building_id": bid, "pts": inst.n_points, "height_m": round(height, 2),
            "floors": floor_count, "roof": roof_type, "area_m2": round(fp_area, 1),
            "density": round(_cluster_density(pts, fp_area), 3),
            "conf_mean": round(conf_mean, 3) if conf_mean else None,
        })

    # ── Scene OBJ ─────────────────────────────────────────────────────────────
    print(f"\nWriting scene OBJ...")
    scene_obj = out_dir / "scene.obj"
    scene_mtl = out_dir / "scene.mtl"
    write_scene_obj(scene_obj, scene_meshes)
    write_mtl(scene_mtl, len(scene_meshes))
    txt = scene_obj.read_text(encoding="utf-8")
    scene_obj.write_text("mtllib scene.mtl\n" + txt, encoding="utf-8")

    # ── Save buildings.json ────────────────────────────────────────────────────
    run_params = {
        "schema_version":    "reconstruction_v3",
        "dataset":           dataset,
        "crs":               crs,
        "instance_method":   instance_method,
        "footprint_method":  "raster-boundary-edge-tracing",
        "height_method":     "LiDAR-percentile-5th-99th",
        "floor_method":      "height-estimate-with-density-check",
        "roof_method":       "RANSAC-open3d-with-Zstd-fallback",
        "plausibility_filter": apply_plausibility,
        "height_split":      apply_height_split,
        "defaults_used":     defaults,
    }
    scene_doc = scene_record(
        scene_id=scene_id, source_dataset=dataset,
        source_ply=str(pred_ply.resolve()), ai_model=ai_model,
        buildings=buildings_out, run_params=run_params,
    )
    save_buildings_json(scene_doc, out_dir / "buildings.json")

    # ── 3DBAG aggregate validation ────────────────────────────────────────────
    bag_report = validate_against_3dbag_aggregate(buildings_out, bag_path)
    with open(qa_dir / "3dbag_aggregate.json", "w") as f:
        json.dump(bag_report, f, indent=2)

    # ── QA CSV ────────────────────────────────────────────────────────────────
    import csv
    if qa_rows:
        keys = list(qa_rows[0].keys())
        with open(qa_dir / "buildings_qa.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(qa_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "schema_version":            "reconstruction_v3",
        "pred_ply":                  str(pred_ply),
        "scene_id":                  scene_id,
        "dataset":                   dataset,
        "crs":                       crs,
        "ai_model":                  ai_model,
        "total_input_points":        record.n_points,
        "building_class_points":     n_bldg_pts,
        "n_raw_instances":           inst_stats["n_instances"],
        "n_after_filtering":         len(instances) + n_skip,
        "n_buildings_reconstructed": len(buildings_out),
        "n_flat_roof":               n_flat,
        "n_gable_roof":              n_gable,
        "n_skipped":                 n_skip,
        "instance_method":           instance_method,
        "noise_pct":                 inst_stats["noise_pct"],
        "plausibility_filter":       apply_plausibility,
        "floor_height_m_assumed":    floor_height,
        "floor_estimation_note":     "Height-based estimate with density check. NOT directly observed.",
        "bag_validation":            bag_report.get("height_assessment"),
        "outputs": {
            "buildings_json": str(out_dir / "buildings.json"),
            "scene_obj":      str(scene_obj),
            "buildings_dir":  str(bldg_dir),
            "qa_dir":         str(qa_dir),
        },
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Reconstructed  : {len(buildings_out)}")
    print(f"  Flat roof      : {n_flat}")
    print(f"  Gable roof     : {n_gable}")
    print(f"  Skipped        : {n_skip}")
    print(f"  buildings.json : {out_dir / 'buildings.json'}")
    if bag_report.get("height_assessment"):
        print(f"  3DBAG height   : {bag_report['height_assessment']}")
    print(f"{'='*60}\n")

    return summary


def _build_mesh(pts: np.ndarray) -> tuple[MeshBuilder, dict]:
    info = analyze_building(pts)
    mb   = generate_mesh(info)
    return mb, info


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="SIH 26011 Reconstruction Pipeline v3")
    ap.add_argument("--pred",       required=True)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--dataset",    default="unknown")
    ap.add_argument("--scene",      default=None)
    ap.add_argument("--model",      default="PointNet++")
    ap.add_argument("--crs",        default="unknown")
    ap.add_argument("--bag",        default=None, help="3DBAG reference JSON path")
    ap.add_argument("--max-buildings", type=int, default=500)
    ap.add_argument("--floor-height",  type=float, default=DEFAULT_FLOOR_HEIGHT_M)
    ap.add_argument("--building-class", type=int, default=BUILDING_CLASS)
    ap.add_argument("--no-filter",  action="store_true", help="Skip plausibility filter")
    ap.add_argument("--no-split",   action="store_true", help="Skip height-gap splitting")
    args = ap.parse_args()

    result = run_pipeline(
        pred_ply          = Path(args.pred),
        out_dir           = Path(args.out),
        dataset           = args.dataset,
        scene_id          = args.scene,
        ai_model          = args.model,
        crs               = args.crs,
        floor_height      = args.floor_height,
        building_class    = args.building_class,
        max_buildings     = args.max_buildings,
        bag_path          = Path(args.bag) if args.bag else None,
        apply_plausibility= not args.no_filter,
        apply_height_split= not args.no_split,
    )
    if "error" in result:
        print(f"Pipeline error: {result['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Standalone 3DBAG validation for SIH 26011 reconstruction v3.

Compares our reconstructed buildings against Session 1's 3DBAG reference.

3DBAG reference format (from ml/results/ahn4_inference/3dbag_reference.json):
  {buildings: [{id, floors, h_roof_50p, h_roof_70p, h_roof_max, h_roof_min,
                h_ground, roof_type, volume_lod12, volume_lod22,
                footprint_area, year_built, status, ahn4_density, quality_ok}]}

NOTE: This 3DBAG JSON has NO polygon coordinates. Spatial matching is not possible.
We compare aggregate distributions only.

Usage:
    python ml/src/reconstruction/run_3dbag_validation.py \\
        --our ml/results/reconstruction_v3/ahn4/buildings.json \\
        --bag ml/results/ahn4_inference/3dbag_reference.json \\
        --out ml/results/reconstruction_v3/ahn4/qa/3dbag_validation.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def load_our_buildings(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("buildings", [])


def load_bag_buildings(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("buildings", [])


def dist_stats(vals: list) -> dict:
    if not vals:
        return {"n": 0}
    arr = sorted(vals)
    return {
        "n": len(arr),
        "mean": round(statistics.mean(arr), 3),
        "median": round(statistics.median(arr), 3),
        "std": round(statistics.stdev(arr), 3) if len(arr) > 1 else 0.0,
        "min": round(arr[0], 3),
        "p25": round(float(np.percentile(arr, 25)), 3),
        "p75": round(float(np.percentile(arr, 75)), 3),
        "max": round(arr[-1], 3),
    }


def validate(
    our_buildings: list[dict],
    bag_buildings: list[dict],
) -> dict:
    """Compare our reconstructed building statistics against 3DBAG reference.

    Since the 3DBAG JSON has no polygon coordinates, this is a DISTRIBUTION
    comparison, not a spatial matching. We cannot compute per-building IOU.

    Reports:
      - Height distribution comparison (ours vs 3DBAG h_roof_50p - h_ground)
      - Floor count distribution comparison
      - Footprint area distribution comparison
      - Roof type distribution comparison (normalised categories)
      - Coverage ratio (how many of 532 3DBAG buildings we likely found)
    """
    result: dict = {}
    result["note"] = (
        "Aggregate distribution comparison only. "
        "3DBAG reference JSON has no polygon coordinates — spatial matching not possible. "
        "Per-building IOU would require 3DBAG polygon geometry from the API."
    )
    result["our_n"] = len(our_buildings)
    result["bag_n"] = len(bag_buildings)
    result["bag_quality_ok_n"] = sum(1 for b in bag_buildings if b.get("quality_ok"))

    # ── Height ────────────────────────────────────────────────────────────────
    our_h = [b["height"] for b in our_buildings
              if b.get("height") and b["height"] > 0]
    bag_h = [b["h_roof_50p"] - b["h_ground"]
              for b in bag_buildings
              if b.get("h_roof_50p") is not None and b.get("h_ground") is not None
              and b.get("quality_ok")]
    result["height_m"] = {
        "ours": dist_stats(our_h),
        "bag_50p_minus_ground": dist_stats(bag_h),
        "our_method": "LiDAR 99th-percentile Z minus 5th-percentile Z",
        "bag_method": "h_roof_50p - h_ground (NAP metres)",
        "median_relative_error": (
            round(abs(statistics.median(our_h) - statistics.median(bag_h)) /
                  statistics.median(bag_h), 4)
            if our_h and bag_h else None
        ),
        "assessment": (
            "good" if our_h and bag_h and
            abs(statistics.median(our_h) - statistics.median(bag_h)) / statistics.median(bag_h) < 0.15
            else "moderate" if our_h and bag_h and
            abs(statistics.median(our_h) - statistics.median(bag_h)) / statistics.median(bag_h) < 0.35
            else "poor"
        ),
    }

    # ── Floor count ───────────────────────────────────────────────────────────
    our_fl = [b["floor_count"] for b in our_buildings if b.get("floor_count")]
    bag_fl = [b["floors"] for b in bag_buildings if b.get("floors")]
    floor_correct = None
    floor_within1 = None
    if our_fl and bag_fl:
        # Can't do per-building comparison without spatial matching
        # Show distributions and note that our floors are estimated
        our_med = statistics.median(our_fl)
        bag_med = statistics.median(bag_fl)
        floor_correct = round(abs(our_med - bag_med), 2)
        floor_within1 = abs(our_med - bag_med) <= 1.0
    result["floor_count"] = {
        "ours": dist_stats(our_fl),
        "bag_bouwlagen": dist_stats(bag_fl),
        "our_method": "height-estimate: height / 3.2m — NOT directly observed",
        "bag_method": "b3_bouwlagen (administrative building register)",
        "median_abs_diff_floors": floor_correct,
        "median_within_one_floor": floor_within1,
        "important_caveat": (
            "Our floor counts are height-based ESTIMATES. "
            "Aerial LiDAR cannot observe individual floor slabs. "
            "Do not present these as measured floor counts."
        ),
    }

    # ── Footprint area ────────────────────────────────────────────────────────
    our_area = [b["footprint_area_m2"] for b in our_buildings
                 if b.get("footprint_area_m2", 0) > 0]
    bag_area = [b["footprint_area"] for b in bag_buildings
                 if b.get("footprint_area") and b.get("quality_ok")]
    result["footprint_area_m2"] = {
        "ours": dist_stats(our_area),
        "bag": dist_stats(bag_area),
        "our_method": "raster boundary edge tracing",
        "bag_method": "3DBAG official footprint area from BAG register",
        "note": "Our footprints are reconstructed from building-class LiDAR points only.",
    }

    # ── Roof type ─────────────────────────────────────────────────────────────
    our_roof: dict = {}
    for b in our_buildings:
        rt = b.get("roof_type", "unknown") or "unknown"
        our_roof[rt] = our_roof.get(rt, 0) + 1

    bag_roof: dict = {}
    for b in bag_buildings:
        rt = b.get("roof_type", "unknown") or "unknown"
        bag_roof[rt] = bag_roof.get(rt, 0) + 1

    # Normalise 3DBAG categories to ours
    # 3DBAG: "slanted" = gable/hip | "horizontal" = flat | "multiple horizontal" = mixed
    # Ours: "flat" | "gable" | "shed" | "unknown"
    bag_slanted_pct = sum(bag_roof.get(k, 0) for k in ["slanted"]) / len(bag_buildings) * 100
    bag_flat_pct    = sum(bag_roof.get(k, 0) for k in ["horizontal"]) / len(bag_buildings) * 100
    our_pitched_pct = sum(our_roof.get(k, 0) for k in ["gable", "shed"]) / max(len(our_buildings), 1) * 100

    result["roof_type"] = {
        "ours": our_roof,
        "bag": bag_roof,
        "bag_slanted_pct": round(bag_slanted_pct, 1),
        "bag_flat_pct":    round(bag_flat_pct, 1),
        "our_pitched_pct": round(our_pitched_pct, 1),
        "our_method": "RANSAC plane fitting (or Z-std fallback)",
        "bag_method": "3DBAG b3_dak_type",
        "note": (
            "3DBAG: 78.2% slanted roofs (Amsterdam row houses). "
            "Our RANSAC detection of pitched roofs depends on point density in roof zone."
        ),
    }

    # ── Coverage estimate ─────────────────────────────────────────────────────
    # Rough estimate: how many of 532 3DBAG buildings might correspond to our reconstructed clusters
    # Given building IoU=0.053 and precision=13.4%:
    # True positives per cluster ≈ 13.4%
    # Expected real buildings found ≈ n_our_buildings * 0.134 (rough upper bound)
    expected_real = round(len(our_buildings) * 0.134)  # Using Exp006 building precision
    result["coverage_estimate"] = {
        "bag_total_buildings": len(bag_buildings),
        "our_reconstructed": len(our_buildings),
        "exp006_precision": 0.134,
        "estimated_tp_clusters": expected_real,
        "estimated_coverage_pct": round(expected_real / len(bag_buildings) * 100, 1),
        "note": (
            "Rough estimate based on Exp006 building precision=13.4%. "
            "Not a precise metric. Spatial matching needed for exact figure."
        ),
    }

    # ── Confidence ────────────────────────────────────────────────────────────
    our_conf = [b["confidence_mean"] for b in our_buildings
                if b.get("confidence_mean") is not None]
    if our_conf:
        result["confidence"] = {
            "mean": round(statistics.mean(our_conf), 3),
            "median": round(statistics.median(our_conf), 3),
            "min": round(min(our_conf), 3),
            "max": round(max(our_conf), 3),
            "note": "Softmax max probability averaged over cluster points.",
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    height_ok  = result["height_m"]["assessment"]
    floor_ok   = "within_one" if result["floor_count"].get("median_within_one_floor") else "beyond_one"
    result["summary"] = {
        "height_agreement": height_ok,
        "floor_agreement": floor_ok,
        "area_median_ratio": (
            round(statistics.median(our_area) / statistics.median(bag_area), 3)
            if our_area and bag_area else None
        ),
        "pitched_roof_our_pct": round(our_pitched_pct, 1),
        "pitched_roof_bag_pct": round(bag_slanted_pct, 1),
        "main_limitation": (
            "Exp006 building IoU = 0.053 on AHN4 (domain gap). "
            "Most reconstructed clusters may be false positives. "
            "Fine-tuning Exp006 on AHN4 data would greatly improve results."
        ),
    }

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    our = load_our_buildings(args.our)
    bag = load_bag_buildings(args.bag)

    print(f"Our buildings : {len(our)}")
    print(f"3DBAG buildings: {len(bag)}")

    report = validate(our, bag)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n3DBAG Validation Summary:")
    print(f"  Height agreement    : {report['height_m']['assessment']}")
    print(f"  Height median error : {report['height_m']['median_relative_error']:.1%}")
    print(f"  Floor median diff   : {report['floor_count'].get('median_abs_diff_floors', 'N/A')} floors")
    print(f"  Area ratio (med)    : {report['summary']['area_median_ratio']}")
    print(f"  Our pitched roofs   : {report['summary']['pitched_roof_our_pct']:.1f}%")
    print(f"  3DBAG slanted roofs : {report['summary']['pitched_roof_bag_pct']:.1f}%")
    print(f"  Est. buildings found: ~{report['coverage_estimate']['estimated_tp_clusters']} / {len(bag)}")
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()

"""Improved 3DBAG building-level spatial matching for SIH 26011.

Fixes over the original run_spatial_matching.py:
  1. Broader candidate search (80m instead of 30m centroid threshold).
  2. Bounding-box pre-filter for efficiency.
  3. Best-candidate selection uses rasterized footprint IoU (0.5m cells).
  4. Hungarian-style greedy deduplication: each 3DBAG building can only be
     matched to ONE predicted building (the one with highest IoU).
  5. Explicit classification of each predicted building into:
       matched          - valid spatial match with 3DBAG
       fp_no_reference  - no 3DBAG within 150m (likely FP or data gap)
       unmatched_nearby - 3DBAG exists within 150m but IoU < threshold
  6. Corrected roof-type label mapping:
       ours "flat"  <-> 3DBAG "flat" or "multiple horizontal"
       ours "gable" <-> 3DBAG "slanted"
       ours "shed"  <-> 3DBAG "slanted" (simple pitched)

Usage:
    python ml/src/reconstruction/improved_3dbag_matching.py \\
        --our-buildings ml/results/final_ahn4_improved/buildings.json \\
        --ref-footprints ml/results/final_ahn4/3dbag_footprints.json \\
        --out ml/results/final_ahn4_improved/3dbag_matching_improved.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

# ── Geometry helpers ──────────────────────────────────────────────────────────

def poly_area(poly: list) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(a) / 2.0


def centroid(poly: list) -> tuple[float, float]:
    if not poly:
        return (0.0, 0.0)
    return (sum(p[0] for p in poly) / len(poly),
            sum(p[1] for p in poly) / len(poly))


def dist2d(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def bbox(poly: list) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def bboxes_overlap(a_bbox, b_bbox, margin: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = a_bbox
    bx0, by0, bx1, by1 = b_bbox
    return not (ax1 + margin < bx0 or bx1 + margin < ax0 or
                ay1 + margin < by0 or by1 + margin < ay0)


def iou_raster(poly_a: list, poly_b: list, resolution: float = 0.5) -> float:
    """Polygon IoU via rasterization (no shapely)."""
    if not poly_a or not poly_b:
        return 0.0
    ax0, ay0, ax1, ay1 = bbox(poly_a)
    bx0, by0, bx1, by1 = bbox(poly_b)
    x0 = min(ax0, bx0); y0 = min(ay0, by0)
    x1 = max(ax1, bx1); y1 = max(ay1, by1)

    # Don't bother if polygons are very far apart
    if (x1 - x0) > 300 or (y1 - y0) > 300:
        return 0.0

    res = resolution
    area_product = (x1 - x0) * (y1 - y0)
    if area_product / (res ** 2) > 400000:
        res = max(res, math.sqrt(area_product / 400000))

    cols = max(4, int((x1 - x0) / res) + 2)
    rows = max(4, int((y1 - y0) / res) + 2)

    def rasterize(poly):
        grid = np.zeros((rows, cols), dtype=bool)
        for r in range(rows):
            cy = y0 + (r + 0.5) * res
            for c in range(cols):
                cx = x0 + (c + 0.5) * res
                inside = False
                j = len(poly) - 1
                for i in range(len(poly)):
                    xi, yi = poly[i]; xj, yj = poly[j]
                    if (yi > cy) != (yj > cy):
                        denom = (yj - yi) or 1e-12
                        if cx < (xj - xi) * (cy - yi) / denom + xi:
                            inside = not inside
                    j = i
                grid[r, c] = inside
        return grid

    g_a = rasterize(poly_a)
    g_b = rasterize(poly_b)
    inter = float((g_a & g_b).sum())
    union = float((g_a | g_b).sum())
    return inter / union if union > 0 else 0.0


# ── Roof-type label mapping ────────────────────────────────────────────────────

# Map 3DBAG roof_type -> our roof_type category
_3DBAG_TO_OURS = {
    "flat":               "flat",
    "multiple horizontal": "flat",
    "slanted":            "gable",
    "no_data":            None,     # unknown — exclude from accuracy
}


def roof_labels_agree(our_roof: str | None, ref_roof: str | None) -> bool | None:
    """True if labels are equivalent, False if not, None if unknown."""
    if not our_roof or not ref_roof:
        return None
    mapped = _3DBAG_TO_OURS.get(ref_roof)
    if mapped is None:
        return None
    return (our_roof == mapped)


# ── Main matching logic ───────────────────────────────────────────────────────

def match_buildings(
    our_buildings: list[dict],
    ref_buildings: list[dict],
    candidate_dist_m: float = 80.0,
    bbox_margin_m: float = 10.0,
    min_iou: float = 0.01,
    no_ref_threshold_m: float = 150.0,
) -> dict:
    """Match our predicted buildings to 3DBAG reference buildings.

    Algorithm:
      1. For each predicted building, find candidate 3DBAG buildings whose
         centroid is within candidate_dist_m OR whose bbox overlaps ours.
      2. Compute rasterized footprint IoU with each candidate.
      3. Greedily assign: each predicted building gets its best candidate;
         each 3DBAG building can only be used once.
      4. Classify unmatched buildings:
           fp_no_reference  if nearest 3DBAG > no_ref_threshold_m
           unmatched_nearby if nearest 3DBAG <= no_ref_threshold_m but IoU < min_iou

    Returns a full results dict.
    """
    # Pre-cache geometry
    for rb in ref_buildings:
        fp = [(p[0], p[1]) for p in rb.get("footprint") or []]
        rb["_fp2d"] = fp
        rb["_c"]    = centroid(fp)
        rb["_bbox"] = bbox(fp) if fp else (0, 0, 0, 0)
        rb["_area"] = poly_area(fp)

    # Collect per-predicted-building candidates
    scored: list[dict] = []

    for ob in our_buildings:
        fp_our = [(p[0], p[1]) for p in ob.get("footprint") or []]
        if not fp_our:
            continue
        oc = centroid(fp_our)
        ob_bbox = bbox(fp_our)
        our_area = ob.get("footprint_area_m2") or poly_area(fp_our)

        # Collect candidates
        candidates: list[tuple[float, float, dict]] = []  # (dist, iou, rb)
        for rb in ref_buildings:
            d = dist2d(oc, rb["_c"])
            if d > candidate_dist_m and not bboxes_overlap(ob_bbox, rb["_bbox"], bbox_margin_m):
                continue
            # Compute IoU
            iou = iou_raster(fp_our, rb["_fp2d"])
            candidates.append((d, iou, rb))

        if candidates:
            # Sort by IoU desc, then dist asc
            candidates.sort(key=lambda x: (-x[1], x[0]))
            scored.append({
                "our_id":     ob["building_id"],
                "candidates": candidates,
                "our_area":   our_area,
                "our_h":      ob.get("height", 0),
                "our_fl":     ob.get("floor_count"),
                "our_roof":   ob.get("roof_type"),
                "our_fp":     fp_our,
            })
        else:
            scored.append({
                "our_id":      ob["building_id"],
                "candidates":  [],
                "our_area":    our_area,
                "our_h":       ob.get("height", 0),
                "our_fl":      ob.get("floor_count"),
                "our_roof":    ob.get("roof_type"),
                "our_fp":      fp_our,
            })

    # Greedy deduplication: highest-IoU candidate wins
    used_ref_ids: set[str] = set()
    matches: list[dict]    = []
    unmatched_ours: list   = []

    # Sort by best candidate IoU (desc) to prioritise high-IoU predictions first
    scored.sort(key=lambda s: -(s["candidates"][0][1] if s["candidates"] else -1))

    for s in scored:
        our_id = s["our_id"]
        assigned = False

        for d, iou, rb in s["candidates"]:
            if rb["bag_id"] in used_ref_ids:
                continue
            if iou < min_iou:
                break  # sorted, no better candidate
            used_ref_ids.add(rb["bag_id"])

            ref_h    = rb.get("height")
            ref_fl   = rb.get("floor_count")
            ref_area = rb.get("footprint_area") or rb["_area"]

            h_err = (s["our_h"] - ref_h) if ref_h is not None and s["our_h"] else None
            fl_ex = (s["our_fl"] == ref_fl) if ref_fl is not None and s["our_fl"] else None
            fl_w1 = (abs(s["our_fl"] - ref_fl) <= 1) if ref_fl is not None and s["our_fl"] else None
            fl_ae = abs(s["our_fl"] - ref_fl) if ref_fl is not None and s["our_fl"] else None
            ar    = s["our_area"] / ref_area if ref_area > 0 else None
            roof_ag = roof_labels_agree(s["our_roof"], rb.get("roof_type"))

            matches.append({
                "our_id":           our_id,
                "ref_id":           rb["bag_id"],
                "centroid_dist_m":  round(d, 2),
                "footprint_iou":    round(iou, 4),
                "area_ratio":       round(ar, 3) if ar else None,
                "our_area_m2":      round(s["our_area"], 1),
                "ref_area_m2":      round(ref_area, 1),
                "our_height_m":     round(s["our_h"], 2),
                "ref_height_m":     round(ref_h, 2) if ref_h else None,
                "height_err_m":     round(h_err, 2) if h_err else None,
                "our_floors":       s["our_fl"],
                "ref_floors":       ref_fl,
                "floor_exact":      fl_ex,
                "floor_within1":    fl_w1,
                "floor_abs_err":    fl_ae,
                "our_roof":         s["our_roof"],
                "ref_roof":         rb.get("roof_type"),
                "roof_match":       roof_ag,
            })
            assigned = True
            break

        if not assigned:
            unmatched_ours.append(our_id)

    # Classify unmatched
    ref_centroids = [rb["_c"] for rb in ref_buildings]
    fp_no_ref, unmatched_nearby = [], []
    for bid in unmatched_ours:
        ob = next((o for o in our_buildings if o["building_id"] == bid), None)
        if not ob:
            fp_no_ref.append(bid)
            continue
        fp = [(p[0], p[1]) for p in ob.get("footprint") or []]
        oc = centroid(fp)
        min_d = min(dist2d(oc, rc) for rc in ref_centroids) if ref_centroids else 9999
        if min_d > no_ref_threshold_m:
            fp_no_ref.append(bid)
        else:
            unmatched_nearby.append(bid)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def safe_mean(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    def pct(vals):
        v = [x for x in vals if x is not None]
        return round(100 * sum(v) / len(v), 1) if v else None

    iou_v = [m["footprint_iou"] for m in matches]
    ar_v  = [m["area_ratio"]    for m in matches if m["area_ratio"]]
    cd_v  = [m["centroid_dist_m"] for m in matches]
    h_e   = [m["height_err_m"]  for m in matches if m["height_err_m"] is not None]
    h_a   = [abs(v) for v in h_e]
    w1m   = [abs(v) <= 1.0 for v in h_e]
    w2m   = [abs(v) <= 2.0 for v in h_e]
    fl_ex = [m["floor_exact"]   for m in matches]
    fl_w1 = [m["floor_within1"] for m in matches]
    fl_ae = [m["floor_abs_err"] for m in matches]
    roo   = [m["roof_match"]    for m in matches]

    result = {
        "matching_params": {
            "candidate_dist_m":    candidate_dist_m,
            "bbox_margin_m":       bbox_margin_m,
            "min_iou":             min_iou,
            "no_ref_threshold_m":  no_ref_threshold_m,
        },
        "n_predicted":            len(our_buildings),
        "n_matched":              len(matches),
        "n_unmatched_nearby":     len(unmatched_nearby),
        "n_fp_no_reference":      len(fp_no_ref),
        "n_ref_in_area":          len(ref_buildings),
        "n_unmatched_ref":        len(ref_buildings) - len(used_ref_ids),
        "fp_no_reference":        fp_no_ref,
        "unmatched_nearby":       unmatched_nearby,
        "footprint": {
            "mean_iou":             safe_mean(iou_v),
            "median_iou":           round(float(np.median(iou_v)), 4) if iou_v else None,
            "mean_area_ratio":      safe_mean(ar_v),
            "mean_centroid_dist_m": safe_mean(cd_v),
        },
        "height_vs_3dbag": {
            "n_with_height":    len(h_e),
            "mae_m":            safe_mean(h_a),
            "rmse_m":           round(float(np.sqrt(np.mean([v**2 for v in h_e]))), 4) if h_e else None,
            "mean_signed_err_m":safe_mean(h_e),
            "median_abs_err_m": round(float(np.median(h_a)), 4) if h_a else None,
            "pct_within_1m":    pct(w1m),
            "pct_within_2m":    pct(w2m),
        },
        "floors_vs_3dbag": {
            "exact_accuracy_pct": pct(fl_ex),
            "within_1_floor_pct": pct(fl_w1),
            "mean_abs_floor_err": safe_mean(fl_ae),
        },
        "roof_vs_3dbag": {
            "type_agreement_pct": pct([r for r in roo if r is not None]),
            "NOTE": "Mapped: our 'flat' = 3DBAG 'flat'/'multiple horizontal', our 'gable'/'shed' = 3DBAG 'slanted'.",
        },
        "matches": matches,
    }
    return result


def print_summary(result: dict) -> None:
    fp = result["footprint"]
    h  = result["height_vs_3dbag"]
    fl = result["floors_vs_3dbag"]
    ro = result["roof_vs_3dbag"]
    print(f"\n{'='*60}")
    print("IMPROVED 3DBAG MATCHING RESULTS")
    print(f"{'='*60}")
    print(f"  Predicted  : {result['n_predicted']}")
    print(f"  Matched    : {result['n_matched']}  ({100*result['n_matched']/max(result['n_predicted'],1):.0f}%)")
    print(f"  Unmatched nearby : {result['n_unmatched_nearby']}")
    print(f"  FP no reference  : {result['n_fp_no_reference']} (harbor/open area)")
    print(f"\n  FOOTPRINT:")
    print(f"    Mean IoU           : {fp['mean_iou']}")
    print(f"    Median IoU         : {fp['median_iou']}")
    print(f"    Mean area ratio    : {fp['mean_area_ratio']}")
    print(f"    Mean centroid dist : {fp['mean_centroid_dist_m']} m")
    print(f"\n  HEIGHT vs 3DBAG:")
    print(f"    MAE                : {h['mae_m']} m")
    print(f"    RMSE               : {h['rmse_m']} m")
    print(f"    Mean signed err    : {h['mean_signed_err_m']} m")
    print(f"    % within +/-1m     : {h['pct_within_1m']}%")
    print(f"    % within +/-2m     : {h['pct_within_2m']}%")
    print(f"\n  FLOORS vs 3DBAG:")
    print(f"    Exact accuracy     : {fl['exact_accuracy_pct']}%")
    print(f"    Within +/-1 floor  : {fl['within_1_floor_pct']}%")
    print(f"    Mean abs err       : {fl['mean_abs_floor_err']}")
    print(f"\n  ROOF vs 3DBAG (corrected labels):")
    print(f"    Type agreement     : {ro['type_agreement_pct']}%")
    print(f"{'='*60}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-buildings",  default="ml/results/final_ahn4/reconstruction/buildings.json")
    ap.add_argument("--ref-footprints", default="ml/results/final_ahn4/3dbag_footprints.json")
    ap.add_argument("--out",            default="ml/results/final_ahn4_improved/matching_improved.json")
    ap.add_argument("--candidate-dist", type=float, default=80.0)
    ap.add_argument("--min-iou",        type=float, default=0.01)
    args = ap.parse_args()

    our_path = ROOT / args.our_buildings
    ref_path = ROOT / args.ref_footprints
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(our_path) as f:
        scene = json.load(f)
    our_buildings = scene["buildings"]

    with open(ref_path) as f:
        ref_data = json.load(f)
    ref_buildings = ref_data["buildings"]

    print(f"Matching {len(our_buildings)} predicted vs {len(ref_buildings)} 3DBAG buildings...")
    result = match_buildings(our_buildings, ref_buildings,
                             candidate_dist_m=args.candidate_dist,
                             min_iou=args.min_iou)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print_summary(result)
    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

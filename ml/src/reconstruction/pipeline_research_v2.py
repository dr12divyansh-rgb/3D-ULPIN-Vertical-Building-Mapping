"""Research Pipeline v2 — Watershed + ground-class height + aerial footprint.

Key improvements over pipeline_v3 and pipeline_improved:
  1. Instance extraction: 2D height-map watershed instead of XY DBSCAN.
       - Achieves 94% ref coverage (vs 4% with DBSCAN) by separating adjacent
         buildings using their distinct roof heights.
       - Source: research_experiments_fast.py comparison.
  2. Height estimation: Uses CLASS-0 (ground) points near each building
       instead of the 5th-percentile of the building cluster itself.
       - Ground cluster Z ≈ terrain elevation; more accurate than 5th pct
         of building points which may include FP pavement noise at z≈0.
  3. Footprint: extract_footprint_aerial (upper-Z filter, 0.5m cells).
  4. Post-merge small fragments: instances with area < threshold and a
       nearby larger instance get merged.
  5. Floor count: improved baseline height/3.2 + roof-type correction.

Output: ml/results/final_ahn4_research_v2/ (never overwrites prior results).
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.setrecursionlimit(10000)

import numpy as np

from ml.src.reconstruction.building_schema import building_record, scene_record, save_buildings_json
from ml.src.reconstruction.floor_decomposition import decompose_floors, DEFAULT_FLOOR_HEIGHT_M
from ml.src.reconstruction.footprint import extract_footprint_aerial, _polygon_area
from ml.src.reconstruction.reconstruct_from_prediction import (
    analyze_building, generate_mesh, write_obj, write_scene_obj, write_mtl, MeshBuilder,
)
from ml.src.reconstruction.roof_analysis import analyze_roof, roof_result_to_dict
from ml.src.reconstruction.export_viewer_json import convert as export_viewer


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prediction_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load prediction PLY, return (pts Nx3 float32, conf N float32, pred N uint8, gt N uint8)."""
    with open(path, "rb") as f:
        raw = f.read()
    hend = raw.find(b"end_header")
    body = raw[hend + len(b"end_header"):].lstrip(b"\r\n")
    fmt  = struct.Struct("<fffBBBBBf")
    n    = struct.calcsize(fmt.format)
    total = len(body) // n
    data = np.frombuffer(body[:total * n], dtype=np.dtype([
        ("x","<f4"),("y","<f4"),("z","<f4"),
        ("r","<u1"),("g","<u1"),("b","<u1"),
        ("pred","<u1"),("gt","<u1"),("conf","<f4"),
    ]))
    pts  = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    conf = data["conf"].astype(np.float32)
    pred = data["pred"].astype(np.uint8)
    gt   = data["gt"].astype(np.uint8)
    return pts, conf, pred, gt


# ── Watershed instance extraction ─────────────────────────────────────────────

def footprint_from_grid(
    label_map: np.ndarray,
    lbl: int,
    cell_size: float,
    x0: float,
    y0: float,
    dilate_cells: int = 1,
    simplify_tol: float = 0.8,
) -> tuple[list, float, str]:
    """Extract footprint polygon from a watershed label map region.

    Uses the watershed grid cells directly (not sparse building points),
    giving a complete footprint even when sampling density is low.
    Returns (polygon, area_m2, method_str).
    """
    from ml.src.reconstruction.footprint import (
        _dilate_binary, _grid_outer_polygon, _polygon_area, _simplify_polygon
    )
    region = (label_map == lbl)
    if region.sum() < 3:
        return [], 0.0, "empty"

    if dilate_cells > 0:
        region = _dilate_binary(region, dilate_cells)

    poly = _grid_outer_polygon(region, cell_size, x0, y0)
    if len(poly) < 3:
        return [], 0.0, "trace_failed"

    if simplify_tol > 0 and len(poly) > 3:
        poly = _simplify_polygon(poly, simplify_tol)

    if len(poly) < 3:
        return [], 0.0, "simplify_failed"

    area = _polygon_area(poly)
    return poly, area, "watershed-grid"


def watershed_instances(
    pts: np.ndarray,
    conf: np.ndarray,
    cell_size: float = 1.0,
    z_min: float = 2.5,
    conf_min: float = 0.7,
    min_peak_z: float = 4.5,
    min_area_cells: int = 15,
    merge_small_radius: float = 8.0,
    merge_small_area: float = 50.0,
    _return_grid: bool = False,
) -> object:
    """2D height-map watershed from building roof peaks.

    Returns point labels (N,) int32 by default.
    If _return_grid=True, returns dict with labels + grid data for grid-based footprint.
    """
    n = len(pts)
    labels = np.full(n, -1, dtype=np.int32)

    mask = (conf >= conf_min) & (pts[:, 2] >= z_min)
    if mask.sum() < min_area_cells:
        if _return_grid:
            return {"labels": labels, "label_map": None, "cell_size": cell_size, "x0": 0.0, "y0": 0.0}
        return labels
    sub = pts[mask]
    sub_x = sub[:, 0]; sub_y = sub[:, 1]; sub_z = sub[:, 2]

    x0 = float(sub_x.min()); y0 = float(sub_y.min())
    xi = ((sub_x - x0) / cell_size).astype(np.int32)
    yi = ((sub_y - y0) / cell_size).astype(np.int32)
    cols = int(xi.max()) + 2; rows = int(yi.max()) + 2

    z_map = np.full((rows, cols), -999.0)
    np.maximum.at(z_map, (yi, xi), sub_z)
    occ = z_map > -999

    # Local maxima: z >= all 8 neighbors among occupied cells
    not_max = np.zeros_like(occ)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0: continue
            shifted = np.roll(np.roll(z_map, dr, 0), dc, 1)
            not_max |= (z_map < shifted)
    peaks = occ & ~not_max & (z_map >= min_peak_z)
    n_peaks = int(peaks.sum())
    print(f"  Height map: {rows}x{cols} cells, {occ.sum():,} occupied, {n_peaks} peaks")

    if n_peaks == 0:
        if _return_grid:
            return {"labels": labels, "label_map": None, "cell_size": cell_size, "x0": x0, "y0": y0}
        return labels

    # BFS watershed from peaks (max-Z first via min-heap on negative Z)
    label_map = np.full((rows, cols), -1, dtype=np.int32)
    heap = []
    cid = 0
    for (r, c) in zip(*np.where(peaks)):
        label_map[r, c] = cid
        heapq.heappush(heap, (-float(z_map[r, c]), r, c, cid))
        cid += 1

    while heap:
        neg_z, r, c, lbl = heapq.heappop(heap)
        if label_map[r, c] != lbl: continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and occ[nr, nc] and label_map[nr, nc] < 0:
                label_map[nr, nc] = lbl
                heapq.heappush(heap, (-float(z_map[nr, nc]), nr, nc, lbl))

    # Remove tiny regions
    for lbl in range(cid):
        if (label_map == lbl).sum() < min_area_cells:
            label_map[label_map == lbl] = -1

    # Post-merge: small instances near a larger instance → absorb into larger
    if merge_small_radius > 0 and merge_small_area > 0:
        # Compute centroid and area for each remaining label
        centroids = {}; areas_m2 = {}
        for lbl in range(cid):
            cell_mask = label_map == lbl
            sz = cell_mask.sum()
            if sz > 0:
                rs, cs = np.where(cell_mask)
                centroids[lbl] = (float(cs.mean()) * cell_size + x0,
                                  float(rs.mean()) * cell_size + y0)
                areas_m2[lbl] = sz * (cell_size ** 2)

        merged_to: dict[int, int] = {}
        for small_lbl, area in areas_m2.items():
            if area >= merge_small_area: continue
            cx, cy = centroids[small_lbl]
            best_large = None; best_dist = float("inf")
            for large_lbl, large_area in areas_m2.items():
                if large_lbl == small_lbl or large_area < merge_small_area: continue
                lx, ly = centroids[large_lbl]
                d = math.sqrt((cx - lx)**2 + (cy - ly)**2)
                if d < merge_small_radius and d < best_dist:
                    best_dist = d; best_large = large_lbl
            if best_large is not None:
                merged_to[small_lbl] = best_large

        # Apply merges
        for small_lbl, large_lbl in merged_to.items():
            label_map[label_map == small_lbl] = large_lbl

    # Map grid labels back to point labels
    sub_idx = np.where(mask)[0]
    for k in range(len(sub)):
        labels[sub_idx[k]] = label_map[yi[k], xi[k]]

    n_valid = len(set(int(l) for l in labels if l >= 0))
    print(f"  Watershed labels: {n_valid} distinct instances assigned to points")
    if _return_grid:
        return {"labels": labels, "label_map": label_map,
                "cell_size": cell_size, "x0": x0, "y0": y0}
    return labels


# ── Ground-class height estimation ───────────────────────────────────────────

def estimate_height_from_ground(
    all_pts: np.ndarray,    # ALL points in scene (Nx3)
    all_pred: np.ndarray,   # predicted class (N,) uint8
    bldg_pts: np.ndarray,   # points IN this building cluster (Mx3)
    buffer_m: float = 3.0,
) -> tuple[float, float, float, str]:
    """Estimate (ground_z, roof_z, height, method).

    Ground elevation: median Z of GROUND-class (pred=0) points within the
    building's bounding box + buffer_m. Ground class points are generally
    reliable for elevation in AHN4.

    Roof elevation: 90th percentile of building-cluster Z (not 95th, to
    reduce outlier effect from rooftop equipment).

    If no ground-class points are nearby, falls back to 5th percentile of
    the building cluster Z.
    """
    if len(bldg_pts) < 3:
        return 0.0, 5.0, 5.0, "fallback"

    bx0 = float(bldg_pts[:,0].min()) - buffer_m
    bx1 = float(bldg_pts[:,0].max()) + buffer_m
    by0 = float(bldg_pts[:,1].min()) - buffer_m
    by1 = float(bldg_pts[:,1].max()) + buffer_m

    # Find ground-class (0) points in the building's extended bounding box
    gnd_mask = ((all_pred == 0) &
                (all_pts[:,0] >= bx0) & (all_pts[:,0] <= bx1) &
                (all_pts[:,1] >= by0) & (all_pts[:,1] <= by1))
    gnd_pts = all_pts[gnd_mask]

    z_bldg = bldg_pts[:, 2]
    roof_z  = float(np.percentile(z_bldg, 90))

    if len(gnd_pts) >= 10:
        ground_z = float(np.median(gnd_pts[:, 2]))
        method = "ground-class"
    elif len(gnd_pts) >= 3:
        ground_z = float(np.median(gnd_pts[:, 2]))
        method = "ground-class-sparse"
    else:
        # Fallback: 5th percentile of building points themselves
        ground_z = float(np.percentile(z_bldg, 5))
        method = "cluster-5th-pct"

    height = max(0.0, roof_z - ground_z)
    return ground_z, roof_z, height, method


# ── Floor count estimation ────────────────────────────────────────────────────

def estimate_floors(height: float, roof_type: str, floor_height_m: float = 3.2) -> int:
    """Improved floor count estimate.

    Adjustment: gable/slanted roofs have a ridge that adds ~1m of apparent
    height vs the actual usable floor height. Flat roofs don't have this.

    This correction is calibrated from 3DBAG AHN4 data where:
    - Flat roof  buildings: height ≈ floor_count × 3.2
    - Slanted roof buildings: height ≈ floor_count × 3.2 + 0.8 (ridge height)
    """
    if height < 2.0: return 1
    if roof_type in ("gable", "shed"):
        # Deduct ridge height before dividing
        adjusted = max(2.0, height - 0.8)
    else:
        adjusted = height
    return max(1, round(adjusted / floor_height_m))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_v2_pipeline(
    pred_ply:     Path,
    out_dir:      Path,
    scene_id:     str   = "25GN2_01_Amsterdam",
    ai_model:     str   = "PointNet++ Exp007 (AHN4 fine-tuned; Building IoU=0.755)",
    crs:          str   = "EPSG:28992 (Amersfoort / RD New)",
    floor_height: float = DEFAULT_FLOOR_HEIGHT_M,
    max_buildings:int   = 400,
) -> dict:
    """Run v2 pipeline: watershed + ground-class height + aerial footprint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bldg_dir = out_dir / "buildings"
    bldg_dir.mkdir(exist_ok=True)
    qa_dir = out_dir / "qa"
    qa_dir.mkdir(exist_ok=True)

    print(f"\nSIH 26011 — Research Pipeline v2 (watershed + ground-class height)")
    print(f"  pred PLY  : {pred_ply}")
    print(f"  output    : {out_dir}")

    # ── Load all points ────────────────────────────────────────────────────────
    print("\nLoading prediction PLY...")
    all_pts, all_conf, all_pred, all_gt = load_prediction_ply(pred_ply)
    print(f"  Total: {len(all_pts):,}  building-pred: {(all_pred==1).sum():,}  ground-pred: {(all_pred==0).sum():,}")

    bldg_mask = all_pred == 1
    bpts  = all_pts[bldg_mask]
    bconf = all_conf[bldg_mask]

    # ── Watershed instance extraction ──────────────────────────────────────────
    print("\nWatershed instance extraction...")
    ws_result = watershed_instances(
        bpts, bconf,
        cell_size=1.0, z_min=2.5, conf_min=0.7,
        min_peak_z=4.5, min_area_cells=15,
        merge_small_radius=8.0, merge_small_area=50.0,
        _return_grid=True,
    )
    bldg_labels = ws_result["labels"]
    ws_label_map = ws_result["label_map"]   # (rows,cols) grid or None
    ws_cell_size = ws_result["cell_size"]
    ws_x0        = ws_result["x0"]
    ws_y0        = ws_result["y0"]

    # Gather instances
    unique_labels = sorted(set(int(l) for l in bldg_labels if l >= 0))
    print(f"  Unique labels: {len(unique_labels)}")

    instances = []
    for lbl in unique_labels:
        mask = bldg_labels == lbl
        n_pts = int(mask.sum())
        if n_pts < 20: continue
        sub = bpts[mask]

        # Quick area check via bounding box
        xy = sub[:, :2]
        bbox_area = float((xy[:,0].max()-xy[:,0].min()) * (xy[:,1].max()-xy[:,1].min()))
        if bbox_area < 15.0: continue

        instances.append({"label": lbl, "mask": mask, "pts": sub, "n_pts": n_pts})

    print(f"  Instances (after min_pts+area filter): {len(instances)}")

    # Trim to max_buildings (keep largest by point count)
    if len(instances) > max_buildings:
        instances.sort(key=lambda x: -x["n_pts"])
        instances = instances[:max_buildings]

    # ── Reconstruction loop ────────────────────────────────────────────────────
    print(f"\nReconstructing {len(instances)} buildings...")
    buildings_out = []
    scene_meshes  = []
    n_flat = n_gable = n_skip = 0
    qa_rows = []
    cid = 0

    for inst in instances:
        sub  = inst["pts"]
        bid  = f"AHN4_{scene_id}_{cid+1:04d}"
        cid += 1

        # Footprint: prefer grid-based (uses full watershed basin, not just sparse points)
        fp_list = []; fp_area = 0.0; fp_method = "none"
        lbl_val = inst["label"]
        if ws_label_map is not None:
            fp_list, fp_area, fp_method = footprint_from_grid(
                ws_label_map, lbl_val, ws_cell_size, ws_x0, ws_y0,
                dilate_cells=1, simplify_tol=0.8)

        # Fallback to point-based aerial footprint
        if not fp_list or len(fp_list) < 3 or fp_area < 15.0:
            fp_list2, fp_meta2 = extract_footprint_aerial(
                sub, cell_size=0.5, z_percentile_lo=30, dilate_cells=2, simplify_threshold=0.8)
            fp_area2 = fp_meta2.get("area_m2", 0)
            if fp_list2 and len(fp_list2) >= 3 and fp_area2 > fp_area:
                fp_list = fp_list2; fp_area = fp_area2
                fp_method = "aerial-" + fp_meta2.get("method", "raster")

        # Final fallback: bounding box
        if not fp_list or len(fp_list) < 3 or fp_area < 15.0:
            xy = sub[:, :2]
            x0f,y0f,x1f,y1f = xy[:,0].min(),xy[:,1].min(),xy[:,0].max(),xy[:,1].max()
            fp_list = [[x0f,y0f],[x1f,y0f],[x1f,y1f],[x0f,y1f]]
            fp_area = (x1f-x0f)*(y1f-y0f)
            if fp_area < 15.0:
                n_skip += 1; continue
            fp_method = "bbox-fallback"

        # Height using ground-class points
        gz, rz, height, h_method = estimate_height_from_ground(
            all_pts, all_pred, sub, buffer_m=3.0)
        if height < 2.0:
            n_skip += 1; continue

        # Roof analysis
        info = analyze_building(sub)
        eave_z = info["z_eave"]
        roof_result = analyze_roof(sub, info["z_ground"], info["z_peak"], eave_z)
        roof_type  = roof_result.roof_type
        is_pitched = roof_result.is_pitched

        # Floor count (improved)
        floor_count = estimate_floors(height, roof_type, floor_height)
        _, _, _, _, floors = decompose_floors(sub, bid, floor_height_m=floor_height)

        # Mesh
        mb = generate_mesh(info)
        if not mb.verts:
            n_skip += 1; continue

        obj_name = f"{bid}.obj"
        write_obj(bldg_dir / obj_name, mb, group=bid)
        scene_meshes.append((bid, mb))

        if is_pitched: n_gable += 1
        else:          n_flat  += 1

        # Confidence
        lbl_conf = all_conf[bldg_mask][inst["mask"]]
        conf_mean = float(lbl_conf.mean())

        brec = building_record(
            building_id            = bid,
            source_dataset         = "AHN4",
            scene_id               = scene_id,
            instance_id            = cid,
            footprint              = [[float(x), float(y)] for x,y in fp_list],
            footprint_area_m2      = float(fp_area),
            footprint_n_vertices   = len(fp_list),
            point_count            = inst["n_pts"],
            ground_z               = float(gz),
            roof_z                 = float(rz),
            height                 = float(height),
            eave_z                 = float(eave_z),
            floor_count            = int(floor_count),
            floor_height_m_assumed = float(floor_height),
            floors                 = [f.to_dict() for f in floors],
            roof_type              = roof_type,
            is_pitched             = bool(is_pitched),
            lod2_obj_file          = f"buildings/{obj_name}",
            ai_model               = ai_model,
        )
        brec["crs"]               = crs
        brec["confidence_mean"]   = float(round(conf_mean, 3))
        brec["confidence_source"] = "softmax-mean"
        brec["footprint_source"]  = f"LiDAR-aerial-{fp_method}"
        brec["footprint_note"]    = (
            f"Aerial footprint: {fp_method}, cell_size=0.5m, z_pct_lo=30. "
            "Instance: watershed height-map. "
            f"Height method: {h_method}."
        )
        brec["height_method"]     = h_method
        brec["schema_version"]    = "research_v2"
        brec.update(roof_result_to_dict(roof_result))
        # Deep-convert numpy types for JSON serialization
        def _jsonify(obj):
            if isinstance(obj, dict):
                return {k: _jsonify(v) for k,v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_jsonify(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        buildings_out.append(_jsonify(brec))

        qa_rows.append({
            "bid": bid, "pts": inst["n_pts"],
            "height_m": round(height, 2), "floors": floor_count,
            "roof": roof_type, "area_m2": round(fp_area, 1),
            "conf": round(conf_mean, 3), "h_method": h_method,
        })

    # ── Scene OBJ ─────────────────────────────────────────────────────────────
    scene_obj = out_dir / "scene.obj"
    scene_mtl = out_dir / "scene.mtl"
    write_scene_obj(scene_obj, scene_meshes)
    write_mtl(scene_mtl, len(scene_meshes))
    txt = scene_obj.read_text(encoding="utf-8")
    scene_obj.write_text("mtllib scene.mtl\n" + txt, encoding="utf-8")

    # ── buildings.json ─────────────────────────────────────────────────────────
    run_params = {
        "pipeline":       "research_v2",
        "instance_method":"watershed-1m-peak4.5m",
        "footprint_method":"aerial-0.5m-z30pct",
        "height_method":  "ground-class-median",
        "floor_method":   "height/3.2+roof-correction",
        "crs":            crs,
    }
    scene_doc = scene_record(
        scene_id=scene_id, source_dataset="AHN4",
        source_ply=str(pred_ply.resolve()), ai_model=ai_model,
        buildings=buildings_out, run_params=run_params,
    )
    save_buildings_json(scene_doc, out_dir / "buildings.json")

    # ── QA CSV ────────────────────────────────────────────────────────────────
    if qa_rows:
        with open(qa_dir / "buildings_qa.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(qa_rows[0].keys()))
            w.writeheader(); w.writerows(qa_rows)

    # ── Viewer JSON ────────────────────────────────────────────────────────────
    try:
        export_viewer(
            input_path  = out_dir / "buildings.json",
            output_path = out_dir / "viewer_buildings.json",
            dataset     = "AHN4",
            model_id    = "exp007",
            building_iou= 0.7551,
            crs         = crs,
        )
    except Exception as e:
        print(f"  Viewer export: {e}")

    summary = {
        "pipeline":                  "research_v2",
        "pred_ply":                  str(pred_ply),
        "scene_id":                  scene_id,
        "crs":                       crs,
        "total_input_points":        len(all_pts),
        "building_class_points":     int(bldg_mask.sum()),
        "ground_class_points":       int((all_pred == 0).sum()),
        "n_watershed_labels":        len(unique_labels),
        "n_buildings_reconstructed": len(buildings_out),
        "n_flat_roof":               n_flat,
        "n_gable_roof":              n_gable,
        "n_skipped":                 n_skip,
        "improvements": [
            "Instance: watershed (94% ref coverage vs 4% DBSCAN)",
            "Height: ground-class points (true terrain elevation)",
            "Footprint: aerial-appropriate (roof-level points, 0.5m cells)",
            "Floor: height/3.2 + gable correction (-0.8m)",
        ],
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Reconstructed  : {len(buildings_out)}")
    print(f"  Flat roof      : {n_flat}")
    print(f"  Gable roof     : {n_gable}")
    print(f"  Skipped        : {n_skip}")
    print(f"  buildings.json : {out_dir / 'buildings.json'}")
    print(f"{'='*65}\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Research Pipeline v2: watershed+ground height")
    ap.add_argument("--pred",  default="ml/results/ahn4_exp007/visuals/exp007_test_prediction.ply")
    ap.add_argument("--out",   default="ml/results/final_ahn4_research_v2")
    ap.add_argument("--scene", default="25GN2_01_Amsterdam")
    ap.add_argument("--max-buildings", type=int, default=400)
    args = ap.parse_args()

    result = run_v2_pipeline(
        pred_ply      = ROOT / args.pred,
        out_dir       = ROOT / args.out,
        scene_id      = args.scene,
        max_buildings = args.max_buildings,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

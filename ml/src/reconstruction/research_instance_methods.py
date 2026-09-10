"""Research: compare 4 instance extraction methods for AHN4 aerial LiDAR.

Tests:
  A: Current DBSCAN baseline (eps=2m, XY only)
  B: Confidence + Z-filtered DBSCAN (removes FP ground noise)
  C: 2D Height-map watershed (separates adjacent buildings by roof height)
  D: 3D DBSCAN with anisotropic distance (XY+Z, connects within-building vertically)

Evaluation: match rate against 3DBAG buildings in the test area.
Uses the improved IoU-based matching (not centroid-distance).

Output: JSON comparison table + best method recommendation.

DOES NOT modify any existing outputs. Writes only to:
  ml/results/final_ahn4_research_v2/instance_method_comparison.json
"""
from __future__ import annotations

import heapq
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np


# ── Load prediction PLY ───────────────────────────────────────────────────────

def load_prediction_ply(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()
    hend = raw.find(b"end_header")
    body = raw[hend + len(b"end_header"):].lstrip(b"\r\n")
    fmt = struct.Struct("<fffBBBBBf")
    n = struct.calcsize(fmt.format)
    total = len(body) // n
    return np.frombuffer(body[:total * n], dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "<u1"), ("g", "<u1"), ("b", "<u1"),
        ("pred", "<u1"), ("gt", "<u1"), ("conf", "<f4"),
    ]))


# ── Method A: DBSCAN in XY (baseline) ────────────────────────────────────────

def _numpy_dbscan_xy(xy: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Pure-numpy 2D DBSCAN using a grid-accelerated neighbourhood search."""
    n = len(xy)
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cell = eps
    x_min, y_min = float(xy[:, 0].min()), float(xy[:, 1].min())
    ci = ((xy[:, 0] - x_min) / cell).astype(np.int32)
    ri = ((xy[:, 1] - y_min) / cell).astype(np.int32)
    grid: dict[tuple, list[int]] = defaultdict(list)
    for idx in range(n):
        grid[(int(ri[idx]), int(ci[idx]))].append(idx)

    def neighbours(idx: int) -> list[int]:
        rc, cc = int(ri[idx]), int(ci[idx])
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for j in grid.get((rc + dr, cc + dc), []):
                    dx = xy[j, 0] - xy[idx, 0]
                    dy = xy[j, 1] - xy[idx, 1]
                    if dx * dx + dy * dy <= eps * eps:
                        out.append(j)
        return out

    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbours(i)
        if len(nb) < min_pts:
            continue
        labels[i] = cluster_id
        seeds = list(nb)
        si = 0
        while si < len(seeds):
            j = seeds[si]
            si += 1
            if not visited[j]:
                visited[j] = True
                nb2 = neighbours(j)
                if len(nb2) >= min_pts:
                    seeds.extend(nb2)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def method_a_dbscan(pts: np.ndarray, eps: float = 2.0, min_pts: int = 15) -> np.ndarray:
    """XY-only DBSCAN (current baseline)."""
    xy = pts[:, :2]
    return _numpy_dbscan_xy(xy, eps, min_pts)


# ── Method B: Confidence + Z-filtered DBSCAN ─────────────────────────────────

def method_b_filtered_dbscan(
    pts: np.ndarray,
    conf: np.ndarray,
    eps: float = 2.0,
    min_pts: int = 15,
    conf_threshold: float = 0.7,
    z_min: float = 2.0,
) -> np.ndarray:
    """DBSCAN after removing low-confidence + low-Z points.

    Low-Z building points near the ground are likely FP pavement noise.
    Low-confidence predictions are also more likely FP.
    Labels remain in original indexing (unselected points get label -1).
    """
    n = len(pts)
    labels = np.full(n, -1, dtype=np.int32)
    mask = (conf >= conf_threshold) & (pts[:, 2] >= z_min)
    if mask.sum() < min_pts:
        return labels
    sub_pts = pts[mask]
    xy = sub_pts[:, :2]
    sub_labels = _numpy_dbscan_xy(xy, eps, min_pts)
    labels[mask] = sub_labels
    return labels


# ── Method C: 2D Height-Map Watershed ────────────────────────────────────────

def _find_local_maxima_2d(z_map: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Find local maxima in z_map (among occupied cells).

    A cell is a local maximum if its Z >= all 8 neighbors.
    Returns boolean mask.
    """
    rows, cols = z_map.shape
    is_max = np.zeros_like(occ)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            rr = np.clip(np.arange(rows) + dr, 0, rows - 1)
            cc = np.clip(np.arange(cols)[:, None] + dc, 0, cols - 1)
            neighbor = z_map[rr[:, None], cc.T]
            is_max = is_max | (z_map < neighbor)   # any neighbor taller → not max
    return occ & ~is_max


def method_c_watershed(
    pts: np.ndarray,
    conf: np.ndarray,
    cell_size: float = 0.5,
    z_min: float = 2.0,
    conf_min: float = 0.7,
    min_peak_z: float = 3.5,
    min_area_cells: int = 10,
) -> np.ndarray:
    """2D height-map watershed from building roof peaks.

    Strategy:
      1. Build a max-Z grid at cell_size resolution.
      2. Find local maxima (building peaks) among cells with Z > min_peak_z.
      3. Expand from each peak via BFS in decreasing-Z order.
         Adjacent buildings with different roof heights get separate labels.
      4. Map grid labels back to point labels.

    This separates adjacent buildings that have DIFFERENT roof heights —
    the primary failure of XY-only DBSCAN on Amsterdam row houses.
    """
    n = len(pts)
    labels = np.full(n, -1, dtype=np.int32)

    # Filter to high-confidence, above-ground points
    mask = (conf >= conf_min) & (pts[:, 2] >= z_min)
    if mask.sum() < 10:
        return labels

    sub_pts = pts[mask]
    sub_x = sub_pts[:, 0]; sub_y = sub_pts[:, 1]; sub_z = sub_pts[:, 2]

    x0 = float(sub_x.min()); y0 = float(sub_y.min())
    xi = ((sub_x - x0) / cell_size).astype(np.int32)
    yi = ((sub_y - y0) / cell_size).astype(np.int32)
    cols = int(xi.max()) + 2; rows = int(yi.max()) + 2

    z_map = np.full((rows, cols), -999.0)
    np.maximum.at(z_map, (yi, xi), sub_z)
    occ_map = z_map > -999

    # Find local maxima (candidate building peaks)
    is_max = np.copy(occ_map)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            shift_r = np.roll(z_map, dr, axis=0); shift_c = np.roll(z_map, dc, axis=1)
            # If ANY neighbor is strictly taller, not a local max
            is_max &= (z_map >= shift_r) & (z_map >= shift_c)
    peak_mask = is_max & occ_map & (z_map >= min_peak_z)

    # Watershed from peaks using priority queue (max-heap via negation)
    label_map = np.full((rows, cols), -1, dtype=np.int32)
    heap = []  # (-z, row, col, label)
    cluster_id = 0

    peak_positions = np.argwhere(peak_mask)
    for (r, c) in peak_positions:
        label_map[r, c] = cluster_id
        heapq.heappush(heap, (-float(z_map[r, c]), r, c, cluster_id))
        cluster_id += 1

    # Process in decreasing Z order (highest first)
    while heap:
        neg_z, r, c, lbl = heapq.heappop(heap)
        if label_map[r, c] != lbl:   # already claimed by higher peak
            continue
        # Expand to 4-connected neighbors
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and occ_map[nr, nc]:
                if label_map[nr, nc] == -1:
                    label_map[nr, nc] = lbl
                    heapq.heappush(heap, (-float(z_map[nr, nc]), nr, nc, lbl))

    # Filter out tiny regions
    for lbl in range(cluster_id):
        if (label_map == lbl).sum() < min_area_cells:
            label_map[label_map == lbl] = -1

    # Map grid labels back to point labels
    valid = mask
    sub_idx = np.where(valid)[0]
    for i, (r, c) in enumerate(zip(yi, xi)):
        g_lbl = label_map[r, c]
        labels[sub_idx[i]] = g_lbl

    return labels


# ── Method D: 3D Anisotropic DBSCAN ──────────────────────────────────────────

def method_d_3d_anisotropic_dbscan(
    pts: np.ndarray,
    conf: np.ndarray,
    xy_eps: float = 2.0,
    z_eps: float = 1.5,
    min_pts: int = 10,
    conf_min: float = 0.7,
    z_min: float = 2.0,
) -> np.ndarray:
    """3D DBSCAN with separate XY and Z distance thresholds.

    Two points are neighbors iff:
      sqrt(dx²+dy²) <= xy_eps  AND  |dz| <= z_eps

    Adjacent buildings at the SAME Z share XY connections.
    Adjacent buildings at DIFFERENT Z (different floor count) are separated
    because their point clouds don't have continuous Z overlap.

    Note: z_eps=1.5m connects points within 1 floor of each other vertically.
    z_eps=0.5m only connects same-height points (too fragmented for flat roofs).
    """
    n = len(pts)
    labels = np.full(n, -1, dtype=np.int32)
    mask = (conf >= conf_min) & (pts[:, 2] >= z_min)
    if mask.sum() < min_pts:
        return labels

    sub_pts = pts[mask]
    sub_idx = np.where(mask)[0]
    xyz = sub_pts[:, :3]
    nn = len(xyz)

    visited = np.zeros(nn, dtype=bool)
    sub_labels = np.full(nn, -1, dtype=np.int32)

    # Grid-accelerated 3D neighbor search
    cell_xy = xy_eps
    cell_z  = z_eps
    x0, y0, z0 = float(xyz[:,0].min()), float(xyz[:,1].min()), float(xyz[:,2].min())
    ci = ((xyz[:,0]-x0)/cell_xy).astype(np.int32)
    ri = ((xyz[:,1]-y0)/cell_xy).astype(np.int32)
    zi = ((xyz[:,2]-z0)/cell_z ).astype(np.int32)
    grid3d: dict[tuple, list[int]] = defaultdict(list)
    for idx in range(nn):
        grid3d[(int(ri[idx]), int(ci[idx]), int(zi[idx]))].append(idx)

    def neighbours3d(idx: int) -> list[int]:
        rc, cc, zc = int(ri[idx]), int(ci[idx]), int(zi[idx])
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid3d.get((rc+dr, cc+dc, zc+dz), []):
                        dx = xyz[j,0] - xyz[idx,0]
                        dy = xyz[j,1] - xyz[idx,1]
                        dz2 = xyz[j,2] - xyz[idx,2]
                        if (dx*dx+dy*dy <= xy_eps**2 and abs(dz2) <= z_eps):
                            out.append(j)
        return out

    cluster_id = 0
    for i in range(nn):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbours3d(i)
        if len(nb) < min_pts:
            continue
        sub_labels[i] = cluster_id
        seeds = list(nb)
        si = 0
        while si < len(seeds):
            j = seeds[si]; si += 1
            if not visited[j]:
                visited[j] = True
                nb2 = neighbours3d(j)
                if len(nb2) >= min_pts:
                    seeds.extend(nb2)
            if sub_labels[j] == -1:
                sub_labels[j] = cluster_id
        cluster_id += 1

    labels[mask] = sub_labels
    return labels


# ── Evaluation: match rate against 3DBAG ─────────────────────────────────────

def poly_area(poly: list) -> float:
    n = len(poly)
    if n < 3: return 0.0
    a = 0.0
    for i in range(n):
        j = (i+1)%n
        a += poly[i][0]*poly[j][1] - poly[j][0]*poly[i][1]
    return abs(a)/2.0


def poly_centroid(poly: list) -> tuple[float, float]:
    if not poly: return (0.0, 0.0)
    return (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))


def raster_footprint(pts_2d: np.ndarray, cell: float = 0.5) -> tuple[list, float]:
    """Raster boundary footprint from 2D points."""
    if len(pts_2d) < 3:
        return [], 0.0
    x0, y0 = pts_2d[:,0].min(), pts_2d[:,1].min()
    xi = ((pts_2d[:,0]-x0)/cell).astype(np.int32)
    yi = ((pts_2d[:,1]-y0)/cell).astype(np.int32)
    cols = int(xi.max())+2; rows = int(yi.max())+2
    grid = np.zeros((rows,cols),dtype=bool)
    grid[yi, xi] = True
    # Dilate 1 cell
    from ml.src.reconstruction.footprint import _dilate_binary, _grid_outer_polygon, _polygon_area
    grid = _dilate_binary(grid, 1)
    poly = _grid_outer_polygon(grid, cell, x0, y0)
    if len(poly) < 3: return [], 0.0
    area = _polygon_area(poly)
    return poly, area


def iou_raster(poly_a, poly_b, res=0.5):
    if not poly_a or not poly_b: return 0.0
    axs = [p[0] for p in poly_a]; ays = [p[1] for p in poly_a]
    bxs = [p[0] for p in poly_b]; bys = [p[1] for p in poly_b]
    x0 = min(min(axs),min(bxs)); y0 = min(min(ays),min(bys))
    x1 = max(max(axs),max(bxs)); y1 = max(max(ays),max(bys))
    if (x1-x0)>200 or (y1-y0)>200: return 0.0
    r = res
    if (x1-x0)*(y1-y0)/r**2 > 200000:
        r = max(r, math.sqrt((x1-x0)*(y1-y0)/200000))
    cols = max(4,int((x1-x0)/r)+2); rows = max(4,int((y1-y0)/r)+2)
    def rast(poly):
        g = np.zeros((rows,cols),dtype=bool)
        for rr in range(rows):
            cy = y0+(rr+0.5)*r
            for cc in range(cols):
                cx = x0+(cc+0.5)*r
                ins=False; j=len(poly)-1
                for i in range(len(poly)):
                    xi,yi=poly[i]; xj,yj=poly[j]
                    if (yi>cy)!=(yj>cy) and cx<(xj-xi)*(cy-yi)/((yj-yi) or 1e-12)+xi:
                        ins=not ins
                    j=i
                g[rr,cc]=ins
        return g
    ga=rast(poly_a); gb=rast(poly_b)
    inter=float((ga&gb).sum()); union=float((ga|gb).sum())
    return inter/union if union>0 else 0.0


def evaluate_instances(
    pts: np.ndarray,
    labels: np.ndarray,
    ref_buildings: list,
    min_pts: int = 20,
    min_area: float = 15.0,
    candidate_dist: float = 80.0,
    min_iou: float = 0.01,
) -> dict:
    """Evaluate instance labels against 3DBAG buildings.

    Returns metrics: n_instances, n_matched, footprint_iou stats, height stats.
    """
    # Build instances
    unique = [l for l in np.unique(labels) if l >= 0]
    instances = []
    sys.setrecursionlimit(10000)
    for lbl in unique:
        mask = labels == lbl
        if mask.sum() < min_pts:
            continue
        sub = pts[mask]
        # Quick footprint for area check
        xy = sub[:, :2]
        from ml.src.reconstruction.footprint import extract_footprint_aerial
        poly, meta = extract_footprint_aerial(sub, cell_size=0.5, z_percentile_lo=30,
                                               dilate_cells=1, simplify_threshold=0.8)
        area = meta.get("area_m2", 0)
        if area < min_area:
            # Try bounding box
            area = float((xy[:,0].max()-xy[:,0].min())*(xy[:,1].max()-xy[:,1].min()))
            if area < min_area:
                continue
            poly = [[xy[:,0].min(),xy[:,1].min()],[xy[:,0].max(),xy[:,1].min()],
                    [xy[:,0].max(),xy[:,1].max()],[xy[:,0].min(),xy[:,1].max()]]

        z = sub[:, 2]
        gz = float(np.percentile(z, 5))
        rz = float(np.percentile(z, 95))
        h  = max(0.0, rz - gz)
        cx, cy = poly_centroid(poly)

        instances.append({
            "label": int(lbl),
            "n_pts": int(mask.sum()),
            "footprint": poly,
            "area_m2": area,
            "centroid": (cx, cy),
            "ground_z": gz,
            "roof_z": rz,
            "height": h,
        })

    # Precompute ref geometry
    for rb in ref_buildings:
        fp = [(p[0],p[1]) for p in rb.get("footprint",[])]
        rb["_fp"] = fp
        rb["_c"] = poly_centroid(fp)

    # Match instances to 3DBAG (IoU-based greedy)
    used_refs = set()
    matches = []
    for inst in instances:
        cx, cy = inst["centroid"]
        best_iou = -1; best_ref = None
        for rb in ref_buildings:
            rc, ry = rb["_c"]
            d = math.sqrt((cx-rc)**2+(cy-ry)**2)
            if d > candidate_dist: continue
            iou = iou_raster(inst["footprint"], rb["_fp"])
            if iou > best_iou and iou >= min_iou:
                best_iou = iou; best_ref = rb
        if best_ref and best_ref["bag_id"] not in used_refs:
            used_refs.add(best_ref["bag_id"])
            ref_h = best_ref.get("height") or 0
            matches.append({
                "iou": best_iou,
                "h_err": inst["height"] - ref_h if ref_h else None,
                "area_ratio": inst["area_m2"] / (best_ref.get("footprint_area") or 1),
                "ref_fl": best_ref.get("floor_count"),
                "our_fl": max(1, round(inst["height"]/3.2)),
            })

    n_inst = len(instances)
    n_match = len(matches)
    iou_vals = [m["iou"] for m in matches]
    h_errs = [m["h_err"] for m in matches if m["h_err"] is not None]
    h_abs = [abs(v) for v in h_errs]
    fl_exact = [m["our_fl"]==m["ref_fl"] for m in matches if m["ref_fl"]]
    fl_w1 = [abs(m["our_fl"]-m["ref_fl"])<=1 for m in matches if m["ref_fl"]]

    def safe_mean(v): return round(float(np.mean(v)),4) if v else None
    def pct(v): return round(100*sum(v)/len(v),1) if v else None

    return {
        "n_instances":         n_inst,
        "n_matched":           n_match,
        "match_rate_pct":      round(100*n_match/max(n_inst,1),1),
        "n_ref_total":         len(ref_buildings),
        "ref_coverage_pct":    round(100*n_match/len(ref_buildings),1),
        "footprint_mean_iou":  safe_mean(iou_vals),
        "footprint_median_iou":round(float(np.median(iou_vals)),4) if iou_vals else None,
        "height_mae_m":        safe_mean(h_abs),
        "height_bias_m":       safe_mean(h_errs),
        "floor_exact_pct":     pct(fl_exact),
        "floor_within1_pct":   pct(fl_w1),
    }


# ── Main experiment runner ─────────────────────────────────────────────────────

def main():
    ply_path = ROOT / "ml/results/ahn4_exp007/visuals/exp007_test_prediction.ply"
    ref_path = ROOT / "ml/results/final_ahn4/3dbag_footprints.json"
    out_dir  = ROOT / "ml/results/final_ahn4_research_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_prediction_ply(ply_path)
    bldg_mask = data["pred"] == 1
    pts  = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    conf = data["conf"].astype(np.float32)
    bldg_pts  = pts[bldg_mask]
    bldg_conf = conf[bldg_mask]
    print(f"Building points: {len(bldg_pts):,}")

    with open(ref_path) as f:
        ref_data = json.load(f)
    ref_buildings = ref_data["buildings"]
    print(f"3DBAG reference buildings: {len(ref_buildings)}")
    print()

    # Run all methods
    methods = {
        "A_dbscan_xy": {
            "desc": "DBSCAN in XY only (eps=2m, baseline)",
            "labels_fn": lambda: method_a_dbscan(bldg_pts, eps=2.0, min_pts=15),
        },
        "B_conf_z_filtered": {
            "desc": "DBSCAN with conf>=0.7 + Z>=2m filter",
            "labels_fn": lambda: method_b_filtered_dbscan(
                bldg_pts, bldg_conf, eps=2.0, min_pts=15, conf_threshold=0.7, z_min=2.0),
        },
        "C_watershed_0.5m": {
            "desc": "2D height-map watershed (0.5m grid, peaks at Z>3.5m)",
            "labels_fn": lambda: method_c_watershed(
                bldg_pts, bldg_conf, cell_size=0.5, z_min=2.0, conf_min=0.7,
                min_peak_z=3.5, min_area_cells=15),
        },
        "D_3d_anisotropic": {
            "desc": "3D DBSCAN (xy_eps=2m, z_eps=1.5m)",
            "labels_fn": lambda: method_d_3d_anisotropic_dbscan(
                bldg_pts, bldg_conf, xy_eps=2.0, z_eps=1.5, min_pts=10, conf_min=0.7, z_min=2.0),
        },
    }

    results = {}
    for key, cfg in methods.items():
        print(f"\n--- Method {key}: {cfg['desc']} ---")
        try:
            labels = cfg["labels_fn"]()
            unique = [l for l in np.unique(labels) if l >= 0]
            print(f"  Raw clusters: {len(unique)}")
            metrics = evaluate_instances(bldg_pts, labels, ref_buildings)
            results[key] = {"desc": cfg["desc"], **metrics}
            print(f"  Instances (after filter):  {metrics['n_instances']}")
            print(f"  Matched to 3DBAG:          {metrics['n_matched']} ({metrics['match_rate_pct']}%)")
            print(f"  3DBAG coverage:            {metrics['ref_coverage_pct']}%")
            print(f"  Footprint mean IoU:        {metrics['footprint_mean_iou']}")
            print(f"  Footprint median IoU:      {metrics['footprint_median_iou']}")
            print(f"  Height MAE:                {metrics['height_mae_m']} m")
            print(f"  Height bias:               {metrics['height_bias_m']} m")
            print(f"  Floor exact:               {metrics['floor_exact_pct']}%")
            print(f"  Floor within +/-1:         {metrics['floor_within1_pct']}%")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            results[key] = {"desc": cfg["desc"], "error": str(exc)}

    # Summary
    print("\n" + "="*75)
    print("SUMMARY TABLE")
    print(f"{'Method':<25} {'Inst':>5} {'Match':>5} {'3DBAG%':>7} {'fp IoU':>7} {'h MAE':>6} {'fl ex':>6} {'fl±1':>5}")
    print("-"*75)
    for key, r in results.items():
        if "error" in r:
            print(f"  {key:<23} ERROR: {r['error'][:30]}")
            continue
        print(f"  {key:<23} {r['n_instances']:>5} {r['n_matched']:>5} "
              f"{r['ref_coverage_pct']:>7.1f}% {r.get('footprint_mean_iou') or 0:>7.4f} "
              f"{r.get('height_mae_m') or 0:>6.2f} "
              f"{r.get('floor_exact_pct') or 0:>6.1f} {r.get('floor_within1_pct') or 0:>5.1f}")
    print("="*75)

    with open(out_dir / "instance_method_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_dir / 'instance_method_comparison.json'}")


if __name__ == "__main__":
    main()

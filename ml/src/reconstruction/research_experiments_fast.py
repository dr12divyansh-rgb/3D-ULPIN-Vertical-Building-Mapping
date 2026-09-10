"""Fast research experiments comparing instance extraction methods.

For method COMPARISON (not final reporting), uses centroid-distance + area-ratio
matching instead of full IoU rasterization. This runs in seconds instead of minutes.
Full IoU is reserved for the final best method.

Methods tested:
  A: DBSCAN in XY only, eps=2m (baseline)
  B: DBSCAN with conf>=0.7 + Z>=2m filter
  C: 2D height-map watershed (roof peaks as seeds, Z-guided expansion)
  C2: Watershed with smaller cell (0.5m) and tighter peak threshold
  D: 3D DBSCAN (xy_eps=2m, z_eps=1.5m)

Output:
  ml/results/final_ahn4_research_v2/method_comparison.json
  (printed comparison table)
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

sys.setrecursionlimit(10000)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ply(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()
    hend = raw.find(b"end_header")
    body = raw[hend + len(b"end_header"):].lstrip(b"\r\n")
    fmt  = struct.Struct("<fffBBBBBf")
    n    = struct.calcsize(fmt.format)
    total = len(body) // n
    return np.frombuffer(body[:total * n], dtype=np.dtype([
        ("x","<f4"),("y","<f4"),("z","<f4"),
        ("r","<u1"),("g","<u1"),("b","<u1"),
        ("pred","<u1"),("gt","<u1"),("conf","<f4"),
    ]))


# ── Shared helpers ────────────────────────────────────────────────────────────

def _grid_dbscan_xy(xy: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Grid-accelerated 2D DBSCAN."""
    n = len(xy)
    labels  = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cell    = eps
    x0 = float(xy[:,0].min()); y0 = float(xy[:,1].min())
    ci = ((xy[:,0]-x0)/cell).astype(np.int32)
    ri = ((xy[:,1]-y0)/cell).astype(np.int32)
    grid: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n):
        grid[(int(ri[i]), int(ci[i]))].append(i)

    eps2 = eps * eps
    def nb(i):
        rc, cc = int(ri[i]), int(ci[i])
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for j in grid.get((rc+dr, cc+dc), []):
                    dx = xy[j,0]-xy[i,0]; dy = xy[j,1]-xy[i,1]
                    if dx*dx+dy*dy <= eps2: out.append(j)
        return out

    cluster_id = 0
    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        nb_i = nb(i)
        if len(nb_i) < min_pts: continue
        labels[i] = cluster_id
        seeds = list(nb_i)
        si = 0
        while si < len(seeds):
            j = seeds[si]; si += 1
            if not visited[j]:
                visited[j] = True
                nb_j = nb(j)
                if len(nb_j) >= min_pts: seeds.extend(nb_j)
            if labels[j] < 0: labels[j] = cluster_id
        cluster_id += 1
    return labels


def poly_area(poly):
    n = len(poly)
    if n < 3: return 0.0
    a = 0.0
    for i in range(n):
        j = (i+1)%n
        a += poly[i][0]*poly[j][1] - poly[j][0]*poly[i][1]
    return abs(a)/2.0


def poly_centroid(poly):
    if not poly: return (0.0, 0.0)
    return (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))


# ── Instance properties ───────────────────────────────────────────────────────

def instances_from_labels(pts: np.ndarray, labels: np.ndarray, min_pts: int = 20):
    """Extract instance properties from point labels."""
    unique = np.unique(labels)
    out = []
    for lbl in unique:
        if lbl < 0: continue
        mask = labels == lbl
        if mask.sum() < min_pts: continue
        sub  = pts[mask]
        xy   = sub[:, :2]
        z    = sub[:, 2]
        x0,y0,x1,y1 = xy[:,0].min(),xy[:,1].min(),xy[:,0].max(),xy[:,1].max()
        area = (x1-x0)*(y1-y0)
        if area < 10.0: continue
        cx = (x0+x1)/2; cy = (y0+y1)/2
        gz = float(np.percentile(z, 5))
        rz = float(np.percentile(z, 95))
        h  = max(0.0, rz - gz)
        out.append({
            "label": int(lbl), "n_pts": int(mask.sum()),
            "cx": cx, "cy": cy, "area": area, "height": h,
            "bbox": (x0, y0, x1, y1),
        })
    return out


# ── Fast matching: centroid distance + area ratio ─────────────────────────────

def fast_match(instances, ref_buildings,
               max_dist: float = 30.0,
               min_area_ratio: float = 0.1,
               max_area_ratio: float = 10.0):
    """Fast greedy matching using centroid distance + area ratio (no IoU raster).

    Intended only for quick method COMPARISON.
    """
    matches = []
    used_refs = set()

    # Build ref centroid array
    ref_cx = np.array([b["_cx"] for b in ref_buildings])
    ref_cy = np.array([b["_cy"] for b in ref_buildings])

    # Sort instances by n_pts desc (larger instances first)
    for inst in sorted(instances, key=lambda x: -x["n_pts"]):
        icx, icy = inst["cx"], inst["cy"]
        dists = np.sqrt((ref_cx - icx)**2 + (ref_cy - icy)**2)
        cands = np.argsort(dists)

        best_d = float("inf"); best_r = None
        for ri in cands:
            d = float(dists[ri])
            if d > max_dist: break
            rb = ref_buildings[ri]
            if rb["bag_id"] in used_refs: continue
            ref_area = rb.get("footprint_area") or 1.0
            ar = inst["area"] / ref_area
            if not (min_area_ratio <= ar <= max_area_ratio): continue
            if d < best_d:
                best_d = d; best_r = ri
        if best_r is not None:
            used_refs.add(ref_buildings[best_r]["bag_id"])
            rb = ref_buildings[best_r]
            ref_h = rb.get("height") or 0
            ref_fl = rb.get("floor_count")
            our_fl = max(1, round(inst["height"] / 3.2))
            matches.append({
                "dist": best_d, "area_ratio": inst["area"]/(rb.get("footprint_area") or 1),
                "h_err": inst["height"] - ref_h,
                "fl_exact": (our_fl == ref_fl) if ref_fl else None,
                "fl_w1": (abs(our_fl - ref_fl) <= 1) if ref_fl else None,
            })
    return matches


def summarize(method_name, instances, matches, n_ref):
    n_inst  = len(instances)
    n_match = len(matches)
    h_errs  = [m["h_err"] for m in matches]
    h_abs   = [abs(v) for v in h_errs]
    fl_ex   = [m["fl_exact"] for m in matches if m["fl_exact"] is not None]
    fl_w1   = [m["fl_w1"]    for m in matches if m["fl_w1"]    is not None]

    def smean(v): return round(float(np.mean(v)),3) if v else None
    def spct(v): return round(100*sum(v)/len(v),1) if v else None

    return {
        "method":             method_name,
        "n_instances":        n_inst,
        "n_matched":          n_match,
        "match_pct":          round(100*n_match/max(n_inst,1),1),
        "ref_coverage_pct":   round(100*n_match/n_ref,1),
        "height_mae_m":       smean(h_abs),
        "height_bias_m":      smean(h_errs),
        "floor_exact_pct":    spct(fl_ex),
        "floor_within1_pct":  spct(fl_w1),
    }


# ── Method implementations ────────────────────────────────────────────────────

def run_method_a(bpts, bconf):
    """Baseline DBSCAN, XY only, all building points."""
    print("  Running DBSCAN eps=2m on all building points...")
    labels = _grid_dbscan_xy(bpts[:, :2], eps=2.0, min_pts=15)
    return instances_from_labels(bpts, labels)


def run_method_b(bpts, bconf):
    """DBSCAN with confidence + Z filter."""
    print("  Filtering conf>=0.7, Z>=2m...")
    mask = (bconf >= 0.7) & (bpts[:, 2] >= 2.0)
    print(f"  Points after filter: {mask.sum():,} (from {len(bpts):,})")
    if mask.sum() < 15: return []
    sub = bpts[mask]
    labels_sub = _grid_dbscan_xy(sub[:, :2], eps=2.0, min_pts=15)
    labels = np.full(len(bpts), -1, dtype=np.int32)
    labels[mask] = labels_sub
    return instances_from_labels(bpts, labels)


def run_method_c(bpts, bconf, cell_size=1.0, z_min=2.5, min_peak_z=4.0,
                  min_area_cells=15, conf_min=0.7):
    """2D height-map watershed from building roof peaks."""
    print(f"  Building height map (cell={cell_size}m)...")
    mask = (bconf >= conf_min) & (bpts[:, 2] >= z_min)
    sub  = bpts[mask]
    if len(sub) < 20: return []

    x = sub[:,0]; y = sub[:,1]; z = sub[:,2]
    x0, y0 = float(x.min()), float(y.min())
    xi = ((x-x0)/cell_size).astype(np.int32)
    yi = ((y-y0)/cell_size).astype(np.int32)
    cols = int(xi.max())+2; rows = int(yi.max())+2
    z_map  = np.full((rows,cols), -999.0)
    np.maximum.at(z_map, (yi, xi), z)
    occ = z_map > -999
    print(f"  Grid: {rows}x{cols}, occupied cells: {occ.sum():,}")

    # Local maxima: z >= all 8 neighbors
    not_max = np.zeros_like(occ)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr==0 and dc==0: continue
            sr = np.roll(z_map, dr, 0); sc = np.roll(z_map, dc, 1)
            not_max |= (z_map < sr) | (z_map < sc)
    peaks = occ & ~not_max & (z_map >= min_peak_z)
    n_peaks = peaks.sum()
    print(f"  Peaks (Z>={min_peak_z}m): {n_peaks}")

    # Watershed: BFS from peaks (priority = Z, highest first)
    label_map = np.full((rows,cols), -1, dtype=np.int32)
    heap = []
    cid = 0
    for (r,c) in zip(*np.where(peaks)):
        label_map[r,c] = cid
        heapq.heappush(heap, (-float(z_map[r,c]), r, c, cid))
        cid += 1

    while heap:
        neg_z, r, c, lbl = heapq.heappop(heap)
        if label_map[r,c] != lbl: continue
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and occ[nr,nc] and label_map[nr,nc]<0:
                label_map[nr,nc] = lbl
                heapq.heappush(heap, (-float(z_map[nr,nc]), nr, nc, lbl))

    # Filter tiny regions
    for lbl in range(cid):
        if (label_map==lbl).sum() < min_area_cells:
            label_map[label_map==lbl] = -1

    # Map back to points
    labels = np.full(len(bpts), -1, dtype=np.int32)
    sub_idx = np.where(mask)[0]
    for k in range(len(sub)):
        labels[sub_idx[k]] = label_map[yi[k], xi[k]]

    return instances_from_labels(bpts, labels)


def run_method_d(bpts, bconf, xy_eps=2.0, z_eps=1.5, min_pts=10,
                  conf_min=0.7, z_min=2.0):
    """3D anisotropic DBSCAN."""
    print(f"  3D DBSCAN: xy_eps={xy_eps}m, z_eps={z_eps}m...")
    mask = (bconf >= conf_min) & (bpts[:,2] >= z_min)
    print(f"  Points after filter: {mask.sum():,}")
    if mask.sum() < min_pts: return []
    sub = bpts[mask]
    xyz = sub[:,:3]
    n = len(xyz)
    labels_sub = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    # Grid-accelerated 3D neighbor search
    cell_xy = xy_eps; cell_z = z_eps
    x0,y0,z0 = xyz[:,0].min(), xyz[:,1].min(), xyz[:,2].min()
    ci = ((xyz[:,0]-x0)/cell_xy).astype(np.int32)
    ri = ((xyz[:,1]-y0)/cell_xy).astype(np.int32)
    zi = ((xyz[:,2]-z0)/cell_z ).astype(np.int32)
    grid3: dict[tuple, list[int]] = defaultdict(list)
    for i in range(n):
        grid3[(int(ri[i]),int(ci[i]),int(zi[i]))].append(i)

    xy_eps2 = xy_eps*xy_eps
    def nb3(i):
        rr,cc,zz = int(ri[i]),int(ci[i]),int(zi[i])
        out = []
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                for dz in (-1,0,1):
                    for j in grid3.get((rr+dr,cc+dc,zz+dz),[]):
                        dx=xyz[j,0]-xyz[i,0]; dy=xyz[j,1]-xyz[i,1]; dz2=xyz[j,2]-xyz[i,2]
                        if dx*dx+dy*dy<=xy_eps2 and abs(dz2)<=z_eps: out.append(j)
        return out

    cid = 0
    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        nb_i = nb3(i)
        if len(nb_i) < min_pts: continue
        labels_sub[i] = cid
        seeds = list(nb_i); si = 0
        while si < len(seeds):
            j = seeds[si]; si += 1
            if not visited[j]:
                visited[j] = True
                nb_j = nb3(j)
                if len(nb_j) >= min_pts: seeds.extend(nb_j)
            if labels_sub[j] < 0: labels_sub[j] = cid
        cid += 1

    labels = np.full(len(bpts), -1, dtype=np.int32)
    labels[mask] = labels_sub
    return instances_from_labels(bpts, labels)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ply_path = ROOT / "ml/results/ahn4_exp007/visuals/exp007_test_prediction.ply"
    ref_path = ROOT / "ml/results/final_ahn4/3dbag_footprints.json"
    out_dir  = ROOT / "ml/results/final_ahn4_research_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_ply(ply_path)
    bldg_mask = data["pred"] == 1
    pts  = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    conf = data["conf"].astype(np.float32)
    bpts = pts[bldg_mask]; bconf = conf[bldg_mask]
    print(f"Building points: {len(bpts):,}")

    with open(ref_path) as f:
        ref_data = json.load(f)
    ref_buildings = ref_data["buildings"]
    for rb in ref_buildings:
        fp = [(p[0],p[1]) for p in rb.get("footprint",[])]
        rb["_cx"] = sum(p[0] for p in fp)/len(fp) if fp else 0
        rb["_cy"] = sum(p[1] for p in fp)/len(fp) if fp else 0
    print(f"3DBAG buildings: {len(ref_buildings)}")
    print()

    all_results = {}
    methods = [
        ("A: baseline DBSCAN XY",       run_method_a),
        ("B: conf+Z filtered DBSCAN",    run_method_b),
        ("C: watershed 1m",              lambda bp,bc: run_method_c(bp,bc,cell_size=1.0,min_peak_z=4.0,min_area_cells=10)),
        ("C2: watershed 0.5m",           lambda bp,bc: run_method_c(bp,bc,cell_size=0.5,min_peak_z=3.5,min_area_cells=15)),
        ("D: 3D anisotropic DBSCAN",     run_method_d),
    ]

    for name, fn in methods:
        print(f"\n{'='*50}")
        print(f"Method: {name}")
        try:
            instances = fn(bpts, bconf)
            matches   = fast_match(instances, ref_buildings, max_dist=30.0)
            stats = summarize(name, instances, matches, len(ref_buildings))
            all_results[name] = stats
            print(f"  Instances:     {stats['n_instances']}")
            print(f"  Matched:       {stats['n_matched']} ({stats['match_pct']}%)")
            print(f"  Ref coverage:  {stats['ref_coverage_pct']}%")
            print(f"  Height MAE:    {stats['height_mae_m']} m")
            print(f"  Floor exact:   {stats['floor_exact_pct']}%")
            print(f"  Floor within1: {stats['floor_within1_pct']}%")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            all_results[name] = {"method": name, "error": str(exc)}

    # Summary table
    print("\n" + "="*80)
    print("COMPARISON TABLE (fast centroid+area matching)")
    print(f"  {'Method':<30} {'Inst':>5} {'Match':>6} {'Ref%':>6} {'hMAE':>6} {'fl ex':>6} {'fl±1':>5}")
    print("-"*80)
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<30} ERROR")
            continue
        print(f"  {name:<30} {r['n_instances']:>5} {r['n_matched']:>6} "
              f"{r['ref_coverage_pct']:>6.1f}% {r.get('height_mae_m') or 0:>6.2f} "
              f"{r.get('floor_exact_pct') or 0:>6.1f} {r.get('floor_within1_pct') or 0:>5.1f}")
    print("="*80)

    with open(out_dir / "method_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_dir / 'method_comparison.json'}")


if __name__ == "__main__":
    main()

"""Reconstruct LOD1/LOD2 building meshes from a semantic segmentation prediction PLY.

Works on ANY prediction PLY with x, y, z, predicted_class fields.
No scene metadata, no ground truth, no database — pure geometry from points.

Algorithm per building:
  1. 2-D grid flood-fill  → separate individual building instances
  2. Convex-hull (Graham scan) → 2-D footprint polygon
  3. Z-percentile analysis → ground level, eave level, roof peak
  4. Roof-type detection (Z std of roof points) → flat or gable
  5. Watertight mesh → walls + flat slab OR gable roof planes

Usage:
    # On your STPLS3D prediction:
    python ml/src/reconstruction/reconstruct_from_prediction.py \
        --pred ml/results/dales2_500k_prediction.ply \
        --out  ml/results/reconstructed_buildings/

    # Adjust sensitivity for sparse aerial data:
    python ml/src/reconstruction/reconstruct_from_prediction.py \
        --pred ml/results/dales2_500k_prediction.ply \
        --out  ml/results/reconstructed_buildings/ \
        --cell-size 2.0 --min-points 20

Output:
    <out>/building_0001.obj  ...  individual buildings
    <out>/scene.obj              all buildings in one file
    <out>/summary.json           per-building stats
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Robust PLY reader — handles all common type names, ascii + both binary modes
# ─────────────────────────────────────────────────────────────────────────────

_PLY_TYPES: dict[str, str] = {
    "float": "f", "float32": "f", "double": "d", "float64": "d",
    "char": "b",  "int8": "b",   "uchar": "B",  "uint8": "B",
    "short": "h", "int16": "h",  "ushort": "H", "uint16": "H",
    "int": "i",   "int32": "i",  "uint": "I",   "uint32": "I",
    "long": "l",  "int64": "l",  "ulong": "L",  "uint64": "L",
}


def _read_ply(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        raw = f.read()
    hend = raw.find(b"end_header")
    if hend == -1:
        raise ValueError(f"{path}: not a valid PLY (no end_header)")
    header = raw[:hend].decode("ascii", errors="replace")
    body   = raw[hend + len(b"end_header"):].lstrip(b"\r\n")

    fmt, n_verts, props = "ascii", 0, []
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3 and parts[1] == "vertex":
            n_verts = int(parts[2])
        elif parts[0] == "property" and len(parts) >= 3 and parts[1] != "list":
            sc = _PLY_TYPES.get(parts[1])
            if sc:
                props.append((parts[2], sc))

    endian = ">" if fmt == "binary_big_endian" else "<"
    data = {name: np.empty(n_verts, dtype=np.dtype(endian + sc))
            for name, sc in props}

    if fmt == "ascii":
        for i, line in enumerate(body.decode("ascii").strip().splitlines()[:n_verts]):
            for (name, sc), val in zip(props, line.split()):
                data[name][i] = float(val) if sc in "fd" else int(val)
    else:
        row_fmt  = struct.Struct(endian + "".join(sc for _, sc in props))
        row_size = row_fmt.size
        for i in range(n_verts):
            row = row_fmt.unpack_from(body, i * row_size)
            for (name, _), val in zip(props, row):
                data[name][i] = val

    return data


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz float32 (N,3), pred_class int32 (N,))."""
    data = _read_ply(path)
    keys = set(data.keys())

    def _get(*candidates: str) -> np.ndarray:
        for k in candidates:
            if k in keys:
                return data[k]
        raise KeyError(f"None of {candidates} found. PLY fields: {sorted(keys)}")

    xyz = np.column_stack([
        _get("x", "X").astype(np.float32),
        _get("y", "Y").astype(np.float32),
        _get("z", "Z").astype(np.float32),
    ])
    pred = _get("predicted_class", "class_id", "classification").astype(np.int32)
    return xyz, pred


# ─────────────────────────────────────────────────────────────────────────────
# 2-D convex hull — Graham scan, no external dependencies
# ─────────────────────────────────────────────────────────────────────────────

def _cross2d(O: tuple, A: tuple, B: tuple) -> float:
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])


def convex_hull_2d(pts_xy: np.ndarray) -> list[tuple[float, float]]:
    """Return CCW convex hull of 2-D points."""
    unique = sorted(set((float(p[0]), float(p[1])) for p in pts_xy))
    if len(unique) < 3:
        return unique
    lower: list = []
    for p in unique:
        while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(unique):
        while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _polygon_area(poly: list[tuple]) -> float:
    """Shoelace formula."""
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Instance separation — 2-D grid + BFS flood-fill
# ─────────────────────────────────────────────────────────────────────────────

def separate_instances(
    xyz: np.ndarray,
    cell_size: float = 1.5,
    min_points: int = 30,
) -> list[np.ndarray]:
    """
    Group building points into per-building instances.
    Returns list of index arrays (into xyz), sorted largest-first.
    """
    if len(xyz) == 0:
        return []

    x_min = float(xyz[:, 0].min())
    y_min = float(xyz[:, 1].min())
    ci = ((xyz[:, 0] - x_min) / cell_size).astype(np.int32)
    cj = ((xyz[:, 1] - y_min) / cell_size).astype(np.int32)

    # Build cell → point-index list
    cell_pts: dict[tuple[int, int], list[int]] = {}
    for idx in range(len(xyz)):
        key = (int(ci[idx]), int(cj[idx]))
        if key not in cell_pts:
            cell_pts[key] = []
        cell_pts[key].append(idx)

    occupied = set(cell_pts)
    visited: set[tuple[int, int]] = set()
    instances: list[np.ndarray] = []

    for start in occupied:
        if start in visited:
            continue
        component_cells: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([start])
        visited.add(start)
        while q:
            cx_, cy_ = q.popleft()
            component_cells.append((cx_, cy_))
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):  # 4-connectivity only
                nb = (cx_ + di, cy_ + dj)
                if nb in occupied and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        pts_list: list[int] = []
        for cell in component_cells:
            pts_list.extend(cell_pts[cell])
        if len(pts_list) >= min_points:
            instances.append(np.array(pts_list, dtype=np.int64))

    instances.sort(key=len, reverse=True)
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Building analysis — heights + roof type
# ─────────────────────────────────────────────────────────────────────────────

def analyze_building(pts: np.ndarray) -> dict:
    """
    Analyse a building's point cloud and return geometry parameters.

    Returns dict with:
      footprint     list[(x,y)]   CCW convex hull in world XY
      z_ground      float         base elevation
      z_eave        float         top-of-wall elevation
      z_peak        float         highest point (ridge or flat top)
      height        float         total height
      is_pitched    bool          True if roof is not flat
      ridge_axis    'x'|'y'|None  dominant ridge direction (if pitched)
      n_points      int
      footprint_area float        m²
    """
    z = pts[:, 2]
    z_ground = float(z.min())
    z_peak   = float(z.max())
    height   = z_peak - z_ground

    # Eave: 80th-percentile Z — robust boundary between walls and roof
    z_eave = float(np.percentile(z, 80))
    # Clamp: eave between 55% and 92% of building height
    z_eave = float(np.clip(z_eave,
                           z_ground + 0.55 * height,
                           z_ground + 0.92 * height))

    footprint = convex_hull_2d(pts[:, :2])

    # Roof type: check Z spread of points above eave
    roof_pts = pts[z > z_eave]
    is_pitched = False
    ridge_axis = None

    if len(roof_pts) >= 12 and height > 2.0:
        roof_z_std = float(np.std(roof_pts[:, 2]))
        if roof_z_std > 0.5:          # >50 cm variation → pitched
            is_pitched = True
            # PCA on all building XY → major axis = ridge direction
            xy   = pts[:, :2] - pts[:, :2].mean(axis=0)
            cov  = xy.T @ xy
            _, vecs = np.linalg.eigh(cov)
            major = vecs[:, -1]       # eigenvector of largest eigenvalue
            ridge_axis = "x" if abs(major[0]) >= abs(major[1]) else "y"

    return {
        "footprint":       footprint,
        "z_ground":        z_ground,
        "z_eave":          z_eave,
        "z_peak":          z_peak,
        "height":          height,
        "is_pitched":      is_pitched,
        "ridge_axis":      ridge_axis,
        "n_points":        len(pts),
        "footprint_area":  _polygon_area(footprint),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mesh builder
# ─────────────────────────────────────────────────────────────────────────────

class MeshBuilder:
    """Accumulates vertices + triangular faces for OBJ export."""

    def __init__(self) -> None:
        self.verts: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]]       = []

    # ── primitives ──────────────────────────────────────────────────────────

    def v(self, x: float, y: float, z: float) -> int:
        """Add vertex; return 1-based OBJ index."""
        self.verts.append((float(x), float(y), float(z)))
        return len(self.verts)

    def tri(self, a: int, b: int, c: int) -> None:
        self.faces.append((a, b, c))

    def quad(self, a: int, b: int, c: int, d: int) -> None:
        """Split quad into two CCW triangles."""
        self.tri(a, b, c)
        self.tri(a, c, d)

    def fan(self, center: int, ring: list[int]) -> None:
        """Fan-triangulate a polygon from a center vertex."""
        n = len(ring)
        for i in range(n):
            self.tri(center, ring[i], ring[(i + 1) % n])


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_flat(mb: MeshBuilder,
                fp: list[tuple[float, float]],
                z_ground: float,
                z_top: float) -> None:
    """Watertight extrusion: walls + flat roof + floor."""
    n = len(fp)
    bot = [mb.v(x, y, z_ground) for x, y in fp]
    top = [mb.v(x, y, z_top)    for x, y in fp]

    # Walls (outward-facing quads)
    for i in range(n):
        j = (i + 1) % n
        mb.quad(bot[i], bot[j], top[j], top[i])

    # Flat roof — fan from centroid
    cx = sum(x for x, y in fp) / n
    cy = sum(y for x, y in fp) / n
    rc = mb.v(cx, cy, z_top)
    mb.fan(rc, top)

    # Floor — fan (reversed winding = downward-facing)
    fc = mb.v(cx, cy, z_ground)
    mb.fan(fc, list(reversed(bot)))


def _build_gable(mb: MeshBuilder,
                 fp: list[tuple[float, float]],
                 z_ground: float,
                 z_eave: float,
                 z_peak: float,
                 ridge_axis: str) -> None:
    """
    Gable-roof building:
      - Walls from z_ground → z_eave using the actual convex-hull footprint
      - Gable roof from bounding-box corners (reliable, always valid geometry)
    """
    fp_arr = np.array(fp)
    n      = len(fp)
    x_min, x_max = float(fp_arr[:, 0].min()), float(fp_arr[:, 0].max())
    y_min, y_max = float(fp_arr[:, 1].min()), float(fp_arr[:, 1].max())
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    # ── walls (actual footprint shape, ground → eave) ────────────────────
    bot  = [mb.v(x, y, z_ground) for x, y in fp]
    eave = [mb.v(x, y, z_eave)   for x, y in fp]
    for i in range(n):
        j = (i + 1) % n
        mb.quad(bot[i], bot[j], eave[j], eave[i])

    # Floor
    fc_xy_x = sum(x for x, y in fp) / n
    fc_xy_y = sum(y for x, y in fp) / n
    fc = mb.v(fc_xy_x, fc_xy_y, z_ground)
    mb.fan(fc, list(reversed(bot)))

    # ── gable roof (bounding-box corners for clean geometry) ─────────────
    if ridge_axis == "x":
        # Ridge runs West→East; slopes face North and South
        r0 = mb.v(x_min, cy, z_peak)   # west ridge end
        r1 = mb.v(x_max, cy, z_peak)   # east ridge end

        nw = mb.v(x_min, y_min, z_eave)
        ne = mb.v(x_max, y_min, z_eave)
        se = mb.v(x_max, y_max, z_eave)
        sw = mb.v(x_min, y_max, z_eave)

        # North slope (y_min side)
        mb.quad(r0, r1, ne, nw)
        # South slope (y_max side)
        mb.quad(r1, r0, sw, se)
        # West gable triangle
        mb.tri(r0, nw, sw)
        # East gable triangle
        mb.tri(r1, se, ne)

    else:
        # Ridge runs South→North; slopes face East and West
        r0 = mb.v(cx, y_min, z_peak)   # south ridge end
        r1 = mb.v(cx, y_max, z_peak)   # north ridge end

        nw = mb.v(x_min, y_min, z_eave)
        ne = mb.v(x_max, y_min, z_eave)
        se = mb.v(x_max, y_max, z_eave)
        sw = mb.v(x_min, y_max, z_eave)

        # West slope (x_min side)
        mb.quad(r0, r1, sw, nw)
        # East slope (x_max side)
        mb.quad(r1, r0, ne, se)
        # South gable triangle
        mb.tri(r0, nw, ne)
        # North gable triangle
        mb.tri(r1, se, sw)


def generate_mesh(info: dict) -> MeshBuilder:
    """Build a complete watertight mesh for one building."""
    mb = MeshBuilder()
    fp = info["footprint"]
    if len(fp) < 3:
        return mb

    if info["is_pitched"] and (info["z_peak"] - info["z_eave"]) > 0.5:
        _build_gable(mb, fp,
                     info["z_ground"], info["z_eave"],
                     info["z_peak"],   info["ridge_axis"])
    else:
        _build_flat(mb, fp, info["z_ground"], info["z_peak"])

    return mb


# ─────────────────────────────────────────────────────────────────────────────
# OBJ export
# ─────────────────────────────────────────────────────────────────────────────

def write_obj(path: Path, mb: MeshBuilder, group: str = "building") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# SIH 26011 - building reconstruction\ng {group}\n"]
    for x, y, z in mb.verts:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}\n")
    for a, b, c in mb.faces:
        lines.append(f"f {a} {b} {c}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_scene_obj(path: Path,
                    buildings: list[tuple[str, MeshBuilder]]) -> None:
    """All buildings in one OBJ, each as a named group (for colour-by-group in CloudCompare)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# SIH 26011 - reconstructed building scene\n\n"]
    v_offset = 0
    for name, mb in buildings:
        lines.append(f"g {name}\n")
        for x, y, z in mb.verts:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for a, b, c in mb.faces:
            lines.append(f"f {a+v_offset} {b+v_offset} {c+v_offset}\n")
        v_offset += len(mb.verts)
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_mtl(path: Path, n_buildings: int) -> None:
    """Generate a simple MTL with one distinct colour per building."""
    rng = np.random.default_rng(26011)
    lines = ["# building colours\n\n"]
    for i in range(1, n_buildings + 1):
        r, g, b = rng.uniform(0.3, 1.0, 3)
        lines.append(f"newmtl building_{i:04d}\n")
        lines.append(f"Kd {r:.3f} {g:.3f} {b:.3f}\n\n")
    path.write_text("".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reconstruct LOD1/LOD2 buildings from a prediction PLY.")
    ap.add_argument("--pred",  required=True,
                    help="Prediction PLY (needs x,y,z + predicted_class fields).")
    ap.add_argument("--out",   required=True,
                    help="Output directory.")
    ap.add_argument("--building-class", type=int, default=1,
                    help="Class integer for buildings (default 1).")
    ap.add_argument("--cell-size",  type=float, default=1.5,
                    help="Grid cell size in metres for instance separation (default 1.5).")
    ap.add_argument("--min-points", type=int,   default=30,
                    help="Minimum points to count as a building (default 30).")
    ap.add_argument("--max-buildings", type=int, default=300,
                    help="Maximum buildings to reconstruct (default 300).")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSIH 26011 - LOD2 Building Reconstruction")
    print(f"  input  : {pred_path}")
    print(f"  output : {out_dir}\n")

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading prediction PLY...")
    xyz, pred_class = load_prediction(pred_path)
    n_total    = len(xyz)
    n_building = int((pred_class == args.building_class).sum())
    pct        = 100.0 * n_building / max(n_total, 1)
    print(f"  {n_total:,} total points | {n_building:,} building points ({pct:.1f}%)")

    if n_building == 0:
        print("\nERROR: no building points found.")
        print(f"       Check that class {args.building_class} is the building class.")
        return 1

    building_xyz = xyz[pred_class == args.building_class]

    # ── Instance separation ───────────────────────────────────────────────
    print(f"\nSeparating instances "
          f"(cell_size={args.cell_size} m, min_points={args.min_points})...")
    instances = separate_instances(building_xyz, args.cell_size, args.min_points)
    print(f"  {len(instances)} instances found")

    if not instances:
        print("\nNo instances found.  Try --cell-size 2.0 or --min-points 15")
        return 1

    instances = instances[:args.max_buildings]
    print(f"  reconstructing top {len(instances)}\n")

    # ── Reconstruct ───────────────────────────────────────────────────────
    scene_meshes: list[tuple[str, MeshBuilder]] = []
    summary_rows: list[dict] = []
    n_flat = n_gable = n_skip = 0

    for i, idx in enumerate(instances):
        pts  = building_xyz[idx]
        info = analyze_building(pts)

        if len(info["footprint"]) < 3 or info["height"] < 1.0:
            n_skip += 1
            continue

        mb = generate_mesh(info)
        if not mb.verts:
            n_skip += 1
            continue

        bid  = i + 1
        name = f"building_{bid:04d}"
        write_obj(out_dir / f"{name}.obj", mb, group=name)
        scene_meshes.append((name, mb))

        roof = "gable" if info["is_pitched"] else "flat"
        if info["is_pitched"]:
            n_gable += 1
        else:
            n_flat += 1

        summary_rows.append({
            "id":               bid,
            "n_points":         info["n_points"],
            "height_m":         round(info["height"], 2),
            "footprint_area_m2":round(info["footprint_area"], 1),
            "roof_type":        roof,
            "z_ground":         round(info["z_ground"], 3),
            "z_eave":           round(info["z_eave"],   3),
            "z_peak":           round(info["z_peak"],   3),
            "n_footprint_verts":len(info["footprint"]),
            "n_mesh_verts":     len(mb.verts),
            "n_mesh_faces":     len(mb.faces),
        })

        print(f"  [{bid:3d}/{len(instances)}]  {name}"
              f"  pts={info['n_points']:6d}"
              f"  h={info['height']:5.1f}m"
              f"  area={info['footprint_area']:6.0f}m2"
              f"  roof={roof}")

    # ── Write scene OBJ and MTL ───────────────────────────────────────────
    scene_obj = out_dir / "scene.obj"
    scene_mtl = out_dir / "scene.mtl"
    write_scene_obj(scene_obj, scene_meshes)
    write_mtl(scene_mtl, len(scene_meshes))

    # Inject mtllib reference into scene.obj header
    txt = scene_obj.read_text(encoding="utf-8")
    scene_obj.write_text("mtllib scene.mtl\n" + txt, encoding="utf-8")

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "input_ply":               str(pred_path),
        "total_input_points":      n_total,
        "building_points":         n_building,
        "n_instances_found":       len(instances),
        "n_buildings_reconstructed": len(summary_rows),
        "n_flat_roof":             n_flat,
        "n_gable_roof":            n_gable,
        "n_skipped":               n_skip,
        "buildings":               summary_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Buildings reconstructed : {len(summary_rows)}")
    print(f"  Flat roof               : {n_flat}")
    print(f"  Gable / pitched roof    : {n_gable}")
    print(f"  Skipped (too small)     : {n_skip}")
    print(f"  Scene OBJ               : {scene_obj}")
    print(f"  Summary                 : {out_dir / 'summary.json'}")
    print(f"{'='*55}")
    print(f"\nTo visualise:")
    print(f"  CloudCompare: File > Open > {scene_obj}")
    print(f"  Blender:  File > Import > Wavefront (.obj) > {scene_obj}")
    print(f"  Or open individual building_XXXX.obj files.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Extract a 2-D footprint polygon from predicted building point clouds.

Methods:
  1. Convex hull (baseline) — always works, over-approximates concave shapes
  2. Raster boundary — preserves concave shapes (L, T, courtyard) via grid tracing
  3. Auto-select — chooses the best method based on shape characteristics

All decisions are recorded in the returned metadata dict so the caller
can report them honestly in provenance.
"""

from __future__ import annotations

import numpy as np

# Reuse the monotone-chain convex hull already in the project.
from synthetic_city.core.mesh import _convex_hull  # noqa: PLC2701


# ── Convex hull method (baseline) ─────────────────────────────────────────────

def extract_footprint_convex(
    xyz: np.ndarray,
    base_frac: float = 0.15,
    min_pts: int = 3,
) -> tuple[list[tuple[float, float]], dict]:
    """Return the convex hull of near-ground XY points and metadata.

    Args:
        xyz:        (N, 3) float32 building points (in world coordinates).
        base_frac:  fraction of the height range considered "near-ground".
                    Points with z ≤ z_min + base_frac * (z_max - z_min) are used.
        min_pts:    minimum number of hull vertices to accept; if the hull is
                    smaller the whole-cloud hull is tried; if still too small
                    an empty list is returned.

    Returns:
        (hull_xy, metadata)
        hull_xy:  list of (x, y) tuples in CCW order, may be empty.
        metadata: dict with keys: method, source, n_vertices, area_m2
    """
    if len(xyz) < min_pts:
        return [], {"method": "convex", "source": "empty", "n_vertices": 0, "area_m2": 0.0}

    z_arr = xyz[:, 2]
    z_min = float(z_arr.min())
    z_max = float(z_arr.max())
    z_thresh = z_min + base_frac * max(z_max - z_min, 1e-6)

    base_mask = z_arr <= z_thresh
    base_pts = xyz[base_mask]

    def _hull(pts: np.ndarray) -> list[tuple[float, float]]:
        return _convex_hull([(float(p[0]), float(p[1])) for p in pts])

    hull = _hull(base_pts)
    if len(hull) >= min_pts:
        area = _polygon_area(hull)
        return hull, {"method": "convex", "source": "points_base_hull", "n_vertices": len(hull), "area_m2": area}

    hull = _hull(xyz)
    if len(hull) >= min_pts:
        area = _polygon_area(hull)
        return hull, {"method": "convex", "source": "points_full_hull", "n_vertices": len(hull), "area_m2": area}

    return [], {"method": "convex", "source": "empty", "n_vertices": 0, "area_m2": 0.0}


# ── Raster boundary method (handles concave shapes) ───────────────────────────

def extract_footprint_raster(
    xyz: np.ndarray,
    cell_size: float = 0.5,
    base_frac: float = 0.15,
    dilate_cells: int = 1,
    simplify_threshold: float = 0.5,
    min_pts: int = 3,
) -> tuple[list[tuple[float, float]], dict]:
    """Raster boundary footprint that preserves concave shapes (L, T, courtyard).

    Algorithm:
      1. Voxelize near-ground XY points into a 2D occupancy grid
      2. Dilate grid to fill small gaps
      3. Trace the outer boundary using CELL EDGE midpoints (not centroids)
         — avoids collinear degenerate polygons for thin 1-cell-wide strips
      4. Simplify the resulting polygon (Douglas-Peucker)

    This method naturally handles L-shaped, T-shaped, and courtyard buildings.
    Convex hull only works for simple convex shapes.

    Args:
        xyz:              (N, 3) building points
        cell_size:        Grid resolution in metres (0.5m = high detail, 1.0m = coarse)
        base_frac:        Height fraction considered near-ground
        dilate_cells:     Dilate occupancy grid by this many cells (fills small gaps)
        simplify_threshold: Douglas-Peucker simplification tolerance in metres
        min_pts:          Minimum vertices to accept

    Returns:
        (polygon_xy, metadata)
        polygon_xy: list of (x, y) tuples in CCW order (world coordinates)
        metadata:   dict with method info
    """
    if len(xyz) < min_pts:
        return [], {"method": "raster", "source": "empty", "n_vertices": 0, "area_m2": 0.0,
                    "cell_size_m": cell_size, "n_cells": 0}

    # Filter to near-ground points
    z_arr = xyz[:, 2]
    z_min = float(z_arr.min())
    z_max = float(z_arr.max())
    z_thresh = z_min + base_frac * max(z_max - z_min, 1e-6)
    base_pts = xyz[z_arr <= z_thresh]

    if len(base_pts) < min_pts:
        base_pts = xyz

    xy = base_pts[:, :2]

    # Voxelize to 2D grid
    grid, x0, y0 = _rasterize_2d(xy, cell_size)
    n_cells_before = int(grid.sum())

    # Dilate to fill small gaps (handles point density variations)
    if dilate_cells > 0:
        grid = _dilate_binary(grid, dilate_cells)

    # Trace the outer boundary using CELL EDGES (not cell centroids)
    # This produces correct non-degenerate polygons even for 1-cell-wide strips
    polygon = _grid_outer_polygon(grid, cell_size, x0, y0)

    if len(polygon) < min_pts:
        return [], {"method": "raster", "source": "boundary_trace_failed",
                    "n_vertices": 0, "area_m2": 0.0,
                    "cell_size_m": cell_size, "n_cells": n_cells_before}

    # Simplify
    if simplify_threshold > 0 and len(polygon) > min_pts:
        polygon = _simplify_polygon(polygon, simplify_threshold)

    if len(polygon) < min_pts:
        return [], {"method": "raster", "source": "simplification_failed",
                    "n_vertices": 0, "area_m2": 0.0,
                    "cell_size_m": cell_size, "n_cells": n_cells_before}

    area = _polygon_area(polygon)
    return polygon, {"method": "raster", "source": "boundary_traced",
                     "n_vertices": len(polygon), "area_m2": area,
                     "cell_size_m": cell_size, "n_cells": n_cells_before}


# ── Auto-selection method (best of both worlds) ───────────────────────────────

def extract_footprint_best(
    xyz: np.ndarray,
    cell_size: float = 0.5,
    **kwargs
) -> tuple[list[tuple[float, float]], dict]:
    """Automatically select the best footprint method.

    Strategy:
      1. Try raster boundary first (handles concave shapes)
      2. Fall back to convex hull if raster fails or produces too few vertices

    Args:
        xyz:        (N, 3) building points
        cell_size:  Raster cell size in metres (smaller = more detail, slower)
        **kwargs:   Passed to both methods

    Returns:
        (polygon_xy, metadata)
        metadata["method"] will be "raster" or "convex-fallback"
    """
    # Try raster first
    raster_poly, raster_meta = extract_footprint_raster(xyz, cell_size=cell_size, **kwargs)
    if raster_poly and len(raster_poly) >= 3:
        return raster_poly, raster_meta

    # Fall back to convex hull
    convex_poly, convex_meta = extract_footprint_convex(xyz, **kwargs)
    convex_meta["method"] = "convex-fallback"
    convex_meta["raster_failed_reason"] = raster_meta.get("source", "unknown")
    return convex_poly, convex_meta


# ── Aerial LiDAR footprint (AHN4 / DALES) ────────────────────────────────────

def extract_footprint_aerial(
    xyz: np.ndarray,
    cell_size: float = 0.5,
    z_percentile_lo: float = 20.0,
    dilate_cells: int = 1,
    simplify_threshold: float = 0.3,
    min_pts: int = 3,
) -> tuple[list[tuple[float, float]], dict]:
    """Footprint extraction tuned for aerial / top-down LiDAR (AHN4, DALES).

    Key differences from the generic raster method (base_frac=0.15):
      - Uses UPPER Z points (above the z_percentile_lo-th Z percentile) instead
        of the bottom 15%. For aerial LiDAR, the roof surface IS the building
        outline; selecting "near-ground" points selects pavement FP noise.
      - Smaller default cell_size (0.5m vs 1.5m) for a tighter footprint.
      - Smaller default dilation (1 cell at 0.5m = 0.5m margin vs 1.5m).
      - Fallback to all points if the upper-Z filter leaves too few.

    Args:
        xyz:             (N, 3) building points in world coordinates.
        cell_size:       Grid resolution in metres (default 0.5m).
        z_percentile_lo: Lower percentile of Z used to remove ground-level FP
                         noise.  Points with z > z_percentile_lo-th pct are kept.
                         Default 20 (removes bottom 20% of Z range, i.e., low-Z
                         pavement/FP noise, keeps roof points).
        dilate_cells:    Grid dilation in cells (0.5m per cell at default res).
        simplify_threshold: Douglas-Peucker tolerance in metres.
        min_pts:         Minimum polygon vertices.

    Returns:
        (polygon_xy, metadata)
    """
    if len(xyz) < min_pts:
        return [], {"method": "aerial", "source": "empty", "n_vertices": 0, "area_m2": 0.0}

    z_arr = xyz[:, 2]
    z_lo = float(np.percentile(z_arr, z_percentile_lo))

    # Keep points ABOVE the low-Z threshold (roof / upper structure)
    upper_mask = z_arr > z_lo
    if upper_mask.sum() >= min_pts:
        pts = xyz[upper_mask]
    else:
        pts = xyz   # fallback: use everything

    xy = pts[:, :2]

    grid, x0, y0 = _rasterize_2d(xy, cell_size)
    n_cells = int(grid.sum())

    if dilate_cells > 0:
        grid = _dilate_binary(grid, dilate_cells)

    polygon = _grid_outer_polygon(grid, cell_size, x0, y0)
    if len(polygon) < min_pts:
        # Fallback: try with all points
        grid2, x0, y0 = _rasterize_2d(xyz[:, :2], cell_size)
        if dilate_cells > 0:
            grid2 = _dilate_binary(grid2, dilate_cells)
        polygon = _grid_outer_polygon(grid2, cell_size, x0, y0)

    if len(polygon) < min_pts:
        return [], {"method": "aerial", "source": "trace_failed",
                    "n_vertices": 0, "area_m2": 0.0, "cell_size_m": cell_size}

    if simplify_threshold > 0 and len(polygon) > min_pts:
        polygon = _simplify_polygon(polygon, simplify_threshold)

    if len(polygon) < min_pts:
        return [], {"method": "aerial", "source": "simplify_failed",
                    "n_vertices": 0, "area_m2": 0.0, "cell_size_m": cell_size}

    area = _polygon_area(polygon)
    return polygon, {
        "method":       "aerial",
        "source":       "boundary_traced_upper_z",
        "n_vertices":   len(polygon),
        "area_m2":      area,
        "cell_size_m":  cell_size,
        "n_cells":      n_cells,
        "z_pct_lo":     z_percentile_lo,
    }


# ── Rasterization helpers ─────────────────────────────────────────────────────

def _rasterize_2d(xy: np.ndarray, cell_size: float) -> tuple[np.ndarray, float, float]:
    """Voxelize 2D XY points into a binary occupancy grid.

    Returns:
        (grid, x0, y0)
        grid: (rows, cols) bool array
        x0, y0: world coordinates of grid[0, 0] lower-left corner
    """
    x_min = float(xy[:, 0].min())
    y_min = float(xy[:, 1].min())
    x_max = float(xy[:, 0].max())
    y_max = float(xy[:, 1].max())

    cols = max(1, int(np.ceil((x_max - x_min) / cell_size)))
    rows = max(1, int(np.ceil((y_max - y_min) / cell_size)))

    grid = np.zeros((rows, cols), dtype=bool)

    ci = ((xy[:, 0] - x_min) / cell_size).astype(np.int32)
    ri = ((xy[:, 1] - y_min) / cell_size).astype(np.int32)

    # Clamp to grid
    ci = np.clip(ci, 0, cols - 1)
    ri = np.clip(ri, 0, rows - 1)

    grid[ri, ci] = True

    return grid, x_min, y_min


def _dilate_binary(grid: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary dilation using a 3×3 cross (orthogonal) kernel.

    Fills small gaps caused by point density variations.
    """
    result = grid.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode='constant', constant_values=False)
        dilated = np.zeros_like(result)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Orthogonal neighbors
            shifted = padded[1 + dr:1 + dr + result.shape[0], 1 + dc:1 + dc + result.shape[1]]
            dilated |= shifted
        result = dilated | result
    return result


# ── Boundary polygon from grid edges ──────────────────────────────────────────

def _grid_outer_polygon(
    grid: np.ndarray,
    cell_size: float,
    x0: float,
    y0: float,
) -> list[tuple[float, float]]:
    """Extract the outer boundary polygon of a binary grid using cell edges.

    For each occupied cell, its 4 edges (shared with an unoccupied cell or
    grid border) are added as directed half-edges. These are then chained
    into an ordered closed polygon.

    This produces correct, non-degenerate polygons for:
      - Single cells (4-corner square)
      - 1-cell-wide strips (rectangle, not a degenerate collinear line)
      - L-shapes, T-shapes (concave polygons)
      - Courtyard buildings (outer boundary only)

    All vertex coordinates are cell CORNERS in world space (metres).

    Returns:
        Ordered list of (x, y) tuples forming a closed polygon.
        Returns [] if no occupied cells or cannot chain edges.
    """
    rows, cols = grid.shape

    # Build directed boundary edges as half-edges: (start_corner → end_corner)
    # Corner coordinates in grid units: (col_index, row_index) — note: col=x, row=y
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if not grid[r, c]:
                continue
            # Check each of the 4 cell edges
            # Convention: directed so the occupied cell is on the LEFT of the edge (CCW walk)
            # Top edge (y=r): left-to-right → (c,r)→(c+1,r)  [above is empty]
            if r == 0 or not grid[r - 1, c]:
                edges.append(((c, r), (c + 1, r)))
            # Bottom edge (y=r+1): right-to-left → (c+1,r+1)→(c,r+1)  [below is empty]
            if r == rows - 1 or not grid[r + 1, c]:
                edges.append(((c + 1, r + 1), (c, r + 1)))
            # Left edge (x=c): bottom-to-top → (c,r+1)→(c,r)  [left is empty]
            if c == 0 or not grid[r, c - 1]:
                edges.append(((c, r + 1), (c, r)))
            # Right edge (x=c+1): top-to-bottom → (c+1,r)→(c+1,r+1)  [right is empty]
            if c == cols - 1 or not grid[r, c + 1]:
                edges.append(((c + 1, r), (c + 1, r + 1)))

    if not edges:
        return []

    # Build adjacency: start_corner → end_corner
    adj: dict[tuple[int, int], tuple[int, int]] = {}
    for start, end in edges:
        adj[start] = end

    # Chain edges into a closed polygon starting from any corner
    start_corner = min(adj.keys())  # Deterministic start: smallest (col, row)
    polygon_corners: list[tuple[int, int]] = [start_corner]
    current = adj[start_corner]
    max_steps = len(adj) + 2

    for _ in range(max_steps):
        if current == start_corner:
            break
        polygon_corners.append(current)
        current = adj.get(current)
        if current is None:
            break

    if len(polygon_corners) < 3:
        return []

    # Convert grid corner coordinates to world space
    return [(x0 + c * cell_size, y0 + r * cell_size) for (c, r) in polygon_corners]


# ── Polygon utilities ─────────────────────────────────────────────────────────

def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    """Shoelace formula for polygon area (unsigned)."""
    if len(vertices) < 3:
        return 0.0
    area = 0.0
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _simplify_polygon(vertices: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker polygon simplification.

    Args:
        vertices:  Ordered list of (x, y) tuples
        tolerance: Maximum perpendicular distance from simplified edge (metres)

    Returns:
        Simplified list of (x, y) tuples
    """
    if len(vertices) <= 2:
        return vertices

    def _perpendicular_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        """Distance from point p to line segment ab."""
        ax, ay = a
        bx, by = b
        px, py = p
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return np.hypot(px - ax, py - ay)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return np.hypot(px - proj_x, py - proj_y)

    def _rdp(pts: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
        if len(pts) <= 2:
            return pts
        # Find farthest point from line segment start->end
        dmax = 0.0
        idx = 0
        for i in range(1, len(pts) - 1):
            d = _perpendicular_dist(pts[i], pts[0], pts[-1])
            if d > dmax:
                dmax = d
                idx = i
        # If max distance exceeds tolerance, subdivide
        if dmax > eps:
            left = _rdp(pts[:idx + 1], eps)
            right = _rdp(pts[idx:], eps)
            return left[:-1] + right
        else:
            return [pts[0], pts[-1]]

    return _rdp(vertices, tolerance)


# ── Quality metrics ───────────────────────────────────────────────────────────

def footprint_quality(
    footprint: list[tuple[float, float]],
    building_xyz: np.ndarray,
) -> dict:
    """Compute quality metrics for a footprint polygon.

    Metrics:
        coverage:      Fraction of building points inside the footprint (2D projection)
        compactness:   area / perimeter² (1.0 for circle, lower for elongated shapes)
        n_vertices:    Number of polygon vertices
        is_simple:     Whether polygon is simple (no self-intersections)

    Args:
        footprint:    List of (x, y) polygon vertices
        building_xyz: (N, 3) original building points

    Returns:
        dict of quality metrics
    """
    if len(footprint) < 3:
        return {"coverage": 0.0, "compactness": 0.0, "n_vertices": 0, "is_simple": False}

    area = _polygon_area(footprint)
    perimeter = sum(np.hypot(footprint[i][0] - footprint[(i + 1) % len(footprint)][0],
                             footprint[i][1] - footprint[(i + 1) % len(footprint)][1])
                    for i in range(len(footprint)))
    compactness = area / (perimeter * perimeter) if perimeter > 0 else 0.0

    # Coverage: fraction of XY points inside footprint
    xy = building_xyz[:, :2]
    inside = sum(_point_in_polygon(x, y, footprint) for x, y in xy)
    coverage = inside / len(xy) if len(xy) > 0 else 0.0

    # Simplicity check: no self-intersections (simplified check)
    is_simple = _is_simple_polygon(footprint)

    return {
        "coverage": coverage,
        "compactness": compactness,
        "n_vertices": len(footprint),
        "is_simple": is_simple,
    }


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray casting algorithm for point-in-polygon test."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def _is_simple_polygon(vertices: list[tuple[float, float]]) -> bool:
    """Check if polygon is simple (no self-intersections).

    Simplified check: just verify no consecutive edges are degenerate.
    A full check would test all edge pairs but is O(n²).
    """
    if len(vertices) < 3:
        return False
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        if x1 == x2 and y1 == y2:
            return False  # Degenerate edge
    return True

"""Building Instance Extraction for SIH 26011.

Takes a set of building points (class=1 from semantic segmentation) and
separates them into individual physical building instances.

Methods:
  1. Grid BFS (baseline) — 2D voxel grid connected-components (4-connectivity)
  2. DBSCAN-2D — density-based clustering via Open3D
  3. Adaptive — auto-selects method based on point density

This is NOT trained instance segmentation. These are spatial clustering
heuristics applied to already-segmented building points.

Deterministic ID generation
---------------------------
Instances are sorted by (floor(centroid_x), floor(centroid_y)) before
numbering. If point coordinates don't change, the same instance gets the
same sequential position within the scene → reproducible IDs.

Output per instance
-------------------
  instance_id   : int        (1-based, scene-local, deterministic)
  building_id   : str        e.g. "DALES_myScene_0001"
  point_indices : np.ndarray (indices into the input building_xyz array)
  centroid_xy   : (float, float)
  method        : str        "grid_bfs" | "dbscan" | "adaptive"
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# pure-numpy DBSCAN — no external deps
HAS_NUMPY_DBSCAN = True  # always available

BUILDING_CLASS = 1


# ── Instance container ─────────────────────────────────────────────────────────

@dataclass
class BuildingInstance:
    instance_id: int                        # 1-based, scene-local
    building_id: str                        # globally-unique prototype ID
    point_indices: np.ndarray               # into the input building_xyz array
    centroid_xy: tuple[float, float]
    method: str = "grid_bfs"               # extraction method used

    @property
    def n_points(self) -> int:
        return int(self.point_indices.shape[0])


# ── Grid BFS (deterministic connected-components) ─────────────────────────────

def _grid_bfs(
    xyz: np.ndarray,
    cell_size: float,
    min_points: int,
) -> list[np.ndarray]:
    """
    2-D grid BFS on XY plane → connected components.

    Returns list of index-arrays (into xyz), each representing one component
    with at least `min_points` points. Not sorted.
    """
    if len(xyz) == 0:
        return []

    x_min = float(xyz[:, 0].min())
    y_min = float(xyz[:, 1].min())
    ci = ((xyz[:, 0] - x_min) / cell_size).astype(np.int32)
    cj = ((xyz[:, 1] - y_min) / cell_size).astype(np.int32)

    cell_pts: dict[tuple[int, int], list[int]] = {}
    for idx in range(len(xyz)):
        key = (int(ci[idx]), int(cj[idx]))
        cell_pts.setdefault(key, []).append(idx)

    occupied = set(cell_pts)
    visited: set[tuple[int, int]] = set()
    components: list[np.ndarray] = []

    for start in occupied:
        if start in visited:
            continue
        cells: list[tuple[int, int]] = []
        q: deque = deque([start])
        visited.add(start)
        while q:
            cx, cy = q.popleft()
            cells.append((cx, cy))
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (cx + di, cy + dj)
                if nb in occupied and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        pts: list[int] = []
        for c in cells:
            pts.extend(cell_pts[c])
        if len(pts) >= min_points:
            components.append(np.array(pts, dtype=np.int64))

    return components


# ── Deterministic sorting and ID generation ────────────────────────────────────

def _centroid_xy(xyz: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    pts = xyz[indices, :2]
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _make_building_id(dataset: str, scene_id: str, sequential: int) -> str:
    """
    Prototype ULPIN-style building ID.

    Format: {DATASET}_{scene_id}_{seq:04d}
    Example: DALES_dales2_500k_0001

    This is a PROTOTYPE identifier, NOT an official ULPIN.
    Official ULPIN requires integration with a government cadastral registry.
    """
    ds = dataset.upper().replace(" ", "_")[:16]
    sc = scene_id.replace(" ", "_")[:32]
    return f"{ds}_{sc}_{sequential:04d}"


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_building_instances(
    building_xyz: np.ndarray,
    scene_id: str,
    source_dataset: str = "unknown",
    cell_size: float = 1.5,
    min_points: int = 30,
    max_instances: int = 500,
) -> list[BuildingInstance]:
    """
    Extract individual building instances from a set of building points.

    This performs Building Instance Extraction — NOT trained instance
    segmentation. The method is deterministic 2-D grid BFS.

    Args:
        building_xyz:    (N, 3) float32 building-class points.
        scene_id:        Scene identifier for ID generation.
        source_dataset:  Dataset name for ID generation.
        cell_size:       BFS grid cell size in metres. Smaller = finer
                         separation; larger = tolerates gaps. Use 1.5 m for
                         terrestrial LiDAR (STPLS3D), 2.0–3.0 m for aerial
                         (DALES) where density is lower.
        min_points:      Minimum points for a component to count as a building.
        max_instances:   Cap on returned instances (largest-first).

    Returns:
        List of BuildingInstance, sorted by centroid XY (deterministic order),
        trimmed to max_instances.
    """
    if len(building_xyz) == 0:
        return []

    raw_components = _grid_bfs(building_xyz, cell_size, min_points)

    if not raw_components:
        return []

    # Compute centroids for sorting
    centroids = [_centroid_xy(building_xyz, idx) for idx in raw_components]

    # Sort by (floor(cx, 1m), floor(cy, 1m)) for determinism
    order = sorted(range(len(raw_components)),
                   key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    # Trim to max_instances (take largest after deterministic sort)
    if len(order) > max_instances:
        # Secondary sort by size descending, then apply deterministic position
        order_by_size = sorted(order, key=lambda i: -len(raw_components[i]))
        order = sorted(order_by_size[:max_instances],
                       key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    instances: list[BuildingInstance] = []
    for seq, raw_idx in enumerate(order, start=1):
        idx_arr = raw_components[raw_idx]
        cx, cy  = centroids[raw_idx]
        bid     = _make_building_id(source_dataset, scene_id, seq)
        instances.append(BuildingInstance(
            instance_id   = seq,
            building_id   = bid,
            point_indices = idx_arr,
            centroid_xy   = (cx, cy),
            method        = "grid_bfs",
        ))

    return instances


# ── Dataset-specific parameter guidance ───────────────────────────────────────

RECOMMENDED_CELL_SIZE: dict[str, float] = {
    "STPLS3D": 1.5,   # terrestrial LiDAR ~100 pts/m²
    "DALES":   2.0,   # aerial LiDAR ~4–10 pts/m²
    "AHN4":    1.5,   # aerial LiDAR ~20–40 pts/m²
    "unknown": 1.5,
}

RECOMMENDED_MIN_POINTS: dict[str, int] = {
    "STPLS3D": 30,
    "DALES":   15,
    "AHN4":    20,
    "unknown": 30,
}

# DBSCAN eps (metres) per dataset — controls the gap allowed between building points
RECOMMENDED_DBSCAN_EPS: dict[str, float] = {
    "STPLS3D": 1.5,   # terrestrial, high density
    "DALES":   2.5,   # aerial, sparser
    "AHN4":    2.0,   # aerial moderate density
    "unknown": 2.0,
}

RECOMMENDED_DBSCAN_MIN_PTS: dict[str, int] = {
    "STPLS3D": 20,
    "DALES":   10,
    "AHN4":    15,
    "unknown": 15,
}


# ── DBSCAN-based instance extraction (via Open3D) ─────────────────────────────

def extract_building_instances_dbscan(
    building_xyz: np.ndarray,
    scene_id: str,
    source_dataset: str = "unknown",
    eps: float | None = None,
    min_points: int | None = None,
    max_instances: int = 500,
) -> list[BuildingInstance]:
    """Extract building instances using 2D DBSCAN clustering via Open3D.

    Works in XY plane only (Z coordinate zeroed out) so vertical separation
    within a building does not cause spurious splits.

    DBSCAN advantages over grid BFS:
    - Adapts to irregular point distributions
    - Handles varying point density
    - Naturally separates buildings with narrow gaps
    - Explicit noise rejection (label=-1)

    DBSCAN limitations:
    - Non-deterministic in edge cases (Open3D uses deterministic implementation)
    - Sensitive to eps parameter — must be tuned per dataset density
    - Slower than grid BFS for large point clouds

    This is NOT trained instance segmentation.

    Args:
        building_xyz:   (N, 3) float32 building-class points.
        scene_id:       Scene identifier for ID generation.
        source_dataset: Dataset name for parameter selection.
        eps:            Cluster radius in metres (auto-selected from dataset if None).
        min_points:     Minimum cluster size (auto-selected if None).
        max_instances:  Cap on returned instances.

    Returns:
        List of BuildingInstance sorted by centroid XY, trimmed to max_instances.
        Returns empty list if Open3D is unavailable.
    """
    if not HAS_OPEN3D:
        print("  [DBSCAN] Open3D not available — falling back to grid BFS")
        return extract_building_instances(
            building_xyz, scene_id, source_dataset,
            max_instances=max_instances,
        )

    if len(building_xyz) == 0:
        return []

    # Use dataset defaults if not specified
    if eps is None:
        eps = RECOMMENDED_DBSCAN_EPS.get(source_dataset, 2.0)
    if min_points is None:
        min_points = RECOMMENDED_DBSCAN_MIN_PTS.get(source_dataset, 15)

    # Project to 2D: set Z=0 so DBSCAN works purely in XY plane
    xy_flat = np.column_stack([building_xyz[:, :2], np.zeros(len(building_xyz), dtype=np.float32)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xy_flat.astype(np.float64))

    labels = np.array(pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
        print_progress=False,
    ))

    # labels == -1 means noise (rejected)
    n_noise = int((labels == -1).sum())
    unique_labels = [l for l in np.unique(labels) if l >= 0]

    if not unique_labels:
        return []

    # Build raw components
    raw_components: list[np.ndarray] = []
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        raw_components.append(idx)

    # Sort and trim identically to grid BFS
    centroids = [_centroid_xy(building_xyz, idx) for idx in raw_components]
    order = sorted(range(len(raw_components)),
                   key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    if len(order) > max_instances:
        order_by_size = sorted(order, key=lambda i: -len(raw_components[i]))
        order = sorted(order_by_size[:max_instances],
                       key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    instances: list[BuildingInstance] = []
    for seq, raw_idx in enumerate(order, start=1):
        idx_arr = raw_components[raw_idx]
        cx, cy  = centroids[raw_idx]
        bid     = _make_building_id(source_dataset, scene_id, seq)
        instances.append(BuildingInstance(
            instance_id   = seq,
            building_id   = bid,
            point_indices = idx_arr,
            centroid_xy   = (cx, cy),
            method        = "dbscan",
        ))

    return instances


# ── Adaptive instance extraction ──────────────────────────────────────────────

def _estimate_point_density(xyz: np.ndarray) -> float:
    """Estimate 2D point density (points per m²) from a building point cloud."""
    if len(xyz) < 4:
        return 0.0
    x_range = float(xyz[:, 0].max() - xyz[:, 0].min())
    y_range = float(xyz[:, 1].max() - xyz[:, 1].min())
    approx_area = max(x_range * y_range, 1.0)
    return len(xyz) / approx_area


def extract_building_instances_adaptive(
    building_xyz: np.ndarray,
    scene_id: str,
    source_dataset: str = "unknown",
    cell_size: float | None = None,
    min_points: int | None = None,
    max_instances: int = 500,
    prefer_dbscan: bool = True,
) -> tuple[list[BuildingInstance], str]:
    """Adaptively select the best instance extraction method.

    Selection logic:
    - DBSCAN (preferred when Open3D available): better separation of adjacent
      buildings, handles irregular density, explicit noise rejection.
    - Grid BFS (fallback / high-density terrestrial): deterministic, fast,
      predictable for very dense point clouds (STPLS3D ~100 pts/m²).

    Args:
        building_xyz:   (N, 3) float32 building points.
        scene_id:       Scene identifier.
        source_dataset: Dataset name.
        cell_size:      Grid BFS cell size (None = auto).
        min_points:     Minimum cluster size (None = auto).
        max_instances:  Cap on instances.
        prefer_dbscan:  If True and Open3D available, use DBSCAN as primary.

    Returns:
        (instances, method_used)
        method_used is "dbscan" or "grid_bfs".
    """
    if prefer_dbscan and HAS_OPEN3D:
        instances = extract_building_instances_dbscan(
            building_xyz, scene_id, source_dataset,
            max_instances=max_instances,
        )
        if instances:
            return instances, "dbscan"
        # DBSCAN returned nothing — fall through to grid BFS
        print("  [adaptive] DBSCAN found 0 instances, retrying with grid BFS")

    # Grid BFS
    if cell_size is None:
        cell_size = RECOMMENDED_CELL_SIZE.get(source_dataset, 1.5)
    if min_points is None:
        min_points = RECOMMENDED_MIN_POINTS.get(source_dataset, 30)

    instances = extract_building_instances(
        building_xyz, scene_id, source_dataset,
        cell_size=cell_size,
        min_points=min_points,
        max_instances=max_instances,
    )
    return instances, "grid_bfs"


# ── Pure-numpy DBSCAN (no sklearn/scipy/open3d) ───────────────────────────────

def _numpy_dbscan_xy(
    xy: np.ndarray,
    eps: float,
    min_pts: int,
) -> np.ndarray:
    """Grid-accelerated DBSCAN on 2D XY points.  Returns label array (-1 = noise).

    Uses a regular cell grid to limit neighbor searches to the 9-cell 3×3
    neighbourhood, making it O(N * density) rather than O(N²).
    """
    n = len(xy)
    labels   = np.full(n, -1, dtype=np.int32)
    visited  = np.zeros(n, dtype=bool)

    # Bucket points into grid cells of size eps
    x_min, y_min = float(xy[:, 0].min()), float(xy[:, 1].min())
    ci = ((xy[:, 0] - x_min) / eps).astype(np.int32)
    cj = ((xy[:, 1] - y_min) / eps).astype(np.int32)
    grid: dict[tuple[int, int], list[int]] = {}
    for idx in range(n):
        k = (int(ci[idx]), int(cj[idx]))
        grid.setdefault(k, []).append(idx)

    def get_neighbors(p_idx: int) -> np.ndarray:
        ci_p, cj_p = int(ci[p_idx]), int(cj[p_idx])
        cands: list[int] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cands.extend(grid.get((ci_p + di, cj_p + dj), []))
        if not cands:
            return np.array([], dtype=np.int64)
        arr = np.array(cands, dtype=np.int64)
        diff = xy[arr] - xy[p_idx]
        mask = (diff[:, 0] ** 2 + diff[:, 1] ** 2) <= eps * eps
        return arr[mask]

    cluster_id = 0
    for p_idx in range(n):
        if visited[p_idx]:
            continue
        visited[p_idx] = True
        neighbors = get_neighbors(p_idx)

        if len(neighbors) < min_pts:
            continue  # noise — stays -1

        labels[p_idx] = cluster_id
        seed: deque[int] = deque(int(x) for x in neighbors)
        while seed:
            q = seed.popleft()
            if labels[q] == -1:
                labels[q] = cluster_id
            if visited[q]:
                continue
            visited[q] = True
            labels[q] = cluster_id
            q_neighbors = get_neighbors(q)
            if len(q_neighbors) >= min_pts:
                for nb in q_neighbors:
                    if not visited[int(nb)]:
                        seed.append(int(nb))
        cluster_id += 1

    return labels


def extract_building_instances_numpy_dbscan(
    building_xyz: np.ndarray,
    scene_id: str,
    source_dataset: str = "unknown",
    eps: float | None = None,
    min_points: int | None = None,
    max_instances: int = 500,
) -> list[BuildingInstance]:
    """Extract building instances using pure-numpy 2D DBSCAN.

    Does NOT require Open3D, sklearn, or scipy — uses only numpy.
    The implementation uses a grid-accelerated neighbour search for efficiency.

    This clusters in the XY plane only (Z is ignored) so vertical separation
    within a single building does not cause spurious splits.

    Args:
        building_xyz:   (N, 3) float32 building-class points.
        scene_id:       Scene identifier for ID generation.
        source_dataset: Dataset name for default parameter selection.
        eps:            Cluster radius in metres (auto-selected if None).
        min_points:     Minimum cluster size (auto-selected if None).
        max_instances:  Cap on returned instances.

    Returns:
        List of BuildingInstance sorted by centroid XY, trimmed to max_instances.
    """
    if len(building_xyz) == 0:
        return []
    if eps is None:
        eps = RECOMMENDED_DBSCAN_EPS.get(source_dataset, 2.0)
    if min_points is None:
        min_points = RECOMMENDED_DBSCAN_MIN_PTS.get(source_dataset, 15)

    xy = building_xyz[:, :2]
    labels = _numpy_dbscan_xy(xy, eps, min_points)

    unique_labels = [l for l in np.unique(labels) if l >= 0]
    if not unique_labels:
        return []

    raw_components = [np.where(labels == lbl)[0] for lbl in unique_labels]
    centroids = [_centroid_xy(building_xyz, idx) for idx in raw_components]
    order = sorted(range(len(raw_components)),
                   key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    if len(order) > max_instances:
        order_by_size = sorted(order, key=lambda i: -len(raw_components[i]))
        order = sorted(order_by_size[:max_instances],
                       key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    instances: list[BuildingInstance] = []
    for seq, raw_idx in enumerate(order, start=1):
        idx_arr = raw_components[raw_idx]
        cx, cy  = centroids[raw_idx]
        bid     = _make_building_id(source_dataset, scene_id, seq)
        instances.append(BuildingInstance(
            instance_id   = seq,
            building_id   = bid,
            point_indices = idx_arr,
            centroid_xy   = (cx, cy),
            method        = "numpy_dbscan",
        ))

    return instances


# ── Voxel connected components (3D grid BFS) ──────────────────────────────────

def extract_building_instances_voxel(
    building_xyz: np.ndarray,
    scene_id: str,
    source_dataset: str = "unknown",
    voxel_size: float | None = None,
    min_points: int | None = None,
    max_instances: int = 500,
) -> list[BuildingInstance]:
    """Extract building instances using 3D voxel grid connected components.

    Unlike the 2D grid BFS, this operates in all three dimensions.  It is
    useful when buildings that share XY proximity are at clearly different
    heights (e.g. a roof terrace touching an adjacent building's facade).

    In practice, for most aerial and terrestrial scans the 2D grid BFS is
    sufficient because building points are laterally separated.  Use this
    as a comparison baseline.

    Args:
        building_xyz:  (N, 3) float32 building-class points.
        scene_id:      Scene identifier.
        source_dataset: Dataset name for defaults.
        voxel_size:    Voxel edge length in metres (None = auto from dataset).
        min_points:    Minimum voxel occupancy to be part of a component.
        max_instances: Cap on returned instances.

    Returns:
        List of BuildingInstance sorted by centroid XY.
    """
    if len(building_xyz) == 0:
        return []
    if voxel_size is None:
        voxel_size = RECOMMENDED_CELL_SIZE.get(source_dataset, 1.5)
    if min_points is None:
        min_points = RECOMMENDED_MIN_POINTS.get(source_dataset, 30)

    x_min = float(building_xyz[:, 0].min())
    y_min = float(building_xyz[:, 1].min())
    z_min = float(building_xyz[:, 2].min())

    ci = ((building_xyz[:, 0] - x_min) / voxel_size).astype(np.int32)
    cj = ((building_xyz[:, 1] - y_min) / voxel_size).astype(np.int32)
    ck = ((building_xyz[:, 2] - z_min) / voxel_size).astype(np.int32)

    voxel_pts: dict[tuple[int, int, int], list[int]] = {}
    for idx in range(len(building_xyz)):
        key = (int(ci[idx]), int(cj[idx]), int(ck[idx]))
        voxel_pts.setdefault(key, []).append(idx)

    occupied = set(voxel_pts)
    visited: set[tuple[int, int, int]] = set()
    components: list[np.ndarray] = []

    for start in occupied:
        if start in visited:
            continue
        voxels: list[tuple[int, int, int]] = []
        q: deque = deque([start])
        visited.add(start)
        while q:
            vi, vj, vk = q.popleft()
            voxels.append((vi, vj, vk))
            for di, dj, dk in (
                (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
            ):
                nb = (vi + di, vj + dj, vk + dk)
                if nb in occupied and nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        pts: list[int] = []
        for v in voxels:
            pts.extend(voxel_pts[v])
        if len(pts) >= min_points:
            components.append(np.array(pts, dtype=np.int64))

    if not components:
        return []

    centroids = [_centroid_xy(building_xyz, idx) for idx in components]
    order = sorted(range(len(components)),
                   key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    if len(order) > max_instances:
        order_by_size = sorted(order, key=lambda i: -len(components[i]))
        order = sorted(order_by_size[:max_instances],
                       key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    instances: list[BuildingInstance] = []
    for seq, raw_idx in enumerate(order, start=1):
        idx_arr = components[raw_idx]
        cx, cy  = centroids[raw_idx]
        bid     = _make_building_id(source_dataset, scene_id, seq)
        instances.append(BuildingInstance(
            instance_id   = seq,
            building_id   = bid,
            point_indices = idx_arr,
            centroid_xy   = (cx, cy),
            method        = "voxel_cc",
        ))

    return instances


# ── Height-aware BFS: XY grid BFS + Z gap splitting ───────────────────────────

def _z_gap_split(
    idx_arr: np.ndarray,
    xyz: np.ndarray,
    gap_m: float = 2.5,
    bin_w: float = 0.5,
) -> list[np.ndarray]:
    """Split a cluster at the largest Z-gap in the middle 30-80% of height range.

    Returns [idx_arr] unchanged if no significant gap is found.
    """
    pts = xyz[idx_arr]
    z = pts[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    z_range = z_max - z_min

    if z_range < gap_m * 2:
        return [idx_arr]

    n_bins = max(4, int(z_range / bin_w))
    counts, edges = np.histogram(z, bins=n_bins, range=(z_min, z_max))

    lo_idx = int(n_bins * 0.3)
    hi_idx = int(n_bins * 0.8)
    mid_counts = counts[lo_idx:hi_idx]

    if len(mid_counts) == 0 or mid_counts.min() > 0:
        return [idx_arr]

    # Find the longest run of zero-count bins in the middle zone
    best_run, run_start, zero_start, zero_end = 0, -1, -1, -1
    for i, c in enumerate(mid_counts):
        if c == 0:
            if run_start == -1:
                run_start = i
        else:
            if run_start != -1 and (i - run_start) > best_run:
                best_run = i - run_start
                zero_start = run_start + lo_idx
                zero_end   = i + lo_idx
                run_start  = -1

    if best_run < 2:
        return [idx_arr]

    gap_z = (edges[zero_start] + edges[zero_end]) / 2.0
    mask_lo = z <= gap_z
    mask_hi = z > gap_z

    if mask_lo.sum() < 10 or mask_hi.sum() < 10:
        return [idx_arr]

    return [idx_arr[mask_lo], idx_arr[mask_hi]]


def extract_building_instances_height_aware(
    building_xyz: np.ndarray,
    scene_id: str,
    source_dataset: str = "unknown",
    cell_size: float | None = None,
    min_points: int | None = None,
    z_gap_m: float = 2.5,
    max_instances: int = 500,
) -> list[BuildingInstance]:
    """Height-aware instance extraction: 2D grid BFS followed by Z-gap splitting.

    This combines the speed of grid BFS with robustness to merged clusters
    that span two vertically distinct buildings (e.g. a building bridge).

    Algorithm:
      1. Run 2D grid BFS on XY coordinates (same as extract_building_instances)
      2. For each cluster, inspect the Z histogram
      3. If a significant gap exists in the middle height range, split the cluster

    This is equivalent to the height-discontinuity splitting in pipeline_v3.py
    but integrated into instance extraction instead of being a post-processing step.

    Returns:
        List of BuildingInstance, method tag "height_aware_bfs".
    """
    if len(building_xyz) == 0:
        return []
    if cell_size is None:
        cell_size = RECOMMENDED_CELL_SIZE.get(source_dataset, 1.5)
    if min_points is None:
        min_points = RECOMMENDED_MIN_POINTS.get(source_dataset, 30)

    # Step 1: 2D BFS components
    raw_components = _grid_bfs(building_xyz, cell_size, min_points)

    # Step 2: Z-gap split each component
    split_components: list[np.ndarray] = []
    for comp in raw_components:
        split_components.extend(_z_gap_split(comp, building_xyz, gap_m=z_gap_m))

    # Filter: only keep components with enough points after splitting
    split_components = [c for c in split_components if len(c) >= min_points]

    if not split_components:
        return []

    centroids = [_centroid_xy(building_xyz, idx) for idx in split_components]
    order = sorted(range(len(split_components)),
                   key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    if len(order) > max_instances:
        order_by_size = sorted(order, key=lambda i: -len(split_components[i]))
        order = sorted(order_by_size[:max_instances],
                       key=lambda i: (round(centroids[i][0]), round(centroids[i][1])))

    instances: list[BuildingInstance] = []
    for seq, raw_idx in enumerate(order, start=1):
        idx_arr = split_components[raw_idx]
        cx, cy  = centroids[raw_idx]
        bid     = _make_building_id(source_dataset, scene_id, seq)
        instances.append(BuildingInstance(
            instance_id   = seq,
            building_id   = bid,
            point_indices = idx_arr,
            centroid_xy   = (cx, cy),
            method        = "height_aware_bfs",
        ))

    return instances


# ── Evaluation utilities ──────────────────────────────────────────────────────

def evaluate_instance_extraction(
    instances: list[BuildingInstance],
    building_xyz: np.ndarray,
    method: str = "unknown",
) -> dict:
    """Report statistics for an instance extraction result.

    Reports:
        n_instances:        Total number of building instances found
        n_noise_pct:        Percentage of input points not assigned to any instance
        mean_pts:           Mean points per instance
        std_pts:            Std deviation of points per instance
        min_pts:            Smallest instance (potential over-segmentation)
        max_pts:            Largest instance (potential under-segmentation)
        size_ratio:         max_pts / min_pts (high = likely under/over segmentation)
        method:             Extraction method used

    Args:
        instances:      List of BuildingInstance
        building_xyz:   Original building point cloud
        method:         Method name for the report

    Returns:
        dict of statistics
    """
    n_total = len(building_xyz)
    if not instances:
        return {
            "method": method,
            "n_instances": 0,
            "n_assigned_pts": 0,
            "n_noise_pts": n_total,
            "noise_pct": 100.0,
            "mean_pts": 0,
            "std_pts": 0,
            "min_pts": 0,
            "max_pts": 0,
            "size_ratio": 0.0,
        }

    sizes = [inst.n_points for inst in instances]
    n_assigned = sum(sizes)
    n_noise = n_total - n_assigned

    return {
        "method": method,
        "n_instances": len(instances),
        "n_assigned_pts": n_assigned,
        "n_noise_pts": max(0, n_noise),
        "noise_pct": round(100.0 * max(0, n_noise) / n_total, 1) if n_total > 0 else 0.0,
        "mean_pts": round(float(np.mean(sizes)), 1),
        "std_pts": round(float(np.std(sizes)), 1),
        "min_pts": int(min(sizes)),
        "max_pts": int(max(sizes)),
        "size_ratio": round(max(sizes) / max(min(sizes), 1), 1),
    }


def compare_extraction_methods(
    building_xyz: np.ndarray,
    scene_id: str,
    source_dataset: str = "unknown",
) -> dict:
    """Run all available instance extraction methods side-by-side.

    Methods compared:
      grid_bfs_default  — 2D grid BFS at dataset-recommended cell size
      grid_bfs_fine     — 2D grid BFS at half cell size (finer)
      grid_bfs_coarse   — 2D grid BFS at double cell size (coarser)
      numpy_dbscan      — Pure-numpy 2D DBSCAN (no external deps)
      height_aware_bfs  — 2D grid BFS + Z-gap splitting
      voxel_cc          — 3D voxel connected components
      open3d_dbscan     — Open3D DBSCAN (if available)

    Useful for validating which method suits a given dataset.
    Does NOT modify any state.
    """
    import time
    results: dict = {}

    cs   = RECOMMENDED_CELL_SIZE.get(source_dataset, 1.5)
    mp   = RECOMMENDED_MIN_POINTS.get(source_dataset, 30)
    eps  = RECOMMENDED_DBSCAN_EPS.get(source_dataset, 2.0)
    mpts = RECOMMENDED_DBSCAN_MIN_PTS.get(source_dataset, 15)

    # ── Grid BFS variants ──────────────────────────────────────────────────────
    for key, size in [("grid_bfs_default", cs),
                      ("grid_bfs_fine",    cs * 0.5),
                      ("grid_bfs_coarse",  cs * 2.0)]:
        t0 = time.perf_counter()
        inst = extract_building_instances(
            building_xyz, scene_id, source_dataset,
            cell_size=size, min_points=mp,
        )
        elapsed = time.perf_counter() - t0
        results[key] = evaluate_instance_extraction(inst, building_xyz, key)
        results[key]["cell_size_m"] = size
        results[key]["min_points"]  = mp
        results[key]["time_s"]      = round(elapsed, 4)

    # ── Pure-numpy DBSCAN ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    nd_inst = extract_building_instances_numpy_dbscan(
        building_xyz, scene_id, source_dataset, eps=eps, min_points=mpts,
    )
    elapsed = time.perf_counter() - t0
    results["numpy_dbscan"] = evaluate_instance_extraction(nd_inst, building_xyz, "numpy_dbscan")
    results["numpy_dbscan"]["eps_m"]      = eps
    results["numpy_dbscan"]["min_points"] = mpts
    results["numpy_dbscan"]["time_s"]     = round(elapsed, 4)

    # ── Height-aware BFS ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    ha_inst = extract_building_instances_height_aware(
        building_xyz, scene_id, source_dataset, cell_size=cs, min_points=mp,
    )
    elapsed = time.perf_counter() - t0
    results["height_aware_bfs"] = evaluate_instance_extraction(ha_inst, building_xyz, "height_aware_bfs")
    results["height_aware_bfs"]["cell_size_m"] = cs
    results["height_aware_bfs"]["min_points"]  = mp
    results["height_aware_bfs"]["time_s"]      = round(elapsed, 4)

    # ── Voxel connected components ─────────────────────────────────────────────
    t0 = time.perf_counter()
    vox_inst = extract_building_instances_voxel(
        building_xyz, scene_id, source_dataset, voxel_size=cs, min_points=mp,
    )
    elapsed = time.perf_counter() - t0
    results["voxel_cc"] = evaluate_instance_extraction(vox_inst, building_xyz, "voxel_cc")
    results["voxel_cc"]["voxel_size_m"] = cs
    results["voxel_cc"]["min_points"]   = mp
    results["voxel_cc"]["time_s"]       = round(elapsed, 4)

    # ── Open3D DBSCAN (when available) ────────────────────────────────────────
    if HAS_OPEN3D:
        t0 = time.perf_counter()
        o3d_inst = extract_building_instances_dbscan(
            building_xyz, scene_id, source_dataset,
        )
        elapsed = time.perf_counter() - t0
        results["open3d_dbscan"] = evaluate_instance_extraction(
            o3d_inst, building_xyz, "open3d_dbscan"
        )
        results["open3d_dbscan"]["eps_m"]      = eps
        results["open3d_dbscan"]["min_points"] = mpts
        results["open3d_dbscan"]["time_s"]     = round(elapsed, 4)
    else:
        results["open3d_dbscan"] = {"available": False, "reason": "open3d not installed"}

    return results

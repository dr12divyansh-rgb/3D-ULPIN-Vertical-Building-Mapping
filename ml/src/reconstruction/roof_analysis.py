"""Improved roof analysis using Open3D RANSAC plane fitting.

Replaces the simple Z-std heuristic in pipeline_v2 with actual plane detection.
Uses Open3D's segment_plane (RANSAC) to fit dominant roof planes.

Method: LiDAR-derived RANSAC plane fitting
  1. Isolate roof-zone points (above eave_z threshold)
  2. Fit a dominant plane via RANSAC
  3. Classify by plane slope angle
  4. Detect secondary planes (hip / multi-pitch)
  5. Estimate ridge direction from principal component

This is a geometric heuristic, NOT a trained classification model.
Label roof_type_source as "LiDAR-derived-RANSAC-heuristic" in all outputs.

Roof types produced:
  "flat"         — slope < FLAT_MAX_DEG (5°)
  "gable"        — dominant slope, PCA suggests single ridge axis
  "hip"          — multiple inclined planes detected
  "shed"         — single inclined plane
  "multi-pitch"  — 3+ significant planes
  "unknown"      — insufficient roof points or no plane fit

Falls back gracefully to the original Z-std heuristic if Open3D is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

# ── Thresholds ────────────────────────────────────────────────────────────────
FLAT_MAX_DEG     = 5.0    # Planes with slope ≤ this are "flat"
PITCHED_MIN_DEG  = 12.0   # Planes steeper than this count as pitched
WALL_MAX_DEG     = 65.0   # Planes steeper than this are rejected as wall artifacts
MAX_RANSAC_ITER  = 200    # RANSAC iterations for plane fitting
PLANE_DIST_THR   = 0.25   # RANSAC inlier distance threshold (metres)
MIN_ROOF_POINTS  = 8      # Minimum points in roof zone to attempt plane fitting
INLIER_RATIO_MIN = 0.20   # Minimum fraction of roof points on dominant plane
HIP_PLANE_RATIO  = 0.15   # Fraction of remaining points to count as a second plane


@dataclass
class RoofPlane:
    """One detected roof plane."""
    normal: tuple[float, float, float]   # unit normal [a, b, c]
    d: float                             # plane offset (ax+by+cz+d=0)
    slope_deg: float                     # angle from horizontal (0=flat, 90=wall)
    inlier_count: int                    # number of points on this plane
    inlier_fraction: float               # fraction of roof-zone points on this plane


@dataclass
class RoofAnalysisResult:
    """Full roof analysis output."""
    roof_type: str                       # "flat"|"gable"|"hip"|"shed"|"multi-pitch"|"unknown"
    is_pitched: bool
    slope_deg: float | None             # dominant slope in degrees
    ridge_axis: str | None              # "x" | "y" — estimated ridge direction
    n_planes: int                       # number of significant planes detected
    planes: list[RoofPlane]
    method: str                         # "RANSAC-open3d" or "Zstd-fallback"
    roof_pts_used: int                  # number of points in roof zone
    notes: str = ""


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_roof(
    building_xyz: np.ndarray,
    ground_z: float,
    roof_z: float,
    eave_z: float,
) -> RoofAnalysisResult:
    """Analyse the roof of a building point cloud.

    Args:
        building_xyz: (N, 3) float32 building points (full building, not just roof).
        ground_z:     Ground elevation (5th-percentile Z).
        roof_z:       Roof peak elevation (99th-percentile Z).
        eave_z:       Eave/wall-plate elevation (80th-percentile clamped).

    Returns:
        RoofAnalysisResult with roof type, slope, planes, and method.
    """
    height = roof_z - ground_z
    if height < 1.0 or len(building_xyz) < MIN_ROOF_POINTS:
        return RoofAnalysisResult(
            roof_type="unknown", is_pitched=False, slope_deg=None,
            ridge_axis=None, n_planes=0, planes=[],
            method="insufficient_data",
            roof_pts_used=0,
            notes=f"Height {height:.1f}m or too few points ({len(building_xyz)})",
        )

    # Extract roof-zone points
    roof_pts = building_xyz[building_xyz[:, 2] > eave_z]

    if len(roof_pts) < MIN_ROOF_POINTS:
        # Fall back to Z-std heuristic
        return _zstd_fallback(building_xyz, ground_z, roof_z, eave_z)

    if HAS_OPEN3D:
        return _analyze_with_ransac(roof_pts, building_xyz, ground_z, roof_z, eave_z)
    else:
        return _zstd_fallback(building_xyz, ground_z, roof_z, eave_z)


# ── RANSAC-based analysis ──────────────────────────────────────────────────────

def _analyze_with_ransac(
    roof_pts: np.ndarray,
    full_pts: np.ndarray,
    ground_z: float,
    roof_z: float,
    eave_z: float,
) -> RoofAnalysisResult:
    """Fit 1–3 planes to roof-zone points via Open3D RANSAC."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(roof_pts.astype(np.float64))

    planes: list[RoofPlane] = []
    remaining_pts = roof_pts.copy()
    n_roof_total = len(roof_pts)

    for _round in range(3):  # Fit at most 3 planes
        if len(remaining_pts) < MIN_ROOF_POINTS:
            break

        pcd_rem = o3d.geometry.PointCloud()
        pcd_rem.points = o3d.utility.Vector3dVector(remaining_pts.astype(np.float64))

        try:
            model, inliers = pcd_rem.segment_plane(
                distance_threshold=PLANE_DIST_THR,
                ransac_n=3,
                num_iterations=MAX_RANSAC_ITER,
            )
        except (RuntimeError, Exception):
            break

        a, b, c, d = model
        # Normalise normal vector
        norm_len = math.sqrt(a*a + b*b + c*c)
        if norm_len < 1e-9:
            break
        nx, ny, nz = a / norm_len, b / norm_len, c / norm_len

        # Slope = angle between plane normal and vertical (0, 0, 1)
        cos_theta = abs(nz)
        slope_deg = math.degrees(math.acos(min(1.0, cos_theta)))

        n_inliers = len(inliers)
        inlier_frac = n_inliers / n_roof_total

        # Only accept planes with sufficient inliers
        if inlier_frac < (INLIER_RATIO_MIN if _round == 0 else HIP_PLANE_RATIO):
            break

        planes.append(RoofPlane(
            normal=(float(nx), float(ny), float(nz)),
            d=float(d),
            slope_deg=float(slope_deg),
            inlier_count=n_inliers,
            inlier_fraction=float(inlier_frac),
        ))

        # Remove inliers from remaining points for next iteration
        remaining_mask = np.ones(len(remaining_pts), dtype=bool)
        remaining_mask[inliers] = False
        remaining_pts = remaining_pts[remaining_mask]

    if not planes:
        # RANSAC found nothing — fall back
        return _zstd_fallback(full_pts, ground_z, roof_z, eave_z)

    return _classify_from_planes(planes, full_pts, n_roof_total)


def _classify_from_planes(
    planes: list[RoofPlane],
    full_pts: np.ndarray,
    n_roof_total: int,
) -> RoofAnalysisResult:
    """Derive roof_type from detected planes.

    Wall-artifact filter: planes with slope > WALL_MAX_DEG (65°) are
    treated as wall detections, not roof planes. RANSAC can latch onto
    a vertical wall face when the roof zone contains too few points
    (happens for short or very dense buildings). Such planes are removed
    before classification; if all planes are wall-artifacts we fall back
    to a Z-std heuristic on the full building.
    """
    # Remove wall-artifact planes (slope > 65°)
    roof_planes = [p for p in planes if p.slope_deg <= WALL_MAX_DEG]
    if not roof_planes:
        # All planes are near-vertical (wall artifacts) — use Z-std
        return _zstd_fallback(full_pts, float(full_pts[:, 2].min()),
                              float(full_pts[:, 2].max()), float(np.percentile(full_pts[:, 2], 80)))

    dominant  = roof_planes[0]
    dom_slope = dominant.slope_deg

    is_flat    = dom_slope <= FLAT_MAX_DEG
    is_pitched = dom_slope >= PITCHED_MIN_DEG

    # Count only non-wall pitched planes
    n_pitched = sum(1 for p in roof_planes if p.slope_deg >= PITCHED_MIN_DEG)
    ridge_axis: str | None = None

    if is_flat or not is_pitched:
        roof_type  = "flat"
        is_pitched = False
    elif n_pitched == 1:
        # Single inclined plane — shed (one big face) or start of gable
        roof_type  = "shed" if dominant.inlier_fraction > 0.85 else "gable"
        ridge_axis = _estimate_ridge_axis(full_pts)
    elif n_pitched == 2:
        roof_type  = "gable"
        ridge_axis = _estimate_ridge_axis(full_pts)
    else:  # n_pitched >= 3
        roof_type  = "hip"
        ridge_axis = _estimate_ridge_axis(full_pts)

    notes = ""
    if len(roof_planes) < len(planes):
        notes = f"Removed {len(planes)-len(roof_planes)} near-vertical wall plane(s)."
    if len(roof_planes) == 1 and dominant.inlier_fraction < 0.35:
        notes += f" Low inlier fraction ({dominant.inlier_fraction:.2f}); uncertain."

    return RoofAnalysisResult(
        roof_type=roof_type,
        is_pitched=is_pitched,
        slope_deg=round(dom_slope, 1),
        ridge_axis=ridge_axis,
        n_planes=len(roof_planes),
        planes=roof_planes,
        method="RANSAC-open3d",
        roof_pts_used=n_roof_total,
        notes=notes.strip(),
    )


# ── Z-std fallback (original heuristic) ──────────────────────────────────────

def _zstd_fallback(
    xyz: np.ndarray,
    ground_z: float,
    roof_z: float,
    eave_z: float,
) -> RoofAnalysisResult:
    """Multi-metric Z heuristic roof classification.

    Used when Open3D is unavailable, RANSAC fails, or all detected planes
    are near-vertical wall artifacts.

    Metrics (all computed in roof zone = top 25% of building height):
      1. Z-std (original)        : high std → pitched
      2. Z-range in roof zone    : large range relative to building height → pitched
      3. Z-skewness in roof zone : large positive skew → dome/shed; negative → gable
      4. Z-gradient (XY PCA)     : strong Z-gradient along principal axis → gable/shed

    A building is classified as pitched if at least 2 of the 4 signals agree.
    """
    roof_pts = xyz[xyz[:, 2] > eave_z]
    height   = roof_z - ground_z
    is_pitched = False
    ridge_axis: str | None = None
    notes_parts: list[str] = []

    if len(roof_pts) < 8 or height < 1.0:
        notes_parts.append("Insufficient data for roof analysis.")
        return RoofAnalysisResult(
            roof_type="flat", is_pitched=False, slope_deg=None, ridge_axis=None,
            n_planes=0, planes=[], method="Zstd-fallback",
            roof_pts_used=len(roof_pts),
            notes=" ".join(notes_parts),
        )

    z_roof = roof_pts[:, 2]
    z_std   = float(np.std(z_roof))
    z_range = float(z_roof.max() - z_roof.min())

    # Signal 1: Z-std in roof zone (relative to building height)
    s1 = z_std > max(0.4, height * 0.05)

    # Signal 2: Z-range in roof zone (relative to building height)
    # A flat roof → z_range ≈ 0; gable → z_range up to 30-50% of height
    s2 = z_range > max(0.6, height * 0.08)

    # Signal 3: Z-std relative to ground-to-eave Z-std (compares roof zone vs wall zone)
    wall_pts = xyz[(xyz[:, 2] > ground_z) & (xyz[:, 2] <= eave_z)]
    if len(wall_pts) >= 8:
        wall_z_std = float(np.std(wall_pts[:, 2]))
        s3 = z_std > wall_z_std * 1.5
        notes_parts.append(f"wall_zstd={wall_z_std:.2f} roof_zstd={z_std:.2f}")
    else:
        s3 = s1  # fallback: trust z_std signal

    # Signal 4: Z-gradient along the XY principal axis
    if len(roof_pts) >= 12:
        xy  = roof_pts[:, :2] - roof_pts[:, :2].mean(axis=0)
        cov = xy.T @ xy
        try:
            _, vecs = np.linalg.eigh(cov)
            major_xy = vecs[:, -1]
            proj   = roof_pts[:, :2] @ major_xy   # 1D projection along principal axis
            # Bin and compute Z-mean per XY slice
            n_bins = max(3, min(8, len(roof_pts) // 5))
            proj_bins = np.array_split(np.argsort(proj), n_bins)
            bin_z_means = [float(z_roof[b].mean()) for b in proj_bins if len(b) > 0]
            z_slope = max(bin_z_means) - min(bin_z_means) if bin_z_means else 0.0
            s4 = z_slope > max(0.5, height * 0.06)
        except np.linalg.LinAlgError:
            s4 = s1
    else:
        s4 = False

    # At least 2 of 4 signals must agree → pitched
    n_signals = sum([s1, s2, s3, s4])
    is_pitched = n_signals >= 2

    notes_parts.append(f"signals={int(s1)}{int(s2)}{int(s3)}{int(s4)}({n_signals}/4) "
                       f"z_std={z_std:.2f} z_range={z_range:.2f}")

    if is_pitched:
        ridge_axis = _estimate_ridge_axis(xyz)

    roof_type = "gable" if is_pitched else "flat"
    return RoofAnalysisResult(
        roof_type=roof_type,
        is_pitched=is_pitched,
        slope_deg=None,
        ridge_axis=ridge_axis,
        n_planes=1 if is_pitched else 0,
        planes=[],
        method="Zstd-fallback",
        roof_pts_used=len(roof_pts),
        notes="; ".join(notes_parts),
    )


# ── Ridge axis estimation ─────────────────────────────────────────────────────

def _estimate_ridge_axis(xyz: np.ndarray) -> str | None:
    """Estimate ridge axis direction from point cloud PCA (XY plane)."""
    if len(xyz) < 4:
        return None
    xy  = xyz[:, :2] - xyz[:, :2].mean(axis=0)
    cov = xy.T @ xy
    try:
        _, vecs = np.linalg.eigh(cov)
        major = vecs[:, -1]   # eigenvector with largest eigenvalue
        return "x" if abs(major[0]) >= abs(major[1]) else "y"
    except np.linalg.LinAlgError:
        return None


# ── Schema export ─────────────────────────────────────────────────────────────

def roof_result_to_dict(result: RoofAnalysisResult) -> dict:
    """Convert RoofAnalysisResult to a serialisable dict for buildings.json."""
    return {
        "roof_type":          result.roof_type,
        "roof_is_pitched":    result.is_pitched,
        "roof_slope_deg":     result.slope_deg,
        "roof_ridge_axis":    result.ridge_axis,
        "roof_n_planes":      result.n_planes,
        "roof_type_source":   f"LiDAR-derived-{result.method}",
        "roof_type_note": (
            f"Method: {result.method}. "
            "This is a geometric heuristic on LiDAR geometry, NOT a trained classifier. "
            + (result.notes or "")
        ).strip(),
    }

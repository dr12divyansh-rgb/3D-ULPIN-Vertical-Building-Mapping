"""Generic data I/O interface.

These loaders/savers are intentionally generic so the same reconstruction
pipeline can later consume *real* drone LiDAR / survey geometry without
rewriting call sites. The synthetic generator is just one data source.
"""

from __future__ import annotations

from pathlib import Path

from . import ply_io
from .mesh import SurfaceMesh


def load_point_cloud(path) -> dict:
    """Load a point cloud (PLY) into a dict of property name -> list of values."""
    data, _count = ply_io.read_pointcloud(path)
    return data


def save_point_cloud(path, xyz, class_id, building_id, intensity=None, return_number=None):
    """Save a point cloud (PLY) using the fixed schema."""
    return ply_io.write_pointcloud(path, xyz, class_id, building_id,
                                   intensity, return_number)


def load_surface_mesh(path) -> SurfaceMesh:
    """Load a group-tagged OBJ mesh into a structured ``SurfaceMesh``."""
    return SurfaceMesh.from_obj(path)


def save_surface_mesh(path, mesh: SurfaceMesh) -> Path:
    """Save a ``SurfaceMesh`` to a group-tagged OBJ file."""
    return mesh.to_obj(path)

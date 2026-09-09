"""Building surface semantics (pure Python, no bpy).

Defines the three structured surface types used by LOD1/LOD2 building
geometry. Integer codes are used for Blender's bmesh face layers; the string
names are the canonical public identifiers (future CityGML-style export can
map directly onto these names).
"""

from __future__ import annotations

WALL = 0
ROOF = 1
GROUND = 2

# int code -> canonical surface type name
SURFACE_CODES = {
    WALL: "WallSurface",
    ROOF: "RoofSurface",
    GROUND: "GroundSurface",
}

# canonical name -> int code
SURFACE_NAMES = {v: k for k, v in SURFACE_CODES.items()}

# Blender bmesh face layer name used to tag face semantics during building.
SEMANTIC_LAYER_NAME = "semantic"

# All surface type names, in a stable order (for reporting).
SURFACE_TYPE_NAMES = ("WallSurface", "RoofSurface", "GroundSurface")

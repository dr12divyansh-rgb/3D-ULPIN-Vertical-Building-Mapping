"""Semantic class definitions shared across the entire pipeline.

The class integer ids are fixed so that datasets produced by different runs
(and by different tools) remain mutually interpretable.
"""

from __future__ import annotations

# Fixed ordering. The integer id is the index in this list.
CLASS_NAMES = ("ground", "building", "vegetation", "road", "vehicle", "other")

CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

# Approximate RGB colors used both for the debug viewer and stored in the PLY.
CLASS_COLORS = {
    "ground": (120, 120, 120),
    "building": (220, 110, 55),
    "vegetation": (45, 160, 80),
    "road": (40, 40, 40),
    "vehicle": (70, 130, 200),
    "other": (200, 200, 40),
}

# Sentinal used in the point cloud for points that do not belong to a building.
NON_BUILDING_ID = -1


def class_id(name: str) -> int:
    return CLASS_IDS[name]


def class_name(cid: int) -> str:
    return CLASS_NAMES[int(cid)]


def class_color(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS[name]


def is_valid_class_id(cid: int) -> bool:
    return 0 <= int(cid) < len(CLASS_NAMES)

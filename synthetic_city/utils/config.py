"""Configuration loading + validation (venv side; standard library + PyYAML)."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    validate_config(cfg)
    return cfg


def _require(cfg, *keys):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise ValueError(f"config missing key: {'/'.join(keys)}")
        node = node[k]
    return node


def _positive_int(cfg, *keys):
    v = _require(cfg, *keys)
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise ValueError(f"config key {'/'.join(keys)} must be a non-negative int, got {v!r}")
    return v


def _positive_float(cfg, *keys):
    v = _require(cfg, *keys)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
        raise ValueError(f"config key {'/'.join(keys)} must be a positive number, got {v!r}")
    return v


def validate_config(cfg: dict) -> None:
    _positive_int(cfg, "dataset", "seed")
    _positive_int(cfg, "dataset", "train_scenes")
    _positive_int(cfg, "dataset", "validation_scenes")
    _positive_int(cfg, "dataset", "test_scenes")
    if cfg["dataset"]["train_scenes"] + cfg["dataset"]["validation_scenes"] + cfg["dataset"]["test_scenes"] < 1:
        raise ValueError("dataset must contain at least one scene")

    for key in ("width_range", "length_range", "floor_height_range", "rotation_range",
                "roof_slope_range"):
        rng = _require(cfg, "building", key)
        if not isinstance(rng, (list, tuple)) or len(rng) != 2 or rng[0] > rng[1]:
            raise ValueError(f"building.{key} must be [min, max], got {rng!r}")

    floors = _require(cfg, "building", "floors_range")
    if not isinstance(floors, (list, tuple)) or len(floors) != 2 or int(floors[0]) < 1 or int(floors[1]) < int(floors[0]):
        raise ValueError(f"building.floors_range must be [min, max] with min >= 1, got {floors!r}")

    _positive_float(cfg, "lidar", "base_points_per_sqm")

    scene_mode = _require(cfg, "scene", "mode")
    if scene_mode not in ("urban", "isolated"):
        raise ValueError(f"scene.mode must be 'urban' or 'isolated', got {scene_mode!r}")

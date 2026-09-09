"""Shared ML utilities: config loading, seeding, device selection, experiment dirs."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_cfg: str):
    if device_cfg in ("cpu", "cuda"):
        return torch.device(device_cfg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def new_experiment_dir(results_root) -> Path:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("_")[-1]) for p in root.glob("experiment_*")
                if p.is_dir() and p.name.split("_")[-1].isdigit()]
    next_id = max(existing) + 1 if existing else 1
    exp_dir = root / f"experiment_{next_id:03d}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir

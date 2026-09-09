"""Deterministic scene manifest + train/validation/test split.

The split happens at the *scene* level: each scene is an independent random
draw, so buildings can never leak between splits. Test scenes additionally use
a distinct seed stream (``dataset.test_seed_offset``).
"""

from __future__ import annotations

import random


def build_manifest(cfg: dict) -> list[dict]:
    d = cfg["dataset"]
    base_seed = int(d["seed"])
    n_train = int(d["train_scenes"])
    n_val = int(d["validation_scenes"])
    n_test = int(d["test_scenes"])
    total = n_train + n_val + n_test

    rng = random.Random(base_seed)

    # Shuffle the split assignment so which scene slots land in which split is
    # random but reproducible.
    splits = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    rng.shuffle(splits)

    b = cfg["building"]
    nmin, nmax = cfg["scene"]["buildings_per_scene_range"]

    manifest = []
    for scene_index, split in enumerate(splits):
        if split == "test":
            seed = base_seed + int(d["test_seed_offset"]) + scene_index
        else:
            seed = base_seed + scene_index
        manifest.append({
            "scene_index": scene_index,
            "scene_id": f"scene_{scene_index + 1:05d}",
            "split": split,
            "seed": seed,
            "mode": cfg["scene"]["mode"],
            "num_buildings": rng.randint(int(nmin), int(nmax)),
            "building_id_base": scene_index * 10000 + 1,
        })

    # Optional isolated single-building debug scenes (excluded from splits).
    for i in range(int(cfg["scene"].get("extra_isolated_scenes", 0))):
        manifest.append({
            "scene_index": total + i,
            "scene_id": f"debug_scene_{i + 1:05d}",
            "split": "debug",
            "seed": base_seed + 900000 + i,
            "mode": "isolated",
            "num_buildings": 1,
            "building_id_base": 900000 + i * 10000 + 1,
        })

    return manifest


def split_counts(manifest: list[dict]) -> dict:
    counts = {}
    for s in manifest:
        counts[s["split"]] = counts.get(s["split"], 0) + 1
    return counts

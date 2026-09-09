#!/usr/bin/env python3
"""Build ``viewer/data/index.json`` for the temporary HTML test viewer.

Scans the existing Step 1/2 dataset (``ml/data/synthetic``) and the Step 3
prediction outputs (``ml/results/experiment_*/predictions``), and writes a
small JSON index the browser uses to populate the scene/building selectors and
load the correct artifacts.

This only *indexes* existing files; it never modifies the dataset or the ML
pipeline.

Usage:
    python viewer/build_index.py [--out viewer/data/index.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml" / "data" / "synthetic"
RESULTS = ROOT / "ml" / "results"
SPLITS = ("train", "validation", "test")


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def find_predictions() -> dict:
    """scene_id -> {experiment, path, metrics_path, metrics_by_building}."""
    out: dict = {}
    experiments = sorted((p for p in RESULTS.glob("experiment_*") if p.is_dir()),
                         key=lambda p: p.name)
    for exp in reversed(experiments):  # most recent first
        metrics_path = exp / "per_building_metrics.json"
        metrics_map = {}
        if metrics_path.exists():
            try:
                entries = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics_map = {int(e["building_id"]): e for e in entries}
            except Exception:
                metrics_map = {}
        pred_root = exp / "predictions"
        if not pred_root.is_dir():
            continue
        for scene_dir in sorted(pred_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            ply = scene_dir / "prediction.ply"
            if not ply.exists():
                continue
            sid = scene_dir.name
            if sid in out:
                continue  # already claimed by a more recent experiment
            out[sid] = {
                "experiment": exp.name,
                "path": _rel(ply),
                "metrics_path": _rel(metrics_path) if metrics_path.exists() else None,
                "metrics_by_building": metrics_map,
            }
    return out


def build_index() -> dict:
    predictions = find_predictions()
    scenes = []
    for split in SPLITS:
        split_dir = DATA / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            building_json = scene_dir / "ground_truth" / "building.json"
            if not building_json.exists():
                continue
            buildings = json.loads(building_json.read_text(encoding="utf-8"))["buildings"]
            blist = []
            for b in buildings:
                bid = int(b["building_id"])
                blist.append({
                    "id": bid,
                    "lod1_path": _rel(scene_dir / "lod1" / f"building_{bid:06d}.obj"),
                    "lod2_path": _rel(scene_dir / "ground_truth" / "lod2" / f"building_{bid:06d}.obj"),
                    "metadata": b,
                })
            scene = {
                "id": scene_dir.name,
                "split": split,
                "lidar_path": _rel(scene_dir / "lidar" / "pointcloud.ply"),
                "buildings": blist,
            }
            if scene_dir.name in predictions:
                scene["prediction"] = predictions[scene_dir.name]
            scenes.append(scene)
    return {
        "dataset_root": _rel(DATA),
        "results_root": _rel(RESULTS),
        "scenes": scenes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "viewer" / "data" / "index.json"))
    args = ap.parse_args()
    index = build_index()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1), encoding="utf-8")
    n_scenes = len(index["scenes"])
    n_buildings = sum(len(s["buildings"]) for s in index["scenes"])
    print(f"wrote {out}  ({n_scenes} scenes, {n_buildings} buildings)")


if __name__ == "__main__":
    main()
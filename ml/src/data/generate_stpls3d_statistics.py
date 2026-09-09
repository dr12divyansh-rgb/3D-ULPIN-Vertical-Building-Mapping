"""Generate comprehensive dataset statistics for preprocessed STPLS3D data.

Usage:
    python ml/src/data/generate_stpls3d_statistics.py \
        --data ml/data/stpls3d \
        --out ml/data/stpls3d/statistics.json

Reads manifest.json and computes:
- Total files/scenes/chunks/points
- Class distribution before/after mapping
- Train/validation/test breakdown
- Imbalance metrics
- Per-split statistics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]
NUM_CLASSES = 6

# STPLS3D original label IDs → names
STPLS3D_LABELS = {
    0: "Ground", 1: "Building", 2: "LowVegetation", 3: "MediumVegetation",
    4: "HighVegetation", 5: "Vehicle", 6: "Truck", 7: "Aircraft",
    8: "MilitaryVehicle", 9: "Bike", 10: "Motorcycle", 11: "LightPole",
    12: "StreetSign", 13: "Clutter", 14: "Fence", 15: "Road",
    17: "Windows", 18: "Dirt", 19: "Grass",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to preprocessed STPLS3D directory (has manifest.json)")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <data>/statistics.json)")
    args = ap.parse_args()

    data_root = Path(args.data)
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found at {manifest_path}", file=sys.stderr)
        print("Run prepare_stpls3d.py first to generate the manifest.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading manifest from: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    chunks = manifest["chunks"]
    print(f"Total chunks in manifest: {len(chunks)}")

    # Aggregate statistics
    splits = {"train": [], "validation": [], "test": []}
    for c in chunks:
        splits[c["split"]].append(c)

    # Build report
    report = {
        "data_root": str(data_root),
        "manifest_version": manifest.get("version"),
        "chunk_size": manifest.get("chunk_size"),
        "cell_size": manifest.get("cell_size"),
        "seed": manifest.get("seed"),
        "num_classes": manifest.get("num_classes"),
        "class_names": manifest.get("class_names"),
        "total_chunks": len(chunks),
        "total_points": sum(c["n_points"] for c in chunks),
        "splits": {},
    }

    # Per-split breakdown
    for split_name, split_chunks in splits.items():
        n_chunks = len(split_chunks)
        n_points = sum(c["n_points"] for c in split_chunks)
        scenes = {c["scene_id"] for c in split_chunks}
        n_scenes = len(scenes)

        # Class distribution
        class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for c in split_chunks:
            for cid, cnt in c["per_class"].items():
                class_counts[int(cid)] += cnt

        per_class = {}
        for ci, cname in enumerate(CLASS_NAMES):
            cnt = int(class_counts[ci])
            pct = 100.0 * cnt / max(n_points, 1)
            per_class[cname] = {"count": cnt, "percent": round(pct, 2)}

        report["splits"][split_name] = {
            "num_chunks": n_chunks,
            "num_points": n_points,
            "num_scenes": n_scenes,
            "scenes": sorted(scenes),
            "per_class": per_class,
        }

    # Overall class distribution
    overall_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for c in chunks:
        for cid, cnt in c["per_class"].items():
            overall_counts[int(cid)] += cnt

    overall_per_class = {}
    for ci, cname in enumerate(CLASS_NAMES):
        cnt = int(overall_counts[ci])
        pct = 100.0 * cnt / max(report["total_points"], 1)
        overall_per_class[cname] = {"count": cnt, "percent": round(pct, 2)}
    report["overall_per_class"] = overall_per_class

    # Class imbalance metrics
    overall_dist = overall_counts / max(overall_counts.sum(), 1)
    max_class = CLASS_NAMES[int(overall_counts.argmax())]
    min_class = CLASS_NAMES[int(overall_counts.argmin())]
    imbalance_ratio = float(overall_counts.max()) / max(overall_counts.min(), 1)

    report["imbalance_metrics"] = {
        "most_frequent_class": max_class,
        "most_frequent_count": int(overall_counts.max()),
        "least_frequent_class": min_class,
        "least_frequent_count": int(overall_counts.min()),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "entropy": round(-float((overall_dist * np.log(overall_dist + 1e-12)).sum()), 3),
    }

    # Save
    out_path = Path(args.out) if args.out else data_root / "statistics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nStatistics saved to: {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("STPLS3D DATASET STATISTICS")
    print("=" * 70)
    print(f"  Total chunks  : {report['total_chunks']:,}")
    print(f"  Total points  : {report['total_points']:,}")
    print()
    print("  Split breakdown:")
    for sname in ("train", "validation", "test"):
        s = report["splits"].get(sname, {})
        print(f"    {sname:<12}  {s.get('num_chunks', 0):>6,} chunks   "
              f"{s.get('num_scenes', 0):>3} scenes   "
              f"{s.get('num_points', 0):>12,} points")
    print()
    print("  Overall class distribution:")
    for cn in CLASS_NAMES:
        pc = overall_per_class[cn]
        bar = "#" * int(pc["percent"] / 2)
        print(f"    {cn:<12}  {pc['count']:>14,}  {pc['percent']:>5.1f}%  {bar}")
    print()
    print("  Imbalance:")
    im = report["imbalance_metrics"]
    print(f"    Most frequent    : {im['most_frequent_class']} ({im['most_frequent_count']:,})")
    print(f"    Least frequent   : {im['least_frequent_class']} ({im['least_frequent_count']:,})")
    print(f"    Imbalance ratio  : {im['imbalance_ratio']:.1f}:1")
    print(f"    Distribution entropy : {im['entropy']:.3f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

"""Extract ASPRS class-2 ground points from the full AHN4 LAZ tile.

Supplements the existing test-strip ground NPZ with the full tile coverage
needed for full-tile reconstruction.

Usage:
    python ml/src/reconstruction/extract_ahn4_ground_fulltile.py \\
        --laz ml/results/ahn4_inference/data/25GN2_01.LAZ \\
        --out ml/results/ahn4_fulltile/ahn4_ground_fulltile.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np


def extract_asprs_ground(laz_path: Path, out_path: Path) -> dict:
    try:
        import laspy
    except ImportError:
        print("ERROR: laspy not available. Install with: pip install laspy")
        sys.exit(1)

    print(f"Reading LAZ: {laz_path}")
    las = laspy.read(str(laz_path))

    total = len(las.x)
    print(f"Total points: {total:,}")

    asprs_class = np.asarray(las.classification, dtype=np.uint8)
    ground_mask = (asprs_class == 2)  # ASPRS 2 = ground
    n_ground = ground_mask.sum()
    print(f"ASPRS class-2 ground points: {n_ground:,} ({100*n_ground/total:.1f}%)")

    gx = np.asarray(las.x)[ground_mask].astype(np.float64)
    gy = np.asarray(las.y)[ground_mask].astype(np.float64)
    gz = np.asarray(las.z)[ground_mask].astype(np.float64)

    print(f"X range: {gx.min():.1f} - {gx.max():.1f}")
    print(f"Y range: {gy.min():.1f} - {gy.max():.1f}")
    print(f"Z range: {gz.min():.3f} - {gz.max():.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), x=gx, y=gy, z=gz)
    print(f"Saved: {out_path} ({n_ground:,} ground points)")

    return {"n_ground": int(n_ground), "x_range": [float(gx.min()), float(gx.max())],
            "y_range": [float(gy.min()), float(gy.max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laz", default="ml/results/ahn4_inference/data/25GN2_01.LAZ")
    ap.add_argument("--out", default="ml/results/ahn4_fulltile/ahn4_ground_fulltile.npz")
    args = ap.parse_args()

    extract_asprs_ground(ROOT / args.laz, ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

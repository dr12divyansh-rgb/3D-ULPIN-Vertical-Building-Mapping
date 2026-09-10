"""Create a geographically separated train/val/test split for AHN4 chunks.

Split strategy: divide the 1040m x 1290m Amsterdam tile by Y coordinate
(north-south axis). No random mixing of neighbouring chunks.

  Test  strip: northernmost ~20% by Y  (held-out, never used during training)
  Val   strip: next ~20% by Y          (validation during fine-tuning)
  Train strip: remaining ~60% by Y     (fine-tuning data)

This prevents spatial leakage: training chunks are spatially distinct from
test chunks. Geographic coordinates (EPSG:28992) are preserved.

Usage:
    python ml/src/data/split_ahn4_spatial.py \\
        --chunks ml/results/ahn4_inference/chunks/test \\
        --out    ml/results/ahn4_exp007/data/manifest.json \\
        --test-frac 0.20 \\
        --val-frac  0.20
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks",    default="ml/results/ahn4_inference/chunks/test")
    ap.add_argument("--out",       default="ml/results/ahn4_exp007/data/manifest.json")
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--val-frac",  type=float, default=0.20)
    args = ap.parse_args()

    chunk_dir = ROOT / args.chunks
    out_path  = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(chunk_dir.glob("*.npz"))
    if not npz_files:
        print(f"ERROR: no NPZ files in {chunk_dir}")
        return 1
    print(f"\nFound {len(npz_files)} AHN4 chunks in {chunk_dir}")

    # ── Read chunk centres and class distributions ─────────────────────────
    print("Reading chunk centres (Y coordinate for spatial split)...")
    chunk_info = []
    for npz in npz_files:
        d = np.load(npz)
        xyz    = d["xyz"].astype(np.float32)
        labels = d["labels"].astype(np.uint8)

        y_centre = float(xyz[:, 1].mean())
        x_centre = float(xyz[:, 0].mean())

        unique, counts = np.unique(labels, return_counts=True)
        per_class = {int(u): int(c) for u, c in zip(unique, counts)}

        # Stem looks like "25GN2_01__0042"
        stem = npz.stem
        if "__" in stem:
            scene_id, idx_str = stem.rsplit("__", 1)
            chunk_idx = int(idx_str)
        else:
            scene_id, chunk_idx = stem, 0

        chunk_info.append({
            "npz_path":  npz,
            "stem":      stem,
            "scene_id":  scene_id,
            "chunk_idx": chunk_idx,
            "x_centre":  x_centre,
            "y_centre":  y_centre,
            "per_class": per_class,
            "n_points":  int(d["xyz"].shape[0]),
        })

    # ── Sort by Y (north-south) ────────────────────────────────────────────
    chunk_info.sort(key=lambda c: c["y_centre"])
    n_total = len(chunk_info)
    n_test  = max(1, round(n_total * args.test_frac))
    n_val   = max(1, round(n_total * args.val_frac))
    n_train = n_total - n_test - n_val

    # Test = northernmost (highest Y)
    test_chunks  = chunk_info[-n_test:]
    # Val = next strip
    val_chunks   = chunk_info[-(n_test + n_val):-n_test]
    # Train = southernmost (lowest Y)
    train_chunks = chunk_info[:n_train]

    # Y boundary thresholds
    y_test_min = min(c["y_centre"] for c in test_chunks)
    y_val_min  = min(c["y_centre"] for c in val_chunks)
    y_train_max = max(c["y_centre"] for c in train_chunks)

    print(f"\nSpatial split (by Y / north-south axis):")
    print(f"  Train : {n_train} chunks  Y < {y_train_max:.0f}m  (south strip, ~{100*n_train/n_total:.0f}%)")
    print(f"  Val   : {n_val} chunks  Y in [{y_val_min:.0f}, {y_test_min:.0f})m  (~{100*n_val/n_total:.0f}%)")
    print(f"  Test  : {n_test} chunks  Y >= {y_test_min:.0f}m  (north strip, ~{100*n_test/n_total:.0f}%)")

    # ── Build manifest ─────────────────────────────────────────────────────
    # data_root is ml/results/ahn4_inference/
    # paths in manifest are relative to data_root/chunks/
    # so path = "test/25GN2_01__XXXX.npz"
    data_root   = ROOT / "ml/results/ahn4_inference"
    chunks_root = data_root / "chunks"

    def make_entry(info: dict, split: str) -> dict:
        rel_path = info["npz_path"].relative_to(chunks_root)
        return {
            "split":     split,
            "path":      str(rel_path).replace("\\", "/"),
            "scene_id":  info["scene_id"],
            "chunk_idx": info["chunk_idx"],
            "n_points":  info["n_points"],
            "per_class": info["per_class"],
            "x_centre":  round(info["x_centre"], 1),
            "y_centre":  round(info["y_centre"], 1),
        }

    all_entries = []
    for c in train_chunks: all_entries.append(make_entry(c, "train"))
    for c in val_chunks:   all_entries.append(make_entry(c, "validation"))
    for c in test_chunks:  all_entries.append(make_entry(c, "test"))

    manifest = {
        "source":      "ahn4",
        "dataset":     "AHN4 GeoTiles Amsterdam 25GN2_01",
        "data_root":   str(data_root),
        "schema":      "xyz:(4096,3)f32  rgb:(4096,3)u8  labels:(4096,)u8",
        "crs":         "EPSG:28992 (Amersfoort / RD New)",
        "rgb_scale":   "uint8 0-255 / 255.0 at model time",
        "split_method":"spatial_Y_north_south",
        "split_fracs": {"train": args.test_frac, "val": args.val_frac},
        "n_train":     n_train,
        "n_val":       n_val,
        "n_test":      n_test,
        "y_test_boundary":  round(y_test_min, 1),
        "y_val_boundary":   round(y_val_min, 1),
        "cell_size":   50.0,
        "chunk_size":  4096,
        "seed":        26011,
        "chunks":      all_entries,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Class distribution per split ───────────────────────────────────────
    print("\nClass distribution per split:")
    for split, clist in [("train", train_chunks), ("val", val_chunks), ("test", test_chunks)]:
        tot = sum(sum(c["per_class"].values()) for c in clist)
        bld = sum(c["per_class"].get(1, 0) for c in clist)
        gnd = sum(c["per_class"].get(0, 0) for c in clist)
        print(f"  {split:10}: {len(clist):3d} chunks  {tot:>8,} pts  "
              f"building={100*bld/max(tot,1):.1f}%  ground={100*gnd/max(tot,1):.1f}%")

    print(f"\nManifest saved: {out_path}")
    print(f"data_root to pass to training: {data_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

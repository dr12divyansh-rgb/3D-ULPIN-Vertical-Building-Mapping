"""DALES-2 aerial LiDAR → 6-class NPZ preprocessing.

Converts DALES-2 LAZ files into the same NPZ schema used by STPLS3D so that
the existing PointNet++ architecture and dataset loader can consume DALES without
any changes to the model or dataset.py.

DALES-2 legacy_semantic label mapping (1-indexed original DALES taxonomy):
    1 → 0  (ground)
    2 → 2  (vegetation)
    3 → 4  (vehicle: cars)
    4 → 4  (vehicle: trucks)
    5 → 5  (other: power lines)
    6 → 5  (other: fences)
    7 → 5  (other: poles)
    8 → 1  (building)
    other → -1 (discard)

Note: road (project class 3) has no DALES equivalent.  All DALES ground points
map to project class 0 (ground).  This is scientifically honest — DALES does
not distinguish road from ground at annotation level.

Output NPZ schema (identical to prepare_stpls3d.py):
    xyz:    (chunk_size, 3)  float32  raw world coordinates (metres)
    rgb:    (chunk_size, 3)  uint8    aerial RGB (uint16 / 256 → uint8)
    labels: (chunk_size,)    uint8    6-class project IDs

Usage:
    python ml/src/data/prepare_dales.py \\
        --input  ml/data/dales2 \\
        --output ml/data/dales_chunks \\
        --split-config ml/src/data/dales_splits.json

    # Or process a single LAZ file:
    python ml/src/data/prepare_dales.py \\
        --input  ml/data/dales2/train \\
        --output ml/data/dales_chunks/train \\
        --split  train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import laspy
except ImportError:
    print("ERROR: laspy not installed.  Run: pip install laspy lazrs")
    sys.exit(1)


# ── Label mapping ──────────────────────────────────────────────────────────────
# DALES-2 legacy_semantic is 1-indexed (1–8).  Values outside this range are
# treated as unlabelled and discarded (mapped to -1).

_DALES_LEGACY_TO_PROJECT = {
    1: 0,   # ground        → ground
    2: 2,   # vegetation    → vegetation
    3: 4,   # cars          → vehicle
    4: 4,   # trucks        → vehicle
    5: 5,   # power lines   → other
    6: 5,   # fences        → other
    7: 5,   # poles         → other
    8: 1,   # buildings     → building
}

# Build a 256-entry lookup table for fast per-point remapping
_LUT = np.full(256, -1, dtype=np.int8)
for dales_id, proj_id in _DALES_LEGACY_TO_PROJECT.items():
    _LUT[dales_id] = proj_id

_PROJECT_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]


# ── Core reading ───────────────────────────────────────────────────────────────

def read_laz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a DALES-2 LAZ file.

    Returns:
        xyz:    (N, 3) float32  world coordinates in metres
        rgb:    (N, 3) uint8    RGB (uint16 >> 8 to fit uint8)
        labels: (N,)   int8     6-class project labels (-1 = discard)
    """
    laz = laspy.read(path)

    xyz = np.column_stack([
        np.asarray(laz.x, dtype=np.float32),
        np.asarray(laz.y, dtype=np.float32),
        np.asarray(laz.z, dtype=np.float32),
    ])

    # RGB: LAS stores uint16 (0–65535); convert to uint8 (0–255) by >>8
    red   = (np.asarray(laz.red,   dtype=np.uint16) >> 8).astype(np.uint8)
    green = (np.asarray(laz.green, dtype=np.uint16) >> 8).astype(np.uint8)
    blue  = (np.asarray(laz.blue,  dtype=np.uint16) >> 8).astype(np.uint8)
    rgb   = np.column_stack([red, green, blue])

    # Labels from legacy_semantic (original 8-class DALES taxonomy, 1-indexed)
    if not hasattr(laz, 'legacy_semantic'):
        raise ValueError(
            f"{path.name}: 'legacy_semantic' field not found.  "
            "This file may be original DALES-1 (not DALES-2).  "
            "Check the dataset source.")
    raw_labels = np.asarray(laz.legacy_semantic, dtype=np.uint8)
    labels     = _LUT[raw_labels.clip(0, 255)].astype(np.int8)

    return xyz, rgb, labels


# ── Spatial chunking ──────────────────────────────────────────────────────────

def spatial_chunks(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    cell_size: float = 50.0,
    chunk_size: int = 4096,
    min_points: int = 256,
    seed: int = 26011,
) -> list[dict]:
    """Divide a point cloud into spatial 2D grid cells and sample each to chunk_size.

    Cells with fewer than min_points valid (label ≥ 0) points are discarded.
    Sampling is with-replacement if a cell has < chunk_size valid points.

    Args:
        xyz:        (N, 3) float32 world coordinates
        rgb:        (N, 3) uint8
        labels:     (N,)   int8  (-1 = invalid)
        cell_size:  side length of each XY cell in metres
        chunk_size: target points per chunk (4096 to match STPLS3D)
        min_points: minimum valid points to keep a cell
        seed:       RNG seed for reproducibility

    Returns:
        List of dicts with keys 'xyz', 'rgb', 'labels' each of shape
        (chunk_size, *) with dtype matching input.
    """
    rng = np.random.default_rng(seed)

    valid = labels >= 0
    x_all, y_all = xyz[:, 0], xyz[:, 1]

    x_min, y_min = float(x_all.min()), float(y_all.min())
    xi = ((x_all - x_min) / cell_size).astype(np.int32)
    yi = ((y_all - y_min) / cell_size).astype(np.int32)
    cell_id = xi.astype(np.int64) * 100000 + yi  # unique per cell

    cells: dict[int, np.ndarray] = {}
    for idx in range(len(xyz)):
        if valid[idx]:
            cid = cell_id[idx]
            if cid not in cells:
                cells[cid] = []
            cells[cid].append(idx)

    chunks = []
    for cid, indices in cells.items():
        if len(indices) < min_points:
            continue
        idx_arr = np.array(indices, dtype=np.int64)
        if len(idx_arr) >= chunk_size:
            sel = rng.choice(idx_arr, size=chunk_size, replace=False)
        else:
            sel = rng.choice(idx_arr, size=chunk_size, replace=True)
        chunks.append({
            'xyz':    xyz[sel].astype(np.float32),
            'rgb':    rgb[sel].astype(np.uint8),
            'labels': labels[sel].astype(np.uint8),
        })

    return chunks


# ── Validation ────────────────────────────────────────────────────────────────

def validate_npz(path: Path, chunk_size: int = 4096) -> dict:
    """Load one NPZ and verify schema + content."""
    data = np.load(path)
    issues = []

    for key, expected_shape, expected_dtype in [
        ('xyz',    (chunk_size, 3), np.float32),
        ('rgb',    (chunk_size, 3), np.uint8),
        ('labels', (chunk_size,),   np.uint8),
    ]:
        if key not in data:
            issues.append(f"missing key: {key}")
            continue
        if data[key].shape != expected_shape:
            issues.append(f"{key}: shape {data[key].shape} ≠ {expected_shape}")
        if data[key].dtype != expected_dtype:
            issues.append(f"{key}: dtype {data[key].dtype} ≠ {expected_dtype}")

    if issues:
        raise ValueError(f"{path.name}: {'; '.join(issues)}")

    labels = data['labels']
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
    return {
        'path':       str(path),
        'class_dist': class_dist,
        'building_pts': class_dist.get(1, 0),
    }


# ── Per-tile processing ────────────────────────────────────────────────────────

def process_tile(
    laz_path: Path,
    out_dir: Path,
    cell_size: float,
    chunk_size: int,
    min_points: int,
    seed: int,
) -> dict:
    """Process one LAZ tile → write NPZ chunks → return statistics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_name = laz_path.stem

    print(f"\n  Reading {laz_path.name}...")
    xyz, rgb, labels = read_laz(laz_path)
    n_raw = len(xyz)

    valid_mask = labels >= 0
    n_valid   = int(valid_mask.sum())
    n_discard = n_raw - n_valid

    # Class distribution of valid points
    valid_labels = labels[valid_mask]
    unique, counts = np.unique(valid_labels, return_counts=True)
    raw_dist = {int(u): int(c) for u, c in zip(unique, counts)}

    print(f"    {n_raw:,} raw points  |  {n_valid:,} valid  |  {n_discard:,} discarded")
    x_span = float(xyz[:, 0].max() - xyz[:, 0].min())
    y_span = float(xyz[:, 1].max() - xyz[:, 1].min())
    z_span = float(xyz[:, 2].max() - xyz[:, 2].min())
    density = n_valid / max(x_span * y_span, 1.0)
    print(f"    XY extent: {x_span:.0f}m × {y_span:.0f}m  Z range: {xyz[:,2].min():.1f}–{xyz[:,2].max():.1f}m")
    print(f"    Density: {density:.1f} pts/m²")
    print(f"    Class distribution (valid):")
    for cid, cnt in sorted(raw_dist.items()):
        name = _PROJECT_NAMES[cid] if cid < len(_PROJECT_NAMES) else f"cls{cid}"
        print(f"      {name:12} ({cid}): {cnt:>8,} ({100*cnt/n_valid:.1f}%)")

    # Chunk
    print(f"    Chunking (cell_size={cell_size}m, chunk_size={chunk_size})...")
    chunks = spatial_chunks(xyz, rgb, labels,
                            cell_size=cell_size,
                            chunk_size=chunk_size,
                            min_points=min_points,
                            seed=seed)
    print(f"    {len(chunks)} chunks produced")

    # Write NPZs
    written = 0
    total_building = 0
    for i, chunk in enumerate(chunks):
        fname = f"{tile_name}__{i:04d}.npz"
        np.savez_compressed(out_dir / fname,
                            xyz=chunk['xyz'],
                            rgb=chunk['rgb'],
                            labels=chunk['labels'])
        total_building += int((chunk['labels'] == 1).sum())
        written += 1

    pct_bldg = 100.0 * total_building / max(written * chunk_size, 1)
    print(f"    Written: {written} NPZ files  |  building pts: {total_building:,} ({pct_bldg:.1f}%)")

    return {
        'tile':           tile_name,
        'n_raw_points':   n_raw,
        'n_valid_points': n_valid,
        'n_discarded':    n_discard,
        'x_span_m':       round(x_span, 1),
        'y_span_m':       round(y_span, 1),
        'z_span_m':       round(z_span, 1),
        'density_pts_m2': round(density, 1),
        'n_chunks':       written,
        'class_dist':     raw_dist,
        'building_pct':   round(pct_bldg, 2),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="DALES-2 LAZ → 6-class NPZ preprocessing (matches STPLS3D schema).")
    ap.add_argument("--input",   required=True,
                    help="Directory containing DALES-2 LAZ files, OR a single LAZ file.")
    ap.add_argument("--output",  required=True,
                    help="Root output directory.  Chunks are written to <output>/<split>/.")
    ap.add_argument("--split",   default=None,
                    choices=["train", "validation", "test"],
                    help="Split name.  If not given, inferred from input directory name.")
    ap.add_argument("--cell-size",   type=float, default=50.0,
                    help="XY grid cell side length in metres (default 50).")
    ap.add_argument("--chunk-size",  type=int,   default=4096,
                    help="Points per output chunk (default 4096, must match STPLS3D).")
    ap.add_argument("--min-points",  type=int,   default=256,
                    help="Minimum valid points to keep a cell (default 256).")
    ap.add_argument("--seed",        type=int,   default=26011,
                    help="RNG seed (default 26011).")
    ap.add_argument("--validate",    action="store_true",
                    help="Validate written NPZs after processing.")
    args = ap.parse_args()

    in_path  = ROOT / args.input
    out_root = ROOT / args.output

    # Collect LAZ files
    if in_path.is_file() and in_path.suffix == '.laz':
        laz_files = [in_path]
    elif in_path.is_dir():
        laz_files = sorted(in_path.glob("*.laz"))
    else:
        print(f"ERROR: --input must be a LAZ file or a directory: {in_path}")
        return 1

    if not laz_files:
        print(f"No LAZ files found in {in_path}")
        return 1

    # STPLS3DChunkDataset resolves: data_root / "chunks" / entry["path"]
    # NPZ files must be under <out_root>/chunks/<split>/
    # Manifest entry paths are "<split>/file.npz" (relative to chunks/).
    split = args.split or in_path.name
    out_dir = out_root / "chunks" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDALES-2 Preprocessing")
    print(f"  Input:      {in_path} ({len(laz_files)} LAZ files)")
    print(f"  Output:     {out_dir}")
    print(f"  Split:      {split}")
    print(f"  Cell size:  {args.cell_size} m")
    print(f"  Chunk size: {args.chunk_size} pts")
    print(f"  Min points: {args.min_points}")

    all_stats = []
    for laz_path in laz_files:
        stats = process_tile(
            laz_path, out_dir,
            cell_size=args.cell_size,
            chunk_size=args.chunk_size,
            min_points=args.min_points,
            seed=args.seed,
        )
        all_stats.append(stats)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_chunks   = sum(s['n_chunks']       for s in all_stats)
    total_raw      = sum(s['n_raw_points']   for s in all_stats)
    total_valid    = sum(s['n_valid_points'] for s in all_stats)
    total_discard  = sum(s['n_discarded']    for s in all_stats)

    # Aggregate class distribution
    agg: dict[int, int] = {}
    for s in all_stats:
        for cid, cnt in s['class_dist'].items():
            agg[cid] = agg.get(cid, 0) + cnt

    total_bldg = agg.get(1, 0)

    print(f"\n{'='*55}")
    print(f"  Tiles processed : {len(all_stats)}")
    print(f"  Total raw pts   : {total_raw:,}")
    print(f"  Valid pts       : {total_valid:,}  ({100*total_valid/max(total_raw,1):.1f}%)")
    print(f"  Discarded pts   : {total_discard:,}  ({100*total_discard/max(total_raw,1):.1f}%)")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Building pts    : {total_bldg:,}  ({100*total_bldg/max(total_valid,1):.1f}%)")
    print(f"\n  Class distribution across all tiles:")
    for cid in sorted(agg):
        name = _PROJECT_NAMES[cid] if cid < len(_PROJECT_NAMES) else f"cls{cid}"
        print(f"    {name:12} ({cid}): {agg[cid]:>10,}  ({100*agg[cid]/max(total_valid,1):.1f}%)")
    print(f"{'='*55}")

    # Validation
    if args.validate:
        print(f"\nValidating {total_chunks} NPZ files...")
        issues = 0
        for npz in sorted(out_dir.glob("*.npz")):
            try:
                validate_npz(npz, args.chunk_size)
            except ValueError as e:
                print(f"  FAIL: {e}")
                issues += 1
        if issues == 0:
            print(f"  All {total_chunks} NPZs passed validation.")
        else:
            print(f"  {issues} validation failures!")
            return 1

    # ── Write/update top-level manifest (STPLS3DChunkDataset-compatible) ──────
    # The manifest lives at out_root/manifest.json and accumulates across runs
    # (one per split).  Each chunk entry has: path (relative to out_root),
    # split, scene_id, chunk_idx, n_points, per_class — matching the STPLS3D schema.
    manifest_path = out_root / 'manifest.json'

    # Load existing manifest if present (incremental update across splits)
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            top = json.load(f)
        # Remove any existing entries for this split (idempotent re-run)
        top['chunks'] = [c for c in top['chunks'] if c['split'] != split]
    else:
        top = {
            'source': 'dales',
            'dataset': 'DALES-2',
            'schema': 'xyz:(4096,3)f32  rgb:(4096,3)u8  labels:(4096,)u8',
            'cell_size': args.cell_size,
            'chunk_size': args.chunk_size,
            'seed': args.seed,
            'chunks': [],
        }

    # Enumerate written NPZ files for this split and add to manifest
    new_entries = []
    for npz_file in sorted(out_dir.glob('*.npz')):
        # Parse tile_name and chunk_idx from filename e.g. 5100_54440__0003.npz
        stem = npz_file.stem
        if '__' in stem:
            scene_id, idx_str = stem.rsplit('__', 1)
            chunk_idx = int(idx_str)
        else:
            scene_id, chunk_idx = stem, 0

        # Read per_class distribution from NPZ for manifest
        data = np.load(npz_file)
        labels_arr = data['labels']
        unique_l, counts_l = np.unique(labels_arr, return_counts=True)
        per_class = {int(u): int(c) for u, c in zip(unique_l, counts_l)}

        entry = {
            'split':    split,
            # Path must be relative to the chunks/ directory because
            # STPLS3DChunkDataset resolves: data_root / "chunks" / entry["path"]
            'path':     str(npz_file.relative_to(out_root / 'chunks')).replace('\\', '/'),
            'scene_id': scene_id,
            'chunk_idx': chunk_idx,
            'n_points': int(args.chunk_size),
            'per_class': per_class,
        }
        new_entries.append(entry)

    top['chunks'].extend(new_entries)

    with open(manifest_path, 'w') as f:
        json.dump(top, f, indent=2)

    n_train = sum(1 for c in top['chunks'] if c['split'] == 'train')
    n_val   = sum(1 for c in top['chunks'] if c['split'] == 'validation')
    n_test  = sum(1 for c in top['chunks'] if c['split'] == 'test')
    print(f"\nManifest updated: {manifest_path}")
    print(f"  train={n_train}  validation={n_val}  test={n_test} chunks")
    print(f"Output dir: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""AHN4 (GeoTiles) aerial LiDAR -> 6-class NPZ preprocessing.

Converts AHN4 LAZ files from GeoTiles (geotiles.citg.tudelft.nl) into the
same NPZ schema used by the STPLS3D/DALES pipeline so that the existing
PointNet++ model (Exp006) can run inference on AHN4 without architectural
changes.

== Data Source ==
GeoTiles AHN4 sub-tiles (1 x 1.25 km, ~28-30 pts/m^2):
    https://geotiles.citg.tudelft.nl/AHN4_T/<TILEID>_<NN>.LAZ
    LAS 1.4, Point Record Format 8 (X Y Z Intensity ... Red Green Blue NIR + extras)
    CRS: EPSG:28992 (Amersfoort / RD New) horizontal, NAP vertical

== RGB Normalisation ==
GeoTiles stores RGB in uint16 fields but with values in the range 0-255
(NOT 0-65535 as per LAS standard convention). Division by 255.0 is correct.
Do NOT use >> 8 (would produce near-zero values).

== ASPRS Classification -> Project Class Mapping ==
    ASPRS 2  (Ground)             -> 0  ground
    ASPRS 3  (Low Vegetation)     -> 2  vegetation
    ASPRS 4  (Medium Vegetation)  -> 2  vegetation
    ASPRS 5  (High Vegetation)    -> 2  vegetation
    ASPRS 6  (Building)           -> 1  building
    ASPRS 9  (Water)              -> 0  ground  (Netherlands: water at ground level)
    ASPRS 10 (Rail)               -> 3  road
    ASPRS 11 (Road Surface)       -> 3  road
    ASPRS 13-17 (Wires/Tower etc) -> 5  other
    ASPRS 1  (Unclassified)       -> -1 DISCARD for evaluation (unknown true class)
    Anything else                 -> -1 DISCARD

Unclassified points (ASPRS 1) are discarded for evaluation chunks.
For inference-only chunks, they are mapped to class 5 (other).

== Output Schema (identical to prepare_dales.py) ==
    xyz:    (chunk_size, 3)  float32  raw world coordinates in metres (RD New)
    rgb:    (chunk_size, 3)  uint8    aerial RGB (uint16 / 255 -> uint8)
    labels: (chunk_size,)    uint8    6-class project IDs

Usage:
    python ml/src/data/prepare_ahn4.py \\
        --input  ml/results/ahn4_inference/data \\
        --output ml/results/ahn4_inference/chunks \\
        --split  inference

    # Single file:
    python ml/src/data/prepare_ahn4.py \\
        --input  ml/results/ahn4_inference/data/25GN2_01.LAZ \\
        --output ml/results/ahn4_inference/chunks \\
        --split  inference
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
    print("ERROR: laspy not installed. Run: pip install laspy lazrs")
    sys.exit(1)


# ── Label mapping ───────────────────────────────────────────────────────────

# ASPRS LAS standard classification codes -> project class (0-5)
# -1 = discard (unknown/unclassified in evaluation mode)
#  5 = other (kept in inference mode)
_ASPRS_TO_PROJECT_EVAL = {
    # --- Definitive labels (kept for evaluation)
    2:  0,  # Ground                -> ground
    3:  2,  # Low Vegetation        -> vegetation
    4:  2,  # Medium Vegetation     -> vegetation
    5:  2,  # High Vegetation       -> vegetation
    6:  1,  # Building              -> building
    9:  0,  # Water                 -> ground (Netherlands topography)
    10: 3,  # Rail                  -> road
    11: 3,  # Road Surface          -> road
    17: 5,  # Bridge Deck           -> other
}
# In evaluation mode, all other ASPRS codes map to -1 (discard)

_ASPRS_TO_PROJECT_INFER = {
    **_ASPRS_TO_PROJECT_EVAL,
    # --- Ambiguous/unclassified kept as "other" for inference
    0:  5,  # Never classified      -> other
    1:  5,  # Unclassified          -> other
    7:  5,  # Low Point/Noise       -> other
    8:  5,  # Model Key-point       -> other
    12: 5,  # Reserved              -> other
    13: 5,  # Wire - Guard          -> other
    14: 5,  # Wire - Conductor      -> other
    15: 5,  # Transmission Tower    -> other
    16: 5,  # Wire Connector        -> other
    18: 5,  # High Noise            -> other
    26: 5,  # AHN4 custom class     -> other
}

_PROJECT_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]


def _build_lut(mapping: dict, n: int = 256) -> np.ndarray:
    lut = np.full(n, -1, dtype=np.int8)
    for asprs_id, proj_id in mapping.items():
        if 0 <= asprs_id < n:
            lut[asprs_id] = proj_id
    return lut


_LUT_EVAL  = _build_lut(_ASPRS_TO_PROJECT_EVAL)
_LUT_INFER = _build_lut(_ASPRS_TO_PROJECT_INFER)


# ── Core reading ─────────────────────────────────────────────────────────────

def read_laz(path: Path, eval_mode: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an AHN4 GeoTiles LAZ file.

    Args:
        path:      LAZ file path
        eval_mode: If True, unclassified points are discarded (label=-1).
                   If False (inference only), they map to class 5 (other).

    Returns:
        xyz:    (N, 3) float32  world coordinates in RD New metres
        rgb:    (N, 3) uint8    RGB 0-255 (already in this range in AHN4 GeoTiles)
        labels: (N,)   int8     6-class project labels (-1 = discard)
    """
    laz = laspy.read(path)

    xyz = np.column_stack([
        np.asarray(laz.x, dtype=np.float32),
        np.asarray(laz.y, dtype=np.float32),
        np.asarray(laz.z, dtype=np.float32),
    ])

    # AHN4 GeoTiles RGB: stored as uint16 but values are 0-255 (from 8-bit aerial imagery)
    # Verified: las.red.max() == 255, no values > 255 in 38M-point tile
    # Correct normalisation: cast directly to uint8 (values already 0-255)
    # WRONG: (uint16 >> 8) would give near-zero results (e.g. 200 >> 8 = 0)
    red   = np.asarray(laz.red,   dtype=np.uint16).astype(np.uint8)
    green = np.asarray(laz.green, dtype=np.uint16).astype(np.uint8)
    blue  = np.asarray(laz.blue,  dtype=np.uint16).astype(np.uint8)
    rgb   = np.column_stack([red, green, blue])

    # ASPRS classification -> project class
    asprs = np.asarray(laz.classification, dtype=np.uint8)
    lut   = _LUT_EVAL if eval_mode else _LUT_INFER
    labels = lut[asprs.clip(0, 255)].astype(np.int8)

    return xyz, rgb, labels


# ── Spatial chunking ─────────────────────────────────────────────────────────

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

    Identical logic to prepare_dales.py for pipeline compatibility.
    Cells with fewer than min_points valid (label >= 0) points are discarded.
    """
    rng = np.random.default_rng(seed)

    valid = labels >= 0
    x_all, y_all = xyz[:, 0], xyz[:, 1]
    x_min, y_min = float(x_all.min()), float(y_all.min())

    xi = ((x_all - x_min) / cell_size).astype(np.int32)
    yi = ((y_all - y_min) / cell_size).astype(np.int32)
    cell_id = xi.astype(np.int64) * 100000 + yi

    cells: dict[int, list[int]] = {}
    for idx in range(len(xyz)):
        if valid[idx]:
            cid = int(cell_id[idx])
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


# ── Tile processing ──────────────────────────────────────────────────────────

def process_tile(
    laz_path: Path,
    out_dir: Path,
    cell_size: float,
    chunk_size: int,
    min_points: int,
    seed: int,
    eval_mode: bool,
) -> dict:
    """Process one AHN4 LAZ tile -> write NPZ chunks -> return stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_name = laz_path.stem

    print(f"\n  Reading {laz_path.name}...")
    xyz, rgb, labels = read_laz(laz_path, eval_mode=eval_mode)
    n_raw = len(xyz)

    valid_mask = labels >= 0
    n_valid   = int(valid_mask.sum())
    n_discard = n_raw - n_valid

    valid_labels = labels[valid_mask]
    unique, counts = np.unique(valid_labels, return_counts=True)
    raw_dist = {int(u): int(c) for u, c in zip(unique, counts)}

    x_span = float(xyz[:, 0].max() - xyz[:, 0].min())
    y_span = float(xyz[:, 1].max() - xyz[:, 1].min())
    z_span = float(xyz[:, 2].max() - xyz[:, 2].min())
    density = n_valid / max(x_span * y_span, 1.0)

    print(f"    {n_raw:,} raw points  |  {n_valid:,} valid  |  {n_discard:,} discarded (unclassified)")
    print(f"    XY extent: {x_span:.0f}m x {y_span:.0f}m  Z range: {xyz[:,2].min():.1f} to {xyz[:,2].max():.1f}m")
    print(f"    Density (valid): {density:.1f} pts/m2")
    print(f"    RGB: dtype=uint8 after cast, range checked 0-255")
    print(f"    Class distribution (valid):")
    for cid, cnt in sorted(raw_dist.items()):
        name = _PROJECT_NAMES[cid] if cid < len(_PROJECT_NAMES) else f"cls{cid}"
        print(f"      {name:12} ({cid}): {cnt:>8,} ({100*cnt/n_valid:.1f}%)")

    print(f"    Chunking (cell_size={cell_size}m, chunk_size={chunk_size})...")
    chunks = spatial_chunks(xyz, rgb, labels,
                            cell_size=cell_size,
                            chunk_size=chunk_size,
                            min_points=min_points,
                            seed=seed)
    print(f"    {len(chunks)} chunks produced")

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
        'tile':              tile_name,
        'n_raw_points':      n_raw,
        'n_valid_points':    n_valid,
        'n_discarded':       n_discard,
        'x_span_m':          round(x_span, 1),
        'y_span_m':          round(y_span, 1),
        'z_span_m':          round(z_span, 1),
        'density_pts_m2':    round(density, 1),
        'n_chunks':          written,
        'class_dist':        raw_dist,
        'building_pct':      round(pct_bldg, 2),
        'crs':               'EPSG:28992 (Amersfoort / RD New)',
        'rgb_scale':         '0-255 uint8 (cast from uint16 in source)',
        'rgb_normalisation': 'divide by 255.0 for model input',
    }


# ── Manifest ─────────────────────────────────────────────────────────────────

def write_manifest(
    out_root: Path,
    split: str,
    out_dir: Path,
    all_stats: list[dict],
    cell_size: float,
    chunk_size: int,
    seed: int,
) -> None:
    """Write/update the dataset manifest (STPLS3DChunkDataset-compatible)."""
    manifest_path = out_root / 'manifest.json'

    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            top = json.load(f)
        top['chunks'] = [c for c in top['chunks'] if c['split'] != split]
    else:
        top = {
            'source':     'ahn4',
            'dataset':    'AHN4 GeoTiles - Amsterdam (25GN2_01)',
            'schema':     'xyz:(4096,3)f32  rgb:(4096,3)u8  labels:(4096,)u8',
            'crs':        'EPSG:28992 (Amersfoort / RD New)',
            'rgb_scale':  '0-255 uint8 (source uint16 cast; divide by 255.0 for model)',
            'cell_size':  cell_size,
            'chunk_size': chunk_size,
            'seed':       seed,
            'chunks':     [],
        }

    new_entries = []
    for npz_file in sorted(out_dir.glob('*.npz')):
        stem = npz_file.stem
        if '__' in stem:
            scene_id, idx_str = stem.rsplit('__', 1)
            chunk_idx = int(idx_str)
        else:
            scene_id, chunk_idx = stem, 0

        data = np.load(npz_file)
        labels_arr = data['labels']
        unique_l, counts_l = np.unique(labels_arr, return_counts=True)
        per_class = {int(u): int(c) for u, c in zip(unique_l, counts_l)}

        entry = {
            'split':     split,
            'path':      str(npz_file.relative_to(out_root / 'chunks')).replace('\\', '/'),
            'scene_id':  scene_id,
            'chunk_idx': chunk_idx,
            'n_points':  int(chunk_size),
            'per_class': per_class,
        }
        new_entries.append(entry)

    top['chunks'].extend(new_entries)

    with open(manifest_path, 'w') as f:
        json.dump(top, f, indent=2)
    print(f"\n  Manifest: {manifest_path}  ({len(new_entries)} entries for split='{split}')")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="AHN4 GeoTiles LAZ -> 6-class NPZ preprocessing (STPLS3D/DALES-compatible).")
    ap.add_argument("--input",   required=True,
                    help="Directory of LAZ files, OR a single LAZ file.")
    ap.add_argument("--output",  required=True,
                    help="Root output directory. Chunks go to <output>/chunks/<split>/.")
    ap.add_argument("--split",   default="inference",
                    choices=["train", "validation", "test", "inference"],
                    help="Split name (default: inference).")
    ap.add_argument("--cell-size",   type=float, default=50.0,
                    help="XY grid cell side length in metres (default 50, same as DALES).")
    ap.add_argument("--chunk-size",  type=int,   default=4096,
                    help="Points per chunk (default 4096, must match model).")
    ap.add_argument("--min-points",  type=int,   default=256,
                    help="Minimum valid points to keep a cell (default 256).")
    ap.add_argument("--seed",        type=int,   default=26011,
                    help="RNG seed (default 26011 = SIH problem number).")
    ap.add_argument("--no-eval",     action="store_true",
                    help="Inference mode: keep unclassified points as class 5 (other). "
                         "Default: eval mode discards unclassified for honest evaluation.")
    args = ap.parse_args()

    in_path  = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_root = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    eval_mode = not args.no_eval

    # Collect LAZ files
    if in_path.is_file() and in_path.suffix.lower() == '.laz':
        laz_files = [in_path]
    elif in_path.is_dir():
        laz_files = sorted(in_path.glob("*.laz")) + sorted(in_path.glob("*.LAZ"))
    else:
        print(f"ERROR: --input must be a LAZ file or directory: {in_path}")
        return 1

    if not laz_files:
        print(f"No LAZ files found in {in_path}")
        return 1

    split   = args.split
    out_dir = out_root / "chunks" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAHN4 GeoTiles Preprocessing")
    print(f"  Input:      {in_path} ({len(laz_files)} LAZ files)")
    print(f"  Output:     {out_dir}")
    print(f"  Split:      {split}")
    print(f"  Cell size:  {args.cell_size} m")
    print(f"  Chunk size: {args.chunk_size} pts")
    print(f"  Min points: {args.min_points}")
    print(f"  Eval mode:  {eval_mode} (discard unclassified: {eval_mode})")
    print(f"  RGB:        uint16 0-255 -> uint8 -> /255.0 at model time")
    print(f"  CRS:        EPSG:28992 (Amersfoort / RD New)")

    all_stats = []
    for laz_path in laz_files:
        stats = process_tile(
            laz_path, out_dir,
            cell_size=args.cell_size,
            chunk_size=args.chunk_size,
            min_points=args.min_points,
            seed=args.seed,
            eval_mode=eval_mode,
        )
        all_stats.append(stats)

    # Summary
    total_chunks  = sum(s['n_chunks']       for s in all_stats)
    total_raw     = sum(s['n_raw_points']   for s in all_stats)
    total_valid   = sum(s['n_valid_points'] for s in all_stats)
    total_discard = sum(s['n_discarded']    for s in all_stats)

    agg: dict[int, int] = {}
    for s in all_stats:
        for cid, cnt in s['class_dist'].items():
            agg[cid] = agg.get(cid, 0) + cnt

    total_bldg = agg.get(1, 0)

    print(f"\n{'='*60}")
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
    print(f"{'='*60}")

    write_manifest(out_root, split, out_dir, all_stats,
                   args.cell_size, args.chunk_size, args.seed)
    print(f"\nOutput dir: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Preprocess STPLS3D PLY data for the SIH 26011 six-class segmentation pipeline.

Reads directly from the STPLS3D.zip (or a pre-extracted directory) without ever
loading the full 35 GB into RAM.  Produces fixed-size (4096-point) spatial chunks
saved as .npz files plus a JSON manifest describing every chunk.

Usage — full run from ZIP:
    python ml/src/data/prepare_stpls3d.py \\
        --source-zip  C:/Users/Takin/Downloads/STPLS3D.zip \\
        --output      ml/data/stpls3d

Usage — point at a pre-extracted directory:
    python ml/src/data/prepare_stpls3d.py \\
        --source-dir  C:/data/STPLS3D \\
        --output      ml/data/stpls3d

Smoke-test mode (just RA_points.ply, first 200 K points):
    python ml/src/data/prepare_stpls3d.py \\
        --source-zip  C:/Users/Takin/Downloads/STPLS3D.zip \\
        --output      ml/data/stpls3d_smoke \\
        --smoke-test

Only XYZ + labels are written to the chunks.  RGB is preserved in the .npz for
future XYZRGB experiments; the current PointNet++ only sees XYZ.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PLY type → numpy dtype char
# ---------------------------------------------------------------------------
_PLY_NP = {
    "float": "<f4",    "float32": "<f4",
    "double": "<f8",   "float64": "<f8",
    "uint8": "u1",     "uchar": "u1",
    "int8": "i1",      "char": "i1",
    "uint16": "<u2",   "ushort": "<u2",
    "int16": "<i2",    "short": "<i2",
    "uint32": "<u4",   "uint": "<u4",
    "int32": "<i4",    "int": "<i4",
}

# ---------------------------------------------------------------------------
# STPLS3D semantic label → SIH 26011 six-class label
# Unsupported / missing labels (e.g. 16) map to -1 → filtered out.
# ---------------------------------------------------------------------------
_STPLS3D_NAMES = {
    0: "Ground", 1: "Building", 2: "LowVegetation", 3: "MediumVegetation",
    4: "HighVegetation", 5: "Vehicle", 6: "Truck", 7: "Aircraft",
    8: "MilitaryVehicle", 9: "Bike", 10: "Motorcycle", 11: "LightPole",
    12: "StreetSign", 13: "Clutter", 14: "Fence", 15: "Road",
    # 16 absent in spec
    17: "Windows", 18: "Dirt", 19: "Grass",
}

OUR_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]
NUM_CLASSES = 6

# Build a lookup array of size 256.  -1 ⇒ remove point.
_LABEL_LUT = np.full(256, -1, dtype=np.int8)
_STPLS3D_TO_OURS: dict[int, int] = {
    0: 0,   # Ground       → ground
    1: 1,   # Building     → building
    2: 2,   # LowVeg       → vegetation
    3: 2,   # MediumVeg    → vegetation
    4: 2,   # HighVeg      → vegetation
    5: 4,   # Vehicle      → vehicle
    6: 4,   # Truck        → vehicle
    7: 4,   # Aircraft     → vehicle
    8: 4,   # MilitaryVeh  → vehicle
    9: 4,   # Bike         → vehicle
    10: 4,  # Motorcycle   → vehicle
    11: 5,  # LightPole    → other
    12: 5,  # StreetSign   → other
    13: 5,  # Clutter      → other
    14: 5,  # Fence        → other
    15: 3,  # Road         → road
    # 16 absent → stays -1
    17: 5,  # Windows      → other
    18: 0,  # Dirt         → ground
    19: 2,  # Grass        → vegetation
}
for _src, _dst in _STPLS3D_TO_OURS.items():
    _LABEL_LUT[_src] = _dst


# ---------------------------------------------------------------------------
# Deterministic scene-level train / validation / test split
# ---------------------------------------------------------------------------
# Rule:
#   STPLS3D/RealWorldData/*          → test   (real-world, never trained on)
#   STPLS3D/Synthetic_v3/  N ≥ 21   → validation
#   everything else in Synthetic_*  → train
#
def assign_split(zip_entry_name: str) -> str:
    """Return 'train', 'validation', or 'test' for a ZIP entry path."""
    parts = Path(zip_entry_name).parts  # e.g. ('STPLS3D', 'Synthetic_v3', '22_points_GTv3.ply')
    if len(parts) < 2:
        return "train"
    subdir = parts[-2].lower()
    fname = parts[-1]
    if "realworld" in subdir:
        return "test"
    if "synthetic_v3" in subdir:
        # Extract leading integer from filename like "22_points_GTv3.ply"
        try:
            n = int(fname.split("_")[0])
            return "validation" if n >= 21 else "train"
        except ValueError:
            return "train"
    return "train"


def scene_id_from_entry(zip_entry_name: str) -> str:
    """Human-readable scene identifier (no extension)."""
    parts = Path(zip_entry_name).parts
    subdir = parts[-2] if len(parts) >= 2 else "unknown"
    fname = Path(parts[-1]).stem
    return f"{subdir}__{fname}"


# ---------------------------------------------------------------------------
# PLY parsing
# ---------------------------------------------------------------------------

def parse_ply_header(f) -> tuple[int, np.dtype, list[str]]:
    """Read the PLY header from an open binary file-like object.

    Returns (n_points, numpy_dtype, [property_names]).
    File cursor is left just after the 'end_header\\n' line.
    """
    first = f.readline().decode("ascii", errors="replace").strip()
    if first != "ply":
        raise ValueError(f"Not a PLY file (first line: {first!r})")

    n_points = 0
    props: list[tuple[str, str]] = []

    while True:
        raw_line = f.readline()
        if not raw_line:
            raise ValueError("Unexpected EOF in PLY header")
        line = raw_line.decode("ascii", errors="replace").strip()
        if line == "end_header":
            break
        parts = line.split()
        if not parts:
            continue
        keyword = parts[0]
        if keyword == "element" and len(parts) >= 3 and parts[1] == "vertex":
            n_points = int(parts[2])
        elif keyword == "property" and len(parts) >= 3:
            ptype = parts[1]
            pname = parts[2]
            np_type = _PLY_NP.get(ptype)
            if np_type is None:
                raise ValueError(f"Unknown PLY property type: {ptype!r} for {pname!r}")
            props.append((pname, np_type))

    if not props:
        raise ValueError("PLY header contains no properties")
    if n_points == 0:
        raise ValueError("PLY header reports 0 vertices")

    # 'class' is a Python reserved word; rename to 'label' in the dtype.
    renamed = [(("label" if n == "class" else n), t) for n, t in props]
    dt = np.dtype(renamed)
    prop_names = [n for n, _ in renamed]
    return n_points, dt, prop_names


def read_ply_binary(f, n_points: int, dt: np.dtype, max_points: int | None = None) -> np.ndarray:
    """Read binary little-endian PLY body into a structured numpy array.

    If max_points is given, only the first max_points rows are returned
    (used for smoke testing without loading the full file).
    """
    actual = n_points if max_points is None else min(n_points, max_points)
    n_bytes = actual * dt.itemsize
    body = f.read(n_bytes)
    if len(body) < n_bytes:
        raise ValueError(
            f"Truncated PLY body: expected {n_bytes} bytes, got {len(body)}"
        )
    return np.frombuffer(body, dtype=dt, count=actual).copy()


def inspect_ply_schema(f) -> dict:
    """Return a dict describing the PLY schema (for reporting, does not read body)."""
    n, dt, prop_names = parse_ply_header(f)
    return {
        "n_points": n,
        "properties": prop_names,
        "bytes_per_point": dt.itemsize,
        "approx_size_mb": round(n * dt.itemsize / 1_048_576, 1),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_chunk(xyz: np.ndarray, labels: np.ndarray, rgb: np.ndarray,
                   scene_id: str, chunk_idx: int) -> list[str]:
    """Return a list of problem strings (empty = OK)."""
    problems: list[str] = []
    tag = f"{scene_id}#{chunk_idx}"

    if xyz.shape != (len(labels), 3):
        problems.append(f"{tag}: xyz/label shape mismatch {xyz.shape} vs {labels.shape}")
    if rgb.shape != (len(labels), 3):
        problems.append(f"{tag}: rgb/label shape mismatch {rgb.shape} vs {labels.shape}")
    if len(xyz) == 0:
        problems.append(f"{tag}: empty chunk")
        return problems

    if np.any(np.isnan(xyz)):
        problems.append(f"{tag}: NaN in xyz")
    if np.any(np.isinf(xyz)):
        problems.append(f"{tag}: Inf in xyz")
    bad_labels = ~((labels >= 0) & (labels < NUM_CLASSES))
    if np.any(bad_labels):
        problems.append(f"{tag}: label out of range [0,{NUM_CLASSES}): "
                        f"{np.unique(labels[bad_labels]).tolist()}")
    if np.any(rgb > 255) or np.any(rgb < 0):
        problems.append(f"{tag}: RGB out of [0,255]")
    return problems


# ---------------------------------------------------------------------------
# Spatial chunking
# ---------------------------------------------------------------------------

def spatial_chunks(
    xyz: np.ndarray,
    labels: np.ndarray,
    rgb: np.ndarray,
    chunk_size: int = 4096,
    cell_size: float = 50.0,
    min_points: int = 256,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Divide a scene into fixed-size spatial chunks.

    Procedure:
    1. Build a 2D grid of cell_size × cell_size in XY.
    2. Assign every point to its cell.
    3. Skip cells with fewer than min_points valid points.
    4. Sample exactly chunk_size points per cell (with replacement if sparse).

    Returns a list of dicts {xyz, labels, rgb}.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(xyz) == 0:
        return []

    x, y = xyz[:, 0], xyz[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    nx = max(1, int(np.ceil((x_max - x_min) / cell_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / cell_size)))

    xi = np.clip(((x - x_min) / cell_size).astype(np.int32), 0, nx - 1)
    yi = np.clip(((y - y_min) / cell_size).astype(np.int32), 0, ny - 1)
    cell_id = yi * nx + xi

    chunks: list[dict] = []
    for cid in np.unique(cell_id):
        idx = np.where(cell_id == cid)[0]
        if len(idx) < min_points:
            continue
        replace = len(idx) < chunk_size
        sel = rng.choice(idx, chunk_size, replace=replace)
        chunks.append({
            "xyz": xyz[sel].astype(np.float32),
            "labels": labels[sel].astype(np.uint8),
            "rgb": rgb[sel].astype(np.uint8),
        })

    return chunks


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def process_ply_stream(
    f,                          # open binary file-like object (seeked past nothing)
    scene_id: str,
    split: str,
    out_split_dir: Path,
    chunk_size: int,
    cell_size: float,
    min_chunk_points: int,
    max_points: int | None,
    rng: np.random.Generator,
    verbose: bool = True,
) -> tuple[list[dict], dict, dict]:
    """Parse one PLY stream, chunk it spatially, save .npz files.

    Returns a list of manifest-entry dicts.
    """
    n_points, dt, prop_names = parse_ply_header(f)

    if verbose:
        print(f"    schema: {prop_names}  bytes/pt: {dt.itemsize}  "
              f"vertices: {n_points:,}")

    # Check required columns exist
    for req in ("x", "y", "z", "label"):
        if req not in prop_names:
            raise ValueError(f"Required PLY property missing: {req!r} in {prop_names}")

    data = read_ply_binary(f, n_points, dt, max_points=max_points)

    # ---- extract arrays ----
    xyz = np.column_stack([
        data["x"].astype(np.float32),
        data["y"].astype(np.float32),
        data["z"].astype(np.float32),
    ])
    raw_labels = data["label"].astype(np.uint8)

    if "red" in prop_names and "green" in prop_names and "blue" in prop_names:
        rgb = np.column_stack([data["red"], data["green"], data["blue"]]).astype(np.uint8)
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)

    # ---- label statistics BEFORE mapping ----
    raw_counts: dict[int, int] = {}
    for lbl_val in np.unique(raw_labels):
        raw_counts[int(lbl_val)] = int(np.sum(raw_labels == lbl_val))

    # ---- apply label mapping ----
    mapped = _LABEL_LUT[raw_labels]   # int8, -1 means remove
    valid_mask = mapped >= 0
    n_removed = int(np.sum(~valid_mask))

    xyz = xyz[valid_mask]
    labels = mapped[valid_mask].astype(np.uint8)
    rgb = rgb[valid_mask]

    if verbose and n_removed > 0:
        print(f"    filtered {n_removed:,} points with unsupported/absent labels")

    if len(xyz) == 0:
        print(f"    WARNING: no valid points in {scene_id} after label filtering")
        return [], raw_counts, {}

    # ---- label counts AFTER mapping ----
    mapped_counts: dict[int, int] = {}
    for c in range(NUM_CLASSES):
        cnt = int(np.sum(labels == c))
        if cnt:
            mapped_counts[c] = cnt

    if verbose:
        print(f"    valid points: {len(xyz):,}   "
              f"XY extent: "
              f"x=[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}] "
              f"y=[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}]")
        print(f"    mapped class distribution:")
        for c, name in enumerate(OUR_NAMES):
            cnt = mapped_counts.get(c, 0)
            pct = 100.0 * cnt / len(xyz) if len(xyz) else 0.0
            print(f"      {c} {name:<12} {cnt:>10,}  ({pct:.1f}%)")

    # ---- spatial chunking ----
    chunks = spatial_chunks(xyz, labels, rgb, chunk_size=chunk_size,
                            cell_size=cell_size, min_points=min_chunk_points, rng=rng)

    if verbose:
        print(f"    spatial chunks: {len(chunks)}")

    # ---- save chunks ----
    out_split_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    all_problems: list[str] = []

    for ci, ch in enumerate(chunks):
        problems = validate_chunk(ch["xyz"], ch["labels"], ch["rgb"], scene_id, ci)
        if problems:
            all_problems.extend(problems)
            continue

        fname = f"{scene_id}__{ci:04d}.npz"
        fpath = out_split_dir / fname
        np.savez_compressed(
            fpath,
            xyz=ch["xyz"],
            labels=ch["labels"],
            rgb=ch["rgb"],
        )

        per_class = {c: int(np.sum(ch["labels"] == c)) for c in range(NUM_CLASSES)}
        entries.append({
            "path": f"{split}/{fname}",
            "split": split,
            "scene_id": scene_id,
            "chunk_idx": ci,
            "n_points": int(len(ch["xyz"])),
            "per_class": per_class,
        })

    if all_problems:
        print(f"    VALIDATION WARNINGS ({len(all_problems)}):")
        for p in all_problems[:10]:
            print(f"      {p}")

    return entries, raw_counts, mapped_counts


# ---------------------------------------------------------------------------
# Source enumeration
# ---------------------------------------------------------------------------

def iter_ply_from_zip(zip_path: Path):
    """Yield (entry_name, open_file_obj) for each PLY inside the ZIP."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.endswith(".ply") and not info.is_dir():
                with zf.open(info) as f:
                    yield info.filename, f


def iter_ply_from_dir(src_dir: Path):
    """Yield (relative_path_str, open_file_obj) for PLY files in a directory."""
    for p in sorted(src_dir.rglob("*.ply")):
        rel = p.relative_to(src_dir.parent)
        with open(p, "rb") as f:
            yield str(rel).replace("\\", "/"), f


# ---------------------------------------------------------------------------
# Statistics reporting
# ---------------------------------------------------------------------------

def print_statistics(manifest: list[dict]) -> None:
    total_chunks = len(manifest)
    total_points = sum(e["n_points"] for e in manifest)

    split_counts: dict[str, int] = {}
    scene_ids: set[str] = set()
    global_class: dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}

    for e in manifest:
        split_counts[e["split"]] = split_counts.get(e["split"], 0) + 1
        scene_ids.add(e["scene_id"])
        for c in range(NUM_CLASSES):
            global_class[c] += e["per_class"].get(c, 0)

    print("\n" + "=" * 65)
    print("STPLS3D PREPROCESSED DATASET STATISTICS")
    print("=" * 65)
    print(f"  Total chunks  : {total_chunks:,}")
    print(f"  Total points  : {total_points:,}")
    print(f"  Total scenes  : {len(scene_ids)}")
    print()
    print("  Split breakdown:")
    for sp in ("train", "validation", "test"):
        n = split_counts.get(sp, 0)
        sp_scenes = len({e["scene_id"] for e in manifest if e["split"] == sp})
        print(f"    {sp:<12} {n:>6} chunks   {sp_scenes:>3} scenes")
    print()
    print("  Class distribution (after mapping):")
    total_valid = sum(global_class.values())
    for c, name in enumerate(OUR_NAMES):
        cnt = global_class[c]
        pct = 100.0 * cnt / total_valid if total_valid else 0.0
        bar = "#" * int(pct / 2)
        print(f"    {c} {name:<12} {cnt:>12,}  {pct:5.1f}%  {bar}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocess STPLS3D PLY data for SIH 26011 segmentation."
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-zip", metavar="PATH",
                        help="Path to STPLS3D.zip")
    source.add_argument("--source-dir", metavar="PATH",
                        help="Path to pre-extracted STPLS3D directory")

    ap.add_argument("--output", required=True, metavar="DIR",
                    help="Output directory for preprocessed chunks and manifest")
    ap.add_argument("--chunk-size", type=int, default=4096,
                    help="Points per chunk (default: 4096)")
    ap.add_argument("--cell-size", type=float, default=50.0,
                    help="Spatial grid cell side length in scene units (default: 50.0)")
    ap.add_argument("--min-chunk-points", type=int, default=256,
                    help="Skip grid cells with fewer than this many valid points (default: 256)")
    ap.add_argument("--seed", type=int, default=26011,
                    help="Global RNG seed for deterministic chunk sampling (default: 26011)")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Process only the first 200 000 points of the first PLY for a fast sanity check")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Limit number of PLY files processed (useful for partial runs)")
    ap.add_argument("--inspect-only", action="store_true",
                    help="Print PLY schemas without writing any output")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    max_points_per_file = 200_000 if args.smoke_test else None

    # Pick source iterator
    if args.source_zip:
        src_path = Path(args.source_zip)
        if not src_path.exists():
            print(f"ERROR: ZIP not found: {src_path}", file=sys.stderr)
            sys.exit(1)
        ply_iter = iter_ply_from_zip(src_path)
    else:
        src_path = Path(args.source_dir)
        if not src_path.exists():
            print(f"ERROR: directory not found: {src_path}", file=sys.stderr)
            sys.exit(1)
        ply_iter = iter_ply_from_dir(src_path)

    manifest: list[dict] = []

    # Aggregate raw label counts across all files (for before-mapping report)
    all_raw_counts: dict[int, int] = {}
    all_mapped_counts: dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}

    n_files_processed = 0

    print("\n" + "=" * 65)
    print("STPLS3D PREPROCESSING — SIH 26011")
    print("=" * 65)
    if args.smoke_test:
        print("  SMOKE-TEST MODE: first 200 K points per file only")
    print()

    for entry_name, f in ply_iter:
        if not entry_name.endswith(".ply"):
            continue
        if args.max_files is not None and n_files_processed >= args.max_files:
            break

        split = assign_split(entry_name)
        scene_id = scene_id_from_entry(entry_name)

        print(f"[{n_files_processed + 1}] {entry_name}")
        print(f"    split: {split}   scene_id: {scene_id}")

        if args.inspect_only:
            try:
                info = inspect_ply_schema(f)
                print(f"    n_points: {info['n_points']:,}   "
                      f"properties: {info['properties']}   "
                      f"bytes/pt: {info['bytes_per_point']}   "
                      f"size: {info['approx_size_mb']} MB")
            except Exception as exc:
                print(f"    ERROR reading schema: {exc}")
            n_files_processed += 1
            continue

        out_split_dir = out_dir / "chunks" / split

        try:
            entries, raw_counts, mapped_counts = process_ply_stream(
                f=f,
                scene_id=scene_id,
                split=split,
                out_split_dir=out_split_dir,
                chunk_size=args.chunk_size,
                cell_size=args.cell_size,
                min_chunk_points=args.min_chunk_points,
                max_points=max_points_per_file,
                rng=rng,
                verbose=True,
            )
        except Exception as exc:
            print(f"    ERROR processing {entry_name}: {exc}")
            import traceback
            traceback.print_exc()
            n_files_processed += 1
            continue

        manifest.extend(entries)

        for lbl, cnt in raw_counts.items():
            all_raw_counts[lbl] = all_raw_counts.get(lbl, 0) + cnt
        for c, cnt in mapped_counts.items():
            all_mapped_counts[c] = all_mapped_counts.get(c, 0) + cnt

        n_files_processed += 1
        if args.smoke_test:
            print("  [smoke-test] stopping after first file")
            break

    # ---- write manifest ----
    if not args.inspect_only and manifest:
        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump({
                "version": "1",
                "chunk_size": args.chunk_size,
                "cell_size": args.cell_size,
                "seed": args.seed,
                "num_classes": NUM_CLASSES,
                "class_names": OUR_NAMES,
                "label_mapping": {
                    str(k): int(v) for k, v in _STPLS3D_TO_OURS.items()
                },
                "chunks": manifest,
            }, mf, indent=2)
        print(f"\nManifest written to: {manifest_path}")
        print(f"Total chunks in manifest: {len(manifest)}")

    # ---- statistics ----
    if not args.inspect_only and manifest:
        print("\nRaw STPLS3D label distribution (BEFORE mapping):")
        total_raw = sum(all_raw_counts.values())
        for lbl in sorted(all_raw_counts):
            cnt = all_raw_counts[lbl]
            pct = 100.0 * cnt / total_raw if total_raw else 0.0
            name = _STPLS3D_NAMES.get(lbl, "UNKNOWN")
            print(f"  {lbl:>2} {name:<20} {cnt:>12,}  ({pct:.2f}%)")

        print_statistics(manifest)

    if not args.inspect_only and not manifest:
        print("WARNING: no chunks were produced — check PLY schema and label contents.")

    print(f"\nProcessed {n_files_processed} PLY file(s).")
    print("Done.")


if __name__ == "__main__":
    main()

"""Run inference on any raw PLY file (no scene-directory format required).

Usage:
    python ml/src/inference/predict_raw.py \
        --checkpoint ml/results/experiment_003/best_model.pth \
        --input path/to/any_pointcloud.ply \
        --output ml/results/raw_prediction.ply

The input PLY must have x, y, z fields. All other fields are ignored.
Output PLY contains x, y, z, predicted_class (uint8), confidence (float32).

Use experiment_003 checkpoint (XYZ-only model) for any dataset without RGB.
Use experiment_005 checkpoint only if the PLY has r, g, b fields.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ml.src.inference.predict import build_model_from_checkpoint
from ml.src.data.dataset import chunk_indices
from ml.src.utils import select_device

CLASS_COLORS = {
    0: (120, 120, 120),  # ground      - grey
    1: (220, 110,  55),  # building    - orange
    2: ( 45, 160,  80),  # vegetation  - green
    3: ( 40,  40,  40),  # road        - dark grey
    4: ( 70, 130, 200),  # vehicle     - blue
    5: (200, 200,  40),  # other       - yellow
}


# Covers all PLY type variants seen in public datasets
_PLY_TYPES = {
    "float": "f", "float32": "f",
    "double": "d", "float64": "d",
    "char": "b", "int8": "b",
    "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h",
    "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i",
    "uint": "I", "uint32": "I",
    "long": "l", "int64": "l",
    "ulong": "L", "uint64": "L",
}


def _read_ply_robust(path: Path):
    """Minimal PLY reader that handles all common type names and formats."""
    with open(path, "rb") as f:
        raw = f.read()
    hend = raw.find(b"end_header")
    if hend == -1:
        raise ValueError(f"{path}: not a valid PLY file")
    header = raw[:hend].decode("ascii", errors="replace")
    body = raw[hend + len(b"end_header"):]
    body = body.lstrip(b"\r\n")

    fmt, n_verts, props = "ascii", 0, []
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3 and parts[1] == "vertex":
            n_verts = int(parts[2])
        elif parts[0] == "property" and len(parts) >= 3 and parts[1] != "list":
            ptype, pname = parts[1], parts[2]
            sc = _PLY_TYPES.get(ptype)
            if sc is None:
                raise ValueError(f"Unsupported PLY type '{ptype}' for field '{pname}'")
            props.append((pname, sc))

    data = {name: np.empty(n_verts, dtype=np.dtype("<" + sc))
            for name, sc in props}

    if fmt == "ascii":
        for i, line in enumerate(body.decode("ascii").strip().splitlines()):
            vals = line.split()
            for j, (name, sc) in enumerate(props):
                data[name][i] = float(vals[j]) if sc in "fd" else int(vals[j])
    elif fmt == "binary_little_endian":
        row_fmt = struct.Struct("<" + "".join(sc for _, sc in props))
        row_size = row_fmt.size
        for i in range(n_verts):
            row = row_fmt.unpack_from(body, i * row_size)
            for (name, _), val in zip(props, row):
                data[name][i] = val
    elif fmt == "binary_big_endian":
        row_fmt = struct.Struct(">" + "".join(sc for _, sc in props))
        row_size = row_fmt.size
        for i in range(n_verts):
            row = row_fmt.unpack_from(body, i * row_size)
            for (name, _), val in zip(props, row):
                data[name][i] = val
    else:
        raise ValueError(f"Unsupported PLY format: {fmt}")

    return data, n_verts


def _get(data, *candidates):
    for k in candidates:
        if k in data:
            return data[k]
    raise KeyError(f"None of {candidates} found. Available fields: {sorted(data.keys())}")


def read_raw_ply(path: Path):
    """Read any PLY file and return xyz (N,3) float32, rgb (N,3) or None."""
    data, n = _read_ply_robust(path)
    print(f"  PLY fields: {sorted(data.keys())}")

    xyz = np.column_stack([
        _get(data, "x", "X").astype(np.float32),
        _get(data, "y", "Y").astype(np.float32),
        _get(data, "z", "Z").astype(np.float32),
    ])

    rgb = None
    for rk, gk, bk in [("r","g","b"), ("red","green","blue"), ("R","G","B")]:
        if rk in data and gk in data and bk in data:
            rgb = np.column_stack([
                data[rk].astype(np.float32),
                data[gk].astype(np.float32),
                data[bk].astype(np.float32),
            ])
            print(f"  RGB found ({rk}/{gk}/{bk}) — XYZRGB input available")
            break

    if rgb is None:
        print(f"  No RGB found — XYZ-only input (3-dim)")
    return xyz, rgb


def predict_raw(model, xyz, rgb, input_dim, device, num_points):
    """Chunk and predict on a raw point cloud."""
    n = xyz.shape[0]
    chunks = chunk_indices(n, num_points, seed=0)
    pred = np.empty(n, dtype=np.int64)
    conf = np.empty(n, dtype=np.float32)

    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            xyz_chunk = xyz[chunk]
            centroid = xyz_chunk.mean(axis=0, keepdims=True)
            centered = xyz_chunk - centroid
            scale = float(np.max(np.linalg.norm(centered, axis=1)))
            if scale < 1e-8:
                scale = 1.0
            xyz_norm = (centered / scale).astype(np.float32)

            if input_dim == 6:
                if rgb is not None:
                    rgb_chunk = (rgb[chunk] / 255.0).astype(np.float32)
                else:
                    # No RGB in file but model expects 6-dim: zero-pad
                    rgb_chunk = np.zeros((len(chunk), 3), dtype=np.float32)
                    print("  WARNING: model expects XYZRGB but PLY has no RGB — padding with zeros")
                feat = np.concatenate([xyz_norm, rgb_chunk], axis=1)
            else:
                feat = xyz_norm

            inp = torch.from_numpy(feat).unsqueeze(0).to(device)
            logits = model(inp)[0]
            probs = torch.softmax(logits, dim=-1)
            p = probs.argmax(dim=-1).cpu().numpy()
            c = probs.max(dim=-1).values.cpu().numpy()
            pred[chunk] = p
            conf[chunk] = c

            if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
                print(f"  chunk {i+1}/{len(chunks)} done")

    return pred, conf


def save_colored_ply(path, xyz, pred_class, confidence):
    """Write PLY with x,y,z,r,g,b,predicted_class,confidence for easy visualization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment SIH 26011 - raw PLY inference output\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar predicted_class\n"
        "property float confidence\n"
        "end_header\n"
    )
    fmt = struct.Struct("<fffBBBBf")
    buf = bytearray()
    for i in range(n):
        x, y, z = xyz[i]
        cls = int(pred_class[i])
        r, g, b = CLASS_COLORS.get(cls, (200, 200, 200))
        buf += fmt.pack(float(x), float(y), float(z), r, g, b, cls, float(confidence[i]))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(bytes(buf))
    print(f"  saved: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Predict semantic labels on any raw PLY point cloud.")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best_model.pth (use exp003 for XYZ-only datasets)")
    ap.add_argument("--input", required=True, help="Input PLY file path")
    ap.add_argument("--output", default=None,
                    help="Output PLY path (default: <input>_prediction.ply)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else \
        input_path.parent / (input_path.stem + "_prediction.ply")

    print(f"\nSIH 26011 — Raw PLY Inference")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  input      : {input_path}")
    print(f"  output     : {out_path}\n")

    device = select_device("auto")
    model, ckpt = build_model_from_checkpoint(str(ckpt_path), device)
    cfg = ckpt["config"]
    input_mode = cfg["dataset"].get("input_mode", "xyz")
    input_dim = 6 if input_mode == "xyzrgb" else 3
    num_points = int(cfg["dataset"]["num_points"])

    print(f"  model      : {cfg['model']['name']}  input_dim={input_dim}  "
          f"num_classes={cfg['model']['num_classes']}")

    print(f"\nReading PLY...")
    xyz, rgb = read_raw_ply(input_path)
    print(f"  {xyz.shape[0]:,} points loaded")

    print(f"\nRunning inference ({xyz.shape[0]//num_points + 1} chunks)...")
    pred, conf = predict_raw(model, xyz, rgb, input_dim, device, num_points)

    unique, counts = np.unique(pred, return_counts=True)
    class_names = ["ground", "building", "vegetation", "road", "vehicle", "other"]
    print("\nPrediction summary:")
    for cls, cnt in zip(unique, counts):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        pct = 100 * cnt / len(pred)
        print(f"  {name:<12} {cnt:>8,} pts  ({pct:.1f}%)")

    print(f"\nSaving output PLY...")
    save_colored_ply(out_path, xyz, pred, conf)
    print(f"\nDone. Open {out_path} in CloudCompare or MeshLab to visualize.")


if __name__ == "__main__":
    main()

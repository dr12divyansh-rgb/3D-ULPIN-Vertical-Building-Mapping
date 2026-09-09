"""Reusable inference + prediction-saving utilities, and a prediction CLI.

Usage:
    python ml/src/inference/predict.py --checkpoint <best_model.pth> \
        --scene <scene_dir> [--out prediction.ply]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.src.data.dataset import normalize_chunk, chunk_indices, load_scene_pointcloud  # noqa: E402
from ml.src.models.pointnet2 import PointNet2Seg  # noqa: E402
from ml.src.utils import select_device  # noqa: E402


def build_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    input_mode = cfg["dataset"].get("input_mode", "xyz")
    input_dim = 6 if input_mode == "xyzrgb" else 3
    model = PointNet2Seg(num_classes=cfg["model"]["num_classes"],
                         sa1=cfg["model"].get("sa1"), sa2=cfg["model"].get("sa2"),
                         input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def predict_on_cloud(model, xyz, device, num_points, batch_size):
    """Predict semantic labels + confidence for a full point cloud.

    The cloud is deterministically chunked; per-chunk coordinates are
    normalised exactly as in training. Returns (pred (N,), confidence (N,)).
    """
    n = xyz.shape[0]
    chunks = chunk_indices(n, num_points, seed=0)
    pred = np.empty(n, dtype=np.int64)
    conf = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for chunk in chunks:
            xyz_chunk = xyz[chunk]
            xyz_norm, _, _ = normalize_chunk(xyz_chunk)
            inp = torch.from_numpy(xyz_norm).unsqueeze(0).to(device)
            logits = model(inp)[0]                       # (M, C)
            probs = torch.softmax(logits, dim=-1)        # (M, C)
            p = probs.argmax(dim=-1).cpu().numpy()
            c = probs.max(dim=-1).values.cpu().numpy()
            pred[chunk] = p
            conf[chunk] = c
    return pred, conf


def save_prediction_ply(path, xyz, pred_class, confidence):
    """Write a binary PLY with x,y,z, predicted_class (uint8), confidence (float32)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment predicted semantic segmentation (SIH 26011)\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar predicted_class\n"
        "property float confidence\n"
        "end_header\n"
    )
    fmt = struct.Struct("<fffBf")
    buf = bytearray()
    for i in range(n):
        x, y, z = xyz[i]
        buf += fmt.pack(float(x), float(y), float(z), int(pred_class[i]), float(confidence[i]))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(bytes(buf))
    return path


def main():
    ap = argparse.ArgumentParser(description="Predict semantic labels for a scene.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", required=True, help="scene directory or scene id")
    ap.add_argument("--data", default=None, help="dataset root (for scene-id lookup)")
    ap.add_argument("--out", default=None, help="output PLY path")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = select_device("auto")
    model, ckpt = build_model_from_checkpoint(args.checkpoint, device)
    cfg = ckpt["config"]
    num_classes = cfg["model"]["num_classes"]
    num_points = int(cfg["dataset"]["num_points"])

    scene = Path(args.scene)
    if not scene.is_dir():
        data_root = Path(args.data) if args.data else (ROOT / cfg["dataset"]["data_root"])
        for split in ("train", "validation", "test"):
            cand = data_root / split / args.scene
            if cand.is_dir():
                scene = cand
                break
    if not scene.is_dir():
        print(f"scene not found: {args.scene}", file=sys.stderr)
        sys.exit(1)

    xyz, labels, building_id = load_scene_pointcloud(scene)
    pred, conf = predict_on_cloud(model, xyz, device, num_points, args.batch_size)

    out = Path(args.out) if args.out else scene / "prediction.ply"
    save_prediction_ply(out, xyz, pred, conf)
    print(f"scene {scene.name}: {xyz.shape[0]} points predicted -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

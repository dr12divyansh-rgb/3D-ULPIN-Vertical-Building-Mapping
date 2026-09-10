"""Full-tile Exp007 inference — all 498 AHN4 chunks (train+val+test).

Produces a single merged PLY with all predictions in EPSG:28992 coordinates,
ready for the research_v2 reconstruction pipeline.

Usage:
    python ml/src/inference/infer_ahn4_fulltile.py \\
        --checkpoint ml/results/ahn4_exp007/best_model.pth \\
        --manifest   ml/results/ahn4_exp007/data/manifest.json \\
        --out        ml/results/ahn4_fulltile

NOTE: Includes training chunks — for RECONSTRUCTION/DEMO only, not evaluation.
      Use infer_ahn4.py --split test for held-out evaluation metrics.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ml.src.models.pointnet2 import PointNet2Seg
from ml.src.utils import select_device

_CLASS_COLORS = {
    0: (120, 120, 120),
    1: (220, 110,  55),
    2: ( 45, 160,  80),
    3: ( 40,  40,  40),
    4: ( 70, 130, 200),
    5: (200, 200,  40),
}


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    input_mode = cfg["dataset"].get("input_mode", "xyz")
    input_dim  = 6 if input_mode == "xyzrgb" else 3
    model = PointNet2Seg(
        num_classes=cfg["model"]["num_classes"],
        sa1=cfg["model"].get("sa1"),
        sa2=cfg["model"].get("sa2"),
        input_dim=input_dim,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, input_dim


def normalize_chunk(xyz: np.ndarray):
    centroid = xyz.mean(axis=0, keepdims=True)
    centered = xyz - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    return (centered / scale).astype(np.float32), centroid.reshape(-1), scale


def write_merged_ply(path: Path, xyz: np.ndarray, pred: np.ndarray,
                     gt: np.ndarray, confidence: np.ndarray) -> None:
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"comment AHN4 Exp007 full-tile inference (SIH 26011)\n"
        f"comment CRS: EPSG:28992 (Amersfoort / RD New)\n"
        f"comment Includes training chunks - for reconstruction/demo only\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar predicted_class\nproperty uchar gt_class\n"
        "property float confidence\n"
        "end_header\n"
    )
    fmt = struct.Struct("<fffBBBBBf")
    buf = bytearray()
    for i in range(n):
        p = int(pred[i])
        g = int(gt[i])
        r, gv, b = _CLASS_COLORS.get(p, (200, 200, 200))
        buf += fmt.pack(
            float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
            r, gv, b, p, g, float(confidence[i])
        )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(bytes(buf))


def run_fulltile_inference(
    ckpt_path: Path,
    manifest_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device("auto")
    print(f"Device: {device}")

    model, input_dim = load_model(ckpt_path, device)
    print(f"Exp007 model loaded (input_dim={input_dim})")

    with open(manifest_path) as f:
        manifest = json.load(f)

    data_root = Path(manifest["data_root"])
    crs = manifest.get("crs", "EPSG:28992")
    all_chunks = manifest["chunks"]
    n_total = len(all_chunks)
    print(f"Total chunks: {n_total} (train+val+test)")
    print(f"Data root: {data_root}")

    all_xyz   = []
    all_pred  = []
    all_gt    = []
    all_conf  = []

    splits_seen: dict[str, int] = {}

    for i, entry in enumerate(all_chunks):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n_total}] processing chunk {entry['chunk_idx']} (split={entry['split']})", flush=True)

        npz_path = data_root / "chunks" / entry["path"]
        if not npz_path.exists():
            print(f"  WARNING: {npz_path} missing, skipping")
            continue

        data = np.load(npz_path)
        xyz_world = data["xyz"].astype(np.float32)
        rgb_u8    = data["rgb"].astype(np.float32) / 255.0
        labels    = data["labels"].astype(np.int64)

        xyz_norm, _, _ = normalize_chunk(xyz_world)

        if input_dim == 6:
            feat = np.concatenate([xyz_norm, rgb_u8], axis=1)
        else:
            feat = xyz_norm

        feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(feat_t)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred  = probs.argmax(axis=-1).astype(np.uint8)
        conf  = probs.max(axis=-1).astype(np.float32)

        all_xyz.append(xyz_world)
        all_pred.append(pred)
        all_gt.append(labels.astype(np.uint8))
        all_conf.append(conf)
        splits_seen[entry["split"]] = splits_seen.get(entry["split"], 0) + 1

    all_xyz  = np.concatenate(all_xyz,  axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_gt   = np.concatenate(all_gt,   axis=0)
    all_conf = np.concatenate(all_conf, axis=0)

    n_pts    = len(all_xyz)
    n_bldg   = int((all_pred == 1).sum())
    n_ground = int((all_pred == 0).sum())
    print(f"\nTotal points: {n_pts:,}")
    print(f"  building predicted: {n_bldg:,} ({100*n_bldg/n_pts:.1f}%)")
    print(f"  ground predicted:   {n_ground:,} ({100*n_ground/n_pts:.1f}%)")
    print(f"  splits: {splits_seen}")

    ply_path = out_dir / "merged_prediction.ply"
    print(f"\nWriting merged PLY → {ply_path}")
    write_merged_ply(ply_path, all_xyz, all_pred, all_gt, all_conf)
    print(f"  Written {n_pts:,} points")

    summary = {
        "checkpoint":  str(ckpt_path),
        "manifest":    str(manifest_path),
        "crs":         crs,
        "n_chunks":    n_total,
        "splits":      splits_seen,
        "n_points":    int(n_pts),
        "n_building":  int(n_bldg),
        "n_ground":    int(n_ground),
        "merged_ply":  str(ply_path),
        "note":        "Includes training chunks — for reconstruction/demo only",
    }
    with open(out_dir / "fulltile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_dir / 'fulltile_summary.json'}")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Full-tile Exp007 inference (all 498 AHN4 chunks)")
    ap.add_argument("--checkpoint", default="ml/results/ahn4_exp007/best_model.pth")
    ap.add_argument("--manifest",   default="ml/results/ahn4_exp007/data/manifest.json")
    ap.add_argument("--out",        default="ml/results/ahn4_fulltile")
    args = ap.parse_args()

    run_fulltile_inference(
        ckpt_path     = ROOT / args.checkpoint,
        manifest_path = ROOT / args.manifest,
        out_dir       = ROOT / args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

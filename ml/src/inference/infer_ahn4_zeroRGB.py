"""AHN4 inference with Exp006 using zero RGB (immediate workaround).

ROOT CAUSE FINDING (Phase 1+2 diagnosis):
- DALES LAZ had all-zero RGB (no camera on scanner)
- Exp006 was fine-tuned on zero RGB -> model expects RGB=0
- AHN4 has real RGB (mean ~113) -> mismatch causes 83% vehicle prediction
- Zeroing AHN4 RGB before inference restores Building IoU from 0.053 -> 0.769

This script is the immediate workaround to use BEFORE Exp007 finishes training.
Use Exp007 (ml/results/ahn4_exp007/best_model.pth) once available for a model
that properly uses real RGB.

Usage:
    python ml/src/inference/infer_ahn4_zeroRGB.py \\
        --checkpoint ml/results/experiment_006/best_model.pth \\
        --manifest   ml/results/ahn4_exp007/data/manifest.json \\
        --split      test \\
        --out        ml/results/ahn4_exp007/exp006_zeroRGB
"""
from __future__ import annotations
import argparse, csv, json, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.src.models.pointnet2 import PointNet2Seg
from ml.src.training.metrics import segmentation_metrics
from ml.src.training.finetune_ahn4 import AHN4SplitDataset, normalize_chunk
from ml.src.utils import select_device

CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]
NUM_CLASSES = 6
CLASS_COLORS = {
    0: (120, 120, 120), 1: (220, 110, 55), 2: (45, 160, 80),
    3: (40, 40, 40),    4: (70, 130, 200), 5: (200, 200, 40),
}


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    input_dim = 6 if cfg["dataset"].get("input_mode", "xyz") == "xyzrgb" else 3
    m = PointNet2Seg(num_classes=NUM_CLASSES, sa1=cfg["model"].get("sa1"),
                     sa2=cfg["model"].get("sa2"), input_dim=input_dim)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    return m, input_dim, cfg


def write_ply(path, xyz, pred, gt, conf):
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = (f"ply\nformat binary_little_endian 1.0\n"
           f"comment AHN4 Exp006 zero-RGB inference. CRS EPSG:28992\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "property uchar predicted_class\nproperty uchar gt_class\n"
           "property float confidence\nend_header\n")
    fmt = struct.Struct("<fffBBBBBf")
    buf = bytearray()
    for i in range(n):
        p = int(pred[i]); g = int(gt[i])
        r, gv, b = CLASS_COLORS.get(p, (200, 200, 200))
        buf += fmt.pack(float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]),
                        r, gv, b, p, g, float(conf[i]))
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(bytes(buf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ml/results/experiment_006/best_model.pth")
    ap.add_argument("--manifest",   default="ml/results/ahn4_exp007/data/manifest.json")
    ap.add_argument("--split",      default="test")
    ap.add_argument("--out",        default="ml/results/ahn4_exp007/exp006_zeroRGB")
    args = ap.parse_args()

    ckpt_path = ROOT / args.checkpoint
    manifest  = ROOT / args.manifest
    out_dir   = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device("auto")
    model, input_dim, cfg = load_model(ckpt_path, device)
    print(f"\nExp006 + zero-RGB workaround inference")
    print(f"  Checkpoint: {ckpt_path.name}  input_dim={input_dim}")
    print(f"  Split: {args.split}  ->  {out_dir}")

    ds = AHN4SplitDataset(manifest, args.split)
    print(f"  Chunks: {len(ds)}")

    all_pred, all_gt, all_xyz, all_conf = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds.npz_paths[i]
            d = np.load(item)
            xyz    = d["xyz"].astype(np.float32)
            labels = d["labels"].astype(np.int64)
            xyz_norm = normalize_chunk(xyz)

            # ZERO OUT RGB — matches DALES training distribution
            rgb_zero = np.zeros((4096, 3), dtype=np.float32)
            feat = np.concatenate([xyz_norm, rgb_zero], axis=1)

            inp = torch.from_numpy(feat).unsqueeze(0).to(device)
            logits = model(inp)
            probs  = torch.softmax(logits, dim=-1)
            pred   = probs.argmax(dim=-1).cpu().numpy()[0]
            conf   = probs.max(dim=-1).values.cpu().numpy()[0]

            all_pred.append(pred)
            all_gt.append(labels)
            all_xyz.append(xyz)
            all_conf.append(conf)

            if (i+1) % 25 == 0:
                print(f"  {i+1}/{len(ds)} chunks done")

    pred_all = np.concatenate(all_pred)
    gt_all   = np.concatenate(all_gt)
    xyz_all  = np.concatenate(all_xyz)
    conf_all = np.concatenate(all_conf)

    metrics = segmentation_metrics(pred_all, gt_all, NUM_CLASSES, CLASS_NAMES)
    b = metrics["per_class"]["building"]
    g = metrics["per_class"]["ground"]

    print(f"\n{'='*60}")
    print(f"Exp006 zero-RGB on AHN4 test")
    print(f"  Building IoU:  {b['iou']:.4f}")
    print(f"  Building Prec: {b['precision']:.4f}")
    print(f"  Building Rec:  {b['recall']:.4f}")
    print(f"  Building F1:   {b['f1']:.4f}")
    print(f"  Ground IoU:    {g['iou']:.4f}")
    print(f"  Overall Acc:   {metrics['overall_accuracy']:.4f}")
    print(f"  Mean IoU:      {metrics['mean_iou']:.4f}")

    n = len(pred_all)
    print(f"\n  Prediction distribution:")
    for c, name in enumerate(CLASS_NAMES):
        cnt = int((pred_all == c).sum())
        if cnt: print(f"    {name:12}: {cnt:>7,} ({100*cnt/n:.1f}%)")
    print(f"{'='*60}")

    # Save PLY
    ply_path = out_dir / "exp006_zeroRGB_prediction.ply"
    write_ply(ply_path, xyz_all, pred_all, gt_all, conf_all)

    # Save JSON
    result = {
        "method": "Exp006 + zero RGB (DALES-condition workaround)",
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_chunks": len(ds),
        "total_points": int(len(pred_all)),
        "building_iou": float(b["iou"]),
        "building_precision": float(b["precision"]),
        "building_recall": float(b["recall"]),
        "building_f1": float(b["f1"]),
        "ground_iou": float(g["iou"]),
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "mean_iou": float(metrics["mean_iou"]),
        "note": "RGB zeroed to 0 before model input (matches DALES training conditions)",
        "prediction_ply": str(ply_path),
        "per_class": {k: {m2: float(v2) for m2, v2 in d2.items()}
                      for k, d2 in metrics["per_class"].items()},
    }
    metrics_path = out_dir / "exp006_zeroRGB_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Metrics: {metrics_path}")
    print(f"  PLY:     {ply_path}")


if __name__ == "__main__":
    main()

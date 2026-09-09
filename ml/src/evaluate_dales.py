"""Evaluate a PointNet++ checkpoint on the DALES-2 preprocessed test split.

Produces per-class IoU, precision, recall, F1, a confusion matrix, and
prediction PLY files for visual inspection.

Usage:
    # Cross-domain baseline (STPLS3D-trained model on DALES):
    python ml/src/evaluate_dales.py \\
        --checkpoint ml/results/experiment_005/best_model.pth \\
        --data       ml/data/dales_chunks \\
        --out        ml/results/dales_baseline/

    # Exp006 fine-tuned model on DALES:
    python ml/src/evaluate_dales.py \\
        --checkpoint ml/results/experiment_006/best_model.pth \\
        --data       ml/data/dales_chunks \\
        --out        ml/results/dales_exp006_eval/
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ml.src.data.dataset import STPLS3DChunkDataset
from ml.src.models.pointnet2 import PointNet2Seg
from ml.src.training.metrics import segmentation_metrics, confusion_matrix as cm_fn
from ml.src.utils import select_device

_CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]

_CLASS_COLORS = {
    0: (120, 120, 120),  # ground      grey
    1: (220, 110,  55),  # building    orange
    2: ( 45, 160,  80),  # vegetation  green
    3: ( 40,  40,  40),  # road        dark grey
    4: ( 70, 130, 200),  # vehicle     blue
    5: (200, 200,  40),  # other       yellow
}


def _load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    return model, cfg, input_dim


def _write_pred_ply(path: Path, xyz_orig: np.ndarray,
                    pred: np.ndarray, gt: np.ndarray) -> None:
    """Write a PLY with original coords + predicted class + GT class + color."""
    n = len(xyz_orig)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"comment DALES evaluation output (SIH 26011)\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar predicted_class\nproperty uchar gt_class\n"
        "end_header\n"
    )
    fmt = struct.Struct("<fffBBBBB")
    buf = bytearray()
    for i in range(n):
        p = int(pred[i])
        g = int(gt[i])
        r, gv, b = _CLASS_COLORS.get(p, (200, 200, 200))
        buf += fmt.pack(float(xyz_orig[i, 0]), float(xyz_orig[i, 1]),
                        float(xyz_orig[i, 2]), r, gv, b, p, g)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(bytes(buf))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate PointNet++ on DALES-2 test split.")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best_model.pth checkpoint.")
    ap.add_argument("--data", default="ml/data/dales_chunks",
                    help="Root directory of preprocessed DALES NPZ chunks.")
    ap.add_argument("--split", default="test",
                    help="Split to evaluate (default: test).")
    ap.add_argument("--out", required=True,
                    help="Output directory for metrics and prediction PLYs.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--save-ply", action="store_true",
                    help="Write per-chunk prediction PLY files for visualization.")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Limit evaluation to first N chunks (for quick testing).")
    args = ap.parse_args()

    ckpt_path = ROOT / args.checkpoint
    data_root = ROOT / args.data
    out_dir   = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDALES-2 Evaluation")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  data root  : {data_root}")
    print(f"  split      : {args.split}")
    print(f"  output     : {out_dir}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    device = select_device("auto")
    model, cfg, input_dim = _load_model(ckpt_path, device)
    num_classes = int(cfg["model"]["num_classes"])
    input_mode  = cfg["dataset"].get("input_mode", "xyz")
    num_points  = int(cfg["dataset"]["num_points"])

    print(f"  model      : {cfg['model']['name']}  input_dim={input_dim}  "
          f"num_classes={num_classes}")
    print(f"  input_mode : {input_mode}\n")

    # ── Load DALES test dataset ───────────────────────────────────────────────
    # Reuse STPLS3DChunkDataset — it reads any NPZ with {xyz, rgb, labels}
    # and the schema is identical.
    split_dir = data_root / args.split
    if not split_dir.exists():
        print(f"ERROR: split directory not found: {split_dir}")
        return 1

    dataset = STPLS3DChunkDataset(
        data_root=data_root,
        split=args.split,
        num_points=num_points,
        input_mode=input_mode,
    )
    if args.max_chunks:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_chunks, len(dataset))))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  Chunks     : {len(dataset)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_pred, all_gt = [], []
    ply_dir = out_dir / "predictions"

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            xyz    = batch["xyz"].to(device)
            labels = batch["labels"].numpy()

            logits = model(xyz)
            pred   = logits.argmax(dim=-1).cpu().numpy()

            all_pred.append(pred.reshape(-1))
            all_gt.append(labels.reshape(-1))

            if args.save_ply:
                xyz_orig = batch.get("xyz_orig", batch["xyz"].cpu()).numpy()
                for bi in range(len(pred)):
                    chunk_idx = batch_idx * args.batch_size + bi
                    scene_id  = batch.get("scene", ["chunk"])[bi] if isinstance(
                                batch.get("scene"), list) else f"chunk{chunk_idx:05d}"
                    ply_path = ply_dir / f"{scene_id}_{chunk_idx:05d}.ply"
                    _write_pred_ply(ply_path, xyz_orig[bi],
                                    pred[bi], labels[bi])

            if (batch_idx + 1) % 20 == 0:
                print(f"  [{batch_idx+1}/{len(loader)}] chunks evaluated...")

    pred_all = np.concatenate(all_pred)
    gt_all   = np.concatenate(all_gt)
    n_pts    = len(pred_all)

    print(f"\n  Total points evaluated: {n_pts:,}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = segmentation_metrics(pred_all, gt_all, num_classes, _CLASS_NAMES)

    print(f"\n{'='*60}")
    print(f"  Overall accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"  Mean IoU         : {metrics['mean_iou']:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<14} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7}")
    for name in _CLASS_NAMES:
        pc = metrics["per_class"].get(name, {})
        print(f"  {name:<14} "
              f"{pc.get('precision', 0):>7.4f} "
              f"{pc.get('recall',    0):>7.4f} "
              f"{pc.get('f1',        0):>7.4f} "
              f"{pc.get('iou',       0):>7.4f}")
    print(f"{'='*60}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = np.asarray(metrics["confusion_matrix"])
    cm_path = out_dir / "confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + _CLASS_NAMES[:num_classes])
        for i, row in enumerate(cm):
            w.writerow([_CLASS_NAMES[i] if i < len(_CLASS_NAMES) else str(i)]
                       + [int(v) for v in row])

    # ── Save JSON metrics ─────────────────────────────────────────────────────
    result = {
        "checkpoint":        str(ckpt_path),
        "data_root":         str(data_root),
        "split":             args.split,
        "num_chunks":        len(dataset),
        "total_points":      n_pts,
        "overall_accuracy":  float(metrics["overall_accuracy"]),
        "mean_iou":          float(metrics["mean_iou"]),
        "building_iou":      float(metrics["per_class"].get("building", {}).get("iou", 0)),
        "building_precision":float(metrics["per_class"].get("building", {}).get("precision", 0)),
        "building_recall":   float(metrics["per_class"].get("building", {}).get("recall", 0)),
        "building_f1":       float(metrics["per_class"].get("building", {}).get("f1", 0)),
        "per_class":         {k: {m: float(v) for m, v in d.items()}
                              for k, d in metrics["per_class"].items()},
    }
    metrics_path = out_dir / "dales_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Metrics saved : {metrics_path}")
    print(f"  Confusion mat : {cm_path}")
    if args.save_ply:
        print(f"  Predictions   : {ply_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

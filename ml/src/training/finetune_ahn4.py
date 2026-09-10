"""Exp007: Fine-tune Exp006 checkpoint on AHN4 (Amsterdam, Netherlands).

Starts from the Exp006 best_model.pth (STPLS3D + DALES) and adapts to
AHN4 aerial LiDAR with a small learning rate.

Key differences from zero-shot Exp006 on AHN4:
  - AHN4 has REAL RGB (mean ~113), unlike DALES (all-zero RGB).
  - Fine-tuning lets the model learn AHN4 ground/building appearances.
  - Uses spatially-split data: no leakage between train/val/test strips.

Usage:
    python ml/src/training/finetune_ahn4.py \\
        --base-checkpoint ml/results/experiment_006/best_model.pth \\
        --manifest        ml/results/ahn4_exp007/data/manifest.json \\
        --data-root       ml/results/ahn4_inference \\
        --out             ml/results/ahn4_exp007 \\
        --epochs          10 \\
        --lr              5e-5

Writes to ml/results/ahn4_exp007/:
  best_model.pth
  latest_checkpoint.pth
  history.json
  config_exp007.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml.src.models.pointnet2 import PointNet2Seg
from ml.src.training.metrics import segmentation_metrics
from ml.src.utils import select_device, set_seed

CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]
NUM_CLASSES = 6


def normalize_chunk(xyz: np.ndarray):
    centroid = xyz.mean(axis=0, keepdims=True)
    centered = xyz - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    return (centered / scale).astype(np.float32)


class AHN4SplitDataset(Dataset):
    """Reads AHN4 NPZ chunks from a spatial split manifest.

    Unlike STPLS3DChunkDataset, this class resolves NPZ paths using
    data_root from the manifest file itself, independent of a shared
    chunks/ directory.
    """

    def __init__(self, manifest_path: Path, split: str, input_mode: str = "xyzrgb"):
        with open(manifest_path) as f:
            manifest = json.load(f)

        data_root = Path(manifest["data_root"])
        if not data_root.is_absolute():
            data_root = ROOT / data_root

        self.split      = split
        self.input_mode = input_mode
        self.entries    = [e for e in manifest["chunks"] if e["split"] == split]
        self.data_root  = data_root

        if not self.entries:
            raise RuntimeError(f"No '{split}' entries in {manifest_path}")

        # Resolve absolute paths: data_root / "chunks" / entry["path"]
        self.npz_paths = [
            data_root / "chunks" / e["path"] for e in self.entries
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        d      = np.load(self.npz_paths[i])
        xyz    = d["xyz"].astype(np.float32)     # (4096, 3) raw world coords
        rgb    = d["rgb"].astype(np.float32) / 255.0  # (4096, 3) [0,1]
        labels = d["labels"].astype(np.int64)    # (4096,)

        xyz_norm = normalize_chunk(xyz)

        if self.input_mode == "xyzrgb":
            feat = np.concatenate([xyz_norm, rgb], axis=1)  # (4096, 6)
        else:
            feat = xyz_norm  # (4096, 3)

        return {
            "xyz":      torch.from_numpy(feat),
            "xyz_orig": torch.from_numpy(xyz),
            "labels":   torch.from_numpy(labels),
        }

    def class_distribution(self) -> dict:
        counts = {c: 0 for c in range(NUM_CLASSES)}
        for e in self.entries:
            for c, n in e["per_class"].items():
                counts[int(c)] += n
        return counts


def load_base_model(ckpt_path: Path, device: torch.device):
    """Load Exp006 model architecture and weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    input_mode = cfg["dataset"].get("input_mode", "xyzrgb")
    input_dim  = 6 if input_mode == "xyzrgb" else 3

    model = PointNet2Seg(
        num_classes=cfg["model"]["num_classes"],
        sa1=cfg["model"].get("sa1"),
        sa2=cfg["model"].get("sa2"),
        input_dim=input_dim,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"  Loaded base model from {ckpt_path.name}  "
          f"(input_dim={input_dim}, num_classes={cfg['model']['num_classes']})")
    return model, cfg, input_dim


def run_epoch(model, loader, device, optimizer=None, training=False):
    """Run one epoch; returns (loss, pred_array, gt_array)."""
    model.train(training)
    total_loss = 0.0
    all_pred, all_gt = [], []
    criterion = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(training):
        for batch in loader:
            xyz    = batch["xyz"].to(device)       # (B, 4096, 6)
            labels = batch["labels"].to(device)    # (B, 4096)

            logits = model(xyz)                    # (B, 4096, 6)
            loss   = criterion(logits.permute(0, 2, 1), labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * xyz.shape[0]
            pred = logits.argmax(dim=-1).cpu().numpy()
            all_pred.append(pred.reshape(-1))
            all_gt.append(labels.cpu().numpy().reshape(-1))

    n     = sum(len(p) for p in all_pred)
    pred  = np.concatenate(all_pred)
    gt    = np.concatenate(all_gt)
    return total_loss / max(len(loader.dataset), 1), pred, gt


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp007: Fine-tune Exp006 on AHN4.")
    ap.add_argument("--base-checkpoint", default="ml/results/experiment_006/best_model.pth")
    ap.add_argument("--manifest",        default="ml/results/ahn4_exp007/data/manifest.json")
    ap.add_argument("--out",             default="ml/results/ahn4_exp007")
    ap.add_argument("--epochs",  type=int,   default=10)
    ap.add_argument("--lr",      type=float, default=5e-5)
    ap.add_argument("--batch",   type=int,   default=4)
    ap.add_argument("--seed",    type=int,   default=26011)
    ap.add_argument("--device",  default="auto")
    ap.add_argument("--resume",  action="store_true",
                    help="Resume from latest_checkpoint.pth in --out dir")
    args = ap.parse_args()

    base_ckpt = ROOT / args.base_checkpoint
    manifest  = ROOT / args.manifest
    out_dir   = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = select_device(args.device)

    print(f"\nExp007 — AHN4 Fine-tuning")
    print(f"  Base checkpoint : {base_ckpt}")
    print(f"  Manifest        : {manifest}")
    print(f"  Output          : {out_dir}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  LR              : {args.lr}")
    print(f"  Device          : {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    if args.resume:
        resume_path = out_dir / "latest_checkpoint.pth"
        model, base_cfg, input_dim = load_base_model(resume_path, device)
        start_epoch = torch.load(resume_path, map_location="cpu",
                                 weights_only=False).get("epoch", 0) + 1
        print(f"  Resuming from epoch {start_epoch}")
    else:
        model, base_cfg, input_dim = load_base_model(base_ckpt, device)
        start_epoch = 1

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = AHN4SplitDataset(manifest, "train")
    val_ds   = AHN4SplitDataset(manifest, "validation")
    print(f"\n  Train chunks : {len(train_ds)}")
    print(f"  Val   chunks : {len(val_ds)}")

    # Print class distribution
    train_dist = train_ds.class_distribution()
    train_tot  = sum(train_dist.values())
    print("  Train class distribution:")
    for c, name in enumerate(CLASS_NAMES):
        n = train_dist.get(c, 0)
        if n > 0:
            print(f"    {name:12} ({c}): {n:>8,}  ({100*n/max(train_tot,1):.1f}%)")

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0)

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # ── Training loop ────────────────────────────────────────────────────────
    history   = []
    best_iou  = 0.0
    best_ckpt = out_dir / "best_model.pth"
    last_ckpt = out_dir / "latest_checkpoint.pth"

    # Compute class weights to handle 80/20 ground/building imbalance
    counts = np.array([train_dist.get(c, 1) for c in range(NUM_CLASSES)], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights /= weights.mean()
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    print(f"\n  Class weights: {[f'{w:.2f}' for w in weights.tolist()]}")
    print(f"\n{'='*65}")

    epoch_end = args.epochs + 1
    for epoch in range(start_epoch, epoch_end):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss, n_batches = 0.0, 0
        all_train_pred, all_train_gt = [], []
        for batch in train_loader:
            xyz    = batch["xyz"].to(device)
            labels = batch["labels"].to(device)
            logits = model(xyz)
            loss   = criterion(logits.permute(0, 2, 1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            n_batches  += 1
            all_train_pred.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
            all_train_gt.append(labels.cpu().numpy().reshape(-1))

        train_pred = np.concatenate(all_train_pred)
        train_gt   = np.concatenate(all_train_gt)
        train_m    = segmentation_metrics(train_pred, train_gt, NUM_CLASSES, CLASS_NAMES)
        train_biou = train_m["per_class"]["building"]["iou"]
        train_miou = train_m["mean_iou"]

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        all_val_pred, all_val_gt = [], []
        with torch.no_grad():
            for batch in val_loader:
                xyz    = batch["xyz"].to(device)
                labels = batch["labels"].to(device)
                logits = model(xyz)
                all_val_pred.append(logits.argmax(dim=-1).cpu().numpy().reshape(-1))
                all_val_gt.append(labels.cpu().numpy().reshape(-1))

        val_pred = np.concatenate(all_val_pred)
        val_gt   = np.concatenate(all_val_gt)
        val_m    = segmentation_metrics(val_pred, val_gt, NUM_CLASSES, CLASS_NAMES)
        val_biou = val_m["per_class"]["building"]["iou"]
        val_miou = val_m["mean_iou"]

        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch":       epoch,
            "train_loss":  round(train_loss / max(n_batches, 1), 4),
            "train_miou":  round(float(train_miou), 4),
            "train_biou":  round(float(train_biou), 4),
            "val_miou":    round(float(val_miou), 4),
            "val_biou":    round(float(val_biou), 4),
            "val_acc":     round(float(val_m["overall_accuracy"]), 4),
            "lr":          scheduler.get_last_lr()[0],
            "time_s":      round(elapsed, 1),
        }
        history.append(row)

        print(f"  Ep {epoch:2d}/{args.epochs}  "
              f"loss={row['train_loss']:.4f}  "
              f"train_bIoU={row['train_biou']:.4f}  "
              f"val_bIoU={row['val_biou']:.4f}  "
              f"val_mIoU={row['val_miou']:.4f}  "
              f"[{elapsed:.0f}s]")

        # Save latest
        save_obj = {
            "epoch":       epoch,
            "config":      base_cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_biou":    val_biou,
            "experiment":  "exp007_ahn4_finetune",
        }
        torch.save(save_obj, last_ckpt)

        # Save best
        if val_biou > best_iou:
            best_iou = val_biou
            torch.save(save_obj, best_ckpt)
            print(f"    *** New best val building IoU: {best_iou:.4f} ***")

        # Save history
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Training complete.")
    print(f"  Best val building IoU : {best_iou:.4f}")
    print(f"  Best checkpoint       : {best_ckpt}")

    # Save config
    config = {
        "experiment":       "exp007",
        "description":      "Exp006 fine-tuned on AHN4 Amsterdam tile 25GN2_01",
        "base_checkpoint":  str(base_ckpt),
        "manifest":         str(manifest),
        "split_method":     "spatial_Y_north_south",
        "n_train":          len(train_ds),
        "n_val":            len(val_ds),
        "epochs":           args.epochs,
        "lr":               args.lr,
        "batch_size":       args.batch,
        "seed":             args.seed,
        "input_mode":       "xyzrgb",
        "input_dim":        input_dim,
        "num_classes":      NUM_CLASSES,
        "best_val_biou":    round(best_iou, 4),
        "dataset":          "AHN4 GeoTiles Amsterdam 25GN2_01",
        "crs":              "EPSG:28992 (Amersfoort / RD New)",
        "rgb_note":         "AHN4 has REAL RGB (mean~113). DALES had zero RGB. Fine-tuning adapts RGB path.",
    }
    with open(out_dir / "config_exp007.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Config saved : {out_dir / 'config_exp007.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

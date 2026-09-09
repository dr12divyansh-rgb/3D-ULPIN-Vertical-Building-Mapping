"""Train a PointNet++ semantic segmentation model.

Usage:
    python ml/src/training/train.py --config ml/config/stpls3d_xyzrgb.yaml

    # Resume an interrupted run (reads latest_checkpoint.pth from the exp dir):
    python ml/src/training/train.py --config ml/config/stpls3d_xyzrgb.yaml \
        --resume ml/results/experiment_005

Writes an experiment directory under ``ml/results/experiment_NNN/`` containing:
  best_model.pth        — best checkpoint by validation mIoU
  latest_checkpoint.pth — most recent epoch checkpoint (used for --resume)
  config.yaml           — effective config
  history.json          — per-epoch metrics
  summary.json          — final summary

Supports:
  gradient_accumulation_steps — accumulate gradients over N batches before step
  preload / preload_train / preload_validation — per-split preload control
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

from synthetic_city.core.labels import CLASS_NAMES  # noqa: E402
from ml.src.data.dataset import (validate_classes,  # noqa: E402
                                 compute_class_weights,
                                 compute_class_weights_stpls3d,
                                 compute_class_weights_mixed,
                                 build_dataset,
                                 _STPLS3D_CLASS_NAMES)
from ml.src.models.pointnet2 import PointNet2Seg  # noqa: E402
from ml.src.training.metrics import segmentation_metrics  # noqa: E402
from ml.src.utils import load_config, set_seed, select_device, new_experiment_dir  # noqa: E402


def _ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1_048_576


def _sys_avail_gb() -> float:
    return psutil.virtual_memory().available / 1_073_741_824


def _cpu_pct() -> float:
    """System-wide CPU utilisation since last call (non-blocking)."""
    return psutil.cpu_percent(interval=None)


def _check_prior_experiments(results_root: str) -> None:
    """Verify that previous experiment checkpoints are untouched."""
    root = Path(results_root)
    for n in range(1, 5):
        exp = root / f"experiment_{n:03d}"
        ckpt = exp / "best_model.pth"
        if exp.exists() and not ckpt.exists():
            raise RuntimeError(
                f"Safety check failed: {exp} exists but best_model.pth is missing."
            )
    print("  [safety] experiments 001-004 checkpoints intact.")


def _resolve_data_root(cfg: dict) -> Path | None:
    """Resolve data_root for single-source configs; returns None for mixed."""
    if "data_root" not in cfg["dataset"]:
        return None
    p = Path(cfg["dataset"]["data_root"])
    if p.is_absolute():
        return p
    return ROOT / p


def predict_dataset(model, dataset, device, num_classes, batch_size):
    """Run model over every chunk in dataset, returning (preds, targets)."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
    all_pred, all_target = [], []
    with torch.no_grad():
        for batch in loader:
            xyz = batch["xyz"].to(device)
            labels = batch["labels"]
            logits = model(xyz)
            pred = logits.argmax(dim=-1).cpu()
            all_pred.append(pred.numpy().reshape(-1))
            all_target.append(labels.numpy().reshape(-1))
    return np.concatenate(all_pred), np.concatenate(all_target)


def main():
    ap = argparse.ArgumentParser(description="Train PointNet++ semantic segmentation.")
    ap.add_argument("--config", default=str(ROOT / "ml" / "config" / "segmentation.yaml"))
    ap.add_argument("--results", default=str(ROOT / "ml" / "results"))
    ap.add_argument("--resume", default=None,
                    help="Path to an experiment dir containing latest_checkpoint.pth to resume")
    args = ap.parse_args()

    cfg = load_config(args.config)
    validate_classes(cfg["classes"])
    seed = int(cfg["training"]["seed"])
    set_seed(seed)
    device = select_device(cfg["training"]["device"])
    num_classes = int(cfg["model"]["num_classes"])
    source = cfg["dataset"].get("source", "synthetic")
    input_mode = cfg["dataset"].get("input_mode", "xyz")
    input_dim = 6 if input_mode == "xyzrgb" else 3
    data_root = _resolve_data_root(cfg)

    _cpu_pct()  # prime the cpu_percent sampler (first call returns 0.0)

    print(f"\n{'='*60}")
    print(f"SIH 26011 — PointNet++ Training")
    print(f"  config      : {args.config}")
    print(f"  source      : {source}")
    print(f"  input_mode  : {input_mode}  (input_dim={input_dim})")
    print(f"  device      : {device}  (CUDA: {torch.cuda.is_available()})")
    ram_gb = psutil.virtual_memory().total / 1e9
    avail_gb = _sys_avail_gb()
    print(f"  system RAM  : {ram_gb:.1f} GB total, {avail_gb:.1f} GB free")
    if avail_gb < 1.5:
        print(f"  WARNING: only {avail_gb:.1f} GB free — close other apps if possible.")
        print(f"           batch_size=2 + preload=false is already the safest config.")
    print(f"{'='*60}\n")

    _check_prior_experiments(args.results)

    # Determine per-split preload — support both unified 'preload' and per-split keys
    def _preload_for(split: str) -> bool:
        key = f"preload_{split}"
        if key in cfg["dataset"]:
            return bool(cfg["dataset"][key])
        return bool(cfg["dataset"].get("preload", False))

    # Temporarily override preload in cfg for build_dataset
    def build_ds(split: str):
        preload_val = _preload_for(split)
        orig = cfg["dataset"].get("preload")
        cfg["dataset"]["preload"] = preload_val
        ds = build_dataset(cfg, split, seed,
                           samples_per_scene=cfg["dataset"].get(
                               "train_samples_per_scene" if split == "train"
                               else "val_samples_per_scene"))
        if orig is None:
            cfg["dataset"].pop("preload", None)
        else:
            cfg["dataset"]["preload"] = orig
        return ds

    print("Loading datasets...")
    train_ds = build_ds("train")
    val_ds = build_ds("validation")
    print(f"  train chunks: {len(train_ds)}   val chunks: {len(val_ds)}")
    print(f"  RAM after load: {_ram_mb():.0f} MB  (system free: {_sys_avail_gb():.2f} GB)")

    class_names_for_logging = (list(CLASS_NAMES) if source == "synthetic"
                                else list(_STPLS3D_CLASS_NAMES))

    model = PointNet2Seg(num_classes=num_classes,
                         sa1=cfg["model"].get("sa1"),
                         sa2=cfg["model"].get("sa2"),
                         input_dim=input_dim)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    tcfg = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(tcfg["learning_rate"]),
                                 weight_decay=float(tcfg.get("weight_decay", 0.0)))

    # Class-weighted loss
    loss_kwargs = {}
    if tcfg.get("class_weighted_loss", False):
        if source == "synthetic":
            weights = torch.from_numpy(compute_class_weights(data_root, num_classes)).to(device)
        elif source == "stpls3d":
            weights = torch.from_numpy(
                compute_class_weights_stpls3d(data_root, num_classes)).to(device)
        elif source == "mixed":
            syn_root = ROOT / cfg["dataset"]["synthetic"]["data_root"]
            stpls3d_root = ROOT / cfg["dataset"]["stpls3d_root"]
            syn_weight = float(cfg["dataset"].get("synthetic_weight", 0.3))
            weights = torch.from_numpy(
                compute_class_weights_mixed(syn_root, stpls3d_root, num_classes, syn_weight)
            ).to(device)
        else:
            raise ValueError(f"Unknown source for class weights: {source}")
        loss_kwargs["weight"] = weights
        print("  class weights:",
              {class_names_for_logging[i]: round(float(weights[i]), 3)
               for i in range(num_classes)})
    criterion = nn.CrossEntropyLoss(**loss_kwargs)

    batch_size = int(tcfg["batch_size"])
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 1))
    effective_bs = batch_size * grad_accum
    print(f"  batch_size={batch_size}  grad_accum={grad_accum}  "
          f"effective_bs={effective_bs}")

    # Resume or create new experiment dir
    start_epoch = 1
    best_iou = -1.0
    best_epoch = -1
    patience_counter = 0
    history: list[dict] = []

    if args.resume:
        resume_dir = Path(args.resume)
        latest_ckpt = resume_dir / "latest_checkpoint.pth"
        if not latest_ckpt.exists():
            print(f"WARNING: --resume given but {latest_ckpt} not found. "
                  "Starting fresh.")
            exp_dir = new_experiment_dir(args.results)
        else:
            print(f"\nResuming from: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_iou = float(ckpt.get("best_iou", -1.0))
            best_epoch = int(ckpt.get("best_epoch", -1))
            patience_counter = int(ckpt.get("patience_counter", 0))
            history = ckpt.get("history", [])
            exp_dir = resume_dir
            print(f"  resumed at epoch {start_epoch - 1}, best_iou={best_iou:.4f}")
    else:
        exp_dir = new_experiment_dir(args.results)

    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"\nexperiment dir: {exp_dir}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=0)

    epochs = int(tcfg["epochs"])
    log_interval = int(tcfg.get("log_interval", 50))
    early_patience = int(tcfg.get("early_stopping_patience", 10 ** 9))

    if start_epoch > epochs:
        print(f"Already completed all {epochs} epochs. Nothing to do.")
        return

    print(f"\nTraining epochs {start_epoch}–{epochs}  "
          f"({len(train_loader)} batches/epoch)\n")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            xyz = batch["xyz"].to(device)
            labels = batch["labels"].to(device)
            logits = model(xyz)
            loss = criterion(logits.reshape(-1, num_classes), labels.reshape(-1))

            # Scale loss for gradient accumulation
            (loss / grad_accum).backward()

            epoch_loss += loss.item()
            n_batches += 1

            if step % grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            if step % log_interval == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}  "
                      f"RAM={_ram_mb():.0f}MB  free={_sys_avail_gb():.2f}GB")

        train_loss = epoch_loss / max(n_batches, 1)

        pred, target = predict_dataset(model, val_ds, device, num_classes, batch_size)
        val_metrics = segmentation_metrics(pred, target, num_classes,
                                           class_names_for_logging)
        val_iou = val_metrics["mean_iou"]
        val_acc = val_metrics["overall_accuracy"]
        pc = val_metrics["per_class"]
        building_iou = pc.get("building", {}).get("iou", float("nan"))
        elapsed = time.time() - t0

        ram_proc = _ram_mb()
        sys_free_gb = _sys_avail_gb()
        cpu_pct = _cpu_pct()

        print(
            f"epoch {epoch:>2}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_mIoU={val_iou:.4f} | "
            f"buildingIoU={building_iou:.4f} | "
            f"acc={val_acc:.4f} | "
            f"time={elapsed:.0f}s | "
            f"CPU={cpu_pct:.0f}% | "
            f"RAM={ram_proc:.0f}MB | "
            f"free={sys_free_gb:.2f}GB"
        )
        # Per-class detail line
        detail = "  classes: " + " ".join(
            f"{n[:3]}={pc.get(n, {}).get('iou', 0):.3f}"
            for n in class_names_for_logging
        )
        print(detail)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_overall_accuracy": val_acc,
            "val_mean_iou": val_iou,
            "val_building_iou": building_iou,
            "val_building_f1": pc.get("building", {}).get("f1", float("nan")),
            "epoch_time_s": elapsed,
            "cpu_pct": cpu_pct,
            "ram_process_mb": ram_proc,
            "system_free_gb": sys_free_gb,
        }
        # Add all per-class IoUs
        for n in class_names_for_logging:
            row[f"val_iou_{n}"] = pc.get(n, {}).get("iou", float("nan"))
        history.append(row)

        # Best checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "seed": seed,
                "epoch": epoch,
                "val_mean_iou": best_iou,
                "class_names": class_names_for_logging,
                "input_dim": input_dim,
            }, exp_dir / "best_model.pth")
            print(f"  -> saved best_model.pth  (val mIoU {best_iou:.4f})")
        else:
            patience_counter += 1

        # Latest checkpoint — always save (enables resume)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "seed": seed,
            "epoch": epoch,
            "val_mean_iou": val_iou,
            "best_iou": best_iou,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "class_names": class_names_for_logging,
            "input_dim": input_dim,
            "history": history,
        }, exp_dir / "latest_checkpoint.pth")

        # Persist history after every epoch
        (exp_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")

        if sys_free_gb < 0.5:
            print(f"WARNING: system RAM critically low ({sys_free_gb:.2f} GB free). "
                  "Consider reducing batch_size or disabling preload.")

        if patience_counter >= early_patience:
            print(f"Early stopping after epoch {epoch} (patience={early_patience})")
            break

    summary = {
        "best_epoch": best_epoch,
        "best_val_mean_iou": best_iou,
        "total_epochs_trained": len(history),
        "model_parameters": n_params,
        "input_mode": input_mode,
        "input_dim": input_dim,
        "source": source,
        "config_path": str(args.config),
    }
    (exp_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTraining complete.")
    print(f"  best epoch  : {best_epoch}  (val mIoU {best_iou:.4f})")
    print(f"  results dir : {exp_dir}")
    print(f"  next        : python ml/src/evaluate_stpls3d.py "
          f"--checkpoint {exp_dir / 'best_model.pth'} "
          f"--data C:/ULPIN_DATA/stpls3d")


if __name__ == "__main__":
    main()

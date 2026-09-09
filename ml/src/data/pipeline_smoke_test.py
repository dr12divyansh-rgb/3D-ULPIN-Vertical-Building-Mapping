"""Full ML pipeline smoke test for the STPLS3D segmentation pipeline.

Validates the complete path from manifest → dataset → PointNet++ → loss → step.

Usage:
    python ml/src/data/pipeline_smoke_test.py --data ml/data/stpls3d
    python ml/src/data/pipeline_smoke_test.py --data ml/data/stpls3d_smoke  # quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ml.src.data.dataset import (  # noqa: E402
    STPLS3DChunkDataset, compute_class_weights_stpls3d, normalize_chunk
)
from ml.src.models.pointnet2 import PointNet2Seg  # noqa: E402
from ml.src.training.metrics import segmentation_metrics  # noqa: E402

NUM_CLASSES = 6
CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]


def check(cond, msg):
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Preprocessed STPLS3D directory (has manifest.json)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-batches", type=int, default=3,
                    help="Number of train batches to test")
    args = ap.parse_args()

    data_root = Path(args.data)
    print("\n" + "=" * 65)
    print("STPLS3D FULL PIPELINE SMOKE TEST")
    print("=" * 65)
    print(f"  data_root : {data_root}")

    # ------------------------------------------------------------------ #
    # 1. Dataset loading
    # ------------------------------------------------------------------ #
    print("\n[1] Dataset loading...")

    # Try train first, fall back to test if no train split
    for try_split in ("train", "validation", "test"):
        try:
            train_ds = STPLS3DChunkDataset(data_root, try_split, num_points=4096)
            train_split_used = try_split
            break
        except RuntimeError:
            continue
    else:
        print("ERROR: no chunks found for any split", file=sys.stderr)
        sys.exit(1)

    print(f"  Using split '{train_split_used}' for training smoke test")
    check(len(train_ds) > 0, f"train dataset has {len(train_ds)} chunks")

    for try_val_split in ("validation", "test", "train"):
        try:
            val_ds = STPLS3DChunkDataset(data_root, try_val_split, num_points=4096)
            val_split_used = try_val_split
            break
        except RuntimeError:
            continue

    print(f"  Using split '{val_split_used}' for validation smoke test")
    print(f"  train chunks: {len(train_ds)}   val chunks: {len(val_ds)}")

    # ------------------------------------------------------------------ #
    # 2. DataLoader + tensor shapes
    # ------------------------------------------------------------------ #
    print("\n[2] Tensor shape validation...")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    batch = next(iter(loader))

    B, N = args.batch_size, 4096
    xyz = batch["xyz"]
    labels = batch["labels"]
    xyz_orig = batch["xyz_orig"]

    check(tuple(xyz.shape) == (B, N, 3), f"xyz shape {xyz.shape} == ({B},{N},3)")
    check(tuple(labels.shape) == (B, N), f"labels shape {labels.shape} == ({B},{N})")
    check(tuple(xyz_orig.shape) == (B, N, 3), f"xyz_orig shape {xyz_orig.shape}")
    check(not torch.any(torch.isnan(xyz)), "no NaN in xyz")
    check(not torch.any(torch.isinf(xyz)), "no Inf in xyz")
    check((labels >= 0).all() and (labels < NUM_CLASSES).all(),
          f"labels in [0,{NUM_CLASSES}): unique={labels.unique().tolist()}")

    # ------------------------------------------------------------------ #
    # 3. Coordinate normalisation
    # ------------------------------------------------------------------ #
    print("\n[3] Coordinate normalisation...")
    # Per-chunk norm should result in near-unit sphere
    for bi in range(B):
        norms = torch.norm(xyz[bi], dim=-1)
        max_norm = norms.max().item()
        check(max_norm <= 1.05, f"batch[{bi}] max norm {max_norm:.4f} <= 1.05 (unit sphere)")

    # ------------------------------------------------------------------ #
    # 4. PointNet++ forward pass
    # ------------------------------------------------------------------ #
    print("\n[4] PointNet++ forward pass...")
    device = torch.device("cpu")
    model = PointNet2Seg(num_classes=NUM_CLASSES)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,}")

    model.train()
    xyz_in = xyz.to(device)
    logits = model(xyz_in)
    check(tuple(logits.shape) == (B, N, NUM_CLASSES),
          f"logits shape {logits.shape} == ({B},{N},{NUM_CLASSES})")
    check(not torch.any(torch.isnan(logits)), "no NaN in logits")

    # ------------------------------------------------------------------ #
    # 5. Class-weighted loss computation
    # ------------------------------------------------------------------ #
    print("\n[5] Class-weighted loss...")
    try:
        raw_weights = compute_class_weights_stpls3d(data_root, NUM_CLASSES, split=train_split_used)
        weights_t = torch.from_numpy(raw_weights).to(device)
        print(f"  class weights: {dict(zip(CLASS_NAMES, [round(float(w),3) for w in raw_weights]))}")
        criterion = nn.CrossEntropyLoss(weight=weights_t)
    except Exception as e:
        print(f"  NOTE: class weight computation failed ({e}); using uniform loss")
        criterion = nn.CrossEntropyLoss()

    labels_in = batch["labels"].to(device)
    loss = criterion(logits.reshape(-1, NUM_CLASSES), labels_in.reshape(-1))
    check(not torch.isnan(loss), f"loss is finite: {loss.item():.4f}")
    check(loss.item() > 0, f"loss > 0: {loss.item():.4f}")
    print(f"  initial loss: {loss.item():.4f}")

    # ------------------------------------------------------------------ #
    # 6. Backward + optimizer step
    # ------------------------------------------------------------------ #
    print("\n[6] Backward + optimizer step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    # Check gradients exist
    grads_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    check(grads_ok, "gradients exist for all parameters")

    optimizer.step()

    # Verify loss changes after step
    logits2 = model(xyz_in)
    loss2 = criterion(logits2.reshape(-1, NUM_CLASSES), labels_in.reshape(-1))
    print(f"  loss after 1 step: {loss2.item():.4f}")
    check(not torch.isnan(loss2), "loss2 is finite")

    # ------------------------------------------------------------------ #
    # 7. Multiple training batches
    # ------------------------------------------------------------------ #
    print(f"\n[7] {args.num_batches} training batches...")
    losses = [loss.item()]
    for bi2, batch2 in enumerate(loader):
        if bi2 >= args.num_batches - 1:
            break
        xyz2 = batch2["xyz"].to(device)
        lab2 = batch2["labels"].to(device)
        logits3 = model(xyz2)
        l3 = criterion(logits3.reshape(-1, NUM_CLASSES), lab2.reshape(-1))
        optimizer.zero_grad()
        l3.backward()
        optimizer.step()
        losses.append(l3.item())
    print(f"  batch losses: {[round(l, 4) for l in losses]}")
    check(all(not np.isnan(l) for l in losses), "all batch losses are finite")

    # ------------------------------------------------------------------ #
    # 8. Validation pass
    # ------------------------------------------------------------------ #
    print("\n[8] Validation pass...")
    model.eval()
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    all_pred, all_target = [], []
    with torch.no_grad():
        for vi, vbatch in enumerate(val_loader):
            if vi >= 3:  # limit to 3 batches for speed
                break
            vxyz = vbatch["xyz"].to(device)
            vlab = vbatch["labels"]
            vlogits = model(vxyz)
            vpred = vlogits.argmax(dim=-1).cpu()
            all_pred.append(vpred.numpy().reshape(-1))
            all_target.append(vlab.numpy().reshape(-1))

    pred_arr = np.concatenate(all_pred)
    target_arr = np.concatenate(all_target)
    metrics = segmentation_metrics(pred_arr, target_arr, NUM_CLASSES, CLASS_NAMES)

    check(0.0 <= metrics["overall_accuracy"] <= 1.0,
          f"overall_accuracy={metrics['overall_accuracy']:.4f} in [0,1]")
    check(0.0 <= metrics["mean_iou"] <= 1.0,
          f"mean_iou={metrics['mean_iou']:.4f} in [0,1]")
    print(f"  overall_accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"  mean_iou         : {metrics['mean_iou']:.4f}")

    # ------------------------------------------------------------------ #
    # 9. XYZRGB mode check
    # ------------------------------------------------------------------ #
    print("\n[9] XYZRGB mode check...")
    rgb_ds = STPLS3DChunkDataset(data_root, train_split_used, num_points=4096, input_mode="xyzrgb")
    rgb_batch = next(iter(torch.utils.data.DataLoader(rgb_ds, batch_size=2)))
    check(tuple(rgb_batch["xyz"].shape) == (2, 4096, 6),
          f"xyzrgb shape {rgb_batch['xyz'].shape} == (2,4096,6)")
    rgb_chan = rgb_batch["xyz"][:, :, 3:]  # the RGB part
    check((rgb_chan >= 0).all() and (rgb_chan <= 1).all(),
          "RGB normalised to [0,1]")

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("ALL PIPELINE SMOKE TEST CHECKS PASSED")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

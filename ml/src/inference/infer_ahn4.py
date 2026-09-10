"""AHN4 inference and evaluation with Exp006 PointNet++ checkpoint.

Runs Exp006 on AHN4 preprocessed NPZ chunks and produces:
  1. Per-chunk predictions
  2. Building IoU and per-class metrics (vs ASPRS ground-truth labels)
  3. Full-scene combined prediction PLY (with real RD New coordinates)
  4. JSON metrics file

Usage:
    python ml/src/inference/infer_ahn4.py \\
        --checkpoint ml/results/experiment_006/best_model.pth \\
        --data       ml/results/ahn4_inference \\
        --split      test \\
        --out        ml/results/ahn4_inference \\
        --save-ply

Evaluation notes:
  - Only chunks / points with ASPRS class 2 (ground) or 6 (building) have
    definitive semantic labels in this tile.  Unclassified points (ASPRS 1)
    are discarded during preprocessing (prepare_ahn4.py eval mode).
  - Classes present: ground (0), building (1).  Vegetation, road, vehicle
    are absent from this tile's labelling; their IoU is reported as N/A.
  - Building IoU is the primary metric for cross-domain comparison with
    Exp006 DALES test result (0.763).
"""
from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ml.src.models.pointnet2 import PointNet2Seg
from ml.src.training.metrics import segmentation_metrics
from ml.src.utils import select_device

_CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]

_CLASS_COLORS = {
    0: (120, 120, 120),   # ground      grey
    1: (220, 110,  55),   # building    orange
    2: ( 45, 160,  80),   # vegetation  green
    3: ( 40,  40,  40),   # road        dark grey
    4: ( 70, 130, 200),   # vehicle     blue
    5: (200, 200,  40),   # other       yellow
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
    return model, cfg, input_dim


def normalize_chunk(xyz: np.ndarray):
    centroid = xyz.mean(axis=0, keepdims=True)
    centered = xyz - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    return (centered / scale).astype(np.float32), centroid.reshape(-1), scale


def load_npz_chunks(data_root: Path, split: str) -> list[dict]:
    """Load all NPZ chunks for a split from the manifest."""
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {data_root}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = [e for e in manifest["chunks"] if e["split"] == split]
    if not entries:
        raise RuntimeError(f"No '{split}' chunks in manifest {manifest_path}")

    chunks = []
    for entry in entries:
        npz_path = data_root / "chunks" / entry["path"]
        data = np.load(npz_path)
        chunks.append({
            "xyz":      data["xyz"].astype(np.float32),
            "rgb":      data["rgb"].astype(np.float32) / 255.0,
            "labels":   data["labels"].astype(np.int64),
            "scene_id": entry["scene_id"],
            "path":     str(npz_path),
        })
    return chunks, manifest.get("crs", "EPSG:28992")


def write_pred_ply(path: Path, xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                   confidence: np.ndarray) -> None:
    """Write PLY with original world coords + prediction + GT + confidence."""
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"comment AHN4 Exp006 inference output (SIH 26011)\n"
        f"comment CRS: EPSG:28992 (Amersfoort / RD New)\n"
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run Exp006 inference + evaluation on AHN4 NPZ chunks.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data",  default="ml/results/ahn4_inference",
                    help="Root dir with manifest.json and chunks/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out",   required=True,
                    help="Output directory for metrics JSON and PLY files.")
    ap.add_argument("--save-ply",  action="store_true",
                    help="Write per-chunk prediction PLY files.")
    ap.add_argument("--save-combined-ply", action="store_true",
                    help="Write one combined PLY for the whole scene (large).")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Limit to first N chunks (for quick tests).")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    data_root = Path(args.data)
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    print(f"\nAHN4 Exp006 Inference")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  data       : {data_root}")
    print(f"  split      : {args.split}")
    print(f"  output     : {out_dir}\n")

    device = select_device("auto")
    model, cfg, input_dim = load_model(ckpt_path, device)
    num_classes = int(cfg["model"]["num_classes"])
    input_mode  = cfg["dataset"].get("input_mode", "xyz")
    num_points  = int(cfg["dataset"]["num_points"])

    print(f"  model      : {cfg['model']['name']}  dim={input_dim}  "
          f"num_classes={num_classes}  input_mode={input_mode}")

    chunks, crs = load_npz_chunks(data_root, args.split)
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    print(f"  chunks     : {len(chunks)}  |  CRS: {crs}\n")

    all_pred, all_gt = [], []
    all_xyz_orig, all_conf = [], []
    ply_dir = out_dir / "predictions"

    model.eval()
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            xyz_orig = chunk["xyz"]       # (4096, 3) raw world coords (RD New metres)
            rgb      = chunk["rgb"]       # (4096, 3) float32 0-1
            gt       = chunk["labels"]    # (4096,)   uint8 project class

            xyz_norm, _, _ = normalize_chunk(xyz_orig)

            if input_dim == 6:
                feat = np.concatenate([xyz_norm, rgb], axis=1)  # (4096, 6)
            else:
                feat = xyz_norm

            inp = torch.from_numpy(feat).unsqueeze(0).to(device)    # (1, 4096, 6)
            logits = model(inp)                                       # (1, 4096, 6)
            probs  = torch.softmax(logits, dim=-1)
            pred   = probs.argmax(dim=-1).cpu().numpy()[0]            # (4096,)
            conf   = probs.max(dim=-1).values.cpu().numpy()[0]        # (4096,)

            all_pred.append(pred)
            all_gt.append(gt)
            all_xyz_orig.append(xyz_orig)
            all_conf.append(conf)

            if args.save_ply:
                ply_path = ply_dir / f"{chunk['scene_id']}_{i:04d}.ply"
                write_pred_ply(ply_path, xyz_orig, pred, gt, conf)

            if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
                print(f"  [{i+1}/{len(chunks)}] chunks done")

    pred_all = np.concatenate(all_pred)
    gt_all   = np.concatenate(all_gt)
    xyz_all  = np.concatenate(all_xyz_orig)
    conf_all = np.concatenate(all_conf)
    n_pts    = len(pred_all)

    print(f"\n  Total points: {n_pts:,}")

    # Prediction class distribution
    print("\n  Prediction distribution:")
    for cls_id, name in enumerate(_CLASS_NAMES[:num_classes]):
        cnt = int((pred_all == cls_id).sum())
        print(f"    {name:<12} {cnt:>8,}  ({100*cnt/n_pts:.1f}%)")

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Classes present in AHN4: 0=ground, 1=building (only these 2 after eval-mode preprocessing)
    metrics = segmentation_metrics(pred_all, gt_all, num_classes, _CLASS_NAMES)

    print(f"\n{'='*60}")
    print(f"  AHN4 Evaluation (Exp006 cross-domain, UNSEEN data)")
    print(f"  Note: Only ground+building labels present in this tile.")
    print(f"  Vegetation/road/vehicle IoU is N/A (no GT in this tile).")
    print(f"  Overall accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"  Mean IoU         : {metrics['mean_iou']:.4f}")
    print(f"\n  {'Class':<14} {'Prec':>7} {'Rec':>7} {'F1':>7} {'IoU':>7}")
    for name in _CLASS_NAMES:
        pc = metrics["per_class"].get(name, {})
        print(f"  {name:<14} "
              f"{pc.get('precision', 0):>7.4f} "
              f"{pc.get('recall',    0):>7.4f} "
              f"{pc.get('f1',        0):>7.4f} "
              f"{pc.get('iou',       0):>7.4f}")
    print(f"{'='*60}")

    building_iou = float(metrics["per_class"].get("building", {}).get("iou", 0))
    print(f"\n  BUILDING IoU  : {building_iou:.4f}")
    print(f"  (DALES baseline: 0.763 — cross-domain drop expected)")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = np.asarray(metrics["confusion_matrix"])
    cm_path = out_dir / "ahn4_confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + _CLASS_NAMES[:num_classes])
        for i, row in enumerate(cm):
            lbl = _CLASS_NAMES[i] if i < len(_CLASS_NAMES) else str(i)
            w.writerow([lbl] + [int(v) for v in row])

    # ── Save combined PLY ─────────────────────────────────────────────────────
    combined_ply_path = out_dir / "ahn4_combined_prediction.ply"
    if args.save_combined_ply:
        print(f"\n  Writing combined PLY ({n_pts:,} points)...")
        write_pred_ply(combined_ply_path, xyz_all, pred_all, gt_all, conf_all)
        print(f"  Saved: {combined_ply_path}")
    else:
        # Write even without flag for a subset (first 50 chunks ~200k points)
        n_preview = min(50 * num_points, n_pts)
        print(f"\n  Writing preview PLY ({n_preview:,} points, first 50 chunks)...")
        write_pred_ply(combined_ply_path,
                       xyz_all[:n_preview], pred_all[:n_preview],
                       gt_all[:n_preview], conf_all[:n_preview])
        print(f"  Saved: {combined_ply_path}")

    # ── JSON metrics ─────────────────────────────────────────────────────────
    result = {
        "experiment":          "Exp006_AHN4_inference",
        "checkpoint":          str(ckpt_path),
        "data_root":           str(data_root),
        "split":               args.split,
        "dataset":             "AHN4 GeoTiles - Amsterdam tile 25GN2_01",
        "crs":                 crs,
        "tile_coverage":       "1040m x 1290m (1.34 km2)",
        "point_density_pm2":   "~28 pts/m2 (total) / ~20 pts/m2 (labeled)",
        "num_chunks":          len(chunks),
        "total_points":        n_pts,
        "classes_with_gt":     ["ground (0)", "building (1)"],
        "classes_absent_in_gt":["vegetation (2)", "road (3)", "vehicle (4)", "other (5)"],
        "overall_accuracy":    float(metrics["overall_accuracy"]),
        "mean_iou":            float(metrics["mean_iou"]),
        "building_iou":        building_iou,
        "building_precision":  float(metrics["per_class"].get("building", {}).get("precision", 0)),
        "building_recall":     float(metrics["per_class"].get("building", {}).get("recall", 0)),
        "building_f1":         float(metrics["per_class"].get("building", {}).get("f1", 0)),
        "ground_iou":          float(metrics["per_class"].get("ground", {}).get("iou", 0)),
        "dales_building_iou":  0.763,
        "cross_domain_note":   (
            "Model trained on STPLS3D (terrestrial) + fine-tuned on DALES (aerial USA). "
            "AHN4 is aerial Netherlands. Some cross-domain drop expected."
        ),
        "per_class":           {k: {m: float(v) for m, v in d.items()}
                                for k, d in metrics["per_class"].items()},
        "combined_ply":        str(combined_ply_path),
        "confusion_matrix":    str(cm_path),
    }
    metrics_path = out_dir / "ahn4_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Metrics JSON  : {metrics_path}")
    print(f"  Confusion mat : {cm_path}")
    print(f"  Preview PLY   : {combined_ply_path}")
    if args.save_ply:
        print(f"  Per-chunk PLY : {ply_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

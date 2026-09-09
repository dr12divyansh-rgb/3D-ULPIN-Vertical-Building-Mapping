"""Evaluate a trained segmentation model on the STPLS3D held-out test set.

Usage:
    python ml/src/evaluate_stpls3d.py \
        --checkpoint ml/results/experiment_003/best_model.pth \
        --data ml/data/stpls3d

Produces (in the checkpoint's experiment directory):
  stpls3d_test_metrics.json     — aggregate metrics
  stpls3d_confusion_matrix.csv  — NxN confusion matrix
  stpls3d_confusion_matrix.png  — normalised heatmap
  predictions/<scene_id>/<chunk_id>.ply   — XYZ + gt_class + predicted_class + confidence
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.src.data.dataset import STPLS3DChunkDataset, normalize_chunk  # noqa: E402
from ml.src.inference.predict import build_model_from_checkpoint  # noqa: E402
from ml.src.training.metrics import segmentation_metrics  # noqa: E402
from ml.src.utils import select_device  # noqa: E402

CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]
NUM_CLASSES = 6


def save_prediction_ply_with_gt(path, xyz, gt_class, pred_class, confidence):
    """Write a PLY with xyz + ground truth + prediction + confidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment STPLS3D prediction (SIH 26011)\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar gt_class\n"
        "property uchar predicted_class\n"
        "property float confidence\n"
        "end_header\n"
    )
    fmt = struct.Struct("<fffBBf")
    buf = bytearray()
    for i in range(n):
        x, y, z = xyz[i]
        buf += fmt.pack(float(x), float(y), float(z),
                        int(gt_class[i]), int(pred_class[i]), float(confidence[i]))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(bytes(buf))


def _save_confusion_plot(path, cm, class_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = cm.shape[0]
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(np.float64), row_sum,
                            out=np.zeros_like(cm, dtype=np.float64), where=row_sum > 0)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, label="row-normalised")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except ImportError:
        print("matplotlib not available — skipping confusion matrix plot")


def main():
    ap = argparse.ArgumentParser(description="Evaluate on STPLS3D test set.")
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    ap.add_argument("--data", required=True, help="Path to preprocessed STPLS3D dir (has manifest.json)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-ply", action="store_true", default=True,
                    help="Save per-chunk prediction PLY files")
    ap.add_argument("--no-save-ply", action="store_false", dest="save_ply")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    exp_dir = ckpt_path.parent
    data_root = Path(args.data)
    device = select_device("auto")

    print(f"\nLoading checkpoint: {ckpt_path}")
    model, ckpt = build_model_from_checkpoint(ckpt_path, device)
    cfg = ckpt["config"]
    num_points = int(cfg["dataset"]["num_points"])
    input_mode = cfg["dataset"].get("input_mode", "xyz")
    print(f"Model: {cfg['model']['name']}  num_classes={cfg['model']['num_classes']}  input_mode={input_mode}")

    print(f"\nLoading STPLS3D test split from: {data_root}")
    test_ds = STPLS3DChunkDataset(data_root, "test", num_points=num_points, input_mode=input_mode)
    print(f"Test chunks: {len(test_ds)}")
    if len(test_ds) == 0:
        print("ERROR: no test chunks found", file=sys.stderr)
        sys.exit(1)

    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_pred, all_target = [], []
    preds_dir = exp_dir / "stpls3d_predictions"

    print("\nRunning inference...")
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            xyz = batch["xyz"].to(device)       # (B, N, 3)
            labels = batch["labels"]            # (B, N)
            xyz_orig = batch["xyz_orig"]        # (B, N, 3)
            scenes = batch["scene"]             # list of str

            logits = model(xyz)                 # (B, N, C)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1).cpu()   # (B, N)
            conf = probs.max(dim=-1).values.cpu()  # (B, N)

            all_pred.append(pred.numpy().reshape(-1))
            all_target.append(labels.numpy().reshape(-1))

            if args.save_ply:
                for bi2 in range(xyz.shape[0]):
                    scene_id = scenes[bi2]
                    global_chunk_idx = bi * args.batch_size + bi2
                    out_ply = preds_dir / scene_id / f"chunk_{global_chunk_idx:05d}.ply"
                    save_prediction_ply_with_gt(
                        out_ply,
                        xyz_orig[bi2].numpy(),
                        labels[bi2].numpy(),
                        pred[bi2].numpy(),
                        conf[bi2].numpy(),
                    )

            if (bi + 1) % 50 == 0:
                print(f"  processed {(bi + 1) * args.batch_size} / {len(test_ds)} chunks")

    pred_arr = np.concatenate(all_pred)
    target_arr = np.concatenate(all_target)

    print("\nComputing metrics...")
    metrics = segmentation_metrics(pred_arr, target_arr, NUM_CLASSES, CLASS_NAMES)
    cm = np.asarray(metrics["confusion_matrix"])

    # Build output report
    out_metrics = {
        "checkpoint": str(ckpt_path),
        "data_root": str(data_root),
        "seed": ckpt.get("seed"),
        "epoch": ckpt.get("epoch"),
        "num_test_chunks": len(test_ds),
        "num_test_points": int(len(target_arr)),
        "overall_accuracy": metrics["overall_accuracy"],
        "mean_iou": metrics["mean_iou"],
        "per_class": metrics["per_class"],
        "class_names": CLASS_NAMES,
    }

    metrics_path = exp_dir / "stpls3d_test_metrics.json"
    metrics_path.write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics saved to: {metrics_path}")

    csv_path = exp_dir / "stpls3d_confusion_matrix.csv"
    np.savetxt(csv_path, cm, fmt="%d", delimiter=",",
               header=",".join(CLASS_NAMES), comments="")
    print(f"Confusion matrix saved to: {csv_path}")

    _save_confusion_plot(exp_dir / "stpls3d_confusion_matrix.png", cm, CLASS_NAMES)

    print("\n===== STPLS3D TEST METRICS =====")
    print(f"overall_accuracy : {out_metrics['overall_accuracy']:.4f}")
    print(f"mean_iou         : {out_metrics['mean_iou']:.4f}")
    print("\nper-class metrics:")
    print(f"  {'class':<12}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'iou':>6}")
    for name in CLASS_NAMES:
        pc = metrics["per_class"][name]
        print(f"  {name:<12}  {pc['precision']:>6.4f}  {pc['recall']:>6.4f}  "
              f"{pc['f1']:>6.4f}  {pc['iou']:>6.4f}")

    print("\nconfusion matrix (rows=true, cols=pred):")
    header_str = "  " + "  ".join(f"{n[:4]:>5}" for n in CLASS_NAMES)
    print(header_str)
    for ri, row in enumerate(cm):
        row_str = "  ".join(f"{v:>5}" for v in row)
        print(f"  {CLASS_NAMES[ri][:4]:<4}  {row_str}")

    print(f"\nResults written to: {exp_dir}")
    if args.save_ply:
        print(f"Prediction PLYs:   {preds_dir}")


if __name__ == "__main__":
    main()

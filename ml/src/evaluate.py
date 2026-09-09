"""Evaluate a trained segmentation model on the (unseen) test split.

Usage:
    python ml/src/evaluate.py --checkpoint ml/results/experiment_001/best_model.pth

Produces, in the checkpoint's experiment directory:
  * test_metrics.json          (aggregate metrics incl. building P/R/F1/IoU)
  * per_building_metrics.json  (one entry per test building)
  * confusion_matrix.png / .csv
  * predictions/<scene_id>/prediction.ply   (original XYZ + predicted class + confidence)

Test scenes are never seen during training (scene-level split from Step 1/2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# evaluate.py lives at <project>/ml/src/evaluate.py.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from synthetic_city.core.labels import CLASS_NAMES, NON_BUILDING_ID  # noqa: E402
from ml.src.data.dataset import list_scene_dirs, load_scene_pointcloud  # noqa: E402
from ml.src.inference.predict import build_model_from_checkpoint, predict_on_cloud, save_prediction_ply  # noqa: E402
from ml.src.training.metrics import segmentation_metrics, confusion_matrix  # noqa: E402
from ml.src.utils import select_device  # noqa: E402


def _point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def load_buildings(scene_dir):
    with open(Path(scene_dir) / "ground_truth" / "building.json", "r", encoding="utf-8") as f:
        return json.load(f)["buildings"]


def per_building_metrics(pred, xyz, building_id, buildings):
    """Per-building building-class precision/recall/F1/IoU."""
    records = []
    for b in buildings:
        bid = b["building_id"]
        poly = b["footprint_world"]
        own = building_id == bid
        n_own = int(own.sum())
        if n_own == 0:
            continue
        tp = int(((pred == 1) & own).sum())
        fn = n_own - tp
        # false positives: non-building points predicted building inside footprint
        fp = 0
        non_b = ~own & (building_id == NON_BUILDING_ID)
        cand = non_b & (pred == 1)
        if cand.any():
            xs, ys = xyz[cand, 0], xyz[cand, 1]
            fp = int(sum(_point_in_polygon(float(x), float(y), poly) for x, y in zip(xs, ys)))
        denom = tp + fp + fn
        iou = tp / denom if denom else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        records.append({
            "building_id": bid,
            "footprint_type": b["footprint_type"],
            "roof_type": b["roof_type"],
            "width": b["width"],
            "length": b["length"],
            "floor_count": b["floor_count"],
            "num_points": n_own,
            "building_iou": float(iou),
            "building_precision": float(prec),
            "building_recall": float(rec),
            "building_f1": float(f1),
        })
    return records


def known_style_sets(data_root):
    """Collect (footprint_type, roof_type) combos seen in the training split."""
    seen = set()
    for scene in list_scene_dirs(data_root, "train"):
        for b in load_buildings(scene):
            seen.add((b["footprint_type"], b["roof_type"]))
    return seen


def main():
    ap = argparse.ArgumentParser(description="Evaluate segmentation model on test split.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    exp_dir = ckpt_path.parent
    device = select_device("auto")
    model, ckpt = build_model_from_checkpoint(ckpt_path, device)
    cfg = ckpt["config"]
    num_classes = cfg["model"]["num_classes"]
    num_points = int(cfg["dataset"]["num_points"])
    data_root = Path(args.data) if args.data else (ROOT / cfg["dataset"]["data_root"])

    test_scenes = list_scene_dirs(data_root, "test")
    if not test_scenes:
        print("no test scenes found", file=sys.stderr)
        sys.exit(1)

    all_pred, all_target, all_xyz = [], [], []
    per_building = []
    preds_dir = exp_dir / "predictions"
    for scene in test_scenes:
        xyz, labels, building_id = load_scene_pointcloud(scene)
        pred, conf = predict_on_cloud(model, xyz, device, num_points, args.batch_size)
        all_pred.append(pred)
        all_target.append(labels)
        all_xyz.append(xyz)

        buildings = load_buildings(scene)
        per_building.extend(per_building_metrics(pred, xyz, building_id, buildings))

        out = preds_dir / scene.name / "prediction.ply"
        save_prediction_ply(out, xyz, pred, conf)
        print(f"scene {scene.name}: {xyz.shape[0]} points -> {out}")

    pred = np.concatenate(all_pred)
    target = np.concatenate(all_target)

    metrics = segmentation_metrics(pred, target, num_classes, list(CLASS_NAMES))
    cm = np.asarray(metrics["confusion_matrix"])

    # generalization analysis: known-style vs unseen (footprint, roof) combos
    seen = known_style_sets(data_root)
    known_rec = [r for r in per_building if (r["footprint_type"], r["roof_type"]) in seen]
    novel_rec = [r for r in per_building if (r["footprint_type"], r["roof_type"]) not in seen]

    def avg_iou(recs):
        return float(np.mean([r["building_iou"] for r in recs])) if recs else None

    out_metrics = {
        "checkpoint": str(ckpt_path),
        "seed": ckpt.get("seed"),
        "epoch": ckpt.get("epoch"),
        "num_test_scenes": len(test_scenes),
        "num_test_points": int(len(target)),
        "overall_accuracy": metrics["overall_accuracy"],
        "mean_iou": metrics["mean_iou"],
        "building_precision": metrics["per_class"]["building"]["precision"],
        "building_recall": metrics["per_class"]["building"]["recall"],
        "building_f1": metrics["per_class"]["building"]["f1"],
        "building_iou": metrics["per_class"]["building"]["iou"],
        "per_class": metrics["per_class"],
        "generalization": {
            "train_seen_combos": sorted([list(c) for c in seen]),
            "test_buildings_known_style": len(known_rec),
            "test_buildings_novel_style": len(novel_rec),
            "avg_building_iou_known_style": avg_iou(known_rec),
            "avg_building_iou_novel_style": avg_iou(novel_rec),
        },
    }

    (exp_dir / "test_metrics.json").write_text(json.dumps(out_metrics, indent=2), encoding="utf-8")
    (exp_dir / "per_building_metrics.json").write_text(
        json.dumps(per_building, indent=2), encoding="utf-8")
    np.savetxt(exp_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASS_NAMES), comments="")

    _save_confusion_plot(exp_dir / "confusion_matrix.png", cm, list(CLASS_NAMES))

    print("\n===== TEST METRICS (unseen buildings) =====")
    print(f"overall_accuracy : {out_metrics['overall_accuracy']:.4f}")
    print(f"mean_iou         : {out_metrics['mean_iou']:.4f}")
    print(f"building prec/rec/f1/iou : {out_metrics['building_precision']:.4f} / "
          f"{out_metrics['building_recall']:.4f} / {out_metrics['building_f1']:.4f} / "
          f"{out_metrics['building_iou']:.4f}")
    print("\nper-class IoU:")
    for name in CLASS_NAMES:
        print(f"  {name:<10} {metrics['per_class'][name]['iou']:.4f}")
    print("\nper-building building IoU:")
    for r in sorted(per_building, key=lambda r: r["building_iou"]):
        print(f"  {r['building_id']} {r['footprint_type']:>10} {r['roof_type']:>6} "
              f"iou={r['building_iou']:.4f} prec={r['building_precision']:.4f} "
              f"rec={r['building_recall']:.4f}")
    print("\nconfusion matrix:")
    print(cm)
    print(f"\nresults written to {exp_dir}")


def _save_confusion_plot(path, cm, class_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = cm.shape[0]
    cm_norm = cm.astype(np.float64)
    row_sum = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sum, out=np.zeros_like(cm_norm), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

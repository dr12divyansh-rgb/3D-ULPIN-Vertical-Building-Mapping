"""Phase 5-6: Evaluate Exp007 on held-out AHN4 test split and compare with Exp006.

Runs inference on the 100-chunk AHN4 TEST strip (northernmost 20% of tile
25GN2_01, never seen during fine-tuning). Reports Building IoU, precision,
recall, F1, confusion matrix, and comparison with Exp006 zero-shot result.

Usage:
    python ml/src/inference/evaluate_ahn4_exp007.py \\
        --checkpoint-exp007 ml/results/ahn4_exp007/best_model.pth \\
        --checkpoint-exp006 ml/results/experiment_006/best_model.pth \\
        --manifest          ml/results/ahn4_exp007/data/manifest.json \\
        --out               ml/results/ahn4_exp007
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
    input_dim = 6 if cfg["dataset"].get("input_mode", "xyz") == "xyzrgb" else 3
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


def run_inference(model, manifest_path: Path, split: str, device: torch.device, input_dim: int):
    ds = AHN4SplitDataset(manifest_path, split)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    all_pred, all_gt, all_xyz, all_conf = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            xyz    = batch["xyz"].to(device)   # (B, 4096, 6)
            labels = batch["labels"].numpy()

            if input_dim == 3:
                inp = xyz[:, :, :3]
            else:
                inp = xyz

            logits = model(inp)
            probs  = torch.softmax(logits, dim=-1)
            pred   = probs.argmax(dim=-1).cpu().numpy()
            conf   = probs.max(dim=-1).values.cpu().numpy()

            all_pred.append(pred.reshape(-1))
            all_gt.append(labels.reshape(-1))
            all_xyz.append(batch["xyz_orig"].numpy().reshape(-1, 3))
            all_conf.append(conf.reshape(-1))

            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(loader)} batches")

    return (
        np.concatenate(all_pred),
        np.concatenate(all_gt),
        np.concatenate(all_xyz),
        np.concatenate(all_conf),
    )


def write_ply(path: Path, xyz: np.ndarray, pred: np.ndarray, gt: np.ndarray, conf: np.ndarray):
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"comment CRS: EPSG:28992 (Amersfoort / RD New)\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar predicted_class\nproperty uchar gt_class\n"
        "property float confidence\nend_header\n"
    )
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
    print(f"    Saved: {path}  ({n:,} pts)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-exp007", default="ml/results/ahn4_exp007/best_model.pth")
    ap.add_argument("--checkpoint-exp006", default="ml/results/experiment_006/best_model.pth")
    ap.add_argument("--manifest",           default="ml/results/ahn4_exp007/data/manifest.json")
    ap.add_argument("--out",                default="ml/results/ahn4_exp007")
    ap.add_argument("--split",              default="test")
    args = ap.parse_args()

    ckpt007  = ROOT / args.checkpoint_exp007
    ckpt006  = ROOT / args.checkpoint_exp006
    manifest = ROOT / args.manifest
    out_dir  = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir  = out_dir / "visuals"
    vis_dir.mkdir(exist_ok=True)

    device = select_device("auto")
    print(f"\nAHN4 Evaluation: Exp006 vs Exp007")
    print(f"  Test split: {args.split}  (100 chunks, north strip)")
    print(f"  Device: {device}")

    results = {}

    # ── Exp007 inference ─────────────────────────────────────────────────────
    if ckpt007.exists():
        print(f"\n[Exp007] Loading {ckpt007.name}...")
        m007, dim007 = load_model(ckpt007, device)
        pred007, gt007, xyz007, conf007 = run_inference(m007, manifest, args.split, device, dim007)
        metrics007 = segmentation_metrics(pred007, gt007, NUM_CLASSES, CLASS_NAMES)
        results["exp007"] = metrics007
        write_ply(vis_dir / "exp007_test_prediction.ply", xyz007, pred007, gt007, conf007)
    else:
        print(f"  WARNING: {ckpt007} not found — skipping Exp007")
        pred007, gt007, xyz007, conf007 = None, None, None, None

    # ── Exp006 inference on same test chunks ──────────────────────────────────
    print(f"\n[Exp006] Loading {ckpt006.name}...")
    m006, dim006 = load_model(ckpt006, device)
    pred006, gt006, xyz006, conf006 = run_inference(m006, manifest, args.split, device, dim006)
    metrics006 = segmentation_metrics(pred006, gt006, NUM_CLASSES, CLASS_NAMES)
    results["exp006"] = metrics006
    write_ply(vis_dir / "exp006_test_prediction.ply", xyz006, pred006, gt006, conf006)

    # Ground truth PLY
    write_ply(vis_dir / "ground_truth.ply", xyz006, gt006, gt006,
              np.ones(len(gt006), dtype=np.float32))

    # ── Print comparison ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"AHN4 TEST RESULTS (100 chunks, north strip, unseen during training)")
    print(f"{'='*65}")
    print(f"\n{'Model':<12}  {'Bldg IoU':>9}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Gnd IoU':>8}  {'Acc':>7}  {'mIoU':>7}")
    print("-"*65)

    def row(name, m):
        b = m["per_class"]["building"]
        g = m["per_class"]["ground"]
        return (f"  {name:<10}  "
                f"{b['iou']:>9.4f}  {b['precision']:>7.4f}  {b['recall']:>7.4f}  {b['f1']:>7.4f}  "
                f"{g['iou']:>8.4f}  {m['overall_accuracy']:>7.4f}  {m['mean_iou']:>7.4f}")

    print(row("Exp006", metrics006))
    if "exp007" in results:
        print(row("Exp007", results["exp007"]))
        b006 = metrics006["per_class"]["building"]["iou"]
        b007 = results["exp007"]["per_class"]["building"]["iou"]
        delta = b007 - b006
        print(f"\n  Building IoU improvement: Exp006={b006:.4f}  ->  Exp007={b007:.4f}  (delta={delta:+.4f})")
    print(f"{'='*65}")

    # ── Prediction distribution ───────────────────────────────────────────────
    for name, pred in [("Exp006", pred006), ("Exp007", pred007 if pred007 is not None else [])]:
        if len(pred) == 0: continue
        n = len(pred)
        print(f"\n  {name} prediction distribution:")
        for c, cn in enumerate(CLASS_NAMES):
            cnt = int((pred == c).sum())
            if cnt > 0:
                print(f"    {cn:12} ({c}): {cnt:>8,}  ({100*cnt/n:.1f}%)")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    for name, m, pred, gt in [
        ("exp006", metrics006, pred006, gt006),
        ("exp007", results.get("exp007"), pred007, gt007),
    ]:
        if m is None or pred is None: continue
        cm = np.asarray(m["confusion_matrix"])
        cm_path = out_dir / f"{name}_test_confusion.csv"
        with open(cm_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + CLASS_NAMES[:NUM_CLASSES])
            for i, row_vals in enumerate(cm):
                lbl = CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i)
                w.writerow([lbl] + [int(v) for v in row_vals])

    # ── Save JSON ─────────────────────────────────────────────────────────────
    def fmt_result(m):
        if m is None: return None
        b = m["per_class"]["building"]
        g = m["per_class"]["ground"]
        return {
            "building_iou":       float(b["iou"]),
            "building_precision": float(b["precision"]),
            "building_recall":    float(b["recall"]),
            "building_f1":        float(b["f1"]),
            "ground_iou":         float(g["iou"]),
            "overall_accuracy":   float(m["overall_accuracy"]),
            "mean_iou":           float(m["mean_iou"]),
            "per_class":          {k: {m2: float(v2) for m2, v2 in d.items()}
                                   for k, d in m["per_class"].items()},
        }

    comparison = {
        "test_split":           args.split,
        "n_test_chunks":        100,
        "n_test_points":        409600,
        "tile":                 "AHN4 25GN2_01 Amsterdam",
        "test_strip_Y_min":     487203,
        "exp006_zero_shot":     fmt_result(metrics006),
        "exp007_finetuned":     fmt_result(results.get("exp007")),
        "classes_with_gt":      ["ground (0)", "building (1)"],
        "classes_absent_in_gt": ["vegetation (2)", "road (3)", "vehicle (4)", "other (5)"],
        "visuals_dir":          str(vis_dir),
    }
    out_json = out_dir / "comparison_exp006_vs_exp007.json"
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Comparison JSON : {out_json}")
    print(f"  Visuals dir     : {vis_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

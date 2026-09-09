"""Compare segmentation + LOD2 reconstruction metrics across two experiments.

Usage:
    # Step 1 – evaluate experiment_002 on the test split:
    python ml/src/evaluate.py --checkpoint ml/results/experiment_002/best_model.pth

    # Step 2 – reconstruct LOD2 from experiment_002 predictions:
    python ml/src/reconstruction/reconstruct.py \\
        --checkpoint ml/results/experiment_002/best_model.pth \\
        --scene scene_00005 \\
        --out ml/results/experiment_002/reconstruction/scene_00005/predicted_lod2

    # Step 3 – evaluate reconstruction:
    python ml/src/evaluation/evaluate_reconstruction.py \\
        --predicted ml/results/experiment_002/reconstruction/scene_00005/predicted_lod2 \\
        --gt ml/data/synthetic/test/scene_00005

    # Step 4 – compare:
    python ml/src/evaluation/compare_experiments.py \\
        --exp1 ml/results/experiment_001 \\
        --exp2 ml/results/experiment_002

Prints a side-by-side table: segmentation IoU per class and LOD2 reconstruction
metrics (height error, footprint IoU, chamfer), then writes comparison.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _fmt(v, digits=4) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _pct_change(a, b) -> str:
    if a is None or b is None or a == 0:
        return ""
    delta = b - a
    pct = 100 * delta / abs(a)
    sign = "+" if pct >= 0 else ""
    return f"({sign}{pct:.1f}%)"


def compare(exp1_dir: Path, exp2_dir: Path) -> dict:
    e1_seg  = _load_json(exp1_dir / "test_metrics.json")
    e2_seg  = _load_json(exp2_dir / "test_metrics.json")
    e1_rec  = _load_json(exp1_dir / "reconstruction" / "scene_00005" / "evaluation.json")
    e2_rec  = _load_json(exp2_dir / "reconstruction" / "scene_00005" / "evaluation.json")
    e1_sum  = _load_json(exp1_dir / "summary.json")
    e2_sum  = _load_json(exp2_dir / "summary.json")

    CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]

    print("\n" + "=" * 78)
    print(f"{'EXPERIMENT COMPARISON':^78}")
    print(f"  exp1: {exp1_dir}")
    print(f"  exp2: {exp2_dir}")
    print("=" * 78)

    # ------------------------------------------------------------------ #
    # Training summary                                                     #
    # ------------------------------------------------------------------ #
    print("\n── Training ─────────────────────────────────────────────────────────────")
    print(f"{'Metric':<40} {'Exp1':>10} {'Exp2':>10} {'Change':>12}")
    print("-" * 78)

    def _row(label, v1, v2, digits=4):
        change = _pct_change(v1, v2) if isinstance(v1, float) else ""
        print(f"{label:<40} {_fmt(v1, digits):>10} {_fmt(v2, digits):>10} {change:>12}")

    _row("Best epoch", e1_sum["best_epoch"] if e1_sum else None,
         e2_sum["best_epoch"] if e2_sum else None, 0)
    _row("Best val mIoU",
         e1_sum["best_val_mean_iou"] if e1_sum else None,
         e2_sum["best_val_mean_iou"] if e2_sum else None)

    # ------------------------------------------------------------------ #
    # Segmentation metrics                                                 #
    # ------------------------------------------------------------------ #
    print("\n── Segmentation (test split, 570k points) ───────────────────────────────")
    print(f"{'Metric':<40} {'Exp1':>10} {'Exp2':>10} {'Change':>12}")
    print("-" * 78)

    def _seg(key, sub=None):
        if e1_seg is None or e2_seg is None:
            return None, None
        v1 = e1_seg.get(key)
        v2 = e2_seg.get(key)
        if sub is not None:
            v1 = v1.get(sub) if isinstance(v1, dict) else None
            v2 = v2.get(sub) if isinstance(v2, dict) else None
        return v1, v2

    v1, v2 = _seg("overall_accuracy"); _row("Overall accuracy", v1, v2)
    v1, v2 = _seg("mean_iou");         _row("Mean IoU (mIoU)", v1, v2)
    v1, v2 = _seg("building_iou");     _row("Building IoU  ← primary metric", v1, v2)
    v1, v2 = _seg("building_f1");      _row("Building F1", v1, v2)

    print()
    for cls in CLASS_NAMES:
        def _cls_iou(d):
            if d is None:
                return None
            return d.get("per_class", {}).get(cls, {}).get("iou")
        v1 = _cls_iou(e1_seg)
        v2 = _cls_iou(e2_seg)
        _row(f"  IoU: {cls}", v1, v2)

    # ------------------------------------------------------------------ #
    # LOD2 reconstruction metrics                                          #
    # ------------------------------------------------------------------ #
    print("\n── LOD2 Reconstruction (scene_00005, 18 buildings) ─────────────────────")
    print(f"{'Metric':<40} {'Exp1':>10} {'Exp2':>10} {'Change':>12}")
    print("-" * 78)

    def _agg(d, key):
        if d is None:
            return None
        return d.get("aggregate", {}).get(key)

    metrics_rec = [
        ("Mean height error (m)",            "mean_height_error_m"),
        ("Mean footprint IoU",               "mean_footprint_iou"),
        ("Mean chamfer distance (m)",        "mean_chamfer_distance_m"),
        ("Mean area error relative",         "mean_footprint_area_error_relative"),
        ("Watertight buildings",             "n_watertight"),
    ]
    for label, key in metrics_rec:
        v1 = _agg(e1_rec, key)
        v2 = _agg(e2_rec, key)
        _row(label, v1, v2,
             digits=(0 if key == "n_watertight" else 4))

    print("\n── Summary ──────────────────────────────────────────────────────────────")
    if e1_seg and e2_seg:
        b1 = e1_seg.get("building_iou", 0)
        b2 = e2_seg.get("building_iou", 0)
        v1_5 = e1_seg.get("per_class", {}).get("vehicle", {}).get("iou", 0)
        v2_5 = e2_seg.get("per_class", {}).get("vehicle", {}).get("iou", 0)
        r1 = e1_seg.get("per_class", {}).get("road", {}).get("iou", 0)
        r2 = e2_seg.get("per_class", {}).get("road", {}).get("iou", 0)

        delta_b = b2 - b1
        delta_v = v2_5 - v1_5
        delta_r = r2 - r1

        print(f"\n  Building IoU:  {b1:.4f} → {b2:.4f}  ({delta_b:+.4f})")
        print(f"  Vehicle  IoU:  {v1_5:.4f} → {v2_5:.4f}  ({delta_v:+.4f})")
        print(f"  Road     IoU:  {r1:.4f} → {r2:.4f}  ({delta_r:+.4f})")

        if delta_b < -0.01:
            print("\n  ⚠  Building IoU dropped >1pp. Class weights over-penalised building.")
            print("     Consider: lower weight_cap, focal loss, or a custom weight schedule.")
        elif delta_v > 0.01 or delta_r > 0.01:
            print("\n  ✓  Rare-class IoU improved without significant building regression.")
        else:
            print("\n  ⟳  No clear winner. Further tuning required.")
    else:
        missing = []
        if not e1_seg: missing.append("exp1/test_metrics.json")
        if not e2_seg: missing.append("exp2/test_metrics.json")
        print(f"\n  Cannot summarise — missing: {', '.join(missing)}")
        print("  Run: python ml/src/evaluate.py --checkpoint <exp_dir>/best_model.pth")

    # ------------------------------------------------------------------ #
    # Write JSON                                                           #
    # ------------------------------------------------------------------ #
    result = {
        "exp1": str(exp1_dir),
        "exp2": str(exp2_dir),
        "segmentation": {
            "exp1": e1_seg,
            "exp2": e2_seg,
        },
        "reconstruction": {
            "exp1": e1_rec,
            "exp2": e2_rec,
        },
    }
    out = exp2_dir / "comparison_vs_exp1.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Full results written → {out}\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two experiment directories.")
    ap.add_argument("--exp1", default="ml/results/experiment_001",
                    help="Path to experiment_001 directory.")
    ap.add_argument("--exp2", default="ml/results/experiment_002",
                    help="Path to experiment_002 directory.")
    args = ap.parse_args()

    compare(ROOT / args.exp1, ROOT / args.exp2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare experiment_001 (synthetic baseline) against experiment_003/004 (STPLS3D).

Usage:
    python ml/src/evaluation/compare_stpls3d_experiments.py \
        --exp001 ml/results/experiment_001 \
        --exp003 ml/results/experiment_003 \
        --exp004 ml/results/experiment_004  # optional

Outputs a JSON + human-readable summary in the last experiment's directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]


def _load_metrics(exp_dir: Path) -> dict | None:
    """Load test metrics from an experiment directory.

    Tries stpls3d_test_metrics.json first (for STPLS3D experiments), then
    test_metrics.json (for synthetic experiments).
    """
    for fname in ("stpls3d_test_metrics.json", "test_metrics.json"):
        p = exp_dir / fname
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_history(exp_dir: Path) -> list:
    p = exp_dir / "history.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return []


def _load_config(exp_dir: Path) -> dict:
    p = exp_dir / "config.yaml"
    if p.exists():
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return {}


def _format_row(name: str, metrics: dict) -> str:
    pc = metrics.get("per_class", {}).get(name, {})
    iou = pc.get("iou", float("nan"))
    prec = pc.get("precision", float("nan"))
    rec = pc.get("recall", float("nan"))
    f1 = pc.get("f1", float("nan"))
    return f"{name:<12}  iou={iou:.4f}  p={prec:.4f}  r={rec:.4f}  f1={f1:.4f}"


def compare(exp_dirs: list[tuple[str, Path, dict | None]]) -> dict:
    """Build the comparison report dict."""
    report = {"experiments": {}}

    for label, exp_dir, metrics in exp_dirs:
        if metrics is None:
            report["experiments"][label] = {"status": "no metrics found"}
            continue
        history = _load_history(exp_dir)
        cfg = _load_config(exp_dir)

        entry = {
            "path": str(exp_dir),
            "dataset_source": cfg.get("dataset", {}).get("source", "synthetic"),
            "overall_accuracy": metrics.get("overall_accuracy"),
            "mean_iou": metrics.get("mean_iou"),
            "num_test_points": metrics.get("num_test_points"),
            "best_epoch": history[-1]["epoch"] if history else None,
            "per_class": {},
        }
        for cn in CLASS_NAMES:
            pc = metrics.get("per_class", {}).get(cn, {})
            entry["per_class"][cn] = pc
        report["experiments"][label] = entry

    # Build comparison tables
    metric_labels = sorted(report["experiments"].keys())
    table_rows = []
    for cn in CLASS_NAMES:
        row = {"class": cn}
        for lbl in metric_labels:
            exp = report["experiments"].get(lbl, {})
            if "per_class" in exp:
                pc = exp["per_class"].get(cn, {})
                row[lbl] = {
                    "iou": pc.get("iou"),
                    "f1": pc.get("f1"),
                    "precision": pc.get("precision"),
                    "recall": pc.get("recall"),
                }
        table_rows.append(row)
    report["per_class_comparison"] = table_rows

    # Summary row
    report["miou_summary"] = {
        lbl: report["experiments"].get(lbl, {}).get("mean_iou")
        for lbl in metric_labels
    }
    return report


def print_comparison(report: dict):
    experiments = report["experiments"]
    exp_labels = sorted(experiments.keys())

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON — SIH 26011")
    print("=" * 70)

    for lbl in exp_labels:
        exp = experiments[lbl]
        print(f"\n  [{lbl}]  {exp.get('path','')}")
        print(f"    Dataset source : {exp.get('dataset_source', '?')}")
        print(f"    Test points    : {exp.get('num_test_points', '?'):,}" if isinstance(exp.get('num_test_points'), int) else f"    Test points    : {exp.get('num_test_points', '?')}")
        print(f"    Overall acc    : {exp.get('overall_accuracy', float('nan')):.4f}")
        print(f"    Mean IoU       : {exp.get('mean_iou', float('nan')):.4f}")

    print("\n  Per-class IoU comparison:")
    print(f"  {'class':<12}", end="")
    for lbl in exp_labels:
        print(f"  {lbl:>10}", end="")
    print()
    print(f"  {'-'*12}", end="")
    for _ in exp_labels:
        print(f"  {'-'*10}", end="")
    print()

    for row in report["per_class_comparison"]:
        cn = row["class"]
        print(f"  {cn:<12}", end="")
        for lbl in exp_labels:
            iou_val = row.get(lbl, {}).get("iou")
            if iou_val is not None:
                print(f"  {iou_val:>10.4f}", end="")
            else:
                print(f"  {'n/a':>10}", end="")
        print()

    print("\n  Mean IoU:")
    for lbl in exp_labels:
        miou = report["miou_summary"].get(lbl)
        print(f"    {lbl}: {miou:.4f}" if miou is not None else f"    {lbl}: n/a")

    print("\n  Building-specific comparison:")
    print(f"  {'exp':<10}  {'iou':>6}  {'prec':>6}  {'rec':>6}  {'f1':>6}")
    for lbl in exp_labels:
        exp = experiments.get(lbl, {})
        pc = exp.get("per_class", {}).get("building", {})
        print(f"  {lbl:<10}  {pc.get('iou', float('nan')):>6.4f}  "
              f"{pc.get('precision', float('nan')):>6.4f}  "
              f"{pc.get('recall', float('nan')):>6.4f}  "
              f"{pc.get('f1', float('nan')):>6.4f}")
    print("=" * 70 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp001", required=True, help="experiment_001 directory")
    ap.add_argument("--exp003", required=True, help="experiment_003 directory")
    ap.add_argument("--exp004", default=None, help="experiment_004 directory (optional)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: exp003/comparison_001_vs_003.json)")
    args = ap.parse_args()

    dirs = [
        ("exp001", Path(args.exp001)),
        ("exp003", Path(args.exp003)),
    ]
    if args.exp004:
        dirs.append(("exp004", Path(args.exp004)))

    exp_data = [(lbl, d, _load_metrics(d)) for lbl, d in dirs]

    report = compare(exp_data)
    print_comparison(report)

    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.exp003) / "comparison_001_vs_003.json"
        if args.exp004:
            out_path = Path(args.exp004) / "comparison_001_vs_003_vs_004.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()

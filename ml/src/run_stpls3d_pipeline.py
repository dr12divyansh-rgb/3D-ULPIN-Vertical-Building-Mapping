"""Full post-training pipeline for the STPLS3D milestone.

Chains: eval exp003 → compare → train exp004 → eval exp004 on both test sets.

Usage:
    python ml/src/run_stpls3d_pipeline.py [--skip-exp003-train] [--skip-exp004-train]

Pass --skip-exp003-train if experiment_003 already has best_model.pth.
Pass --skip-exp004-train if experiment_004 already has best_model.pth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

def run(cmd: list[str], desc: str = "") -> int:
    print(f"\n{'='*60}")
    print(f"RUNNING: {desc or ' '.join(cmd[:3])}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"ERROR: command failed with code {result.returncode}: {' '.join(cmd)}")
    return result.returncode


def find_latest_experiment(results_dir: Path, prefix: str = "experiment_") -> Path | None:
    dirs = sorted([d for d in results_dir.iterdir()
                   if d.is_dir() and d.name.startswith(prefix)],
                  key=lambda d: d.name)
    return dirs[-1] if dirs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-exp003-train", action="store_true")
    ap.add_argument("--skip-exp004-train", action="store_true")
    ap.add_argument("--results", default=str(ROOT / "ml" / "results"))
    args = ap.parse_args()

    results_dir = Path(args.results)
    stpls3d_data = ROOT / "ml" / "data" / "stpls3d"
    syn_data = ROOT / "ml" / "data" / "synthetic"

    # -----------------------------------------------------------------------
    # 1. Train Experiment 003 (STPLS3D-only) if not already done
    # -----------------------------------------------------------------------
    exp003 = results_dir / "experiment_003"
    if not args.skip_exp003_train:
        if (exp003 / "best_model.pth").exists():
            print(f"\nexp003 checkpoint already exists: {exp003 / 'best_model.pth'}")
            print("Skipping training (use --skip-exp003-train to suppress this check)")
        else:
            rc = run([PYTHON, "ml/src/training/train.py",
                      "--config", "ml/config/stpls3d_semantic.yaml",
                      "--results", str(results_dir)],
                     "Train Experiment 003: STPLS3D-only")
            if rc != 0:
                print("ERROR: exp003 training failed")
                sys.exit(rc)
            exp003 = find_latest_experiment(results_dir)
    else:
        print(f"\nSkipping exp003 training. Using: {exp003}")

    if not (exp003 / "best_model.pth").exists():
        print(f"ERROR: best_model.pth not found in {exp003}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Evaluate Experiment 003 on STPLS3D test set
    # -----------------------------------------------------------------------
    if not (exp003 / "stpls3d_test_metrics.json").exists():
        rc = run([PYTHON, "ml/src/evaluate_stpls3d.py",
                  "--checkpoint", str(exp003 / "best_model.pth"),
                  "--data", str(stpls3d_data),
                  "--batch-size", "8"],
                 "Evaluate Experiment 003 on STPLS3D test set")
        if rc != 0:
            print("WARNING: evaluation failed")
    else:
        print(f"\nStpls3d test metrics already exist for exp003. Skipping evaluation.")

    # -----------------------------------------------------------------------
    # 3. Train Experiment 004 (mixed synthetic + STPLS3D)
    # -----------------------------------------------------------------------
    exp004 = results_dir / "experiment_004"
    if not args.skip_exp004_train:
        if (exp004 / "best_model.pth").exists():
            print(f"\nexp004 checkpoint already exists: {exp004 / 'best_model.pth'}")
        else:
            rc = run([PYTHON, "ml/src/training/train.py",
                      "--config", "ml/config/mixed_semantic.yaml",
                      "--results", str(results_dir)],
                     "Train Experiment 004: Mixed synthetic + STPLS3D")
            if rc != 0:
                print("ERROR: exp004 training failed")
                sys.exit(rc)
            exp004 = find_latest_experiment(results_dir)
    else:
        print(f"\nSkipping exp004 training. Using: {exp004}")

    if not (exp004 / "best_model.pth").exists():
        print(f"ERROR: best_model.pth not found in {exp004}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 4a. Evaluate Experiment 004 on STPLS3D test set
    # -----------------------------------------------------------------------
    if not (exp004 / "stpls3d_test_metrics.json").exists():
        rc = run([PYTHON, "ml/src/evaluate_stpls3d.py",
                  "--checkpoint", str(exp004 / "best_model.pth"),
                  "--data", str(stpls3d_data),
                  "--batch-size", "8"],
                 "Evaluate Experiment 004 on STPLS3D test set")
        if rc != 0:
            print("WARNING: exp004 stpls3d evaluation failed")
    else:
        print(f"\nStpls3d test metrics already exist for exp004. Skipping.")

    # -----------------------------------------------------------------------
    # 4b. Evaluate Experiment 004 on Synthetic test set
    # -----------------------------------------------------------------------
    if not (exp004 / "test_metrics.json").exists():
        rc = run([PYTHON, "ml/src/evaluate.py",
                  "--checkpoint", str(exp004 / "best_model.pth"),
                  "--data", str(syn_data),
                  "--batch-size", "8"],
                 "Evaluate Experiment 004 on Synthetic test set")
        if rc != 0:
            print("WARNING: exp004 synthetic evaluation failed — mixed model may not match synthetic schema")
    else:
        print(f"\nSynthetic test metrics already exist for exp004. Skipping.")

    # -----------------------------------------------------------------------
    # 5. Generate comparison: exp001 vs exp003 vs exp004
    # -----------------------------------------------------------------------
    exp001 = results_dir / "experiment_001"
    compare_out = exp004 / "comparison_001_vs_003_vs_004.json"
    if not compare_out.exists():
        rc = run([PYTHON, "ml/src/evaluation/compare_stpls3d_experiments.py",
                  "--exp001", str(exp001),
                  "--exp003", str(exp003),
                  "--exp004", str(exp004),
                  "--out", str(compare_out)],
                 "Compare experiments 001, 003, 004")
        if rc != 0:
            print("WARNING: comparison failed")
    else:
        print(f"\nComparison already exists: {compare_out}")

    # -----------------------------------------------------------------------
    # 6. Print summary of available results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — AVAILABLE RESULTS")
    print("=" * 60)
    for exp_dir, exp_label in [(exp001, "exp001"), (exp003, "exp003"), (exp004, "exp004")]:
        print(f"\n  {exp_label}: {exp_dir}")
        for fname in ["test_metrics.json", "stpls3d_test_metrics.json",
                      "history.json", "summary.json",
                      "comparison_001_vs_003_vs_004.json"]:
            p = exp_dir / fname
            status = "EXISTS" if p.exists() else "MISSING"
            print(f"    [{status}] {fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()

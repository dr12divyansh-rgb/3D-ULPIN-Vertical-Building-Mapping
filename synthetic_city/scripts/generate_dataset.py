"""Orchestrator: generate the full synthetic dataset.

Usage:
    python synthetic_city/scripts/generate_dataset.py [--config PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from synthetic_city.utils import config as config_util  # noqa: E402
from synthetic_city.utils import manifest as manifest_util  # noqa: E402
from synthetic_city.utils import blender_runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic building + LiDAR dataset.")
    ap.add_argument("--config", default=str(ROOT / "synthetic_city" / "config" / "default.yaml"))
    ap.add_argument("--out", default=None, help="override dataset.output_dir")
    args = ap.parse_args()

    cfg = config_util.load_config(args.config)

    out_root = Path(args.out) if args.out else (ROOT / cfg["dataset"]["output_dir"])
    out_root = out_root.resolve()

    # Clear previously generated scene output so stale scenes from an earlier
    # run (with a different split) can never leak into this dataset.
    for sub in ("train", "validation", "test"):
        p = out_root / sub
        if p.exists():
            shutil.rmtree(p)
    debug_dir = out_root / "metadata" / "debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)

    manifest = manifest_util.build_manifest(cfg)
    split_counts = manifest_util.split_counts(manifest)

    # ---- write dataset-level metadata (reproducibility) ----
    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    (meta_dir / "dataset_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (meta_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (meta_dir / "split_summary.json").write_text(
        json.dumps(split_counts, indent=2), encoding="utf-8")

    blender = blender_runner.find_blender(cfg["blender"].get("executable"))
    blender_version = _blender_version(blender)

    job_path = meta_dir / "job.json"
    job_path.write_text(json.dumps({"config": cfg, "manifest": manifest}), encoding="utf-8")

    generation_info = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "blender": str(blender),
        "blender_version": blender_version,
        "config_file": str(Path(args.config).resolve()),
        "output_dir": str(out_root),
        "python": sys.version.split()[0],
        "split_counts": split_counts,
    }
    (meta_dir / "generation_info.json").write_text(
        json.dumps(generation_info, indent=2), encoding="utf-8")

    print(f"Blender: {blender} ({blender_version})")
    print(f"Scenes: {split_counts}")
    print(f"Output : {out_root}")
    print("Generating scenes with Blender (this may take a while)...")

    script = ROOT / "synthetic_city" / "blender" / "generate_scenes.py"
    proc = blender_runner.run_blender_script(blender, script,
                                             ["--job", str(job_path), "--out", str(out_root)])

    # Relay Blender stderr (progress) to the user.
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode != 0:
        print(f"Blender failed with exit code {proc.returncode}", file=sys.stderr)
        sys.exit(1)

    _parse_summary(proc.stdout)
    print("Dataset generation complete.")
    print(f"Run validation:  python synthetic_city/scripts/validate_dataset.py --config {args.config}")


def _blender_version(blender: Path) -> str:
    try:
        proc = subprocess.run([str(blender), "--version"], capture_output=True, text=True, timeout=60)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line and "Blender" in line:
                return line
    except Exception:
        pass
    return "unknown"


def _parse_summary(stdout: str) -> None:
    if "SUMMARY_JSON_START" not in stdout:
        print("Warning: no summary returned from Blender.", file=sys.stderr)
        return
    payload = stdout.split("SUMMARY_JSON_START", 1)[1].split("SUMMARY_JSON_END", 1)[0]
    try:
        results = json.loads(payload)
    except json.JSONDecodeError:
        print("Warning: could not parse Blender summary.", file=sys.stderr)
        return
    print("\nPer-scene results:")
    for r in results:
        print(f"  [{r['split']:>10}] {r['scene_id']}  points={r['num_points']:>8}  "
              f"buildings={r['num_buildings']}")


if __name__ == "__main__":
    main()

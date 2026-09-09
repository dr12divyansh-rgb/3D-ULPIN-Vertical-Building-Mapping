"""Evaluate a predicted LOD2 mesh against the ground-truth LOD2 mesh.

This step does not train a model, so today it is exercised as a sanity test:
comparing the ground truth against itself must report (near) zero error.

Usage:
    # compare two OBJ files directly
    python synthetic_city/scripts/evaluate_lod2.py --pred pred.obj --gt gt.obj

    # sanity check: ground-truth LOD2 vs. itself (must report ~0 error)
    python synthetic_city/scripts/evaluate_lod2.py --scene <scene_dir> [--building <id>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthetic_city.core import evaluation  # noqa: E402
from synthetic_city.core.mesh import SurfaceMesh  # noqa: E402


def _find_scene_dir(scene_arg, data_root):
    p = Path(scene_arg)
    if p.is_dir():
        return p
    for split in ("train", "validation", "test", "metadata/debug"):
        cand = Path(data_root) / split / scene_arg
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"scene not found: {scene_arg}")


def _print(results, title):
    print(f"{title}")
    g = results["geometry"]
    t = results["topology"]
    c = results["counts"]
    print(f"  height          : pred {g['height_pred']:.4f}  gt {g['height_gt']:.4f}  "
          f"err {g['height_error']:.6f}")
    print(f"  footprint area  : pred {g['footprint_area_pred']:.4f}  "
          f"gt {g['footprint_area_gt']:.4f}  err {g['footprint_area_error']:.6f}")
    print(f"  footprint IoU   : {g['footprint_iou']:.6f}")
    print(f"  chamfer distance: {g['chamfer_distance']:.6f}")
    print(f"  watertight      : pred {t['watertight_pred']}  gt {t['watertight_gt']}")
    print(f"  non-manifold    : pred {t['non_manifold_edges_pred']}  "
          f"gt {t['non_manifold_edges_gt']}")
    print(f"  zero-area faces : pred {t['zero_area_faces_pred']}  "
          f"gt {t['zero_area_faces_gt']}")
    print(f"  faces/verts     : pred {c['faces_pred']}/{c['vertices_pred']}  "
          f"gt {c['faces_gt']}/{c['vertices_gt']}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate predicted LOD2 vs ground truth.")
    ap.add_argument("--pred", default=None, help="predicted LOD2 OBJ")
    ap.add_argument("--gt", default=None, help="ground-truth LOD2 OBJ")
    ap.add_argument("--scene", default=None, help="scene dir or id (sanity self-check)")
    ap.add_argument("--data", default=None, help="dataset root")
    ap.add_argument("--building", type=int, default=None)
    ap.add_argument("--samples", type=int, default=2000)
    args = ap.parse_args()

    if args.pred and args.gt:
        pred = SurfaceMesh.from_obj(args.pred)
        gt = SurfaceMesh.from_obj(args.gt)
        results = evaluation.evaluate_lod2(pred, gt, args.samples)
        _print(results, f"Evaluation: {args.pred} vs {args.gt}")
        print(f"  sane (near-zero error): {evaluation.is_sane(results)}")
        return

    if args.scene:
        data_root = Path(args.data) if args.data else (ROOT / "ml" / "data" / "synthetic")
        scene_dir = _find_scene_dir(args.scene, data_root)
        lod2_meta = json.loads((scene_dir / "ground_truth" / "lod2" / "metadata.json")
                               .read_text(encoding="utf-8"))
        bids = [b["building_id"] for b in lod2_meta["buildings"]]
        bid = args.building if args.building is not None else bids[0]
        meta = next(b for b in lod2_meta["buildings"] if b["building_id"] == bid)
        gt = SurfaceMesh.from_obj(scene_dir / "ground_truth" / "lod2" / meta["obj"])
        results = evaluation.evaluate_lod2(gt, gt, args.samples)
        _print(results, f"Sanity check: scene={scene_dir.name} building={bid} "
                        f"(GT vs itself)")
        ok = evaluation.is_sane(results)
        print(f"  sane (near-zero error): {ok}")
        sys.exit(0 if ok else 1)

    print("Specify either --pred/--gt or --scene.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()

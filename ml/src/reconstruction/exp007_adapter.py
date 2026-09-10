"""Exp007 Pipeline Adapter — SIH 26011.

Wraps pipeline_v3.py for the specific case of AHN4 Exp007 inference output.

When Session 1 produces Exp007 prediction, run this ONE command:

    python ml/src/reconstruction/exp007_adapter.py \\
        --pred <path-to-exp007-prediction.ply> \\
        --out  ml/results/reconstruction_v3_future/ahn4_exp007 \\
        [--export-viewer]

The pipeline is IDENTICAL to what was used for Exp006/zeroRGB.
Only the input PLY path changes.

Expected input PLY schema (same as all other prediction PLYs):
    float x, y, z           ← EPSG:28992 (Amersfoort / RD New)
    uchar red, green, blue  ← class colour (optional — ignored)
    uchar predicted_class   ← 0-5 project class
    [uchar gt_class]        ← optional ground truth
    [float confidence]      ← optional softmax probability

Output schema: reconstruction_v3 (same as all other runs).

Zero architecture changes needed for Exp007 — this script is the adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.src.reconstruction.pipeline_v3 import run_pipeline
from ml.src.reconstruction.export_viewer_json import convert as export_viewer_json

# ── Hard-coded AHN4 parameters (do not change) ────────────────────────────────

AHN4_PARAMS = {
    "dataset":    "AHN4",
    "scene_id":   "25GN2_01_Amsterdam",
    "crs":        "EPSG:28992 (Amersfoort / RD New)",
    "bag_path":   ROOT / "ml/results/ahn4_inference/3dbag_reference.json",
    "max_buildings": 300,
    "floor_height":  3.2,
    "apply_plausibility": True,
    "apply_height_split": True,
}

# The model description is updated here when Exp007 metrics are known.
# Fill in the actual Building IoU after Session 1 evaluates the model.
MODEL_DESCRIPTION_TEMPLATE = (
    "PointNet++ Exp007 (AHN4 fine-tuned from Exp006; "
    "AHN4 Building IoU={building_iou})"
)


def get_exp007_iou() -> str:
    """Try to read Exp007 test IoU from Session 1's evaluation output.

    Returns the IoU as a string, or 'TBD' if not yet available.
    """
    eval_path = ROOT / "ml/results/ahn4_exp007"

    # Check for comparison file (produced by evaluate_ahn4_exp007.py)
    comp = eval_path / "comparison_exp006_vs_exp007.json"
    if comp.exists():
        try:
            with open(comp) as f:
                d = json.load(f)
            iou = d.get("exp007", {}).get("building_iou")
            if iou is not None:
                return f"{float(iou):.4f}"
        except (json.JSONDecodeError, KeyError):
            pass

    # Check training history for best val IoU
    history = eval_path / "history.json"
    if history.exists():
        try:
            with open(history) as f:
                h = json.load(f)
            if isinstance(h, list) and h:
                best_val = max(ep.get("val_biou", 0) for ep in h)
                return f"{best_val:.4f} (val, training)"
        except (json.JSONDecodeError, KeyError):
            pass

    return "TBD"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run reconstruction pipeline on Exp007 AHN4 prediction. "
                    "Identical to Exp006 pipeline — only the PLY path changes.")

    ap.add_argument("--pred", required=True,
                    help="Path to Exp007 prediction PLY "
                         "(e.g. ml/results/ahn4_exp007/full_inference/merged.ply)")
    ap.add_argument("--out", default="ml/results/reconstruction_v3_future/ahn4_exp007",
                    help="Output directory (default: reconstruction_v3_future/ahn4_exp007)")
    ap.add_argument("--export-viewer", action="store_true",
                    help="Also export viewer_buildings.json for Session 3")
    ap.add_argument("--building-iou", default=None,
                    help="Exp007 AHN4 Building IoU (override auto-detect)")

    args = ap.parse_args()

    pred_ply = Path(args.pred)
    if not pred_ply.exists():
        print(f"ERROR: Prediction PLY not found: {pred_ply}", file=sys.stderr)
        print("  Hint: Has Session 1 completed Exp007 inference?", file=sys.stderr)
        print("  Expected: ml/results/ahn4_exp007/visuals/exp007_test_prediction.ply", file=sys.stderr)
        return 1

    building_iou = args.building_iou or get_exp007_iou()
    ai_model = MODEL_DESCRIPTION_TEMPLATE.format(building_iou=building_iou)

    print(f"SIH 26011 — Exp007 Pipeline Adapter")
    print(f"  pred PLY      : {pred_ply}")
    print(f"  output        : {args.out}")
    print(f"  Exp007 IoU    : {building_iou}")
    print(f"  ai_model      : {ai_model}")
    print()

    out_dir = Path(args.out)

    summary = run_pipeline(
        pred_ply           = pred_ply,
        out_dir            = out_dir,
        dataset            = AHN4_PARAMS["dataset"],
        scene_id           = AHN4_PARAMS["scene_id"],
        ai_model           = ai_model,
        crs                = AHN4_PARAMS["crs"],
        floor_height       = AHN4_PARAMS["floor_height"],
        max_buildings      = AHN4_PARAMS["max_buildings"],
        bag_path           = AHN4_PARAMS["bag_path"],
        apply_plausibility = AHN4_PARAMS["apply_plausibility"],
        apply_height_split = AHN4_PARAMS["apply_height_split"],
    )

    if "error" in summary:
        print(f"Pipeline error: {summary['error']}", file=sys.stderr)
        return 1

    if args.export_viewer:
        buildings_json = out_dir / "buildings.json"
        viewer_json    = out_dir / "viewer_buildings.json"
        if buildings_json.exists():
            try:
                iou_val = float(building_iou) if building_iou not in ("TBD", None) else 0.0
                export_viewer_json(
                    input_path   = buildings_json,
                    output_path  = viewer_json,
                    dataset      = "AHN4",
                    model_id     = "exp007",
                    building_iou = iou_val,
                    crs          = AHN4_PARAMS["crs"],
                )
                print(f"  viewer_buildings.json -> {viewer_json}")
            except Exception as e:
                print(f"  Viewer export failed (non-fatal): {e}")

    print(f"\nExp007 reconstruction complete.")
    print(f"  Buildings: {summary.get('n_buildings_reconstructed', '?')}")
    print(f"  Output:    {out_dir}")
    print(f"\nTo update Session 3 viewer, load:")
    print(f"  ml/results/reconstruction_v3_future/ahn4_exp007/viewer_buildings.json")
    print(f"  (same schema as other datasets — no viewer code changes needed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

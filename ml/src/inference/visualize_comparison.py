"""Phase 7: Visual comparison — AHN4 raw vs Exp006 vs Exp007 vs ground truth.

Creates color-coded PLY files from the first N chunks of the test strip,
showing the per-point class assignments for each model side by side.

Usage:
    python ml/src/inference/visualize_comparison.py \\
        --checkpoint-exp007 ml/results/ahn4_exp007/best_model.pth \\
        --checkpoint-exp006 ml/results/experiment_006/best_model.pth \\
        --manifest          ml/results/ahn4_exp007/data/manifest.json \\
        --out               ml/results/ahn4_exp007/visuals \\
        --n-chunks          20
"""
from __future__ import annotations
import argparse, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ml.src.models.pointnet2 import PointNet2Seg
from ml.src.training.finetune_ahn4 import AHN4SplitDataset, normalize_chunk
from ml.src.utils import select_device

NUM_CLASSES = 6
CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]
CLASS_COLORS = {
    0: (120, 120, 120),
    1: (220, 110,  55),
    2: ( 45, 160,  80),
    3: ( 40,  40,  40),
    4: ( 70, 130, 200),
    5: (200, 200,  40),
}
GT_COLOR_OVERRIDE = {  # use distinct colors for GT visualization
    0: (80,  80,  80),
    1: (255, 80,  0),
}


def write_ply_rgb_class(path: Path, xyz: np.ndarray, rgb_src: np.ndarray,
                        class_colors: np.ndarray):
    """Write PLY with BOTH source RGB and class-color RGB."""
    n = len(xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"comment CRS: EPSG:28992\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    fmt = struct.Struct("<fffBBB")
    buf = bytearray()
    for i in range(n):
        buf += fmt.pack(float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]),
                        int(class_colors[i,0]), int(class_colors[i,1]), int(class_colors[i,2]))
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(bytes(buf))
    print(f"  Saved: {path.name}  ({n:,} pts)")


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    input_dim = 6 if cfg["dataset"].get("input_mode", "xyz") == "xyzrgb" else 3
    m = PointNet2Seg(
        num_classes=cfg["model"]["num_classes"],
        sa1=cfg["model"].get("sa1"),
        sa2=cfg["model"].get("sa2"),
        input_dim=input_dim,
    )
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()
    return m, input_dim


def predict_chunk(model, xyz_norm: np.ndarray, rgb_norm: np.ndarray,
                  input_dim: int, device: torch.device):
    if input_dim == 6:
        feat = np.concatenate([xyz_norm, rgb_norm], axis=1)
    else:
        feat = xyz_norm
    inp = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        pred   = logits.argmax(dim=-1).cpu().numpy()[0]
    return pred


def class_to_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((len(labels), 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        mask = labels == c
        if mask.any():
            rgb[mask] = color
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-exp007", default="ml/results/ahn4_exp007/best_model.pth")
    ap.add_argument("--checkpoint-exp006", default="ml/results/experiment_006/best_model.pth")
    ap.add_argument("--manifest",           default="ml/results/ahn4_exp007/data/manifest.json")
    ap.add_argument("--out",                default="ml/results/ahn4_exp007/visuals")
    ap.add_argument("--n-chunks",           type=int, default=20)
    ap.add_argument("--split",              default="test")
    args = ap.parse_args()

    ckpt007  = ROOT / args.checkpoint_exp007
    ckpt006  = ROOT / args.checkpoint_exp006
    manifest = ROOT / args.manifest
    out_dir  = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device("auto")
    print(f"\nPhase 7: Visual Comparison (first {args.n_chunks} chunks of {args.split} strip)")
    print(f"  Output: {out_dir}")

    m006, dim006 = load_model(ckpt006, device)
    print(f"  Exp006 loaded (input_dim={dim006})")

    m007 = None
    if ckpt007.exists():
        m007, dim007 = load_model(ckpt007, device)
        print(f"  Exp007 loaded (input_dim={dim007})")
    else:
        print(f"  WARNING: Exp007 checkpoint not found: {ckpt007}")
        dim007 = 6

    ds = AHN4SplitDataset(manifest, args.split)
    n_vis = min(args.n_chunks, len(ds))

    # Collect all points from first n_vis chunks
    all_xyz, all_rgb_uint8 = [], []
    all_gt, all_pred006, all_pred007 = [], [], []

    for i in range(n_vis):
        item = ds.npz_paths[i]
        d = np.load(item)
        xyz     = d["xyz"].astype(np.float32)     # world coords RD New
        rgb_u8  = d["rgb"].astype(np.uint8)       # uint8 0-255
        labels  = d["labels"].astype(np.int64)

        xyz_norm = normalize_chunk(xyz)
        rgb_norm = rgb_u8.astype(np.float32) / 255.0

        p006 = predict_chunk(m006, xyz_norm, rgb_norm, dim006, device)
        p007 = predict_chunk(m007, xyz_norm, rgb_norm, dim007, device) if m007 else labels

        all_xyz.append(xyz)
        all_rgb_uint8.append(rgb_u8)
        all_gt.append(labels)
        all_pred006.append(p006)
        all_pred007.append(p007)

        if (i+1) % 5 == 0:
            print(f"  {i+1}/{n_vis} chunks processed")

    xyz_all     = np.concatenate(all_xyz)
    rgb_all     = np.concatenate(all_rgb_uint8)
    gt_all      = np.concatenate(all_gt)
    pred006_all = np.concatenate(all_pred006)
    pred007_all = np.concatenate(all_pred007)
    n_total     = len(xyz_all)

    print(f"\n  Writing {n_total:,} points to {out_dir.name}/...")

    # 1. Raw AHN4 aerial RGB
    write_ply_rgb_class(out_dir / "raw_aerial_rgb.ply",
                        xyz_all, rgb_all, rgb_all)

    # 2. Ground truth
    write_ply_rgb_class(out_dir / "ground_truth.ply",
                        xyz_all, rgb_all, class_to_rgb(gt_all))

    # 3. Exp006 predictions
    write_ply_rgb_class(out_dir / "exp006_prediction.ply",
                        xyz_all, rgb_all, class_to_rgb(pred006_all))

    # 4. Exp007 predictions
    if m007 is not None:
        write_ply_rgb_class(out_dir / "exp007_prediction.ply",
                            xyz_all, rgb_all, class_to_rgb(pred007_all))

    # Print summary stats
    n_pts = len(gt_all)
    print(f"\n  Ground truth distribution:")
    for c, name in enumerate(CLASS_NAMES):
        cnt = int((gt_all == c).sum())
        if cnt: print(f"    {name:12}: {cnt:>7,} ({100*cnt/n_pts:.1f}%)")

    print(f"\n  Exp006 prediction distribution:")
    for c, name in enumerate(CLASS_NAMES):
        cnt = int((pred006_all == c).sum())
        if cnt: print(f"    {name:12}: {cnt:>7,} ({100*cnt/n_pts:.1f}%)")

    if m007 is not None:
        print(f"\n  Exp007 prediction distribution:")
        for c, name in enumerate(CLASS_NAMES):
            cnt = int((pred007_all == c).sum())
            if cnt: print(f"    {name:12}: {cnt:>7,} ({100*cnt/n_pts:.1f}%)")

    print(f"\n  Done. Open PLY files in CloudCompare or MeshLab.")
    print(f"  Files in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

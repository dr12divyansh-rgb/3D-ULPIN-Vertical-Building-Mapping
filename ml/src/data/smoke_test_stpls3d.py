"""Quick smoke test for STPLS3DChunkDataset — run after prepare_stpls3d.py --smoke-test."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ml.src.data.dataset import STPLS3DChunkDataset  # noqa: E402

SMOKE_DATA = ROOT / "ml" / "data" / "stpls3d_smoke"


def check(condition, msg):
    if not condition:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  OK  : {msg}")


def main():
    print("\n" + "=" * 60)
    print("STPLS3DChunkDataset smoke test")
    print("=" * 60)

    if not SMOKE_DATA.exists():
        print(f"ERROR: smoke data not found at {SMOKE_DATA}")
        print("Run: python ml/src/data/prepare_stpls3d.py --source-zip ... --output ml/data/stpls3d_smoke --smoke-test")
        sys.exit(1)

    # XYZ mode
    print("\n[1] Loading test split (XYZ mode)...")
    ds = STPLS3DChunkDataset(SMOKE_DATA, "test", num_points=4096, input_mode="xyz")
    print(ds.summary())

    check(len(ds) > 0, f"dataset has {len(ds)} chunks (expected > 0)")

    s = ds[0]
    check(tuple(s["xyz"].shape) == (4096, 3), f"xyz shape {s['xyz'].shape} == (4096,3)")
    check(tuple(s["xyz_orig"].shape) == (4096, 3), f"xyz_orig shape {s['xyz_orig'].shape} == (4096,3)")
    check(tuple(s["labels"].shape) == (4096,), f"labels shape {s['labels'].shape} == (4096,)")
    check(isinstance(s["scene"], str), "scene is str")

    import torch
    check(not torch.any(torch.isnan(s["xyz"])), "no NaN in xyz")
    check(not torch.any(torch.isinf(s["xyz"])), "no Inf in xyz")
    labels_arr = s["labels"]
    check((labels_arr >= 0).all() and (labels_arr < 6).all(),
          f"labels in [0,6): unique={labels_arr.unique().tolist()}")

    # XYZRGB mode
    print("\n[2] Loading test split (XYZRGB mode)...")
    ds_rgb = STPLS3DChunkDataset(SMOKE_DATA, "test", num_points=4096, input_mode="xyzrgb")
    s_rgb = ds_rgb[0]
    check(tuple(s_rgb["xyz"].shape) == (4096, 6), f"xyzrgb shape {s_rgb['xyz'].shape} == (4096,6)")

    # DataLoader round-trip
    print("\n[3] DataLoader batch test...")
    import torch
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    check(tuple(batch["xyz"].shape) == (2, 4096, 3), f"batch xyz {batch['xyz'].shape} == (2,4096,3)")
    check(tuple(batch["labels"].shape) == (2, 4096), f"batch labels {batch['labels'].shape} == (2,4096)")
    print(f"  batch scenes: {batch['scene']}")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

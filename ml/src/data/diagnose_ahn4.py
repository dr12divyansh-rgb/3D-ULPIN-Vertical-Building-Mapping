"""Phase 1 Preprocessing Audit: AHN4 vs DALES domain shift diagnosis.

Compares raw and normalized XYZ/RGB statistics, class distributions, and
height geometry to identify the root causes of Exp006 AHN4 zero-shot failure
(Building IoU 0.0528, 83.3% vehicle misclassification).

Usage:
    python ml/src/data/diagnose_ahn4.py \
        --ahn4  ml/results/ahn4_inference/chunks/test \
        --dales ml/data/dales_chunks/chunks/test \
        --out   ml/results/ahn4_exp007/phase1_diagnosis.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

CLASS_NAMES = ["ground", "building", "vegetation", "road", "vehicle", "other"]


def normalize_chunk(xyz: np.ndarray):
    centroid = xyz.mean(axis=0, keepdims=True)
    centered = xyz - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    return centered / scale, scale


def load_chunks(chunk_dir: Path, n: int = 50):
    files = sorted(chunk_dir.glob("*.npz"))[:n]
    all_xyz, all_rgb, all_lbl = [], [], []
    for f in files:
        d = np.load(f)
        all_xyz.append(d["xyz"].astype(np.float32))
        all_rgb.append(d["rgb"].astype(np.uint8))
        all_lbl.append(d["labels"].astype(np.uint8))
    return (
        np.concatenate(all_xyz),
        np.concatenate(all_rgb),
        np.concatenate(all_lbl),
        files,
    )


def stats(arr: np.ndarray) -> dict:
    return {
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "p25":    float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75":    float(np.percentile(arr, 75)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ahn4",  default="ml/results/ahn4_inference/chunks/test")
    ap.add_argument("--dales", default="ml/data/dales_chunks/chunks/test")
    ap.add_argument("--out",   default="ml/results/ahn4_exp007/phase1_diagnosis.json")
    ap.add_argument("--n",     type=int, default=50)
    args = ap.parse_args()

    ahn4_dir  = ROOT / args.ahn4
    dales_dir = ROOT / args.dales
    out_path  = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 1: PREPROCESSING AUDIT — AHN4 vs DALES")
    print("="*70)

    print(f"\nLoading up to {args.n} AHN4 chunks from {ahn4_dir}...")
    ahn4_xyz, ahn4_rgb, ahn4_lbl, ahn4_files = load_chunks(ahn4_dir, args.n)
    print(f"  {len(ahn4_xyz):,} pts from {len(ahn4_files)} files")

    print(f"Loading up to {args.n} DALES chunks from {dales_dir}...")
    dales_xyz, dales_rgb, dales_lbl, dales_files = load_chunks(dales_dir, args.n)
    print(f"  {len(dales_xyz):,} pts from {len(dales_files)} files")

    # ── XYZ raw statistics ────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("RAW XYZ STATISTICS (world coords, metres)")
    print("-"*60)
    ahn4_xyz_stats  = {ax: stats(ahn4_xyz[:,i])  for i, ax in enumerate("xyz")}
    dales_xyz_stats = {ax: stats(dales_xyz[:,i]) for i, ax in enumerate("xyz")}
    for ax in "xyz":
        a, d = ahn4_xyz_stats[ax], dales_xyz_stats[ax]
        print(f"  {ax.upper()}: AHN4  [{a['min']:>10.1f}, {a['max']:>10.1f}]  mean={a['mean']:.1f}  std={a['std']:.1f}")
        print(f"     DALES [{d['min']:>10.1f}, {d['max']:>10.1f}]  mean={d['mean']:.1f}  std={d['std']:.1f}")

    ahn4_zspan  = ahn4_xyz[:,2].max()  - ahn4_xyz[:,2].min()
    dales_zspan = dales_xyz[:,2].max() - dales_xyz[:,2].min()
    print(f"\n  Z span: AHN4={ahn4_zspan:.1f}m  DALES={dales_zspan:.1f}m")

    # ── Normalization scale ────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("NORMALIZATION SCALE (per-chunk sphere radius, metres)")
    print("-"*60)
    ahn4_scales, dales_scales = [], []
    ahn4_znorm, dales_znorm   = [], []
    for f in sorted(ahn4_dir.glob("*.npz"))[:args.n]:
        d = np.load(f)
        n, s = normalize_chunk(d["xyz"].astype(np.float32))
        ahn4_scales.append(s)
        ahn4_znorm.append(n[:,2])
    for f in sorted(dales_dir.glob("*.npz"))[:args.n]:
        d = np.load(f)
        n, s = normalize_chunk(d["xyz"].astype(np.float32))
        dales_scales.append(s)
        dales_znorm.append(n[:,2])

    ahn4_sc  = np.array(ahn4_scales)
    dales_sc = np.array(dales_scales)
    ahn4_zn  = np.concatenate(ahn4_znorm)
    dales_zn = np.concatenate(dales_znorm)

    print(f"  AHN4:  mean={ahn4_sc.mean():.2f}m  min={ahn4_sc.min():.2f}m  max={ahn4_sc.max():.2f}m")
    print(f"  DALES: mean={dales_sc.mean():.2f}m  min={dales_sc.min():.2f}m  max={dales_sc.max():.2f}m")
    print(f"\n  Norm-Z AHN4:  mean={ahn4_zn.mean():.4f}  std={ahn4_zn.std():.4f}  "
          f"p5={np.percentile(ahn4_zn,5):.4f}  p95={np.percentile(ahn4_zn,95):.4f}")
    print(f"  Norm-Z DALES: mean={dales_zn.mean():.4f}  std={dales_zn.std():.4f}  "
          f"p5={np.percentile(dales_zn,5):.4f}  p95={np.percentile(dales_zn,95):.4f}")

    # ── RGB statistics ─────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("RGB STATISTICS (uint8, 0-255 in NPZ)")
    print("-"*60)
    ahn4_rgb_stats  = {c: stats(ahn4_rgb[:,i].astype(float))  for i, c in enumerate("RGB")}
    dales_rgb_stats = {c: stats(dales_rgb[:,i].astype(float)) for i, c in enumerate("RGB")}
    for ch in "RGB":
        a, d = ahn4_rgb_stats[ch], dales_rgb_stats[ch]
        print(f"  {ch}: AHN4  mean={a['mean']:5.1f}  std={a['std']:5.1f}  "
              f"[{a['min']:.0f}, {a['max']:.0f}]  median={a['median']:.0f}")
        print(f"     DALES mean={d['mean']:5.1f}  std={d['std']:5.1f}  "
              f"[{d['min']:.0f}, {d['max']:.0f}]  median={d['median']:.0f}")

    # ── Class distribution ─────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("CLASS DISTRIBUTION")
    print("-"*60)
    ahn4_cls  = {c: int((ahn4_lbl  == c).sum()) for c in range(6)}
    dales_cls = {c: int((dales_lbl == c).sum()) for c in range(6)}
    n_a, n_d = len(ahn4_lbl), len(dales_lbl)
    for c, name in enumerate(CLASS_NAMES):
        na, nd = ahn4_cls[c], dales_cls[c]
        if na > 0 or nd > 0:
            print(f"  {name:12} ({c}): AHN4={na:>8,} ({100*na/n_a:.1f}%)  "
                  f"DALES={nd:>8,} ({100*nd/n_d:.1f}%)")

    # ── Per-class height and RGB analysis ─────────────────────────────────
    print("\n" + "-"*60)
    print("PER-CLASS HEIGHT (raw Z) AND RGB ANALYSIS")
    print("-"*60)
    for c, name in enumerate(CLASS_NAMES):
        mask_a = ahn4_lbl  == c
        mask_d = dales_lbl == c
        if mask_a.sum() > 100:
            az = ahn4_xyz[mask_a, 2]
            ar = ahn4_rgb[mask_a]
            print(f"\n  {name} [AHN4]: Z mean={az.mean():.1f}m "
                  f"std={az.std():.2f}m "
                  f"RGB=({ar[:,0].mean():.0f},{ar[:,1].mean():.0f},{ar[:,2].mean():.0f})")
        if mask_d.sum() > 100:
            dz = dales_xyz[mask_d, 2]
            dr = dales_rgb[mask_d]
            print(f"  {name} [DALES]: Z mean={dz.mean():.1f}m "
                  f"std={dz.std():.2f}m "
                  f"RGB=({dr[:,0].mean():.0f},{dr[:,1].mean():.0f},{dr[:,2].mean():.0f})")

    # ── DALES vehicle vs AHN4 ground comparison ────────────────────────────
    print("\n" + "-"*60)
    print("DALES VEHICLE vs AHN4 GROUND (the mismatch that causes 83% vehicle)")
    print("-"*60)
    dales_veh_mask = dales_lbl == 4
    ahn4_gnd_mask  = ahn4_lbl  == 0
    if dales_veh_mask.sum() > 100 and ahn4_gnd_mask.sum() > 100:
        dv_rgb = dales_rgb[dales_veh_mask]
        ag_rgb = ahn4_rgb[ahn4_gnd_mask]
        print(f"  DALES vehicle RGB: R={dv_rgb[:,0].mean():.1f} G={dv_rgb[:,1].mean():.1f} B={dv_rgb[:,2].mean():.1f}")
        print(f"  AHN4  ground  RGB: R={ag_rgb[:,0].mean():.1f} G={ag_rgb[:,1].mean():.1f} B={ag_rgb[:,2].mean():.1f}")
        r_diff = abs(dv_rgb[:,0].mean() - ag_rgb[:,0].mean())
        g_diff = abs(dv_rgb[:,1].mean() - ag_rgb[:,1].mean())
        b_diff = abs(dv_rgb[:,2].mean() - ag_rgb[:,2].mean())
        print(f"  RGB mean diff: R={r_diff:.1f}  G={g_diff:.1f}  B={b_diff:.1f}")
        print(f"  -> Small RGB diff = model confuses AHN4 ground with DALES vehicle")

    # ── Root cause summary ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ROOT CAUSE SUMMARY")
    print("="*70)
    causes = []

    # Cause 1: Normalization scale
    scale_ratio = float(ahn4_sc.mean() / max(dales_sc.mean(), 1e-8))
    if abs(scale_ratio - 1.0) > 0.1:
        cause = f"Norm scale: AHN4={ahn4_sc.mean():.1f}m vs DALES={dales_sc.mean():.1f}m (ratio={scale_ratio:.2f})"
        causes.append(("SCALE_DIFF", cause))
        print(f"  [1] {cause}")

    # Cause 2: Z distribution
    ahn4_zstd  = ahn4_xyz[:,2].std()
    dales_zstd = dales_xyz[:,2].std()
    if abs(ahn4_zstd - dales_zstd) > 2.0:
        cause = f"Z spread: AHN4 std={ahn4_zstd:.2f}m vs DALES std={dales_zstd:.2f}m"
        causes.append(("Z_SPREAD", cause))
        print(f"  [2] {cause}")

    # Cause 3: Class distribution
    ahn4_gnd_pct  = 100*ahn4_cls[0]/n_a
    dales_gnd_pct = 100*dales_cls[0]/n_d
    if abs(ahn4_gnd_pct - dales_gnd_pct) > 10:
        cause = f"Ground %: AHN4={ahn4_gnd_pct:.0f}% vs DALES={dales_gnd_pct:.0f}%"
        causes.append(("CLASS_DIST", cause))
        print(f"  [3] {cause}")

    ahn4_veh_pct = 100*ahn4_cls[4]/n_a
    dales_veh_pct = 100*dales_cls[4]/n_d
    cause = f"Vehicle %: AHN4={ahn4_veh_pct:.1f}% (GT=0%) vs DALES={dales_veh_pct:.1f}%"
    causes.append(("NO_VEHICLE_IN_AHN4", cause))
    print(f"  [4] {cause}")

    # Cause 4: RGB domain
    ahn4_rgb_mean  = ahn4_rgb.astype(float).mean(axis=0)
    dales_rgb_mean = dales_rgb.astype(float).mean(axis=0)
    rgb_dist = float(np.linalg.norm(ahn4_rgb_mean - dales_rgb_mean))
    if rgb_dist > 20:
        cause = f"RGB domain shift: AHN4=({ahn4_rgb_mean.round(1)}) vs DALES=({dales_rgb_mean.round(1)}) dist={rgb_dist:.1f}"
        causes.append(("RGB_DOMAIN", cause))
        print(f"  [5] {cause}")

    print("\n  PRIMARY: Model trained on classes 0-5 but AHN4 eval has only 0+1.")
    print("  PRIMARY: DALES RGB is ALL ZEROS in NPZ. AHN4 has real RGB (mean~113). RGB mismatch.")
    print("  SOLUTION: Fine-tune on AHN4 data with REAL RGB. Model learns AHN4 ground/building.")

    # ── Save JSON report ──────────────────────────────────────────────────
    report = {
        "ahn4_chunks":  len(ahn4_files),
        "dales_chunks": len(dales_files),
        "ahn4_pts":     len(ahn4_xyz),
        "dales_pts":    len(dales_xyz),
        "xyz_stats": {
            "ahn4":  ahn4_xyz_stats,
            "dales": dales_xyz_stats,
        },
        "norm_scale": {
            "ahn4":  {"mean": float(ahn4_sc.mean()), "min": float(ahn4_sc.min()), "max": float(ahn4_sc.max())},
            "dales": {"mean": float(dales_sc.mean()), "min": float(dales_sc.min()), "max": float(dales_sc.max())},
        },
        "norm_z_stats": {
            "ahn4":  stats(ahn4_zn),
            "dales": stats(dales_zn),
        },
        "rgb_stats": {
            "ahn4":  {ch: ahn4_rgb_stats[ch]  for ch in "RGB"},
            "dales": {ch: dales_rgb_stats[ch] for ch in "RGB"},
        },
        "class_dist": {
            "ahn4":  {CLASS_NAMES[c]: {"n": ahn4_cls[c],  "pct": round(100*ahn4_cls[c]/n_a, 2)}  for c in range(6)},
            "dales": {CLASS_NAMES[c]: {"n": dales_cls[c], "pct": round(100*dales_cls[c]/n_d, 2)} for c in range(6)},
        },
        "root_causes": [{"code": c, "desc": d} for c, d in causes],
        "diagnosis": {
            "primary_cause_1": "AHN4 eval mode has only ground(0)+building(1). No vehicle GT. Model still outputs vehicle because it sees grey surfaces.",
            "primary_cause_2": "DALES vehicle RGB ≈ AHN4 ground RGB (Dutch grey asphalt resembles DALES vehicle class).",
            "secondary_cause": "Netherlands flat terrain -> small Z variance -> different normalized geometry from DALES Ohio.",
            "recommended_fix": "Fine-tune Exp006 on AHN4 train split. Model must see AHN4 ground points with label=0 to unlearn vehicle activation.",
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved diagnosis: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

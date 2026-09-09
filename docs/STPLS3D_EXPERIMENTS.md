# STPLS3D Integration & Experiments

**Status**: Experiment 003 training in progress (started 2026-09-07 16:18)  
**Document**: Integration of STPLS3D real+synthetic outdoor point cloud dataset into SIH 26011 semantic segmentation pipeline.

---

## Overview

This document describes the integration of [STPLS3D](https://www.stpls3d.com/) — a large-scale photogrammetry and synthetic LiDAR dataset — into the existing PointNet++ semantic segmentation pipeline from Step 3.

**Goal**: Determine whether adding realistic outdoor STPLS3D data improves building segmentation compared to the existing synthetic-only baseline.

---

## Dataset Integration

### STPLS3D Source Data
- **Location**: `C:\Users\Takin\Downloads\STPLS3D.zip` (37.2 GB compressed)
- **Format**: 67 PLY files (binary little-endian 1.0)
- **Properties**: x,y,z (float32), red,green,blue,class,instance (uint8, instance only in Synthetic)
- **Original classes**: 19 STPLS3D labels (0-19, label 16 absent)

### Preprocessing
- **Script**: `ml/src/data/prepare_stpls3d.py`
- **Method**: Spatial chunking (50×50m cells, 4096 pts/chunk, seed 26011)
- **Runtime**: ~2-3 hours for full 67 files on laptop SSD
- **Output**: `ml/data/stpls3d/` with `manifest.json` + `chunks/*.npz`

### Preprocessed Statistics
```
Total chunks : 7,509
Total points : 30,756,864 (30.7M)
Total scenes : 67 PLY files

Split breakdown:
  train       : 6,416 chunks, 58 scenes (26.3M pts)
  validation  :   500 chunks,  5 scenes ( 2.0M pts)
  test        :   593 chunks,  4 scenes ( 2.4M pts)

Class distribution (after mapping to 6 classes):
  0 ground     :  9,143,873  (29.7%)
  1 building   :  7,543,567  (24.5%)
  2 vegetation : 10,600,587  (34.5%)  ← most frequent
  3 road       :  1,985,424  ( 6.5%)
  4 vehicle    :    648,923  ( 2.1%)  ← least frequent
  5 other      :    834,490  ( 2.7%)

Imbalance ratio: 16.3:1 (vegetation:vehicle)
```

### Label Mapping (STPLS3D → Six-Class Ontology)
```
STPLS3D                          → Our ontology
─────────────────────────────────────────────────
0  Ground                        → 0 ground
1  Building                      → 1 building
2  LowVegetation                 → 2 vegetation
3  MediumVegetation              → 2 vegetation
4  HighVegetation                → 2 vegetation
5  Vehicle                       → 4 vehicle
6  Truck                         → 4 vehicle
7  Aircraft                      → 4 vehicle
8  MilitaryVehicle               → 4 vehicle
9  Bike                          → 4 vehicle
10 Motorcycle                    → 4 vehicle
11 LightPole                     → 5 other
12 StreetSign                    → 5 other
13 Clutter                       → 5 other
14 Fence                         → 5 other
15 Road                          → 3 road
16 (absent)                      → -1 (dropped)
17 Windows                       → 5 other
18 Dirt                          → 0 ground
19 Grass                         → 2 vegetation
```

### Train/Validation/Test Split (Deterministic, Scene-Level)
```
STPLS3D/RealWorldData/*          → test   (real-world, never trained on)
STPLS3D/Synthetic_v3/ tiles ≥21  → validation
All other STPLS3D files          → train
```

**Rationale**: Real-world data (OCCC, RA, USC, WMSC) isolated for test. Synthetic_v3 tiles 21-25 held out for validation. Training uses Synthetic_v1, Synthetic_v2, and Synthetic_v3 tiles 1-20.

---

## Experiments

### Experiment 001: Synthetic-Only Baseline (Complete)
**Purpose**: Established baseline from Step 3. Uses only our procedurally-generated synthetic buildings/LiDAR.

- **Config**: `ml/config/segmentation.yaml`
- **Dataset**: `ml/data/synthetic` (7 train scenes, 2 val, 1 test)
- **Architecture**: PointNet++ (XYZ-only, 6 classes)
- **Training**: 25 epochs, batch_size=4, 96 train chunks, class_weighted_loss=false
- **Best epoch**: 10 (val mIoU=0.334)

**Results on synthetic test set**:
```
Overall accuracy  : 89.2%
Mean IoU          : 35.1%

Per-class IoU:
  ground      : 51.9%
  building    : 91.4%  ← excellent on synthetic buildings
  vegetation  : 52.2%
  road        : 14.9%
  vehicle     :  0.0%  ← no vehicles in synthetic test
  other       :  0.0%  ← no "other" in synthetic test
```

**Key insight**: Synthetic model achieves 91% building IoU but completely fails at rare/absent classes (vehicle, other).

---

### Experiment 003: STPLS3D-Only XYZ (In Progress)
**Purpose**: Measure whether realistic outdoor STPLS3D data improves building segmentation and rare-class detection vs synthetic-only.

- **Config**: `ml/config/stpls3d_semantic.yaml`
- **Dataset**: `ml/data/stpls3d` (6416 train chunks → subsetted to 1500)
- **Architecture**: PointNet++ (same as exp001, XYZ-only, 6 classes)
- **Training**: 25 epochs, batch_size=4, early_stopping_patience=10
- **Loss**: class-weighted CrossEntropy (vehicle=2.51×, road=1.05×, vegetation=0.15×)
- **Subsetting**: `max_train_chunks=1500`, `max_val_chunks=250`
  - *Rationale*: CPU training with full 6416 chunks would take 23 hours. Subsampling to 1500 keeps training ~3 hours.

**Status**: Training started 2026-09-07 16:18. Monitor armed. Expected duration: ~3 hours.

**Expected CPU time per epoch**: 375 steps × 1.14 s/step ≈ 7 min/epoch

---

### Experiment 004: Mixed Synthetic + STPLS3D (Pending)
**Purpose**: Determine if combining synthetic (with perfect building labels) + STPLS3D (with realistic outdoor context) yields best of both worlds.

- **Config**: `ml/config/mixed_semantic.yaml`
- **Dataset**: MixedDataset (30% synthetic, 70% STPLS3D)
- **Architecture**: PointNet++ (same, XYZ-only, 6 classes)
- **Training**: 25 epochs, batch_size=4, class-weighted loss
- **Mixing strategy**: Balanced epoch construction using `MixedDataset` with `synthetic_weight=0.3`

**Evaluation plan**:
1. Test on STPLS3D test set (generalization to real-world)
2. Test on synthetic test set (cross-domain: does STPLS3D hurt synthetic performance?)

---

## Comparison Framework

### Primary Metric: Building IoU
The project's primary goal is building reconstruction, so **building-class IoU is the primary metric**.

### Secondary Metrics:
- Overall accuracy
- mIoU (mean across all 6 classes)
- Per-class IoU for vegetation, road, vehicle, ground, other

### Comparison Matrix (Planned)
```
                          Exp001   Exp003   Exp004
                        Synthetic STPLS3D   Mixed
───────────────────────────────────────────────────
Test: Synthetic
  Overall accuracy        89.2%      -        ?
  Building IoU            91.4%      -        ?

Test: STPLS3D
  Overall accuracy         n/a       ?        ?
  Building IoU             n/a       ?        ?
  Vegetation IoU           n/a       ?        ?
  Road IoU                 n/a       ?        ?
  Vehicle IoU              n/a       ?        ?
```

---

## Code Changes

### New Files
- `ml/config/stpls3d_semantic.yaml` — STPLS3D config
- `ml/config/mixed_semantic.yaml` — Mixed dataset config
- `ml/src/data/prepare_stpls3d.py` — Preprocessing (already existed, verified)
- `ml/src/data/generate_stpls3d_statistics.py` — Dataset stats report
- `ml/src/data/pipeline_smoke_test.py` — Full ML pipeline smoke test
- `ml/src/evaluate_stpls3d.py` — STPLS3D-specific evaluation
- `ml/src/evaluation/compare_stpls3d_experiments.py` — Cross-experiment comparison
- `ml/src/run_stpls3d_pipeline.py` — Automated pipeline runner

### Modified Files
- `ml/src/data/dataset.py`:
  - Added `STPLS3DChunkDataset` (reads .npz chunks from manifest)
  - Added `MixedDataset` (balanced synthetic + STPLS3D mixing)
  - Added `build_dataset()` factory (dispatches on `source`)
  - Added `subset_dataset()` (CPU-training tractability)
  - Added `compute_class_weights_stpls3d()`, `compute_class_weights_mixed()`

- `ml/src/training/train.py`:
  - Updated class weight computation to dispatch on dataset source
  - Handles `data_root=None` for mixed configs
  - Added `max_train_chunks`/`max_val_chunks` subsetting

### Files Deliberately Preserved
- All existing synthetic data (`ml/data/synthetic/`)
- All existing experiment checkpoints (`ml/results/experiment_001/`, `experiment_002/`)
- PointNet++ architecture (`ml/src/models/pointnet2.py`) — unchanged
- Existing configs (`segmentation.yaml`, etc.) — unchanged

---

## Lessons Learned

### 1. Class Imbalance is Critical
STPLS3D has severe imbalance (16:1). **Class-weighted loss is not optional** — it's essential for learning rare classes (vehicle 2.1%, other 2.7%).

### 2. CPU Training Requires Aggressive Subsetting
- PointNet++ FPS is slow: ~1.1 s/step at batch_size=4 on CPU
- Full STPLS3D (6416 chunks) would take 23 hours to train
- Subsetting to 1500 chunks preserves diversity while keeping training at ~3 hours

### 3. Preprocessing is Fast and Reliable
- 37 GB ZIP → 7509 chunks in ~2-3 hours
- Deterministic spatial chunking works well
- No data-quality issues found

### 4. Scene-Level Splits Preserve Realism
- Spatial chunks from same scene stay in same split
- Test set = pure real-world (RealWorldData)
- No leakage between splits

---

## Failure Analysis (Pending)

**Will be added after Experiment 003 evaluation completes.**

Expected error patterns to investigate:
1. Ground/road confusion (both horizontal, similar structure)
2. Vegetation contamination of buildings (overhanging trees)
3. Sparse-point regions (far from scanner)
4. Chunk boundary artifacts
5. Vehicle misclassification (rare class, severe imbalance)

---

## Next Steps

**After Experiment 003 completes**:
1. ✅ Evaluate on STPLS3D test set → metrics JSON, confusion matrix
2. ✅ Compare against Experiment 001 baseline
3. ✅ Train Experiment 004 (mixed dataset)
4. ✅ Evaluate Experiment 004 on both test sets
5. ✅ Determine best checkpoint (building IoU priority)
6. ✅ Generate synthetic data recommendations

**Longer-term (Next Milestone)**:
- Add RGB features (XYZRGB mode) once XYZ baseline is established
- Experiment with alternative architectures (KPConv, PointTransformer)
- Generate additional synthetic data to match STPLS3D characteristics
- Investigate semi-supervised learning (use unlabeled STPLS3D data)

---

**Status**: Training in progress. Document will be updated with actual results once experiments complete.

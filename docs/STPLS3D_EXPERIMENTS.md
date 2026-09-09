# STPLS3D Integration & Experiments

**Status**: All experiments complete (2026-09-09)  
**Summary**: 5 experiments run. Best real-world model: Exp005 (XYZRGB, building IoU 65.9%). Best synthetic model: Exp001 (XYZ, building IoU 91.4%).

Full results: `ml/results/comparison_all_experiments.json`

---

## Dataset Integration

### STPLS3D Source Data
- **Location**: `C:\Users\Takin\Downloads\STPLS3D.zip` (37.2 GB compressed)
- **Format**: 67 PLY files (binary little-endian 1.0)
- **Properties**: x,y,z (float32), red,green,blue,class,instance (uint8)
- **Original classes**: 19 STPLS3D labels (0-19, label 16 absent)
- **Preprocessed**: `ml/src/data/prepare_stpls3d.py` → `C:/ULPIN_DATA/stpls3d/`

### Preprocessed Statistics
```
Total chunks : 7,509  (4096 pts each, 50m cells, seed 26011)
Total points : 30.7M

Split breakdown:
  train      : 6,416 chunks, 58 scenes (26.3M pts)
  validation :   500 chunks,  5 scenes ( 2.0M pts)
  test        :  593 chunks,  4 scenes ( 2.4M pts)  ← RealWorldData only

Class distribution:
  0 ground     :  9,143,873  (29.7%)
  1 building   :  7,543,567  (24.5%)
  2 vegetation : 10,600,587  (34.5%)  ← most frequent
  3 road       :  1,985,424  ( 6.5%)
  4 vehicle    :    648,923  ( 2.1%)  ← least frequent
  5 other      :    834,490  ( 2.7%)
```

### Label Mapping (STPLS3D → 6-Class)
```
STPLS3D 0  Ground            → 0 ground
STPLS3D 1  Building          → 1 building
STPLS3D 2/3/4 Vegetation     → 2 vegetation
STPLS3D 5-10  Vehicles       → 4 vehicle
STPLS3D 11-14,17 Other       → 5 other
STPLS3D 15 Road              → 3 road
STPLS3D 18 Dirt              → 0 ground
STPLS3D 19 Grass             → 2 vegetation
STPLS3D 16 (absent)          → dropped
```

---

## Complete Results Matrix

### On STPLS3D Test Set (RealWorldData — never seen during training)

| | Exp001 | Exp003 | Exp004 | **Exp005** |
|---|---|---|---|---|
| Dataset | Synthetic | STPLS3D | Mixed | **STPLS3D** |
| Input | XYZ | XYZ | XYZ | **XYZRGB** |
| Best epoch | 10/25 | 5/25 | 10/25 | **17/25** |
| Overall acc | n/a | 47.6% | 43.6% | **65.9%** |
| mIoU | n/a | 25.4% | 23.2% | **39.0%** |
| Building IoU | n/a | 52.0% | 41.1% | **65.9%** |
| Vegetation IoU | n/a | 25.6% | 20.1% | **61.2%** |
| Road IoU | n/a | 36.2% | 34.5% | **51.5%** |
| Vehicle IoU | n/a | 14.6% | 16.5% | **23.9%** |
| Ground IoU | n/a | 16.5% | 18.2% | 22.0% |
| Other IoU | n/a | 7.5% | 8.7% | 9.2% |

### On Synthetic Test Set (scene_00005, 18 buildings, 570k pts)

| | **Exp001** | Exp005 (zero-pad RGB) |
|---|---|---|
| Overall acc | **89.2%** | 22.3% |
| mIoU | **35.1%** | 9.1% |
| Building IoU | **91.4%** | 22.8% |
| Building Precision | 96.1% | **99.5%** |
| Building Recall | **94.9%** | 22.9% |

**Cross-domain finding**: Exp005 (trained on real STPLS3D with RGB) collapses on synthetic XYZ-only data — precision stays high (99.5%) but recall drops to 22.9%. The model learned to use RGB as a strong signal; without it, it is extremely conservative.

---

## Key Findings

### 1. RGB Features Are the Biggest Lever
Exp005 (XYZRGB) vs Exp003 (XYZ), same dataset: building IoU jumps from 52% → 65.9%, mIoU from 25.4% → 39.0%. Adding RGB is more impactful than mixing datasets or training longer at XYZ.

### 2. Mixed Training Hurts
Exp004 (mixed 30%syn+70%STPLS3D) underperforms Exp003 (STPLS3D-only) on real test data. The synthetic data distribution is too different — adding it confuses the model. Don't mix unless the synthetic is made more realistic.

### 3. Exp003/004 Early-Stopped — Exp005 Did Not
Exp003 stopped at epoch 5, Exp004 at epoch 10 (OOM on higher batch sizes). Exp005 solved this with `batch_size=2` + `gradient_accumulation_steps=2` + local SSD path, allowing 25 full epochs — explaining much of its improvement.

### 4. Domain Gap Is Real
No cross-domain generalization: synthetic-trained model fails on real data; real-trained model fails on synthetic. Use the right checkpoint for each domain.

### 5. Vehicle Detection Requires Real Data
Exp001 (synthetic-only): vehicle IoU = 0%. All STPLS3D experiments achieve 14-24% vehicle IoU. Real-world diversity is essential for rare classes.

---

## Experiment Details

### Experiment 001 — Synthetic XYZ Baseline
- **Config**: `ml/config/segmentation.yaml`
- **Training**: 25 epochs, batch_size=4, 96 train chunks, unweighted loss
- **Best epoch**: 10 (val mIoU=0.334)
- **Use case**: ULPIN synthetic demo, best building IoU on synthetic data

### Experiment 002 — Synthetic Weighted Loss
- **Config**: `ml/config/segmentation_weighted.yaml`
- **Training**: 25 epochs, class-weighted loss, same data as 001
- **Result**: Similar to exp001, no major improvement from weighting on balanced synthetic data

### Experiment 003 — STPLS3D XYZ
- **Config**: `ml/config/stpls3d_semantic.yaml`
- **Training**: 5 epochs (early stop due to OOM at higher batch), 1500 chunks/epoch
- **Lesson**: Needed lower batch size and local SSD path for stability

### Experiment 004 — Mixed Synthetic + STPLS3D
- **Config**: `ml/config/mixed_semantic.yaml`
- **Training**: 10 epochs, 30% synthetic + 70% STPLS3D per step
- **Lesson**: Mixed training does NOT improve over STPLS3D-only on real test data

### Experiment 005 — STPLS3D XYZRGB (Best Real-World)
- **Config**: `ml/config/stpls3d_xyzrgb.yaml`
- **Training**: 25 full epochs, batch_size=2, gradient_accumulation=2, data from C:/ULPIN_DATA/stpls3d
- **Best epoch**: 17 (val mIoU peaked, then slight overfit)
- **Total training time**: ~55 hours (CPU-only, Core i7 240H)
- **Checkpoint**: `ml/results/experiment_005/best_model.pth`

---

## Failure Analysis

### Ground/Road Confusion
Exp005 ground IoU: 22.0% despite high recall (75.3%). Ground is confused with road — both horizontal, similar point density. The model high-recalls ground but precision is only 23.7%.

### Vegetation Bleed
Buildings partially covered by trees get misclassified as vegetation. Exp005 reduces this significantly vs exp003 (building recall 73.8% vs 63.5%).

### Vehicle Recall
All experiments have high vehicle recall (60-76%) but low precision (16-26%). Vehicles are over-predicted. Class-weighted loss helps recall but causes false positives on similar small structures.

### Chunk Boundary Artifacts
50m spatial chunks occasionally split large buildings. The reconstruction step handles this with the flood-fill instance segmentation in `reconstruct_from_prediction.py`.

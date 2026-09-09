# SIH 26011 — 3D ULPIN Project

## What this project is

**Goal**: 3D building reconstruction + semantic segmentation for ULPIN (Unique Land Parcel Identification Number) assignment, producing PLATEAU-equivalent CityGML output.

**Competition**: SIH (Smart India Hackathon) problem 26011.

**Status** (as of 2026-09-09): All 5 training experiments complete. CityGML export, ULPIN assignment, and real-world inference pipeline implemented and tested.

---

## Architecture

```
Stage 1: Synthetic Data Generation
  synthetic_city/ (Blender procedural buildings + synthetic LiDAR)
  → ml/data/synthetic/  (7 train / 2 val / 1 test scenes, 164 buildings)

Stage 2: External Real-World Training Data
  STPLS3D (37 GB ZIP) → ml/src/data/prepare_stpls3d.py
  → C:/ULPIN_DATA/stpls3d/  (7509 NPZ chunks, LOCAL SSD, not OneDrive)
  DALES (real aerial LiDAR) → ml/data/dales2/

Stage 3: Semantic Segmentation (PointNet++)
  ml/src/models/pointnet2.py  (264K params, 2SA+2FP, CPU-only)
  ml/src/training/train.py    (Adam, gradient accumulation, early stop)
  5 experiments: exp001 (synthetic XYZ) → exp005 (STPLS3D XYZRGB)

Stage 4: LOD2 Reconstruction
  ml/src/reconstruction/reconstruct.py              (needs scene metadata)
  ml/src/reconstruction/reconstruct_from_prediction.py  (any PLY, no metadata)

Stage 5: CityGML Export (PLATEAU-equivalent)
  ml/src/export/ulpin.py    (ULPIN-26011-S{scene}-B{building} IDs)
  ml/src/export/citygml.py  (CityGML 2.0, stdlib only, no new deps)
  ml/src/pipeline_plateau.py (end-to-end CLI)

Stage 6: Viewer
  viewer/ (Three.js r160, offline, PLATEAU dark theme, ULPIN hero panel)
```

---

## Key Commands

### Run the full PLATEAU pipeline (synthetic scene)
```bash
python ml/src/pipeline_plateau.py \
    --scene scene_00005 \
    --checkpoint ml/results/experiment_001/best_model.pth \
    --out ml/results/plateau_demo
```

### Run inference on any raw PLY (real-world)
```bash
python ml/src/inference/predict_raw.py \
    --checkpoint ml/results/experiment_005/best_model.pth \
    --input path/to/cloud.ply \
    --output ml/results/prediction_output.ply
```

### Reconstruct buildings from any prediction (no metadata)
```bash
python ml/src/reconstruction/reconstruct_from_prediction.py \
    --pred ml/results/dales2_500k_prediction.ply \
    --out  ml/results/reconstructed_buildings/
```

### Run viewer
```bash
python viewer/build_index.py
python -m http.server 8000
# Open http://localhost:8000/viewer/
```

### Evaluate a checkpoint
```bash
# On synthetic test:
python ml/src/evaluate.py --checkpoint ml/results/experiment_001/best_model.pth

# On STPLS3D test:
python ml/src/evaluate_stpls3d.py \
    --checkpoint ml/results/experiment_005/best_model.pth \
    --data C:/ULPIN_DATA/stpls3d
```

### Train a new experiment
```bash
python ml/src/training/train.py --config ml/config/stpls3d_xyzrgb.yaml
# Resume: add --resume ml/results/experiment_005
```

---

## Experiment Results Summary

| Exp | Dataset | Input | Best Epoch | Building IoU | mIoU | Use for |
|---|---|---|---|---|---|---|
| 001 | Synthetic | XYZ | 10/25 | **91.4%** | 35.1% | Synthetic/ULPIN demo |
| 002 | Synthetic | XYZ | 25/25 | ~91% | ~35% | Weighted loss variant |
| 003 | STPLS3D | XYZ | 5/25 | 52.0% | 25.4% | Real-world XYZ baseline |
| 004 | Mixed | XYZ | 10/25 | 41.1% | 23.2% | Worse than 003 — don't use |
| 005 | STPLS3D | XYZRGB | 17/25 | **65.9%** | 39.0% | **Best for real-world PLY** |

**Cross-domain finding**: Exp005 on synthetic test (zero-padded RGB) achieves only 22.8% building IoU — confirms domain gap between STPLS3D and synthetic data.

**Best checkpoint for real-world**: `ml/results/experiment_005/best_model.pth`
**Best checkpoint for synthetic/ULPIN**: `ml/results/experiment_001/best_model.pth`

---

## Data Locations

| Data | Path | Size | In git? |
|---|---|---|---|
| Synthetic scenes | `ml/data/synthetic/` | ~100 MB | No (generated) |
| STPLS3D chunks | `C:/ULPIN_DATA/stpls3d/` | 414 MB | No (local SSD) |
| STPLS3D chunks (mirror) | `ml/data/stpls3d/` | 414 MB | No |
| DALES real LiDAR | `ml/data/dales2/test/` | ~400 MB | No |
| Experiment checkpoints | `ml/results/experiment_00*/` | ~200 MB | Partially |
| PLATEAU demo output | `ml/results/plateau_demo/` | ~5 MB | No |
| DALES prediction | `ml/results/dales2_500k_prediction.ply` | ~15 MB | No |

---

## Immutable Constraints

**DO NOT change these — changing them breaks trained checkpoints and all existing data:**
- `CLASS_IDS` in `synthetic_city/core/labels.py` — 6 classes, fixed order
- `seed: 26011` in all configs — tied to SIH problem number
- `vendor/three.module.js` — Three.js r160, locked; any upgrade needs full viewer QA
- Any `best_model.pth` checkpoint — training is complete

---

## NOT Implemented (future work)

- Real ULPIN assignment from cadastral shapefile (synthetic ULPINs used for demo)
- Georeferencing (GML uses local XYZ, not real EPSG coordinates)
- LOD3/LOD4 (doors, windows, interior)
- Predicted LOD2 meshes in the viewer (shows ground truth LOD2 currently)
- CesiumJS upgrade (currently Three.js offline viewer)

---

## Hardware (this machine)

- CPU: Intel Core 7 240H (10 cores, 16 logical)
- RAM: 16.9 GB total
- GPU: Intel integrated — NO CUDA
- All training is CPU-only (PyTorch 2.14.0+cpu)
- venv: `.venv\Scripts\python.exe`

# SIH 26011 — 3D ULPIN Building Reconstruction Pipeline

End-to-end pipeline for **semantic segmentation + 3D building reconstruction + ULPIN assignment** for the Smart India Hackathon problem 26011. Produces PLATEAU-equivalent CityGML output from aerial LiDAR point clouds.

## Project Status

| Stage | Description | Status |
|---|---|---|
| Synthetic data | Procedural Blender buildings + synthetic LiDAR | ✅ Complete |
| ML segmentation | PointNet++ semantic segmentation (6 classes) | ✅ 6 experiments |
| Real-world data | STPLS3D + DALES-2 + AHN4 aerial LiDAR | ✅ Integrated |
| Reconstruction | Instance extraction → LOD2 mesh → building schema | ✅ Pipeline v2/v3 |
| CityGML export | PLATEAU-equivalent LOD1/LOD2 GML output | ✅ Complete |
| Viewer | Three.js offline viewer with ULPIN panel | ✅ Complete |

## Experiment Results

| Exp | Dataset | Input | Building IoU | mIoU | Checkpoint |
|---|---|---|---|---|---|
| 001 | Synthetic | XYZ | **91.4%** | 35.1% | `ml/results/experiment_001/best_model.pth` |
| 002 | Synthetic | XYZ | ~91% | ~35% | `ml/results/experiment_002/best_model.pth` |
| 003 | STPLS3D | XYZ | 52.0% | 25.4% | `ml/results/experiment_003/best_model.pth` |
| 004 | Mixed | XYZ | 41.1% | 23.2% | `ml/results/experiment_004/best_model.pth` |
| 005 | STPLS3D | XYZRGB | **65.9%** | 39.0% | `ml/results/experiment_005/best_model.pth` |
| 006 | STPLS3D→DALES | XYZRGB | **76.3%** | — | `ml/results/experiment_006/best_model.pth` |

**Best checkpoint for real-world aerial LiDAR**: `ml/results/experiment_006/best_model.pth` (fine-tuned on DALES-2)

**Best checkpoint for synthetic/ULPIN demo**: `ml/results/experiment_001/best_model.pth`

Cross-domain note: Exp005 on synthetic data achieves only 22.8% building IoU — domain gap between STPLS3D and synthetic is significant.

## Quick Start

```bash
# 1. Create a Python 3.12 venv and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate synthetic dataset (requires Blender 4.2 LTS)
python synthetic_city/scripts/generate_dataset.py

# 3. Run full PLATEAU pipeline on a synthetic scene
python ml/src/pipeline_plateau.py \
    --scene scene_00005 \
    --checkpoint ml/results/experiment_001/best_model.pth \
    --out ml/results/plateau_demo

# 4. Run inference on any raw aerial LiDAR PLY (real-world)
python ml/src/inference/predict_raw.py \
    --checkpoint ml/results/experiment_006/best_model.pth \
    --input path/to/cloud.ply \
    --output ml/results/prediction_output.ply

# 5. Run reconstruction pipeline v2 on a prediction
python ml/src/reconstruction/pipeline_v2.py \
    --pred ml/results/prediction_output.ply \
    --out ml/results/reconstruction_output/

# 6. Start the offline viewer
python viewer/build_index.py
python -m http.server 8000
# Open http://localhost:8000/viewer/
```

## Pipeline Architecture

```
Stage 1: Synthetic Data Generation
  synthetic_city/ (Blender procedural buildings + synthetic LiDAR)
  → ml/data/synthetic/  (7 train / 2 val / 1 test scenes, 164 buildings)

Stage 2: Real-World Training Data
  STPLS3D (37 GB ZIP) → ml/src/data/prepare_stpls3d.py → C:/ULPIN_DATA/stpls3d/
  DALES-2 aerial LiDAR → ml/src/data/prepare_dales.py → ml/data/dales_chunks/
  AHN4 aerial LiDAR   → ml/src/data/prepare_ahn4.py

Stage 3: Semantic Segmentation (PointNet++)
  ml/src/models/pointnet2.py  (264K params, 2SA+2FP, CPU-only)
  ml/src/training/train.py    (Adam, gradient accumulation, early stop)
  6-class labels: ground / vegetation / building / water / vehicle / unknown

Stage 4: Building Reconstruction (Pipeline v2/v3)
  ml/src/reconstruction/prediction_contract.py  — PLY loader, PredictionRecord
  ml/src/reconstruction/instance_extraction.py  — Grid BFS instance segmentation
  ml/src/reconstruction/floor_decomposition.py  — Height-based floor estimation
  ml/src/reconstruction/roof_analysis.py        — RANSAC roof plane fitting
  ml/src/reconstruction/building_schema.py      — Unified building output schema
  ml/src/reconstruction/pipeline_v2.py          — End-to-end CLI (primary entry point)
  ml/src/reconstruction/pipeline_v3.py          — Improved instance methods

Stage 5: CityGML Export (PLATEAU-equivalent)
  ml/src/export/ulpin.py      — ULPIN-26011-S{scene}-B{building} ID generation
  ml/src/export/citygml.py    — CityGML 2.0 (stdlib only, no extra deps)
  ml/src/pipeline_plateau.py  — End-to-end CLI with CityGML output

Stage 6: Viewer
  viewer/ulpin_demo.html      — Three.js r160, offline, PLATEAU dark theme
  viewer/build_index.py       — Regenerates viewer scene index
```

## Key Commands

### Training
```bash
# Train from scratch
python ml/src/training/train.py --config ml/config/stpls3d_xyzrgb.yaml

# Fine-tune on DALES (Exp006-style)
python ml/src/training/train.py \
    --config ml/config/dales_xyzrgb.yaml \
    --init-checkpoint ml/results/experiment_005/best_model.pth

# Resume interrupted training
python ml/src/training/train.py --config ml/config/stpls3d_xyzrgb.yaml \
    --resume ml/results/experiment_005
```

### Evaluation
```bash
# Synthetic test split
python ml/src/evaluate.py --checkpoint ml/results/experiment_001/best_model.pth

# STPLS3D test set
python ml/src/evaluate_stpls3d.py \
    --checkpoint ml/results/experiment_005/best_model.pth \
    --data C:/ULPIN_DATA/stpls3d

# DALES test set
python ml/src/evaluate_dales.py \
    --checkpoint ml/results/experiment_006/best_model.pth
```

### Real-World Inference (AHN4)
```bash
# Prepare AHN4 LAZ tiles
python ml/src/data/prepare_ahn4.py --input path/to/ahn4.laz --out ml/data/ahn4/

# Run inference
python ml/src/inference/infer_ahn4.py \
    --checkpoint ml/results/experiment_006/best_model.pth \
    --data ml/data/ahn4/

# Full tile inference (large scenes)
python ml/src/inference/infer_ahn4_fulltile.py \
    --checkpoint ml/results/experiment_006/best_model.pth \
    --input path/to/tile.laz
```

### Reconstruction
```bash
# Pipeline v2 (primary)
python ml/src/reconstruction/pipeline_v2.py \
    --pred ml/results/prediction_output.ply \
    --out ml/results/reconstruction_output/

# Reconstruct from any prediction PLY (no scene metadata needed)
python ml/src/reconstruction/reconstruct_from_prediction.py \
    --pred ml/results/dales2_500k_prediction.ply \
    --out ml/results/reconstructed_buildings/
```

## Data Locations

| Data | Path | Size | Committed? |
|---|---|---|---|
| Synthetic scenes | `ml/data/synthetic/` | ~100 MB | No (regenerate) |
| STPLS3D chunks | `C:/ULPIN_DATA/stpls3d/` | 414 MB | No (local SSD) |
| DALES-2 LiDAR | `ml/data/dales2/` | ~400 MB | No |
| DALES chunks | `ml/data/dales_chunks/` | ~50 MB | No (gitignored) |
| AHN4 tiles | `ml/data/ahn4/` | varies | No |
| Experiment checkpoints | `ml/results/experiment_00*/best_model.pth` | ~200 MB | Yes |

## Building ID Format

```
{DATASET}_{scene_id}_{sequential:04d}    # e.g. STPLS3D_RA_0001
{building_id}/F{floor_number:02d}        # e.g. STPLS3D_RA_0001/F01
```

These are **prototype IDs only** — not official ULPIN assignments. Official ULPIN requires cadastral shapefile georeferencing (not yet implemented).

## Requirements

- Python 3.12 + PyTorch 2.x (CPU wheel — no CUDA needed)
- Blender 4.2 LTS for synthetic data generation: `winget install BlenderFoundation.Blender.LTS.4.2`
- `numpy`, `PyYAML`, `open3d>=0.19`, `matplotlib`
- All training/inference runs CPU-only (Intel Core 7 240H, 16 GB RAM)

```bash
pip install -r requirements.txt
```

## Docs

- [`docs/SYNTHETIC_DATA.md`](docs/SYNTHETIC_DATA.md) — synthetic dataset pipeline
- [`docs/LOD_PIPELINE.md`](docs/LOD_PIPELINE.md) — LOD1/LOD2 architecture
- [`docs/ML_SEGMENTATION.md`](docs/ML_SEGMENTATION.md) — PointNet++ training
- [`docs/TEST_VIEWER.md`](docs/TEST_VIEWER.md) — viewer setup
- [`CLAUDE.md`](CLAUDE.md) — project architecture and constraints reference

## Constraints (DO NOT change)

- `CLASS_IDS` in `synthetic_city/core/labels.py` — 6 classes, fixed order, breaking change
- `seed: 26011` in all configs — tied to SIH problem number
- `vendor/three.module.js` — Three.js r160 locked; upgrade needs full viewer QA
- `ml/results/experiment_00*/best_model.pth` — training complete, do not retrain

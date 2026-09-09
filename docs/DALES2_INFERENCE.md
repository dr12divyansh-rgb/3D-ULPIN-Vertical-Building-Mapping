# DALES2 Real-World Inference Test

**Status**: Complete  
**Date**: 2026-09-09  
**Purpose**: Demonstrate the full SIH 26011 pipeline on real aerial LiDAR data (no ground truth labels).

---

## Dataset: DALES (Digital Aerial LiDAR Edmonton Survey)

- **Source**: HuggingFace — `fraunhoferisi/DALES` (free, public)
- **Location**: `ml/data/dales2/test/` (4 tiles downloaded)
- **Format**: LAZ (compressed LAS), converted to PLY for inference
- **Coverage**: City of Edmonton, Alberta, Canada — aerial LiDAR survey
- **Classes** (original, not used for inference): ground, vegetation, cars, trucks, power lines, fences, poles, buildings

### Files Downloaded
```
ml/data/dales2/test/
├── 5080_54470.laz        (60 MB compressed)
├── 5100_54440.laz        (59 MB compressed)
├── 5100_54440.ply        (162 MB — full tile, ~4M points)
├── 5100_54440_500k.ply   (7.2 MB — downsampled to 500k points)
├── 5120_54445.laz        (75 MB compressed)
└── 5135_54430.laz        (65 MB compressed)
```

**Tile used for inference**: `5100_54440_500k.ply` — 500,000 points, downsampled from full tile.

---

## Inference: predict_raw.py on Real LiDAR

```bash
python ml/src/inference/predict_raw.py \
    --checkpoint ml/results/experiment_005/best_model.pth \
    --input ml/data/dales2/test/5100_54440_500k.ply \
    --output ml/results/dales2_500k_prediction.ply
```

- **Checkpoint**: Experiment 005 (STPLS3D XYZRGB, epoch 17, building IoU 65.9%)
- **Input**: XYZ-only PLY (DALES has no RGB in point cloud)
- **RGB handling**: Model auto-detected as 6-channel; RGB zero-padded automatically
- **Output**: `ml/results/dales2_500k_prediction.ply` — 500k points with predicted_class and confidence

### Output format
```
PLY fields: x, y, z (float32) | red, green, blue (uint8, class colors) | predicted_class (uint8) | confidence (float32)
Color legend: grey=ground | orange=building | green=vegetation | dark=road | blue=vehicle | yellow=other
```

---

## Reconstruction: reconstruct_from_prediction.py on Real Buildings

```bash
python ml/src/reconstruction/reconstruct_from_prediction.py \
    --pred ml/results/dales2_500k_prediction.ply \
    --out  ml/results/reconstructed_buildings_dales/
```

- **Script**: `ml/src/reconstruction/reconstruct_from_prediction.py` (metadata-free approach)
- **Algorithm**: No scene metadata — pure geometry from point predictions
  1. Filter building-class points (predicted_class == 1)
  2. 2D grid flood-fill (cell_size=1.5m) → separate building instances
  3. Convex hull (Graham scan) → 2D footprint polygon
  4. Z-percentile analysis → ground level, eave level, roof peak
  5. Roof-type detection (Z std of roof points) → flat or gable
  6. Watertight mesh: walls + flat slab or gable roof planes

### Output
```
ml/results/reconstructed_buildings_dales/
├── building_0001.obj  ...  building_00NN.obj   (individual watertight meshes)
├── scene.obj                                   (all buildings combined)
├── scene.mtl                                   (material file with class colors)
└── summary.json                                (per-building stats)
```

---

## Key Differences vs Synthetic Pipeline

| Aspect | Synthetic (reconstruct.py) | Real DALES (reconstruct_from_prediction.py) |
|---|---|---|
| Requires scene metadata | Yes (building.json) | **No** |
| Building instance source | building_id field in PLY | 2D flood-fill on predictions |
| Footprint source | Convex hull or metadata rectangle | Convex hull of base points |
| Roof type source | Metadata (ground truth) | Z std heuristic |
| Works on any PLY | No | **Yes** |
| Requires ground truth | Yes | **No** |

The `reconstruct_from_prediction.py` approach is the production-ready path for real-world data.

---

## Significance for SIH 26011

This demonstrates the **complete production pipeline** on real LiDAR data:

```
Real LAZ tile
  → convert to PLY
  → predict_raw.py (Exp005 checkpoint)
  → reconstruct_from_prediction.py
  → per-building OBJ meshes
  → (+ citygml.py export → CityGML)
```

No synthetic data, no ground truth labels, no scene metadata required anywhere in the chain.

---

## Visualization

Open in CloudCompare or MeshLab:
- `dales2_500k_prediction.ply` — colored point cloud by semantic class
- `reconstructed_buildings_dales/scene.obj` — all reconstructed building meshes
- Individual `building_000N.obj` files — per-building meshes

Or extend the viewer:
```bash
python viewer/build_index.py
python -m http.server 8000
# Load dales2_500k_prediction.ply via the "AI Prediction" toggle
```

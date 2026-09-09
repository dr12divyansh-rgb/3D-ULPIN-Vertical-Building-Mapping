# SIH 26011 — ULPIN Synthetic LiDAR Dataset Pipeline

Reproducible **synthetic building + LiDAR dataset generation** for the ULPIN
problem statement (SIH 26011). Implemented so far:

- **Step 1** — procedural building generation with Blender, synthetic LiDAR
  point clouds with ground-truth semantic labels, scene-level dataset splitting.
- **Step 2** — PLATEAU-inspired **LOD1 → LOD2** architecture: every building
  now produces a simple LOD1 block, a ground-truth LOD2 (detailed, semantic
  surfaces), and the synthetic LiDAR, plus LOD comparison metadata, geometry
  validation, and visual comparison / evaluation tools.
- **Step 3** — CPU-compatible PointNet++ semantic segmentation baseline,
  evaluated on the separate unseen-building test split with checkpoints,
  confusion matrix, per-building metrics, and prediction visualization.
- **Test viewer** — temporary local HTML/Three.js viewer (`viewer/`) to
  inspect LOD1, ground-truth LOD2, LiDAR, AI predictions, and metadata for any
  existing building.

ML training (PointNet++ / RandLA-Net) and the other pipeline stages are **not**
implemented yet.

## Quick start

```bash
# 1. Create a Python 3.12 venv and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate the dataset (Blender 4.2 LTS must be installed)
python synthetic_city/scripts/generate_dataset.py

# 3. Validate it (LOD1/LOD2 geometry + point cloud + split)
python synthetic_city/scripts/validate_dataset.py

# 4. Visual comparison of LOD1 / LOD2 / LiDAR for one building
python synthetic_city/scripts/compare_building.py --scene ml/data/synthetic/train/scene_00002

# 5. Evaluation sanity check (ground truth vs itself → ~0 error)
python synthetic_city/scripts/evaluate_lod2.py --scene ml/data/synthetic/train/scene_00002

# 6. Train PointNet++ (CPU works; checkpoint under ml/results/)
python ml/src/training/train.py --config ml/config/segmentation.yaml

# 7. Evaluate the untouched test split
python ml/src/evaluate.py --checkpoint ml/results/experiment_001/best_model.pth

# 8. Start the temporary HTML test viewer (from the project root)
python viewer/build_index.py          # regenerate the viewer index (after new data)
python -m http.server 8000            # then open http://localhost:8000/viewer/
```

Docs: [`docs/SYNTHETIC_DATA.md`](docs/SYNTHETIC_DATA.md) (Step 1),
[`docs/LOD_PIPELINE.md`](docs/LOD_PIPELINE.md) (Step 2),
[`docs/ML_SEGMENTATION.md`](docs/ML_SEGMENTATION.md) (Step 3), and
[`docs/TEST_VIEWER.md`](docs/TEST_VIEWER.md) (viewer).

## Requirements

- Blender 4.2 LTS (`winget install BlenderFoundation.Blender.LTS.4.2`)
- Python 3.12 (recommended) + `numpy`, `PyYAML`, `open3d`
- CPU PyTorch + `matplotlib` for Step 3 (`torch` should be installed from the
  CPU wheel index; see `docs/ML_SEGMENTATION.md`)
- CPU only — no GPU/CUDA needed

## Output (per scene)

```
scene_XXXXX/
├── lod1/            building_XXXXXX.obj + metadata.json
├── lidar/           pointcloud.ply + metadata.json
├── ground_truth/
│   ├── lod2/        building_XXXXXX.obj + metadata.json (incl. lod_comparison)
│   └── building.json
└── scene.json
```

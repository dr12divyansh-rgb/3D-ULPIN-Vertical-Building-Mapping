# STPLS3D Integration Milestone — COMPLETE

**Status**: All tasks complete as of 2026-09-09
**Branch**: `ml-segmentation`
**Last commit**: `a1de81c`

---

## Summary

All 5 experiments complete. PLATEAU-equivalent CityGML export implemented.
Real-world inference tested on DALES aerial LiDAR. Full results in `ml/results/comparison_all_experiments.json`.

---

## Final Results

| Experiment | Dataset | Building IoU | mIoU | Status |
|---|---|---|---|---|
| Exp001 | Synthetic XYZ | **91.4%** | 35.1% | Best synthetic |
| Exp002 | Synthetic XYZ (weighted) | ~91% | ~35% | Complete |
| Exp003 | STPLS3D XYZ | 52.0% | 25.4% | Complete |
| Exp004 | Mixed XYZ | 41.1% | 23.2% | Complete |
| Exp005 | STPLS3D XYZRGB | **65.9%** | 39.0% | Best real-world |

---

## Completed Tasks

- [x] STPLS3D dataset downloaded and preprocessed (7509 chunks)
- [x] Experiment 003 trained and evaluated (STPLS3D XYZ)
- [x] Experiment 004 trained and evaluated (Mixed)
- [x] Experiment 005 trained and evaluated (STPLS3D XYZRGB — 25 full epochs)
- [x] Cross-experiment comparison (`comparison_all_experiments.json`)
- [x] Exp005 cross-domain eval on synthetic test (domain gap confirmed: 22.8% bldg IoU)
- [x] Real-world inference on DALES aerial LiDAR (`predict_raw.py`)
- [x] Metadata-free building reconstruction (`reconstruct_from_prediction.py`)
- [x] CityGML 2.0 export with ULPIN identifiers (`ml/src/export/`)
- [x] End-to-end PLATEAU pipeline CLI (`ml/src/pipeline_plateau.py`)
- [x] Viewer upgraded: PLATEAU dark theme + ULPIN hero panel
- [x] All new code committed to `ml-segmentation` branch
- [x] CLAUDE.md created with full project context

---

## Remaining (Not Blocking SIH Demo)

- [ ] Predicted LOD2 meshes in viewer (shows GT LOD2 currently)
- [ ] Real ULPIN from cadastral shapefile (synthetic ULPINs used for demo)
- [ ] Georeferencing (real EPSG coordinates in GML)

---

## Key Artifacts

```
ml/results/experiment_001/best_model.pth    <- Use for synthetic/ULPIN demo
ml/results/experiment_005/best_model.pth    <- Use for real-world PLY inference
ml/results/comparison_all_experiments.json  <- Full 4-experiment results table
ml/results/plateau_demo/city_model.gml      <- CityGML 2.0 output (18 buildings)
ml/results/plateau_demo/ulpin_registry.csv  <- ULPIN + attribute table
ml/results/dales2_500k_prediction.ply       <- Real-world inference output (DALES)
ml/results/reconstructed_buildings_dales/   <- Real-world LOD2 reconstruction
CLAUDE.md                                   <- Full project context for AI sessions
docs/DALES2_INFERENCE.md                    <- Real-world test documentation
docs/STPLS3D_EXPERIMENTS.md                 <- Full experiment results and analysis
```

# Step 3: Semantic Segmentation

Step 3 adds the first ML component: per-point semantic classification of the
existing synthetic LiDAR. It does **not** generate geometry or predicted LOD2.

```text
LiDAR XYZ
   ↓
PointNet++
   ↓
per-point semantic labels
   ↓
building points
   ↓
Step 4: geometry reconstruction (not implemented)
```

## Dataset inspected

The loader uses the Step 2 layout directly:

```text
<scene>/lidar/pointcloud.ply
<scene>/ground_truth/building.json
<scene>/scene.json
```

The PLY contains `x`, `y`, `z`, `class_id`, and `building_id`, plus RGB,
synthetic `intensity`, and synthetic `return_number`. The model uses **XYZ
only**. Intensity and return number are deliberately excluded because Step 2
documents them as synthetic placeholders rather than meaningful sensor
features.

Canonical labels are imported from `synthetic_city.core.labels` and verified
against the YAML configuration:

| ID | Class |
|---:|---|
| 0 | ground |
| 1 | building |
| 2 | vegetation |
| 3 | road |
| 4 | vehicle |
| 5 | other |

The existing split contains 7 training scenes / 116 buildings, 2 validation
scenes / 30 buildings, and 1 test scene / 18 buildings. The existing dataset
validator confirms no building ID crosses split boundaries.

## Floor-marker decision

Step 2 floor-marker slabs are retained in the input. They are not a separate
label or feature: they are ordinary `building` XYZ points in the PLY and are
indistinguishable from other observed facade points. They represent realistic
facade floor-line/cornice detail rather than a synthetic class shortcut. They
are not used as LOD2 geometry and the model never receives a marker flag.

## Model input and preprocessing

The first baseline uses only normalized XYZ:

1. Load original XYZ, labels, and building IDs.
2. Sample deterministic fixed-size chunks (`4096` points by default) from
   training/validation scenes.
3. Center each chunk at its centroid and scale it to a unit sphere.
4. Keep original coordinates unchanged in `xyz_orig` and in saved prediction
   PLY files.

Training chunks are generated only from training scenes. Validation chunks are
generated only from validation scenes. Test inference processes the complete
order.

## PointNet++ baseline

Implemented in `ml/src/models/pointnet2.py`:

- two set-abstraction layers with farthest-point sampling and ball query;
- two feature-propagation layers;
- per-point logits with shape `N x 6`.

The model is CPU-compatible and exposes:

```python
logits = model(points)  # (B, N, 3) -> (B, N, 6)
```

The generic `PointCloudSegmenter` interface remains available in
`synthetic_city/core/segmentation.py` for future PointNet++/RandLA-Net
interchangeability.

## Training and evaluation

Install CPU PyTorch explicitly:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Train:

```bash
python ml/src/training/train.py --config ml/config/segmentation.yaml
```

Evaluate the untouched test split:

```bash
python ml/src/evaluate.py \
  --checkpoint ml/results/experiment_001/best_model.pth
```

Predict one scene:

```bash
python ml/src/inference/predict.py \
  --checkpoint ml/results/experiment_001/best_model.pth \
  --scene ml/data/synthetic/test/scene_00005
```

Visualize ground truth, predictions, or errors:

```bash
python ml/src/inference/visualize_prediction.py \
  --scene ml/data/synthetic/test/scene_00005 \
  --view errors --save test_errors.png
```

Each prediction PLY preserves original XYZ and stores `predicted_class` and
`confidence`.

## Metrics

The evaluation reports:

- overall accuracy;
- per-class precision, recall, F1, and IoU;
- mean IoU;
- building precision, recall, F1, and IoU;
- confusion matrix;
- per-building building IoU/precision/recall/F1;
- a simple known-style vs novel `(footprint_type, roof_type)` comparison.

The per-building metric counts building points correctly classified as
building and non-building points predicted as building inside that building's
world footprint as false positives. This is intended to identify difficult
building instances, not to replace a future object-level evaluation.

## Experiment 001 result

Training used seed `26011`, CPU PyTorch, XYZ only, 25 epochs, and selected the
best checkpoint by validation mean IoU at epoch 10.

Test evaluation was performed on the separate 18-building test scene:

```text
overall accuracy : 0.8918
mean IoU         : 0.3506
building precision: 0.9608
building recall   : 0.9494
building F1       : 0.9550
building IoU      : 0.9140
```

Per-class IoU:

```text
ground      0.5187
building    0.9140
vegetation  0.5219
road        0.1489
vehicle     0.0000
other       0.0000
```

The low vehicle/other scores are expected from the current class imbalance:
the test scene contains only 3,263 vehicle points and 20 `other` points out of
570,141 total points. The baseline intentionally uses plain cross-entropy so
building performance remains interpretable; optional inverse-frequency loss
is supported by the config but disabled for this reported experiment.

The test scene's footprint/roof combinations were all already represented in
the training metadata, although the buildings themselves and their full
parameters were unseen. Therefore this is a valid unseen-building test, but
not a strong held-out-combination experiment. A future dataset revision should
reserve genuinely novel footprint/roof combinations if that analysis is
required.

## Results layout

```text
ml/results/experiment_001/
├── best_model.pth
├── config.yaml
├── history.json
├── summary.json
├── test_metrics.json
├── per_building_metrics.json
├── confusion_matrix.csv
├── confusion_matrix.png
└── predictions/scene_00005/prediction.ply
```

## Limitations

- This is a baseline, not a tuned benchmark model.
- The model sees synthetic geometry and synthetic LiDAR; real-survey domain
  gap remains unaddressed.
- Class imbalance is severe, especially for `vehicle` and `other`.
- The current test split contains unseen building instances but no novel
  footprint/roof combination relative to training.
- No LOD2 reconstruction, government data, ULPIN integration, or real-data
  processing is included in Step 3.

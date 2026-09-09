# Synthetic LiDAR Dataset Pipeline

This document describes the **synthetic building + LiDAR dataset generation
pipeline** for SIH 26011 (ULPIN). This is Step 1 only: it produces a
reproducible, labelled point-cloud dataset for later segmentation-model
training. No ML training, government-data processing, ULPIN integration, or
frontend is implemented here.

> **Important:** the point clouds produced here are **synthetic**. They are
> generated from procedural 3D geometry and are **not** equivalent to a real
> LiDAR survey. They contain correct geometry and correct semantic labels by
> construction, but their noise model, intensity, and return numbers are
> deliberate simplifications.

---

## 1. Overview

The pipeline:

1. Generates procedural buildings with **Blender** (source of truth for all
   geometry) with exact, recorded ground-truth metadata.
2. Assembles small urban scenes (ground, roads, buildings, trees, vehicles,
   streetlights).
3. Samples LiDAR-like point clouds from the Blender geometry, applying
   configurable noise, dropout, variable density, and optional occlusion.
4. Splits scenes into `train` / `validation` / `test` **at the scene level**
   (no building ever leaks between splits).
5. Validates the dataset automatically.

```
Synthetic LiDAR dataset
        ↓
PointNet++ / RandLA-Net (NEXT step, not implemented)
        ↓
Train on unseen-building split
        ↓
Evaluate generalization
```

## 2. Requirements

- **Blender 4.2 LTS** (tested with `4.2.16`). Install via
  `winget install BlenderFoundation.Blender.LTS.4.2`, or set
  `blender.executable` in the config to your `blender.exe` path.
- **Python 3.12** (recommended). `open3d` does not yet ship wheels for
  Python 3.13, which is why 3.12 is recommended for the viewer.
- No GPU / CUDA is required. Everything runs on CPU.

Install Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

`requirements.txt` pulls `numpy`, `PyYAML`, and `open3d` (the last is only
needed for `view_pointcloud.py`).

## 3. Project layout

```
synthetic_city/
├── config/
│   ├── default.yaml          # recommended initial configuration
│   └── small.yaml            # small/fast configuration for smoke tests
├── core/                     # pure Python (no bpy / no GPU)
│   ├── labels.py             # fixed semantic class ids + colors
│   ├── geometry.py           # footprint shapes, roof polygons, triangulation
│   ├── lidar.py              # area-weighted surface sampling + imperfections
│   └── ply_io.py             # dependency-free binary PLY reader/writer
├── blender/                  # runs inside Blender's bundled Python
│   ├── building.py           # floor-aware building mesh + ground truth
│   ├── objects.py            # ground, roads, trees, vehicles, streetlights
│   ├── meshutil.py           # bmesh helpers
│   └── generate_scenes.py    # Blender entry point
├── utils/                    # runs in the venv (orchestrator side)
│   ├── config.py
│   ├── manifest.py           # scene manifest + split
│   └── blender_runner.py     # locate Blender + subprocess
└── scripts/
    ├── generate_dataset.py   # generate the dataset
    ├── validate_dataset.py   # run QC checks
    └── view_pointcloud.py    # Open3D viewer
ml/data/synthetic/            # generated output (gitignored)
docs/SYNTHETIC_DATA.md
```

## 4. Running the pipeline

```bash
# full dataset (7 train / 2 validation / 1 test, per default.yaml)
python synthetic_city/scripts/generate_dataset.py

# use a different config
python synthetic_city/scripts/generate_dataset.py --config synthetic_city/config/small.yaml

# validate
python synthetic_city/scripts/validate_dataset.py --config synthetic_city/config/default.yaml

# view a scene (interactive Open3D window)
python synthetic_city/scripts/view_pointcloud.py --input ml/data/synthetic/train/scene_00002

# render a PNG without opening a window
python synthetic_city/scripts/view_pointcloud.py --input ml/data/synthetic/train/scene_00002 --save out.png
```

The orchestrator (`generate_dataset.py`) writes a **job manifest**, invokes
Blender headlessly (`blender --background --python generate_scenes.py`), and
Blender produces one `pointcloud.ply` + `ground_truth.json` per scene.

## 5. How procedural buildings are generated

Buildings are generated with Blender's `bmesh` data model (Blender geometry).
For each building the generator draws, from a seeded RNG:

| Parameter | Range (configurable) |
|---|---|
| footprint type | `rectangle`, `L`, `T`, `courtyard`, `irregular` |
| width | `building.width_range` (default `[8, 30]` m) |
| length | `building.length_range` (default `[10, 50]` m) |
| floor count | `building.floors_range` (default `[1, 10]`) |
| floor height | `building.floor_height_range` (default `[2.8, 3.5]` m) |
| rotation | `building.rotation_range` (default `[0, 360]` deg) |
| roof type | `flat`, `gable`, `hip`, `shed` (simple sloped) |
| roof slope | `building.roof_slope_range` (roof rise = slope · min(width, length)) |

Footprint shapes are simple 2D polygons centred on the origin:

- **rectangle** — 4 corners (width ≤ length, so roof ridges are well defined).
- **L-shape** — rectangle with a corner notch removed (6 vertices).
- **T-shape** — a top bar plus a central stem (8 vertices).
- **courtyard** — an outer rectangle with a rectangular hole, modelled as four
  wall "bars".
- **irregular** — a 5–8 vertex, non-rectangular, star-shaped polygon.

Buildings are **floor-aware**. Floors are extruded wall segments; thin
protruding slabs are added at each internal floor boundary (configurable via
`building.floor_markers*`) so floors are visible in the point cloud — this is
intended to support later floor-level estimation. The exact floor Z-levels are
stored in metadata.

Roofs: `gable`, `hip`, and `shed` are only generated for **rectangle**
footprints (their analytic geometry assumes a rectangle plan); `L`, `T`,
`courtyard`, and `irregular` buildings receive `flat` roofs (optionally with a
parapet). This is a deliberate, documented simplification.

## 6. Ground truth

Every building's ground truth is the **exact parameters used to build its
Blender geometry** — never estimated from the point cloud. For each building we
record:

`building_id`, `footprint_type`, `roof_type`, `width`, `length`,
`footprint_local` / `footprint_world`, `hole_local` / `hole_world` (courtyard),
`position`, `rotation_deg`, `floor_count`, `floor_heights`, `floor_levels`,
`wall_height`, `roof_height`, `total_height`, `roof_slope`, `parapet_height`,
`num_triangles`.

As a correctness guarantee, the generated bmesh's bounding box is
cross-checked against the recorded `width` / `length` / `total_height` before
the building is used; a mismatch raises an error.

`floor_levels` has length `floor_count + 1` and always starts at `0.0`, e.g.:

```json
{ "floor_count": 4, "floor_levels": [0.0, 3.1, 6.2, 9.3, 12.4] }
```

## 7. Semantic classes

Fixed integer ids (order is stable across runs):

| id | class |
|---|---|
| 0 | ground |
| 1 | building |
| 2 | vegetation |
| 3 | road |
| 4 | vehicle |
| 5 | other |

Non-building objects: ground, roads, trees (trunk + canopy), vehicles
(body + cabin), and streetlights (`other`). Every point is labelled
automatically from the Blender object that produced it.

## 8. Synthetic LiDAR

Points are sampled **uniformly over object surfaces** (area-weighted triangle
sampling) in world coordinates — this guarantees correct XYZ and semantic
labels. The public interface mirrors the requested shape:

```python
point_cloud = generate_lidar(scene, config)
```

Output PLY columns: `x, y, z` (float32), `red, green, blue` (uint8, semantic
color), `class_id` (uint8), `building_id` (int32, `-1` for non-building),
`intensity` (uint8, **synthetic**), `return_number` (uint8, **synthetic**).

`intensity` and `return_number` are **not physically meaningful**; they are
present only for schema parity with real LiDAR pipelines. `intensity` is a
crude function of height/orientation and `return_number` is always 1.

Point density is configurable: `lidar.base_points_per_sqm` with per-class
multipliers (`lidar.class_multipliers`) and a hard per-scene cap
(`lidar.max_points_per_scene`, which down-samples uniformly if exceeded).

### Imperfections (all configurable, all reproducible)

| Imperfection | Config |
|---|---|
| Gaussian XYZ noise | `lidar.noise_std_m` |
| Random point dropout | `lidar.point_dropout_rate` |
| Variable point density | `lidar.variable_density`, `lidar.variable_density_range` |
| Partial occlusion / viewing angle | `lidar.sensor.occlusion`, `lidar.sensor.position` |

Occlusion (optional, off by default) simulates a single sensor position and
drops any sampled point that is hidden from that position via Blender's
`BVHTree` ray casting. Changing the sensor position changes the viewing
angle/height and thus the pattern of occlusion.

The **noiseless geometry** is the ground truth; noise/dropout are applied
*after* sampling so the clean geometry is implicitly preserved in the metadata
(recorded parameters) and reflected in the point cloud before degradation.

## 9. Dataset organization and split

```
ml/data/synthetic/
├── train/
│   └── scene_00001/
│       ├── pointcloud.ply
│       └── ground_truth.json
├── validation/…
├── test/…
└── metadata/
    ├── manifest.json           # full scene plan (seeds, splits, ids)
    ├── dataset_config.yaml     # effective config snapshot
    ├── split_summary.json
    ├── generation_info.json    # Blender version, timestamp, paths
    ├── job.json                # what was sent to Blender
    └── debug/                  # isolated single-building scenes (not in splits)
```

The split is done **by scene, not by point**. The manifest assigns each scene a
`split` and a unique `building_id_base`, so building ids are globally unique
and can never appear in two splits. Default proportions are 70 / 15 / 15
(7 train / 2 validation / 1 test).

### Test set is genuinely different

Test scenes are generated from a **different seed stream**
(`dataset.test_seed_offset`), so their dimensions, rotations, footprints,
roofs, floor counts, and surroundings are not part of the train/val seed
space. The model will be evaluated on unseen building configurations.

### Reproducibility

Every scene uses its own `random.Random(seed)` derived from the base
`dataset.seed`. The same config + seed always produces byte-identical
`pointcloud.ply` and `ground_truth.json` (verified). Change the seed to get a
different dataset: seed `26011` → dataset A, seed `26012` → dataset B, etc.

## 10. Validation

```bash
python synthetic_city/scripts/validate_dataset.py --config synthetic_city/config/default.yaml
```

Checks performed:

- **Building geometry**: height > 0, floor count ≥ 1, valid footprint
  (≥ 3 vertices), valid non-decreasing floor levels starting at 0, all required
  metadata fields present, known footprint/roof types.
- **Point cloud**: non-empty, all XYZ finite (no NaN), all `class_id` valid,
  all `building_id` either `-1` or a known building.
- **Ground-truth consistency**: each building's sampled points match its
  metadata (height, and width/length in the building's local frame, within
  noise tolerance).
- **Dataset split**: no building id appears in more than one split; scene
  counts match the config.

Validation exits non-zero on any failure.

## 11. Visualization

```bash
python synthetic_city/scripts/view_pointcloud.py --input <scene_dir_or.ply> \
    --color semantic|height [--save out.png]
```

Colors points by semantic class (default) or by height. `--save` renders
offscreen to a PNG (useful on headless machines).

## 12. Known limitations / assumptions

- Point clouds are **synthetic**; not a substitute for a real LiDAR survey.
- A single point is sampled once per surface with no multi-return physics;
  `intensity` / `return_number` are placeholders.
- `gable` / `hip` / `shed` roofs are restricted to rectangle footprints.
- Roads are modelled as slightly raised flat strips that overlap the ground
  plane (a few ground points coincide with road cells).
- With aggressive building sizes, placement can fall back to a random position
  that may overlap a neighbour (placement is best-effort, not a layout solver).
- Density defaults are starting points; class balance should be tuned for the
  eventual segmentation task.

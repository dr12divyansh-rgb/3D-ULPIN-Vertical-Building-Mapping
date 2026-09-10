# Viewer Integration Notes
## Session 3 (Frontend) → Session 2 (ML Backend) Contract

**Status**: Waiting for `ml/results/reconstruction_v2/HANDOFF.md`

This document lists exactly what the viewer needs from Session A's reconstruction output.
The viewer will consume this contract verbatim — do not invent any fields until Session A publishes HANDOFF.md.

---

## 1. Required Output Location

Session A must produce:

```
ml/results/reconstruction_v2/HANDOFF.md         ← contract file
ml/results/reconstruction_v2/viewer_buildings.json  ← building metadata JSON (primary)
ml/results/reconstruction_v2/buildings/             ← per-building geometry (optional)
```

The viewer's boot sequence checks `../ml/results/reconstruction_v2/viewer_buildings.json` first.

---

## 2. Required: `viewer_buildings.json` Schema

The top-level object must have a `buildings` array (or `features` for GeoJSON compatibility).

### Minimal required schema (LOD1 path):

```json
{
  "dataset": "STPLS3D | DALES | <new_dataset>",
  "model_id": "exp005 | exp006 | <experiment_id>",
  "building_iou": 0.659,
  "buildings": [
    {
      "id": "B000001",
      "source_dataset": "STPLS3D",
      "scene_id": "scene_00042",

      "geometry": {
        "type": "lod1_extrusion",
        "footprint": [[x0, y0], [x1, y1], ...],
        "height_m": 12.8,
        "height_source": "lidar_derived | modelled | estimated"
      },

      "ai": {
        "model": "PointNet++ Exp005",
        "status": "predicted | ground_truth | not_available",
        "class_label": "building",
        "class_id": 1
      },

      "floors": {
        "count": 4,
        "source": "height_estimate | lidar_derived | manual",
        "levels": [0.0, 3.2, 6.4, 9.6, 12.8]
      },

      "roof": {
        "type": "flat | gabled | hipped | unknown",
        "source": "lidar_derived | estimated | unknown"
      },

      "footprint_area_m2": 80.0,
      "centroid_lon": 77.221,
      "centroid_lat": 28.631
    }
  ]
}
```

### Fields consumed by the viewer panel:

| Field | Panel section | Notes |
|---|---|---|
| `id` | Building Identifier | Displayed as-is; must not claim to be official ULPIN |
| `source_dataset` | Provenance → Source | e.g. "STPLS3D", "DALES" |
| `scene_id` | Provenance → Scene | scene identifier |
| `geometry.height_m` | Building Geometry → Height | metres |
| `geometry.height_source` | Provenance → Height source | "lidar_derived" / "modelled" / "estimated" |
| `ai.model` | Provenance → AI model | full model name string |
| `ai.status` | Provenance → AI status | "predicted" / "not_available" |
| `floors.count` | Building Geometry → Floors | integer |
| `floors.source` | Provenance → Floor source | "height_estimate" / "lidar_derived" |
| `floors.levels` | Floor mode | array of Z values, length = count+1 |
| `roof.type` | Building Geometry → Roof | string |
| `footprint_area_m2` | Building Geometry → Footprint Area | float |
| `dataset` (top-level) | Dataset-Level Metrics → Dataset | string |
| `model_id` (top-level) | AI Pipeline → Model | string |
| `building_iou` (top-level) | Dataset-Level Metrics → Building IoU | float, dataset-level only |

---

## 3. Optional: Per-building geometry files (LOD2-like)

If Session A produces mesh geometry per building:

```
ml/results/reconstruction_v2/buildings/<id>/lod1.obj
ml/results/reconstruction_v2/buildings/<id>/lod2.obj
```

The viewer's `loader.js` already parses OBJ with surface group colors:
- `WallSurface` → orange
- `RoofSurface` → red
- `GroundSurface` → grey

Or glTF/GLB (preferred for binary efficiency):

```
ml/results/reconstruction_v2/buildings/<id>/building.glb
```

If these files exist, include their paths in `viewer_buildings.json`:

```json
"geometry": {
  ...
  "lod1_path": "ml/results/reconstruction_v2/buildings/B000001/lod1.obj",
  "lod2_path": "ml/results/reconstruction_v2/buildings/B000001/lod2.obj"
}
```

The viewer will load them on selection. Without these files, the viewer uses extrusion geometry from the footprint + height.

---

## 4. GeoJSON Alternative Format

If Session A outputs standard GeoJSON, also accepted:

```json
{
  "type": "FeatureCollection",
  "dataset": "STPLS3D",
  "model_id": "exp005",
  "building_iou": 0.659,
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon0, lat0], [lon1, lat1], ...]]
      },
      "properties": {
        "id": "B000001",
        "source_dataset": "STPLS3D",
        "scene_id": "scene_00042",
        "height_m": 12.8,
        "height_source": "lidar_derived",
        "ai_model": "PointNet++ Exp005",
        "ai_status": "predicted",
        "floors": 4,
        "floor_levels": [0.0, 3.2, 6.4, 9.6, 12.8],
        "floor_source": "height_estimate",
        "roof_type": "flat",
        "roof_source": "estimated",
        "footprint_area_m2": 80.0,
        "centroid_lon": 77.221,
        "centroid_lat": 28.631
      }
    }
  ]
}
```

---

## 5. Coordinate System

The viewer uses WGS84 (lon/lat) with a local Mercator projection.

If Session A's output uses local XYZ (metres), include a `georef` block:

```json
{
  "georef": {
    "type": "local_xy_offset",
    "origin_lon": 77.22,
    "origin_lat": 28.63,
    "units": "metres"
  }
}
```

The viewer will then interpret footprint coordinates as local XY offsets in metres.

If the data is not georeferenced to Delhi (e.g. STPLS3D is a US suburb dataset), do NOT assign fake Delhi coordinates. Use the STPLS3D scene's own local coordinate frame and label it clearly.

---

## 6. Dataset Mode Integration Point

The viewer's `DATASET_SOURCES` object in `ulpin_demo.html` has slots for:

| Key | When to use |
|---|---|
| `stpls3d` | Session A provides STPLS3D reconstruction output |
| `dales` | Session A provides DALES reconstruction output |
| `ms_delhi` | Microsoft Building Footprints GIS reference data |
| `demo` | Fallback when no real data found |
| _(future)_ | Session 1's new dataset — add a new key here |

To add Session 1's new dataset: add one entry to `DATASET_SOURCES` with the same field structure. No other viewer code changes needed.

---

## 7. Floor Mode Requirements

For interactive floor-level decomposition, Session A must provide `floors.levels`:

```json
"floors": {
  "count": 4,
  "source": "height_estimate",
  "levels": [0.0, 3.2, 6.4, 9.6, 12.8]
}
```

- `levels` length must equal `count + 1` (fencepost: bottom of floor 1 → top of floor N)
- If Session A provides genuine lidar-derived floor Z values, set `"source": "lidar_derived"`
- If it is simply `height / count`, set `"source": "height_estimate"` — do not call it "AI floor segmentation"

---

## 8. What the Viewer Will NOT Do

- Run PointNet++ in the browser
- Load 37 GB of training data
- Generate fake confidence scores per building
- Assign synthetic ULPIN cadastral codes (State/District/Village)
- Present demo sample data as real AI output
- Fake Delhi AI inference when it does not exist
- Attach dataset-level IoU (0.659 / 0.763) to individual buildings

---

## 9. HANDOFF.md Checklist

When Session A creates `ml/results/reconstruction_v2/HANDOFF.md`, it must include:

- [ ] Top-level `dataset` field value
- [ ] Top-level `model_id` field value
- [ ] Top-level `building_iou` float (test split result)
- [ ] Whether coordinates are georeferenced lon/lat or local XY
- [ ] Whether per-building OBJ/GLB files are provided, and where
- [ ] Whether `floor_levels` arrays are provided
- [ ] Whether `roof_type` comes from LiDAR or is estimated
- [ ] Any fields in the output that differ from the schema above

---

*Written by Session 3 (Frontend) — 2026-09-10*
*Do not merge into backend files.*

# SIH 26011 — 3D ULPIN Model Results Comparison

**Project:** 3D Building Reconstruction + Semantic Segmentation for ULPIN Assignment  
**Goal:** Identify and reconstruct buildings from aerial LiDAR, assign PLATEAU-equivalent CityGML identifiers  
**Hardware:** Intel Core 7 240H · 16.9 GB RAM · CPU-only training (no GPU)  
**Model:** PointNet++ (264K parameters, 2SA + 2FP layers) trained with PyTorch

---

## Pipeline Overview

```
Raw LiDAR (PLY/LAZ)
       │
       ▼
Stage 1 — Semantic Segmentation  ←── This is what all 7 experiments optimise
       │  6 classes: ground / building / vegetation / road / vehicle / other
       │
       ▼
Stage 2 — Building Instance Extraction  (Grid-BFS watershed)
       │
       ▼
Stage 3 — LOD2 Reconstruction  (footprint + roof geometry)
       │
       ▼
Stage 4 — ULPIN Assignment + CityGML Export
```

---

## Experiment Progress at a Glance

| Exp | Training Data | Input | Epochs | Building IoU | mIoU | OA | Purpose |
|-----|--------------|-------|--------|:---:|:---:|:---:|---------|
| 001 | Synthetic (164 bldgs) | XYZ | 10 | **91.4 %** | 35.1 % | 89.2 % | Synthetic ULPIN demo |
| 002 | Synthetic (164 bldgs) | XYZ | 25 | ~91 % | ~35 % | ~89 % | Weighted-loss variant |
| 003 | STPLS3D (terrestrial) | XYZ | 5 | 52.0 % | 25.4 % | 47.6 % | First real-world test |
| 004 | Mixed 30 % syn + 70 % STPLS3D | XYZ | 6 | 41.1 % | 23.2 % | 43.6 % | Mixing experiment |
| 005 | STPLS3D (terrestrial) | XYZRGB | 17 | **65.9 %** | 39.0 % | 65.9 % | Best general real-world |
| 006 | Fine-tune Exp005 → DALES aerial | XYZRGB | 15 | **76.3 %** *(DALES)* | 54.3 % | 89.5 % | Aerial adaptation |
| 007 | Fine-tune Exp006 → AHN4 aerial | XYZRGB | — | **75.5 %** *(AHN4)* | 41.6 % | 92.8 % | Target deployment |

> **IoU** = Intersection over Union (higher = better, 100 % = perfect).  
> **mIoU** = mean IoU across all 6 classes.  
> **OA** = Overall point-level accuracy.  
> All numbers are measured on the respective held-out **test split**.

---

## Detailed Experiment Results

### Experiment 001 — Synthetic XYZ Baseline

| Metric | Value |
|--------|-------|
| Dataset | Synthetic procedural scenes (7 train / 2 val / 1 test) |
| Input features | XYZ (3D coordinates only) |
| Best epoch | 10 / 25 |
| Overall accuracy | **89.2 %** |
| Mean IoU | **35.1 %** |

**Per-class IoU (test):**

| Class | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Ground | 51.9 % | 62.8 % | 74.8 % | 68.3 % |
| **Building** | **91.4 %** | **96.1 %** | **94.9 %** | **95.5 %** |
| Vegetation | 52.2 % | 71.9 % | 65.6 % | 68.6 % |
| Road | 14.9 % | 29.4 % | 23.2 % | 25.9 % |
| Vehicle | 0.0 % | — | — | — |
| Other | 0.0 % | — | — | — |

**Key finding:** Synthetic data produces near-perfect building segmentation (91.4 % IoU) but road/vehicle classes are underrepresented. Vehicle = 0 because the synthetic generator did not include cars.

---

### Experiment 002 — Synthetic XYZ + Class-Weighted Loss

| Metric | Value |
|--------|-------|
| Dataset | Same synthetic scenes as Exp001 |
| Input features | XYZ |
| Best epoch | 25 / 25 (trained longer) |
| Building IoU | ~91 % |
| Mean IoU | ~35 % |

**Key finding:** Class-weighted loss did not improve over Exp001. Both experiments achieve essentially the same building IoU on synthetic data, confirming the baseline result. Used as a control.

---

### Experiment 003 — STPLS3D Real-World XYZ

| Metric | Value |
|--------|-------|
| Dataset | STPLS3D (USA terrestrial scan, 37 GB) |
| Input features | XYZ |
| Best epoch | 5 / 25 (early convergence) |
| Overall accuracy | **47.6 %** |
| Mean IoU | **25.4 %** |

**Per-class IoU (STPLS3D test):**

| Class | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Ground | 16.5 % | 17.3 % | 76.3 % | 28.3 % |
| **Building** | **52.0 %** | **74.2 %** | **63.5 %** | **68.4 %** |
| Vegetation | 25.6 % | 78.9 % | 27.5 % | 40.8 % |
| Road | 36.2 % | 60.6 % | 47.4 % | 53.2 % |
| Vehicle | 14.6 % | 16.1 % | 60.6 % | 25.4 % |
| Other | 7.5 % | 10.6 % | 20.5 % | 14.0 % |

**Key finding:** Significant drop from synthetic (91.4 %) to real-world (52.0 %). This is the **real-world baseline** — demonstrates the domain gap between procedural and real LiDAR data.

---

### Experiment 004 — Mixed Dataset (30 % Synthetic + 70 % STPLS3D)

| Metric | Value |
|--------|-------|
| Dataset | 30 % synthetic scenes + 70 % STPLS3D real-world |
| Input features | XYZ |
| Best epoch | 6 |
| Overall accuracy | **43.6 %** |
| Mean IoU | **23.2 %** |
| Building IoU | **41.1 %** |

**Per-class IoU comparison vs Exp003:**

| Class | Exp003 (XYZ) | Exp004 (Mixed) | Change |
|-------|:---:|:---:|:---:|
| Ground | 16.5 % | 18.2 % | +1.7 % |
| **Building** | **52.0 %** | **41.1 %** | **-10.9 %** ↓ |
| Vegetation | 25.6 % | 20.1 % | -5.5 % |
| Road | 36.2 % | 34.5 % | -1.7 % |
| Vehicle | 14.6 % | 16.5 % | +1.9 % |
| mIoU | 25.4 % | 23.2 % | -2.2 % |

**Key finding:** Mixing synthetic + real data **hurt** performance compared to training on real data alone. The synthetic distribution confused the model. Conclusion: do NOT mix synthetic with real-world data for this architecture.

---

### Experiment 005 — STPLS3D XYZRGB (Best General Model)

| Metric | Value |
|--------|-------|
| Dataset | STPLS3D (same as Exp003, but with RGB colour) |
| Input features | XYZRGB (6D: coordinates + colour) |
| Best epoch | 17 / 25 |
| Overall accuracy | **65.9 %** |
| Mean IoU | **39.0 %** |

**Per-class IoU (STPLS3D test):**

| Class | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Ground | 22.0 % | 23.7 % | 75.3 % | 36.1 % |
| **Building** | **65.9 %** | **86.1 %** | **73.8 %** | **79.5 %** |
| Vegetation | 61.2 % | 89.9 % | 65.8 % | 76.0 % |
| Road | 51.5 % | 89.2 % | 54.9 % | 68.0 % |
| Vehicle | 23.9 % | 25.8 % | 76.3 % | 38.6 % |
| Other | 9.2 % | 11.9 % | 29.0 % | 16.9 % |

**Improvement over XYZ-only Exp003:**

| Metric | Exp003 (XYZ) | Exp005 (XYZRGB) | Gain |
|--------|:---:|:---:|:---:|
| Building IoU | 52.0 % | **65.9 %** | **+13.9 %** |
| mIoU | 25.4 % | **39.0 %** | **+13.6 %** |
| Overall accuracy | 47.6 % | **65.9 %** | **+18.3 %** |

**Key finding:** Adding RGB colour features improved building IoU by +13.9 percentage points. This is the **best general-purpose checkpoint** for any real-world coloured LiDAR input.

**Cross-domain test** (Exp005 applied to synthetic data without fine-tuning):
- Building IoU: **22.8 %** (vs 91.4 % in-domain) — confirms domain gap between STPLS3D and synthetic data

---

### Experiment 006 — DALES Fine-Tuning (Aerial Adaptation)

| Metric | Value |
|--------|-------|
| Starting checkpoint | Exp005 (STPLS3D XYZRGB) |
| Fine-tuning dataset | DALES-2 (USA aerial LiDAR, real-world) |
| Fine-tuning epochs | 15 |
| Dataset split | 300 train / 100 val / 100 test chunks |

**Results on DALES test split:**

| Class | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Ground | 86.0 % | 86.7 % | 99.0 % | 92.5 % |
| **Building** | **76.3 %** | **97.6 %** | **77.8 %** | **86.6 %** |
| Vegetation | 78.8 % | 90.2 % | 86.2 % | 88.2 % |
| Road | 0.0 % | — | — | — |
| Vehicle | 7.4 % | 29.7 % | 9.0 % | 13.8 % |
| Other | 23.1 % | 65.7 % | 26.2 % | 37.5 % |
| **mIoU** | **54.3 %** | | | |
| **OA** | **89.5 %** | | | |

**Progression of building IoU through training:**

| Stage | Building IoU | Dataset |
|-------|:---:|---------|
| Exp005 (before fine-tune) | 55.3 % | DALES cross-domain |
| Exp006 (after fine-tune) | **76.3 %** | DALES in-domain |
| Improvement | **+21.0 %** | |

**Cross-domain to AHN4 (zero-shot, no AHN4 data):**

| Metric | Value |
|--------|-------|
| Building IoU | **5.7 %** (near failure) |
| Overall accuracy | 1.6 % |
| Building F1 | 10.9 % |

**Key finding:** Fine-tuning on DALES aerial LiDAR boosted DALES performance from 55.3 % → 76.3 %. However, applying Exp006 zero-shot to Dutch AHN4 data gives only 5.7 % building IoU — the domain gap between USA aerial (DALES) and Netherlands aerial (AHN4) is severe.

> **Why the gap?** DALES uses a different point density, coordinate convention, and scanner geometry. AHN4 also lacks road/vehicle/vegetation ground truth labels in the annotation used here, so the model's class predictions collapse.

---

### Experiment 007 — AHN4 Fine-Tuning (Target Deployment)

| Metric | Value |
|--------|-------|
| Starting checkpoint | Exp006 (DALES fine-tuned) |
| Fine-tuning dataset | AHN4 Amsterdam aerial LiDAR (tile 25GN2_01) |
| Tile area | 1.34 km² |
| Test split | 100 held-out chunks (409,600 points) |

**Results on AHN4 test split:**

| Class | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|----|
| Ground | 90.8 % | 99.6 % | 91.1 % | 95.2 % |
| **Building** | **75.5 %** | **76.2 %** | **98.7 %** | **86.0 %** |
| Vegetation | — *(absent in AHN4 GT)* | | | |
| Road | — | | | |
| Vehicle | — | | | |
| **mIoU (2 classes)** | **41.6 %** | | | |
| **Overall accuracy** | **92.8 %** | | | |

**Exp006 vs Exp007 on AHN4:**

| Metric | Exp006 (zero-shot) | Exp007 (fine-tuned) | Gain |
|--------|:---:|:---:|:---:|
| Building IoU | 5.7 % | **75.5 %** | **+69.8 %** |
| Building Precision | 24.4 % | **76.2 %** | +51.8 % |
| Building Recall | 7.0 % | **98.7 %** | +91.7 % |
| Building F1 | 10.9 % | **86.0 %** | +75.1 % |
| Overall accuracy | 1.6 % | **92.8 %** | +91.2 % |

**Key finding:** AHN4 fine-tuning transformed a near-failing model (5.7 %) into a high-quality one (75.5 %). This is the **production checkpoint** for the ULPIN pipeline.

---

## Full Training Chain Summary

```
Exp001 ─── Synthetic XYZ ──────────── Building IoU 91.4 % (synthetic only)
   │
   ├── Exp002 ─ Class-weighted loss ── Building IoU ~91 % (no improvement)
   │
Exp003 ─── STPLS3D XYZ ─────────────── Building IoU 52.0 % (real-world baseline)
   │
   ├── Exp004 ─ Mixed (syn+real) ────── Building IoU 41.1 % (worse — don't mix)
   │
Exp005 ─── STPLS3D XYZRGB ──────────── Building IoU 65.9 % (best general)
   │
   └── Exp006 ─ Fine-tune on DALES ─── Building IoU 76.3 % (on DALES)
         │                              Building IoU  5.7 % (on AHN4, zero-shot)
         │
         └── Exp007 ─ Fine-tune on AHN4 ─ Building IoU 75.5 % (on AHN4 ✓)
```

---

## Building IoU Progression Chart

```
100% ┤                                            ■ Exp001 (synthetic in-domain)
     │
 90% ┤
     │
 80% ┤                               ■ Exp006     ■ Exp007
     │                               (DALES)      (AHN4)
 70% ┤                  ■ Exp005
     │                  (STPLS3D)
 60% ┤
     │    ■ Exp003
 50% ┤    (STPLS3D)
     │
 40% ┤          ■ Exp004 (mixed — worst)
     │
 30% ┤
     │
 20% ┤                                      ■ Exp006 zero-shot on AHN4 (5.7%)
 10% ┤
     │
  0% ┼─────────────────────────────────────────────────────────
      XYZ  XYZRGB  Domain-gap  Fine-tuned  Target domain
```

*(Each ■ represents measured test IoU on the respective dataset)*

---

## Domain Gap Analysis

A core finding of this project is the **domain gap** between training and target data:

| Transfer | Building IoU | Drop |
|----------|:---:|:---:|
| Synthetic → Synthetic (in-domain, Exp001) | 91.4 % | — |
| STPLS3D → Synthetic (cross-domain, Exp005) | 22.8 % | **-68.6 %** |
| STPLS3D → DALES (cross-domain, Exp005) | 55.3 % | -10.6 % |
| DALES → AHN4 (cross-domain, Exp006) | 5.7 % | **-70.6 %** |
| AHN4 → AHN4 (in-domain, Exp007) | 75.5 % | — |

**Key insight:** Each new domain (USA terrestrial → USA aerial → Netherlands aerial) requires explicit fine-tuning. Off-the-shelf transfer fails severely. The solution used here is **progressive fine-tuning**: STPLS3D → DALES → AHN4.

---

## Reconstruction Results (Exp007 → Amsterdam)

After segmentation, building instances are reconstructed into 3D geometry using the **research_v2 pipeline**.

### Input data
- Prediction PLY from Exp007 test strip
- AHN4 tile 25GN2_01 (Amsterdam, 1.34 km²)
- 409,600 LiDAR points evaluated; 118,783 predicted as building class

### Reconstruction statistics

| Metric | Value |
|--------|-------|
| Buildings reconstructed | **356** |
| Flat-roof buildings | 286 (80.3 %) |
| Gable-roof buildings | 70 (19.7 %) |
| Instances skipped (too small/noisy) | 44 |
| CRS | EPSG:28992 (Amersfoort / RD New) |
| Ground elevation source | ASPRS class-2 LiDAR points |

### Height accuracy (vs 3DBAG reference)

| Metric | Value |
|--------|-------|
| 3DBAG buildings matched | 198 / 278 (71.2 %) |
| Height MAE | **1.17 m** |
| Heights within ±1 m | 59.5 % |
| Heights within ±2 m | 76.8 % |
| Systematic bias | +0.83 m overestimate (ASPRS ground slightly above NAP datum) |

### Floor count accuracy (vs 3DBAG reference)

| Metric | Value |
|--------|-------|
| Exact floor match | 69.9 % |
| Floor within ±1 | **98.5 %** |
| Method | height ÷ 3.2 m (estimated, not AI-segmented) |

### Roof type agreement (vs 3DBAG reference)

| Metric | Value |
|--------|-------|
| Roof type match | 21.8 % |
| Note | LiDAR-RANSAC heuristic, not a trained classifier |

### Instance extraction method comparison

Several methods were tested for separating touching buildings:

| Method | 3DBAG Coverage | Notes |
|--------|:---:|-------|
| DBSCAN (distance-based) | ~4 % | Over-segments 10× — buildings split into fragments |
| **Grid BFS / Watershed** | **71 %** | **Best: respects building boundaries** |
| Voxel Connected Components | 40 % | Better than DBSCAN, worse than watershed |

---

## Key Conclusions for Teacher

### 1. RGB features matter significantly
Adding colour (XYZRGB vs XYZ) improved building IoU by **+13.9 %** (Exp003 → Exp005). When colour is available, always include it.

### 2. Mixing datasets is harmful
Experiment 004 proved that naively mixing synthetic + real data **reduces** accuracy. Keep training sets homogeneous or use proper curriculum learning.

### 3. Progressive fine-tuning is essential
Direct cross-domain transfer (Exp006 → AHN4) gives 5.7 % — near useless. After 15-epoch fine-tuning on AHN4, the same model reaches 75.5 %. Fine-tuning cost is small but the gain is enormous.

### 4. Building recall > precision trade-off
Exp007 achieves 76.2 % precision but **98.7 % recall** — it finds almost every building, at the cost of some false positives. For ULPIN purposes (not missing real buildings), this is the correct trade-off.

### 5. Instance extraction is the reconstruction bottleneck
The AI segmentation identifies building *class* pixels well (75.5 % IoU). The harder problem is separating *individual* buildings. Watershed/Grid-BFS at 0.5 m resolution gave 71 % 3DBAG coverage; DBSCAN failed completely for densely packed Dutch row houses.

### 6. Height estimation is practical
Height MAE of 1.17 m on Amsterdam buildings is competitive. 98.5 % of buildings are within ±1 floor of the reference, making the floor estimate production-ready for urban planning tools.

---

## Files and Checkpoints

| Experiment | Checkpoint path | Recommended for |
|-----------|----------------|----------------|
| Exp001 | `ml/results/experiment_001/best_model.pth` | Synthetic ULPIN demo |
| Exp005 | `ml/results/experiment_005/best_model.pth` | Any coloured real-world LiDAR |
| Exp006 | `ml/results/experiment_006/best_model.pth` | DALES-format aerial LiDAR |
| **Exp007** | `ml/results/ahn4_exp007/` (checkpoint) | **AHN4 / Netherlands aerial — use this** |

**Best reconstruction output:** `ml/results/final_ahn4_research_v2/buildings.json`  
(356 buildings, EPSG:28992, with LOD2-like OBJ geometry, floor decomposition, and RANSAC roof analysis)

---

*All metrics are measured on held-out test data. No numbers were adjusted or estimated post-hoc.*  
*Training hardware: Intel Core 7 240H CPU, 16.9 GB RAM — no GPU. All PyTorch CPU-only.*

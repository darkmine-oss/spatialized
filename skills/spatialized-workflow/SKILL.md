---
name: spatialized-workflow
description: Use when running spatialized workflows for spatial random forest data preparation, supervised prediction, unsupervised clustering, feature layout metadata, zone-of-influence export, or end-to-end GeoTIFF prediction.
---

# Spatialized Workflow

Use this skill when a task involves this repository's spatial random forest
workflow. Prefer the CLI for repeatable file-based workflows and the Python API
for tests, notebooks, or custom orchestration.

## Feature Layout Metadata

Use the CLI when the user needs auditable feature-column metadata:

```bash
spatialized feature-layout \
  --layer mag:7 \
  --layer gravity:5:0,12,24 \
  --rotations \
  --output feature-layout.json
```

Use `feature_layout(layers).to_dict()` from Python when layer objects already
exist in memory.

## Full-Grid Prediction

For file-based prediction, use:

```bash
spatialized predict-grid \
  --model model.pkl \
  --layer mag:mag.tif:7 \
  --layer gravity:gravity.tif:5 \
  --mask-raster prediction-grid.tif \
  --mask-mode nan \
  --output classes.tif \
  --entropy-output entropy.tif
```

The model must be a pickled fitted spatialized model wrapper. Use `--mask-mode
all`, `valid`, `nan`, or `nodata` to choose prediction cells.

## Python API Steps

Typical supervised workflow:

```python
from spatialized import (
    SpatialRandomForestClassifier,
    predict_grid_to_raster,
    read_raster,
    read_spatial_layer,
)

layers = [
    read_spatial_layer("mag.tif", name="mag", window_size=7),
    read_spatial_layer("gravity.tif", name="gravity", window_size=5),
]
model = SpatialRandomForestClassifier(n_estimators=500, random_state=42)
model.fit(layers, training_centers, training_labels, rotations=True)

reference = read_raster("prediction-grid.tif", masked=True)
predict_grid_to_raster(
    model,
    layers,
    prediction_mask=mask,
    reference=reference,
    output_path="classes.tif",
    entropy_path="entropy.tif",
)
```

Typical unsupervised workflow:

```python
from spatialized import UnsupervisedSpatialRandomForest

model = UnsupervisedSpatialRandomForest(n_estimators=500, random_state=42)
model.fit(layers, centers, rotations=True)
clusters = model.spectral_cluster(n_clusters=4)
embedding = model.mds(n_components=2)
```

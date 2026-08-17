---
name: spatial-classify
description: Use when training a SpatialRandomForestClassifier, predicting full-grid classes with entropy/uncertainty rasters, computing out-of-bag accuracy, or exporting feature importance / zone-of-influence maps for a classification workflow. Requires spatial-data-prep first (build SpatialLayer objects and training centers).
---

# Spatial Classify

Supervised classification over vectorised spatial patterns — e.g.
reproducing interpreted geology classes from geophysical covariates, the
same task as the original R `NWMP.R` case study. Do `spatial-data-prep`
first: you need `layers` and `training_centers` before anything here.

## Fit

```python
from spatialized import SpatialRandomForestClassifier

model = SpatialRandomForestClassifier(
    n_estimators=1000,
    random_state=42,
    encoder_kwargs={"numeric_missing_strategy": "window_mean"},
    estimator_kwargs={"oob_score": True, "bootstrap": True},  # needed for OOB below
)
model.fit(layers, training_centers, training_labels, rotations=True)
```

## Out-of-bag diagnostics

Requires `estimator_kwargs={"oob_score": True, "bootstrap": True}` at fit
time:

```python
oob_predictions = model.oob_prediction()
oob_accuracy = model.oob_accuracy_score()
oob_probabilities = model.oob_decision_function()
```

## Full-grid prediction with entropy

```python
from spatialized import predict_grid_to_raster
import numpy as np

predict_grid_to_raster(
    model,
    layers,
    prediction_mask=np.isnan(reference.values),  # or any boolean mask
    reference=reference,          # a RasterGrid, e.g. from read_raster()
    output_path="classes.tif",
    entropy_path="entropy.tif",   # standardised Shannon entropy, base = n_classes
    chunk_size=20_000,
)
```

Or via CLI, from a pickled fitted model:

```bash
spatialized predict-grid \
  --model model.pkl \
  --layer mag_rtp:mag_rtp.tif:7 \
  --layer gravity:gravity.tif:5 \
  --mask-raster prediction-grid.tif \
  --mask-mode nan \
  --output classes.tif \
  --entropy-output entropy.tif
```

For an in-memory prediction (no raster I/O), use `predict_grid(...)` — same
arguments minus `reference`/`output_path`, returns a `GridPrediction`.

## Feature importance / zone of influence

```python
importance = model.feature_importance()          # per fitted-column impurity importance
zones = model.zone_of_influence(layers)           # per-layer 2D window importance grids
mag_zone = zones["mag_rtp"]
```

`zone_of_influence` maps each column's importance back to its window
row/col position, so you get a per-layer heatmap of where in the local
neighbourhood the model is drawing signal from.

## Validated real-data example

`examples/validate_realcase_geology_classification.py` is a complete,
working reference: loads real cropped rasters, builds a stratified training
sample matching the original R script's `mysam.csv`, fits with
`window_size=10` (even — see `spatial-data-prep`), predicts the full grid,
and compares against a recovered R reference raster (92%+ direct class
agreement). Use it as a template for a real classification workflow.

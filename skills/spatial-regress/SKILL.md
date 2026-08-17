---
name: spatial-regress
description: Use when training a SpatialRandomForestRegressor, predicting a full-grid continuous target (e.g. geochemistry concentration), computing out-of-bag R^2, or exporting feature importance / zone-of-influence maps for a regression workflow. Requires spatial-data-prep first (build SpatialLayer objects and training centers).
---

# Spatial Regress

Supervised regression over vectorised spatial patterns — e.g. predicting Cu
geochemistry concentration from magnetic and gravity covariates, the same
task as the original R `NWMP.R` (`realCase2`) Cu-regression case study. Do
`spatial-data-prep` first: you need `layers` and `training_centers` before
anything here.

## Fit

```python
from spatialized import SpatialRandomForestRegressor

model = SpatialRandomForestRegressor(
    n_estimators=1000,
    random_state=42,
    encoder_kwargs={"numeric_missing_strategy": "window_mean"},
    estimator_kwargs={"oob_score": True, "bootstrap": True},  # needed for OOB below
)
model.fit(layers, training_centers, target_values, rotations=True)
```

## Out-of-bag diagnostics

Requires `estimator_kwargs={"oob_score": True, "bootstrap": True}` at fit
time:

```python
oob_predictions = model.oob_prediction()
oob_r2 = model.oob_r2_score()
```

For diagnostics against held-out predictions computed elsewhere (not OOB),
use the standalone helpers:

```python
from spatialized import regression_r2_score, regression_residuals

r2 = regression_r2_score(observed, predicted)          # matches R's 1 - SSE/SST formula
residuals = regression_residuals(observed, predicted)   # predicted - observed
```

## Full-grid prediction

```python
from spatialized import predict_grid_to_raster
import numpy as np

predict_grid_to_raster(
    model,
    layers,
    prediction_mask=np.isnan(reference.values),
    reference=reference,
    output_path="prediction.tif",
    chunk_size=20_000,
)
```

Regression models don't have an entropy output (that's classification-only);
use OOB R² or held-out R² for uncertainty framing instead.

## Feature importance / zone of influence

Same API as classification:

```python
importance = model.feature_importance()
zones = model.zone_of_influence(layers)
```

## Validated real-data example

`examples/validate_realcase2_cu_regression.py` is a complete, working
reference: loads real cropped rasters, target values from a geochem CSV,
fits with `window_size=10` (even — see `spatial-data-prep`), predicts the
full grid, and compares OOB R² and full-grid predictions against the
original R script's saved output (`gheochemcuSRF.tif`) and the paper's
reported R² (0.684). Use it as a template for a real regression workflow.

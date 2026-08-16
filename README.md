# spatialized

Spatial random forest workflows for gridded geoscience data.

`spatialized` turns local raster neighbourhoods into vectorised spatial patterns,
then uses those patterns for classification, regression, unsupervised clustering,
full-grid GeoTIFF prediction, entropy maps, and feature-importance zone-of-
influence maps.

The core spatial-pattern technique follows:

> Hassan Talebi, Luk J. M. Peeters, Alex Otto, Raimon Tolosana-Delgado (2022).
> A Truly Spatial Random Forests Algorithm for Geoscience Data Analysis and
> Modelling. Mathematical Geosciences 54, 1-22.
> https://doi.org/10.1007/s11004-021-09946-w

This implementation is based on the paper and an authorised review of the
original R code with permission from first author Hassan Talebi. Original source
material and paper test data are not distributed in this repository.

## Install

```bash
pip install "spatialized[full]"
```

## Train and Predict

```python
import numpy as np

from spatialized import (
    SpatialRandomForestClassifier,
    centers_from_mask,
    predict_grid_to_raster,
    read_raster,
    read_spatial_layer,
)

mag = read_spatial_layer("mag.tif", window_size=7)
grav = read_spatial_layer("gravity.tif", window_size=5)
prediction = read_raster("prediction-grid.tif", masked=True)

training_centers = np.array([[120, 240], [380, 410], [615, 190]])
training_labels = np.array(["class-a", "class-b", "class-a"])

model = SpatialRandomForestClassifier(n_estimators=500, random_state=42)
model.fit([mag, grav], training_centers, training_labels, rotations=True)

predict_grid_to_raster(
    model,
    [mag, grav],
    prediction_mask=np.isnan(prediction.values),
    reference=prediction,
    output_path="classes.tif",
    entropy_path="entropy.tif",
    chunk_size=10_000,
)
```

## Unsupervised Clustering

```python
from spatialized import UnsupervisedSpatialRandomForest

centers = centers_from_mask(np.isnan(prediction.values))

unsupervised = UnsupervisedSpatialRandomForest(n_estimators=500, random_state=42)
unsupervised.fit([mag, grav], centers, rotations=True)

clusters = unsupervised.spectral_cluster(n_clusters=4)
embedding = unsupervised.mds(n_components=2)
distance = unsupervised.distance_
```

## Paper-Style Synthetic Experiment

Run a synthetic experiment that mirrors the paper workflow structure without
using restricted paper data:

```bash
python examples/paper_like_experiment.py --output-dir paper_like_outputs
```

It creates magnetic/gravity-like covariates, geology-like classes, full-grid
class predictions, entropy, zone-of-influence rasters, and unsupervised clusters.

For ferricrete/paleovalley-style targeting, use `predict_target_proxy_transfer`
to train on labelled target units in one area and predict equivalent target
proxies in another area.

For unsupervised potential-field domains, use `predict_unsupervised_domains` to
cluster sampled spatial patterns and predict domain labels and entropy over a
full grid.

Deep-feature potential-field clustering helpers are also available for
normalising raster channels, extracting patches, and clustering feature vectors.
The pretrained CNN extractor itself is planned as an optional heavier workflow.

## Feature Importance

```python
importance = model.feature_importance()
zones = model.zone_of_influence([mag, grav])

mag_zone = zones["mag"]
grav_zone = zones["gravity"]
```

## CLI

```bash
spatialized feature-layout \
  --layer mag:7 \
  --layer gravity:5 \
  --rotations \
  --output feature-layout.json

spatialized predict-grid \
  --model model.pkl \
  --layer mag:mag.tif:7 \
  --layer gravity:gravity.tif:5 \
  --mask-raster prediction-grid.tif \
  --mask-mode nan \
  --output classes.tif \
  --entropy-output entropy.tif
```

## Validation against the original R workflow

The point of this package is to match the original R implementation, not just
approximate its spirit. Real Paper Author R scripts and data (not distributed
in this repository) have been used to validate two supervised cases end to
end, comparing this package's output directly against the R scripts' own
saved and re-generated outputs:

- **Cu geochemistry regression** — `examples/validate_realcase2_cu_regression.py`.
  Full-grid predictions correlate r>=0.85 with R's saved output raster; OOB
  R^2 tracks the paper's reported value closely, and the qualitative finding
  (spatial RF beats classical non-spatial RF) reproduces.
- **Geology classification** — `examples/validate_realcase_geology_classification.py`.
  Full-grid accuracy against the true labels matches R's within ~1
  percentage point (77.1% vs 77.8%), and 92% of individual cell predictions
  agree directly between the two implementations.

The unsupervised workflow does **not** yet match this closely.
`UnsupervisedSpatialRandomForest` uses the same real-vs-synthetic-pattern
discrimination mechanism `randomForestSRC` documents itself using internally
in unsupervised mode, but validating it against a real R distance matrix
(`examples/validate_unsupervised_srf_distance.py`) shows only moderate
correlation (Pearson r ~= 0.35) and weak-to-moderate cluster agreement
(adjusted Rand index ~= 0.29) — see `.features/PLAN.md` ("Improve
Unsupervised SRF Parity") for the full finding and open questions.

See `.features/PLAN.md` for the current status of every workflow, including
what's been validated, what's approximated and how, and what's still
blocked on missing reference data.

## Notes

- Numeric and categorical raster values are supported by the model wrappers.
- Missing values are handled by configurable encoder strategies: constant,
  numeric mean (global per-column), spatially-local `window_mean` (fills
  from the other valid cells in the same local window, closer to but still
  coarser than R's `randomForestSRC(na.action="na.impute")`), and
  categorical most-frequent fills.
- `SpatialLayer` supports both odd and even `window_size`, matching the real
  R workflows (which use both, depending on the script).
- GeoTIFF I/O is optional and uses `rasterio`; `read_raster`/
  `read_spatial_layer` support `window=`/`bounds=` for cropped reads.

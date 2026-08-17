# spatialized

Spatial random forest workflows for gridded geoscience data.

`spatialized` turns local raster neighbourhoods into vectorised spatial patterns,
then uses those patterns for classification, regression, unsupervised clustering,
full-grid GeoTIFF prediction, entropy maps, and feature-importance zone-of-
influence maps.

## What is this? (ELI5)

Ordinary random forests look at one pixel at a time:

```text
this one pixel's magnetic value, gravity value -> what rock type is this?
```

That throws away context. A geologist doesn't identify a rock unit by
looking at a single point on a map — they look at the *shape* around it:
is this pixel next to a sharp magnetic edge? Is it inside a circular
gravity high? Spatial context like that often matters more than the raw
value at one point.

`spatialized` gives the model that context by cropping a small window
around each point, for every input layer, and flattening those windows into
one long list of numbers — a "vectorised spatial pattern":

```text
   magnetic          gravity
  ┌─┬─┬─┬─┬─┐       ┌─┬─┬─┬─┬─┐
  ├─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┤
  ├─┼─●─┼─┼─┤   +   ├─┼─●─┼─┼─┤   ->  [flattened magnetic window, flattened gravity window]
  ├─┼─┼─┼─┼─┤       ├─┼─┼─┼─┼─┤          -> one long row of numbers -> random forest
  └─┴─┴─┴─┴─┘       └─┴─┴─┴─┴─┘
   (● = the pixel being predicted, window cropped around it)
```

Every input raster contributes its own window (they don't have to be the
same size — a coarse gravity survey can use a smaller window than a
high-resolution magnetic survey), and all the windows for one point get
concatenated into a single row before it goes into the forest. That row is
one training or prediction example. Do this for every pixel in the raster
and you get a full training matrix or a full prediction grid.

A few extra tricks the library handles for you:

- **Rotation invariance**: each window is also fed in rotated 90°, 180°, and
  270°, so the model doesn't learn "north-facing edges only" by accident —
  it learns the *shape*, regardless of orientation.
- **Entropy / uncertainty maps**: alongside a predicted class, you get a
  0–1 confidence score per pixel (low = the model is unsure) — a whole
  raster you can render alongside the prediction.
- **Zone of influence**: which parts of each input window actually mattered
  to the model, mapped back onto the same window shape — a heatmap of "the
  model looked here" for every input layer.

What you can build with this:

| I have... | I want... | Use |
|---|---|---|
| a few labelled points (e.g. "this is granite", "this is basalt") | a full-grid rock-type map | `SpatialRandomForestClassifier` |
| a few sampled measurements (e.g. Cu ppm at drill holes) | a full-grid concentration map | `SpatialRandomForestRegressor` |
| no labels at all, just raw covariate rasters | groups of pixels that "look similar" spatially (candidate geological domains) | `UnsupervisedSpatialRandomForest` |
| covariate rasters, no labels, want deep-learning texture features instead of hand-built spatial patterns | domains found via a pretrained image model (ResNet50) instead of a forest | `extract_resnet_features` + `cluster_feature_vectors` |

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

## Quickstart (no files needed)

This runs end to end with made-up data — copy, paste, run:

```python
import numpy as np
from spatialized import (
    SpatialLayer,
    SpatialRandomForestClassifier,
    centers_from_shape,
    predict_grid,
)

# Stand-in for two real raster layers, e.g. RTP magnetics and gravity.
rng = np.random.default_rng(0)
rows, cols = np.indices((20, 20))
magnetic = np.sin(rows / 3) + rng.normal(scale=0.1, size=(20, 20))
gravity = np.cos(cols / 3) + rng.normal(scale=0.1, size=(20, 20))

mag_layer = SpatialLayer("magnetic", magnetic, window_size=5)
grav_layer = SpatialLayer("gravity", gravity, window_size=5)

# A handful of "known" labelled points to train on.
training_centers = centers_from_shape((20, 20))[::7]
training_labels = np.where(
    magnetic[training_centers[:, 0], training_centers[:, 1]] > 0, "granite", "basalt"
)

model = SpatialRandomForestClassifier(n_estimators=200, random_state=42)
model.fit([mag_layer, grav_layer], training_centers, training_labels, rotations=True)

# Predict rock type (and confidence) for every pixel in the grid.
prediction = predict_grid(
    model, [mag_layer, grav_layer],
    prediction_mask=np.ones((20, 20), dtype=bool),
    entropy=True,
)
print(prediction.prediction.shape)   # (20, 20) array of "granite"/"basalt"
print(prediction.entropy.shape)      # (20, 20) array of confidence scores
```

## Classify: predict categories over a grid

Real GeoTIFF workflow — train on a handful of labelled points, predict rock
type/lithology class for every pixel, with an uncertainty raster:

```python
import numpy as np

from spatialized import (
    SpatialRandomForestClassifier,
    predict_grid_to_raster,
    read_raster,
    read_spatial_layer,
)

mag = read_spatial_layer("mag.tif", window_size=7)
grav = read_spatial_layer("gravity.tif", window_size=5)
prediction_grid = read_raster("prediction-grid.tif", masked=True)

training_centers = np.array([[120, 240], [380, 410], [615, 190]])
training_labels = np.array(["granite", "basalt", "granite"])

model = SpatialRandomForestClassifier(
    n_estimators=500,
    random_state=42,
    estimator_kwargs={"oob_score": True, "bootstrap": True},  # for OOB accuracy below
)
model.fit([mag, grav], training_centers, training_labels, rotations=True)

print("out-of-bag accuracy:", model.oob_accuracy_score())

predict_grid_to_raster(
    model,
    [mag, grav],
    prediction_mask=np.isnan(prediction_grid.values),
    reference=prediction_grid,
    output_path="classes.tif",
    entropy_path="entropy.tif",   # low = the model is unsure
    chunk_size=10_000,
)
```

## Regress: predict a continuous value over a grid

Same idea, but the target is a number instead of a category — e.g.
predicting a geochemical concentration from geophysics:

```python
from spatialized import SpatialRandomForestRegressor, regression_r2_score

training_values = np.array([420.0, 55.0, 610.0])  # e.g. Cu concentration, ppm

model = SpatialRandomForestRegressor(
    n_estimators=500,
    random_state=42,
    estimator_kwargs={"oob_score": True, "bootstrap": True},
)
model.fit([mag, grav], training_centers, training_values, rotations=True)

print("out-of-bag R^2:", model.oob_r2_score())

predict_grid_to_raster(
    model,
    [mag, grav],
    prediction_mask=np.isnan(prediction_grid.values),
    reference=prediction_grid,
    output_path="cu_ppm_prediction.tif",
    chunk_size=10_000,
)
```

`regression_r2_score(observed, predicted)` and `regression_residuals(...)`
are also available standalone, for scoring predictions computed elsewhere.

## Unsupervised Clustering: find spatial groupings with no labels

No labelled data at all — just group pixels by how similar their local
spatial *shape* is, a candidate first pass at finding geological domains:

```python
from spatialized import UnsupervisedSpatialRandomForest, centers_from_mask

centers = centers_from_mask(np.isnan(prediction_grid.values))

unsupervised = UnsupervisedSpatialRandomForest(n_estimators=500, random_state=42)
unsupervised.fit([mag, grav], centers, rotations=True)

clusters = unsupervised.spectral_cluster(n_clusters=4)   # which group each center falls in
embedding = unsupervised.mds(n_components=2)              # for a 2D scatter-plot view
distance = unsupervised.distance_                          # the underlying center-to-center distances
```

**Known limitation**: this approximates the original R workflow's
`randomForestSRC(distance="all")` behaviour, but validated against real R
output it's only moderately correlated (see "Validation against the
original R workflow" below) — treat clusters as a candidate grouping to
sanity-check, not a numerically exact reproduction of what the R script
would produce.

To predict cluster/domain membership over every pixel in a full grid (not
just the sampled centers), plus an entropy raster:

```python
from spatialized import predict_unsupervised_domains

domains = predict_unsupervised_domains(
    [mag, grav], centers, np.isnan(prediction_grid.values),
    n_clusters=4, rotations=True,
)
```

## Deep-Feature Clustering: cluster with a pretrained CNN instead

An alternative to the forest-based approach above: use a pretrained
ResNet50 as a feature extractor over small image patches, then cluster
those features. Requires the `deep` extra (`pip install "spatialized[deep]"`
— installs `torch`, `torchvision`, `umap-learn`):

```python
from spatialized import (
    normalize_channels,
    patch_centers,
    extract_patches,
    extract_resnet_features,
    point_intensities_at_centers,
    cluster_feature_vectors,
)

# Stack same-shaped rasters into one normalised (0..1) 3-channel image.
image = normalize_channels([rtp, one_vd, gravity])

# 16x16 patches (deliberately small -- avoids edge effects in dense
# sliding-window extraction), every 8 pixels.
centers = patch_centers(image.shape[:2], patch_size=16, stride=8)
patches = extract_patches(image, centers, patch_size=16)

cnn_features = extract_resnet_features(patches, batch_size=64)   # 2048-dim per patch
intensities = point_intensities_at_centers(image, centers)        # raw pixel values at each center

result = cluster_feature_vectors(
    cnn_features, centers,
    output_shape=image.shape[:2],
    n_clusters=8, n_components=6, random_state=42,
    point_intensities=intensities,   # concatenated onto the reduced embedding
)
result.label_grid   # cluster labels mapped back onto the raster, -1 = unsampled
```

Note: unlike the forest-based workflows above, no real R run of this
specific workflow exists to validate numerically against — see
`skills/spatial-deep-cluster/SKILL.md` for the full caveat.

## Ferricrete / Target-Proxy Transfer

Have a mapped unit in one area (e.g. a ferricrete/paleovalley polygon, as a
boolean mask) and covariate layers for a second, unmapped area with matching
inputs? Train a target-vs-background classifier from the mask in the first
area, then predict where that same unit likely occurs in the second:

```python
from spatialized import predict_target_proxy_transfer

transfer = predict_target_proxy_transfer(
    [mag_train, grav_train],           # covariate layers, training area
    [mag_transfer, grav_transfer],     # matching covariate layers, transfer area
    train_target_mask,                 # boolean mask: True where the mapped unit is
    target_prediction_mask,            # boolean mask: where to predict in the transfer area
    rotations=True,
)
transfer.prediction.prediction   # predicted class grid in the transfer area
transfer.prediction.entropy      # uncertainty grid
```

## Paper-Style Synthetic Experiment

Run a synthetic experiment that mirrors the paper workflow structure without
using restricted paper data:

```bash
python examples/paper_like_experiment.py --output-dir paper_like_outputs
```

It creates magnetic/gravity-like covariates, geology-like classes, full-grid
class predictions, entropy, zone-of-influence rasters, and unsupervised clusters.

## Feature Importance & Zone of Influence

Which input layers — and which part of each layer's local window — actually
drove the model's predictions:

```python
importance = model.feature_importance()
zones = model.zone_of_influence([mag, grav])

mag_zone = zones["mag"]      # same shape as mag's window: a heatmap of "the model looked here"
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

The unsupervised workflow does **not** match this closely, and by design
isn't going to. `UnsupervisedSpatialRandomForest` uses the same
real-vs-synthetic-pattern discrimination mechanism `randomForestSRC`
documents itself using internally in unsupervised mode, but validating it
against a real R distance matrix (`examples/validate_unsupervised_srf_distance.py`)
shows only moderate correlation (Pearson r ~= 0.35) and weak-to-moderate
cluster agreement (adjusted Rand index ~= 0.29). This is confirmed to be a
real algorithmic gap, not run-to-run noise (R's own reproducibility on the
same input is r ~= 0.99), and the decision was made not to pursue a closer
`randomForestSRC`-compatible backend given the effort involved — see
`.features/PLAN.md` ("Improve Unsupervised SRF Parity") for the full
finding. Treat this workflow as directionally similar to R, not numerically
interchangeable with it.

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
  R workflows (which use both, depending on the script). For an even
  `window_size`, there's no unique center pixel: `(window_size - 1) // 2`
  cells are taken before the target pixel and the rest after — this exact
  convention is verified against real recovered R output, not assumed.
- GeoTIFF I/O is optional and uses `rasterio`; `read_raster`/
  `read_spatial_layer` support `window=`/`bounds=` for cropped reads.

## Agent Skills

For step-by-step, copy-paste-ready guidance (including the details behind
each snippet above), see `skills/`: `spatial-data-prep` (loading rasters,
choosing window sizes, building training centers — start here), then
`spatial-classify`, `spatial-regress`, `spatial-cluster`, or
`spatial-deep-cluster` depending on which workflow you need.

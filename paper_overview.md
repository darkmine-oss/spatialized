# Paper Overview

## Premise

The paper introduces Spatial Random Forests (SRF) for geoscience modelling.
Classical random forests usually treat each sample or pixel as an independent
feature vector. That is a poor fit for geoscience data, where spatial context,
textures, contacts, lineaments, gradients, and anomaly shapes often matter as
much as point values.

Instead of modelling a location as:

```text
location -> magnetic value, gravity value, radiometric value
```

SRF models a location as:

```text
location -> local magnetic window + local gravity window + local radiometric window
```

Each local window is flattened into a vectorised spatial pattern. The random
forest then learns from spatial context around each observation, not just the
single pixel value at the observation.

## Data Used in the Paper

The real case study uses data from the North West Minerals Province in
Queensland, Australia.

Input datasets include:

- DEM
- magnetic VRTP
- magnetic first vertical derivative
- gamma-ray spectrometry / radiometrics
- Bouguer gravity
- interpreted solid geology map
- soil geochemistry samples, specifically Cu at 3,755 locations

The paper demonstrates three broad workflows:

- **Unsupervised SRF:** discover spatial structures and clusters from covariates.
- **Classification SRF:** reproduce interpreted geology classes from geophysical
  and radiometric covariates.
- **Regression SRF:** predict Cu concentration from magnetic and gravity
  covariates.

The real predictive mineralisation example is Cu concentration prediction, not a
gold occurrence classifier. However, the method is directly relevant to mineral
prospectivity modelling.

## Prospectivity Interpretation

This workflow can be adapted to gold prospectivity modelling.

Given:

- magnetics
- gravity
- radiometrics
- geology / lithology
- structures / faults
- geochemistry
- known gold hits, deposits, or anomalous assays

we can train a spatial model to produce a gold prospectivity surface.

Possible targets:

- classification: `gold hit` vs `background`
- multiclass classification: deposit style, lithological association, or
  prospectivity class
- regression: Au grade, anomaly intensity, or geochemical score
- ranking: probability-like prospectivity score

The advantage of SRF is that it can learn spatial associations such as:

- edges of magnetic anomalies
- gravity/magnetic relationships
- structural corridors
- lithological contacts
- alteration halos
- local texture and anisotropy

The main caveat is target design. Gold hits alone are not enough. A reliable
workflow needs defensible background/negative samples, spatial validation, and
careful treatment of sampling bias. Otherwise the model may learn where people
sampled or drilled instead of where gold is likely.

## Code Pattern

The spatialized package supports the main workflow pieces: raster loading,
vectorised spatial pattern preparation, supervised SRF-style modelling,
unsupervised clustering, entropy maps, and zone-of-influence maps.

### Supervised Prospectivity Classification

```python
import numpy as np

from spatialized import (
    SpatialRandomForestClassifier,
    predict_grid_to_raster,
    read_raster,
    read_spatial_layer,
)

mag = read_spatial_layer("mag.tif", name="mag", window_size=7)
gravity = read_spatial_layer("gravity.tif", name="gravity", window_size=5)
radiometrics = read_spatial_layer("thorium.tif", name="thorium", window_size=5)
prediction_grid = read_raster("prediction_grid.tif", masked=True)

training_centers = np.array([
    [120, 240],
    [380, 410],
    [615, 190],
])
training_labels = np.array([
    "background",
    "gold_hit",
    "background",
])

model = SpatialRandomForestClassifier(
    n_estimators=500,
    random_state=42,
    encoder_kwargs={
        "numeric_missing_strategy": "mean",
        "categorical_missing_strategy": "most_frequent",
    },
)
model.fit(
    [mag, gravity, radiometrics],
    training_centers,
    training_labels,
    rotations=True,
)

predict_grid_to_raster(
    model,
    [mag, gravity, radiometrics],
    prediction_mask=np.isfinite(prediction_grid.values),
    reference=prediction_grid,
    output_path="gold_prospectivity_classes.tif",
    entropy_path="gold_prospectivity_entropy.tif",
    probabilities_path="gold_prospectivity_probabilities.tif",
    chunk_size=10_000,
)
```

### Feature Importance and Zone of Influence

```python
importance = model.feature_importance()
zones = model.zone_of_influence([mag, gravity, radiometrics])

mag_zone = zones["mag"]
gravity_zone = zones["gravity"]
thorium_zone = zones["thorium"]
```

The zone-of-influence arrays show where, inside each local spatial window, the
model found useful information for prediction.

### Unsupervised Spatial Clustering

```python
from spatialized import UnsupervisedSpatialRandomForest, centers_from_mask

centers = centers_from_mask(np.isfinite(prediction_grid.values))

unsupervised = UnsupervisedSpatialRandomForest(
    n_estimators=500,
    random_state=42,
)
unsupervised.fit(
    [mag, gravity, radiometrics],
    centers[::10],
    rotations=True,
)

clusters = unsupervised.spectral_cluster(n_clusters=4)
embedding = unsupervised.mds(n_components=2)
distance = unsupervised.distance_
```

### Paper-Style Synthetic Experiment

The repository includes a synthetic workflow that mimics the structure of the
paper experiments without using restricted paper data:

```bash
python examples/paper_like_experiment.py --output-dir paper_like_outputs
```

It creates magnetic/gravity-like covariates, geology-like classes, SRF
classification outputs, entropy rasters, zone-of-influence rasters, and
unsupervised clusters.

Real Paper Author R scripts and data are also now available (gitignored,
not distributed in this repository) and have been used to validate this
Python implementation directly against the original R workflow's own
outputs — see `examples/validate_realcase2_cu_regression.py`,
`examples/validate_realcase_geology_classification.py`,
`examples/validate_unsupervised_srf_distance.py`, and `.features/PLAN.md`
for current validation status per workflow (what matches closely, what's
approximate and by how much, and what's still unvalidated).

## Repository Goal

**The primary goal of this repository is to reimplement the original R SRF
workflow in Python as faithfully as possible** — not just capture its
spirit, but reproduce its actual numerical behaviour where that's
achievable, and clearly document the gap where it isn't (e.g. the
unsupervised distance/clustering step, which is only moderately correlated
with R's own output — see `.features/PLAN.md`). Concretely that means:

1. Match the original workflow's mechanics — window sizes (including R's
   even, no-unique-center windows), rotation ordering, pattern flattening,
   missing-value handling, entropy formulas — not just a "spatial ML"
   workflow that produces plausible-looking outputs.
2. Validate against the original R code and Paper Author data whenever
   possible, using the R scripts' own saved or re-generated outputs as
   ground truth, rather than only checking internal consistency.
3. Where exact parity isn't practical (different underlying RF library
   internals, R algorithm details not fully documented), say so explicitly
   in code and docs rather than presenting an approximation as equivalent.

Once R parity is well-established, the package should also remain open to
additional spatial algorithms that share the same practical workflow (load
covariates, build spatial features, fit, predict over a grid, export
interpretable rasters and metadata) — but that is secondary to the
R-matching goal above.

Future overview documents can follow this same pattern: explain the method,
describe suitable data, show when to use it, include runnable code snippets,
and state validation status against the original R implementation.

# spatialized

`spatialized` implements data-preparation building blocks for spatial random
forests: extracting local spatial patterns around grid cells, flattening them into
predictor rows, and optionally adding 90/180/270 degree rotation variants.

The technique follows the vectorised spatial pattern preparation described in:

> Hassan Talebi, Luk J. M. Peeters, Alex Otto, Raimon Tolosana-Delgado (2022).
> A Truly Spatial Random Forests Algorithm for Geoscience Data Analysis and
> Modelling. Mathematical Geosciences 54, 1-22.
> https://doi.org/10.1007/s11004-021-09946-w

This implementation is based on the paper and an authorised review of the
original R code with permission from first author Hassan Talebi. The original
source material and paper test data are not distributed in this repository.

## Current scope

The current implementation is focused on vectorised spatial data preparation. It
also includes thin classifier/regressor wrappers around scikit-learn random
forests. It does not yet read/write GeoTIFFs directly.

Install modelling support with:

```bash
pip install "spatialized[model]"
```

```python
import numpy as np

from spatialized import SpatialLayer, prepare_patterns

mag = SpatialLayer("mag", np.arange(25, dtype=float).reshape(5, 5), window_size=3)
grav = SpatialLayer("grav", np.arange(100, 125, dtype=float).reshape(5, 5), window_size=3)

patterns = prepare_patterns([mag, grav], centers=[(2, 2)], rotations=True)

# 4 rows: original, 90, 180, and 270 degree variants.
# 18 columns: 9 mag cells followed by 9 gravity cells.
print(patterns.shape)
```

For larger prediction grids, build centers from a boolean mask and prepare them in
chunks:

```python
from spatialized import centers_from_mask, grid_from_centers, iter_pattern_batches

centers = centers_from_mask(np.isnan(prediction_grid))

for batch in iter_pattern_batches([mag, grav], centers, chunk_size=10_000):
    print(batch.centers.shape, batch.patterns.shape)

predicted_grid = grid_from_centers(prediction_grid.shape, centers, predicted_classes)
```

For supervised training data, responses are repeated to match rotation-augmented
rows:

```python
from spatialized import prepare_training_data

dataset = prepare_training_data(
    [mag, grav],
    centers=[(2, 2)],
    target=["class-a"],
    rotations=True,
)

print(dataset.patterns.shape, dataset.target)
```

Raster metadata can be adapted from common north-up transform formats:

```python
from spatialized import GridTransform

prediction_transform = GridTransform.from_gdal((500000, 25, 0, 7000000, 0, -25))
```

Train a spatial random forest classifier from vectorised patterns:

```python
from spatialized import SpatialRandomForestClassifier

model = SpatialRandomForestClassifier(n_estimators=500, random_state=42)
model.fit([mag, grav], centers=[(2, 2)], target=["class-a"], rotations=True)

classes = model.predict([mag, grav], centers=[(2, 2)])
entropy = model.entropy([mag, grav], centers=[(2, 2)])

for batch in model.iter_predict([mag, grav], centers, chunk_size=10_000, entropy=True):
    print(batch.centers.shape, batch.prediction.shape, batch.entropy.shape)
```

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
does not yet train a random forest model or read/write GeoTIFFs directly.

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
from spatialized import centers_from_mask, iter_pattern_batches

centers = centers_from_mask(np.isnan(prediction_grid))

for batch in iter_pattern_batches([mag, grav], centers, chunk_size=10_000):
    print(batch.centers.shape, batch.patterns.shape)
```

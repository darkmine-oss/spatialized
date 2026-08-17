---
name: spatial-data-prep
description: Use when loading GeoTIFF raster layers, building SpatialLayer objects and vectorised spatial patterns, choosing window sizes (odd or even) or training centers, handling categorical/missing values, or exporting feature-layout metadata. Prerequisite step before spatial-classify, spatial-regress, spatial-cluster, or spatial-deep-cluster.
---

# Spatial Data Prep

Every spatialized workflow starts here: load rasters into `SpatialLayer`
objects, decide window sizes and training centers, and (optionally) export
feature-layout metadata before fitting any model. This skill covers that
common first step; see the sibling skills for what to do with prepared data.

## Load rasters into layers

```python
from spatialized import read_spatial_layer

layers = [
    read_spatial_layer("mag_rtp.tif", name="mag_rtp", window_size=7),
    read_spatial_layer("gravity.tif", name="gravity", window_size=5),
]
```

For a cropped sub-extent (e.g. matching a specific study area rather than a
whole file), pass `window=(col_off, row_off, width, height)` or
`bounds=(left, bottom, right, top)`:

```python
layers = [
    read_spatial_layer("mag_rtp.tif", name="mag_rtp", window_size=7,
                        bounds=(422000, 7606000, 450000, 7634000)),
]
```

Reading a raster's raw values/metadata without wrapping it in a layer:
`read_raster(path, masked=True)` returns a `RasterGrid`.

## Choosing window_size: odd or even

`SpatialLayer.window_size` accepts **both odd and even** integers — this
matters because the original R workflows this package matches use both,
depending on the script (odd for some unsupervised patterns, even 10x10 for
the NWMP geology/geochemistry case studies). There is no "safer default":
match whatever the source workflow you're reproducing actually used.

- Odd sizes have a unique center pixel (symmetric window).
- Even sizes have no center pixel: `(window_size - 1) // 2` cells are taken
  before the target pixel, the rest after. This convention was verified
  **exactly** against real recovered R output (see
  `tests/test_patterns.py::test_even_window_matches_recovered_r_pattern_exactly`
  and `.features/PLAN.md`) — do not assume a symmetric split for even sizes.

The same before/after convention applies to `patch_size` in the deep-feature
workflow (`spatial-deep-cluster`), for consistency.

## Training centers

- `centers_from_mask(mask)` — every `True` cell in a boolean mask, row-major.
- `centers_from_shape(shape)` — every cell in a grid.
- For stratified/per-class sampling (matching how the R case studies drew
  training samples), build indices manually per class and stack them — see
  `examples/validate_realcase_geology_classification.py`'s `mysam.csv`
  loading logic for a real worked example of loading a fixed, R-compatible
  sample.

## Rotation augmentation

Pass `rotations=True` to `.fit(...)` (all model classes) or
`prepare_patterns`/`prepare_training_data` directly. This quadruples the
row count: each center contributes 4 rows (0/90/180/270 degrees), consecutive
per center. This matches the original R workflows' rotation-invariance
augmentation exactly (`myrot` applied 3 times, `rbind`'d).

## Missing values and categorical layers

Pass `encoder_kwargs` to any model constructor to control `PatternEncoder`:

- `numeric_missing_strategy`: `"constant"` (default), `"mean"` (global
  per-column), or `"window_mean"` (spatially-local — fills from the other
  valid cells in the same window/row, falling back to the global mean only
  when a whole window is missing; closer to, but still coarser than, R's
  `randomForestSRC(na.action="na.impute")`).
- `categorical_missing_strategy`: `"constant"` or `"most_frequent"`.

Categorical/object-dtype layer arrays are supported directly — no manual
one-hot encoding needed.

## Feature layout metadata

For auditability (tracing any model input column back to its source layer,
window position, and rotation), export the layout:

```bash
spatialized feature-layout \
  --layer mag_rtp:7 \
  --layer gravity:5:0,12,24 \
  --rotations \
  --output feature-layout.json
```

Or in Python: `feature_layout(layers, rotations=True).to_dict()`. A fitted
model's own layout (validated against its actual feature count) is available
via `model.feature_layout(layers)`.

## Validating against the original R workflow

If you're checking a data-prep step against real R output, `original_source/`
(gitignored, not distributed) holds recovered ground truth for several
workflows — see `.features/PLAN.md` for what's available and
`examples/validate_realcase2_cu_regression.py` /
`examples/validate_realcase_geology_classification.py` for worked
comparison scripts.

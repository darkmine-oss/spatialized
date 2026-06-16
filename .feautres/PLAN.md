# Spatialized Implementation Plan

## Remaining Work

### Validate Against Paper/Test Data

Status: synthetic paper-style workflow exists in
`examples/paper_like_experiment.py`; validation against actual Paper Author data
is still pending.

When the Paper Author provides the authorised paper/test data, compare the Python
implementation against the original workflow:

- pattern matrices
- rotation ordering
- sparse pattern selection
- multi-resolution layer alignment
- distance/proximity matrices
- clusters and embeddings
- prediction rasters
- entropy rasters

These checks should become regression tests or reproducible validation notebooks.

### Feature Importance and Zone of Influence

Status: implemented for fitted scikit-learn-backed spatialized model wrappers.
Feature scores can be mapped back to each layer's local spatial pattern window.

Required outputs:

- per-feature importance
- per-layer importance arrays
- zone-of-influence grids for each regionalised variable
- support for sparse pattern indices
- support for multi-layer feature offsets

### Improve Unsupervised SRF Parity

The current unsupervised implementation is a practical scikit-learn analogue of
the original `randomForestSRC(distance = "all")` workflow. It trains a forest to
separate real vectorised spatial patterns from synthetic shuffled patterns, then
derives center-level distances from tree leaf co-occurrence.

Remaining work:

- confirm distance behaviour against Paper Author test data
- compare cluster stability against the original R workflow
- decide whether a closer randomForestSRC-compatible backend is needed
- document where the implementation is exact versus approximate

### Categorical Raster Handling

Status: implemented for model wrappers via `PatternEncoder`.

Pattern preparation supports categorical/object arrays, but scikit-learn models
need numeric features.

Add:

- categorical layer encoding
- category mapping metadata
- consistent handling during prediction
- support for writing categorical prediction outputs with class metadata where
  practical

### Missing-Value Strategy

Status: partially implemented via `PatternEncoder`.

The original R workflow relies on random forest imputation. The current Python
implementation preserves missing values during pattern preparation and encodes
them before fitting/prediction.

Implemented:

- configurable numeric imputation
- categorical missing handling
- model-level validation before fitting
- documentation of defaults and tradeoffs

Remaining work:

- missingness indicators if useful
- validation against Paper Author data

### Feature Layout Metadata

Status: partially implemented. A formal feature layout object now lets every
model input column be traced back to its source.

Metadata should include:

- layer name
- layer order
- window row and column
- flattened feature index
- sparse source index, when used
- rotation augmentation policy
- total feature offsets per layer

This is required for auditability and feature-importance reconstruction.

Remaining work:

- persist layout metadata beside trained models and raster outputs
- include categorical encoding metadata in exported model metadata
- include missing-value strategy metadata once implemented
- add metadata validation against Paper Author test data

### CLI and Examples

Status: partially implemented. A CLI entrypoint now supports feature-layout JSON
export and chunked full-grid prediction from a pickled fitted model. Synthetic
end-to-end and paper-style examples have been added under `examples/`.

Continue adding command-line or script workflows for common use cases:

- read GeoTIFF layers
- prepare training samples
- train classifier/regressor
- run full-grid prediction
- write prediction, probability, and entropy GeoTIFFs
- run unsupervised clustering

Examples should use synthetic data until Paper Author test data is available.

### Agent Skills Workflows

Status: initial `skills/spatialized-workflow/` skill added.

Continue improving the agent `skills/` directory so Codex or other local agents
can call the different spatialized workflow steps safely and consistently.

Skills should cover:

- feature layout metadata export
- GeoTIFF layer inspection
- vectorised pattern preparation
- supervised classifier/regressor training
- full-grid prediction and GeoTIFF writing
- feature importance and zone-of-influence export
- categorical raster handling checks
- unsupervised SRF clustering
- end-to-end workflow orchestration from source rasters to outputs

Each skill should document required inputs, expected outputs, validation checks,
and which CLI/API calls it uses.

### Packaging and CI Polish

Status: partially implemented. Project classifiers, URLs, CLI entry point,
`full`/`dev` extras, GitHub Actions test/build workflow, and local build
validation have been added.

Improve release readiness:

- lint/format configuration
- source distribution and wheel builds verified locally with `twine check`

### Versioned Packages for Pip and PyPI Upload

Prepare versioned packages for pip installation and publishing to PyPI.

Required work:

- define a versioning policy for public releases
- verify `pip install spatialized` from published package
- build versioned artifacts with `python -m build` (local check passes)
- validate artifacts with `twine check` (local check passes)
- configure trusted publishing or PyPI API token handling
- upload releases to PyPI
- add release notes for each published version
- document install extras such as `spatialized[model]`, `spatialized[raster]`,
  and any future `spatialized[full]`

### Performance Testing

Benchmark and tune large-raster workflows.

Areas to measure:

- chunk size versus memory use
- `sliding_window_view` behaviour for large grids
- training matrix size under rotation augmentation
- random forest fit time
- unsupervised distance matrix memory cost
- raster write performance

### Documentation

Expand the README into fuller API documentation.

Documentation should explain:

- paper provenance and permission context
- vectorised spatial pattern preparation
- rotation augmentation
- multi-resolution transform mapping
- supervised classifier/regressor workflows
- full-grid prediction workflows
- unsupervised workflow and its current approximation status
- missing value and categorical-data handling
- validation status once Paper Author data is available

# Spatialized Implementation Plan

## Remaining Work

### Validate Against Paper/Test Data

Status: Paper Author data has arrived, in `original_source/` (gitignored, not
distributed). One real case is now validated end to end; several others are
blocked on missing reference outputs rather than missing input data.

Validated:

- **Cu geochemistry regression (`original_source/code/realCase2/`)** — fully
  validated in `examples/validate_realcase2_cu_regression.py`. Python's
  `SpatialRandomForestRegressor` reproduces the paper's qualitative finding
  (SRF clearly beats classical non-spatial RF); OOB R² (~0.62-0.64 depending
  on tree count and imputation) tracks the paper's reported 0.684 reasonably
  closely, and full-grid predictions correlate r≈0.85-0.89 with the R
  script's saved `gheochemcuSRF.tif`. This validation drove three real fixes:
  `SpatialLayer` now supports the R script's true even `window_size` (10x10,
  not an odd approximation); the even-window offset itself was then found
  and fixed to be **mirrored from R's actual convention** (Python took 5
  cells before the target pixel and 4 after; confirmed via a real recovered
  R pattern that R actually takes 4 before and 5 after — see
  `test_even_window_matches_recovered_r_pattern_exactly` in
  `tests/test_patterns.py`, which pins this against real data and is the
  authoritative check going forward); and `PatternEncoder` gained a
  `window_mean` imputation strategy that is closer to spatially-local than
  the previous global-constant/global-mean fill (still coarser than R's
  `randomForestSRC(na.action="na.impute")`, which imputes iteratively during
  tree growth). After the window-offset fix, rerun at `window_size=10`,
  `ntree=1000`: OOB R²=0.621, full-grid predictions r=0.885/R²=0.687 vs R's
  saved `gheochemcuSRF.tif` (`original_source/code/realCase2/gheochemcu_python_srf_fixed.tif`,
  `python_validation_summary_fixed.json`) — consistent with the pre-fix
  numbers within run-to-run noise (neither R nor Python fixes a training
  seed here), which is expected: the fix corrects per-cell window alignment,
  not aggregate model accuracy. The fix's correctness is proven by the exact
  byte-level match in `test_even_window_matches_recovered_r_pattern_exactly`,
  not by this R² movement.

Still blocked, and why (not a missing-data problem, a missing-reference-output
problem):

- **Geology classification (`original_source/code/realCase/`)** — RESOLVED.
  The R script's own spatial-SRF prediction raster (`channelClass.tiff`) was
  never saved to disk by the original authors; only the non-spatial
  baseline's outputs (`RFLithiPred.tif`/`RFLithiEntro.tif`/
  `RFLithimismatch.tif`) survived, plus the training sample (`mysam.csv`)
  and a confusion-matrix summary (`ctsp.csv`) for the spatial run.
  `code/realCase/.RDatanonspatial.RData` (30MB) turned out to be the
  *non-spatial* session cache (confirmed via `raster`/R after installing R
  — its `patbase_train` has only 8 columns, matching the non-spatial
  workflow's raw-pixel features, not the 700-column windowed spatial
  pattern), so it didn't help. With an actual R install (Homebrew `r`
  4.6.1), re-ran the real spatial `NWMP.R` workflow directly — patched only
  to (a) reuse the already-saved `mysam.csv` training sample instead of a
  fresh `sample()` call, for direct comparability with the saved `ctsp.csv`,
  and (b) parallelise the per-cell pattern-extraction loop across cores via
  `parallel::mclapply` (no algorithmic change, same `crop`/`extend` calls
  per cell). Recovered outputs are saved under
  `original_source/code/realCase/recovered_from_RData/`
  (`channelClass_recovered.tif`, `channelprob_entropy_recovered.tif`,
  `channelmismatch_recovered.tif`, `ctsp_recovered.csv`,
  `patbase_train_spatial.csv`, and the `run_NWMP_recover.R` script used).
  Overall classification accuracy from the rerun (77.77%) matches the
  original saved `ctsp.csv` summary (77.89%) almost exactly — the small gap
  is fully explained by `randomForestSRC` never being seeded in the
  original script. This is now a genuine, usable reference for validating
  the Python spatial classifier against, the same way
  `examples/validate_realcase2_cu_regression.py` does for the Cu case.
  Python-side validation is done in
  `examples/validate_realcase_geology_classification.py`: fits
  `SpatialRandomForestClassifier` on the same `mysam.csv` training sample,
  `window_size=10`, `ntree=1000`. Result: Python's full-grid accuracy
  against the true geomask is **77.13%**, matching R's 77.77% within 0.64
  points, and **92.4%** of individual cell predictions agree directly
  between Python and R. Outputs saved alongside the R reference in
  `original_source/code/realCase/recovered_from_RData/`
  (`channelClass_python_srf.tif`, `python_validation_summary.json`). After
  the window-offset fix, rerun: full-grid accuracy **78.52%**, class
  agreement **92.35%** (`channelClass_python_srf_fixed.tif`,
  `python_validation_summary_fixed.json`) — again consistent within
  run-to-run noise, same reasoning as the Cu case above. Both real cases
  (Cu regression and geology classification) are now validated end to end.
- **Unsupervised SRF / deep-feature clustering
  (`original_source/SRFGeoTIFF.R`, matching top-level rasters and
  `MinExP9_unsupervised_potential_field_modelling2.docx`)** — algorithm and
  matching input rasters are present, but the script's own final outputs
  (`srfPredictRasterHighRes8clust.tif` and its entropy raster) were never
  saved either. Attempted recovery from `code/.Rhistory.RData` (7.3MB) —
  this one *does* parse with the pure-Python `rdata` library, but
  cross-checking its cached `patbase` pattern matrix against its own cached
  `inputData_raster` cell-by-cell showed the two are **not mutually
  consistent** (only edge/corner rows matched, coincidentally, due to NaN
  padding overlap) — the autosaved session was very likely re-run multiple
  times with fresh random data without regenerating every cached object, so
  it cannot be trusted as numeric ground truth. No reliable reference
  recovered from this file.

  Full-scale recovery (predicting the unsupervised domain classifier over
  every one of the ~11.3M cells in the RTP/1VD rasters) is still **not
  practical** — extrapolating from the NWMP full-grid prediction time, that's
  on the order of a full day of compute, not attempted.

  The bounded version (distance matrix + clustering only, skipping the
  full-grid domain raster) IS done: re-ran the training half of
  `SRFGeoTIFF.R` directly in R — sample 1000 cells, build rotation-augmented
  patterns, `rfsrc(distance="all", ntree=100)`, collapse to a 1000x1000
  distance matrix, PAM clustering with a silhouette sweep over k=2..20. Took
  under a minute total (patterns: 9s, forest fit: 20s) — much cheaper than
  expected, because the real per-layer window sizes here are **7x7 (RTP/1VD)
  and 5x5 (gravity)**, not the 19x19/12x12 the docx described (worth noting
  as a docx/actual-code discrepancy, same pattern as the `ntree` mismatches
  flagged elsewhere). No `set.seed()` in the original script and no prior run
  was ever saved to reproduce, so this is a fresh, seeded (`20260816`)
  ground-truth generation, not a reproduction of a specific lost run.
  Outputs saved to `original_source/recovered_from_RData/`:
  `patbase_train.csv` (4000 rows x 123 cols — the exact vectorised,
  rotation-augmented patterns, useful for validating
  `prepare_patterns`/`vectorize_layer` directly), `mySamp.csv` (which cells,
  with coordinates), `myDist.csv` (the 1000x1000 collapsed distance matrix),
  `silhouette_scores.csv` (k=2..20 sweep — best score at k=4, 0.138, matching
  the original script's own choice of `pam(..., 4, ...)` as its baseline
  clustering before the spectral k=8 step), `pam_k4_clusters.csv`, and the
  `run_SRFGeoTIFF_bounded.R` script used. Python-side validation against
  this (comparing `UnsupervisedSpatialRandomForest`'s distance/clustering
  behaviour to `myDist.csv`/`pam_k4_clusters.csv`) has not been written yet.

These checks should become regression tests or reproducible validation notebooks
as more reference outputs become available (either by re-running the R scripts
directly, or with a real R install to load the `.RData` caches).

### Ferricrete / Paleovalley Gold Targeting Workflow

Status: planned.

Implement a workflow inspired by the ferricrete/paleovalley paper:

- train on a labelled ferricrete/inset-valley area
- predict target units into one or more transfer areas
- export target probability/class rasters
- export entropy/uncertainty rasters
- export feature-importance and zone-of-influence rasters
- support high-resolution magnetic layers such as RTP, high-frequency residual,
  and 1VD

Data to gather:

- high-resolution aeromagnetic grids for at least one training area
- matching labelled ferricrete/inset-valley polygons or mask rasters
- one or more target areas with equivalent aeromagnetic processing
- optional validation polygons for transfer areas
- analogue datasets if the paper data cannot be distributed

### Magnetic Preprocessing Recipes

Status: planned. The package should not initially reimplement specialist
geophysical transformations, but it should document expected layers and provide
workflow hooks.

Recipes to document/support:

- TMI gridded at consistent cell size
- continuation to common survey height
- RTP or other regional magnetic correction as externally prepared input
- high-frequency residual: RTP minus upward-continued RTP
- 1VD of the high-frequency residual

Data to gather:

- processed magnetic layer examples
- metadata describing survey height, line spacing, cell size, and processing
  sequence
- raw and processed pairs if we later implement preprocessing checks

### Unsupervised Domain Diagnostics and Prediction

Status: planned.

Extend the unsupervised SRF workflow with:

- silhouette scores across candidate cluster counts
- eigengap analysis
- sampled-domain clustering
- supervised domain classifier trained from unsupervised cluster labels
- full-grid domain prediction
- domain entropy/uncertainty raster

Reference R silhouette scores (k=2..20) over the real Eastern Yilgarn
potential-field data are now available at
`original_source/recovered_from_RData/silhouette_scores.csv` (see
"Improve Unsupervised SRF Parity" below) — best score at k=4 (0.138), usable
as a target for `cluster_diagnostics`'s own silhouette sweep once it's
pointed at the same sampled-cell distance matrix (`myDist.csv`). Eigengap
analysis and full-grid prediction remain out of scope for now (the latter is
a ~11.3M-cell, full-day-of-compute job in R too, per the note above).

Data to gather:

- potential-field rasters for a coherent area — have real data now
  (`original_source/magmap_v7_2019_RTP_clip.tif`,
  `magmap_v7_2019_1VD_clip.tif`, `residual160km_eastern_yilgarn.tif`)
- interpreted geology boundaries for qualitative comparison — still missing
- optional expert domain labels for validation — still missing

### Deep Feature Potential-Field Clustering

Status: implemented end to end and verified working (real `torch`/
`torchvision` installed and exercised, not just written against). All steps
from the second workflow in the unsupervised potential-field modelling
document:

- normalize potential-field rasters into a 3-channel image (`normalize_channels`)
- extract sliding image patches, **true 16x16 as the docx specifies**
  (`patch_centers`/`extract_patches` — both now support even patch sizes,
  not just odd, using the same before/after split convention as the
  `SpatialLayer` even-window fix above, for consistency across the package)
- pretrained ResNet50 feature extractor with its final layer removed
  (`extract_resnet_features`) — a 16x16 input does survive ResNet50's
  strided conv stack (spatial map collapses to 1x1 by the last stage;
  adaptive average pooling handles this), matching the docx's own reasoning
  for choosing 16x16 over 224x224 (avoiding edge/off-center effects in dense
  sliding-window extraction)
- reduce CNN features with UMAP, PCA fallback if `umap-learn` isn't
  installed (`cluster_feature_vectors`)
- concatenate point intensity values (`point_intensities_at_centers`, wired
  into `cluster_feature_vectors` via a new `point_intensities` parameter,
  concatenated *after* dimensionality reduction per the docx's ordering)
- cluster with k-means (`cluster_feature_vectors`)
- map clusters back to raster space (`label_grid` on the result)

Verified end to end: 16x16 patches -> ResNet50 (2048-dim) -> UMAP (6-dim) ->
concatenate 3 raw channel intensities (9-dim) -> k-means(k=8) -> label grid,
on synthetic data with real `torch`/`torchvision` installed (not mocked).
8/9 `tests/test_deep_features.py` tests exercise this without skipping when
torch is present; 2 skip cleanly when it isn't (it's an optional `deep`
extra, matching the existing package convention).

Not yet done:

- no reference R/docx output exists to numerically validate against (this
  workflow was only ever described in the docx, not run and saved anywhere
  in `original_source/`) — unlike the two NWMP cases and the unsupervised
  SRF distance matrix, there is no ground truth to check pixel-for-pixel or
  cluster-for-cluster; the docx's own figures are illustrative images only
- no example script demonstrating this on the real Eastern Yilgarn rasters
  (`original_source/magmap_v7_2019_*`) — the verification above used
  synthetic data for speed; a real-raster run would need the same chunked/
  sliding-window treatment the NWMP full-grid predictions use, since the
  real rasters are ~11.3M cells

Data to gather:

- RTP, 1VD, and gravity rasters over the same area — have real data now
  (`original_source/magmap_v7_2019_RTP_clip.tif`,
  `magmap_v7_2019_1VD_clip.tif`, `residual160km_eastern_yilgarn.tif`)
- interpreted geology or structural boundaries for qualitative comparison —
  still missing

Dependency notes:

- `deep` extra now includes `torch>=2.0` and `torchvision>=0.15` alongside
  `umap-learn>=0.5` (`pyproject.toml`)

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

Status: CLOSED (decision: accept as documented limitation, not pursuing
further). The current
unsupervised implementation is a scikit-learn analogue of the original
`randomForestSRC(distance = "all")` workflow. It trains a classifier to
separate real vectorised spatial patterns from synthetic patterns generated
by independently permuting each feature column, then derives center-level
distances from tree leaf co-occurrence, collapsed across the 4 rotation
variants via `min` — both details match how `randomForestSRC` documents its
own internal unsupervised mechanism and how `SRFGeoTIFF.R` collapses its own
rotation blocks (`apply(array(...), 3, min)`), so this is not an invented
simplification, it is the same real-vs-synthetic-via-marginal-permutation
technique made explicit.

**Empirically tested** against real data: `examples/validate_unsupervised_srf_distance.py`
feeds R's own recovered patterns (`original_source/recovered_from_RData/patbase_train.csv`,
1000 samples x 4 rotations x 123 features from the real `SRFGeoTIFF.R`
run — see "Unsupervised SRF / deep-feature clustering" above) through
`UnsupervisedSpatialRandomForest.fit_prepared` (added specifically to support
this kind of validation — feeds pre-vectorised patterns in directly, bypassing
raster extraction) and compares the resulting distance matrix and clustering
against R's own `myDist.csv`/`pam_k4_clusters.csv`. Result:

- Distance matrix Pearson correlation: **~0.35** (Spearman: ~0.36)
- PAM(k=4)-equivalent clustering agreement (adjusted Rand index): **~0.29**

This is a real, moderate, but not close, match — the two approaches are
directionally related (not uncorrelated, not a bug in the permutation
mechanism, which this result confirms is structurally sound) but not
numerically interchangeable. The gap is most likely `randomForestSRC`'s
internal splitting/distance computation differing from scikit-learn's
`RandomForestClassifier`, not the synthetic-generation or rotation-collapse
logic (both confirmed structurally correct against the R script's own code).

**R-vs-R noise baseline** (resolves the open question above): re-ran
`rfsrc(distance="all")` a second time on the exact same `patbase_train.csv`
patterns, with a fresh independent forest fit (no `set.seed()`, matching the
original script) — script and outputs at
`original_source/recovered_from_RData/rerun_srf_distance_r_vs_r.R`,
`myDist_rerun2.csv`, `pam_k4_clusters_rerun2.csv`. Result: **R vs R itself is
highly reproducible** (distance Pearson r=0.993, Spearman r=0.991, PAM(k=4)
adjusted Rand index=0.885) — night and day versus the Python-vs-R numbers
above. This settles the question: the ~0.35 correlation / ~0.29 ARI gap
between Python and R is **not run-to-run randomness, it is a real
algorithmic difference** between `randomForestSRC`'s internal unsupervised
splitting/distance computation and scikit-learn's `RandomForestClassifier`.

**Decision (made after seeing this result): do not pursue a closer
`randomForestSRC`-compatible backend.** Reimplementing R's actual internal
unsupervised splitting rule would mean porting `randomForestSRC`'s C-level
mechanics, a substantial undertaking with no guaranteed payoff. The
sklearn-based approximation stays as the implementation, clearly documented
(here, in the `unsupervised.py` docstring, and in the README) as
"directionally similar, not numerically interchangeable" with R's own
output. This item is closed; revisit only if a specific downstream use case
demands tighter numeric parity.

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

- configurable numeric imputation: `"constant"`, `"mean"` (global per-column),
  and `"window_mean"` (spatially-local: fills a missing cell from the other
  valid cells in the same window/row, falling back to the global column mean
  only when a whole window is missing; auto-wires the required `layer_spans`
  from `.fit(layers, ...)`)
- categorical missing handling
- model-level validation before fitting
- documentation of defaults and tradeoffs

Remaining work:

- missingness indicators if useful
- `window_mean` is still coarser than R's `randomForestSRC`
  `na.action="na.impute"`, which imputes iteratively during tree growth
  rather than from a single local/global statistic; a closer analogue would
  need genuine multiple-imputation-during-fit, which sklearn's
  `RandomForestRegressor`/`Classifier` don't support natively
- quantified against Paper Author data for the Cu regression case
  (`examples/validate_realcase2_cu_regression.py --imputation window_mean`
  vs `--imputation mean`, same tree count: OOB R² 0.617 vs 0.614, full-grid
  R² vs R's saved output 0.575 vs 0.554 — real but modest improvement, not
  yet validated at full 1000-tree scale); not yet checked against the other
  cases

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
end-to-end and paper-style examples have been added under `examples/`, plus
`examples/validate_realcase2_cu_regression.py`, which now uses real Paper
Author data (`original_source/`, gitignored) rather than synthetic data — see
"Validate Against Paper/Test Data" above.

Continue adding command-line or script workflows for common use cases:

- read GeoTIFF layers
- prepare training samples
- train classifier/regressor
- run full-grid prediction
- write prediction, probability, and entropy GeoTIFFs
- run unsupervised clustering

Examples should use synthetic data until Paper Author test data is available.

### Agent Skills Workflows

Status: decomposed into one data-prep skill plus one skill per downstream
workflow, replacing the earlier single `skills/spatialized-workflow/` skill
(removed — its content was migrated, not duplicated):

- `skills/spatial-data-prep/` — GeoTIFF loading, `SpatialLayer` construction,
  odd/even window sizing (with the R-verified even-offset convention),
  training centers, rotation augmentation, categorical/missing-value
  handling, feature-layout metadata export. Prerequisite for the rest.
- `skills/spatial-classify/` — `SpatialRandomForestClassifier`, OOB accuracy,
  full-grid prediction with entropy, feature importance / zone of influence,
  CLI `predict-grid`, pointer to the real geology-classification validation.
- `skills/spatial-regress/` — `SpatialRandomForestRegressor`, OOB R²,
  `regression_r2_score`/`regression_residuals`, full-grid prediction,
  feature importance, pointer to the real Cu-regression validation.
- `skills/spatial-cluster/` — `UnsupervisedSpatialRandomForest`, distance
  matrix, spectral clustering, silhouette/eigengap diagnostics, MDS,
  `fit_prepared` for R-validation use cases, and the documented parity
  limitation stated up front (moderate correlation only, not close — see
  "Improve Unsupervised SRF Parity" above).
- `skills/spatial-deep-cluster/` — the ResNet50 deep-feature clustering
  workflow implemented this session: channel normalisation, even-size patch
  extraction, `extract_resnet_features`, UMAP reduction,
  `point_intensities_at_centers` concatenation, k-means, label-grid mapping;
  notes the `deep` extra requirement and that no R ground truth exists to
  validate this specific workflow against.
- `skills/setup/` — unchanged, cross-reference updated to point at the new
  skill names instead of the removed one.

Remaining work:

- categorical raster handling checks and end-to-end orchestration (source
  rasters through to final outputs) aren't yet their own skill sections —
  currently folded into `spatial-data-prep`/`spatial-classify`/
  `spatial-regress` rather than called out separately; revisit if agents
  need a single "run the whole pipeline" entry point beyond the individual
  workflow skills
- none of the five skills have been exercised end-to-end by an actual agent
  session yet (only written and cross-checked against real function
  signatures) — worth a dry run

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

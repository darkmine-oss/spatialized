---
name: spatial-deep-cluster
description: Use when clustering potential-field rasters with a pretrained CNN (ResNet50) feature extractor instead of spatial random forest patterns -- normalizing channels, extracting sliding-window patches, extracting CNN features, reducing with UMAP, concatenating point intensities, and mapping k-means clusters back to raster space.
---

# Spatial Deep Cluster

The deep-feature alternative to `spatial-cluster`: instead of vectorised
spatial-forest patterns, this clusters potential-field data using a
pretrained ResNet50 as a feature extractor. Matches the second workflow
described in the unsupervised potential-field modelling document. This is
independent of `spatial-data-prep` (no `SpatialLayer`/patterns involved) --
it works directly on stacked raster arrays.

## Requires the `deep` extra

```bash
pip install "spatialized[deep]"   # torch, torchvision, umap-learn
```

Without `torch`/`torchvision` installed, `extract_resnet_features` raises a
clear `ImportError` telling you to install the extra. `cluster_feature_vectors`
falls back to PCA if `umap-learn` isn't installed (no hard failure).

## Full pipeline

```python
import numpy as np
from spatialized import (
    normalize_channels, patch_centers, extract_patches,
    extract_resnet_features, point_intensities_at_centers,
    cluster_feature_vectors,
)

# 1. Stack same-shaped rasters into a normalized 0..1 band-last image.
image = normalize_channels([rtp, one_vd, gravity])   # each 2D, same shape

# 2. Sliding-window patch centers. 16x16 (even!) is the documented choice --
#    deliberately smaller than the usual 224x224, to avoid edge/off-center
#    effects in dense sliding-window extraction. See spatial-data-prep for
#    the even-size before/after convention (same one used here).
centers = patch_centers(image.shape[:2], patch_size=16, stride=8)
patches = extract_patches(image, centers, patch_size=16)

# 3. Pretrained ResNet50, final layer removed -> 2048-dim feature per patch.
cnn_features = extract_resnet_features(patches, device="cpu", batch_size=64)

# 4. Reduce (UMAP to 6-dim, matching the doc) and concatenate the original
#    3-channel point intensities at each center -> 9-dim combined space.
intensities = point_intensities_at_centers(image, centers)
result = cluster_feature_vectors(
    cnn_features, centers,
    output_shape=image.shape[:2],
    n_clusters=8, n_components=6, random_state=42,
    point_intensities=intensities,
)

result.label_grid   # cluster labels mapped back to raster space, -1 = unsampled
```

## Scale note

The real Eastern Yilgarn rasters this workflow targets are ~11.3M cells.
Running this densely over the full extent needs the same chunking discipline
as full-grid SRF prediction (`spatial-classify`/`spatial-regress`) --
`extract_resnet_features`'s `batch_size` controls GPU/CPU memory per batch,
but you are still responsible for tiling `patch_centers`/`extract_patches`
over manageable chunks rather than materialising every patch at once for a
raster that size.

## No R ground truth exists for this workflow

Unlike `spatial-classify`/`spatial-regress`/`spatial-cluster`, this specific
workflow was only ever described in a docx document, never run and saved
anywhere in the recovered R materials. There's no reference output to
validate pixel-for-pixel or cluster-for-cluster against — treat correctness
here as "matches the documented method's steps and parameters exactly"
(verified: true 16x16 patches, ResNet50 minus final layer, UMAP to 6-dim,
concatenate 3-channel intensities, k-means), not "numerically validated
against a real R run" the way the other three workflows are.

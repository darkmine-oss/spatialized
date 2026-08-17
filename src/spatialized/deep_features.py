"""Optional deep-feature clustering helpers for potential-field rasters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DeepFeatureClusteringResult:
    """Clustered potential-field feature outputs."""

    embedding: np.ndarray
    labels: np.ndarray
    label_grid: np.ndarray


def normalize_channels(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Normalize same-shaped arrays to a band-last 0..1 image stack."""

    if not arrays:
        raise ValueError("at least one array is required")
    stack = np.stack([np.asarray(array, dtype=float) for array in arrays], axis=-1)
    if stack.ndim != 3:
        raise ValueError("arrays must be 2D")
    normalized = np.empty_like(stack, dtype=float)
    for band in range(stack.shape[2]):
        values = stack[:, :, band]
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        if max_value == min_value:
            normalized[:, :, band] = 0.0
        else:
            normalized[:, :, band] = (values - min_value) / (max_value - min_value)
    return normalized


def patch_centers(shape: tuple[int, int], patch_size: int, *, stride: int = 1) -> np.ndarray:
    """Return valid patch-center row/col indices for an image shape.

    ``patch_size`` may be odd or even. The deep-feature workflow described in
    the unsupervised potential-field modelling document deliberately uses
    16x16 (even) patches, not the usual odd, unique-center-pixel size -- see
    :func:`extract_patches` for the before/after split convention used.
    """

    before, _ = _patch_offsets(patch_size)
    if stride < 1:
        raise ValueError("stride must be positive")
    after = patch_size - 1 - before
    rows = np.arange(before, shape[0] - after, stride)
    cols = np.arange(before, shape[1] - after, stride)
    return np.array(np.meshgrid(rows, cols, indexing="ij")).reshape(2, -1).T


def extract_patches(
    image: np.ndarray,
    centers: np.ndarray,
    *,
    patch_size: int,
) -> np.ndarray:
    """Extract band-last image patches centered on row/col centers.

    For even ``patch_size`` there is no unique center pixel: following the
    same convention used for even ``SpatialLayer`` windows (see
    ``patterns._extract_windows``), ``(patch_size - 1) // 2`` cells are taken
    before the center and the rest after.
    """

    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError("image must be a 3D band-last array")
    before, after = _patch_offsets(patch_size)
    patches = []
    for row, col in np.asarray(centers, dtype=int):
        if (
            row - before < 0
            or row + after >= image_array.shape[0]
            or col - before < 0
            or col + after >= image_array.shape[1]
        ):
            raise ValueError("patch center falls outside valid patch area")
        patches.append(
            image_array[
                row - before : row + after + 1,
                col - before : col + after + 1,
                :,
            ]
        )
    return np.asarray(patches)


def _patch_offsets(patch_size: int) -> tuple[int, int]:
    if patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    before = (patch_size - 1) // 2
    after = patch_size - 1 - before
    return before, after


def cluster_feature_vectors(
    features: np.ndarray,
    centers: np.ndarray,
    *,
    output_shape: tuple[int, int],
    n_clusters: int,
    n_components: int = 6,
    random_state: int | None = None,
    point_intensities: np.ndarray | None = None,
) -> DeepFeatureClusteringResult:
    """Reduce feature vectors with UMAP if available, then cluster with k-means.

    ``point_intensities``, if given, is concatenated onto the UMAP/PCA
    embedding *after* dimensionality reduction (matching the documented
    deep-feature workflow: CNN features reduced to a small embedding, then
    the original per-pixel channel intensities at each patch center are
    appended before clustering -- e.g. a 6-dim embedding plus 3 raw channel
    values gives the 9-dim combined space the workflow clusters on). Use
    :func:`point_intensities_at_centers` to build this array from a
    normalized channel image and the same ``centers``.
    """

    from sklearn.cluster import KMeans

    feature_array = np.asarray(features, dtype=float)
    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array")
    intensity_array = None
    if point_intensities is not None:
        intensity_array = np.asarray(point_intensities, dtype=float)
        if intensity_array.shape[0] != feature_array.shape[0]:
            raise ValueError("point_intensities length must match features length")
    embedding = _umap_embedding(
        feature_array,
        n_components=n_components,
        random_state=random_state,
    )
    if intensity_array is not None:
        embedding = np.concatenate([embedding, intensity_array], axis=1)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(
        embedding
    )
    label_grid = np.full(output_shape, -1, dtype=int)
    center_array = np.asarray(centers, dtype=int)
    label_grid[center_array[:, 0], center_array[:, 1]] = labels
    return DeepFeatureClusteringResult(
        embedding=embedding,
        labels=labels,
        label_grid=label_grid,
    )


def point_intensities_at_centers(image: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return per-center raw channel values from a band-last image.

    Builds the "concatenate point intensity values" input for
    :func:`cluster_feature_vectors`: the original (e.g. normalized 3-channel)
    pixel values at each patch center, to be appended to the reduced CNN
    embedding before clustering.
    """

    image_array = np.asarray(image, dtype=float)
    if image_array.ndim != 3:
        raise ValueError("image must be a 3D band-last array")
    center_array = np.asarray(centers, dtype=int)
    return image_array[center_array[:, 0], center_array[:, 1], :]


def extract_resnet_features(
    patches: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract 2048-dim features from band-last image patches with ResNet50.

    Uses a pretrained ResNet50 with its final classification layer removed,
    matching the deep-feature potential-field clustering workflow: small
    patches (e.g. 16x16, not the usual 224x224) are used deliberately, to
    avoid the edge/off-center effects a full-size ImageNet patch would
    introduce in a dense sliding-window feature-extraction setting. A patch
    this small still round-trips through ResNet50's strided conv stack (the
    spatial map collapses to 1x1 by the last stage, which the final adaptive
    average pool handles regardless of size). Requires `torch` and
    `torchvision` (install `spatialized[deep]`).
    """

    torch, models = _import_torch()
    patch_array = np.asarray(patches, dtype=np.float32)
    if patch_array.ndim != 4 or patch_array.shape[-1] != 3:
        raise ValueError("patches must be a band-last array of 3-channel images")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    tensor = torch.from_numpy(patch_array).permute(0, 3, 1, 2).contiguous()
    features = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            features.append(model(batch).cpu().numpy())
    return np.concatenate(features, axis=0)


def _import_torch():
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise ImportError(
            "torch and torchvision are required for ResNet feature extraction; "
            "install spatialized[deep]"
        ) from exc
    return torch, models


def _umap_embedding(
    features: np.ndarray,
    *,
    n_components: int,
    random_state: int | None,
) -> np.ndarray:
    try:
        from umap import UMAP
    except ImportError:
        from sklearn.decomposition import PCA

        components = min(n_components, features.shape[0], features.shape[1])
        return PCA(n_components=components, random_state=random_state).fit_transform(features)
    return UMAP(n_components=n_components, random_state=random_state).fit_transform(features)

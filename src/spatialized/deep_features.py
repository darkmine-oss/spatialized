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
    """Return valid patch-center row/col indices for an image shape."""

    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer")
    if stride < 1:
        raise ValueError("stride must be positive")
    radius = patch_size // 2
    rows = np.arange(radius, shape[0] - radius, stride)
    cols = np.arange(radius, shape[1] - radius, stride)
    return np.array(np.meshgrid(rows, cols, indexing="ij")).reshape(2, -1).T


def extract_patches(
    image: np.ndarray,
    centers: np.ndarray,
    *,
    patch_size: int,
) -> np.ndarray:
    """Extract band-last image patches centered on row/col centers."""

    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError("image must be a 3D band-last array")
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer")
    radius = patch_size // 2
    patches = []
    for row, col in np.asarray(centers, dtype=int):
        if (
            row - radius < 0
            or row + radius >= image_array.shape[0]
            or col - radius < 0
            or col + radius >= image_array.shape[1]
        ):
            raise ValueError("patch center falls outside valid patch area")
        patches.append(
            image_array[
                row - radius : row + radius + 1,
                col - radius : col + radius + 1,
                :,
            ]
        )
    return np.asarray(patches)


def cluster_feature_vectors(
    features: np.ndarray,
    centers: np.ndarray,
    *,
    output_shape: tuple[int, int],
    n_clusters: int,
    n_components: int = 6,
    random_state: int | None = None,
) -> DeepFeatureClusteringResult:
    """Reduce feature vectors with UMAP if available, then cluster with k-means."""

    from sklearn.cluster import KMeans

    feature_array = np.asarray(features, dtype=float)
    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array")
    embedding = _umap_embedding(
        feature_array,
        n_components=n_components,
        random_state=random_state,
    )
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

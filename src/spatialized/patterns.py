"""Vectorised spatial pattern preparation.

The core operation is the "vectorised spatial patterns" preprocessing step from
Talebi et al. (2022): extract a local window around each target location for each
regionalised variable, flatten each window row-wise, and concatenate variables into
a single predictor row. Optional rotation augmentation appends 90, 180, and 270
degree rotations of each multivariate pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


Center = tuple[int, int]


@dataclass(frozen=True)
class GridTransform:
    """Affine metadata for a north-up regular grid.

    Parameters are expressed at the outer grid bounds, matching common raster
    conventions. Cell centers are located at ``left + (col + 0.5) * x_size`` and
    ``top - (row + 0.5) * y_size``.
    """

    left: float
    top: float
    x_size: float
    y_size: float | None = None

    def __post_init__(self) -> None:
        y_size = self.x_size if self.y_size is None else self.y_size
        if self.x_size <= 0 or y_size <= 0:
            raise ValueError("cell sizes must be positive")
        object.__setattr__(self, "y_size", y_size)

    @classmethod
    def from_gdal(cls, geotransform: Sequence[float]) -> "GridTransform":
        """Create a transform from a GDAL geotransform tuple.

        Only north-up grids are supported: x rotation and y rotation must be zero,
        x pixel size must be positive, and y pixel size must be negative.
        """

        if len(geotransform) != 6:
            raise ValueError("GDAL geotransform must have six values")
        left, x_size, x_rotation, top, y_rotation, y_size = geotransform
        if x_rotation != 0 or y_rotation != 0:
            raise ValueError("rotated geotransforms are not supported")
        if x_size <= 0 or y_size >= 0:
            raise ValueError("expected positive x pixel size and negative y pixel size")
        return cls(left=left, top=top, x_size=x_size, y_size=abs(y_size))

    @classmethod
    def from_affine(cls, affine: object) -> "GridTransform":
        """Create a transform from a rasterio/affine-style object.

        The object must expose ``a``, ``b``, ``c``, ``d``, ``e``, and ``f``
        attributes following ``x = a * col + b * row + c`` and
        ``y = d * col + e * row + f``.
        """

        values = tuple(getattr(affine, name) for name in ("c", "a", "b", "f", "d", "e"))
        return cls.from_gdal(values)

    def xy(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return x/y cell-center coordinates for row/column indices."""

        row_array = np.asarray(rows)
        col_array = np.asarray(cols)
        return (
            self.left + (col_array + 0.5) * self.x_size,
            self.top - (row_array + 0.5) * self.y_size,
        )

    def rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest row/column indices for x/y cell-center coordinates."""

        x_array = np.asarray(x)
        y_array = np.asarray(y)
        cols = np.floor((x_array - self.left) / self.x_size).astype(int)
        rows = np.floor((self.top - y_array) / self.y_size).astype(int)
        return rows, cols


@dataclass(frozen=True)
class SpatialLayer:
    """A gridded regionalised variable and its extraction settings."""

    name: str
    values: np.ndarray
    window_size: int
    transform: GridTransform | None = None
    sparse_indices: Sequence[int] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError(f"{self.name!r} values must be a 2D array")
        _validate_window_size(self.window_size)
        if self.sparse_indices is not None:
            _normalise_sparse_indices(self.sparse_indices, self.window_size)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class PatternBatch:
    """A chunk of target centers and their prepared pattern matrix."""

    centers: np.ndarray
    patterns: np.ndarray


@dataclass(frozen=True)
class PatternDataset:
    """Prepared spatial patterns with optional response values."""

    centers: np.ndarray
    patterns: np.ndarray
    target: np.ndarray | None = None


def pattern_size_from_edge(edge: float, cell_size: float) -> int:
    """Return the square pattern width used by the original R implementation.

    The R code computes ``floor(edge * 2 / cell_size)`` and then squares that
    value to obtain the number of cells in a square pattern. This helper returns
    the one-dimensional width and requires it to be odd so the pattern has a
    unique center cell.
    """

    if edge <= 0 or cell_size <= 0:
        raise ValueError("edge and cell_size must be positive")
    size = int(np.floor((edge * 2) / cell_size))
    _validate_window_size(size)
    return size


def centers_from_shape(shape: tuple[int, int]) -> np.ndarray:
    """Return all grid centers for ``shape`` in row-major order."""

    _validate_shape(shape)
    return np.indices(shape).reshape(2, -1).T


def centers_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return row/column centers where ``mask`` is true.

    This mirrors the R workflow of selecting prediction raster cells before
    pattern extraction, while keeping the selection independent from any raster
    file format.
    """

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("mask must be a 2D array")
    return np.argwhere(mask_array)


def iter_centers(
    centers: Iterable[Center] | np.ndarray,
    *,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    """Yield row-major center chunks of at most ``chunk_size`` rows."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    center_array = _as_centers(centers)
    for start in range(0, len(center_array), chunk_size):
        yield center_array[start : start + chunk_size]


def iter_pattern_batches(
    layers: Sequence[SpatialLayer],
    centers: Iterable[Center] | np.ndarray,
    *,
    chunk_size: int,
    prediction_transform: GridTransform | None = None,
    rotations: bool = False,
) -> Iterator[PatternBatch]:
    """Yield prepared pattern matrices in bounded center chunks."""

    for center_chunk in iter_centers(centers, chunk_size=chunk_size):
        yield PatternBatch(
            centers=center_chunk,
            patterns=prepare_patterns(
                layers,
                center_chunk,
                prediction_transform=prediction_transform,
                rotations=rotations,
            ),
        )


def prepare_training_data(
    layers: Sequence[SpatialLayer],
    centers: Iterable[Center] | np.ndarray,
    target: Sequence[object] | np.ndarray,
    *,
    prediction_transform: GridTransform | None = None,
    rotations: bool = True,
) -> PatternDataset:
    """Build pattern and response arrays for supervised SRF training.

    When ``rotations`` is true, each center contributes four pattern rows and its
    response value is repeated four times in the same center-major order.
    """

    center_array = _as_centers(centers)
    target_array = np.asarray(target)
    if target_array.ndim != 1:
        raise ValueError("target must be one-dimensional")
    if len(target_array) != len(center_array):
        raise ValueError("target length must match centers length")

    patterns = prepare_patterns(
        layers,
        center_array,
        prediction_transform=prediction_transform,
        rotations=rotations,
    )
    repeats = 4 if rotations else 1
    return PatternDataset(
        centers=center_array,
        patterns=patterns,
        target=np.repeat(target_array, repeats),
    )


def prepare_patterns(
    layers: Sequence[SpatialLayer],
    centers: Iterable[Center] | np.ndarray,
    *,
    prediction_transform: GridTransform | None = None,
    rotations: bool = False,
) -> np.ndarray:
    """Build a multivariate pattern matrix for target grid cells.

    Parameters
    ----------
    layers:
        Regionalised variables. Each layer contributes one flattened square window
        to the final feature matrix, in the order provided.
    centers:
        Target locations as ``(row, col)`` pairs on the prediction grid.
    prediction_transform:
        Transform for the prediction grid. Required when any layer uses its own
        transform. If omitted, layer arrays are assumed to share the prediction
        grid indexing.
    rotations:
        If true, each center yields four rows: original, 90, 180, and 270 degree
        rotated patterns. This matches the rotation-invariant augmentation used in
        the original R code.
    """

    if not layers:
        raise ValueError("at least one layer is required")

    center_array = _as_centers(centers)
    feature_blocks = [
        vectorize_layer(
            layer,
            center_array,
            prediction_transform=prediction_transform,
            rotations=rotations,
        )
        for layer in layers
    ]
    return np.concatenate(feature_blocks, axis=1)


def vectorize_layer(
    layer: SpatialLayer,
    centers: Iterable[Center] | np.ndarray,
    *,
    prediction_transform: GridTransform | None = None,
    rotations: bool = False,
) -> np.ndarray:
    """Extract flattened local windows for one spatial layer."""

    center_array = _as_centers(centers)
    rows, cols = _centers_for_layer(layer, center_array, prediction_transform)
    _validate_centers_in_bounds(layer, rows, cols)
    windows = _extract_windows(layer.values, rows, cols, layer.window_size)

    if rotations:
        variants = np.stack(
            [np.rot90(windows, k=k, axes=(1, 2)) for k in range(4)],
            axis=1,
        )
        windows = variants.reshape(windows.shape[0] * 4, layer.window_size, layer.window_size)

    flattened = windows.reshape(windows.shape[0], -1)
    if layer.sparse_indices is not None:
        flattened = flattened[:, _normalise_sparse_indices(layer.sparse_indices, layer.window_size)]
    return flattened


def _extract_windows(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    window_size: int,
) -> np.ndarray:
    radius = window_size // 2
    pad_values = _values_for_padding(values)
    padded = np.pad(
        pad_values,
        pad_width=radius,
        mode="constant",
        constant_values=np.nan,
    )
    views = sliding_window_view(padded, (window_size, window_size))
    return views[rows, cols].copy()


def _values_for_padding(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float, copy=False)
    return values.astype(object, copy=False)


def _centers_for_layer(
    layer: SpatialLayer,
    centers: np.ndarray,
    prediction_transform: GridTransform | None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = centers[:, 0]
    cols = centers[:, 1]

    if layer.transform is None:
        return rows, cols
    if prediction_transform is None:
        raise ValueError(
            "prediction_transform is required when a layer has its own transform"
        )

    x, y = prediction_transform.xy(rows, cols)
    return layer.transform.rowcol(x, y)


def _validate_centers_in_bounds(
    layer: SpatialLayer,
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    height, width = layer.values.shape
    if np.any(rows < 0) or np.any(rows >= height) or np.any(cols < 0) or np.any(cols >= width):
        raise ValueError(f"one or more centers fall outside layer {layer.name!r}")


def _as_centers(centers: Iterable[Center] | np.ndarray) -> np.ndarray:
    center_array = np.asarray(list(centers) if not isinstance(centers, np.ndarray) else centers)
    if center_array.ndim != 2 or center_array.shape[1] != 2:
        raise ValueError("centers must be an array-like collection of (row, col) pairs")
    if not np.issubdtype(center_array.dtype, np.integer):
        if not np.all(np.equal(center_array, np.floor(center_array))):
            raise ValueError("centers must contain integer row/column indices")
        center_array = center_array.astype(int)
    return center_array


def _validate_shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2:
        raise ValueError("shape must have two dimensions")
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError("shape dimensions must be positive")


def _validate_window_size(window_size: int) -> None:
    if not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError("window_size must be a positive odd integer")


def _normalise_sparse_indices(indices: Sequence[int], window_size: int) -> np.ndarray:
    sparse = np.asarray(indices)
    if sparse.ndim != 1:
        raise ValueError("sparse_indices must be one-dimensional")
    if not np.issubdtype(sparse.dtype, np.integer):
        raise TypeError("sparse_indices must contain integers")
    max_index = window_size * window_size
    if np.any(sparse < 0) or np.any(sparse >= max_index):
        raise ValueError(f"sparse_indices must be between 0 and {max_index - 1}")
    return sparse.astype(int)

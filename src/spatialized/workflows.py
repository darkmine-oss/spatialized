"""End-to-end spatial prediction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .patterns import GridTransform, SpatialLayer, centers_from_mask, grid_from_centers
from .raster import RasterGrid, write_raster


@dataclass(frozen=True)
class GridPrediction:
    """Prediction outputs reconstructed onto a target grid."""

    centers: np.ndarray
    prediction: np.ndarray
    probabilities: np.ndarray | None = None
    entropy: np.ndarray | None = None


def predict_grid(
    model: object,
    layers: Sequence[SpatialLayer],
    prediction_mask: np.ndarray,
    *,
    chunk_size: int = 10_000,
    prediction_transform: GridTransform | None = None,
    probabilities: bool = False,
    entropy: bool = False,
    fill_value: object = np.nan,
) -> GridPrediction:
    """Predict every true cell in ``prediction_mask`` and rebuild grids."""

    mask = np.asarray(prediction_mask)
    if mask.ndim != 2:
        raise ValueError("prediction_mask must be a 2D array")

    centers = centers_from_mask(mask)
    prediction_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    entropy_parts: list[np.ndarray] = []

    for batch in _iter_model_prediction(
        model,
        layers,
        centers,
        chunk_size=chunk_size,
        prediction_transform=prediction_transform,
        probabilities=probabilities,
        entropy=entropy,
    ):
        prediction_parts.append(batch.prediction)
        if probabilities and batch.probabilities is not None:
            probability_parts.append(batch.probabilities)
        if entropy and batch.entropy is not None:
            entropy_parts.append(batch.entropy)

    flat_prediction = _concat_or_empty(prediction_parts)
    prediction_grid = grid_from_centers(
        mask.shape,
        centers,
        flat_prediction,
        fill_value=fill_value,
    )

    probability_grid = None
    if probabilities:
        probability_grid = grid_from_centers(
            mask.shape,
            centers,
            _concat_or_empty(probability_parts),
            fill_value=np.nan,
        )

    entropy_grid = None
    if entropy:
        entropy_grid = grid_from_centers(
            mask.shape,
            centers,
            _concat_or_empty(entropy_parts),
            fill_value=np.nan,
        )

    return GridPrediction(
        centers=centers,
        prediction=prediction_grid,
        probabilities=probability_grid,
        entropy=entropy_grid,
    )


def predict_grid_to_raster(
    model: object,
    layers: Sequence[SpatialLayer],
    prediction_mask: np.ndarray,
    reference: RasterGrid,
    output_path: str | Path,
    *,
    chunk_size: int = 10_000,
    prediction_transform: GridTransform | None = None,
    entropy_path: str | Path | None = None,
    probabilities_path: str | Path | None = None,
    fill_value: object = np.nan,
    dtype: str | np.dtype | None = None,
) -> GridPrediction:
    """Run ``predict_grid`` and write requested outputs as GeoTIFFs."""

    result = predict_grid(
        model,
        layers,
        prediction_mask,
        chunk_size=chunk_size,
        prediction_transform=prediction_transform,
        probabilities=probabilities_path is not None,
        entropy=entropy_path is not None,
        fill_value=fill_value,
    )
    write_raster(output_path, result.prediction, reference, dtype=dtype)
    if entropy_path is not None and result.entropy is not None:
        write_raster(entropy_path, result.entropy, reference, dtype="float32")
    if probabilities_path is not None and result.probabilities is not None:
        write_raster(probabilities_path, result.probabilities, reference, dtype="float32")
    return result


def _concat_or_empty(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.array([])
    return np.concatenate(parts)


def _iter_model_prediction(
    model: object,
    layers: Sequence[SpatialLayer],
    centers: np.ndarray,
    *,
    chunk_size: int,
    prediction_transform: GridTransform | None,
    probabilities: bool,
    entropy: bool,
):
    if probabilities or entropy:
        return model.iter_predict(
            layers,
            centers,
            chunk_size=chunk_size,
            prediction_transform=prediction_transform,
            probabilities=probabilities,
            entropy=entropy,
        )
    return model.iter_predict(
        layers,
        centers,
        chunk_size=chunk_size,
        prediction_transform=prediction_transform,
    )

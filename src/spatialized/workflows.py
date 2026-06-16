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


@dataclass(frozen=True)
class DomainPrediction:
    """Unsupervised domain modelling outputs."""

    centers: np.ndarray
    labels: np.ndarray
    prediction: GridPrediction
    unsupervised_model: object
    classifier: object


@dataclass(frozen=True)
class TransferPrediction:
    """Train-area to target-area prediction workflow output."""

    training_centers: np.ndarray
    model: object
    prediction: GridPrediction


def train_target_proxy_classifier(
    layers: Sequence[SpatialLayer],
    target_mask: np.ndarray,
    *,
    positive_label: str = "target",
    background_label: str = "background",
    max_background: int | None = None,
    random_state: int | None = None,
    classifier: object | None = None,
    prediction_transform: GridTransform | None = None,
    rotations: bool = True,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Train a binary classifier from a target-unit mask and sampled background."""

    from .models import SpatialRandomForestClassifier

    mask = np.asarray(target_mask, dtype=bool)
    positives = centers_from_mask(mask)
    background = centers_from_mask(~mask)
    if max_background is not None and len(background) > max_background:
        rng = np.random.default_rng(random_state)
        background = background[rng.choice(len(background), size=max_background, replace=False)]

    centers = np.vstack([positives, background])
    labels = np.concatenate(
        [
            np.full(len(positives), positive_label, dtype=object),
            np.full(len(background), background_label, dtype=object),
        ]
    )
    model = (
        SpatialRandomForestClassifier(random_state=random_state)
        if classifier is None
        else classifier
    )
    model.fit(
        layers,
        centers,
        labels,
        prediction_transform=prediction_transform,
        rotations=rotations,
    )
    return model, centers, labels


def predict_target_proxy_transfer(
    train_layers: Sequence[SpatialLayer],
    target_layers: Sequence[SpatialLayer],
    train_target_mask: np.ndarray,
    target_prediction_mask: np.ndarray,
    *,
    train_transform: GridTransform | None = None,
    target_transform: GridTransform | None = None,
    positive_label: str = "target",
    background_label: str = "background",
    max_background: int | None = None,
    random_state: int | None = None,
    classifier: object | None = None,
    rotations: bool = True,
    chunk_size: int = 10_000,
    entropy: bool = True,
) -> TransferPrediction:
    """Train on one area and predict target-proxy units into another area."""

    model, centers, _labels = train_target_proxy_classifier(
        train_layers,
        train_target_mask,
        positive_label=positive_label,
        background_label=background_label,
        max_background=max_background,
        random_state=random_state,
        classifier=classifier,
        prediction_transform=train_transform,
        rotations=rotations,
    )
    prediction = predict_grid(
        model,
        target_layers,
        target_prediction_mask,
        chunk_size=chunk_size,
        prediction_transform=target_transform,
        probabilities=True,
        entropy=entropy,
        fill_value=None,
    )
    return TransferPrediction(
        training_centers=centers,
        model=model,
        prediction=prediction,
    )


def predict_unsupervised_domains(
    layers: Sequence[SpatialLayer],
    sample_centers: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    n_clusters: int,
    unsupervised_model: object | None = None,
    classifier: object | None = None,
    chunk_size: int = 10_000,
    prediction_transform: GridTransform | None = None,
    rotations: bool = True,
    entropy: bool = True,
) -> DomainPrediction:
    """Cluster sampled spatial patterns and predict domains over a grid."""

    from .models import SpatialRandomForestClassifier
    from .unsupervised import UnsupervisedSpatialRandomForest

    unsupervised = (
        UnsupervisedSpatialRandomForest(random_state=42)
        if unsupervised_model is None
        else unsupervised_model
    )
    unsupervised.fit(
        layers,
        sample_centers,
        prediction_transform=prediction_transform,
        rotations=rotations,
    )
    labels = unsupervised.spectral_cluster(n_clusters=n_clusters)
    domain_classifier = (
        SpatialRandomForestClassifier(random_state=42)
        if classifier is None
        else classifier
    )
    domain_classifier.fit(
        layers,
        sample_centers,
        labels.astype(str),
        prediction_transform=prediction_transform,
        rotations=rotations,
    )
    prediction = predict_grid(
        domain_classifier,
        layers,
        prediction_mask,
        chunk_size=chunk_size,
        prediction_transform=prediction_transform,
        probabilities=False,
        entropy=entropy,
        fill_value=None,
    )
    return DomainPrediction(
        centers=np.asarray(sample_centers),
        labels=labels,
        prediction=prediction,
        unsupervised_model=unsupervised,
        classifier=domain_classifier,
    )


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

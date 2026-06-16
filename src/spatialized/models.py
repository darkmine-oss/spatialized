"""Random forest estimators that consume vectorised spatial patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .encoding import PatternEncoder
from .patterns import (
    Center,
    GridTransform,
    PatternDataset,
    SpatialLayer,
    feature_layout,
    iter_centers,
    prepare_patterns,
    prepare_training_data,
    zone_of_influence,
)


def classification_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return standardised Shannon entropy for class probability rows."""

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    n_classes = probs.shape[1]
    if n_classes < 2:
        return np.zeros(probs.shape[0], dtype=float)

    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1) / np.log(n_classes)
    return entropy


@dataclass(frozen=True)
class PredictionBatch:
    """A chunk of prediction centers and model outputs."""

    centers: np.ndarray
    prediction: np.ndarray
    probabilities: np.ndarray | None = None
    entropy: np.ndarray | None = None


@dataclass
class SpatialRandomForestClassifier:
    """Spatial random forest classifier using vectorised local patterns."""

    n_estimators: int = 100
    max_features: str | int | float | None = "sqrt"
    random_state: int | None = None
    n_jobs: int | None = None
    encoder_kwargs: dict[str, object] = field(default_factory=dict)
    estimator_kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.estimator_kwargs,
        )

    def fit(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        target: Sequence[object] | np.ndarray,
        *,
        prediction_transform: GridTransform | None = None,
        rotations: bool = True,
    ) -> "SpatialRandomForestClassifier":
        dataset = prepare_training_data(
            layers,
            centers,
            target,
            prediction_transform=prediction_transform,
            rotations=rotations,
        )
        return self.fit_dataset(dataset)

    def fit_dataset(self, dataset: PatternDataset) -> "SpatialRandomForestClassifier":
        if dataset.target is None:
            raise ValueError("dataset target is required")
        self.feature_encoder_ = PatternEncoder(**self.encoder_kwargs).fit(dataset.patterns)
        self.estimator.fit(self.feature_encoder_.transform(dataset.patterns), dataset.target)
        return self

    def predict(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        *,
        prediction_transform: GridTransform | None = None,
    ) -> np.ndarray:
        patterns = prepare_patterns(
            layers,
            centers,
            prediction_transform=prediction_transform,
            rotations=False,
        )
        return self.estimator.predict(self._transform_patterns(patterns))

    def iter_predict(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        *,
        chunk_size: int,
        prediction_transform: GridTransform | None = None,
        probabilities: bool = False,
        entropy: bool = False,
    ) -> Iterable[PredictionBatch]:
        for center_chunk in iter_centers(centers, chunk_size=chunk_size):
            patterns = prepare_patterns(
                layers,
                center_chunk,
                prediction_transform=prediction_transform,
                rotations=False,
            )
            encoded = self._transform_patterns(patterns)
            prediction = self.estimator.predict(encoded)
            proba = self.estimator.predict_proba(encoded) if probabilities or entropy else None
            yield PredictionBatch(
                centers=center_chunk,
                prediction=prediction,
                probabilities=proba if probabilities else None,
                entropy=classification_entropy(proba) if entropy and proba is not None else None,
            )

    def predict_proba(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        *,
        prediction_transform: GridTransform | None = None,
    ) -> np.ndarray:
        patterns = prepare_patterns(
            layers,
            centers,
            prediction_transform=prediction_transform,
            rotations=False,
        )
        return self.estimator.predict_proba(self._transform_patterns(patterns))

    def entropy(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        *,
        prediction_transform: GridTransform | None = None,
    ) -> np.ndarray:
        return classification_entropy(
            self.predict_proba(
                layers,
                centers,
                prediction_transform=prediction_transform,
            )
        )

    def feature_importance(self) -> np.ndarray:
        """Return fitted per-feature impurity importances."""

        return np.asarray(self.estimator.feature_importances_, dtype=float)

    def zone_of_influence(self, layers: Sequence[SpatialLayer]) -> dict[str, np.ndarray]:
        """Map fitted feature importances back to each layer's local pattern."""

        return zone_of_influence(self.feature_importance(), layers)

    def feature_layout(self, layers: Sequence[SpatialLayer]):
        """Return feature-column metadata for the model input layers."""

        layout = feature_layout(layers)
        if len(layout) != self.estimator.n_features_in_:
            raise ValueError("layer feature layout does not match fitted model")
        return layout

    def _transform_patterns(self, patterns: np.ndarray) -> np.ndarray:
        if not hasattr(self, "feature_encoder_"):
            raise ValueError("model has not been fitted")
        return self.feature_encoder_.transform(patterns)


@dataclass
class SpatialRandomForestRegressor:
    """Spatial random forest regressor using vectorised local patterns."""

    n_estimators: int = 100
    max_features: str | int | float | None = 1.0
    random_state: int | None = None
    n_jobs: int | None = None
    encoder_kwargs: dict[str, object] = field(default_factory=dict)
    estimator_kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        self.estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.estimator_kwargs,
        )

    def fit(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        target: Sequence[float] | np.ndarray,
        *,
        prediction_transform: GridTransform | None = None,
        rotations: bool = True,
    ) -> "SpatialRandomForestRegressor":
        dataset = prepare_training_data(
            layers,
            centers,
            target,
            prediction_transform=prediction_transform,
            rotations=rotations,
        )
        return self.fit_dataset(dataset)

    def fit_dataset(self, dataset: PatternDataset) -> "SpatialRandomForestRegressor":
        if dataset.target is None:
            raise ValueError("dataset target is required")
        self.feature_encoder_ = PatternEncoder(**self.encoder_kwargs).fit(dataset.patterns)
        self.estimator.fit(
            self.feature_encoder_.transform(dataset.patterns),
            dataset.target.astype(float),
        )
        return self

    def predict(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        *,
        prediction_transform: GridTransform | None = None,
    ) -> np.ndarray:
        patterns = prepare_patterns(
            layers,
            centers,
            prediction_transform=prediction_transform,
            rotations=False,
        )
        return self.estimator.predict(self._transform_patterns(patterns))

    def iter_predict(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        *,
        chunk_size: int,
        prediction_transform: GridTransform | None = None,
    ) -> Iterable[PredictionBatch]:
        for center_chunk in iter_centers(centers, chunk_size=chunk_size):
            patterns = prepare_patterns(
                layers,
                center_chunk,
                prediction_transform=prediction_transform,
                rotations=False,
            )
            yield PredictionBatch(
                centers=center_chunk,
                prediction=self.estimator.predict(self._transform_patterns(patterns)),
            )

    def feature_importance(self) -> np.ndarray:
        """Return fitted per-feature impurity importances."""

        return np.asarray(self.estimator.feature_importances_, dtype=float)

    def zone_of_influence(self, layers: Sequence[SpatialLayer]) -> dict[str, np.ndarray]:
        """Map fitted feature importances back to each layer's local pattern."""

        return zone_of_influence(self.feature_importance(), layers)

    def feature_layout(self, layers: Sequence[SpatialLayer]):
        """Return feature-column metadata for the model input layers."""

        layout = feature_layout(layers)
        if len(layout) != self.estimator.n_features_in_:
            raise ValueError("layer feature layout does not match fitted model")
        return layout

    def _transform_patterns(self, patterns: np.ndarray) -> np.ndarray:
        if not hasattr(self, "feature_encoder_"):
            raise ValueError("model has not been fitted")
        return self.feature_encoder_.transform(patterns)

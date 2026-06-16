"""Random forest estimators that consume vectorised spatial patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .patterns import (
    Center,
    GridTransform,
    PatternDataset,
    SpatialLayer,
    prepare_patterns,
    prepare_training_data,
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


@dataclass
class SpatialRandomForestClassifier:
    """Spatial random forest classifier using vectorised local patterns."""

    n_estimators: int = 100
    max_features: str | int | float | None = "sqrt"
    random_state: int | None = None
    n_jobs: int | None = None
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
        self.estimator.fit(dataset.patterns, dataset.target)
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
        return self.estimator.predict(patterns)

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
        return self.estimator.predict_proba(patterns)

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


@dataclass
class SpatialRandomForestRegressor:
    """Spatial random forest regressor using vectorised local patterns."""

    n_estimators: int = 100
    max_features: str | int | float | None = 1.0
    random_state: int | None = None
    n_jobs: int | None = None
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
        self.estimator.fit(dataset.patterns, dataset.target.astype(float))
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
        return self.estimator.predict(patterns)

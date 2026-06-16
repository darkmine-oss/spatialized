"""Unsupervised spatial random forest utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .encoding import PatternEncoder
from .patterns import Center, GridTransform, SpatialLayer, prepare_patterns


@dataclass(frozen=True)
class UnsupervisedResult:
    """Fitted unsupervised SRF state and center-level distances."""

    centers: np.ndarray
    patterns: np.ndarray
    synthetic_patterns: np.ndarray
    distance: np.ndarray
    estimator: object
    rotations: bool


@dataclass(frozen=True)
class ClusterDiagnostics:
    """Cluster-count diagnostics for a fitted unsupervised distance matrix."""

    k_values: np.ndarray
    silhouette_scores: np.ndarray
    eigenvalues: np.ndarray
    eigengaps: np.ndarray


def synthetic_patterns(
    patterns: np.ndarray,
    *,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate synthetic patterns by independently permuting each feature."""

    pattern_array = np.asarray(patterns)
    if pattern_array.ndim != 2:
        raise ValueError("patterns must be a 2D array")
    rng = np.random.default_rng(random_state)
    synthetic = pattern_array.copy()
    for column in range(synthetic.shape[1]):
        synthetic[:, column] = rng.permutation(synthetic[:, column])
    return synthetic


@dataclass
class UnsupervisedSpatialRandomForest:
    """Unsupervised SRF approximation using synthetic-pattern discrimination."""

    n_estimators: int = 100
    max_features: str | int | float | None = "sqrt"
    random_state: int | None = None
    n_jobs: int | None = None
    encoder_kwargs: dict[str, object] | None = None
    estimator_kwargs: dict[str, object] | None = None

    def __post_init__(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        kwargs = {} if self.estimator_kwargs is None else self.estimator_kwargs
        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **kwargs,
        )
        self.result_: UnsupervisedResult | None = None

    def fit(
        self,
        layers: Sequence[SpatialLayer],
        centers: Iterable[Center] | np.ndarray,
        *,
        prediction_transform: GridTransform | None = None,
        rotations: bool = True,
    ) -> "UnsupervisedSpatialRandomForest":
        center_array = np.asarray(list(centers) if not isinstance(centers, np.ndarray) else centers)
        patterns = prepare_patterns(
            layers,
            center_array,
            prediction_transform=prediction_transform,
            rotations=rotations,
        )
        synthetic = synthetic_patterns(patterns, random_state=self.random_state)
        encoder_kwargs = {} if self.encoder_kwargs is None else self.encoder_kwargs
        self.feature_encoder_ = PatternEncoder(**encoder_kwargs).fit(
            np.vstack([patterns, synthetic])
        )
        encoded_patterns = self.feature_encoder_.transform(patterns)
        encoded_synthetic = self.feature_encoder_.transform(synthetic)
        x = np.vstack([encoded_patterns, encoded_synthetic])
        y = np.concatenate(
            [
                np.ones(patterns.shape[0], dtype=int),
                np.zeros(synthetic.shape[0], dtype=int),
            ]
        )
        self.estimator.fit(x, y)
        self.result_ = UnsupervisedResult(
            centers=center_array,
            patterns=patterns,
            synthetic_patterns=synthetic,
            distance=_center_distance_from_leaves(
                self.estimator.apply(encoded_patterns),
                n_centers=len(center_array),
                rotations=4 if rotations else 1,
            ),
            estimator=self.estimator,
            rotations=rotations,
        )
        return self

    @property
    def distance_(self) -> np.ndarray:
        return self._require_result().distance

    def spectral_cluster(self, n_clusters: int) -> np.ndarray:
        """Cluster fitted centers using spectral clustering on RF affinities."""

        return spectral_cluster(
            self._require_result().distance,
            n_clusters=n_clusters,
            random_state=self.random_state,
        )

    def diagnostics(self, k_values: Sequence[int] = tuple(range(2, 11))) -> ClusterDiagnostics:
        """Return silhouette and eigengap diagnostics for candidate cluster counts."""

        return cluster_diagnostics(
            self._require_result().distance,
            k_values=k_values,
            random_state=self.random_state,
        )

    def mds(self, n_components: int = 2) -> np.ndarray:
        """Embed fitted center distances with metric MDS."""

        from sklearn.manifold import MDS

        model = MDS(
            n_components=n_components,
            metric="precomputed",
            init="random",
            n_jobs=1,
            random_state=self.random_state,
            normalized_stress="auto",
        )
        return model.fit_transform(self._require_result().distance)

    def _require_result(self) -> UnsupervisedResult:
        if self.result_ is None:
            raise ValueError("model has not been fitted")
        return self.result_


def spectral_cluster(
    distance: np.ndarray,
    *,
    n_clusters: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Cluster a distance matrix using spectral clustering on an RBF affinity."""

    from sklearn.cluster import SpectralClustering

    distance_array = _as_distance_matrix(distance)
    affinity = affinity_from_distance(distance_array)
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        n_jobs=1,
        random_state=random_state,
    )
    return model.fit_predict(affinity)


def cluster_diagnostics(
    distance: np.ndarray,
    *,
    k_values: Sequence[int] = tuple(range(2, 11)),
    random_state: int | None = None,
) -> ClusterDiagnostics:
    """Compute silhouette scores and affinity eigengaps for candidate cluster counts."""

    from sklearn.metrics import silhouette_score

    distance_array = _as_distance_matrix(distance)
    k_array = np.asarray(k_values, dtype=int)
    if np.any(k_array < 2):
        raise ValueError("k_values must be at least 2")
    if np.any(k_array >= len(distance_array)):
        raise ValueError("k_values must be smaller than the number of samples")

    scores = []
    for k in k_array:
        labels = spectral_cluster(distance_array, n_clusters=int(k), random_state=random_state)
        scores.append(silhouette_score(distance_array, labels, metric="precomputed"))

    affinity = affinity_from_distance(distance_array)
    eigenvalues = np.linalg.eigvalsh(affinity)[::-1]
    max_k = int(np.max(k_array))
    eigengaps = eigenvalues[:max_k] - eigenvalues[1 : max_k + 1]
    return ClusterDiagnostics(
        k_values=k_array,
        silhouette_scores=np.asarray(scores, dtype=float),
        eigenvalues=eigenvalues,
        eigengaps=eigengaps,
    )


def affinity_from_distance(distance: np.ndarray) -> np.ndarray:
    """Convert a distance matrix into the RBF affinity used by SRF clustering."""

    distance_array = _as_distance_matrix(distance)
    affinity = np.exp(-(distance_array**2) / (2 * _sigma(distance_array) ** 2))
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _center_distance_from_leaves(
    leaves: np.ndarray,
    *,
    n_centers: int,
    rotations: int,
) -> np.ndarray:
    if leaves.shape[0] != n_centers * rotations:
        raise ValueError("leaf rows must equal n_centers * rotations")

    same_leaf = leaves[:, None, :] == leaves[None, :, :]
    row_distance = 1.0 - same_leaf.mean(axis=2)
    grouped = row_distance.reshape(n_centers, rotations, n_centers, rotations)
    distance = grouped.min(axis=(1, 3))
    np.fill_diagonal(distance, 0.0)
    return (distance + distance.T) / 2


def _as_distance_matrix(distance: np.ndarray) -> np.ndarray:
    distance_array = np.asarray(distance, dtype=float)
    if distance_array.ndim != 2 or distance_array.shape[0] != distance_array.shape[1]:
        raise ValueError("distance must be a square matrix")
    return distance_array


def _sigma(distance: np.ndarray) -> float:
    nonzero = distance[distance > 0]
    if nonzero.size == 0:
        return 1.0
    return float(np.quantile(nonzero, 0.5)) or 1.0

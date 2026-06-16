import numpy as np
import pytest

from spatialized import (
    SpatialLayer,
    UnsupervisedSpatialRandomForest,
    affinity_from_distance,
    cluster_diagnostics,
    synthetic_patterns,
)


def test_synthetic_patterns_permute_each_column():
    patterns = np.arange(20).reshape(5, 4)

    synthetic = synthetic_patterns(patterns, random_state=7)

    assert synthetic.shape == patterns.shape
    for column in range(patterns.shape[1]):
        np.testing.assert_array_equal(
            np.sort(synthetic[:, column]),
            np.sort(patterns[:, column]),
        )
    assert not np.array_equal(synthetic, patterns)


def test_unsupervised_spatial_random_forest_builds_center_distance():
    values = np.array(
        [
            [0, 0, 9, 9],
            [0, 0, 9, 9],
            [1, 1, 8, 8],
            [1, 1, 8, 8],
        ],
        dtype=float,
    )
    layer = SpatialLayer("x", values, window_size=1)
    centers = [(0, 0), (0, 1), (0, 2), (0, 3), (3, 0), (3, 1), (3, 2), (3, 3)]

    model = UnsupervisedSpatialRandomForest(n_estimators=25, random_state=7, n_jobs=1)
    model.fit([layer], centers, rotations=False)

    distance = model.distance_
    assert distance.shape == (len(centers), len(centers))
    np.testing.assert_array_equal(np.diag(distance), np.zeros(len(centers)))
    np.testing.assert_allclose(distance, distance.T)
    assert np.all((distance >= 0) & (distance <= 1))


def test_unsupervised_spatial_random_forest_encodes_categorical_layers():
    layer = SpatialLayer(
        "lithology",
        np.array(
            [
                ["basalt", "basalt", "granite", "granite"],
                ["basalt", "basalt", "granite", "granite"],
            ],
            dtype=object,
        ),
        window_size=1,
    )
    centers = [(0, 0), (0, 1), (0, 2), (0, 3)]

    model = UnsupervisedSpatialRandomForest(n_estimators=20, random_state=7, n_jobs=1)
    model.fit([layer], centers, rotations=False)

    assert model.distance_.shape == (4, 4)
    assert model.feature_encoder_.columns_[0].kind == "categorical"


def test_unsupervised_spectral_cluster_and_mds_return_center_outputs():
    layer = SpatialLayer(
        "x",
        np.array(
            [
                [0, 0, 9, 9],
                [0, 0, 9, 9],
                [1, 1, 8, 8],
                [1, 1, 8, 8],
            ],
            dtype=float,
        ),
        window_size=1,
    )
    centers = [(0, 0), (0, 1), (0, 2), (0, 3), (3, 0), (3, 1), (3, 2), (3, 3)]

    model = UnsupervisedSpatialRandomForest(n_estimators=25, random_state=7, n_jobs=1)
    model.fit([layer], centers, rotations=False)

    labels = model.spectral_cluster(n_clusters=2)
    embedding = model.mds(n_components=2)

    assert labels.shape == (len(centers),)
    assert set(labels) <= {0, 1}
    assert embedding.shape == (len(centers), 2)


def test_affinity_from_distance_has_unit_diagonal():
    distance = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, 0.6], [0.8, 0.6, 0.0]])

    affinity = affinity_from_distance(distance)

    assert affinity.shape == distance.shape
    np.testing.assert_array_equal(np.diag(affinity), [1.0, 1.0, 1.0])
    assert affinity[0, 1] > affinity[0, 2]


def test_cluster_diagnostics_returns_silhouette_and_eigengaps():
    distance = np.array(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.85, 0.88],
            [0.8, 0.85, 0.0, 0.12],
            [0.9, 0.88, 0.12, 0.0],
        ]
    )

    diagnostics = cluster_diagnostics(distance, k_values=[2], random_state=7)

    np.testing.assert_array_equal(diagnostics.k_values, [2])
    assert diagnostics.silhouette_scores.shape == (1,)
    assert diagnostics.eigenvalues.shape == (4,)
    assert diagnostics.eigengaps.shape == (2,)


def test_unsupervised_model_diagnostics_uses_fitted_distance():
    layer = SpatialLayer("x", np.arange(16, dtype=float).reshape(4, 4), window_size=1)
    centers = [(0, 0), (0, 1), (3, 2), (3, 3)]
    model = UnsupervisedSpatialRandomForest(n_estimators=25, random_state=7, n_jobs=1)
    model.fit([layer], centers, rotations=False)

    diagnostics = model.diagnostics(k_values=[2])

    np.testing.assert_array_equal(diagnostics.k_values, [2])
    assert diagnostics.silhouette_scores.shape == (1,)


def test_unsupervised_methods_require_fit():
    model = UnsupervisedSpatialRandomForest(n_estimators=2, random_state=7)

    with pytest.raises(ValueError, match="not been fitted"):
        _ = model.distance_

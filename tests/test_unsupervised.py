import numpy as np
import pytest

from spatialized import (
    SpatialLayer,
    UnsupervisedSpatialRandomForest,
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


def test_unsupervised_methods_require_fit():
    model = UnsupervisedSpatialRandomForest(n_estimators=2, random_state=7)

    with pytest.raises(ValueError, match="not been fitted"):
        _ = model.distance_

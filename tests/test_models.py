import numpy as np
import pytest

from spatialized import (
    SpatialLayer,
    SpatialRandomForestClassifier,
    SpatialRandomForestRegressor,
    classification_entropy,
    prepare_training_data,
)


def test_classification_entropy_is_standardised():
    probabilities = np.array([[1.0, 0.0], [0.5, 0.5], [0.9, 0.1]])

    entropy = classification_entropy(probabilities)

    np.testing.assert_allclose(entropy[:2], [0.0, 1.0], atol=1e-9)
    assert 0 < entropy[2] < 1


def test_spatial_random_forest_classifier_fits_prepared_patterns():
    layer = SpatialLayer("x", np.array([[0, 0], [1, 1]], dtype=float), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = ["low", "low", "high", "high"]

    model = SpatialRandomForestClassifier(n_estimators=30, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    prediction = model.predict([layer], centers)
    probabilities = model.predict_proba([layer], centers)
    entropy = model.entropy([layer], centers)

    np.testing.assert_array_equal(prediction, target)
    assert probabilities.shape == (4, 2)
    assert entropy.shape == (4,)


def test_spatial_random_forest_regressor_fits_prepared_patterns():
    layer = SpatialLayer("x", np.arange(4, dtype=float).reshape(2, 2), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = np.array([0.0, 1.0, 2.0, 3.0])

    model = SpatialRandomForestRegressor(n_estimators=40, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    prediction = model.predict([layer], centers)

    assert prediction.shape == (4,)
    assert np.all(np.isfinite(prediction))
    assert np.corrcoef(prediction, target)[0, 1] > 0.9


def test_fit_dataset_requires_target():
    dataset = prepare_training_data(
        [SpatialLayer("x", np.arange(4).reshape(2, 2), window_size=1)],
        centers=[(0, 0)],
        target=[1],
        rotations=False,
    )
    model = SpatialRandomForestClassifier(n_estimators=2, random_state=7)

    with pytest.raises(ValueError, match="target"):
        model.fit_dataset(type(dataset)(centers=dataset.centers, patterns=dataset.patterns))

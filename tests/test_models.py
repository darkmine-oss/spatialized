import numpy as np
import pytest

from spatialized import (
    SpatialLayer,
    SpatialRandomForestClassifier,
    SpatialRandomForestRegressor,
    classification_entropy,
    prepare_training_data,
    regression_r2_score,
    regression_residuals,
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


def test_spatial_random_forest_classifier_encodes_categorical_layers():
    layer = SpatialLayer(
        "lithology",
        np.array([["basalt", "basalt"], ["granite", "granite"]], dtype=object),
        window_size=1,
    )
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = ["mafic", "mafic", "felsic", "felsic"]

    model = SpatialRandomForestClassifier(n_estimators=30, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    prediction = model.predict([layer], centers)

    np.testing.assert_array_equal(prediction, target)
    assert model.feature_encoder_.columns_[0].kind == "categorical"
    assert model.feature_encoder_.columns_[0].categories == ("basalt", "granite")


def test_spatial_random_forest_classifier_accepts_encoder_kwargs():
    layer = SpatialLayer(
        "lithology",
        np.array([["basalt", None], ["granite", "granite"]], dtype=object),
        window_size=1,
    )
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = ["mafic", "mafic", "felsic", "felsic"]

    model = SpatialRandomForestClassifier(
        n_estimators=20,
        random_state=7,
        encoder_kwargs={"categorical_missing_strategy": "most_frequent"},
    )
    model.fit([layer], centers, target, rotations=False)

    assert model.feature_encoder_.columns_[0].fill_value == 1.0


def test_model_feature_importance_and_zone_of_influence():
    layer = SpatialLayer("x", np.arange(25, dtype=float).reshape(5, 5), window_size=3)
    centers = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
    target = [0, 0, 1, 1, 1]

    model = SpatialRandomForestClassifier(n_estimators=30, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    importance = model.feature_importance()
    zones = model.zone_of_influence([layer])

    assert importance.shape == (9,)
    assert zones["x"].shape == (3, 3)
    np.testing.assert_allclose(zones["x"].ravel(), importance)


def test_spatial_random_forest_classifier_iter_predict_chunks_outputs():
    layer = SpatialLayer("x", np.array([[0, 0], [1, 1]], dtype=float), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = ["low", "low", "high", "high"]

    model = SpatialRandomForestClassifier(n_estimators=30, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    batches = list(
        model.iter_predict(
            [layer],
            centers,
            chunk_size=3,
            probabilities=True,
            entropy=True,
        )
    )

    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0].centers, [[0, 0], [0, 1], [1, 0]])
    np.testing.assert_array_equal(batches[1].centers, [[1, 1]])
    np.testing.assert_array_equal(
        np.concatenate([batch.prediction for batch in batches]),
        target,
    )
    assert batches[0].probabilities.shape == (3, 2)
    assert batches[0].entropy.shape == (3,)


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


def test_spatial_random_forest_regressor_encodes_categorical_layers():
    layer = SpatialLayer(
        "class",
        np.array([["low", "low"], ["high", "high"]], dtype=object),
        window_size=1,
    )
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = np.array([1.0, 1.0, 10.0, 10.0])

    model = SpatialRandomForestRegressor(n_estimators=40, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    prediction = model.predict([layer], centers)

    assert prediction[:2].mean() < prediction[2:].mean()
    assert model.feature_encoder_.columns_[0].kind == "categorical"


def test_spatial_random_forest_regressor_iter_predict_chunks_outputs():
    layer = SpatialLayer("x", np.arange(4, dtype=float).reshape(2, 2), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = np.array([0.0, 1.0, 2.0, 3.0])

    model = SpatialRandomForestRegressor(n_estimators=40, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    batches = list(model.iter_predict([layer], centers, chunk_size=2))

    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0].centers, [[0, 0], [0, 1]])
    np.testing.assert_array_equal(batches[1].centers, [[1, 0], [1, 1]])
    assert np.concatenate([batch.prediction for batch in batches]).shape == (4,)


def test_spatial_random_forest_regressor_oob_prediction_and_r2():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(6, 6))
    layer = SpatialLayer("x", values, window_size=1)
    centers = [(row, col) for row in range(6) for col in range(6)]
    target = values.ravel() * 2 + 1

    model = SpatialRandomForestRegressor(
        n_estimators=50,
        random_state=7,
        estimator_kwargs={"oob_score": True, "bootstrap": True},
    )
    model.fit([layer], centers, target, rotations=False)

    oob_prediction = model.oob_prediction()

    assert oob_prediction.shape == target.shape
    assert model.oob_r2_score() > 0.0


def test_spatial_random_forest_regressor_oob_prediction_requires_oob_score():
    layer = SpatialLayer("x", np.arange(4, dtype=float).reshape(2, 2), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = np.array([0.0, 1.0, 2.0, 3.0])

    model = SpatialRandomForestRegressor(n_estimators=10, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    with pytest.raises(ValueError, match="oob_score"):
        model.oob_prediction()


def test_spatial_random_forest_classifier_oob_prediction_and_accuracy():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(6, 6))
    layer = SpatialLayer("x", values, window_size=1)
    centers = [(row, col) for row in range(6) for col in range(6)]
    target = np.where(values.ravel() > 0, "high", "low")

    model = SpatialRandomForestClassifier(
        n_estimators=50,
        random_state=7,
        estimator_kwargs={"oob_score": True, "bootstrap": True},
    )
    model.fit([layer], centers, target, rotations=False)

    oob_prediction = model.oob_prediction()

    assert oob_prediction.shape == target.shape
    assert 0.0 <= model.oob_accuracy_score() <= 1.0


def test_spatial_random_forest_classifier_oob_prediction_requires_oob_score():
    layer = SpatialLayer("x", np.array([[0, 0], [1, 1]], dtype=float), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target = ["low", "low", "high", "high"]

    model = SpatialRandomForestClassifier(n_estimators=10, random_state=7)
    model.fit([layer], centers, target, rotations=False)

    with pytest.raises(ValueError, match="oob_score"):
        model.oob_decision_function()


def test_regression_r2_score_matches_r_formula():
    observed = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.0, 2.0, 3.0, 4.0])

    assert regression_r2_score(observed, predicted) == pytest.approx(1.0)

    predicted_mean_only = np.full_like(observed, observed.mean())
    assert regression_r2_score(observed, predicted_mean_only) == pytest.approx(0.0)


def test_regression_residuals_is_predicted_minus_observed():
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.5, 1.5, 4.0])

    np.testing.assert_allclose(regression_residuals(observed, predicted), [0.5, -0.5, 1.0])


def test_spatial_random_forest_regressor_window_mean_imputation_is_wired_from_fit():
    # Two layers with a 2x2 window each; some cells at the grid edge will be
    # NaN-padded, exercising the auto-computed layer_spans wiring from fit().
    layer_a = SpatialLayer("a", np.arange(9, dtype=float).reshape(3, 3), window_size=2)
    layer_b = SpatialLayer("b", np.arange(9, 18, dtype=float).reshape(3, 3), window_size=2)
    centers = [(0, 0), (1, 1), (2, 2)]
    target = np.array([1.0, 2.0, 3.0])

    model = SpatialRandomForestRegressor(
        n_estimators=10,
        random_state=7,
        encoder_kwargs={"numeric_missing_strategy": "window_mean"},
    )
    model.fit([layer_a, layer_b], centers, target, rotations=False)

    prediction = model.predict([layer_a, layer_b], centers)

    assert prediction.shape == (3,)
    assert np.all(np.isfinite(prediction))
    assert model.feature_encoder_.layer_spans == ((0, 4), (4, 8))


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

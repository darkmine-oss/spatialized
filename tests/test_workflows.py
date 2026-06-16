import numpy as np
import rasterio

from spatialized import (
    GridTransform,
    RasterGrid,
    SpatialLayer,
    SpatialRandomForestClassifier,
    SpatialRandomForestRegressor,
    UnsupervisedSpatialRandomForest,
    predict_grid,
    predict_grid_to_raster,
    predict_target_proxy_transfer,
    predict_unsupervised_domains,
    train_target_proxy_classifier,
)


def test_predict_grid_reconstructs_classifier_outputs():
    layer = SpatialLayer("x", np.array([[0, 0], [1, 1]], dtype=float), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    model = SpatialRandomForestClassifier(n_estimators=30, random_state=7)
    model.fit([layer], centers, [0, 0, 1, 1], rotations=False)

    result = predict_grid(
        model,
        [layer],
        prediction_mask=np.ones((2, 2), dtype=bool),
        chunk_size=3,
        probabilities=True,
        entropy=True,
    )

    np.testing.assert_array_equal(result.centers, centers)
    np.testing.assert_array_equal(result.prediction, [[0, 0], [1, 1]])
    assert result.probabilities.shape == (2, 2, 2)
    assert result.entropy.shape == (2, 2)


def test_predict_grid_reconstructs_regression_outputs_for_mask():
    layer = SpatialLayer("x", np.arange(4, dtype=float).reshape(2, 2), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    model = SpatialRandomForestRegressor(n_estimators=40, random_state=7)
    model.fit([layer], centers, [0, 1, 2, 3], rotations=False)

    result = predict_grid(
        model,
        [layer],
        prediction_mask=np.array([[True, False], [False, True]]),
        chunk_size=1,
    )

    assert np.isfinite(result.prediction[0, 0])
    assert np.isnan(result.prediction[0, 1])
    assert np.isnan(result.prediction[1, 0])
    assert np.isfinite(result.prediction[1, 1])
    assert result.probabilities is None
    assert result.entropy is None


def test_predict_grid_to_raster_writes_prediction_and_entropy(tmp_path):
    layer = SpatialLayer("x", np.array([[0, 0], [1, 1]], dtype=float), window_size=1)
    centers = [(0, 0), (0, 1), (1, 0), (1, 1)]
    model = SpatialRandomForestClassifier(n_estimators=30, random_state=7)
    model.fit([layer], centers, [0, 0, 1, 1], rotations=False)
    reference = RasterGrid(
        values=np.zeros((2, 2), dtype=np.float32),
        transform=GridTransform(left=0, top=2, x_size=1),
    )
    prediction_path = tmp_path / "prediction.tif"
    entropy_path = tmp_path / "entropy.tif"
    probabilities_path = tmp_path / "probabilities.tif"

    result = predict_grid_to_raster(
        model,
        [layer],
        prediction_mask=np.ones((2, 2), dtype=bool),
        reference=reference,
        output_path=prediction_path,
        entropy_path=entropy_path,
        probabilities_path=probabilities_path,
        dtype="int16",
    )

    np.testing.assert_array_equal(result.prediction, [[0, 0], [1, 1]])
    with rasterio.open(prediction_path) as dataset:
        np.testing.assert_array_equal(dataset.read(1), [[0, 0], [1, 1]])
    with rasterio.open(entropy_path) as dataset:
        assert dataset.read(1).shape == (2, 2)
    with rasterio.open(probabilities_path) as dataset:
        assert dataset.count == 2


def test_predict_unsupervised_domains_trains_classifier_and_predicts_grid():
    values = np.array(
        [
            [0, 0, 0, 8],
            [0, 0, 8, 8],
            [1, 1, 9, 9],
            [1, 1, 9, 9],
        ],
        dtype=float,
    )
    layer = SpatialLayer("x", values, window_size=1)
    sample_centers = np.array([[0, 0], [0, 1], [2, 2], [3, 3]])

    result = predict_unsupervised_domains(
        [layer],
        sample_centers,
        prediction_mask=np.ones(values.shape, dtype=bool),
        n_clusters=2,
        unsupervised_model=UnsupervisedSpatialRandomForest(
            n_estimators=25,
            random_state=7,
            n_jobs=1,
        ),
        classifier=SpatialRandomForestClassifier(n_estimators=25, random_state=7, n_jobs=1),
        rotations=False,
        chunk_size=4,
    )

    assert result.labels.shape == (len(sample_centers),)
    assert result.prediction.prediction.shape == values.shape
    assert result.prediction.entropy.shape == values.shape


def test_train_target_proxy_classifier_balances_background_limit():
    values = np.arange(16, dtype=float).reshape(4, 4)
    layer = SpatialLayer("mag_hf_1vd", values, window_size=1)
    target_mask = values > 12

    model, centers, labels = train_target_proxy_classifier(
        [layer],
        target_mask,
        max_background=3,
        random_state=7,
        classifier=SpatialRandomForestClassifier(n_estimators=20, random_state=7, n_jobs=1),
        rotations=False,
    )

    assert centers.shape == (6, 2)
    assert list(labels).count("target") == 3
    assert list(labels).count("background") == 3
    assert hasattr(model, "feature_encoder_")


def test_predict_target_proxy_transfer_predicts_into_target_area():
    train_values = np.arange(16, dtype=float).reshape(4, 4)
    target_values = train_values + 1
    train_layer = SpatialLayer("mag_hf_1vd", train_values, window_size=1)
    target_layer = SpatialLayer("mag_hf_1vd", target_values, window_size=1)
    train_target_mask = train_values > 12

    result = predict_target_proxy_transfer(
        [train_layer],
        [target_layer],
        train_target_mask,
        target_prediction_mask=np.ones((4, 4), dtype=bool),
        max_background=3,
        random_state=7,
        classifier=SpatialRandomForestClassifier(n_estimators=25, random_state=7, n_jobs=1),
        rotations=False,
        chunk_size=4,
    )

    assert result.training_centers.shape == (6, 2)
    assert result.prediction.prediction.shape == (4, 4)
    assert result.prediction.probabilities.shape == (4, 4, 2)
    assert result.prediction.entropy.shape == (4, 4)

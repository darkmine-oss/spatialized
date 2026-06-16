import numpy as np
import rasterio

from spatialized import (
    GridTransform,
    RasterGrid,
    SpatialLayer,
    SpatialRandomForestClassifier,
    SpatialRandomForestRegressor,
    predict_grid,
    predict_grid_to_raster,
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

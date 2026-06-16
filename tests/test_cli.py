import json
import pickle

import numpy as np
import rasterio
from affine import Affine

from spatialized import SpatialLayer, SpatialRandomForestClassifier
from spatialized.cli import main


def test_feature_layout_cli_writes_json(tmp_path):
    output = tmp_path / "layout.json"

    exit_code = main(
        [
            "feature-layout",
            "--layer",
            "mag:3",
            "--layer",
            "grav:5:0,12,24",
            "--rotations",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["rotations"] is True
    assert payload["n_features"] == 12
    assert payload["layers"][1]["layer_name"] == "grav"
    assert payload["layers"][1]["sparse_indices"] == [0, 12, 24]


def test_predict_grid_cli_writes_prediction_raster(tmp_path):
    layer_values = np.array([[0, 0], [1, 1]], dtype=np.float32)
    layer_path = tmp_path / "layer.tif"
    mask_path = tmp_path / "mask.tif"
    model_path = tmp_path / "model.pkl"
    output_path = tmp_path / "prediction.tif"
    entropy_path = tmp_path / "entropy.tif"
    transform = Affine(1, 0, 0, 0, -1, 2)

    with rasterio.open(
        layer_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        transform=transform,
    ) as dataset:
        dataset.write(layer_values, 1)
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        transform=transform,
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype=np.float32), 1)

    training_layer = SpatialLayer("x", layer_values, window_size=1)
    model = SpatialRandomForestClassifier(n_estimators=30, random_state=7)
    model.fit(
        [training_layer],
        centers=[(0, 0), (0, 1), (1, 0), (1, 1)],
        target=[0, 0, 1, 1],
        rotations=False,
    )
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    exit_code = main(
        [
            "predict-grid",
            "--model",
            str(model_path),
            "--layer",
            f"x:{layer_path}:1",
            "--mask-raster",
            str(mask_path),
            "--mask-mode",
            "all",
            "--output",
            str(output_path),
            "--entropy-output",
            str(entropy_path),
            "--dtype",
            "int16",
        ]
    )

    assert exit_code == 0
    with rasterio.open(output_path) as dataset:
        np.testing.assert_array_equal(dataset.read(1), [[0, 0], [1, 1]])
    with rasterio.open(entropy_path) as dataset:
        assert dataset.read(1).shape == (2, 2)

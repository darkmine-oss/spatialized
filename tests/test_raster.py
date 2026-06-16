import numpy as np
import rasterio
from affine import Affine

from spatialized import (
    GridTransform,
    RasterGrid,
    read_raster,
    read_spatial_layer,
    write_raster,
)


def test_read_raster_preserves_values_and_metadata(tmp_path):
    path = tmp_path / "input.tif"
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    transform = Affine(10, 0, 100, 0, -20, 200)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="float32",
        transform=transform,
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)

    grid = read_raster(path)

    np.testing.assert_array_equal(grid.values, values)
    assert grid.transform == GridTransform(left=100, top=200, x_size=10, y_size=20)
    assert grid.nodata == -9999
    assert grid.shape == (2, 3)


def test_read_spatial_layer_builds_layer_from_raster(tmp_path):
    path = tmp_path / "layer.tif"

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        transform=Affine(1, 0, 0, 0, -1, 2),
        nodata=-9999,
    ) as dataset:
        dataset.write(np.array([[1, -9999], [3, 4]], dtype=np.float32), 1)

    layer = read_spatial_layer(path, window_size=3)

    assert layer.name == "layer"
    assert layer.window_size == 3
    assert np.isnan(layer.values[0, 1])


def test_write_raster_writes_2d_grid(tmp_path):
    path = tmp_path / "output.tif"
    reference = RasterGrid(
        values=np.zeros((2, 3), dtype=np.float32),
        transform=GridTransform(left=100, top=200, x_size=10, y_size=20),
        nodata=-9999,
    )
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    write_raster(path, values, reference)

    with rasterio.open(path) as dataset:
        np.testing.assert_array_equal(dataset.read(1), values)
        assert dataset.transform == Affine(10, 0, 100, 0, -20, 200)
        assert dataset.nodata == -9999


def test_write_raster_writes_band_last_grid(tmp_path):
    path = tmp_path / "probability.tif"
    reference = RasterGrid(
        values=np.zeros((2, 2), dtype=np.float32),
        transform=GridTransform(left=0, top=2, x_size=1),
    )
    values = np.array(
        [
            [[0.8, 0.2], [0.7, 0.3]],
            [[0.1, 0.9], [0.4, 0.6]],
        ],
        dtype=np.float32,
    )

    write_raster(path, values, reference)

    with rasterio.open(path) as dataset:
        assert dataset.count == 2
        np.testing.assert_array_equal(dataset.read(1), values[:, :, 0])
        np.testing.assert_array_equal(dataset.read(2), values[:, :, 1])

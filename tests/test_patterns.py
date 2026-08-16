import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from spatialized import (
    GridTransform,
    FeatureLayout,
    SpatialLayer,
    centers_from_mask,
    centers_from_shape,
    feature_layout,
    grid_from_centers,
    iter_centers,
    iter_pattern_batches,
    pattern_size_from_edge,
    prepare_patterns,
    prepare_training_data,
    vectorize_layer,
    zone_of_influence,
)


@dataclass(frozen=True)
class FakeAffine:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float


def test_vectorize_layer_flattens_square_window_row_wise():
    layer = SpatialLayer("x", np.arange(25).reshape(5, 5), window_size=3)

    result = vectorize_layer(layer, [(2, 2)])

    np.testing.assert_array_equal(
        result,
        np.array([[6, 7, 8, 11, 12, 13, 16, 17, 18]], dtype=float),
    )


def test_vectorize_layer_supports_even_window_size():
    # Some of the original R workflows use even-width patterns (e.g. a 10x10
    # window) with no unique center cell. Verified against a real recovered R
    # pattern (see .features/PLAN.md, "Improve Unsupervised SRF Parity"): for
    # an even window_size, `(window_size - 1) // 2` cells are taken before
    # the target pixel and the rest after -- i.e. window_size=4 takes 1 cell
    # before, the center, and 2 after.
    layer = SpatialLayer("x", np.arange(36).reshape(6, 6), window_size=4)

    result = vectorize_layer(layer, [(2, 2)])

    np.testing.assert_array_equal(
        result,
        np.array(
            [[7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28]],
            dtype=float,
        ),
    )


def test_edges_are_padded_with_nan():
    layer = SpatialLayer("x", np.arange(9).reshape(3, 3), window_size=3)

    result = vectorize_layer(layer, [(0, 0)])

    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])
    assert np.isnan(result[0, 3])
    np.testing.assert_array_equal(result[0, [4, 5, 7, 8]], [0, 1, 3, 4])


def test_categorical_layers_are_supported_with_nan_padding():
    layer = SpatialLayer(
        "class",
        np.array([["a", "b"], ["c", "d"]], dtype=object),
        window_size=3,
    )

    result = vectorize_layer(layer, [(0, 0)])

    assert result.dtype == object
    assert np.isnan(result[0, 0])
    np.testing.assert_array_equal(result[0, [4, 5, 7, 8]], ["a", "b", "c", "d"])


def test_prepare_patterns_concatenates_layers():
    first = SpatialLayer("a", np.arange(25).reshape(5, 5), window_size=3)
    second = SpatialLayer("b", np.arange(100, 125).reshape(5, 5), window_size=3)

    result = prepare_patterns([first, second], [(2, 2)])

    assert result.shape == (1, 18)
    np.testing.assert_array_equal(result[0, :9], [6, 7, 8, 11, 12, 13, 16, 17, 18])
    np.testing.assert_array_equal(
        result[0, 9:], [106, 107, 108, 111, 112, 113, 116, 117, 118]
    )


def test_feature_layout_describes_layer_offsets_and_sparse_cells():
    first = SpatialLayer("dense", np.zeros((3, 3)), window_size=3)
    second = SpatialLayer("sparse", np.zeros((3, 3)), window_size=3, sparse_indices=[0, 4, 8])

    layout = feature_layout([first, second])

    assert len(layout) == 12
    assert layout.layer_names() == ("dense", "sparse")
    np.testing.assert_array_equal(layout.indices_for_layer("sparse"), [9, 10, 11])
    assert layout.features[9].layer_name == "sparse"
    assert layout.features[9].window_row == 0
    assert layout.features[9].window_col == 0
    assert layout.features[11].window_row == 2
    assert layout.features[11].window_col == 2
    assert layout.layers[0].start == 0
    assert layout.layers[0].stop == 9
    assert layout.layers[1].start == 9
    assert layout.layers[1].stop == 12
    assert layout.layers[1].sparse_indices == (0, 4, 8)


def test_feature_layout_round_trips_through_dict():
    layer = SpatialLayer("x", np.zeros((3, 3)), window_size=3, sparse_indices=[0, 8])

    layout = feature_layout([layer], rotations=True)
    restored = FeatureLayout.from_dict(layout.to_dict())

    assert restored == layout
    assert restored.rotations is True


def test_zone_of_influence_reconstructs_per_layer_windows():
    first = SpatialLayer("dense", np.zeros((3, 3)), window_size=3)
    second = SpatialLayer("sparse", np.zeros((3, 3)), window_size=3, sparse_indices=[0, 4, 8])
    importances = np.arange(12, dtype=float)

    zones = zone_of_influence(importances, [first, second])

    np.testing.assert_array_equal(zones["dense"], np.arange(9, dtype=float).reshape(3, 3))
    np.testing.assert_array_equal(
        zones["sparse"],
        np.array([[9, 0, 0], [0, 10, 0], [0, 0, 11]], dtype=float),
    )


def test_rotations_quadruple_rows_per_center():
    layer = SpatialLayer("x", np.arange(25).reshape(5, 5), window_size=3)

    result = vectorize_layer(layer, [(2, 2)], rotations=True)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [6, 7, 8, 11, 12, 13, 16, 17, 18],
                [8, 13, 18, 7, 12, 17, 6, 11, 16],
                [18, 17, 16, 13, 12, 11, 8, 7, 6],
                [16, 11, 6, 17, 12, 7, 18, 13, 8],
            ],
            dtype=float,
        ),
    )


def test_rotations_are_grouped_by_center():
    layer = SpatialLayer("x", np.arange(36).reshape(6, 6), window_size=3)

    result = vectorize_layer(layer, [(2, 2), (3, 3)], rotations=True)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [7, 8, 9, 13, 14, 15, 19, 20, 21],
                [9, 15, 21, 8, 14, 20, 7, 13, 19],
                [21, 20, 19, 15, 14, 13, 9, 8, 7],
                [19, 13, 7, 20, 14, 8, 21, 15, 9],
                [14, 15, 16, 20, 21, 22, 26, 27, 28],
                [16, 22, 28, 15, 21, 27, 14, 20, 26],
                [28, 27, 26, 22, 21, 20, 16, 15, 14],
                [26, 20, 14, 27, 21, 15, 28, 22, 16],
            ],
            dtype=float,
        ),
    )


def test_sparse_indices_are_applied_after_rotation():
    layer = SpatialLayer(
        "x",
        np.arange(25).reshape(5, 5),
        window_size=3,
        sparse_indices=[0, 4, 8],
    )

    result = vectorize_layer(layer, [(2, 2)], rotations=True)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [6, 12, 18],
                [8, 12, 16],
                [18, 12, 6],
                [16, 12, 8],
            ],
            dtype=float,
        ),
    )


def test_layer_transform_maps_prediction_cells_to_layer_cells():
    prediction_transform = GridTransform(left=0, top=5, x_size=1)
    coarse_transform = GridTransform(left=0, top=6, x_size=2)
    coarse = SpatialLayer(
        "coarse",
        np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]]),
        window_size=3,
        transform=coarse_transform,
    )

    result = vectorize_layer(
        coarse,
        [(2, 2)],
        prediction_transform=prediction_transform,
    )

    np.testing.assert_array_equal(
        result,
        np.array([[10, 11, 12, 20, 21, 22, 30, 31, 32]], dtype=float),
    )


def test_grid_transform_round_trips_xy_and_rowcol():
    transform = GridTransform(left=100, top=200, x_size=10, y_size=20)

    x, y = transform.xy(np.array([0, 2]), np.array([1, 3]))
    rows, cols = transform.rowcol(x, y)

    np.testing.assert_array_equal(x, [115, 135])
    np.testing.assert_array_equal(y, [190, 150])
    np.testing.assert_array_equal(rows, [0, 2])
    np.testing.assert_array_equal(cols, [1, 3])


def test_grid_transform_from_gdal_accepts_north_up_transform():
    transform = GridTransform.from_gdal((100, 10, 0, 200, 0, -20))

    assert transform == GridTransform(left=100, top=200, x_size=10, y_size=20)


def test_grid_transform_from_gdal_rejects_rotated_transform():
    with pytest.raises(ValueError, match="rotated"):
        GridTransform.from_gdal((100, 10, 1, 200, 0, -20))


def test_grid_transform_from_affine_accepts_rasterio_style_object():
    transform = GridTransform.from_affine(FakeAffine(a=10, b=0, c=100, d=0, e=-20, f=200))

    assert transform == GridTransform(left=100, top=200, x_size=10, y_size=20)


def test_centers_outside_layer_bounds_raise():
    layer = SpatialLayer("x", np.arange(9).reshape(3, 3), window_size=3)

    with pytest.raises(ValueError, match="outside layer 'x'"):
        vectorize_layer(layer, [(-1, 0)])


def test_pattern_size_from_edge_matches_r_formula():
    assert pattern_size_from_edge(edge=3.99, cell_size=1) == 7
    assert pattern_size_from_edge(edge=2.99, cell_size=1) == 5


def test_pattern_size_from_edge_supports_even_widths():
    # realCase2's Cu-regression workflow uses `myband <- res * 5`, giving a
    # 10x10 (even) pattern: `floor(5 * 2 / 1) == 10`.
    assert pattern_size_from_edge(edge=5, cell_size=1) == 10


def test_window_size_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive integer"):
        SpatialLayer("x", np.zeros((3, 3)), window_size=0)


def test_centers_from_shape_returns_row_major_grid():
    result = centers_from_shape((2, 3))

    np.testing.assert_array_equal(
        result,
        np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]),
    )


def test_centers_from_mask_returns_true_cells_in_row_major_order():
    mask = np.array(
        [
            [False, True, False],
            [True, False, True],
        ]
    )

    result = centers_from_mask(mask)

    np.testing.assert_array_equal(result, np.array([[0, 1], [1, 0], [1, 2]]))


def test_grid_from_centers_reconstructs_numeric_grid():
    result = grid_from_centers(
        (2, 3),
        centers=[(0, 1), (1, 2)],
        values=np.array([10, 20]),
    )

    expected = np.array([[np.nan, 10, np.nan], [np.nan, np.nan, 20]])
    np.testing.assert_array_equal(result, expected)


def test_grid_from_centers_reconstructs_categorical_grid():
    result = grid_from_centers(
        (2, 2),
        centers=[(0, 0), (1, 1)],
        values=np.array(["ore", "waste"]),
        fill_value=None,
    )

    expected = np.array([["ore", None], [None, "waste"]], dtype=object)
    assert result.dtype == object
    np.testing.assert_array_equal(result, expected)


def test_grid_from_centers_supports_trailing_probability_dimensions():
    result = grid_from_centers(
        (2, 2),
        centers=[(0, 0), (1, 1)],
        values=np.array([[0.8, 0.2], [0.1, 0.9]]),
    )

    assert result.shape == (2, 2, 2)
    np.testing.assert_array_equal(result[0, 0], [0.8, 0.2])
    np.testing.assert_array_equal(result[1, 1], [0.1, 0.9])
    assert np.all(np.isnan(result[0, 1]))


def test_grid_from_centers_rejects_centers_outside_shape():
    with pytest.raises(ValueError, match="outside output shape"):
        grid_from_centers((2, 2), centers=[(2, 0)], values=[1])


def test_iter_centers_chunks_without_reordering():
    centers = centers_from_shape((2, 3))

    chunks = list(iter_centers(centers, chunk_size=2))

    assert len(chunks) == 3
    np.testing.assert_array_equal(chunks[0], [[0, 0], [0, 1]])
    np.testing.assert_array_equal(chunks[1], [[0, 2], [1, 0]])
    np.testing.assert_array_equal(chunks[2], [[1, 1], [1, 2]])


def test_iter_pattern_batches_prepares_chunked_patterns():
    layer = SpatialLayer("x", np.arange(16).reshape(4, 4), window_size=3)
    centers = np.array([[1, 1], [1, 2], [2, 1]])

    batches = list(iter_pattern_batches([layer], centers, chunk_size=2))

    assert len(batches) == 2
    np.testing.assert_array_equal(batches[0].centers, [[1, 1], [1, 2]])
    assert batches[0].patterns.shape == (2, 9)
    np.testing.assert_array_equal(batches[1].centers, [[2, 1]])
    assert batches[1].patterns.shape == (1, 9)


def test_iter_pattern_batches_preserves_centers_when_rotations_expand_rows():
    layer = SpatialLayer("x", np.arange(16).reshape(4, 4), window_size=3)

    batch = next(iter_pattern_batches([layer], [(1, 1), (2, 2)], chunk_size=2, rotations=True))

    np.testing.assert_array_equal(batch.centers, [[1, 1], [2, 2]])
    assert batch.patterns.shape == (8, 9)


def test_prepare_training_data_repeats_target_for_rotation_rows():
    layer = SpatialLayer("x", np.arange(36).reshape(6, 6), window_size=3)

    dataset = prepare_training_data(
        [layer],
        centers=[(2, 2), (3, 3)],
        target=np.array(["ore", "waste"]),
        rotations=True,
    )

    np.testing.assert_array_equal(dataset.centers, [[2, 2], [3, 3]])
    assert dataset.patterns.shape == (8, 9)
    np.testing.assert_array_equal(
        dataset.target,
        ["ore", "ore", "ore", "ore", "waste", "waste", "waste", "waste"],
    )


def test_prepare_training_data_validates_target_length():
    layer = SpatialLayer("x", np.arange(9).reshape(3, 3), window_size=3)

    with pytest.raises(ValueError, match="target length"):
        prepare_training_data([layer], centers=[(1, 1)], target=[])


_REPO_ROOT = Path(__file__).resolve().parents[1]
_RECOVERED_REALCASE = _REPO_ROOT / "original_source" / "code" / "realCase"
_RECOVERED_PATTERN_CSV = _RECOVERED_REALCASE / "recovered_from_RData" / "patbase_train_spatial.csv"
_MYSAM_CSV = _RECOVERED_REALCASE / "mysam.csv"
_DEM_TIF = _RECOVERED_REALCASE / "dem.tif"


@pytest.mark.skipif(
    not (_RECOVERED_PATTERN_CSV.exists() and _MYSAM_CSV.exists() and _DEM_TIF.exists()),
    reason="requires original_source/ (gitignored, not distributed) with the recovered R ground truth",
)
def test_even_window_matches_recovered_r_pattern_exactly():
    # Pins the even-window offset convention (see _extract_windows) against
    # a real pattern recovered by re-running the original R NWMP.R workflow
    # -- see .features/PLAN.md, "Validate Against Paper/Test Data".
    import rasterio
    from rasterio.windows import Window

    with open(_RECOVERED_PATTERN_CSV, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        row = next(reader)
    r_window = np.array([np.nan if value == "NA" else float(value) for value in row[:100]])

    with open(_MYSAM_CSV, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        cell_number = int(next(reader)[1])

    ncols = 311
    row_index, col_index = divmod(cell_number - 1, ncols)

    crop_window = Window(col_off=657, row_off=2692, width=311, height=311)
    with rasterio.open(_DEM_TIF) as dataset:
        values = dataset.read(1, window=crop_window, masked=True).astype(float).filled(np.nan)

    layer = SpatialLayer("dem", values, window_size=10)
    py_window = vectorize_layer(layer, [(row_index, col_index)])[0]

    np.testing.assert_allclose(py_window, r_window, equal_nan=True)

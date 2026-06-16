import numpy as np
import pytest

from spatialized import (
    GridTransform,
    SpatialLayer,
    centers_from_mask,
    centers_from_shape,
    iter_centers,
    iter_pattern_batches,
    pattern_size_from_edge,
    prepare_patterns,
    prepare_training_data,
    vectorize_layer,
)


def test_vectorize_layer_flattens_square_window_row_wise():
    layer = SpatialLayer("x", np.arange(25).reshape(5, 5), window_size=3)

    result = vectorize_layer(layer, [(2, 2)])

    np.testing.assert_array_equal(
        result,
        np.array([[6, 7, 8, 11, 12, 13, 16, 17, 18]], dtype=float),
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


def test_centers_outside_layer_bounds_raise():
    layer = SpatialLayer("x", np.arange(9).reshape(3, 3), window_size=3)

    with pytest.raises(ValueError, match="outside layer 'x'"):
        vectorize_layer(layer, [(-1, 0)])


def test_pattern_size_from_edge_matches_r_formula():
    assert pattern_size_from_edge(edge=3.99, cell_size=1) == 7
    assert pattern_size_from_edge(edge=2.99, cell_size=1) == 5


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

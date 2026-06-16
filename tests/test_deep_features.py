import numpy as np

from spatialized import (
    cluster_feature_vectors,
    extract_patches,
    normalize_channels,
    patch_centers,
)


def test_normalize_channels_returns_band_last_unit_stack():
    first = np.array([[0, 1], [2, 3]], dtype=float)
    second = np.array([[10, 10], [10, 10]], dtype=float)

    stack = normalize_channels([first, second])

    assert stack.shape == (2, 2, 2)
    np.testing.assert_array_equal(stack[:, :, 0], [[0, 1 / 3], [2 / 3, 1]])
    np.testing.assert_array_equal(stack[:, :, 1], np.zeros((2, 2)))


def test_patch_centers_and_extract_patches():
    image = np.arange(5 * 5 * 3).reshape(5, 5, 3)

    centers = patch_centers((5, 5), patch_size=3, stride=2)
    patches = extract_patches(image, centers, patch_size=3)

    np.testing.assert_array_equal(centers, [[1, 1], [1, 3], [3, 1], [3, 3]])
    assert patches.shape == (4, 3, 3, 3)
    np.testing.assert_array_equal(patches[0], image[0:3, 0:3, :])


def test_cluster_feature_vectors_maps_labels_to_grid():
    features = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.8],
        ]
    )
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    result = cluster_feature_vectors(
        features,
        centers,
        output_shape=(2, 2),
        n_clusters=2,
        n_components=2,
        random_state=7,
    )

    assert result.embedding.shape == (4, 2)
    assert result.labels.shape == (4,)
    assert result.label_grid.shape == (2, 2)
    assert set(result.labels) == {0, 1}

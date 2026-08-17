import numpy as np
import pytest

from spatialized import (
    cluster_feature_vectors,
    extract_patches,
    extract_resnet_features,
    normalize_channels,
    patch_centers,
    point_intensities_at_centers,
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


def test_patch_centers_and_extract_patches_support_even_patch_size():
    # The deep-feature workflow's documented patch size is 16x16 (even).
    image = np.arange(6 * 6 * 3).reshape(6, 6, 3)

    centers = patch_centers((6, 6), patch_size=4, stride=6)
    patches = extract_patches(image, centers, patch_size=4)

    # before=(4-1)//2=1, after=2 -> valid center rows/cols: [1, 6-2) = [1,4)
    np.testing.assert_array_equal(centers, [[1, 1]])
    assert patches.shape == (1, 4, 4, 3)
    np.testing.assert_array_equal(patches[0], image[0:4, 0:4, :])


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


def test_point_intensities_at_centers_reads_raw_channel_values():
    image = np.arange(5 * 5 * 3).reshape(5, 5, 3).astype(float)
    centers = np.array([[0, 0], [1, 1], [4, 4]])

    intensities = point_intensities_at_centers(image, centers)

    np.testing.assert_array_equal(intensities, image[[0, 1, 4], [0, 1, 4], :])


def test_cluster_feature_vectors_concatenates_point_intensities():
    features = np.array(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.8],
        ]
    )
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    point_intensities = np.array([[1.0], [1.1], [9.0], [9.2]])

    result = cluster_feature_vectors(
        features,
        centers,
        output_shape=(2, 2),
        n_clusters=2,
        n_components=2,
        random_state=7,
        point_intensities=point_intensities,
    )

    # embedding (2 UMAP/PCA dims) + 1 concatenated intensity column = 3 dims
    assert result.embedding.shape == (4, 3)
    np.testing.assert_array_equal(result.embedding[:, 2], point_intensities[:, 0])


def test_cluster_feature_vectors_rejects_mismatched_point_intensities():
    features = np.zeros((4, 2))
    centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    with pytest.raises(ValueError, match="point_intensities length"):
        cluster_feature_vectors(
            features,
            centers,
            output_shape=(2, 2),
            n_clusters=2,
            n_components=2,
            point_intensities=np.zeros((3, 1)),
        )


def test_extract_resnet_features_returns_2048_dim_vectors():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    patches = np.random.default_rng(0).random((2, 16, 16, 3)).astype(np.float32)
    features = extract_resnet_features(patches, device="cpu", batch_size=1)

    assert features.shape == (2, 2048)


def test_extract_resnet_features_rejects_wrong_channel_count():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    patches = np.zeros((2, 16, 16, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="3-channel"):
        extract_resnet_features(patches)

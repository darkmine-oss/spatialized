import numpy as np
import pytest

from spatialized import PatternEncoder


def test_pattern_encoder_uses_constant_missing_values_by_default():
    patterns = np.array([[1.0, "a"], [np.nan, None]], dtype=object)

    encoder = PatternEncoder().fit(patterns)
    encoded = encoder.transform(patterns)

    np.testing.assert_array_equal(encoded, [[1.0, 0.0], [-1.0, -1.0]])
    assert encoder.columns_[0].fill_value == -1.0
    assert encoder.columns_[1].fill_value == -1.0


def test_pattern_encoder_can_impute_numeric_mean_and_categorical_mode():
    patterns = np.array(
        [
            [1.0, "basalt"],
            [3.0, "basalt"],
            [np.nan, None],
            [5.0, "granite"],
        ],
        dtype=object,
    )

    encoder = PatternEncoder(
        numeric_missing_strategy="mean",
        categorical_missing_strategy="most_frequent",
    ).fit(patterns)
    encoded = encoder.transform(patterns)

    assert encoder.columns_[0].fill_value == 3.0
    assert encoder.columns_[1].categories == ("basalt", "granite")
    assert encoder.columns_[1].fill_value == 0.0
    np.testing.assert_array_equal(encoded[:, 0], [1.0, 3.0, 3.0, 5.0])
    np.testing.assert_array_equal(encoded[:, 1], [0.0, 0.0, 0.0, 1.0])


def test_pattern_encoder_metadata_is_json_ready():
    patterns = np.array([[1.0, "a"], [2.0, "b"]], dtype=object)

    payload = PatternEncoder().fit(patterns).to_dict()

    assert payload["numeric_missing_strategy"] == "constant"
    assert payload["columns"][1]["categories"] == ["a", "b"]


def test_pattern_encoder_rejects_unknown_strategies():
    with pytest.raises(ValueError, match="numeric_missing_strategy"):
        PatternEncoder(numeric_missing_strategy="median")


def test_pattern_encoder_window_mean_requires_layer_spans():
    with pytest.raises(ValueError, match="layer_spans"):
        PatternEncoder(numeric_missing_strategy="window_mean")


def test_pattern_encoder_window_mean_fills_from_local_window_first():
    # Two layers of two columns each: [layer_a_col0, layer_a_col1, layer_b_col0, layer_b_col1]
    patterns = np.array(
        [
            [1.0, 2.0, 5.0, 6.0],
            [1.0, np.nan, 5.0, 6.0],  # one missing cell in layer a's window
            [np.nan, np.nan, 5.0, 7.0],  # layer a's whole window missing -> fall back
            [3.0, 4.0, np.nan, np.nan],  # layer b's whole window missing -> fall back
        ]
    )

    encoder = PatternEncoder(
        numeric_missing_strategy="window_mean",
        layer_spans=((0, 2), (2, 4)),
    ).fit(patterns)
    encoded = encoder.transform(patterns)

    # Row 1: the missing cell is filled from the *other* value in the same
    # window/row (1.0), not the global column mean (which would be 1.6667).
    assert encoded[1, 1] == pytest.approx(1.0)

    # Row 2: the whole window is missing, so both cells fall back to the
    # fitted global column means.
    assert encoded[2, 0] == pytest.approx(encoder.columns_[0].fill_value)
    assert encoded[2, 1] == pytest.approx(encoder.columns_[1].fill_value)

    # Row 3: same fallback behaviour for layer b's window.
    assert encoded[3, 2] == pytest.approx(encoder.columns_[2].fill_value)
    assert encoded[3, 3] == pytest.approx(encoder.columns_[3].fill_value)

    # Untouched cells are passed through unchanged.
    np.testing.assert_array_equal(encoded[0], [1.0, 2.0, 5.0, 6.0])

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

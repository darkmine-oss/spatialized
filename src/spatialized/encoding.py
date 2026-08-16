"""Feature encoding for prepared spatial pattern matrices."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncodedColumn:
    """Encoding metadata for one prepared feature column."""

    index: int
    kind: str
    categories: tuple[object, ...] = ()
    fill_value: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "kind": self.kind,
            "categories": list(self.categories),
            "fill_value": self.fill_value,
        }


@dataclass
class PatternEncoder:
    """Convert mixed numeric/categorical pattern matrices to numeric arrays.

    ``numeric_missing_strategy="window_mean"`` is a spatially-aware fill: a
    missing cell is filled with the mean of the *other* non-missing cells from
    the same local window (same layer, same pattern row) rather than a single
    dataset-wide constant. This matters most at raster edges, where a whole
    window's worth of cells can be NaN-padded and a global column mean flattens
    out real local texture. It requires ``layer_spans`` (column ranges per
    source layer, e.g. from ``FeatureLayout``) so the encoder knows which
    columns belong to the same window; rows/layers with no valid cells at all
    fall back to the fitted global column mean, same as ``"mean"``. This is
    still a coarser approximation than the original R workflows' proximity-based
    ``randomForestSRC`` imputation (``na.action="na.impute"``), which imputes
    iteratively during tree growth, but it is local rather than global.
    """

    numeric_missing_strategy: str = "constant"
    categorical_missing_strategy: str = "constant"
    unknown_value: float = -1.0
    missing_value: float = -1.0
    layer_spans: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.numeric_missing_strategy not in {"constant", "mean", "window_mean"}:
            raise ValueError(
                "numeric_missing_strategy must be 'constant', 'mean', or 'window_mean'"
            )
        if self.categorical_missing_strategy not in {"constant", "most_frequent"}:
            raise ValueError(
                "categorical_missing_strategy must be 'constant' or 'most_frequent'"
            )
        if self.numeric_missing_strategy == "window_mean" and not self.layer_spans:
            raise ValueError("numeric_missing_strategy='window_mean' requires layer_spans")

    def fit(self, patterns: np.ndarray) -> "PatternEncoder":
        matrix = _as_2d(patterns)
        self.columns_: tuple[EncodedColumn, ...] = tuple(
            self._fit_column(matrix[:, index], index) for index in range(matrix.shape[1])
        )
        return self

    def fit_transform(self, patterns: np.ndarray) -> np.ndarray:
        return self.fit(patterns).transform(patterns)

    def transform(self, patterns: np.ndarray) -> np.ndarray:
        matrix = _as_2d(patterns)
        if not hasattr(self, "columns_"):
            raise ValueError("encoder has not been fitted")
        if matrix.shape[1] != len(self.columns_):
            raise ValueError("pattern column count does not match fitted encoder")

        encoded = np.empty(matrix.shape, dtype=float)
        raw_numeric: dict[int, np.ndarray] = {}
        for column in self.columns_:
            values = matrix[:, column.index]
            if column.kind == "numeric":
                if self.numeric_missing_strategy == "window_mean":
                    raw_numeric[column.index] = _numeric_values(values, missing_value=np.nan)
                encoded[:, column.index] = _numeric_values(
                    values,
                    self.missing_value if column.fill_value is None else column.fill_value,
                )
            else:
                encoded[:, column.index] = _categorical_values(
                    values,
                    column.categories,
                    unknown_value=self.unknown_value,
                    missing_value=(
                        self.missing_value if column.fill_value is None else column.fill_value
                    ),
                )

        if self.numeric_missing_strategy == "window_mean" and self.layer_spans:
            self._apply_window_mean(encoded, raw_numeric)
        return encoded

    def _apply_window_mean(
        self, encoded: np.ndarray, raw_numeric: dict[int, np.ndarray]
    ) -> None:
        for start, stop in self.layer_spans:
            span_indices = [index for index in range(start, stop) if index in raw_numeric]
            if not span_indices:
                continue
            span_raw = np.stack([raw_numeric[index] for index in span_indices], axis=1)
            missing = np.isnan(span_raw)
            if not missing.any():
                continue
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                row_means = np.nanmean(span_raw, axis=1)
            has_valid = ~np.isnan(row_means)
            for column_offset, index in enumerate(span_indices):
                rows_to_fill = missing[:, column_offset] & has_valid
                encoded[rows_to_fill, index] = row_means[rows_to_fill]

    def _fit_column(self, values: np.ndarray, index: int) -> EncodedColumn:
        if _is_numeric_column(values):
            numeric = _numeric_values(values, missing_value=np.nan)
            fill_value = self.missing_value
            if self.numeric_missing_strategy in {"mean", "window_mean"}:
                fill_value = float(np.nanmean(numeric)) if not np.all(np.isnan(numeric)) else 0.0
            return EncodedColumn(index=index, kind="numeric", fill_value=fill_value)
        categories = tuple(sorted(_non_missing_unique(values), key=lambda value: repr(value)))
        fill_value = self.missing_value
        if self.categorical_missing_strategy == "most_frequent" and categories:
            fill_value = float(categories.index(_most_frequent(values, categories)))
        return EncodedColumn(
            index=index,
            kind="categorical",
            categories=categories,
            fill_value=fill_value,
        )

    def to_dict(self) -> dict[str, object]:
        if not hasattr(self, "columns_"):
            raise ValueError("encoder has not been fitted")
        return {
            "numeric_missing_strategy": self.numeric_missing_strategy,
            "categorical_missing_strategy": self.categorical_missing_strategy,
            "unknown_value": self.unknown_value,
            "missing_value": self.missing_value,
            "layer_spans": None if self.layer_spans is None else [list(span) for span in self.layer_spans],
            "columns": [column.to_dict() for column in self.columns_],
        }


def _as_2d(patterns: np.ndarray) -> np.ndarray:
    matrix = np.asarray(patterns)
    if matrix.ndim != 2:
        raise ValueError("patterns must be a 2D array")
    return matrix


def _is_numeric_column(values: np.ndarray) -> bool:
    try:
        _numeric_values(values, missing_value=np.nan)
    except (TypeError, ValueError):
        return False
    return True


def _numeric_values(values: np.ndarray, missing_value: float) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    for index, value in enumerate(values):
        if _is_missing(value):
            result[index] = missing_value
        else:
            result[index] = float(value)
    return result


def _categorical_values(
    values: np.ndarray,
    categories: tuple[object, ...],
    *,
    unknown_value: float,
    missing_value: float,
) -> np.ndarray:
    lookup = {category: float(index) for index, category in enumerate(categories)}
    result = np.empty(len(values), dtype=float)
    for index, value in enumerate(values):
        if _is_missing(value):
            result[index] = missing_value
        else:
            result[index] = lookup.get(value, unknown_value)
    return result


def _non_missing_unique(values: np.ndarray) -> list[object]:
    unique: list[object] = []
    for value in values:
        if _is_missing(value):
            continue
        if not any(value == existing for existing in unique):
            unique.append(value)
    return unique


def _most_frequent(values: np.ndarray, categories: tuple[object, ...]) -> object:
    counts = []
    for category in categories:
        counts.append(sum(value == category for value in values if not _is_missing(value)))
    return categories[int(np.argmax(counts))]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False

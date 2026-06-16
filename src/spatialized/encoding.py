"""Feature encoding for prepared spatial pattern matrices."""

from __future__ import annotations

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
    """Convert mixed numeric/categorical pattern matrices to numeric arrays."""

    numeric_missing_strategy: str = "constant"
    categorical_missing_strategy: str = "constant"
    unknown_value: float = -1.0
    missing_value: float = -1.0

    def __post_init__(self) -> None:
        if self.numeric_missing_strategy not in {"constant", "mean"}:
            raise ValueError("numeric_missing_strategy must be 'constant' or 'mean'")
        if self.categorical_missing_strategy not in {"constant", "most_frequent"}:
            raise ValueError(
                "categorical_missing_strategy must be 'constant' or 'most_frequent'"
            )

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
        for column in self.columns_:
            values = matrix[:, column.index]
            if column.kind == "numeric":
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
        return encoded

    def _fit_column(self, values: np.ndarray, index: int) -> EncodedColumn:
        if _is_numeric_column(values):
            numeric = _numeric_values(values, missing_value=np.nan)
            fill_value = self.missing_value
            if self.numeric_missing_strategy == "mean":
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

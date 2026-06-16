"""Spatial random forest preparation utilities."""

from .encoding import EncodedColumn, PatternEncoder
from .models import (
    PredictionBatch,
    SpatialRandomForestClassifier,
    SpatialRandomForestRegressor,
    classification_entropy,
)
from .patterns import (
    FeatureLayout,
    FeatureSpec,
    GridTransform,
    PatternBatch,
    PatternDataset,
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
from .raster import RasterGrid, read_raster, read_spatial_layer, write_raster
from .unsupervised import (
    UnsupervisedResult,
    UnsupervisedSpatialRandomForest,
    synthetic_patterns,
)
from .workflows import GridPrediction, predict_grid, predict_grid_to_raster

__all__ = [
    "EncodedColumn",
    "FeatureLayout",
    "FeatureSpec",
    "GridTransform",
    "GridPrediction",
    "PatternBatch",
    "PatternDataset",
    "PatternEncoder",
    "PredictionBatch",
    "RasterGrid",
    "SpatialRandomForestClassifier",
    "SpatialRandomForestRegressor",
    "SpatialLayer",
    "UnsupervisedResult",
    "UnsupervisedSpatialRandomForest",
    "centers_from_mask",
    "centers_from_shape",
    "classification_entropy",
    "feature_layout",
    "grid_from_centers",
    "iter_centers",
    "iter_pattern_batches",
    "pattern_size_from_edge",
    "prepare_patterns",
    "prepare_training_data",
    "predict_grid",
    "predict_grid_to_raster",
    "read_raster",
    "read_spatial_layer",
    "synthetic_patterns",
    "vectorize_layer",
    "write_raster",
    "zone_of_influence",
]

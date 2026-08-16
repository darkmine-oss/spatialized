"""Validate the SRF classification workflow against the R realCase geology case.

Replicates the workflow in ``original_source/code/realCase/NWMP.R``: crop
dem/radk/radth/radu/grav/magrtp/magvd rasters to the study extent, build an
8-class lithology target from ``geomask.tif``, fit a rotation-augmented
spatial random forest classifier on the same 672-cell stratified training
sample the R script used (``mysam.csv``), predict over the full grid, and
compare against the spatial-SRF reference recovered by actually re-running
``NWMP.R`` in R (see ``original_source/code/realCase/recovered_from_RData/``,
whose ``ctsp_recovered.csv`` overall accuracy of 77.77% already matches the
original authors' saved ``ctsp.csv`` at 77.89%).

``SpatialLayer`` now supports the R script's true even ``window_size`` (10x10,
matching ``myband <- res * 5``), so no window-size approximation is needed
here (see the Cu-regression validation script for that history).

Run with:

    python examples/validate_realcase_geology_classification.py --output-dir /tmp/geology_validation
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from spatialized import (
    GridTransform,
    RasterGrid,
    SpatialLayer,
    SpatialRandomForestClassifier,
    write_raster,
)
from spatialized.workflows import predict_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "original_source" / "code" / "realCase"
RECOVERED_DIR = DATA_DIR / "recovered_from_RData"

LAYER_NAMES = ["dem", "radk", "radth", "radu", "grav", "magrtp", "magvd"]

# Verified against the R script's cropped raster extent
# (422044.1, 450034.1, 7606011, 7634001; 311x311 cells) — same extent NWMP.R
# and NWMP.R (realCase2) both use, and the same crop window used by
# validate_realcase2_cu_regression.py.
CROP_WINDOW = Window(col_off=657, row_off=2692, width=311, height=311)

# R's as.factor() on the sorted class codes {4,5,9,13,28,33,34,35} assigns
# levels 1..8 in ascending order; the recovered/reference rasters store the
# factor level (1-8), not the raw geomask class code.
FACTOR_ID_TO_CLASS_CODE = {1: 4, 2: 5, 3: 9, 4: 13, 5: 28, 6: 33, 7: 34, 8: 35}


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_grids = {}
    transform = crs = None
    for name in LAYER_NAMES:
        values, transform, crs, _ = _read_cropped(DATA_DIR / f"{name}.tif")
        layer_grids[name] = values
    shape = layer_grids["dem"].shape
    print(f"Cropped grid shape: {shape}, cell size {transform.x_size}")

    geomask, _, _, _ = _read_cropped(DATA_DIR / "geomask.tif")
    geomask[geomask == 22] = np.nan
    valid_mask = ~np.isnan(geomask)
    print(f"geomask: {geomask.size} cells total, {int(valid_mask.sum())} valid (non-22, non-NaN) cells")

    training_centers = _load_mysam_centers(DATA_DIR / "mysam.csv", shape)
    target = geomask[training_centers[:, 0], training_centers[:, 1]]
    print(f"Training sample: {len(training_centers)} cells (matches R's mysam.csv)")

    layers = [
        SpatialLayer(name, layer_grids[name], window_size=args.window_size)
        for name in LAYER_NAMES
    ]

    model = SpatialRandomForestClassifier(
        n_estimators=args.trees,
        random_state=args.seed,
        n_jobs=-1,
        encoder_kwargs={"numeric_missing_strategy": args.imputation},
        estimator_kwargs={"oob_score": True, "bootstrap": True},
    )
    print(
        f"Fitting SpatialRandomForestClassifier: {args.trees} trees, "
        f"window_size={args.window_size}, imputation={args.imputation}, rotations=True "
        f"({len(training_centers)} centers -> {4 * len(training_centers)} rotated rows)"
    )
    model.fit(layers, training_centers, target, rotations=True)

    oob_pred = model.oob_prediction()
    oob_accuracy = model.oob_accuracy_score()
    print(f"Python SRF OOB accuracy (rotation rows, not rotation-averaged): {oob_accuracy:.4f}")

    print("Predicting full grid...")
    prediction = predict_grid(
        model,
        layers,
        prediction_mask=valid_mask,
        chunk_size=args.chunk_size,
    )

    reference = RasterGrid(
        values=layer_grids["dem"].astype(np.float32),
        transform=transform,
        crs=crs,
        nodata=-9999,
    )
    pred_path = output_dir / "channelClass_python_srf.tif"
    pred_int = np.where(np.isnan(prediction.prediction.astype(float)), -9999, prediction.prediction)
    write_raster(pred_path, pred_int.astype(np.int16), reference, dtype="int16", nodata=-9999)
    print(f"Wrote prediction raster: {pred_path}")

    # --- compare against the recovered R spatial-SRF reference ---
    ref_path = RECOVERED_DIR / "channelClass_recovered.tif"
    if ref_path.exists():
        ref_ids, ref_nodata = _read_reference(ref_path)
        ref_codes = np.vectorize(lambda v: FACTOR_ID_TO_CLASS_CODE.get(int(v), np.nan) if not np.isnan(v) else np.nan)(ref_ids)

        py_pred = prediction.prediction.astype(float)
        mask = valid_mask & ~np.isnan(ref_codes)
        agreement = float(np.mean(py_pred[mask] == ref_codes[mask]))
        n_compared = int(mask.sum())
        print(f"Python vs recovered-R spatial-SRF class agreement: {agreement:.4f} over {n_compared} cells")

        py_accuracy = float(np.mean(py_pred[mask] == geomask[mask]))
        r_accuracy = float(np.mean(ref_codes[mask] == geomask[mask]))
        print(f"Python full-grid accuracy vs true geomask (same cells): {py_accuracy:.4f}")
        print(f"Recovered-R full-grid accuracy vs true geomask (same cells): {r_accuracy:.4f}")
    else:
        agreement = None
        n_compared = 0
        py_accuracy = float(np.mean(prediction.prediction.astype(float)[valid_mask] == geomask[valid_mask]))
        r_accuracy = None
        print(f"No recovered R reference found at {ref_path}; skipping comparison")

    summary = {
        "grid_shape": list(shape),
        "n_training_cells": int(len(training_centers)),
        "window_size_used": args.window_size,
        "n_trees": args.trees,
        "imputation": args.imputation,
        "python_srf_oob_accuracy": oob_accuracy,
        "python_full_grid_accuracy_vs_geomask": py_accuracy,
        "recovered_r_full_grid_accuracy_vs_geomask": r_accuracy,
        "python_vs_recovered_r_class_agreement": agreement,
        "n_cells_compared": n_compared,
        "r_ctsp_recovered_overall_accuracy": 0.7777,
        "r_ctsp_original_overall_accuracy": 0.7789,
        "outputs": [str(pred_path)],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _read_cropped(path: Path):
    with rasterio.open(path) as dataset:
        values = dataset.read(1, window=CROP_WINDOW, masked=True).astype(float)
        values = np.asarray(values.filled(np.nan), dtype=float)
        window_transform = dataset.window_transform(CROP_WINDOW)
        return values, GridTransform.from_affine(window_transform), dataset.crs, dataset.nodata


def _read_reference(path: Path):
    with rasterio.open(path) as dataset:
        values = np.asarray(dataset.read(1), dtype=float)
        return values, dataset.nodata


def _load_mysam_centers(path: Path, shape: tuple[int, int]) -> np.ndarray:
    ncols = shape[1]
    centers = []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header
        for row in reader:
            cell_number = int(row[1])  # 1-indexed, row-major, top-left origin
            idx0 = cell_number - 1
            r, c = divmod(idx0, ncols)
            centers.append((r, c))
    return np.array(centers, dtype=int)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("geology_validation_outputs"))
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument(
        "--imputation",
        choices=["mean", "window_mean"],
        default="mean",
        help="numeric_missing_strategy passed to PatternEncoder via encoder_kwargs",
    )
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()

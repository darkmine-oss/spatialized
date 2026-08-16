"""Validate the SRF regression workflow against the R realCase2 Cu case study.

Replicates (as closely as the current Python API allows) the workflow in
``original_source/code/realCase2/NWMP.R``: crop grav/magrtp/magvd rasters to
the study extent, build a Cu target grid from ``geochemgrid.csv``, fit a
rotation-augmented spatial random forest regressor on the ~3754 non-NaN Cu
cells, predict over the full grid, and compare against the R script's saved
outputs (``gheochemcuSRF.tif``, ``gheochemcuRF.tif``, ``CuPredFinalRF.csv``).

The window size now matches the R script exactly: ``SpatialLayer`` supports
even window sizes (R's ``myband <- res * 5`` gives a 10x10 pattern with no
unique center pixel), so this uses ``window_size=10`` by default.

Remaining known approximation vs. the R script (see report for discussion):
  * mean-imputation of missing pattern values at fit/predict time instead of
    randomForestSRC's own na.action="na.impute" multiple-imputation-like fill.

Run with:

    python examples/validate_realcase2_cu_regression.py --output-dir /tmp/cu_validation
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
    SpatialRandomForestRegressor,
    centers_from_mask,
    write_raster,
)
from spatialized.workflows import predict_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "original_source" / "code" / "realCase2"

# Verified against the R script's cropped raster extent
# (422044.1, 450034.1, 7606011, 7634001; 311x311 cells) by matching
# gheochemcuSRF.tif / gheochemcuRF.tif bounds exactly.
CROP_WINDOW = Window(col_off=657, row_off=2692, width=311, height=311)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grav, transform, crs, nodata = _read_cropped(DATA_DIR / "grav.tif")
    magrtp, _, _, _ = _read_cropped(DATA_DIR / "magrtp.tif")
    magvd, _, _, _ = _read_cropped(DATA_DIR / "magvd.tif")
    shape = grav.shape
    print(f"Cropped grid shape: {shape}, transform left/top: "
          f"{transform.left:.3f},{transform.top:.3f}, cell size {transform.x_size}")

    cu_grid = _load_cu_grid(DATA_DIR / "geochemgrid.csv", shape)
    n_valid = int(np.sum(~np.isnan(cu_grid)))
    print(f"Cu target grid: {cu_grid.size} cells total, {n_valid} non-NaN training cells")

    layers = [
        SpatialLayer("grav", grav, window_size=args.window_size),
        SpatialLayer("magrtp", magrtp, window_size=args.window_size),
        SpatialLayer("magvd", magvd, window_size=args.window_size),
    ]

    training_centers = np.argwhere(~np.isnan(cu_grid))
    target = cu_grid[training_centers[:, 0], training_centers[:, 1]]

    model = SpatialRandomForestRegressor(
        n_estimators=args.trees,
        random_state=args.seed,
        n_jobs=-1,
        encoder_kwargs={"numeric_missing_strategy": args.imputation},
        estimator_kwargs={"oob_score": True, "bootstrap": True},
    )
    print(f"Fitting SpatialRandomForestRegressor: {args.trees} trees, "
          f"window_size={args.window_size}, rotations=True "
          f"({len(training_centers)} centers -> {4 * len(training_centers)} rotated rows)")
    model.fit(layers, training_centers, target, rotations=True)

    # OOB predictions, averaged across the 4 rotation variants per original
    # sample (mirrors the R script's (oob[i]+oob[i+n]+oob[i+2n]+oob[i+3n])/4).
    oob_pred = model.estimator.oob_prediction_.reshape(len(training_centers), 4).mean(axis=1)
    oob_valid = np.isfinite(oob_pred)
    srf_oob_r2 = _r2(target[oob_valid], oob_pred[oob_valid])
    srf_oob_corr = float(np.corrcoef(target[oob_valid], oob_pred[oob_valid])[0, 1])
    print(f"Python SRF OOB R2 (rotation-averaged): {srf_oob_r2:.4f} over "
          f"{oob_valid.sum()}/{len(training_centers)} samples with finite OOB predictions")

    print("Predicting full grid...")
    prediction = predict_grid(
        model,
        layers,
        prediction_mask=np.ones(shape, dtype=bool),
        chunk_size=args.chunk_size,
    )

    reference = RasterGrid(
        values=grav.astype(np.float32),
        transform=transform,
        crs=crs,
        nodata=-3.4e38,
    )
    pred_path = output_dir / "gheochemcu_python_srf.tif"
    write_raster(pred_path, prediction.prediction.astype(np.float32), reference, dtype="float32")
    print(f"Wrote prediction raster: {pred_path}")

    # --- classical (non-spatial) RF baseline, matching NWMP_nonspatial.R ---
    from sklearn.ensemble import RandomForestRegressor

    rows, cols = training_centers[:, 0], training_centers[:, 1]
    X_classical = np.column_stack([grav[rows, cols], magrtp[rows, cols], magvd[rows, cols]])
    rf_classical = RandomForestRegressor(
        n_estimators=args.trees,
        random_state=args.seed,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    rf_classical.fit(X_classical, target)
    rf_oob_pred = rf_classical.oob_prediction_
    rf_valid = np.isfinite(rf_oob_pred)
    rf_oob_r2 = _r2(target[rf_valid], rf_oob_pred[rf_valid])
    print(f"Python classical RF OOB R2: {rf_oob_r2:.4f} over {rf_valid.sum()}/{len(target)} samples")

    # --- comparisons against R script's saved outputs ---
    ref_srf, ref_srf_nodata = _read_reference(DATA_DIR / "gheochemcuSRF.tif")
    ref_rf, ref_rf_nodata = _read_reference(DATA_DIR / "gheochemcuRF.tif")

    py_pred = prediction.prediction.astype(float)
    mask_srf = np.isfinite(py_pred) & (ref_srf != ref_srf_nodata) & np.isfinite(ref_srf)
    mask_rf = np.isfinite(py_pred) & (ref_rf != ref_rf_nodata) & np.isfinite(ref_rf)

    corr_vs_r_srf = float(np.corrcoef(py_pred[mask_srf], ref_srf[mask_srf])[0, 1])
    r2_vs_r_srf = _r2(ref_srf[mask_srf], py_pred[mask_srf])
    corr_vs_r_rf = float(np.corrcoef(py_pred[mask_rf], ref_rf[mask_rf])[0, 1])
    r2_vs_r_rf = _r2(ref_rf[mask_rf], py_pred[mask_rf])

    cu_pred_final_rf = _load_cu_pred_final_rf(DATA_DIR / "CuPredFinalRF.csv")
    rf_corr_vs_r = None
    if len(cu_pred_final_rf) == len(rf_oob_pred):
        rf_corr_vs_r = float(np.corrcoef(rf_oob_pred, cu_pred_final_rf)[0, 1])

    summary = {
        "grid_shape": list(shape),
        "n_training_cells": int(n_valid),
        "window_size_used": args.window_size,
        "imputation_strategy": args.imputation,
        "n_trees": args.trees,
        "python_srf_oob_r2": srf_oob_r2,
        "python_srf_oob_corr": srf_oob_corr,
        "python_classical_rf_oob_r2": rf_oob_r2,
        "r_paper_reported_srf_r2": 0.684,
        "r_paper_reported_rf_r2": 0.602,
        "python_pred_vs_r_srf_tif": {"pearson_r": corr_vs_r_srf, "r2": r2_vs_r_srf, "n_cells": int(mask_srf.sum())},
        "python_pred_vs_r_rf_tif": {"pearson_r": corr_vs_r_rf, "r2": r2_vs_r_rf, "n_cells": int(mask_rf.sum())},
        "python_classical_rf_oob_vs_r_CuPredFinalRF_corr": rf_corr_vs_r,
        "outputs": [str(pred_path)],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _read_cropped(path: Path):
    with rasterio.open(path) as dataset:
        values = dataset.read(1, window=CROP_WINDOW, masked=True)
        values = np.asarray(values.filled(np.nan), dtype=float)
        window_transform = dataset.window_transform(CROP_WINDOW)
        return values, GridTransform.from_affine(window_transform), dataset.crs, dataset.nodata


def _read_reference(path: Path):
    with rasterio.open(path) as dataset:
        values = np.asarray(dataset.read(1), dtype=float)
        return values, dataset.nodata


def _load_cu_grid(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values: list[float] = []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            cu = row[2].strip()
            values.append(np.nan if cu in ("NA", "") else float(cu))
    if len(values) != shape[0] * shape[1]:
        raise ValueError(
            f"geochemgrid.csv has {len(values)} rows but grid has {shape[0] * shape[1]} cells"
        )
    return np.array(values, dtype=float).reshape(shape)


def _load_cu_pred_final_rf(path: Path) -> np.ndarray:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return np.array([float(row[0]) for row in reader], dtype=float)


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    sse = np.sum((observed - predicted) ** 2)
    sst = np.sum((observed - np.mean(observed)) ** 2)
    return float(1.0 - sse / sst)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("cu_validation_outputs"))
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

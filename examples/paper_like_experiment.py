"""Paper-style synthetic SRF experiment.

This workflow mimics the structure of the Math Geoscience SRF experiments
without using the paper data:

- multi-resolution magnetic and gravity-like covariates
- sampled training cells from an interpreted geology-like class grid
- rotation-augmented spatial random forest classification
- full-grid prediction, entropy, and zone-of-influence rasters
- unsupervised SRF clustering from vectorised potential-field patterns

Run with:

    python examples/paper_like_experiment.py --output-dir /tmp/spatialized-paper-like
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spatialized import (
    GridTransform,
    RasterGrid,
    SpatialLayer,
    SpatialRandomForestClassifier,
    UnsupervisedSpatialRandomForest,
    centers_from_mask,
    grid_from_centers,
    predict_grid,
    write_raster,
)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    covariates = _make_covariates(args.size, rng)
    classes = _make_geology_classes(covariates["mag_rtp"], covariates["gravity_high_res"])
    training_centers = _stratified_training_centers(classes, samples_per_class=args.samples_per_class, rng=rng)

    layers = [
        SpatialLayer(
            "mag_rtp",
            covariates["mag_rtp"],
            window_size=7,
            transform=covariates["prediction_transform"],
        ),
        SpatialLayer(
            "mag_1vd",
            covariates["mag_1vd"],
            window_size=7,
            transform=covariates["prediction_transform"],
        ),
        SpatialLayer(
            "gravity",
            covariates["gravity"],
            window_size=5,
            transform=covariates["gravity_transform"],
        ),
    ]

    classifier = SpatialRandomForestClassifier(
        n_estimators=args.trees,
        random_state=args.seed,
        n_jobs=1,
        encoder_kwargs={"numeric_missing_strategy": "mean"},
    )
    classifier.fit(
        layers,
        training_centers,
        classes[training_centers[:, 0], training_centers[:, 1]],
        prediction_transform=covariates["prediction_transform"],
        rotations=True,
    )

    prediction = predict_grid(
        classifier,
        layers,
        prediction_mask=np.ones(classes.shape, dtype=bool),
        chunk_size=args.chunk_size,
        prediction_transform=covariates["prediction_transform"],
        entropy=True,
    )
    zones = classifier.zone_of_influence(layers)

    unsupervised_centers = _sample_centers(
        centers_from_mask(np.ones(classes.shape, dtype=bool)),
        count=args.unsupervised_samples,
        rng=rng,
    )
    unsupervised = UnsupervisedSpatialRandomForest(
        n_estimators=max(25, args.trees // 2),
        random_state=args.seed,
        n_jobs=1,
        encoder_kwargs={"numeric_missing_strategy": "mean"},
    )
    unsupervised.fit(
        layers,
        unsupervised_centers,
        prediction_transform=covariates["prediction_transform"],
        rotations=True,
    )
    clusters = unsupervised.spectral_cluster(n_clusters=args.clusters)
    cluster_grid = grid_from_centers(
        classes.shape,
        unsupervised_centers,
        clusters.astype(np.int16),
        fill_value=-1,
    )

    reference = RasterGrid(
        values=np.zeros(classes.shape, dtype=np.float32),
        transform=covariates["prediction_transform"],
        nodata=-9999,
    )
    write_raster(output_dir / "covariate_mag_rtp.tif", covariates["mag_rtp"].astype(np.float32), reference)
    write_raster(output_dir / "covariate_mag_1vd.tif", covariates["mag_1vd"].astype(np.float32), reference)
    write_raster(output_dir / "truth_classes.tif", classes.astype(np.int16), reference, dtype="int16")
    write_raster(output_dir / "predicted_classes.tif", prediction.prediction, reference, dtype="int16")
    write_raster(output_dir / "prediction_entropy.tif", prediction.entropy, reference, dtype="float32")
    write_raster(output_dir / "unsupervised_clusters_sampled.tif", cluster_grid, reference, dtype="int16")

    for layer_name, zone in zones.items():
        zone_reference = RasterGrid(
            values=np.zeros(zone.shape, dtype=np.float32),
            transform=GridTransform(left=0, top=zone.shape[0], x_size=1),
        )
        write_raster(output_dir / f"zone_of_influence_{layer_name}.tif", zone, zone_reference, dtype="float32")

    summary = {
        "seed": args.seed,
        "grid_shape": list(classes.shape),
        "training_samples": int(len(training_centers)),
        "unsupervised_samples": int(len(unsupervised_centers)),
        "class_counts": _counts(classes.ravel()),
        "predicted_class_counts": _counts(prediction.prediction.ravel()),
        "unsupervised_cluster_counts": _counts(clusters),
        "outputs": sorted(path.name for path in output_dir.glob("*.tif")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _make_covariates(size: int, rng: np.random.Generator) -> dict[str, np.ndarray | GridTransform]:
    rows, cols = np.indices((size, size))
    x = cols / size
    y = rows / size
    fault = np.tanh((x - y + 0.08 * np.sin(8 * y)) * 18)
    fold = np.sin(7 * x + 3 * np.sin(5 * y))
    intrusive = np.exp(-(((x - 0.68) ** 2) / 0.012 + ((y - 0.32) ** 2) / 0.02))
    basin = np.exp(-(((x - 0.28) ** 2) / 0.025 + ((y - 0.72) ** 2) / 0.018))

    mag_rtp = 60 * fold + 90 * intrusive - 30 * basin + 15 * fault
    mag_rtp += rng.normal(0, 4, size=(size, size))
    mag_1vd = np.gradient(mag_rtp, axis=0)

    coarse_size = size // 2
    coarse_rows, coarse_cols = np.indices((coarse_size, coarse_size))
    cx = coarse_cols / coarse_size
    cy = coarse_rows / coarse_size
    gravity = 120 * np.exp(-(((cx - 0.3) ** 2) / 0.08 + ((cy - 0.7) ** 2) / 0.05))
    gravity -= 70 * np.exp(-(((cx - 0.72) ** 2) / 0.05 + ((cy - 0.28) ** 2) / 0.05))
    gravity += 20 * np.sin(3 * cx + 4 * cy)
    gravity += rng.normal(0, 2, size=(coarse_size, coarse_size))
    gravity_high_res = np.repeat(np.repeat(gravity, 2, axis=0), 2, axis=1)[:size, :size]

    return {
        "mag_rtp": mag_rtp,
        "mag_1vd": mag_1vd,
        "gravity": gravity,
        "gravity_high_res": gravity_high_res,
        "prediction_transform": GridTransform(left=0, top=size, x_size=1),
        "gravity_transform": GridTransform(left=0, top=size, x_size=2),
    }


def _make_geology_classes(mag_rtp: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    classes = np.zeros(mag_rtp.shape, dtype=np.int16)
    classes[(mag_rtp > np.quantile(mag_rtp, 0.67)) & (gravity < np.quantile(gravity, 0.55))] = 1
    classes[(gravity > np.quantile(gravity, 0.7))] = 2
    classes[(mag_rtp < np.quantile(mag_rtp, 0.22)) & (gravity < np.quantile(gravity, 0.45))] = 3
    return classes


def _stratified_training_centers(
    classes: np.ndarray,
    *,
    samples_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = []
    for class_id in np.unique(classes):
        class_centers = np.argwhere(classes == class_id)
        centers.append(_sample_centers(class_centers, count=samples_per_class, rng=rng))
    return np.vstack(centers)


def _sample_centers(
    centers: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = min(count, len(centers))
    selected = rng.choice(len(centers), size=count, replace=False)
    return centers[selected]


def _counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("paper_like_outputs"))
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--samples-per-class", type=int, default=24)
    parser.add_argument("--unsupervised-samples", type=int, default=96)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--trees", type=int, default=120)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Synthetic spatialized workflow.

Run with:

    python examples/synthetic_workflow.py
"""

from __future__ import annotations

import numpy as np

from spatialized import (
    SpatialLayer,
    SpatialRandomForestClassifier,
    UnsupervisedSpatialRandomForest,
    centers_from_shape,
    predict_grid,
)


def main() -> None:
    rows, cols = np.indices((12, 12))
    mag = SpatialLayer("mag", rows + cols, window_size=3)
    grav = SpatialLayer("gravity", rows - cols, window_size=3)
    centers = centers_from_shape((12, 12))
    labels = np.where(centers[:, 0] + centers[:, 1] > 12, "high", "low")

    classifier = SpatialRandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit([mag, grav], centers, labels, rotations=True)

    prediction = predict_grid(
        classifier,
        [mag, grav],
        prediction_mask=np.ones((12, 12), dtype=bool),
        chunk_size=50,
        entropy=True,
    )
    zones = classifier.zone_of_influence([mag, grav])

    unsupervised = UnsupervisedSpatialRandomForest(n_estimators=100, random_state=42)
    unsupervised.fit([mag, grav], centers[::4], rotations=True)
    clusters = unsupervised.spectral_cluster(n_clusters=2)

    print("prediction grid:", prediction.prediction.shape)
    print("entropy grid:", prediction.entropy.shape)
    print("mag zone:", zones["mag"].shape)
    print("unsupervised clusters:", np.bincount(clusters))


if __name__ == "__main__":
    main()

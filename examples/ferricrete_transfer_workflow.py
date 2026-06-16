"""Synthetic ferricrete/paleovalley target-proxy transfer workflow.

This mirrors the ferricrete paper structure without restricted data: train a
classifier from labelled target units in one area, then predict similar
magnetic-pattern target proxies into another area.

Run with:

    python examples/ferricrete_transfer_workflow.py
"""

from __future__ import annotations

import numpy as np

from spatialized import SpatialLayer, predict_target_proxy_transfer


def main() -> None:
    train = _magnetic_area(32, shift=0.0)
    target = _magnetic_area(32, shift=0.12)
    train_mask = train > np.quantile(train, 0.82)

    result = predict_target_proxy_transfer(
        [SpatialLayer("mag_hf_1vd", train, window_size=5)],
        [SpatialLayer("mag_hf_1vd", target, window_size=5)],
        train_mask,
        target_prediction_mask=np.ones(target.shape, dtype=bool),
        max_background=int(train_mask.sum()),
        random_state=42,
        rotations=True,
        chunk_size=128,
    )

    print("training samples:", len(result.training_centers))
    print("prediction grid:", result.prediction.prediction.shape)
    print("probability grid:", result.prediction.probabilities.shape)
    print("entropy grid:", result.prediction.entropy.shape)


def _magnetic_area(size: int, *, shift: float) -> np.ndarray:
    rows, cols = np.indices((size, size))
    x = cols / size
    y = rows / size
    channel = np.sin(25 * (x + shift + 0.2 * np.sin(4 * y)))
    dendritic = np.maximum(channel, 0) * np.exp(-((y - 0.55) ** 2) / 0.15)
    background = 0.3 * np.sin(4 * x + 2 * y)
    return dendritic + background


if __name__ == "__main__":
    main()

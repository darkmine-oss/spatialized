"""Command-line entrypoints for spatialized workflows."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from .patterns import SpatialLayer, feature_layout
from .raster import read_raster, read_spatial_layer
from .workflows import predict_grid_to_raster


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spatialized")
    subparsers = parser.add_subparsers(dest="command", required=True)

    layout_parser = subparsers.add_parser(
        "feature-layout",
        help="write feature layout metadata for layer specs",
    )
    layout_parser.add_argument(
        "--layer",
        action="append",
        required=True,
        metavar="NAME:WINDOW[:SPARSE]",
        help="layer metadata, e.g. mag:7 or mag:7:0,4,8",
    )
    layout_parser.add_argument(
        "--rotations",
        action="store_true",
        help="mark layout as rotation-augmented training metadata",
    )
    layout_parser.add_argument("--output", type=Path, help="optional JSON output path")
    layout_parser.set_defaults(func=_feature_layout_command)

    predict_parser = subparsers.add_parser(
        "predict-grid",
        help="run chunked full-grid prediction from a pickled fitted model",
    )
    predict_parser.add_argument("--model", required=True, type=Path, help="pickle file")
    predict_parser.add_argument(
        "--layer",
        action="append",
        required=True,
        metavar="NAME:PATH:WINDOW[:SPARSE]",
        help="GeoTIFF layer spec, e.g. mag:mag.tif:7 or mag:mag.tif:7:0,4,8",
    )
    predict_parser.add_argument("--mask-raster", required=True, type=Path)
    predict_parser.add_argument("--output", required=True, type=Path)
    predict_parser.add_argument("--entropy-output", type=Path)
    predict_parser.add_argument("--probabilities-output", type=Path)
    predict_parser.add_argument("--chunk-size", type=int, default=10_000)
    predict_parser.add_argument("--dtype")
    predict_parser.add_argument(
        "--mask-mode",
        choices=("valid", "nan", "nodata", "all"),
        default="valid",
        help="cells to predict from mask raster",
    )
    predict_parser.set_defaults(func=_predict_grid_command)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


def _feature_layout_command(args: argparse.Namespace) -> None:
    layers = [_metadata_layer_from_spec(spec) for spec in args.layer]
    payload = feature_layout(layers, rotations=args.rotations).to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")


def _predict_grid_command(args: argparse.Namespace) -> None:
    with args.model.open("rb") as handle:
        model = pickle.load(handle)
    layers = [_raster_layer_from_spec(spec) for spec in args.layer]
    reference = read_raster(args.mask_raster, masked=True)
    mask = _prediction_mask(reference.values, reference.nodata, mode=args.mask_mode)
    predict_grid_to_raster(
        model,
        layers,
        mask,
        reference,
        args.output,
        chunk_size=args.chunk_size,
        prediction_transform=reference.transform,
        entropy_path=args.entropy_output,
        probabilities_path=args.probabilities_output,
        dtype=args.dtype,
    )


def _metadata_layer_from_spec(spec: str) -> SpatialLayer:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("layout layer specs must be NAME:WINDOW[:SPARSE]")
    name = parts[0]
    window_size = int(parts[1])
    sparse = _parse_sparse(parts[2]) if len(parts) == 3 else None
    return SpatialLayer(
        name=name,
        values=np.zeros((1, 1)),
        window_size=window_size,
        sparse_indices=sparse,
    )


def _raster_layer_from_spec(spec: str) -> SpatialLayer:
    parts = spec.split(":")
    if len(parts) not in {3, 4}:
        raise ValueError("raster layer specs must be NAME:PATH:WINDOW[:SPARSE]")
    name, path, window_size = parts[:3]
    sparse = _parse_sparse(parts[3]) if len(parts) == 4 else None
    return read_spatial_layer(
        path,
        name=name,
        window_size=int(window_size),
        sparse_indices=sparse,
    )


def _parse_sparse(value: str) -> list[int]:
    if value == "":
        return []
    return [int(part) for part in value.split(",")]


def _prediction_mask(values: np.ndarray, nodata: object, *, mode: str) -> np.ndarray:
    if mode == "all":
        return np.ones(values.shape, dtype=bool)
    if mode == "nan":
        return np.isnan(values)
    if mode == "nodata":
        if nodata is None:
            raise ValueError("mask raster has no nodata value")
        return values == nodata
    return np.isfinite(values)


if __name__ == "__main__":
    raise SystemExit(main())

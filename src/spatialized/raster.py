"""Optional rasterio adapters for GeoTIFF workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .patterns import GridTransform, SpatialLayer


@dataclass(frozen=True)
class RasterGrid:
    """Array data plus raster metadata needed for GeoTIFF output."""

    values: np.ndarray
    transform: GridTransform
    crs: object | None = None
    nodata: float | int | None = None
    profile: dict[str, object] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape


def read_raster(
    path: str | Path,
    *,
    band: int = 1,
    masked: bool = False,
) -> RasterGrid:
    """Read a single raster band into an array and lightweight metadata."""

    rasterio = _import_rasterio()
    with rasterio.open(path) as dataset:
        values = dataset.read(band, masked=masked)
        if masked:
            values = values.filled(np.nan)
        return RasterGrid(
            values=np.asarray(values),
            transform=GridTransform.from_affine(dataset.transform),
            crs=dataset.crs,
            nodata=dataset.nodata,
            profile=dataset.profile.copy(),
        )


def read_spatial_layer(
    path: str | Path,
    *,
    name: str | None = None,
    window_size: int,
    sparse_indices: Sequence[int] | None = None,
    band: int = 1,
    masked: bool = True,
) -> SpatialLayer:
    """Read a raster band as a ``SpatialLayer``."""

    grid = read_raster(path, band=band, masked=masked)
    layer_name = name if name is not None else Path(path).stem
    return SpatialLayer(
        name=layer_name,
        values=grid.values,
        window_size=window_size,
        transform=grid.transform,
        sparse_indices=sparse_indices,
    )


def write_raster(
    path: str | Path,
    values: np.ndarray,
    reference: RasterGrid,
    *,
    nodata: float | int | None = None,
    dtype: str | np.dtype | None = None,
    driver: str = "GTiff",
    overwrite_profile: dict[str, object] | None = None,
) -> None:
    """Write a 2D grid or band-last 3D grid using reference raster metadata."""

    rasterio = _import_rasterio()
    array = np.asarray(values)
    if array.ndim not in {2, 3}:
        raise ValueError("values must be a 2D grid or 3D band-last grid")
    if array.shape[:2] != reference.shape:
        raise ValueError("values shape must match reference grid shape")

    bands = 1 if array.ndim == 2 else array.shape[2]
    output_dtype = np.dtype(dtype) if dtype is not None else array.dtype
    profile = {
        "driver": driver,
        "height": reference.shape[0],
        "width": reference.shape[1],
        "count": bands,
        "dtype": output_dtype.name,
        "transform": _to_affine(reference.transform),
        "crs": reference.crs,
        "nodata": reference.nodata if nodata is None else nodata,
    }
    if overwrite_profile is not None:
        profile.update(overwrite_profile)

    with rasterio.open(path, "w", **profile) as dataset:
        if array.ndim == 2:
            dataset.write(array.astype(output_dtype, copy=False), 1)
        else:
            dataset.write(np.moveaxis(array, 2, 0).astype(output_dtype, copy=False))


def _to_affine(transform: GridTransform):
    from affine import Affine

    return Affine(transform.x_size, 0, transform.left, 0, -transform.y_size, transform.top)


def _import_rasterio():
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for raster I/O; install spatialized[raster]"
        ) from exc
    return rasterio

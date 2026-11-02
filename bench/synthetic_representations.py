"""Frozen image representations for harmonized synthetic radiomics tasks.

This module is generator-side infrastructure only.  It never imports or runs a
radiomics adapter.  Array axes follow the source NIfTI ``x, y, z`` order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RepresentationError(ValueError):
    """Raised when a representation cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class FBNRepresentation:
    """One mask-specific fixed-bin-number representation and its provenance."""

    array: np.ndarray
    configured_levels: int
    occupied_levels: int
    roi_min: float
    roi_max: float
    roi_voxels: int
    background_value: int = 0


@dataclass(frozen=True)
class FBSRepresentation:
    """Mask-specific one-based fixed-bin-size representation."""

    array: np.ndarray
    bin_width: float
    anchor: float
    configured_levels: int
    occupied_levels: int
    roi_min: float
    roi_max: float
    roi_voxels: int
    background_value: int = 0


def compile_mask_specific_fbn(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    levels: int = 32,
) -> FBNRepresentation:
    """Compile the IBSI fixed-bin-number equation once for an exact ROI.

    The maximum intensity is assigned explicitly to ``levels``.  Zero is
    reserved outside the mask and never represents an occupied grey level.
    Constant ROIs fail closed because FBN contrast is undefined there.
    """

    values = np.asarray(image)
    mask_values = np.asarray(mask)
    if values.shape != mask_values.shape:
        raise RepresentationError("image and mask shapes differ")
    if values.ndim != 3:
        raise RepresentationError("harmonized representations require a 3D image")
    if not np.all(np.isfinite(mask_values)) or not np.all(
        (mask_values == 0) | (mask_values == 1)
    ):
        raise RepresentationError("mask must be finite and canonical binary {0, 1}")
    roi = mask_values == 1
    if not isinstance(levels, (int, np.integer)) or int(levels) < 2:
        raise RepresentationError("levels must be an integer of at least two")
    roi_values = np.asarray(values[roi], dtype=np.float64)
    if roi_values.size == 0:
        raise RepresentationError("cannot discretize an empty ROI")
    if not np.all(np.isfinite(roi_values)):
        raise RepresentationError("ROI contains a non-finite intensity")
    minimum = float(np.min(roi_values))
    maximum = float(np.max(roi_values))
    if maximum <= minimum:
        raise RepresentationError("FBN is undefined for a constant ROI")

    n_levels = int(levels)
    bins = np.floor(n_levels * (roi_values - minimum) / (maximum - minimum)).astype(
        np.int64
    )
    bins = np.clip(bins + 1, 1, n_levels)
    bins[roi_values == maximum] = n_levels
    dtype = np.uint8 if n_levels <= np.iinfo(np.uint8).max else np.uint16
    discrete = np.zeros(values.shape, dtype=dtype)
    discrete[roi] = bins.astype(dtype, copy=False)
    observed = np.unique(discrete[roi])
    if observed[0] < 1 or observed[-1] > n_levels:
        raise RepresentationError("compiled FBN values violate the one-based grid")
    return FBNRepresentation(
        array=discrete,
        configured_levels=n_levels,
        occupied_levels=int(observed.size),
        roi_min=minimum,
        roi_max=maximum,
        roi_voxels=int(roi_values.size),
    )


def texture_workload(discrete: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Measure the texture work that one discrete ROI image implies.

    Zones are 26-connected regions of one grey level, as in the IBSI GLSZM.
    Boundary voxels are ROI voxels with at least one 6-neighbour outside it.
    """

    from scipy import ndimage

    roi = np.asarray(mask) > 0
    box = ndimage.find_objects(roi.astype(np.uint8))[0]
    roi = np.pad(roi[box], 1)
    values = np.pad(np.asarray(discrete)[box], 1)
    structure = np.ones((3, 3, 3), dtype=bool)
    zones = sum(
        int(ndimage.label((values == level) & roi, structure=structure)[1])
        for level in np.unique(values[roi])
    )
    interior = ndimage.binary_erosion(
        roi, structure=ndimage.generate_binary_structure(3, 1)
    )
    roi_voxels = int(roi.sum())
    return {
        "zone_count_26": zones,
        "mean_voxels_per_zone": roi_voxels / zones,
        "boundary_voxels_6": roi_voxels - int(interior.sum()),
    }


def compile_mask_specific_fbs(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    bin_width: float = 1.0,
    anchor: float | None = None,
) -> FBSRepresentation:
    """Compile a one-based IBSI fixed-bin-size grid outside every adapter.

    ``anchor`` defaults to the exact ROI minimum.  The stored values are bin
    indices, not calibrated intensities; provenance retains the affine mapping
    ``intensity = anchor + (level - 1) * bin_width``.  Zero remains reserved
    outside the ROI.
    """

    values = np.asarray(image)
    mask_values = np.asarray(mask)
    if values.shape != mask_values.shape:
        raise RepresentationError("image and mask shapes differ")
    if values.ndim != 3:
        raise RepresentationError("harmonized representations require a 3D image")
    if not np.all(np.isfinite(mask_values)) or not np.all(
        (mask_values == 0) | (mask_values == 1)
    ):
        raise RepresentationError("mask must be finite and canonical binary {0, 1}")
    roi = mask_values == 1
    width = float(bin_width)
    if not np.isfinite(width) or width <= 0:
        raise RepresentationError("bin_width must be finite and positive")
    roi_values = np.asarray(values[roi], dtype=np.float64)
    if roi_values.size == 0:
        raise RepresentationError("cannot discretize an empty ROI")
    if not np.all(np.isfinite(roi_values)):
        raise RepresentationError("ROI contains a non-finite intensity")
    minimum = float(np.min(roi_values))
    maximum = float(np.max(roi_values))
    lower = minimum if anchor is None else float(anchor)
    if not np.isfinite(lower) or lower > minimum:
        raise RepresentationError(
            "FBS anchor must be finite and no greater than ROI minimum"
        )
    bins = np.floor((roi_values - lower) / width).astype(np.int64) + 1
    if np.any(bins < 1):
        raise RepresentationError("compiled FBS grid produced a non-positive ROI level")
    configured = int(np.max(bins))
    if configured <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif configured <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    elif configured <= np.iinfo(np.uint32).max:
        dtype = np.uint32
    else:
        raise RepresentationError("compiled FBS grid exceeds uint32 level capacity")
    discrete = np.zeros(values.shape, dtype=dtype)
    discrete[roi] = bins.astype(dtype, copy=False)
    observed = np.unique(discrete[roi])
    return FBSRepresentation(
        array=discrete,
        bin_width=width,
        anchor=lower,
        configured_levels=configured,
        occupied_levels=int(observed.size),
        roi_min=minimum,
        roi_max=maximum,
        roi_voxels=int(roi_values.size),
    )


__all__ = [
    "FBNRepresentation",
    "FBSRepresentation",
    "RepresentationError",
    "compile_mask_specific_fbn",
    "compile_mask_specific_fbs",
]

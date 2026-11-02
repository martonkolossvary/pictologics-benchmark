"""Shared command-line and result protocol for isolated adapter processes."""

from __future__ import annotations

import argparse
import math
from importlib import metadata
from typing import Any, Iterable

from bench.adapters.base import write_event
from bench.adapters.registry import get_adapter, validate_families
from bench.ibsi_families import FAMILY_ORDER


ADAPTER_PROTOCOL_VERSION = 1
TIMING_CONTRACT_VERSION = 1
REQUIRED_AGGREGATION = "3d_merge"

DIRECTIONAL_TEXTURE_FAMILIES = frozenset({"glcm", "glrlm"})
NONDIRECTIONAL_TEXTURE_FAMILIES = frozenset({"glszm", "gldzm", "ngtdm", "ngldm"})
AGGREGATION_CHOICES = (
    "native",
    "2d_average",
    "2d_slice_merge",
    "2.5d_direction_merge",
    "2.5d_merge",
    "3d_merge",
    "3d_average",
)
_NATIVE_AGGREGATION = {
    "pictologics": "3d_merge",
    "pyradiomics": "3d_average",
    "mirp": "3d_merge",
    "medimage": "3d_merge",
    "zrad": "3d_merge",
}
_SUPPORTED_DIRECTIONAL_AGGREGATIONS = {
    "pictologics": frozenset({"3d_merge"}),
    # PyRadiomics defaults to per-direction feature averaging, but its
    # documented ``weightingNorm='no_weighting'`` mode assigns unit weights and
    # sums the direction matrices before feature calculation. That is the
    # unweighted IBSI 3D-merged operation required by this project.
    "pyradiomics": frozenset({"3d_average", "3d_merge"}),
    "mirp": frozenset(AGGREGATION_CHOICES[1:]),
    "medimage": frozenset({"3d_merge"}),
    "zrad": frozenset(AGGREGATION_CHOICES[1:]),
}


def aggregation_dimension(aggregation: str) -> str:
    token = str(aggregation).strip().lower()
    if token.startswith("2.5d_"):
        return "2.5d"
    if token.startswith("2d_"):
        return "2d"
    if token.startswith("3d_"):
        return "3d"
    raise ValueError(f"Aggregation has no dimensional profile: {aggregation!r}")


# Radius of the IBSI 1 cm^3 local-intensity sphere, in millimetres.
LOCAL_INTENSITY_RADIUS_MM = 10.0 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)


def roi_crop(
    mask: Any,
    spacing_mm: Iterable[float],
    families: Iterable[str],
) -> tuple[tuple[slice, ...], dict[str, Any]]:
    """Return the shared pre-timer ROI bounding-box crop and its audit record.

    One background voxel is kept on every side.  Local intensity keeps the
    whole sphere radius because its neighbourhood means include voxels outside
    the ROI.  ``spacing_mm`` must follow the array axis order of ``mask``.
    """

    import numpy as np

    array = np.asarray(mask)
    nonzero = np.nonzero(array)
    if nonzero[0].size == 0:
        raise ValueError("Cannot crop an empty ROI mask")
    margin_mm = LOCAL_INTENSITY_RADIUS_MM if "local_intensity" in set(families) else 0.0
    padding = [max(1, math.ceil(margin_mm / float(value))) for value in spacing_mm]
    slices = tuple(
        slice(max(0, int(indices.min()) - pad), min(size, int(indices.max()) + 1 + pad))
        for indices, pad, size in zip(nonzero, padding, array.shape)
    )
    record = {
        "rule": "roi_bounding_box_before_timer",
        "padding_voxels": padding,
        "input_shape": list(array.shape),
        "cropped_shape": [item.stop - item.start for item in slices],
    }
    return slices, record


def parse_csv(value: str | None, *, lowercase: bool = True) -> list[str]:
    if not value:
        return []
    tokens = [part.strip() for part in value.split(",") if part.strip()]
    if lowercase:
        tokens = [part.lower() for part in tokens]
    return list(dict.fromkeys(tokens))


def _sha256_argument(value: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise argparse.ArgumentTypeError(
            "expected a 64-character hexadecimal SHA-256 digest"
        )
    return digest


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument(
        "--image-sha256",
        type=_sha256_argument,
        default=None,
        help="Optional controller-supplied SHA-256 binding for the image input",
    )
    parser.add_argument(
        "--mask-sha256",
        type=_sha256_argument,
        default=None,
        help="Optional controller-supplied SHA-256 binding for the mask input",
    )
    parser.add_argument(
        "--source-image-sha256",
        type=_sha256_argument,
        default=None,
        help="SHA-256 of the original image from which this representation derives",
    )
    parser.add_argument(
        "--input-contract",
        default="adapter_internal_preprocessing",
        help="Controller-frozen representation-routing contract",
    )
    parser.add_argument(
        "--input-representation-id",
        default="single_image_internal_preprocessing",
    )
    parser.add_argument(
        "--representation-derivation-sha256",
        type=_sha256_argument,
        default=None,
    )
    parser.add_argument("--configured-levels", type=int, default=None)
    parser.add_argument("--occupied-levels", type=int, default=None)
    parser.add_argument(
        "--modality", default=None, help="Optional dataset modality metadata"
    )
    parser.add_argument(
        "--discretization",
        choices=["fbn", "fbs", "identity", "raw"],
        default="fbn",
        help=(
            "Grey-level preprocessing. 'raw' is restricted to families such as "
            "first-order intensity that do not require discretisation."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATION_CHOICES,
        default=REQUIRED_AGGREGATION,
        help=(
            "Directional texture aggregation; project runs require 3d_merge, "
            "and unsupported upstream methods must not be relabelled"
        ),
    )
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--bin-width", type=float, default=32.0)
    parser.add_argument(
        "--intensity-min",
        type=float,
        default=None,
        help="Optional inclusive lower resegmentation/discretisation bound",
    )
    parser.add_argument(
        "--intensity-max",
        type=float,
        default=None,
        help="Optional inclusive upper resegmentation/discretisation bound",
    )
    parser.add_argument("--include-values", action="store_true")
    parser.add_argument(
        "--include-ibsi-codes", default=None, help="Comma-separated IBSI feature codes"
    )
    parser.add_argument(
        "--families",
        default=None,
        help="Comma-separated canonical IBSI families (default: all families)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of measured calls; one additional warmup is unreported unless disabled",
    )
    parser.add_argument("--timed", action="store_true")
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable the unreported warmup calculation for every requested timing",
    )


def parse_intensity_range(
    args: argparse.Namespace,
) -> tuple[float, float] | None:
    """Return a validated common resegmentation range from adapter arguments."""

    lower = getattr(args, "intensity_min", None)
    upper = getattr(args, "intensity_max", None)
    if (lower is None) != (upper is None):
        raise ValueError("Provide both --intensity-min and --intensity-max, or neither")
    if lower is None:
        return None

    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Intensity range bounds must both be finite")
    if lower >= upper:
        raise ValueError("Intensity range requires --intensity-min < --intensity-max")
    return lower, upper


def resolve_aggregation(
    adapter: str,
    requested: str,
    families: Iterable[str],
) -> str:
    """Resolve and validate directional aggregation without relabelling native output."""

    token = str(requested or "native").strip().lower()
    if token not in AGGREGATION_CHOICES:
        raise ValueError(f"Unknown aggregation: {requested!r}")
    try:
        effective = _NATIVE_AGGREGATION[adapter] if token == "native" else token
        supported = _SUPPORTED_DIRECTIONAL_AGGREGATIONS[adapter]
    except KeyError as exc:
        raise ValueError(
            f"Aggregation capability is not declared for adapter {adapter!r}"
        ) from exc
    family_set = set(families)
    if (
        DIRECTIONAL_TEXTURE_FAMILIES.intersection(family_set)
        and effective not in supported
    ):
        raise ValueError(
            f"{adapter} cannot calculate directional textures using {effective}; "
            f"supported: {', '.join(sorted(supported))}"
        )
    if NONDIRECTIONAL_TEXTURE_FAMILIES.intersection(family_set):
        supported_dimensions = {aggregation_dimension(value) for value in supported}
        if aggregation_dimension(effective) not in supported_dimensions:
            raise ValueError(
                f"{adapter} cannot calculate non-directional textures in "
                f"{aggregation_dimension(effective)}; supported dimensions: "
                f"{', '.join(sorted(supported_dimensions))}"
            )
    return effective


def supports_aggregation(
    adapter: str,
    requested: str,
    families: Iterable[str],
) -> bool:
    """Return whether the adapter path implements the requested exact profile."""

    try:
        resolve_aggregation(adapter, requested, families)
    except ValueError:
        return False
    return True


def requested_families(
    adapter: str, args: argparse.Namespace
) -> tuple[list[str], list[str]]:
    # All built-in adapters call this helper. Validate the common range here so
    # even an adapter that does not consume it cannot silently accept an invalid
    # half-range or non-finite value.
    parse_intensity_range(args)
    bins = getattr(args, "bins", None)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("--bins must be a positive integer")
    bin_width = float(getattr(args, "bin_width", float("nan")))
    if not math.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("--bin-width must be finite and positive")
    requested = parse_csv(getattr(args, "families", None)) or list(FAMILY_ORDER)
    return validate_families(adapter, requested)


def package_version(adapter: str) -> str:
    distribution = get_adapter(adapter).distribution
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def run_adapter_timing(
    compute_fn,
    *,
    iterations: int,
    no_warmup: bool,
    prepare_fn=None,
    finalize_fn=None,
):
    """Apply the calculation-only preparation/warmup policy."""

    from bench.adapters.base import run_timed_computation

    optional = {"event_fn": write_event}
    if prepare_fn is not None:
        optional["prepare_fn"] = prepare_fn
    if finalize_fn is not None:
        optional["finalize_fn"] = finalize_fn
    return run_timed_computation(
        compute_fn,
        iterations=iterations,
        no_warmup=no_warmup,
        **optional,
    )


def timing_contract_metadata() -> dict[str, Any]:
    """Return the stable cross-adapter timing-scope declaration."""

    return {
        "version": TIMING_CONTRACT_VERSION,
        "scope": "prepared_family_inputs_to_radiomic_calculations",
        "includes_required_preprocessing": False,
        "excludes_file_io": True,
        "excludes_mask_preparation": True,
        # Four adapters extract ROI voxels inside their own feature code, so
        # every adapter does this step inside the timer.
        "includes_roi_voxel_extraction": True,
        "excludes_resegmentation": True,
        "excludes_discretization": True,
        "excludes_result_serialization": True,
        "includes_matrix_mesh_neighborhood_construction": True,
        "iterations_meaning": "measured_calls_excluding_optional_warmup",
        "comparison_unit": "adapter_case_family_process_repeat",
        "normalization": "none; feature coverage and calculated-output counts reported separately",
    }


def make_payload(
    *,
    adapter: str,
    feature_names: Iterable[str],
    values: dict[str, float] | None = None,
    timing: dict[str, Any] | None = None,
    requested: Iterable[str] = (),
    unsupported: Iterable[str] = (),
    metadata_payload: dict[str, Any] | None = None,
    image_sha256: str | None = None,
    source_image_sha256: str | None = None,
    mask_sha256: str | None = None,
    modality: str | None = None,
    input_contract: str = "adapter_internal_preprocessing",
    input_representation_id: str = "single_image_internal_preprocessing",
    representation_derivation_sha256: str | None = None,
    configured_levels: int | None = None,
    occupied_levels: int | None = None,
) -> dict[str, Any]:
    capabilities = get_adapter(adapter)
    payload: dict[str, Any] = {
        "schema_version": ADAPTER_PROTOCOL_VERSION,
        "adapter": adapter,
        "software": {
            "distribution": capabilities.distribution,
            "version": package_version(adapter),
        },
        "selection": {
            "requested_families": list(requested),
            "unsupported_families": list(unsupported),
            "mode": capabilities.selection_mode,
        },
        "features": {"all": list(feature_names)},
    }
    if values is not None:
        payload["values"] = {"all": values}
    if timing is not None:
        payload["timing"] = timing
    payload_metadata = dict(metadata_payload or {})
    payload_metadata["input"] = {
        "image_sha256": image_sha256,
        "source_image_sha256": source_image_sha256 or image_sha256,
        "mask_sha256": mask_sha256,
        "modality": modality,
        "input_contract": input_contract,
        "representation_id": input_representation_id,
        "representation_derivation_sha256": representation_derivation_sha256,
        "configured_levels": configured_levels,
        "occupied_levels": occupied_levels,
    }
    if timing is not None:
        payload_metadata["timing_contract"] = timing_contract_metadata()
    payload["metadata"] = payload_metadata
    return payload

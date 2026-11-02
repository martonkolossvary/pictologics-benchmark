from __future__ import annotations

import os
import time
from numbers import Real
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bench.adapters.base import write_event, write_json
from bench.ibsi_families import MORPHOLOGY_SHAPE_REQUESTS, MORPHOLOGY_SPATIAL_REQUESTS
from bench.adapters.protocol import (
    add_common_arguments,
    make_payload,
    parse_csv,
    parse_intensity_range,
    requested_families,
    resolve_aggregation,
    roi_crop,
    run_adapter_timing,
)


def _needs_discretisation(families: List[str]) -> bool:
    tokens = set(families)
    return any(
        fam in tokens
        for fam in (
            "histogram",
            "ivh",
            "glcm",
            "glrlm",
            "glszm",
            "gldzm",
            "ngtdm",
            "ngldm",
        )
    )


def _to_mask_image(image, mask):
    from pictologics.loader import Image

    mask_arr = (mask.array > 0).astype(np.uint8)
    return Image(
        array=mask_arr,
        spacing=mask.spacing,
        origin=mask.origin,
        direction=mask.direction,
        modality=mask.modality,
    )


def _crop_image(image, slices):
    """Return a contiguous ROI crop whose origin keeps its world position."""
    from pictologics.loader import Image

    start_mm = np.array([item.start for item in slices], dtype=float) * np.asarray(
        image.spacing, dtype=float
    )
    direction = (
        np.eye(3)
        if image.direction is None
        else np.asarray(image.direction, dtype=float).reshape(3, 3)
    )
    return Image(
        array=np.ascontiguousarray(image.array[slices]),
        spacing=image.spacing,
        origin=tuple(np.asarray(image.origin, dtype=float) + direction @ start_mm),
        direction=image.direction,
        modality=image.modality,
    )


def _identity_discrete_image(image, intensity_mask):
    """Return an unchanged positive-integer grey-level image for IBSI phantoms."""
    from pictologics.loader import Image

    roi = np.asarray(image.array)[np.asarray(intensity_mask.array) > 0]
    if roi.size == 0:
        raise ValueError("Identity discretization requires a non-empty ROI")
    if (
        not np.isfinite(roi).all()
        or np.any(roi < 1.0)
        or not np.allclose(roi, np.rint(roi))
    ):
        raise ValueError(
            "Identity discretization is valid only for finite positive-integer ROI grey levels"
        )
    discrete = np.zeros(np.asarray(image.array).shape, dtype=np.int32)
    roi_mask = np.asarray(intensity_mask.array) > 0
    discrete[roi_mask] = np.rint(np.asarray(image.array)[roi_mask]).astype(np.int32)
    return Image(
        array=discrete,
        spacing=image.spacing,
        origin=image.origin,
        direction=image.direction,
        modality=image.modality,
    )


def _pictologics_feature_modules():
    """Import the package feature APIs before any measured iteration."""

    from pictologics import features
    from pictologics import preprocessing

    return preprocessing, features


def _clear_feature_caches() -> list[str]:
    """Keep every call cold: clear each cached table of the feature modules.

    The tables (sphere offsets, texture directions, zone offsets) are then
    built again inside each timed call, as in a first call.  Z-Rad gets the
    same rule for its local-intensity cache.  Returns the cleared names.
    """

    from pictologics.features import intensity, morphology, texture

    cleared = []
    for module in (intensity, morphology, texture):
        for name, value in vars(module).items():
            if callable(getattr(value, "cache_clear", None)):
                value.cache_clear()
                cleared.append(f"{module.__name__}.{name}")
    return cleared


def _finite_feature_mapping(values: Dict[object, object]) -> Dict[str, float]:
    """Require one finite scalar for every calculated Pictologics output."""

    normalized: Dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name).strip()
        if not name or name in normalized:
            raise RuntimeError(
                f"Pictologics returned an empty or duplicate feature name: {name!r}"
            )
        if isinstance(raw_value, (bool, np.bool_)):
            raise RuntimeError(f"Pictologics feature {name} is not a numeric scalar")
        try:
            array = np.asarray(raw_value).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Pictologics feature {name} is not a numeric scalar"
            ) from exc
        if array.size != 1:
            raise RuntimeError(f"Pictologics feature {name} is not one finite scalar")
        item = array[0]
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise RuntimeError(f"Pictologics feature {name} is not a numeric scalar")
        scalar = float(item)
        if not np.isfinite(scalar):
            raise RuntimeError(f"Pictologics feature {name} is not one finite scalar")
        normalized[name] = scalar
    return normalized


def _merge_feature_mapping(
    target: Dict[str, float],
    values: Dict[object, object],
) -> None:
    """Merge one native family without silently overwriting another."""

    normalized = _finite_feature_mapping(values)
    duplicated = sorted(set(target).intersection(normalized))
    if duplicated:
        raise RuntimeError(
            "Pictologics returned duplicate feature names across families: "
            + ", ".join(duplicated[:5])
        )
    target.update(normalized)


def _compute_features(
    *,
    image,
    mask,
    families: List[str],
    discretization: str,
    bins: int,
    bin_width: float,
    intensity_range: Optional[Tuple[float, float]],
    feature_modules=None,
) -> Dict[str, float]:
    preprocessing, feature_api = (
        feature_modules
        if feature_modules is not None
        else _pictologics_feature_modules()
    )
    apply_mask = preprocessing.apply_mask
    discretise_image = preprocessing.discretise_image
    resegment_mask = preprocessing.resegment_mask
    calculate_all_texture_matrices = feature_api.calculate_all_texture_matrices
    calculate_glcm_features = feature_api.calculate_glcm_features
    calculate_gldzm_features = feature_api.calculate_gldzm_features
    calculate_glrlm_features = feature_api.calculate_glrlm_features
    calculate_glszm_features = feature_api.calculate_glszm_features
    calculate_intensity_features = feature_api.calculate_intensity_features
    calculate_intensity_histogram_features = (
        feature_api.calculate_intensity_histogram_features
    )
    calculate_ivh_features = feature_api.calculate_ivh_features
    calculate_local_intensity_features = feature_api.calculate_local_intensity_features
    calculate_morphology_features = feature_api.calculate_morphology_features
    calculate_ngldm_features = feature_api.calculate_ngldm_features
    calculate_ngtdm_features = feature_api.calculate_ngtdm_features
    calculate_spatial_intensity_features = (
        feature_api.calculate_spatial_intensity_features
    )

    features: Dict[str, float] = {}

    def add_features(values: Dict[object, object]) -> None:
        _merge_feature_mapping(features, values)

    if discretization == "raw" and _needs_discretisation(families):
        raise ValueError(
            "Raw extraction is valid only for non-discretized feature families"
        )
    intensity_mask = mask
    if intensity_range is not None:
        intensity_mask = resegment_mask(
            image,
            mask,
            range_min=float(intensity_range[0]),
            range_max=float(intensity_range[1]),
        )

    if "intensity" in families:
        roi_values = apply_mask(image, intensity_mask)
        add_features(calculate_intensity_features(roi_values))

    if MORPHOLOGY_SHAPE_REQUESTS.intersection(families):
        add_features(
            calculate_morphology_features(
                mask,
                image=image,
                intensity_mask=intensity_mask,
            )
        )
    if MORPHOLOGY_SPATIAL_REQUESTS.intersection(families):
        # Moran's I and Geary's C are IBSI morphology features.  Benchmark
        # timing measures them as the separate morphology_spatial workload.
        add_features(calculate_spatial_intensity_features(image, intensity_mask))

    if "local_intensity" in families:
        add_features(calculate_local_intensity_features(image, intensity_mask))

    disc_image = None
    disc_values = None
    n_bins = None

    if _needs_discretisation(families):
        method = {"fbn": "FBN", "fbs": "FBS", "identity": "IDENTITY"}[discretization]
        if method == "FBS" and intensity_range is None:
            raise ValueError(
                "Pictologics FBS extraction requires an explicit intensity range "
                "for the IBSI lower-bin anchor"
            )
        if method == "IDENTITY":
            disc_image = _identity_discrete_image(image, intensity_mask)
        elif method == "FBN":
            # FBN uses the observed min/max of the resegmented intensity ROI.
            disc_image = discretise_image(
                image,
                method=method,
                roi_mask=intensity_mask,
                n_bins=bins,
            )
        else:
            assert intensity_range is not None
            disc_image = discretise_image(
                image,
                method=method,
                roi_mask=intensity_mask,
                bin_width=bin_width,
                min_val=float(intensity_range[0]),
                max_val=float(intensity_range[1]),
            )

        disc_values = apply_mask(disc_image, intensity_mask)
        if disc_values.size:
            # The histogram API's ``n_bins`` is the full discretisation grid,
            # not merely the largest occupied bin.  FBN therefore retains the
            # requested N_g even for a flat/sparse ROI; FBS and identity have
            # no separately configured grid size and use the largest observed
            # one-based grey level.
            n_bins = int(bins) if method == "FBN" else int(np.max(disc_values))

    if "histogram" in families and disc_values is not None:
        add_features(
            calculate_intensity_histogram_features(
                disc_values,
                n_bins=n_bins,
            )
        )

    if "ivh" in families and disc_values is not None:
        if discretization == "fbs":
            if intensity_range is None:  # guarded above; keeps the contract local
                raise ValueError(
                    "Pictologics FBS IVH requires an explicit intensity range"
                )
            lower, upper = intensity_range
            add_features(
                calculate_ivh_features(
                    disc_values,
                    bin_width=float(bin_width),
                    min_val=float(lower),
                    max_val=float(upper),
                    target_range_min=float(lower),
                    target_range_max=float(upper),
                )
            )
        else:
            # Preserve the package's FBN IVH convention: its intensity axis is
            # the one-based discrete grey-level index with unit steps.
            add_features(calculate_ivh_features(disc_values, bin_width=1.0))

    texture_families = {
        "glcm",
        "glrlm",
        "glszm",
        "gldzm",
        "ngtdm",
        "ngldm",
    }
    if texture_families.intersection(families) and disc_image is not None and n_bins:
        mask_arr = intensity_mask.array
        morphology_mask_arr = mask.array
        matrices = calculate_all_texture_matrices(
            disc_image.array,
            mask_arr,
            n_bins,
            distance_mask=morphology_mask_arr,
            calc_glcm="glcm" in families,
            calc_glrlm="glrlm" in families,
            calc_glszm="glszm" in families,
            calc_gldzm="gldzm" in families,
            calc_ngtdm="ngtdm" in families,
            calc_ngldm="ngldm" in families,
        )
        if "glcm" in families:
            add_features(
                calculate_glcm_features(
                    disc_image.array, mask_arr, n_bins, glcm_matrix=matrices["glcm"]
                )
            )
        if "glrlm" in families:
            add_features(
                calculate_glrlm_features(
                    disc_image.array, mask_arr, n_bins, glrlm_matrix=matrices["glrlm"]
                )
            )
        if "glszm" in families:
            add_features(
                calculate_glszm_features(
                    disc_image.array, mask_arr, n_bins, glszm_matrix=matrices["glszm"]
                )
            )
        if "gldzm" in families:
            add_features(
                calculate_gldzm_features(
                    disc_image.array,
                    mask_arr,
                    n_bins,
                    gldzm_matrix=matrices["gldzm"],
                    distance_mask=morphology_mask_arr,
                )
            )
        if "ngtdm" in families:
            add_features(
                calculate_ngtdm_features(
                    disc_image.array,
                    mask_arr,
                    n_bins,
                    ngtdm_matrices=(matrices["ngtdm_s"], matrices["ngtdm_n"]),
                )
            )
        if "ngldm" in families:
            add_features(
                calculate_ngldm_features(
                    disc_image.array,
                    mask_arr,
                    n_bins,
                    ngldm_matrix=matrices["ngldm"],
                )
            )

    if families and not features:
        raise RuntimeError(
            "Pictologics returned zero features for requested families: "
            + ", ".join(families)
        )
    return features


def _prepare_calculation_only(
    *,
    image,
    mask,
    families: List[str],
    discretization: str,
    bins: int,
    bin_width: float,
    intensity_range: Optional[Tuple[float, float]],
    feature_modules=None,
):
    """Prepare immutable family inputs and return the calculation callable.

    Mask resegmentation and discrete-image validation or construction happen
    here.  The returned callable extracts the ROI voxel values and then runs the
    mesh, neighbourhood/matrix, and feature-formula calculations.  The other
    adapters extract ROI voxels inside their own feature code, so this step is
    timed for every adapter.
    """

    preprocessing, feature_api = (
        feature_modules
        if feature_modules is not None
        else _pictologics_feature_modules()
    )
    if discretization == "raw" and _needs_discretisation(families):
        raise ValueError(
            "Raw extraction is valid only for non-discretized feature families"
        )
    intensity_mask = mask
    if intensity_range is not None:
        intensity_mask = preprocessing.resegment_mask(
            image,
            mask,
            range_min=float(intensity_range[0]),
            range_max=float(intensity_range[1]),
        )
    disc_image = None
    disc_values = None
    n_bins = None
    if _needs_discretisation(families):
        method = {"fbn": "FBN", "fbs": "FBS", "identity": "IDENTITY"}[discretization]
        if method == "FBS" and intensity_range is None:
            raise ValueError(
                "Pictologics FBS extraction requires an explicit intensity range "
                "for the IBSI lower-bin anchor"
            )
        if method == "IDENTITY":
            disc_image = _identity_discrete_image(image, intensity_mask)
        elif method == "FBN":
            disc_image = preprocessing.discretise_image(
                image,
                method=method,
                roi_mask=intensity_mask,
                n_bins=bins,
            )
        else:
            assert intensity_range is not None
            disc_image = preprocessing.discretise_image(
                image,
                method=method,
                roi_mask=intensity_mask,
                bin_width=bin_width,
                min_val=float(intensity_range[0]),
                max_val=float(intensity_range[1]),
            )
        disc_values = preprocessing.apply_mask(disc_image, intensity_mask)
        if disc_values.size:
            n_bins = int(bins) if method == "FBN" else int(np.max(disc_values))

    mask_arr = np.asarray(intensity_mask.array)
    morphology_mask_arr = np.asarray(mask.array)
    disc_array = np.asarray(disc_image.array) if disc_image is not None else None

    def calculate() -> List[Mapping[object, object]]:
        calculated: List[Mapping[object, object]] = []

        def add(values: Dict[object, object]) -> None:
            calculated.append(values)

        if "intensity" in families:
            roi_values = preprocessing.apply_mask(image, intensity_mask)
            add(feature_api.calculate_intensity_features(roi_values))
        if MORPHOLOGY_SHAPE_REQUESTS.intersection(families):
            add(
                feature_api.calculate_morphology_features(
                    mask,
                    image=image,
                    intensity_mask=intensity_mask,
                )
            )
        if MORPHOLOGY_SPATIAL_REQUESTS.intersection(families):
            add(feature_api.calculate_spatial_intensity_features(image, intensity_mask))
        if "local_intensity" in families:
            add(feature_api.calculate_local_intensity_features(image, intensity_mask))
        roi_disc_values = (
            preprocessing.apply_mask(disc_image, intensity_mask)
            if disc_image is not None and {"histogram", "ivh"}.intersection(families)
            else None
        )
        if "histogram" in families and roi_disc_values is not None:
            add(
                feature_api.calculate_intensity_histogram_features(
                    roi_disc_values,
                    n_bins=n_bins,
                )
            )
        if "ivh" in families and roi_disc_values is not None:
            if discretization == "fbs":
                assert intensity_range is not None
                lower, upper = intensity_range
                add(
                    feature_api.calculate_ivh_features(
                        roi_disc_values,
                        bin_width=float(bin_width),
                        min_val=float(lower),
                        max_val=float(upper),
                        target_range_min=float(lower),
                        target_range_max=float(upper),
                    )
                )
            else:
                add(feature_api.calculate_ivh_features(roi_disc_values, bin_width=1.0))

        texture_families = {
            "glcm",
            "glrlm",
            "glszm",
            "gldzm",
            "ngtdm",
            "ngldm",
        }
        if (
            texture_families.intersection(families)
            and disc_array is not None
            and n_bins
        ):
            matrices = feature_api.calculate_all_texture_matrices(
                disc_array,
                mask_arr,
                n_bins,
                distance_mask=morphology_mask_arr,
                calc_glcm="glcm" in families,
                calc_glrlm="glrlm" in families,
                calc_glszm="glszm" in families,
                calc_gldzm="gldzm" in families,
                calc_ngtdm="ngtdm" in families,
                calc_ngldm="ngldm" in families,
            )
            if "glcm" in families:
                add(
                    feature_api.calculate_glcm_features(
                        disc_array, mask_arr, n_bins, glcm_matrix=matrices["glcm"]
                    )
                )
            if "glrlm" in families:
                add(
                    feature_api.calculate_glrlm_features(
                        disc_array,
                        mask_arr,
                        n_bins,
                        glrlm_matrix=matrices["glrlm"],
                    )
                )
            if "glszm" in families:
                add(
                    feature_api.calculate_glszm_features(
                        disc_array,
                        mask_arr,
                        n_bins,
                        glszm_matrix=matrices["glszm"],
                    )
                )
            if "gldzm" in families:
                add(
                    feature_api.calculate_gldzm_features(
                        disc_array,
                        mask_arr,
                        n_bins,
                        gldzm_matrix=matrices["gldzm"],
                        distance_mask=morphology_mask_arr,
                    )
                )
            if "ngtdm" in families:
                add(
                    feature_api.calculate_ngtdm_features(
                        disc_array,
                        mask_arr,
                        n_bins,
                        ngtdm_matrices=(matrices["ngtdm_s"], matrices["ngtdm_n"]),
                    )
                )
            if "ngldm" in families:
                add(
                    feature_api.calculate_ngldm_features(
                        disc_array,
                        mask_arr,
                        n_bins,
                        ngldm_matrix=matrices["ngldm"],
                    )
                )
        if families and not calculated:
            raise RuntimeError(
                "Pictologics returned zero features for requested families: "
                + ", ".join(families)
            )
        return calculated

    def finalize(
        calculated: Sequence[Mapping[object, object]], _state=None
    ) -> Dict[str, float]:
        features: Dict[str, float] = {}
        for values in calculated:
            _merge_feature_mapping(features, values)
        return features

    return calculate, finalize


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="pictologics-adapter")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    intensity_range = parse_intensity_range(args)

    # Package import must never inherit an ambient request to warm JIT kernels.
    # The reviewed policy schedules its one explicit warmup below, outside the
    # measured region, or omits it entirely under --no-warmup.
    os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "1"

    from pictologics import get_num_threads
    from pictologics.loader import load_image
    from pictologics.warmup import warmup_jit

    feature_modules = _pictologics_feature_modules()

    def _run_warmup() -> None:
        prev = os.environ.get("PICTOLOGICS_DISABLE_WARMUP")
        os.environ["PICTOLOGICS_DISABLE_WARMUP"] = "0"
        try:
            warmup_jit()
        finally:
            if prev is None:
                os.environ.pop("PICTOLOGICS_DISABLE_WARMUP", None)
            else:
                os.environ["PICTOLOGICS_DISABLE_WARMUP"] = prev

    # The package JIT warmup is process start-up work, like the imports, so it
    # runs before the files are read.  It always runs, as at a plain import:
    # --no-warmup drops only the discarded warmup calculation.  Pictologics
    # 1.0.0 can crash in its texture kernels when they load from the numba
    # cache without this warmup.
    start = time.perf_counter()
    _run_warmup()
    jit_warmup_sec = time.perf_counter() - start
    jit_warmup_performed = True
    image = load_image(args.image)
    mask = load_image(args.mask)
    if image.array.shape != mask.array.shape:
        raise ValueError("Shape mismatch between image and mask")
    mask = _to_mask_image(image, mask)
    # Setup time runs from this event to the first timed or warmup call.
    write_event("inputs_loaded")

    families, unsupported_families = requested_families("pictologics", args)
    selected_codes = parse_csv(args.include_ibsi_codes, lowercase=False)
    if selected_codes:
        from bench.ibsi_mapping import pictologics_families_for_codes

        families = pictologics_families_for_codes(selected_codes)
    effective_aggregation = resolve_aggregation(
        "pictologics", args.aggregation, families
    )
    crop_slices, roi_crop_record = roi_crop(mask.array, image.spacing, families)
    image = _crop_image(image, crop_slices)
    mask = _crop_image(mask, crop_slices)

    timing = None
    if args.timed:
        compute_fn, finalize_fn = _prepare_calculation_only(
            image=image,
            mask=mask,
            families=families,
            discretization=args.discretization,
            bins=args.bins,
            bin_width=args.bin_width,
            intensity_range=intensity_range,
            feature_modules=feature_modules,
        )

        def prepare_call() -> None:
            _clear_feature_caches()

        def finalize_call(calculated, state):
            _clear_feature_caches()
            return finalize_fn(calculated, state)

        features, timing = run_adapter_timing(
            lambda _state: compute_fn(),
            iterations=args.iterations,
            no_warmup=args.no_warmup,
            prepare_fn=prepare_call,
            finalize_fn=finalize_call,
        )
    else:
        features = _compute_features(
            image=image,
            mask=mask,
            families=families,
            discretization=args.discretization,
            bins=args.bins,
            bin_width=args.bin_width,
            intensity_range=intensity_range,
            feature_modules=feature_modules,
        )

    if selected_codes:
        from bench.ibsi_mapping import classify_feature

        selected = set(selected_codes)
        features = {
            name: value
            for name, value in features.items()
            if classify_feature("pictologics", name)[0] in selected
        }

    value_payload: Dict[str, float] | None = None
    if args.include_values:
        value_payload = dict(features)

    payload = make_payload(
        adapter="pictologics",
        feature_names=features,
        values=value_payload,
        timing=timing,
        requested=families,
        unsupported=unsupported_families,
        metadata_payload={
            "package_initialization": {
                "jit_warmup_performed": jit_warmup_performed,
                "jit_warmup_sec": jit_warmup_sec,
                "outside_measured_region": True,
                # The package's own default: its fast cores, unless the
                # environment sets PICTOLOGICS_NUM_THREADS.
                "threads": get_num_threads(),
            },
            "roi_crop": roi_crop_record,
            "feature_cache_policy": {
                "rule": "cleared_before_and_after_each_calculation",
                "caches": _clear_feature_caches(),
            },
            "preprocessing": {
                "discretization": args.discretization,
                "bins": int(args.bins),
                "bin_width": float(args.bin_width),
                "intensity_range": list(intensity_range)
                if intensity_range is not None
                else None,
                "fbs_anchor": float(intensity_range[0])
                if args.discretization == "fbs" and intensity_range is not None
                else None,
                "identity_contract": (
                    "native positive-integer grey levels; no grey-level transformation"
                    if args.discretization == "identity"
                    else None
                ),
            },
            "aggregation": {
                "requested": args.aggregation,
                "effective_directional": effective_aggregation,
                "omnidirectional": "3d",
            },
        },
        image_sha256=args.image_sha256,
        source_image_sha256=args.source_image_sha256,
        mask_sha256=args.mask_sha256,
        modality=args.modality,
        input_contract=args.input_contract,
        input_representation_id=args.input_representation_id,
        representation_derivation_sha256=args.representation_derivation_sha256,
        configured_levels=args.configured_levels,
        occupied_levels=args.occupied_levels,
    )

    write_json(payload)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


_DISCRETIZED_FAMILIES = frozenset(
    {"intensity_histogram", "ivh", "glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"}
)


_SPATIAL_FEATURES = frozenset({"morph_moran_i", "morph_geary_c"})


def _keep_morphology_part(name: str, requested: Sequence[str]) -> bool:
    """Keep a MIRP morphology output only when its benchmark part is requested."""

    if not name.startswith("morph_"):
        return True
    wanted = (
        MORPHOLOGY_SPATIAL_REQUESTS
        if name in _SPATIAL_FEATURES
        else MORPHOLOGY_SHAPE_REQUESTS
    )
    return bool(wanted.intersection(requested))


def _infer_modality(image_path: str) -> str | None:
    name = Path(image_path).name.lower()
    if "ct" in name:
        return "ct"
    if "mri" in name or "mr" in name:
        return "mr"
    if "pet" in name or "pt" in name:
        return "pt"
    return None


def _mirp_modality(modality: str | None) -> str | None:
    """Translate benchmark-only modality labels at the MIRP API boundary.

    ``synthetic`` is the provenance label for this benchmark's calibrated-HU
    CT phantom, so it must execute as CT. MIRP 2.7 rejects an explicit
    ``generic`` replacement for NIfTI input. ``other`` therefore leaves the
    native generic modality unchanged by omitting the replacement argument.
    Other values are deliberately left untouched: MIRP validates aliases such
    as CT, MRI/MR, and PET/PT.
    """

    if modality is not None and modality.strip().lower() == "synthetic":
        return "ct"
    if modality is not None and modality.strip().lower() == "other":
        return None
    return modality


def _require_fbs_range(
    *,
    discretization: str,
    families: Sequence[str],
    intensity_range: Optional[Tuple[float, float]],
) -> None:
    if (
        discretization == "fbs"
        and set(families).intersection(_DISCRETIZED_FAMILIES)
        and intensity_range is None
    ):
        raise ValueError(
            "MIRP fixed-bin-size extraction requires an explicit intensity range "
            "for the IBSI lower-bin anchor"
        )


def _build_settings(
    *,
    families: Sequence[str],
    discretization: str,
    bins: int,
    bin_width: float,
    intensity_range: Optional[Tuple[float, float]],
    aggregation: str = "3d_merge",
    identity_ivh_bins: Optional[int] = None,
    ibsi_compliant: bool = True,
):
    """Build public MIRP 2.7 settings for one native family selection."""

    from mirp.settings.feature_parameters import FeatureExtractionSettingsClass
    from mirp.settings.generic import SettingsClass
    from mirp.settings.resegmentation_parameters import ResegmentationSettingsClass

    _require_fbs_range(
        discretization=discretization,
        families=families,
        intensity_range=intensity_range,
    )
    if discretization == "raw" and set(families).intersection(_DISCRETIZED_FAMILIES):
        raise ValueError(
            "Raw extraction is valid only for non-discretized feature families"
        )
    if discretization == "fbn" and int(bins) <= 0:
        raise ValueError("MIRP fixed-bin-number discretization requires bins > 0")
    if discretization == "fbs" and (
        not math.isfinite(float(bin_width)) or float(bin_width) <= 0
    ):
        raise ValueError("MIRP fixed-bin-size discretization requires bin width > 0")

    method = {
        "fbn": "fixed_bin_number",
        "fbs": "fixed_bin_size",
        "identity": "none",
        "raw": "none",
    }[discretization]
    ivh_method = (
        "fixed_bin_number"
        if discretization == "identity" and identity_ivh_bins is not None
        else method
    )
    feature_kwargs: Dict[str, Any] = {
        "by_slice": False,
        # Prefer the exact IBSI formulation (not MIRP's optional approximation
        # for expensive morphology-correlation features). Runner guardrails
        # handle the resulting cost on larger images.
        "no_approximation": True,
        # In MIRP this switch filters the generated feature objects; it does
        # not merely annotate them.  Strict code-selected conformance calls
        # retain the filter, whereas performance calls disable it so the
        # benchmark exercises MIRP's complete eligible native family surface.
        "ibsi_compliant": bool(ibsi_compliant),
        "base_feature_families": list(families),
        "base_discretisation_method": method,
        "ivh_discretisation_method": ivh_method,
    }
    directional_method = {
        "2d_average": "2d_average",
        "2d_slice_merge": "2d_slice_merge",
        "2.5d_direction_merge": "2.5d_direction_merge",
        "2.5d_merge": "2.5d_volume_merge",
        "3d_average": "3d_average",
        "3d_merge": "3d_volume_merge",
    }[aggregation]
    texture_dimension = (
        "2.5d"
        if aggregation.startswith("2.5d_")
        else "2d"
        if aggregation.startswith("2d_")
        else "3d"
    )
    feature_kwargs.update(
        {
            "glcm_spatial_method": directional_method,
            "glrlm_spatial_method": directional_method,
            "glszm_spatial_method": texture_dimension,
            "gldzm_spatial_method": texture_dimension,
            "ngtdm_spatial_method": texture_dimension,
            "ngldm_spatial_method": texture_dimension,
        }
    )
    if discretization == "fbn":
        feature_kwargs.update(
            {
                "base_discretisation_n_bins": int(bins),
                "ivh_discretisation_n_bins": int(bins),
            }
        )
    elif discretization == "fbs":
        feature_kwargs.update(
            {
                "base_discretisation_bin_width": float(bin_width),
                "ivh_discretisation_bin_width": float(bin_width),
            }
        )
    elif identity_ivh_bins is not None:
        if int(identity_ivh_bins) < 2:
            raise ValueError("MIRP identity IVH requires at least two integer levels")
        feature_kwargs["ivh_discretisation_n_bins"] = int(identity_ivh_bins)

    feature_settings = FeatureExtractionSettingsClass(**feature_kwargs)
    resegmentation_settings = ResegmentationSettingsClass(
        resegmentation_intensity_range=(
            [float(intensity_range[0]), float(intensity_range[1])]
            if intensity_range is not None
            else None
        )
    )
    return SettingsClass(
        feature_extr_settings=feature_settings,
        roi_resegment_settings=resegmentation_settings,
    )


def _load_native_pair(
    mirp_module,
    *,
    image_path: str,
    mask_path: str,
    modality: str | None,
):
    """Load NIfTI once through MIRP's public native-image export path."""

    kwargs: Dict[str, Any] = {
        "image": image_path,
        "mask": mask_path,
        "image_file_type": "nifti",
        "mask_file_type": "nifti",
        "write_images": False,
        "export_images": True,
        "image_export_format": "native",
        "num_cpus": 1,
    }
    if modality is not None:
        kwargs["image_modality"] = modality

    result = mirp_module.extract_images(**kwargs)
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        count = len(result) if isinstance(result, (list, tuple)) else "non-sequence"
        raise RuntimeError(
            f"MIRP native loading expected one workflow result, got {count}"
        )

    entry = result[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise RuntimeError(
            "MIRP native loading returned an unexpected result structure"
        )
    images, masks = entry
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise RuntimeError("MIRP native loading expected exactly one image")
    if not isinstance(masks, (list, tuple)) or len(masks) != 1:
        raise RuntimeError("MIRP native loading expected exactly one mask")
    if images[0] is None or masks[0] is None:
        raise RuntimeError("MIRP native loading returned an empty image or mask")
    return images[0], masks[0]


def _identity_ivh_bin_count(image, mask) -> int:
    """Return the full integer grey-level span for identity IVH in MIRP.

    MIRP's generic-image fallback changes ``ivh_discretisation_method='none'``
    to FBN 1000.  FBN over the complete integer span is exactly the direct
    phantom grid, including absent intermediate levels, and avoids that
    modality-dependent fallback.
    """

    import numpy as np

    image_values = np.asarray(image.get_voxel_grid(), dtype=float)
    roi_image = getattr(mask, "roi_intensity", None)
    if roi_image is None:
        roi_image = getattr(mask, "roi", None)
    if roi_image is None:
        raise ValueError("MIRP identity IVH requires a readable intensity ROI")
    roi_mask = np.asarray(roi_image.get_voxel_grid(), dtype=bool)
    values = image_values[roi_mask]
    if (
        values.size == 0
        or not np.isfinite(values).all()
        or np.any(values < 1.0)
        or not np.allclose(values, np.rint(values))
        or float(np.min(values)) != 1.0
    ):
        raise ValueError(
            "MIRP identity IVH requires finite positive-integer ROI levels with minimum 1"
        )
    return int(np.max(values) - np.min(values) + 1)


def _table_to_values(table, *, families: Sequence[str]) -> Dict[str, float]:
    if table is None or bool(getattr(table, "empty", False)):
        raise RuntimeError(
            "MIRP returned an empty feature table for requested families: "
            + ", ".join(families)
        )
    shape = getattr(table, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) != 1:
        raise RuntimeError(f"MIRP expected one feature row, got {shape[0]}")

    row = table.iloc[0]
    values: Dict[str, float] = {}
    names_seen: set[str] = set()
    for column in table.columns:
        name = str(column).strip()
        if not name or name in names_seen:
            raise RuntimeError(
                f"MIRP returned an empty or duplicate feature column: {name!r}"
            )
        names_seen.add(name)
        # Public MIRP tables prepend workflow/provenance columns. They are not
        # calculated imaging features and must not inflate feature coverage.
        if name == "sample_name" or name.startswith("image_"):
            continue
        raw = row[column]
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise RuntimeError(
                f"MIRP returned a nonnumeric calculated feature {name!r}: {raw!r}"
            )
        value = float(raw)
        if not math.isfinite(value):
            raise RuntimeError(
                f"MIRP returned a non-finite calculated feature {name!r}: {raw!r}"
            )
        values[name] = value

    if not values:
        raise RuntimeError(
            "MIRP returned zero finite calculated features for requested families: "
            + ", ".join(families)
        )
    return values


def _extract_public(
    mirp_module,
    *,
    image,
    mask,
    settings,
    families: Sequence[str],
    requested: Sequence[str] = ("morphology",),
) -> Dict[str, float]:
    """Execute the documented MIRP public feature-extraction workflow."""

    try:
        result = mirp_module.extract_features(
            image=image,
            mask=mask,
            settings=settings,
            write_features=False,
            export_features=True,
            num_cpus=1,
        )
    except Exception as exc:
        raise RuntimeError(
            "MIRP public extraction failed for "
            + ", ".join(families)
            + f": {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(result, (list, tuple)) or len(result) != 1:
        count = len(result) if isinstance(result, (list, tuple)) else "non-sequence"
        raise RuntimeError(
            f"MIRP expected one exported feature table for {', '.join(families)}, got {count}"
        )
    values = _table_to_values(result[0], families=families)
    return {
        name: value
        for name, value in values.items()
        if _keep_morphology_part(name, requested)
    }


def _prepare_calculation_only(
    *,
    image,
    mask,
    settings,
    families: Sequence[str],
    identity_input: bool = False,
    requested: Sequence[str] = ("morphology",),
):
    """Prepare MIRP feature objects and return calculation/finalization hooks.

    MIRP's public workflow is retained for conformance and untimed extraction.
    For the calculation-only performance endpoint we use the pinned MIRP 2.7
    feature-generator API: image loading/registration, mask resegmentation,
    feature-object generation, table construction and name normalization are
    all outside the calculation clock.  Matrix-cache sharing between adjacent
    compatible features remains inside because it is part of MIRP's native
    calculation strategy.
    """

    import copy

    from mirp._features.feature_generator import feature_to_table, generate_features

    prepared_mask = copy.deepcopy(mask)
    resegmentation = settings.roi_resegment
    if resegmentation is not None:
        prepared_mask.resegmentise_mask(
            image=image,
            resegmentation_method=resegmentation.resegmentation_method,
            intensity_range=resegmentation.intensity_range,
            sigma=resegmentation.sigma,
        )

    feature_settings = settings.feature_extr
    features = [
        feature
        for feature in generate_features(settings=feature_settings)
        if _keep_morphology_part(str(feature.abbr_name or ""), requested)
    ]
    if feature_settings.ibsi_compliant:
        features = [
            feature for feature in features if feature.is_ibsi_compliant(image=image)
        ]
    if not features:
        raise RuntimeError(
            "MIRP generated zero feature objects for requested families: "
            + ", ".join(families)
        )

    if identity_input and "ivh" in families:
        # The benchmark IVH file already stores the exact one-based FBS1 bin
        # indices.  MIRP's modality fallback would otherwise discretize that
        # grid again.  Retain MIRP's native IVH curve and feature calculation,
        # but select its documented direct-integer path on this prepared image.
        image.get_default_ivh_discretisation_method = lambda: "none"
        for feature in features:
            if hasattr(feature, "_get_data"):
                feature.discretisation_method = "none"
                feature.bin_number = None
                feature.bin_width = None

    def prime_discretization() -> None:
        for feature in features:
            if hasattr(feature, "_get_data"):
                # For IVH, _get_data also builds the histogram and cumulative
                # curve, so it is intentionally left inside the timer.
                continue
            discretise = getattr(feature, "discretise_image", None)
            if discretise is None:
                continue
            discretise(
                image=image,
                mask=prepared_mask,
                discretisation_method=feature.discretisation_method,
                bin_width=feature.bin_width,
                bin_number=feature.bin_number,
                cropping_distance=feature.cropping_distance,
            )

    prime_discretization()

    def calculate():
        previous_feature = None
        for feature in features:
            feature.compute(image=image, mask=prepared_mask)
            if previous_feature is not None:
                previous_feature.clear_local_cache(other=feature)
            previous_feature = feature
        return features

    def finalize(calculated_features, _state=None) -> Dict[str, float]:
        try:
            table = feature_to_table(calculated_features)
            table = image.parse_feature_names(table)
            return _table_to_values(table, families=families)
        finally:
            for feature in calculated_features:
                feature.clear_cache()
            prime_discretization()

    return calculate, finalize


def main(argv: List[str] | None = None) -> int:
    import argparse
    import logging

    parser = argparse.ArgumentParser(prog="mirp-adapter")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    intensity_range = parse_intensity_range(args)

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("mirp").setLevel(logging.ERROR)

    import mirp

    families, unsupported_families = requested_families("mirp", args)
    selected_codes = parse_csv(args.include_ibsi_codes, lowercase=False)
    if selected_codes:
        from bench.ibsi_families import CODE_TO_FAMILY, FAMILY_ORDER
        from bench.ibsi_mapping import mirp_families_for_codes

        native_families = mirp_families_for_codes(selected_codes)
        selected_family_set = {
            CODE_TO_FAMILY[code] for code in selected_codes if code in CODE_TO_FAMILY
        }
        families = [family for family in FAMILY_ORDER if family in selected_family_set]
        if not native_families or not families:
            raise RuntimeError(
                "No MIRP feature family matched the requested IBSI codes"
            )
    else:
        from bench.ibsi_mapping import IBSI_FAMILY_TO_MIRP

        native_families = list(
            dict.fromkeys(IBSI_FAMILY_TO_MIRP[family] for family in families)
        )

    effective_aggregation = resolve_aggregation("mirp", args.aggregation, families)

    _require_fbs_range(
        discretization=args.discretization,
        families=native_families,
        intensity_range=intensity_range,
    )
    source_modality = args.modality or _infer_modality(args.image)
    modality = _mirp_modality(source_modality)
    image, mask = _load_native_pair(
        mirp,
        image_path=args.image,
        mask_path=args.mask,
        modality=modality,
    )
    # Setup time runs from this event to the first timed or warmup call.
    write_event("inputs_loaded")
    # MIRP's feature objects scan the whole grid they get, so crop to the ROI
    # box before any timer, as the other adapters do.  MIRP axes are (z, y, x).
    crop_slices, roi_crop_record = roi_crop(
        mask.roi.get_voxel_grid(), image.image_spacing, families
    )
    bounds = {
        name: [item.start, item.stop - 1]
        for name, item in zip(("ind_ext_z", "ind_ext_y", "ind_ext_x"), crop_slices)
    }
    image.crop(**bounds)
    mask.crop(**bounds)
    # MIRP keeps a view of the full grid; a copy frees it.  Masks are encoded.
    import numpy as np

    image.set_voxel_grid(np.ascontiguousarray(image.get_voxel_grid()))
    identity_ivh_bins = (
        _identity_ivh_bin_count(image, mask)
        if args.discretization == "identity" and "ivh" in families
        else None
    )
    settings = _build_settings(
        families=native_families,
        discretization=args.discretization,
        bins=args.bins,
        bin_width=args.bin_width,
        intensity_range=intensity_range,
        aggregation=effective_aggregation,
        identity_ivh_bins=identity_ivh_bins,
        ibsi_compliant=bool(selected_codes),
    )

    timing = None
    if args.timed:
        compute_fn, finalize_fn = _prepare_calculation_only(
            image=image,
            mask=mask,
            settings=settings,
            families=native_families,
            identity_input=args.discretization == "identity",
            requested=families,
        )
        values, timing = run_adapter_timing(
            compute_fn,
            iterations=args.iterations,
            no_warmup=args.no_warmup,
            finalize_fn=finalize_fn,
        )
    else:
        values = _extract_public(
            mirp,
            image=image,
            mask=mask,
            settings=settings,
            families=native_families,
            requested=families,
        )

    if selected_codes:
        from bench.ibsi_mapping import classify_feature

        selected = set(selected_codes)
        values = {
            name: value
            for name, value in values.items()
            if classify_feature("mirp", name)[0] in selected
        }
        if not values:
            raise RuntimeError(
                "MIRP returned zero features matching the requested IBSI codes"
            )

    payload = make_payload(
        adapter="mirp",
        feature_names=values,
        values=values if args.include_values else None,
        timing=timing,
        requested=families,
        unsupported=unsupported_families,
        metadata_payload={
            "roi_crop": roi_crop_record,
            "input_execution_scope": (
                (
                    "native_objects_mask_and_feature_objects_prepared_outside_timing; "
                    "matrix_cache_sharing_and_feature_formulas_inside_timing"
                )
                if args.timed
                else (
                    "native_objects_preloaded_outside_timing; "
                    "mirp_public_extract_features_internal_copy_inside_timing"
                )
            ),
            "api": (
                "mirp.extract_images + pinned mirp._features.feature_generator "
                "calculation boundary"
                if args.timed
                else "mirp.extract_images + mirp.extract_features"
            ),
            "feature_selection": {
                "contract": (
                    "strict_ibsi_compliant_code_selection"
                    if selected_codes
                    else "complete_native_3d_family_surface"
                ),
                "mirp_ibsi_compliant_filter": bool(selected_codes),
            },
            "modality_bridge": {
                "benchmark": args.modality,
                "effective_mirp": modality,
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
                    "native integer base grid plus full-span FBN-equivalent direct IVH grid"
                    if args.discretization == "identity"
                    else None
                ),
                "identity_ivh_bins": identity_ivh_bins,
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

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from bench.adapters.base import write_event, write_json
from bench.adapters.protocol import (
    DIRECTIONAL_TEXTURE_FAMILIES,
    add_common_arguments,
    make_payload,
    parse_csv,
    parse_intensity_range,
    requested_families,
    resolve_aggregation,
    roi_crop,
    run_adapter_timing,
)
from bench.ibsi_families import CODE_TO_FAMILY


_PYRADIOMICS_NATIVE_CLASS_BY_FAMILY = {
    "morphology": "shape",
    "morphology_shape": "shape",
    "glcm": "glcm",
    "glrlm": "glrlm",
    "glszm": "glszm",
    "ngtdm": "ngtdm",
    "ngldm": "gldm",
}
# Native classes whose constructor performs measured calculation work.
_TIMED_CONSTRUCTOR_CLASSES = frozenset({"shape"})


def _filter_selection(
    selection: Mapping[str, Sequence[str]],
    feature_registry: Mapping[str, Mapping[str, bool]],
) -> Dict[str, List[str]]:
    """Retain only feature names exposed by the installed PyRadiomics build."""

    filtered: Dict[str, List[str]] = {}
    for class_name, features in selection.items():
        available = feature_registry.get(class_name, {})
        keep = sorted({feature for feature in features if feature in available})
        if keep:
            filtered[class_name] = keep
    return filtered


def _native_family_selection(
    families: Sequence[str],
    feature_registry: Mapping[str, Mapping[str, bool]],
) -> Dict[str, List[str]]:
    """Select the complete eligible native 3D surface for performance tasks.

    Compliance requests use :func:`pyradiomics_feature_selection` directly and
    therefore remain limited to reviewed IBSI identities.  Performance tasks
    additionally retain calculated upstream outputs that have no IBSI Table 2
    identity.  Deprecated methods remain disabled except for the three reviewed
    shape methods that calculate direct IBSI definitions when named explicitly.

    PyRadiomics exposes raw intensity statistics and discretised histogram
    statistics from the same ``firstorder`` class.  Partition that class so the
    native outputs are calculated once across the canonical family workloads:
    entropy/uniformity belong to ``histogram`` and all other active outputs,
    including non-IBSI ``TotalEnergy``, belong to ``intensity``.
    """

    from bench.ibsi_mapping import (
        PYRADIOMICS_EXPLICIT_DEPRECATED_FEATURES,
        pyradiomics_feature_selection,
    )

    requested = set(families)
    missing_classes = sorted(
        {
            class_name
            for family, class_name in _PYRADIOMICS_NATIVE_CLASS_BY_FAMILY.items()
            if family in requested and class_name not in feature_registry
        }
        | (
            {"firstorder"}
            if requested.intersection({"intensity", "histogram"})
            and "firstorder" not in feature_registry
            else set()
        )
    )
    if missing_classes:
        raise RuntimeError(
            "Installed PyRadiomics build lacks required feature classes: "
            + ", ".join(missing_classes)
        )

    explicit_deprecated = set(PYRADIOMICS_EXPLICIT_DEPRECATED_FEATURES)

    def eligible(class_name: str) -> set[str]:
        return {
            feature
            for feature, deprecated in feature_registry.get(class_name, {}).items()
            if not deprecated or (class_name, feature) in explicit_deprecated
        }

    selected: Dict[str, set[str]] = {}
    for family, class_name in _PYRADIOMICS_NATIVE_CLASS_BY_FAMILY.items():
        if family in requested:
            selected.setdefault(class_name, set()).update(eligible(class_name))

    histogram_codes = [
        code for code, family in CODE_TO_FAMILY.items() if family == "histogram"
    ]
    histogram_features = set(
        pyradiomics_feature_selection(histogram_codes).get("firstorder", [])
    ).intersection(eligible("firstorder"))
    if "histogram" in requested:
        selected.setdefault("firstorder", set()).update(histogram_features)
    if "intensity" in requested:
        selected.setdefault("firstorder", set()).update(
            eligible("firstorder").difference(histogram_features)
        )

    return {
        class_name: sorted(features)
        for class_name, features in sorted(selected.items())
        if features
    }


def _aggregation_settings(
    effective_aggregation: str,
    families: List[str],
) -> dict:
    """Translate the reviewed aggregation contract to PyRadiomics settings.

    PyRadiomics' default (``weightingNorm=None``) calculates a feature for each
    direction and averages the feature values. Its documented
    ``no_weighting`` mode instead applies unit weights, sums the GLCM/GLRLM
    direction matrices, normalises the resultant matrix, and then calculates
    the feature. With ``force2D=False`` this is IBSI 3D merged aggregation.
    """

    if (
        effective_aggregation == "3d_merge"
        and DIRECTIONAL_TEXTURE_FAMILIES.intersection(families)
    ):
        return {"weightingNorm": "no_weighting"}
    return {}


def _default_settings() -> dict:
    return {
        "binWidth": 1.0,
        "resampledPixelSpacing": None,
        "interpolator": "sitkNearestNeighbor",
        "label": 1,
        "force2D": False,
        "additionalInfo": False,
    }


def _normalize_value(feature_name: str, value: float) -> float:
    if feature_name == "original_firstorder_Kurtosis":
        return value - 3.0
    return value


def _finite_calculated_values(result) -> Dict[str, float]:
    """Normalize calculated outputs and fail on any invalid scalar value."""

    import numbers

    import numpy as np

    values: Dict[str, float] = {}
    for raw_name, raw_value in result.items():
        name = str(raw_name).strip()
        if name.startswith("diagnostics"):
            continue
        if not name or name in values:
            raise RuntimeError(
                f"PyRadiomics returned an empty or duplicate feature name: {name!r}"
            )
        if isinstance(raw_value, (bool, np.bool_)):
            raise RuntimeError(f"PyRadiomics feature {name} is not a numeric scalar")
        try:
            array = np.asarray(raw_value).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"PyRadiomics feature {name} is not a numeric scalar"
            ) from exc
        if array.size != 1:
            raise RuntimeError(f"PyRadiomics feature {name} is not one finite scalar")
        scalar = array[0]
        if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, numbers.Real):
            raise RuntimeError(f"PyRadiomics feature {name} is not a numeric scalar")
        value = float(scalar)
        if not np.isfinite(value):
            raise RuntimeError(f"PyRadiomics feature {name} is not one finite scalar")
        values[name] = _normalize_value(name, value)
    if not values:
        raise RuntimeError("PyRadiomics returned zero calculated finite features")
    return values


def _validate_identity_image(image, mask, sitk_module) -> None:
    import numpy as np

    image_values = np.asarray(sitk_module.GetArrayFromImage(image), dtype=float)
    mask_values = np.asarray(sitk_module.GetArrayFromImage(mask))
    roi = image_values[mask_values > 0]
    if roi.size == 0:
        raise ValueError("Identity discretization requires a non-empty ROI")
    if (
        not np.isfinite(roi).all()
        or np.any(roi < 1.0)
        or not np.allclose(roi, np.rint(roi))
        or float(np.min(roi)) != 1.0
    ):
        raise ValueError(
            "PyRadiomics identity emulation requires finite positive-integer ROI "
            "grey levels with minimum 1"
        )


def _binary_mask(mask, sitk_module):
    """Normalize the preloaded ROI mask before measured extraction."""

    return sitk_module.Cast(mask > 0, sitk_module.sitkUInt8)


def _prepare_calculation_only(
    *,
    image,
    mask,
    selection: Mapping[str, Sequence[str]],
    settings: Mapping[str, object],
    radiomics_module,
):
    """Prepare PyRadiomics feature classes outside the calculation clock.

    The public ``RadiomicsFeatureExtractor.execute`` method intentionally owns
    loading, mask validation, resegmentation, cropping and class construction.
    That is the correct native end-to-end API, but it is too broad for this
    benchmark's calculation-only endpoint.  First-order and texture
    constructors convert the already cropped SimpleITK inputs to arrays and
    perform grey-level bin assignment; their ``execute`` methods then build
    the matrices/neighbourhoods and evaluate the selected feature formulas.
    The shape constructor instead calculates the surface mesh, mesh volume,
    maximum diameters and principal axes, so the shape class is constructed
    inside the calculation clock.
    """

    from radiomics import imageoperations

    prepared_settings = dict(settings)
    label = int(prepared_settings.get("label", 1))
    bounding_box, corrected_mask = imageoperations.checkMask(
        image, mask, **prepared_settings
    )
    if corrected_mask is not None:
        mask = corrected_mask

    resegmented_mask = None
    if prepared_settings.get("resegmentRange") is not None:
        resegmented_mask = imageoperations.resegmentMask(
            image, mask, **prepared_settings
        )
        bounding_box, corrected_mask = imageoperations.checkMask(
            image, resegmented_mask, **prepared_settings
        )
        if corrected_mask is not None:
            resegmented_mask = corrected_mask

    resegment_shape = bool(prepared_settings.get("resegmentShape", False))
    shape_mask = (
        resegmented_mask if resegment_shape and resegmented_mask is not None else mask
    )
    intensity_mask = (
        resegmented_mask
        if not resegment_shape and resegmented_mask is not None
        else shape_mask
    )
    feature_classes = radiomics_module.getFeatureClasses()

    def build(class_name, cropped_image, cropped_mask, feature_names):
        feature_class = feature_classes[class_name](
            cropped_image,
            cropped_mask,
            **prepared_settings,
        )
        feature_class.label = label
        for feature_name in feature_names:
            feature_class.enableFeatureByName(feature_name)
        return feature_class

    prepared = []
    for class_name, feature_names in selection.items():
        class_mask = shape_mask if class_name == "shape" else intensity_mask
        # One background voxel, as ``roi_crop`` keeps for the other adapters.
        # The shape class pads its mask by one voxel itself, so it gets the
        # native crop with no padding and the same final array.
        cropped_image, cropped_mask = imageoperations.cropToTumorMask(
            image,
            class_mask,
            bounding_box,
            padDistance=0 if class_name == "shape" else 1,
        )
        inputs = (class_name, cropped_image, cropped_mask, feature_names)
        if class_name in _TIMED_CONSTRUCTOR_CLASSES:
            prepared.append((class_name, None, inputs))
        else:
            prepared.append((class_name, build(*inputs), None))

    def calculate() -> List[tuple[str, Mapping[str, object]]]:
        calculated: List[tuple[str, Mapping[str, object]]] = []
        for class_name, feature_class, deferred in prepared:
            if feature_class is None:
                feature_class = build(*deferred)
            calculated.append((class_name, feature_class.execute()))
        return calculated

    def finalize(
        calculated: Sequence[tuple[str, Mapping[str, object]]], _state=None
    ) -> Dict[str, float]:
        raw_values: Dict[str, object] = {}
        for class_name, class_values in calculated:
            for feature_name, value in class_values.items():
                output_name = f"original_{class_name}_{feature_name}"
                if output_name in raw_values:
                    raise RuntimeError(
                        "PyRadiomics returned a duplicate prepared feature name: "
                        + output_name
                    )
                raw_values[output_name] = value
        return _finite_calculated_values(raw_values)

    return calculate, finalize


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="pyradiomics-adapter")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    intensity_range = parse_intensity_range(args)

    import logging
    import radiomics
    import SimpleITK as sitk

    logging.getLogger("radiomics").setLevel(logging.ERROR)

    image = sitk.ReadImage(args.image)
    mask = _binary_mask(sitk.ReadImage(args.mask), sitk)
    # Setup time runs from this event to the first timed or warmup call.
    write_event("inputs_loaded")

    families, unsupported_families = requested_families("pyradiomics", args)
    requested_codes = parse_csv(args.include_ibsi_codes, lowercase=False)
    if requested_codes:
        requested_family_set = {
            CODE_TO_FAMILY[code] for code in requested_codes if code in CODE_TO_FAMILY
        }
        families = [family for family in families if family in requested_family_set]
    effective_aggregation = resolve_aggregation(
        "pyradiomics", args.aggregation, families
    )

    settings = _default_settings()
    settings.update(_aggregation_settings(effective_aggregation, families))
    if args.discretization == "fbn":
        settings["binCount"] = int(args.bins)
    elif args.discretization == "fbs":
        settings["binWidth"] = float(args.bin_width)
    elif args.discretization == "identity":
        _validate_identity_image(image, mask, sitk)
        # PyRadiomics has no no-discretisation mode.  With positive integer
        # grey levels starting at one, fixed-bin-size width one is exactly the
        # identity transform; this equivalence is validated above and recorded.
        settings["binWidth"] = 1.0
    elif set(families).intersection(
        {"histogram", "ivh", "glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"}
    ):
        raise ValueError(
            "Raw extraction is valid only for non-discretized feature families"
        )
    if intensity_range is not None:
        settings["resegmentRange"] = [
            float(intensity_range[0]),
            float(intensity_range[1]),
        ]
        settings["resegmentMode"] = "absolute"

    feature_registry = {
        name: dict(cls.getFeatureNames())
        for name, cls in radiomics.getFeatureClasses().items()
    }

    if requested_codes:
        from bench.ibsi_mapping import pyradiomics_feature_selection

        primary_selection = _filter_selection(
            pyradiomics_feature_selection(requested_codes), feature_registry
        )
        if not primary_selection:
            raise RuntimeError("No PyRadiomics features matched provided IBSI codes")
        selection_contract = "reviewed_ibsi_codes"
    else:
        primary_selection = _native_family_selection(families, feature_registry)
        if not primary_selection:
            raise RuntimeError(
                "No eligible native PyRadiomics features matched requested families"
            )
        selection_contract = "complete_native_3d_family_surface"
    _, roi_crop_record = roi_crop(
        sitk.GetArrayViewFromImage(mask), mask.GetSpacing()[::-1], families
    )
    timing = None
    compute_fn, finalize_fn = _prepare_calculation_only(
        image=image,
        mask=mask,
        selection=primary_selection,
        settings=settings,
        radiomics_module=radiomics,
    )
    # The prepared classes hold cropped copies, so the full images go.
    del image, mask

    if args.timed:
        result, timing = run_adapter_timing(
            compute_fn,
            iterations=args.iterations,
            no_warmup=args.no_warmup,
            finalize_fn=finalize_fn,
        )
    else:
        result = finalize_fn(compute_fn())

    features: Dict[str, List[str]] = {"all": list(result)}
    values: Dict[str, Dict[str, float]] = {"all": dict(result)}

    payload = make_payload(
        adapter="pyradiomics",
        feature_names=features["all"],
        values=values["all"] if args.include_values else None,
        timing=timing,
        requested=families,
        unsupported=unsupported_families,
        metadata_payload={
            "roi_crop": roi_crop_record,
            "preprocessing": {
                "discretization": args.discretization,
                "bins": int(args.bins),
                "bin_width": float(args.bin_width),
                "intensity_range": list(intensity_range)
                if intensity_range is not None
                else None,
                "identity_contract": (
                    "validated FBS width 1 equivalence for positive integer ROI with minimum 1"
                    if args.discretization == "identity"
                    else None
                ),
            },
            "aggregation": {
                "requested": args.aggregation,
                "effective_directional": effective_aggregation,
                "omnidirectional": "3d",
                "pyradiomics_weighting_norm": settings.get("weightingNorm"),
                "directional_matrix_operation": (
                    "unit-weighted sum before feature calculation"
                    if settings.get("weightingNorm") == "no_weighting"
                    else "per-direction feature averaging"
                    if DIRECTIONAL_TEXTURE_FAMILIES.intersection(families)
                    else "not applicable"
                ),
            },
            "feature_selection": {
                "contract": selection_contract,
                "dimension": "3d",
                "shape2d_class_enabled": False,
                "selected_by_class": primary_selection,
            },
            "timing_boundary": {
                "classes_constructed_inside_timer": sorted(
                    _TIMED_CONSTRUCTOR_CLASSES.intersection(primary_selection)
                ),
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

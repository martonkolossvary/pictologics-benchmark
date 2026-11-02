from __future__ import annotations

import math
import importlib
from collections.abc import Mapping, MutableMapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bench.adapters.base import write_event, write_json
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


ZRAD_NATIVE_FAMILIES: Dict[str, Tuple[str, ...]] = {
    # Moran's I and Geary's C are IBSI morphology features. Z-Rad puts them
    # in an opt-in native group because their calculation can be expensive.
    "morphology": ("morphology", "morphology_correlation"),
    "morphology_shape": ("morphology",),
    "morphology_spatial": ("morphology_correlation",),
    "local_intensity": ("local_intensity",),
    "intensity": ("intensity_statistics",),
    "histogram": ("intensity_histogram",),
    "ivh": ("ivh",),
    "glcm": ("glcm",),
    "glrlm": ("glrlm",),
    "glszm": ("glszm",),
    "gldzm": ("gldzm",),
    "ngtdm": ("ngtdm",),
    "ngldm": ("ngldm",),
}

_TEXTURE_DISCRETIZED_FAMILIES = frozenset(
    {"histogram", "glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"}
)


class ZRadExtractionError(RuntimeError):
    """A Z-Rad adapter failure with machine-inspectable execution context."""

    def __init__(
        self,
        *,
        phase: str,
        families: Sequence[str],
        message: str,
    ) -> None:
        self.phase = phase
        self.families = tuple(families)
        family_text = ",".join(self.families) or "none"
        super().__init__(
            f"Z-Rad adapter failure [phase={phase}; families={family_text}]: {message}"
        )


def _native_families(families: Sequence[str]) -> List[str]:
    native: List[str] = []
    for family in families:
        try:
            mapped = ZRAD_NATIVE_FAMILIES[family]
        except KeyError as exc:
            raise ValueError(f"Unsupported Z-Rad family: {family}") from exc
        for native_family in mapped:
            if native_family not in native:
                native.append(native_family)
    if not native:
        raise ValueError("At least one Z-Rad feature family must be selected")
    return native


def _load_nifti_pair(image_path: Path, mask_path: Path):
    """Load an aligned image/mask pair through the Z-Rad 26.8 public API."""

    from zrad.image import Image

    image = Image.from_nifti(str(image_path))
    mask = Image.from_nifti_mask(str(mask_path), reference=image)

    if image.array is None or mask.array is None:
        raise RuntimeError("Z-Rad failed to load image/mask arrays")
    if image.array.shape != mask.array.shape:
        raise RuntimeError(
            f"Image/mask shape mismatch for Z-Rad: {image.array.shape} vs {mask.array.shape}"
        )
    return image, mask


def _zrad_preprocessing_classes() -> Dict[str, Any]:
    """Import Z-Rad preprocessing APIs before any measured iteration."""

    from zrad.preprocessing import (
        IVHIntensityDiscretizer,
        IntensityMaskBuilder,
        Resegmenter,
        RoiCropper,
        RoiData,
        TextureDiscretizer,
    )

    return {
        "IVHIntensityDiscretizer": IVHIntensityDiscretizer,
        "IntensityMaskBuilder": IntensityMaskBuilder,
        "Resegmenter": Resegmenter,
        "RoiCropper": RoiCropper,
        "RoiData": RoiData,
        "TextureDiscretizer": TextureDiscretizer,
    }


def _clear_zrad_local_intensity_cache(families: Sequence[str]) -> None:
    """Keep every local-intensity repeat cold and release cached image arrays."""

    if "local_intensity" not in families:
        return
    module = importlib.import_module("zrad.radiomics.intensity")
    cache = getattr(module, "_LOCAL_MEANS_CACHE", None)
    if not isinstance(cache, MutableMapping):
        raise RuntimeError(
            "Z-Rad local-intensity cache control is unavailable for the pinned runtime"
        )
    cache.clear()


def _prepare_roi_data(
    *,
    image,
    mask,
    families: Sequence[str],
    discretization: str,
    bins: int,
    bin_width: float,
    intensity_range: Optional[Tuple[float, float]],
    preprocessing_classes: Optional[Dict[str, Any]] = None,
):
    """Build the prepared ``RoiData`` required by selected native families."""

    classes = (
        preprocessing_classes
        if preprocessing_classes is not None
        else _zrad_preprocessing_classes()
    )
    IVHIntensityDiscretizer = classes["IVHIntensityDiscretizer"]
    IntensityMaskBuilder = classes["IntensityMaskBuilder"]
    Resegmenter = classes["Resegmenter"]
    RoiCropper = classes["RoiCropper"]
    RoiData = classes["RoiData"]
    TextureDiscretizer = classes["TextureDiscretizer"]

    if discretization not in {"fbn", "fbs", "identity", "raw"}:
        raise ValueError(f"Unsupported discretization method: {discretization}")
    if discretization == "fbn" and int(bins) <= 0:
        raise ValueError("Z-Rad fixed-bin-number discretization requires bins > 0")
    if discretization == "fbs" and (
        not math.isfinite(float(bin_width)) or float(bin_width) <= 0
    ):
        raise ValueError("Z-Rad fixed-bin-size discretization requires bin width > 0")

    selected = set(families)
    need_texture = bool(selected.intersection(_TEXTURE_DISCRETIZED_FAMILIES))
    need_ivh = "ivh" in selected
    if discretization == "raw" and (need_texture or need_ivh):
        raise ValueError(
            "Raw extraction is valid only for non-discretized feature families"
        )
    # Z-Rad 26.8 deliberately requires a stable, explicit FBS anchor. Silently
    # substituting the per-ROI minimum would change the discretization contract.
    if (
        discretization == "fbs"
        and (need_texture or need_ivh)
        and intensity_range is None
    ):
        raise ValueError(
            "Z-Rad fixed-bin-size texture/IVH extraction requires --intensity-min "
            "and --intensity-max so the lower discretization anchor is explicit"
        )

    # Crop first, so the masks and discretisers see only the ROI box, as in the
    # other adapters.  Z-Rad arrays are (z, y, x) while its spacing is (x, y, z).
    _, crop_record = roi_crop(mask.array, tuple(mask.spacing)[::-1], families)
    cropped = RoiCropper(padding=crop_record["padding_voxels"]).apply(
        RoiData(image=image, morphological_mask=mask, intensity_mask=mask)
    )
    image, mask = cropped.image, cropped.morphological_mask
    # The cropper returns views; contiguous copies free the full arrays.
    image.array = np.ascontiguousarray(image.array)
    mask.array = np.ascontiguousarray(mask.array)
    roi_data = RoiData(image=image, morphological_mask=mask)
    roi_data = IntensityMaskBuilder().apply(roi_data)

    if intensity_range is not None:
        roi_data = Resegmenter(intensity_range=intensity_range).apply(roi_data)

    if discretization == "identity":
        image_values = np.asarray(image.array, dtype=float)
        mask_values = np.asarray(mask.array) > 0
        if intensity_range is not None:
            mask_values &= image_values >= float(intensity_range[0])
            mask_values &= image_values <= float(intensity_range[1])
        roi = image_values[mask_values]
        if (
            roi.size == 0
            or not np.isfinite(roi).all()
            or np.any(roi < 1.0)
            or not np.allclose(roi, np.rint(roi))
            or float(np.min(roi)) != 1.0
        ):
            raise ValueError(
                "Z-Rad identity emulation requires finite positive-integer ROI grey levels "
                "with minimum 1"
            )
        # Z-Rad 26.8 obtains the fixed-bin-size anchor from
        # ``RoiData.intensity_range``.  Record the observed full phantom range
        # without invoking range resegmentation: the mask and intensity
        # population remain unchanged, while unit-width bins map 1 -> 1, etc.
        roi_data.intensity_range = (1.0, float(np.max(roi)))

    if need_texture:
        texture_discretizer = (
            TextureDiscretizer(number_of_bins=int(bins))
            if discretization == "fbn"
            else TextureDiscretizer(
                bin_size=1.0 if discretization == "identity" else float(bin_width)
            )
        )
        roi_data = texture_discretizer.apply(roi_data)

    if need_ivh:
        ivh_discretizer = (
            IVHIntensityDiscretizer(
                "fixed_bin_number",
                number_of_bins=int(bins),
            )
            if discretization == "fbn"
            else IVHIntensityDiscretizer("direct")
            if discretization == "identity"
            else IVHIntensityDiscretizer(
                "fixed_bin_size",
                bin_size=float(bin_width),
            )
        )
        roi_data = ivh_discretizer.apply(roi_data)

    return roi_data


def _coerce_scalar(value: Any) -> Optional[float]:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Real):
        x = float(value)
        if math.isfinite(x):
            return x
        return None

    try:
        arr = np.asarray(value).reshape(-1)
    except Exception:
        return None
    if arr.size != 1:
        return None
    item = arr[0]
    if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
        return None
    x = float(item)
    if math.isfinite(x):
        return x
    return None


def _compute_zrad_features(
    *,
    roi_data,
    families: Sequence[str],
    aggr_dim: str,
    aggr_method: str,
    radiomics_class=None,
    radiomics_engine=None,
) -> tuple[List[str], Dict[str, float]]:
    """Calculate exactly the requested canonical families through Z-Rad 26.8."""

    raw = _compute_zrad_raw_features(
        roi_data=roi_data,
        families=families,
        aggr_dim=aggr_dim,
        aggr_method=aggr_method,
        radiomics_class=radiomics_class,
        radiomics_engine=radiomics_engine,
    )
    return _finalize_zrad_features(raw, families=families)


def _compute_zrad_raw_features(
    *,
    roi_data,
    families: Sequence[str],
    aggr_dim: str,
    aggr_method: str,
    radiomics_class=None,
    radiomics_engine=None,
) -> Mapping[str, Any]:
    """Invoke the native Z-Rad extractor without adapter result coercion."""

    native_families = _native_families(families)
    engine = radiomics_engine
    if engine is None:
        if radiomics_class is None:
            from zrad.radiomics import Radiomics

            radiomics_class = Radiomics
        engine = radiomics_class(aggr_dim=aggr_dim, aggr_method=aggr_method)
    try:
        raw = engine.extract_features(
            roi_data=roi_data,
            families=native_families,
            include_metadata=False,
        )
    except Exception as exc:
        raise ZRadExtractionError(
            phase="extract",
            families=families,
            message=f"{type(exc).__name__}: {exc}",
        ) from exc

    return raw


def _finalize_zrad_features(
    raw: Any, *, families: Sequence[str]
) -> tuple[List[str], Dict[str, float]]:
    if not isinstance(raw, Mapping):
        raise ZRadExtractionError(
            phase="validate_result",
            families=families,
            message=f"expected a feature dictionary, got {type(raw).__name__}",
        )
    if not raw:
        raise ZRadExtractionError(
            phase="validate_result",
            families=families,
            message="zero features returned",
        )

    names = [str(key).strip() for key in raw]
    if not all(names) or len(names) != len(set(names)):
        raise ZRadExtractionError(
            phase="validate_result",
            families=families,
            message="empty or duplicate feature names returned",
        )
    values: Dict[str, float] = {}
    invalid: List[str] = []
    for key, value in raw.items():
        scalar = _coerce_scalar(value)
        if scalar is None:
            invalid.append(str(key).strip())
        else:
            values[str(key).strip()] = scalar
    if invalid:
        raise ZRadExtractionError(
            phase="validate_result",
            families=families,
            message=(
                "non-finite or non-scalar feature values returned: "
                + ", ".join(invalid[:5])
            ),
        )
    return names, values


def _filter_by_ibsi_codes(
    names: Sequence[str],
    values: Dict[str, float],
    selected_codes: Optional[Sequence[str]],
) -> tuple[List[str], Dict[str, float]]:
    """Retain code filtering without using it as a family-selection substitute."""

    if not selected_codes:
        return list(names), dict(values)
    selected = {code.strip() for code in selected_codes if code and code.strip()}
    if not selected:
        return list(names), dict(values)

    from bench.ibsi_mapping import classify_feature

    out_names: List[str] = []
    out_values: Dict[str, float] = {}
    for name in names:
        code, status = classify_feature("zrad", name)
        if status != "mapped" or not code or code not in selected:
            continue
        out_names.append(name)
        if name in values:
            out_values[name] = values[name]
    return out_names, out_values


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="zrad-adapter")
    add_common_arguments(parser)
    parser.add_argument("--aggr-dim", choices=["2D", "2.5D", "3D"], default=None)
    parser.add_argument(
        "--aggr-method",
        choices=["MERG", "AVER", "SLICE_MERG", "DIR_MERG"],
        default=None,
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve()

    selected_families, unsupported_families = requested_families("zrad", args)
    selected_codes = parse_csv(args.include_ibsi_codes, lowercase=False)
    if selected_codes:
        from bench.ibsi_mapping import zrad_families_for_codes

        selected_families = zrad_families_for_codes(selected_codes)
        if not selected_families:
            raise ZRadExtractionError(
                phase="select_ibsi_codes",
                families=(),
                message="no supported Z-Rad family matched the requested IBSI codes",
            )

    effective_aggregation = resolve_aggregation(
        "zrad", args.aggregation, selected_families
    )
    expected_dimension, expected_method = {
        "2d_average": ("2D", "AVER"),
        "2d_slice_merge": ("2D", "MERG"),
        "2.5d_direction_merge": ("2.5D", "AVER"),
        "2.5d_merge": ("2.5D", "MERG"),
        "3d_average": ("3D", "AVER"),
        "3d_merge": ("3D", "MERG"),
    }[effective_aggregation]
    if args.aggr_dim is not None and args.aggr_dim != expected_dimension:
        raise ValueError(
            f"--aggr-dim {args.aggr_dim} conflicts with --aggregation "
            f"{args.aggregation} (effective {effective_aggregation})"
        )
    if args.aggr_method is not None and args.aggr_method != expected_method:
        raise ValueError(
            f"--aggr-method {args.aggr_method} conflicts with --aggregation "
            f"{args.aggregation} (effective {effective_aggregation})"
        )
    aggregation_method = args.aggr_method or expected_method
    aggregation_dimension = args.aggr_dim or expected_dimension

    intensity_range = parse_intensity_range(args)
    preprocessing_classes = _zrad_preprocessing_classes()
    from zrad.radiomics import Radiomics

    radiomics_engine = Radiomics(
        aggr_dim=aggregation_dimension,
        aggr_method=aggregation_method,
    )
    image, mask = _load_nifti_pair(image_path, mask_path)
    # Setup time runs from this event to the first timed or warmup call.
    write_event("inputs_loaded")
    _, roi_crop_record = roi_crop(
        mask.array, tuple(mask.spacing)[::-1], selected_families
    )
    roi_data = _prepare_roi_data(
        image=image,
        mask=mask,
        families=selected_families,
        discretization=args.discretization,
        bins=args.bins,
        bin_width=args.bin_width,
        intensity_range=intensity_range,
        preprocessing_classes=preprocessing_classes,
    )
    # The prepared ROI data hold cropped copies, so the full images go.
    del image, mask
    identity_anchor_range = None
    if args.discretization == "identity":
        raw_anchor = getattr(roi_data, "intensity_range", None)
        if not isinstance(raw_anchor, (tuple, list)) or len(raw_anchor) != 2:
            raise RuntimeError("Z-Rad identity extraction did not retain its anchor")
        identity_anchor_range = (float(raw_anchor[0]), float(raw_anchor[1]))

    def compute_selected(current_families: Sequence[str], current_roi_data=roi_data):
        return _compute_zrad_features(
            roi_data=current_roi_data,
            families=current_families,
            aggr_dim=aggregation_dimension,
            aggr_method=aggregation_method,
            radiomics_engine=radiomics_engine,
        )

    def compute_selected_raw(
        current_families: Sequence[str], current_roi_data=roi_data
    ):
        return _compute_zrad_raw_features(
            roi_data=current_roi_data,
            families=current_families,
            aggr_dim=aggregation_dimension,
            aggr_method=aggregation_method,
            radiomics_engine=radiomics_engine,
        )

    def prepare_selected():
        _clear_zrad_local_intensity_cache(selected_families)
        return roi_data

    def finalize_selected(raw, _state):
        _clear_zrad_local_intensity_cache(selected_families)
        names, values = _finalize_zrad_features(raw, families=selected_families)
        return names, values, identity_anchor_range

    timing = None
    if args.timed:

        def compute_fn(current_roi_data):
            return compute_selected_raw(selected_families, current_roi_data)

        (names, values, identity_anchor_range), timing = run_adapter_timing(
            compute_fn,
            iterations=args.iterations,
            no_warmup=args.no_warmup,
            prepare_fn=prepare_selected,
            finalize_fn=finalize_selected,
        )
    else:
        names, values = compute_selected(selected_families)

    names, values = _filter_by_ibsi_codes(names, values, selected_codes)
    if selected_codes and not names:
        raise ZRadExtractionError(
            phase="select_ibsi_codes",
            families=selected_families,
            message="zero extracted features matched the requested IBSI codes",
        )

    payload = make_payload(
        adapter="zrad",
        feature_names=names,
        values={key: values[key] for key in names if key in values}
        if args.include_values
        else None,
        timing=timing,
        requested=selected_families,
        unsupported=unsupported_families,
        metadata_payload={
            "roi_crop": roi_crop_record,
            "timing_execution_scope": "native_selected_families",
            "local_intensity_cache_policy": (
                "cleared_before_and_after_each_calculation"
                if "local_intensity" in selected_families
                else "not_applicable"
            ),
            "native_families": _native_families(selected_families),
            "preprocessing": {
                "discretization": args.discretization,
                "bins": int(args.bins),
                "bin_width": float(args.bin_width),
                "intensity_range": list(intensity_range)
                if intensity_range is not None
                else None,
                "identity_contract": (
                    "validated unit-width texture bins plus direct IVH on positive integer ROI levels"
                    if args.discretization == "identity"
                    else None
                ),
                "identity_anchor_range": (
                    list(identity_anchor_range)
                    if args.discretization == "identity"
                    else None
                ),
            },
            "aggregation": {
                "requested": args.aggregation,
                "effective_directional": effective_aggregation,
                "native_dimension": aggregation_dimension,
                "native_method": aggregation_method,
                "omnidirectional": aggregation_dimension.casefold(),
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

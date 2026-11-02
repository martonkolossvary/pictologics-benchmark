from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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


_DISCRETIZED_FAMILIES = frozenset(
    {"histogram", "ivh", "glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"}
)


def _load_nifti(path: Path) -> tuple[Any, tuple[float, float, float]]:
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if arr.ndim != 3:
        raise RuntimeError(
            f"Expected 3D NIfTI volume, got shape={arr.shape} for {path}"
        )
    zooms = img.header.get_zooms()
    spacing = (
        float(zooms[0]) if len(zooms) > 0 else 1.0,
        float(zooms[1]) if len(zooms) > 1 else 1.0,
        float(zooms[2]) if len(zooms) > 2 else 1.0,
    )
    return arr, spacing


def _prepare_roi_volumes(
    image: Any,
    mask: Any,
    *,
    discretization: str,
    bins: int,
    bin_width: float,
    intensity_range: Optional[tuple[float, float]],
    apply_discretization: bool = True,
) -> tuple[Any, Any, Any, float, Any]:
    mask_bool = np.asarray(mask > 0, dtype=bool)
    if not np.any(mask_bool):
        raise RuntimeError("Mask is empty")

    image_values = np.asarray(image, dtype=np.float32)
    intensity_mask = mask_bool
    if intensity_range is not None:
        lower, upper = intensity_range
        intensity_mask = (
            intensity_mask & (image_values >= lower) & (image_values <= upper)
        )

    vol_raw = np.where(intensity_mask, image_values, np.nan)

    valid = vol_raw[~np.isnan(vol_raw)]
    if valid.size == 0:
        raise RuntimeError("No voxels inside ROI")

    if not apply_discretization:
        # Continuous first-order families still require ROI masking and
        # resegmentation on every measured iteration, but they do not consume
        # a discretized image. Returning the raw ROI in the unused slots keeps
        # the extraction interface uniform without doing quantization work.
        return vol_raw, vol_raw, vol_raw, 1.0, intensity_mask

    minv = float(np.min(valid))
    maxv = float(np.max(valid))

    vol_quant = np.full(vol_raw.shape, np.nan, dtype=np.float32)
    if discretization == "raw":
        vol_quant[intensity_mask] = vol_raw[intensity_mask]
        vol_ivh = vol_quant.copy()
        wd = 1.0
    elif discretization == "identity":
        if (
            not np.isfinite(valid).all()
            or np.any(valid < 1.0)
            or not np.allclose(valid, np.rint(valid))
        ):
            raise ValueError(
                "MEDimage identity discretization requires finite positive-integer ROI grey levels"
            )
        vol_quant[intensity_mask] = np.rint(vol_raw[intensity_mask]).astype(np.float32)
        vol_ivh = vol_quant.copy()
        wd = 1.0
    elif discretization == "fbn":
        n_bins = max(1, int(bins))
        if maxv <= minv:
            vol_quant[intensity_mask] = 1.0
            wd = 1.0
        else:
            scaled = (
                np.floor((vol_raw[intensity_mask] - minv) / (maxv - minv) * n_bins)
                + 1.0
            )
            scaled = np.clip(scaled, 1.0, float(n_bins))
            vol_quant[intensity_mask] = scaled.astype(np.float32)
            # MEDimage's FBN IVH operates on one-based grey-level indices.
            wd = 1.0
        vol_ivh = vol_quant.copy()
    else:
        if intensity_range is None:
            raise ValueError(
                "MEDimage FBS extraction requires an explicit intensity range "
                "for the IBSI lower-bin anchor"
            )
        width = max(float(bin_width), 1e-6)
        anchor = float(intensity_range[0])
        scaled = np.floor((vol_raw[intensity_mask] - anchor) / width) + 1.0
        scaled = np.maximum(scaled, 1.0)
        vol_quant[intensity_mask] = scaled.astype(np.float32)
        wd = width
        # MEDimage's own discretisation implementation converts FBS bins to
        # physical bin centres for IVH while texture families use indices.
        vol_ivh = anchor + (vol_quant - 0.5) * width

    return vol_raw, vol_quant, vol_ivh, float(max(wd, 1e-6)), intensity_mask


def _ensure_pkg(name: str, path: Path) -> None:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__package__ = name
        mod.__path__ = [str(path)]
        sys.modules[name] = mod


def _import_module(name: str):
    return importlib.import_module(name)


@lru_cache(maxsize=1)
def _medimage_modules() -> Dict[str, Any]:
    pkg_name = None
    spec = importlib.util.find_spec("MEDiml")
    if spec is not None and spec.submodule_search_locations:
        pkg_name = "MEDiml"
    else:
        spec = importlib.util.find_spec("MEDimage")
        if spec is not None and spec.submodule_search_locations:
            pkg_name = "MEDimage"

    if pkg_name is None or spec is None or not spec.submodule_search_locations:
        raise RuntimeError("MEDimage package not found.")

    root = Path(list(spec.submodule_search_locations)[0])
    _ensure_pkg(pkg_name, root)
    _ensure_pkg(f"{pkg_name}.biomarkers", root / "biomarkers")
    _ensure_pkg(f"{pkg_name}.utils", root / "utils")
    _ensure_pkg(f"{pkg_name}.processing", root / "processing")

    return {
        "stats": _import_module(f"{pkg_name}.biomarkers.stats"),
        "intensity_histogram": _import_module(
            f"{pkg_name}.biomarkers.intensity_histogram"
        ),
        "morph": _import_module(f"{pkg_name}.biomarkers.morph"),
        "local_intensity": _import_module(f"{pkg_name}.biomarkers.local_intensity"),
        "glcm": _import_module(f"{pkg_name}.biomarkers.glcm"),
        "glrlm": _import_module(f"{pkg_name}.biomarkers.glrlm"),
        "glszm": _import_module(f"{pkg_name}.biomarkers.glszm"),
        "gldzm": _import_module(f"{pkg_name}.biomarkers.gldzm"),
        "ngtdm": _import_module(f"{pkg_name}.biomarkers.ngtdm"),
        "ngldm": _import_module(f"{pkg_name}.biomarkers.ngldm"),
        "int_vol_hist": _import_module(f"{pkg_name}.biomarkers.int_vol_hist"),
    }


def _coerce_scalar(value: Any) -> float:
    try:
        arr = np.asarray(value)
    except Exception as exc:
        raise ValueError("feature output is not a numeric scalar") from exc

    if arr.size == 0:
        raise ValueError("feature output is empty")
    if arr.size != 1:
        raise ValueError(
            f"feature output must contain exactly one value, got {arr.size}"
        )

    item = arr.reshape(-1)[0]
    if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
        raise ValueError("feature output is not a numeric scalar")
    scalar = float(item)
    if not np.isfinite(scalar):
        raise ValueError("feature output is not finite")
    return scalar


def _normalize_features(values: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, raw in values.items():
        name = str(key).strip()
        if not name or name in out:
            raise ValueError(
                f"Invalid MEDimage feature name (empty or duplicate): {name!r}"
            )
        try:
            out[name] = _coerce_scalar(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid MEDimage feature {key!r}: {exc}") from exc
    return out


def _extract_family(
    *,
    family: str,
    modules: Dict[str, Any],
    image_full: Any,
    vol_raw: Any,
    vol_quant: Any,
    vol_ivh: Any,
    morphology_mask_bool: Any,
    intensity_mask_bool: Any,
    spacing: Sequence[float],
    intensity_type: str,
    wd: float,
    discretization: str,
    intensity_range: Optional[tuple[float, float]],
    prepared_ivh_range: Optional[tuple[float, float]] = None,
) -> Dict[str, Any]:
    if family in {"morphology", "morphology_shape"}:
        spatial = family == "morphology"
        values = modules["morph"].extract_all(
            vol=vol_raw,
            mask_int=intensity_mask_bool,
            mask_morph=morphology_mask_bool,
            res=np.asarray(spacing, dtype=float),
            intensity_type=intensity_type,
            compute_moran_i=spatial,
            compute_geary_c=spatial,
        )
        if not spatial:
            # ``extract_all`` keeps empty placeholders for skipped features.
            for name in ("Fmorph_moran_i", "Fmorph_geary_c"):
                values.pop(name, None)
        return values
    if family == "morphology_spatial":
        # MEDimage's public single-feature wrappers repeat the padding and
        # ROI masking that ``extract_all`` applies before these two features.
        arguments = {
            "vol": vol_raw,
            "mask_int": intensity_mask_bool,
            "mask_morph": morphology_mask_bool,
            "res": np.asarray(spacing, dtype=float),
        }
        return {
            "Fmorph_moran_i": modules["morph"].moran_i(
                **arguments, compute_moran_i=True
            ),
            "Fmorph_geary_c": modules["morph"].geary_c(
                **arguments, compute_geary_c=True
            ),
        }

    if family == "local_intensity":
        # MEDimage's local-intensity implementation needs the continuous image
        # around the ROI as well as a separate ROI mask.  Replacing every
        # out-of-ROI voxel with zero changes the spherical-kernel means at the
        # ROI boundary and is not the package's documented API contract.
        img_obj = np.asarray(image_full, dtype=np.float32)
        values = modules["local_intensity"].extract_all(
            img_obj=img_obj,
            roi_obj=intensity_mask_bool,
            res=np.asarray(spacing, dtype=float),
            intensity_type=intensity_type,
            compute_global=True,
        )
        return values

    if family == "intensity":
        values = modules["stats"].extract_all(
            vol=vol_raw, intensity_type=intensity_type
        )
        return values

    if family == "histogram":
        values = modules["intensity_histogram"].extract_all(vol=vol_quant)
        return values

    if family == "ivh":
        if prepared_ivh_range is None:
            roi_vals = vol_raw[~np.isnan(vol_raw)]
            if roi_vals.size == 0:
                return {}
            prepared_ivh_range = (
                float(np.min(roi_vals)),
                float(np.max(roi_vals)),
            )
        if discretization == "fbs":
            if intensity_range is None:
                raise RuntimeError(
                    "MEDimage cannot honor FBS IVH without an explicit range"
                )
            ivh_cfg = {"type": "FBS"}
            im_range = [float(intensity_range[0]), float(intensity_range[1])]
            ivh_volume = vol_ivh
        else:
            ivh_cfg = {"type": "FBN"}
            im_range = [prepared_ivh_range[0], prepared_ivh_range[1]]
            ivh_volume = vol_quant
        values = modules["int_vol_hist"].extract_all(
            vol=ivh_volume,
            vol_int_re=vol_raw,
            wd=wd,
            ivh=ivh_cfg,
            im_range=im_range,
            medscan=None,
        )
        return values

    if family == "glcm":
        values = modules["glcm"].extract_all(
            vol=vol_quant, dist_correction=None, merge_method="vol_merge"
        )
        return values

    if family == "glrlm":
        values = modules["glrlm"].extract_all(
            vol=vol_quant, dist_correction=None, merge_method="vol_merge"
        )
        return values

    if family == "glszm":
        values = modules["glszm"].extract_all(vol=vol_quant)
        return values

    if family == "gldzm":
        values = modules["gldzm"].extract_all(
            vol_int=vol_quant,
            mask_morph=morphology_mask_bool,
        )
        return values

    if family == "ngtdm":
        # MEDimage treats non-bool values (including None) as distance
        # correction enabled.  Its own API documents False as the setting that
        # reproduces the IBSI NGTDM formulation.
        values = modules["ngtdm"].extract_all(vol=vol_quant, dist_correction=False)
        return values

    if family == "ngldm":
        values = modules["ngldm"].extract_all(vol=vol_quant)
        return values

    raise RuntimeError(f"Unsupported MEDimage family: {family}")


def _compute_features(
    *,
    families: Sequence[str],
    modules: Dict[str, Any],
    image_full: Any,
    vol_raw: Any,
    vol_quant: Any,
    vol_ivh: Any,
    morphology_mask_bool: Any,
    intensity_mask_bool: Any,
    spacing: Sequence[float],
    intensity_type: str,
    wd: float,
    discretization: str,
    intensity_range: Optional[tuple[float, float]],
    prepared_ivh_range: Optional[tuple[float, float]] = None,
) -> Dict[str, float]:
    return _finalize_feature_payloads(
        _compute_feature_payloads(
            families=families,
            modules=modules,
            image_full=image_full,
            vol_raw=vol_raw,
            vol_quant=vol_quant,
            vol_ivh=vol_ivh,
            morphology_mask_bool=morphology_mask_bool,
            intensity_mask_bool=intensity_mask_bool,
            spacing=spacing,
            intensity_type=intensity_type,
            wd=wd,
            discretization=discretization,
            intensity_range=intensity_range,
            prepared_ivh_range=prepared_ivh_range,
        )
    )


def _compute_feature_payloads(
    *,
    families: Sequence[str],
    modules: Dict[str, Any],
    image_full: Any,
    vol_raw: Any,
    vol_quant: Any,
    vol_ivh: Any,
    morphology_mask_bool: Any,
    intensity_mask_bool: Any,
    spacing: Sequence[float],
    intensity_type: str,
    wd: float,
    discretization: str,
    intensity_range: Optional[tuple[float, float]],
    prepared_ivh_range: Optional[tuple[float, float]] = None,
) -> List[tuple[str, Mapping[str, Any]]]:
    payloads: List[tuple[str, Mapping[str, Any]]] = []

    for family in families:
        try:
            fam_values = _extract_family(
                family=family,
                modules=modules,
                image_full=image_full,
                vol_raw=vol_raw,
                vol_quant=vol_quant,
                vol_ivh=vol_ivh,
                morphology_mask_bool=morphology_mask_bool,
                intensity_mask_bool=intensity_mask_bool,
                spacing=spacing,
                intensity_type=intensity_type,
                wd=wd,
                discretization=discretization,
                intensity_range=intensity_range,
                prepared_ivh_range=prepared_ivh_range,
            )
            if not fam_values:
                raise RuntimeError("returned zero finite features")
        except Exception as exc:
            raise RuntimeError(
                f"MEDimage failed supported feature family {family}: {exc}"
            ) from exc
        payloads.append((family, fam_values))
    return payloads


def _finalize_feature_payloads(
    payloads: Sequence[tuple[str, Mapping[str, Any]]], _state=None
) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for family, raw_values in payloads:
        fam_values = _normalize_features(dict(raw_values))
        duplicated = sorted(set(values).intersection(fam_values))
        if duplicated:
            raise RuntimeError(
                "MEDimage returned duplicate feature names across families "
                f"after {family}: " + ", ".join(duplicated[:5])
            )
        values.update(fam_values)
    return values


def _filter_by_ibsi_codes(
    names: Iterable[str],
    values: Dict[str, float],
    selected_codes: Sequence[str],
) -> tuple[List[str], Dict[str, float]]:
    selected = {c.strip() for c in selected_codes if c and c.strip()}
    if not selected:
        return list(names), dict(values)

    from bench.ibsi_mapping import classify_feature

    out_names: List[str] = []
    out_values: Dict[str, float] = {}
    for name in names:
        code, status = classify_feature("medimage", name)
        if status != "mapped" or not code:
            continue
        if code not in selected:
            continue
        out_names.append(name)
        if name in values:
            out_values[name] = values[name]
    return out_names, out_values


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="medimage-adapter")
    add_common_arguments(parser)
    parser.add_argument(
        "--intensity-type",
        default="definite",
        choices=["definite", "arbitrary", "filtered"],
    )
    args = parser.parse_args(argv)
    intensity_range = parse_intensity_range(args)

    image_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve()

    families, unsupported_families = requested_families("medimage", args)
    ibsi_codes = parse_csv(args.include_ibsi_codes, lowercase=False)
    if ibsi_codes:
        from bench.ibsi_mapping import medimage_families_for_codes

        families = medimage_families_for_codes(ibsi_codes)
    if args.discretization == "raw" and set(families).intersection(
        _DISCRETIZED_FAMILIES
    ):
        raise ValueError(
            "Raw extraction is valid only for non-discretized feature families"
        )
    effective_aggregation = resolve_aggregation("medimage", args.aggregation, families)

    modules = _medimage_modules()
    image, spacing = _load_nifti(image_path)
    # Only the boolean mask is kept, so the full float mask is freed.
    mask_bool = np.asarray(_load_nifti(mask_path)[0] > 0, dtype=bool)
    # Setup time runs from this event to the first timed or warmup call.
    write_event("inputs_loaded")
    # MEDimage's own workflow crops to an ROI box; its low-level family
    # modules otherwise scan every background voxel of the full image.
    crop_slices, roi_crop_record = roi_crop(mask_bool, spacing, families)
    image = np.ascontiguousarray(image[crop_slices])
    mask_bool = np.ascontiguousarray(mask_bool[crop_slices])
    if (
        args.discretization == "fbs"
        and set(families).intersection(_DISCRETIZED_FAMILIES)
        and intensity_range is None
    ):
        raise ValueError(
            "MEDimage FBS extraction is unsupported without an explicit intensity range; "
            "refusing to use a per-ROI implicit anchor"
        )

    apply_discretization = bool(set(families).intersection(_DISCRETIZED_FAMILIES))
    vol_raw, vol_quant, vol_ivh, wd, intensity_mask_bool = _prepare_roi_volumes(
        image,
        mask_bool,
        discretization=args.discretization,
        bins=args.bins,
        bin_width=args.bin_width,
        intensity_range=intensity_range,
        apply_discretization=apply_discretization,
    )
    morphology_mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
    intensity_mask_u8 = np.asarray(intensity_mask_bool, dtype=np.uint8)
    spacing_array = np.asarray(spacing, dtype=float)
    if "local_intensity" in families:
        image_float32 = np.asarray(image, dtype=np.float32)
        if np.isnan(image_float32).any():
            raise ValueError(
                "MEDimage local intensity requires a full image without NaNs"
            )
    valid_raw = vol_raw[~np.isnan(vol_raw)]
    if valid_raw.size == 0:
        raise RuntimeError("MEDimage prepared ROI contains no finite voxels")
    prepared_ivh_range = (float(np.min(valid_raw)), float(np.max(valid_raw)))

    def compute_selected(current_families: Sequence[str]) -> Dict[str, float]:
        return _compute_features(
            families=current_families,
            modules=modules,
            image_full=image,
            vol_raw=vol_raw,
            vol_quant=vol_quant,
            vol_ivh=vol_ivh,
            morphology_mask_bool=morphology_mask_u8,
            intensity_mask_bool=intensity_mask_u8,
            spacing=spacing_array,
            intensity_type=args.intensity_type,
            wd=wd,
            discretization=args.discretization,
            intensity_range=intensity_range,
            prepared_ivh_range=prepared_ivh_range,
        )

    def compute_selected_raw(
        current_families: Sequence[str],
    ) -> List[tuple[str, Mapping[str, Any]]]:
        return _compute_feature_payloads(
            families=current_families,
            modules=modules,
            image_full=image,
            vol_raw=vol_raw,
            vol_quant=vol_quant,
            vol_ivh=vol_ivh,
            morphology_mask_bool=morphology_mask_u8,
            intensity_mask_bool=intensity_mask_u8,
            spacing=spacing_array,
            intensity_type=args.intensity_type,
            wd=wd,
            discretization=args.discretization,
            intensity_range=intensity_range,
            prepared_ivh_range=prepared_ivh_range,
        )

    timing = None
    if args.timed:

        def compute_fn():
            return compute_selected_raw(families)

        values, timing = run_adapter_timing(
            compute_fn,
            iterations=args.iterations,
            no_warmup=args.no_warmup,
            finalize_fn=_finalize_feature_payloads,
        )
    else:
        values = compute_selected(families)

    names = list(values.keys())
    if ibsi_codes:
        names, values = _filter_by_ibsi_codes(names, values, ibsi_codes)
        if not names:
            raise RuntimeError(
                "MEDimage returned zero features matching requested IBSI codes"
            )

    payload = make_payload(
        adapter="medimage",
        feature_names=names,
        values=values if args.include_values else None,
        timing=timing,
        requested=families,
        unsupported=unsupported_families,
        metadata_payload={
            "roi_crop": roi_crop_record,
            "preprocessing": {
                "discretization": args.discretization,
                "bins": int(args.bins),
                "bin_width": float(args.bin_width),
                "intensity_type": args.intensity_type,
                "intensity_range": list(intensity_range)
                if intensity_range is not None
                else None,
                "fbs_anchor": float(intensity_range[0])
                if args.discretization == "fbs" and intensity_range is not None
                else None,
                "identity_contract": (
                    "native positive-integer texture levels; IVH evaluated on the same unit grid"
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

"""Strict package-native IBSI 2 response-map generation.

This module is deliberately small enough to be imported inside each adapter's
isolated environment.  It performs NIfTI I/O around the *installed package's*
public filtering implementation; it does not provide fallback convolutions.

The normalized parameter vocabulary is shared by all adapters.  A backend must
raise :class:`UnsupportedNativeFilter` when it cannot express every requested
operation exactly.  In particular, accepting a filter family is not sufficient
when the installed API cannot select the requested dimensionality, boundary
condition, rotation pooling, or steering mode.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import importlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


FILTERS = frozenset(
    {
        "none",
        "mean",
        "log",
        "laws",
        "gabor",
        "wavelet",
        "simoncelli",
        "riesz_log",
        "riesz_simoncelli",
    }
)
ADAPTERS = frozenset({"pictologics", "pyradiomics", "mirp", "medimage", "zrad"})
BOUNDARIES = frozenset({"zero", "nearest", "periodic", "mirror"})
POOLING_METHODS = frozenset({"max", "average", "min", "sum"})

_COMMON_KEYS = frozenset({"filter", "dimensionality", "boundary"})
_FILTER_KEYS = {
    "none": frozenset(),
    "mean": frozenset({"support"}),
    "log": frozenset({"sigma_mm", "truncate"}),
    "laws": frozenset(
        {
            "kernels",
            "rotation_invariant",
            "pooling",
            "compute_energy",
            "energy_distance",
        }
    ),
    "gabor": frozenset(
        {
            "sigma_mm",
            "lambda_mm",
            "gamma",
            "theta",
            "rotation_invariant",
            "delta_theta",
            "pooling",
            "average_over_planes",
        }
    ),
    "wavelet": frozenset(
        {"wavelet", "level", "decomposition", "rotation_invariant", "pooling"}
    ),
    "simoncelli": frozenset({"level"}),
    "riesz_log": frozenset({"sigma_mm", "truncate", "order", "tensor_sigma"}),
    "riesz_simoncelli": frozenset({"level", "order", "tensor_sigma"}),
}

_DISTRIBUTIONS = {
    "pictologics": "pictologics",
    "pyradiomics": "pyradiomics",
    "mirp": "mirp",
    "medimage": "medimage-pkg",
    "zrad": "z-rad",
}

_IMPLEMENTATION_SELECTED_BOUNDARIES = {
    # FFT-native filters are executed periodically when Phase 2 permits the
    # implementation to select padding. Other filters use mirror padding.
    "pictologics": {
        "simoncelli": "periodic",
        "riesz_simoncelli": "periodic",
    },
    "mirp": {
        "simoncelli": "periodic",
        "riesz_simoncelli": "periodic",
    },
    # PyRadiomics' stationary-wavelet API is intrinsically periodic.
    "pyradiomics": {"wavelet": "periodic"},
}

_MEDIMAGE_PACKAGE_INFO: tuple[str, Path] | None = None


class UnsupportedNativeFilter(RuntimeError):
    """Requested configuration is not exactly expressible by an adapter API."""

    def __init__(
        self,
        adapter: str,
        filter_name: str,
        reason: str,
        *,
        evidence: Sequence[str] | None = None,
    ) -> None:
        self.adapter = adapter
        self.filter_name = filter_name
        self.reason = reason
        self.evidence = tuple(
            evidence
            or (
                "Every requested parameter must be selectable through the installed "
                "package's public native API.",
                "No substitute convolution or boundary/pooling approximation was used.",
            )
        )
        super().__init__(f"{adapter} cannot natively express {filter_name!r}: {reason}")


class UncontrolledNativeParameter(UnsupportedNativeFilter):
    """The package has the filter, but its API cannot set a boundary or truncation.

    An exact run rejects the configuration.  A native-default run executes it
    with the package's own setting for these controls, and the reports keep
    that result in a separate class.  Pooling, rotation, steering, dimension
    and voxel-size limits change the operation itself, so they never qualify.
    """

    def __init__(
        self,
        adapter: str,
        filter_name: str,
        reason: str,
        *,
        uncontrolled: Sequence[str],
        effective_boundary: str,
    ) -> None:
        super().__init__(adapter, filter_name, reason)
        self.uncontrolled = tuple(uncontrolled)
        self.effective_boundary = effective_boundary


def _unsupported(adapter: str, params: Mapping[str, Any], reason: str) -> None:
    raise UnsupportedNativeFilter(adapter, str(params.get("filter", "unknown")), reason)


def _uncontrolled(
    adapter: str,
    params: Mapping[str, Any],
    reason: str,
    *,
    uncontrolled: Sequence[str],
    effective_boundary: str,
) -> dict[str, Any]:
    """Reject an exact run; in a native-default run, record what the package used."""

    if not params.get("native_default"):
        raise UncontrolledNativeParameter(
            adapter,
            str(params.get("filter", "unknown")),
            reason,
            uncontrolled=uncontrolled,
            effective_boundary=effective_boundary,
        )
    return {
        "uncontrolled": list(uncontrolled),
        "effective_boundary": effective_boundary,
        "reason": reason,
    }


def _require_bool(params: Mapping[str, Any], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a JSON boolean")
    return value


def _require_number(
    params: Mapping[str, Any],
    key: str,
    *,
    default: float | None = None,
    positive: bool = False,
) -> float:
    if key not in params:
        if default is None:
            raise ValueError(f"Missing required filter parameter: {key}")
        value: Any = default
    else:
        value = params[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{key} must be greater than zero")
    return result


def _require_int(
    params: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int = 0,
) -> int:
    if key not in params:
        if default is None:
            raise ValueError(f"Missing required filter parameter: {key}")
        value: Any = default
    else:
        value = params[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def _parse_laws_kernels(value: Any, dimensionality: int) -> str:
    if not isinstance(value, str):
        raise TypeError("kernels must be a Laws kernel string such as L5E5E5")
    kernels = value.upper()
    tokens = re.findall(r"[LESWR][35]", kernels)
    if "".join(tokens) != kernels or len(tokens) != dimensionality:
        raise ValueError(
            f"kernels must contain exactly {dimensionality} valid Laws components"
        )
    valid = {"L3", "E3", "S3", "L5", "E5", "S5", "W5", "R5"}
    if any(token not in valid for token in tokens):
        raise ValueError(f"Unsupported Laws kernel component in {kernels!r}")
    return kernels


def _parse_order(value: Any, dimensionality: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise TypeError("order must be a JSON list of non-negative integers")
    if len(value) != dimensionality:
        raise ValueError(f"order must contain {dimensionality} integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError("order must contain only integers")
    order = tuple(int(item) for item in value)
    if any(item < 0 for item in order) or sum(order) == 0:
        raise ValueError(
            "order values must be non-negative and have positive total order"
        )
    return order


def _implementation_selected_boundary(adapter: str | None, filter_name: str) -> str:
    if adapter is not None:
        selected = _IMPLEMENTATION_SELECTED_BOUNDARIES.get(adapter, {}).get(filter_name)
        if selected is not None:
            return selected
    if filter_name in {"simoncelli", "riesz_simoncelli"}:
        return "periodic"
    return "mirror"


def normalize_parameters(
    parameters: Mapping[str, Any],
    *,
    adapter: str | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize a JSON-compatible native-filter mapping."""

    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a JSON-compatible mapping")
    if not isinstance(parameters.get("filter"), str):
        raise ValueError("parameters.filter is required")
    filter_name = str(parameters["filter"]).lower()
    if filter_name not in FILTERS:
        raise ValueError(
            f"Unknown filter {filter_name!r}; expected one of {sorted(FILTERS)}"
        )

    dimensionality = parameters.get("dimensionality")
    if isinstance(dimensionality, bool) or dimensionality not in (2, 3):
        raise ValueError("dimensionality must be integer 2 or 3")
    dimensionality = int(dimensionality)

    allowed = _COMMON_KEYS | _FILTER_KEYS[filter_name]
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(
            f"Parameters not applicable to {filter_name!r}: {', '.join(unknown)}"
        )

    normalized_adapter = str(adapter).lower() if adapter is not None else None
    if normalized_adapter is not None and normalized_adapter not in ADAPTERS:
        raise ValueError(
            f"Unknown adapter {adapter!r}; expected one of {sorted(ADAPTERS)}"
        )
    result: dict[str, Any] = {
        "filter": filter_name,
        "dimensionality": dimensionality,
    }

    if filter_name == "none":
        return result

    boundary_default = _implementation_selected_boundary(
        normalized_adapter, filter_name
    )
    boundary = parameters.get("boundary", boundary_default)
    if not isinstance(boundary, str) or boundary.lower() not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {sorted(BOUNDARIES)}")
    result["boundary"] = boundary.lower()

    if filter_name == "mean":
        support = _require_int(parameters, "support", minimum=1)
        if support % 2 == 0:
            raise ValueError("support must be odd for an IBSI mean filter")
        result["support"] = support

    elif filter_name in {"log", "riesz_log"}:
        result["sigma_mm"] = _require_number(parameters, "sigma_mm", positive=True)
        result["truncate"] = _require_number(
            parameters, "truncate", default=4.0, positive=True
        )
        if filter_name == "riesz_log":
            result["order"] = list(
                _parse_order(parameters.get("order"), dimensionality)
            )

    elif filter_name == "laws":
        result["kernels"] = _parse_laws_kernels(
            parameters.get("kernels"), dimensionality
        )
        result["rotation_invariant"] = _require_bool(
            parameters, "rotation_invariant", False
        )
        pooling = parameters.get("pooling", "max")
        if not isinstance(pooling, str) or pooling.lower() not in POOLING_METHODS:
            raise ValueError(f"pooling must be one of {sorted(POOLING_METHODS)}")
        result["pooling"] = pooling.lower()
        result["compute_energy"] = _require_bool(parameters, "compute_energy", False)
        if result["compute_energy"]:
            result["energy_distance"] = _require_int(
                parameters, "energy_distance", minimum=0
            )
        elif "energy_distance" in parameters:
            raise ValueError(
                "energy_distance is only valid when compute_energy is true"
            )

    elif filter_name == "gabor":
        result["sigma_mm"] = _require_number(parameters, "sigma_mm", positive=True)
        result["lambda_mm"] = _require_number(parameters, "lambda_mm", positive=True)
        result["gamma"] = _require_number(
            parameters, "gamma", default=1.0, positive=True
        )
        result["theta"] = _require_number(parameters, "theta", default=0.0)
        result["rotation_invariant"] = _require_bool(
            parameters, "rotation_invariant", False
        )
        pooling = parameters.get("pooling", "average")
        if not isinstance(pooling, str) or pooling.lower() not in POOLING_METHODS:
            raise ValueError(f"pooling must be one of {sorted(POOLING_METHODS)}")
        result["pooling"] = pooling.lower()
        result["average_over_planes"] = _require_bool(
            parameters, "average_over_planes", False
        )
        if result["rotation_invariant"]:
            delta = _require_number(parameters, "delta_theta", positive=True)
            rotations = 2.0 * math.pi / delta
            if not math.isclose(rotations, round(rotations), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "delta_theta must divide 2*pi into an integer number of steps"
                )
            result["delta_theta"] = delta
        elif "delta_theta" in parameters:
            raise ValueError("delta_theta requires rotation_invariant=true")

    elif filter_name == "wavelet":
        wavelet = parameters.get("wavelet")
        if not isinstance(wavelet, str) or not wavelet:
            raise ValueError("wavelet must be a non-empty PyWavelets family name")
        result["wavelet"] = wavelet.lower()
        result["level"] = _require_int(parameters, "level", default=1, minimum=1)
        decomposition = parameters.get("decomposition")
        if not isinstance(decomposition, str):
            raise TypeError("decomposition must be a low/high response-map string")
        decomposition = decomposition.upper()
        if len(decomposition) != dimensionality or set(decomposition) - {"L", "H"}:
            raise ValueError(
                f"decomposition must contain {dimensionality} letters drawn from L and H"
            )
        result["decomposition"] = decomposition
        result["rotation_invariant"] = _require_bool(
            parameters, "rotation_invariant", False
        )
        pooling = parameters.get("pooling", "average")
        if not isinstance(pooling, str) or pooling.lower() not in POOLING_METHODS:
            raise ValueError(f"pooling must be one of {sorted(POOLING_METHODS)}")
        result["pooling"] = pooling.lower()

    elif filter_name in {"simoncelli", "riesz_simoncelli"}:
        result["level"] = _require_int(parameters, "level", default=1, minimum=1)
        if filter_name == "riesz_simoncelli":
            result["order"] = list(
                _parse_order(parameters.get("order"), dimensionality)
            )

    if filter_name in {"riesz_log", "riesz_simoncelli"}:
        tensor_sigma = parameters.get("tensor_sigma")
        if tensor_sigma is not None:
            result["tensor_sigma"] = _require_number(
                parameters, "tensor_sigma", positive=True
            )
        else:
            result["tensor_sigma"] = None

    return result


def _validate_paths(
    input_path: str | Path, output_path: str | Path
) -> tuple[Path, Path]:
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input NIfTI does not exist: {source}")
    if source == destination:
        raise ValueError("input_path and output_path must be different")
    lower_name = destination.name.lower()
    if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
        raise ValueError("output_path must end in .nii or .nii.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return source, destination


def _package_version(adapter: str) -> str:
    try:
        return importlib.metadata.version(_DISTRIBUTIONS[adapter])
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _array_metadata(array: Any) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(array)
    return {
        "shape": [int(item) for item in values.shape],
        "dtype": str(values.dtype),
        "finite": bool(np.isfinite(values).all()),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _copy_identity(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copyfile(source, destination)
    return {"identity_copy": True}


def _load_nibabel(source: Path) -> tuple[Any, Any, tuple[float, float, float]]:
    import nibabel as nib
    import numpy as np

    image = nib.load(str(source))
    array = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a three-dimensional NIfTI, got {array.shape}")
    zooms = image.header.get_zooms()
    spacing = tuple(float(value) for value in zooms[:3])
    if len(spacing) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in spacing
    ):
        raise ValueError(f"Invalid NIfTI spacing: {spacing}")
    return image, array, spacing


def _save_nibabel(reference: Any, response: Any, destination: Path) -> None:
    import nibabel as nib
    import numpy as np

    output = np.asarray(response, dtype=np.float32)
    if output.shape != reference.shape:
        raise RuntimeError(
            f"Native filter changed image shape: {output.shape} versus {reference.shape}"
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Native filter produced non-finite response values")

    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    result = nib.Nifti1Image(output, reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        result.set_qform(qform, int(qcode or 0))
    if sform is not None:
        result.set_sform(sform, int(scode or 0))
    nib.save(result, str(destination))


def _is_isotropic(spacing: Sequence[float], axes: Sequence[int] = (0, 1, 2)) -> bool:
    values = [float(spacing[index]) for index in axes]
    return all(
        math.isclose(values[0], value, rel_tol=1e-6, abs_tol=1e-8)
        for value in values[1:]
    )


def _validate_pictologics_static(params: Mapping[str, Any]) -> None:
    name = str(params["filter"])
    dimensionality = int(params["dimensionality"])
    if name == "mean" and dimensionality != 3:
        _unsupported(
            "pictologics", params, "the documented mean-filter API is three-dimensional"
        )
    if name == "log" and dimensionality != 3:
        _unsupported(
            "pictologics", params, "the public LoG function is three-dimensional"
        )
    if name == "laws" and dimensionality != 3:
        _unsupported(
            "pictologics", params, "the public Laws function requires three kernels"
        )
    if name == "gabor" and dimensionality != 2:
        _unsupported("pictologics", params, "Gabor uses a two-dimensional kernel")
    if name == "wavelet" and dimensionality != 3:
        _unsupported(
            "pictologics",
            params,
            "the public separable-wavelet API is three-dimensional",
        )
    if name in {"simoncelli", "riesz_log", "riesz_simoncelli"} and dimensionality != 3:
        _unsupported(
            "pictologics", params, f"the documented {name} API is three-dimensional"
        )
    if name in {"laws", "gabor", "wavelet"} and params.get("pooling") == "sum":
        _unsupported(
            "pictologics",
            params,
            "the public pooling API supports max, average, and min, but not sum",
        )


def _pictologics_backend(
    source: Path, destination: Path, params: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_pictologics_static(params)

    from pictologics import filters

    reference, image, spacing = _load_nibabel(source)
    name = str(params["filter"])
    boundary = str(params["boundary"]).upper()

    if name == "mean":
        response = filters.mean_filter(
            image, support=int(params["support"]), boundary=boundary
        )
    elif name == "log":
        response = filters.laplacian_of_gaussian(
            image,
            sigma_mm=float(params["sigma_mm"]),
            spacing_mm=spacing,
            truncate=float(params["truncate"]),
            boundary=boundary,
        )
    elif name == "laws":
        response = filters.laws_filter(
            image,
            kernels=str(params["kernels"]),
            boundary=boundary,
            rotation_invariant=bool(params["rotation_invariant"]),
            pooling=str(params["pooling"]),
            compute_energy=bool(params["compute_energy"]),
            energy_distance=int(params.get("energy_distance", 7)),
        )
    elif name == "gabor":
        response = filters.gabor_filter(
            image,
            sigma_mm=float(params["sigma_mm"]),
            lambda_mm=float(params["lambda_mm"]),
            gamma=float(params["gamma"]),
            theta=float(params["theta"]),
            spacing_mm=spacing,
            boundary=boundary,
            rotation_invariant=bool(params["rotation_invariant"]),
            delta_theta=params.get("delta_theta"),
            pooling=str(params["pooling"]),
            average_over_planes=bool(params["average_over_planes"]),
        )
    elif name == "wavelet":
        response = filters.wavelet_transform(
            image,
            wavelet=str(params["wavelet"]),
            level=int(params["level"]),
            decomposition=str(params["decomposition"]),
            boundary=boundary,
            rotation_invariant=bool(params["rotation_invariant"]),
            pooling=str(params["pooling"]),
        )
    elif name == "simoncelli":
        response = filters.simoncelli_wavelet(
            image,
            level=int(params["level"]),
            boundary=boundary,
        )
    elif name == "riesz_log":
        if params["tensor_sigma"] is not None:
            _unsupported(
                "pictologics", params, "the API does not implement Riesz steering"
            )
        response = filters.riesz_log(
            image,
            sigma_mm=float(params["sigma_mm"]),
            spacing_mm=spacing,
            order=tuple(params["order"]),
            truncate=float(params["truncate"]),
            boundary=boundary,
        )
    elif name == "riesz_simoncelli":
        if params["tensor_sigma"] is not None:
            _unsupported(
                "pictologics", params, "the API does not implement Riesz steering"
            )
        response = filters.riesz_simoncelli(
            image,
            level=int(params["level"]),
            order=tuple(params["order"]),
            boundary=boundary,
        )
    else:
        _unsupported(
            "pictologics", params, "filter family is absent from the public API"
        )

    _save_nibabel(reference, response, destination)
    capability = filters.get_filter_capabilities(name)
    boundary_implementation = capability.effective_boundary
    if (
        boundary_implementation == "as_specified_via_padding"
        and params["boundary"] == "periodic"
        and name in {"simoncelli", "riesz_simoncelli"}
    ):
        boundary_implementation = "native_periodic_fft"
    return {
        **_array_metadata(response),
        "geometry_preserved": True,
        "native_capability": {
            "schema_version": filters.CAPABILITIES_SCHEMA_VERSION,
            "filter": name,
            **asdict(capability),
        },
        "boundary_implementation": boundary_implementation,
    }


def _validate_pyradiomics_static(params: Mapping[str, Any]) -> dict[str, Any] | None:
    if params["filter"] == "log" and int(params["dimensionality"]) == 3:
        return _uncontrolled(
            "pyradiomics",
            params,
            "PyRadiomics LoG uses the SimpleITK recursive Gaussian, which has no "
            "truncation or boundary control",
            uncontrolled=("truncate", "boundary"),
            effective_boundary="simpleitk_recursive_gaussian",
        )
    if params["filter"] != "wavelet":
        _unsupported(
            "pyradiomics",
            params,
            "PyRadiomics exposes only the 3D LoG and the stationary-wavelet filters",
        )
    if params["rotation_invariant"]:
        _unsupported(
            "pyradiomics", params, "getWaveletImage exposes no rotation pooling"
        )
    if params["boundary"] != "periodic":
        return _uncontrolled(
            "pyradiomics",
            params,
            "getWaveletImage uses periodic padding and exposes no boundary option",
            uncontrolled=("boundary",),
            effective_boundary="periodic",
        )
    return None


def _pyradiomics_backend(
    source: Path, destination: Path, params: Mapping[str, Any]
) -> dict[str, Any]:
    native_default = _validate_pyradiomics_static(params)

    import numpy as np
    import SimpleITK as sitk
    from radiomics import imageoperations

    image = sitk.ReadImage(str(source))
    if params["filter"] == "log":
        candidates = list(
            imageoperations.getLoGImage(image, None, sigma=[float(params["sigma_mm"])])
        )
        if len(candidates) != 1:
            _unsupported(
                "pyradiomics", params, "getLoGImage emitted no image for this sigma"
            )
        response_image = candidates[0][0]
        response_image.CopyInformation(image)
        response = np.asarray(sitk.GetArrayFromImage(response_image))
        if not np.isfinite(response).all():
            raise RuntimeError("PyRadiomics produced an invalid response-map grid")
        sitk.WriteImage(response_image, str(destination))
        return {
            **_array_metadata(response),
            "geometry_preserved": True,
            "native_default": native_default,
        }
    level = int(params["level"])
    decomposition = str(params["decomposition"])
    force_2d = int(params["dimensionality"]) == 2
    candidates = list(
        imageoperations.getWaveletImage(
            image,
            None,
            wavelet=str(params["wavelet"]),
            level=level,
            force2D=force_2d,
            force2Ddimension=0,
        )
    )
    expected_name = (
        f"wavelet-{decomposition}" if level == 1 else f"wavelet{level}-{decomposition}"
    )
    matches = [candidate for candidate, name, _ in candidates if name == expected_name]
    if len(matches) != 1:
        _unsupported(
            "pyradiomics",
            params,
            f"requested decomposition {expected_name!r} was not emitted by getWaveletImage",
        )
    response_image = matches[0]
    response_image.CopyInformation(image)
    response = np.asarray(sitk.GetArrayFromImage(response_image))
    if (
        response.shape != tuple(reversed(image.GetSize()))
        or not np.isfinite(response).all()
    ):
        raise RuntimeError("PyRadiomics produced an invalid response-map grid")
    sitk.WriteImage(response_image, str(destination))
    result = {**_array_metadata(response), "geometry_preserved": True}
    if native_default is not None:
        result["native_default"] = native_default
    return result


def _mirp_kwargs(params: Mapping[str, Any]) -> dict[str, Any]:
    name = str(params["filter"])
    dimensionality = int(params["dimensionality"])
    if name in {"riesz_log", "riesz_simoncelli"} and dimensionality != 3:
        _unsupported(
            "mirp",
            params,
            "MIRP constructs a three-axis Riesz frequency grid and "
            "cannot execute its public Riesz filters slice-wise",
        )
    boundary = {
        "zero": "constant",
        "nearest": "nearest",
        "periodic": "wrap",
        "mirror": "reflect",
    }[str(params["boundary"])]
    pooling = {"average": "mean"}.get(str(params.get("pooling")), params.get("pooling"))
    common: dict[str, Any] = {
        "by_slice": dimensionality == 2,
        "response_map_feature_families": "none",
    }

    if name == "mean":
        return {
            **common,
            "filter_kernels": "mean",
            "mean_filter_kernel_size": int(params["support"]),
            "mean_filter_boundary_condition": boundary,
        }
    if name in {"log", "riesz_log"}:
        if name == "riesz_log":
            steered = params["tensor_sigma"] is not None
            filter_name = "riesz_steered_log" if steered else "riesz_log"
        else:
            steered = False
            filter_name = "laplacian_of_gaussian"
        result = {
            **common,
            "filter_kernels": filter_name,
            "laplacian_of_gaussian_sigma": float(params["sigma_mm"]),
            "laplacian_of_gaussian_kernel_truncate": float(params["truncate"]),
            "laplacian_of_gaussian_pooling_method": "none",
            "laplacian_of_gaussian_boundary_condition": boundary,
        }
        if name == "riesz_log":
            # MIRP stores voxel grids in z-y-x order; the normalized IBSI
            # contract states Riesz orders in image x-y-z order.
            result["riesz_filter_order"] = list(reversed(params["order"]))
            # MIRP validates ``riesz_filter_tensor_sigma`` for every
            # Riesz filter, although LaplacianOfGaussianFilter reads it only
            # when the selected kernel name is explicitly steered. Supplying
            # this ignored positive sentinel keeps the public extraction path
            # usable without enabling or approximating steering.
            result["riesz_filter_tensor_sigma"] = (
                float(params["tensor_sigma"]) if steered else 1.0
            )
        if steered:
            result["ibsi_compliant"] = False
        return result
    if name == "laws":
        return {
            **common,
            "filter_kernels": "laws",
            "laws_kernel": str(params["kernels"]).lower(),
            "laws_compute_energy": bool(params["compute_energy"]),
            "laws_delta": int(params.get("energy_distance", 7)),
            "laws_rotation_invariance": bool(params["rotation_invariant"]),
            "laws_pooling_method": pooling,
            "laws_boundary_condition": boundary,
        }
    if name == "gabor":
        if dimensionality != 2:
            _unsupported("mirp", params, "Gabor uses a two-dimensional kernel")
        if params["average_over_planes"] and not params["rotation_invariant"]:
            _unsupported(
                "mirp",
                params,
                "orthogonal-plane pooling is only activated with rotation invariance",
            )
        theta_degrees = math.degrees(float(params["theta"]))
        delta = params.get("delta_theta")
        return {
            **common,
            # by_slice=False activates all three Gabor stack axes while retaining a 2D kernel.
            "by_slice": not bool(params["average_over_planes"]),
            "filter_kernels": "gabor",
            "gabor_sigma": float(params["sigma_mm"]),
            "gabor_lambda": float(params["lambda_mm"]),
            "gabor_gamma": float(params["gamma"]),
            "gabor_theta": theta_degrees,
            "gabor_theta_step": math.degrees(float(delta))
            if delta is not None
            else None,
            "gabor_response": "modulus",
            "gabor_rotation_invariance": bool(params["rotation_invariant"]),
            "gabor_pooling_method": pooling,
            "gabor_boundary_condition": boundary,
        }
    if name == "wavelet":
        return {
            **common,
            "filter_kernels": "separable_wavelet",
            "separable_wavelet_families": str(params["wavelet"]),
            "separable_wavelet_set": str(params["decomposition"]).lower(),
            "separable_wavelet_stationary": True,
            "separable_wavelet_decomposition_level": int(params["level"]),
            "separable_wavelet_rotation_invariance": bool(params["rotation_invariant"]),
            "separable_wavelet_pooling_method": pooling,
            "separable_wavelet_boundary_condition": boundary,
        }
    if name in {"simoncelli", "riesz_simoncelli"}:
        native_default = None
        if params["boundary"] != "periodic":
            native_default = _uncontrolled(
                "mirp",
                params,
                "the nonseparable FFT sets pad_image=False, so boundary is periodic",
                uncontrolled=("boundary",),
                effective_boundary="periodic",
            )
        if name == "riesz_simoncelli":
            steered = params["tensor_sigma"] is not None
            filter_name = (
                "riesz_steered_nonseparable_wavelet"
                if steered
                else "riesz_nonseparable_wavelet"
            )
        else:
            steered = False
            filter_name = "nonseparable_wavelet"
        result = {
            **common,
            "filter_kernels": filter_name,
            "nonseparable_wavelet_families": "simoncelli",
            "nonseparable_wavelet_decomposition_level": int(params["level"]),
            "nonseparable_wavelet_response": "real",
            "nonseparable_wavelet_boundary_condition": "wrap",
        }
        if name == "riesz_simoncelli":
            result["riesz_filter_order"] = list(reversed(params["order"]))
            result["riesz_filter_tensor_sigma"] = (
                float(params["tensor_sigma"]) if steered else 1.0
            )
        if steered:
            result["ibsi_compliant"] = False
        if native_default is not None:
            # Not a MIRP argument: the backend removes it before the call.
            result["_native_default"] = native_default
        return result
    _unsupported(
        "mirp", params, "filter family is absent from the public extraction API"
    )
    raise AssertionError("unreachable")


def _write_mirp_image(image: Any, destination: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix="ibsi2-native-", dir=str(destination.parent)
    ) as tmp:
        image.write(dir_path=tmp, file_name="response", file_format="nifti")
        produced = Path(tmp) / "response.nii.gz"
        if not produced.is_file():
            raise RuntimeError("MIRP did not write the requested response map")
        if destination.name.lower().endswith(".nii.gz"):
            os.replace(produced, destination)
        else:
            with (
                gzip.open(produced, "rb") as source_file,
                destination.open("wb") as output_file,
            ):
                shutil.copyfileobj(source_file, output_file)


def _is_mirp_riesz_grid_error(error: Exception) -> bool:
    """Identify MIRP failures caused by its fixed three-axis Riesz grid."""

    if isinstance(error, IndexError):
        return "list index out of range" in str(error).lower()
    if not isinstance(error, ValueError):
        return False
    message = str(error).lower()
    return (
        "could not be broadcast together with shapes" in message
        or "shape mismatch" in message
    )


def _mirp_backend(
    source: Path, destination: Path, params: Mapping[str, Any]
) -> dict[str, Any]:
    import numpy as np
    from mirp import extract_images

    kwargs = _mirp_kwargs(params)
    native_default = kwargs.pop("_native_default", None)
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = extract_images(
                image=str(source),
                export_images=True,
                write_images=False,
                image_export_format="native",
                **kwargs,
            )
    except (IndexError, ValueError) as error:
        if str(params["filter"]) in {
            "riesz_log",
            "riesz_simoncelli",
        } and _is_mirp_riesz_grid_error(error):
            _unsupported(
                "mirp",
                params,
                "MIRP cannot execute its public Riesz implementation "
                "for this input geometry because its frequency-grid axes do "
                "not match the response-map axes",
            )
        raise
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError("MIRP expected exactly one workflow result")
    entry = result[0]
    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
        raise RuntimeError("MIRP returned an unexpected image-export structure")
    images = entry[0]
    transformed = []
    for image in images:
        attributes = image.get_export_attributes()
        if attributes.get("filter_type") is not None:
            transformed.append(image)
    if len(transformed) != 1:
        raise RuntimeError(
            f"MIRP expected exactly one transformed response map, got {len(transformed)}"
        )
    response_image = transformed[0]
    response = np.asarray(response_image.get_voxel_grid())
    if response.ndim != 3 or not np.isfinite(response).all():
        raise RuntimeError("MIRP produced an invalid response map")
    _write_mirp_image(response_image, destination)
    result = {**_array_metadata(response), "geometry_preserved": True}
    if native_default is not None:
        result["native_default"] = native_default
    return result


def _ensure_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _medimage_filter_module(module_name: str) -> Any:
    """Import a filter without executing MEDimage's heavyweight top-level package.

    ``MEDimage.__init__`` imports its learning stack, which can fail in the
    pinned environment before any filter code is reached (for example while
    loading XGBoost/OpenMP).  Namespace stubs mirror the established MEDimage
    feature adapter and still load the installed package's unmodified filter
    modules from their real paths.
    """

    global _MEDIMAGE_PACKAGE_INFO
    if _MEDIMAGE_PACKAGE_INFO is None:
        package_name = "MEDiml"
        specification = importlib.util.find_spec(package_name)
        if specification is None or not specification.submodule_search_locations:
            package_name = "MEDimage"
            specification = importlib.util.find_spec(package_name)
        if specification is None or not specification.submodule_search_locations:
            raise RuntimeError("Installed MEDimage package could not be located")
        root = Path(next(iter(specification.submodule_search_locations)))
        _MEDIMAGE_PACKAGE_INFO = package_name, root
    else:
        package_name, root = _MEDIMAGE_PACKAGE_INFO

    _ensure_namespace_package(package_name, root)
    _ensure_namespace_package(f"{package_name}.filters", root / "filters")
    _ensure_namespace_package(f"{package_name}.utils", root / "utils")
    return importlib.import_module(f"{package_name}.filters.{module_name}")


def _medimage_backend(
    source: Path, destination: Path, params: Mapping[str, Any]
) -> dict[str, Any]:
    import numpy as np

    reference, image, spacing = _load_nibabel(source)
    name = str(params["filter"])
    dimensionality = int(params["dimensionality"])
    backend_details: dict[str, Any] = {}
    padding = {
        "zero": "constant",
        "nearest": "edge",
        "periodic": "wrap",
        "mirror": "symmetric",
    }[str(params["boundary"])]

    if name == "mean":
        apply_mean = _medimage_filter_module("mean").apply_mean

        response = apply_mean(
            image,
            ndims=dimensionality,
            size=int(params["support"]),
            padding=padding,
            orthogonal_rot=False,
        )
    elif name == "log":
        LaplacianOfGaussian = _medimage_filter_module("log").LaplacianOfGaussian

        axes = (0, 1, 2) if dimensionality == 3 else (0, 1)
        if not _is_isotropic(spacing, axes):
            _unsupported(
                "medimage",
                params,
                "the LoG API accepts one voxel length and needs isotropic filtered axes",
            )
        sigma = float(params["sigma_mm"]) / spacing[0]
        size = 1 + 2 * int(math.floor(float(params["truncate"]) * sigma + 0.5))
        native_filter = LaplacianOfGaussian(
            ndims=dimensionality, size=size, sigma=sigma, padding=padding
        )
        response = np.squeeze(
            native_filter.convolve(
                np.expand_dims(image.astype(np.float64), 0), orthogonal_rot=False
            ),
            axis=0,
        )
    elif name == "laws":
        apply_laws = _medimage_filter_module("laws").apply_laws

        if params["rotation_invariant"] and params["pooling"] != "max":
            _unsupported(
                "medimage",
                params,
                "Laws rotation invariance is hard-coded to max pooling",
            )
        tokens = re.findall(r"[LESWR][35]", str(params["kernels"]))
        # MEDimage constructs a requested [a,b,c] kernel as [c,b,a], flips
        # native kernel axes 1 and 2, and swaps caller axes 0 and 2 around the
        # convolution.  In caller coordinates the resulting kernel is
        # [reverse(a), reverse(b), c].  Conjugating the native operation by
        # reflections of caller axes 0 and 1 restores the requested IBSI
        # coordinate frame without changing the package implementation.
        reflected_image = np.flip(image, axis=(0, 1))
        response = apply_laws(
            reflected_image,
            config=tokens,
            energy_distance=int(params.get("energy_distance", 7)),
            padding=padding,
            rot_invariance=bool(params["rotation_invariant"]),
            orthogonal_rot=False,
            energy_image=bool(params["compute_energy"]),
        )
        response = np.flip(np.asarray(response), axis=(0, 1))
        backend_details["coordinate_frame_correction"] = (
            "conjugated_reflection_axes_0_1"
        )
    elif name == "gabor":
        Gabor = _medimage_filter_module("gabor").Gabor

        if dimensionality != 2:
            _unsupported("medimage", params, "Gabor uses a two-dimensional kernel")
        axes = (0, 1, 2) if params["average_over_planes"] else (0, 1)
        if not _is_isotropic(spacing, axes):
            _unsupported(
                "medimage",
                params,
                "the Gabor API accepts one voxel length and needs isotropic requested planes",
            )
        if params["pooling"] not in {"average", "max"}:
            _unsupported(
                "medimage", params, "Gabor supports only average or max pooling"
            )
        sigma = float(params["sigma_mm"]) / spacing[0]
        wavelength = float(params["lambda_mm"]) / spacing[0]
        size = 2 * int(7.0 * sigma + 0.5) + 1
        # MEDimage's configured/native pipeline negates the requested angle
        # before constructing ``Gabor``. Its array axes use the opposite
        # handedness to the IBSI image-coordinate convention.
        angle = -float(params.get("delta_theta", params["theta"]))
        native_filter = Gabor(
            size=size,
            sigma=sigma,
            lamb=wavelength,
            gamma=float(params["gamma"]),
            theta=angle,
            rot_invariance=bool(params["rotation_invariant"]),
            padding=padding,
        )
        response = np.squeeze(
            native_filter.convolve(
                np.expand_dims(image.astype(np.float64), 0),
                orthogonal_rot=bool(params["average_over_planes"]),
                pooling_method="mean" if params["pooling"] == "average" else "max",
            ),
            axis=0,
        )
    elif name == "wavelet":
        Wavelet = _medimage_filter_module("wavelet").Wavelet

        if params["rotation_invariant"] and params["pooling"] != "average":
            _unsupported(
                "medimage",
                params,
                "wavelet rotation invariance is hard-coded to average pooling",
            )
        image_values = image.astype(np.float64)
        decomposition = str(params["decomposition"])
        level = int(params["level"])
        exact_2d_level1_rotation = (
            dimensionality == 2 and bool(params["rotation_invariant"]) and level == 1
        )
        native_filter = Wavelet(
            ndims=dimensionality,
            wavelet_name=str(params["wavelet"]),
            padding=padding,
            # MEDimage's level-1 2D "rotation" path averages reflections and
            # permuted subbands, which is not the IBSI four-quarter-turn
            # representation.  For that exact case, orchestration below rotates
            # the image around unchanged standard native DWT calls as explicitly
            # permitted by IBSI 2 section 3.3.
            rot_invariance=(
                bool(params["rotation_invariant"]) and not exact_2d_level1_rotation
            ),
        )
        if dimensionality == 2:
            # MEDimage Wavelet.convolve indexes only images[0]. A 2D native
            # call therefore consumes exactly one (batch, x, y) slice; passing
            # the complete (batch, x, y, z) volume makes pywt transform three
            # axes and produces keys such as "aad" instead of the requested
            # two-axis key. Apply that same native class independently to each
            # axial slice, as required by the IBSI 2D filtering definition.
            slices = []
            expected_slice_shape = (1, *image_values.shape[:2])
            for slice_index in range(image_values.shape[2]):
                image_slice = image_values[:, :, slice_index]
                if exact_2d_level1_rotation:
                    rotated_responses = []
                    for quarter_turns in range(4):
                        rotated_slice = np.rot90(image_slice, quarter_turns)
                        rotated_response = np.asarray(
                            native_filter.convolve(
                                np.expand_dims(rotated_slice, 0),
                                _filter=decomposition,
                                level=level,
                            )
                        )
                        expected_rotated_shape = (1, *rotated_slice.shape)
                        if rotated_response.shape != expected_rotated_shape:
                            raise RuntimeError(
                                "MEDimage rotated 2D wavelet changed slice shape: "
                                f"{rotated_response.shape} versus "
                                f"{expected_rotated_shape}"
                            )
                        rotated_responses.append(
                            np.rot90(rotated_response[0], -quarter_turns)
                        )
                    slice_response = np.expand_dims(
                        np.mean(rotated_responses, axis=0),
                        0,
                    )
                else:
                    slice_response = np.asarray(
                        native_filter.convolve(
                            np.expand_dims(image_slice, 0),
                            _filter=decomposition,
                            level=level,
                        )
                    )
                if slice_response.shape != expected_slice_shape:
                    raise RuntimeError(
                        "MEDimage 2D wavelet changed slice shape: "
                        f"{slice_response.shape} versus {expected_slice_shape}"
                    )
                slices.append(slice_response[0])
            response = np.stack(slices, axis=2)
            if exact_2d_level1_rotation:
                backend_details["rotation_orchestration"] = (
                    "four_quarter_turn_native_dwt_average"
                )
        else:
            response = np.squeeze(
                native_filter.convolve(
                    np.expand_dims(image_values, 0),
                    _filter=decomposition,
                    level=level,
                ),
                axis=0,
            )
    else:
        _unsupported(
            "medimage",
            params,
            "MEDimage exposes mean, LoG, Laws, Gabor, and separable wavelet only",
        )

    _save_nibabel(reference, response, destination)
    return {
        **_array_metadata(response),
        "geometry_preserved": True,
        **backend_details,
    }


def _validate_zrad_static(params: Mapping[str, Any]) -> dict[str, Any] | None:
    name = str(params["filter"])
    dimensionality = int(params["dimensionality"])
    if name == "laws":
        if params["rotation_invariant"] and params["pooling"] != "max":
            _unsupported(
                "zrad",
                params,
                "the average Laws path is invalid and no exact min/sum path exists",
            )
        if params["compute_energy"] and params["boundary"] != "mirror":
            return _uncontrolled(
                "zrad",
                params,
                "Laws energy computation hard-codes reflect (IBSI mirror) padding",
                uncontrolled=("energy_boundary",),
                effective_boundary="mirror_for_the_energy_step",
            )
    elif name == "gabor":
        if dimensionality != 2:
            _unsupported("zrad", params, "Gabor uses a two-dimensional kernel")
        if params["rotation_invariant"] and params["pooling"] != "average":
            _unsupported("zrad", params, "rotation-invariant Gabor always averages")
        if params["average_over_planes"] and not params["rotation_invariant"]:
            _unsupported(
                "zrad",
                params,
                "orthogonal_planes is ignored unless rotation_invariance is enabled",
            )
    elif name == "wavelet":
        if params["rotation_invariant"] and params["pooling"] != "average":
            _unsupported("zrad", params, "rotation-invariant wavelets always average")
    elif name not in {"mean", "log"}:
        _unsupported(
            "zrad",
            params,
            "Z-Rad exposes mean, LoG, Laws, Gabor, and separable wavelet only",
        )
    return None


def _zrad_gabor_kwargs(
    params: Mapping[str, Any],
    *,
    spacing: Sequence[float],
    boundary: str,
) -> dict[str, Any]:
    """Map the IBSI ±7σ Gabor support to Z-Rad's full-width convention."""

    return {
        "padding_type": boundary,
        "res_mm": spacing[0],
        "sigma_mm": float(params["sigma_mm"]),
        "lambda_mm": float(params["lambda_mm"]),
        "gamma": float(params["gamma"]),
        "theta": float(params.get("delta_theta", params["theta"])),
        "rotation_invariance": bool(params["rotation_invariant"]),
        "orthogonal_planes": bool(params["average_over_planes"]),
        # Z-Rad interprets n_stds as the full kernel width in standard
        # deviations.  IBSI truncates at seven standard deviations on each
        # side, hence a full width of fourteen.
        "n_stds": 14.0,
    }


def _zrad_backend(
    source: Path, destination: Path, params: Mapping[str, Any]
) -> dict[str, Any]:
    native_default = _validate_zrad_static(params)

    import numpy as np
    from zrad.filtering import Gabor, Laws, LoG, Mean, Wavelets2D, Wavelets3D
    from zrad.image import Image

    image = Image.from_nifti(str(source))
    if image.array is None or np.asarray(image.array).ndim != 3:
        raise RuntimeError("Z-Rad failed to load a three-dimensional NIfTI")
    spacing = tuple(float(value) for value in image.spacing)
    name = str(params["filter"])
    dimensionality = int(params["dimensionality"])
    boundary = {
        "zero": "constant",
        "nearest": "nearest",
        "periodic": "wrap",
        "mirror": "reflect",
    }[str(params["boundary"])]
    dim = f"{dimensionality}D"

    if name == "mean":
        native_filter = Mean(
            padding_type=boundary, support=int(params["support"]), dimensionality=dim
        )
    elif name == "log":
        axes = (0, 1, 2) if dimensionality == 3 else (0, 1)
        if not _is_isotropic(spacing, axes):
            _unsupported(
                "zrad",
                params,
                "the LoG implementation uses spacing[0] and needs isotropic filtered axes",
            )
        native_filter = LoG(
            padding_type=boundary,
            sigma_mm=float(params["sigma_mm"]),
            cutoff=float(params["truncate"]),
            dimensionality=dim,
        )
    elif name == "laws":
        native_filter = Laws(
            response_map=str(params["kernels"]),
            padding_type=boundary,
            distance=int(params.get("energy_distance", 7)),
            energy_map=bool(params["compute_energy"]),
            dimensionality=dim,
            rotation_invariance=bool(params["rotation_invariant"]),
            pooling="max" if params["rotation_invariant"] else None,
        )
    elif name == "gabor":
        axes = (0, 1, 2) if params["average_over_planes"] else (0, 1)
        if not _is_isotropic(spacing, axes):
            _unsupported(
                "zrad", params, "the Gabor implementation accepts one resolution value"
            )
        native_filter = Gabor(
            **_zrad_gabor_kwargs(
                params,
                spacing=spacing,
                boundary=boundary,
            )
        )
    else:
        wavelet_class = Wavelets2D if dimensionality == 2 else Wavelets3D
        native_filter = wavelet_class(
            wavelet_type=str(params["wavelet"]),
            padding_type=boundary,
            response_map=str(params["decomposition"]),
            decomposition_level=int(params["level"]),
            rotation_invariance=bool(params["rotation_invariant"]),
        )

    response_image = native_filter.apply(image)
    response = np.asarray(response_image.array)
    if (
        response.shape != np.asarray(image.array).shape
        or not np.isfinite(response).all()
    ):
        raise RuntimeError("Z-Rad produced an invalid response-map grid")
    response_image.save_as_nifti(str(destination))
    result = {**_array_metadata(response), "geometry_preserved": True}
    if native_default is not None:
        result["native_default"] = native_default
    return result


_BACKENDS = {
    "pictologics": _pictologics_backend,
    "pyradiomics": _pyradiomics_backend,
    "mirp": _mirp_backend,
    "medimage": _medimage_backend,
    "zrad": _zrad_backend,
}


def apply_native_filter(
    adapter: str,
    input_path: str | Path,
    output_path: str | Path,
    parameters: Mapping[str, Any],
    *,
    native_default: bool = False,
) -> dict[str, Any]:
    """Apply one exact package-native filter and write its NIfTI response map.

    The input must already be on the intended IBSI grid.  This function never
    resamples, crops, resegments, masks, or discretizes it.  That separation is
    required because IBSI 2 filtering operates on the complete prepared image,
    while Phase 2 statistics use a separately prepared ROI mask.

    With ``native_default``, a filter that the package has but whose boundary
    or truncation its API cannot set runs with the package's own setting.  The
    result records the uncontrolled parameters and belongs to a separate
    reporting class, never to exact native support.
    """

    normalized_adapter = str(adapter).lower()
    if normalized_adapter not in ADAPTERS:
        raise ValueError(
            f"Unknown adapter {adapter!r}; expected one of {sorted(ADAPTERS)}"
        )
    params = normalize_parameters(parameters, adapter=normalized_adapter)
    if native_default:
        params["native_default"] = True
    if params["filter"] == "none":
        boundary_policy = "not_applicable"
    else:
        boundary_policy = (
            "protocol_explicit"
            if "boundary" in parameters
            else "implementation_selected"
        )
    source, destination = _validate_paths(input_path, output_path)

    if params["filter"] == "none":
        result = _copy_identity(source, destination)
    else:
        result = _BACKENDS[normalized_adapter](source, destination, params)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"Native backend did not create a response map: {destination}"
        )

    boundary_implementation = result.pop(
        "boundary_implementation",
        "not_applicable" if params["filter"] == "none" else "as_specified",
    )
    default_record = result.pop("native_default", None)
    effective_boundary = None if params["filter"] == "none" else params["boundary"]
    if native_default and default_record is None:
        # The exact run would succeed, so there is nothing to add.
        destination.unlink()
        raise UnsupportedNativeFilter(
            normalized_adapter,
            str(params["filter"]),
            "the exact run supports this configuration; no native-default run",
        )
    if default_record is not None:
        effective_boundary = default_record["effective_boundary"]
        boundary_implementation = "package_default"
    return {
        "schema_version": 1,
        "adapter": normalized_adapter,
        "distribution": _DISTRIBUTIONS[normalized_adapter],
        "distribution_version": _package_version(normalized_adapter),
        "parameters": params,
        "boundary_execution": {
            "policy": boundary_policy,
            "selected": params.get("boundary"),
            "effective": effective_boundary,
            "implementation": boundary_implementation,
        },
        "support_class": "native_default" if native_default else "native",
        "native_default": default_record,
        "input_path": str(source),
        "output_path": str(destination),
        "output_size_bytes": int(destination.stat().st_size),
        **result,
    }


def _load_cli_parameters(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.parameters_file is not None:
        with Path(args.parameters_file).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    else:
        value = json.loads(args.parameters_json)
    if not isinstance(value, Mapping):
        raise TypeError("Filter parameters JSON must contain an object")
    # The bundle generator passes the complete reviewed filter-config document so
    # that the exact bytes can be checksummed as provenance.  Direct callers may
    # still pass the normalized mapping itself.
    if "parameters" in value:
        value = value["parameters"]
        if not isinstance(value, Mapping):
            raise TypeError("Filter config envelope .parameters must contain an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for isolated adapter environments."""

    parser = argparse.ArgumentParser(prog="ibsi2-native-filter")
    parser.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parameters-json")
    source.add_argument("--parameters-file")
    parser.add_argument(
        "--native-default",
        action="store_true",
        help="Run a filter whose boundary or truncation the API cannot set "
        "with the package default (a separate reporting class)",
    )
    args = parser.parse_args(argv)

    try:
        metadata = apply_native_filter(
            args.adapter,
            args.input,
            args.output,
            _load_cli_parameters(args),
            native_default=args.native_default,
        )
    except UnsupportedNativeFilter as exc:
        print(
            json.dumps(
                {
                    "status": "unsupported_native_filter",
                    "adapter": exc.adapter,
                    "filter": exc.filter_name,
                    "reason": exc.reason,
                    "evidence": list(exc.evidence),
                    "native_default_possible": isinstance(
                        exc, UncontrolledNativeParameter
                    ),
                    "uncontrolled": list(getattr(exc, "uncontrolled", ())),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

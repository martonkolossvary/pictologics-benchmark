"""Reviewed IBSI 2 phase-1 and phase-2 filter configuration catalogue.

The values in this module transcribe the IBSI 2 reference manual v9 benchmark
tables.  They are deliberately package-neutral: adapter wrappers may execute a
configuration only when the installed package can express every required
parameter.  Phase 1 boundary conditions are exact test parameters.  Phase 2
uses the manual's implementation-selected padding allowance, so the selected
and effective boundary must be recorded by each native execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Mapping

from bench.compliance.references import (
    IBSI2_PHASE1_SOURCE_MASK_SHA256,
    IBSI2_PHASE1_TEST_IDS,
    IBSI2_PHASE2_FILTER_IDS,
)


IBSI2_MANUAL_VERSION = "9"
IBSI2_PHASE2_BOUNDARY_POLICY = "implementation_selected"
IBSI2_PROTOCOL_REVIEW = (
    "IBSI 2 reference manual v9, Phase 1 Table 6.1 and Phase 2 "
    "Tables 6.2-6.3, including implementation-selected Phase 2 padding"
)


@dataclass(frozen=True)
class Phase1FilterSpec:
    """One Phase 1 response-map test and its exact official source phantom."""

    test_id: str
    phantom: str
    source_image_sha256: str
    parameters: Mapping[str, Any]

    @property
    def source_image_relative_path(self) -> str:
        return f"phase1/{self.phantom}/image/{self.phantom}.nii.gz"

    @property
    def source_mask_relative_path(self) -> str:
        return f"phase1/{self.phantom}/mask/mask.nii.gz"

    def filter_config(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "specification": "IBSI 2",
            "phase": "phase1",
            "reference_manual_version": IBSI2_MANUAL_VERSION,
            "test_id": self.test_id,
            "source_phantom": self.phantom,
            "source_image_sha256": self.source_image_sha256,
            "parameters": dict(self.parameters),
        }

    def preprocessing_config(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "specification": "IBSI 2",
            "phase": "phase1",
            "reference_manual_version": IBSI2_MANUAL_VERSION,
            "test_id": self.test_id,
            "settings": {
                "input": "official_nifti_unmodified",
                "crop": False,
                "resampling": {"enabled": False},
                "intensity_resegmentation": "none",
                "response_map_discretization": "none",
                "output_geometry": "source_image",
            },
        }


@dataclass(frozen=True)
class Phase2FilterSpec:
    """One A/B Phase 2 filter configuration."""

    filter_id: str
    parameters: Mapping[str, Any]

    @property
    def dimension(self) -> str:
        return self.filter_id.rsplit(".", 1)[1]

    @property
    def published(self) -> bool:
        return int(self.filter_id.split(".", 1)[0]) <= 9

    def filter_config(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "specification": "IBSI 2",
            "phase": "phase2",
            "reference_manual_version": IBSI2_MANUAL_VERSION,
            "filter_id": self.filter_id,
            "boundary_policy": IBSI2_PHASE2_BOUNDARY_POLICY,
            "parameters": dict(self.parameters),
        }


_PHASE1_IMAGE_SHA256 = {
    "checkerboard": "55cb8915c22178a420904e94a74f7fca94f3a887095957ef679809bfdff7a23e",
    "impulse": "8a8314fdcf41a55f4de3f8e7d0f78a09563ffd4d11f20d73995e8b13369a78bd",
    "pattern_1": "e2ec165e237a9fb8e017c5ef7c5472fa82d10da1433d4bb296762916f8e7570f",
    "sphere": "ab6136aa65c6378fc680f3a3dffc9f2ac12950635b3ad8d15aa3a4896f745064",
}


def _p1(
    test_id: str,
    phantom: str,
    filter_name: str,
    dimensionality: int,
    boundary: str | None,
    **parameters: Any,
) -> Phase1FilterSpec:
    return Phase1FilterSpec(
        test_id=test_id,
        phantom=phantom,
        source_image_sha256=_PHASE1_IMAGE_SHA256[phantom],
        parameters={
            "filter": filter_name,
            "dimensionality": dimensionality,
            **({"boundary": boundary} if boundary is not None else {}),
            **parameters,
        },
    )


PHASE1_FILTER_SPECS = (
    _p1("1.a.1", "checkerboard", "mean", 3, "zero", support=15),
    _p1("1.a.2", "checkerboard", "mean", 3, "nearest", support=15),
    _p1("1.a.3", "checkerboard", "mean", 3, "periodic", support=15),
    _p1("1.a.4", "checkerboard", "mean", 3, "mirror", support=15),
    _p1("1.b.1", "impulse", "mean", 2, "zero", support=15),
    _p1("2.a", "impulse", "log", 3, "zero", sigma_mm=3.0, truncate=4.0),
    _p1("2.b", "checkerboard", "log", 3, "mirror", sigma_mm=5.0, truncate=4.0),
    _p1("2.c", "checkerboard", "log", 2, "mirror", sigma_mm=5.0, truncate=4.0),
    _p1("3.a.1", "impulse", "laws", 3, "zero", kernels="E5L5S5"),
    _p1(
        "3.a.2",
        "impulse",
        "laws",
        3,
        "zero",
        kernels="E5L5S5",
        rotation_invariant=True,
        pooling="max",
    ),
    _p1(
        "3.a.3",
        "impulse",
        "laws",
        3,
        "zero",
        kernels="E5L5S5",
        rotation_invariant=True,
        pooling="max",
        compute_energy=True,
        energy_distance=7,
    ),
    _p1("3.b.1", "checkerboard", "laws", 3, "mirror", kernels="E3W5R5"),
    _p1(
        "3.b.2",
        "checkerboard",
        "laws",
        3,
        "mirror",
        kernels="E3W5R5",
        rotation_invariant=True,
        pooling="max",
    ),
    _p1(
        "3.b.3",
        "checkerboard",
        "laws",
        3,
        "mirror",
        kernels="E3W5R5",
        rotation_invariant=True,
        pooling="max",
        compute_energy=True,
        energy_distance=7,
    ),
    _p1("3.c.1", "checkerboard", "laws", 2, "mirror", kernels="L5S5"),
    _p1(
        "3.c.2",
        "checkerboard",
        "laws",
        2,
        "mirror",
        kernels="L5S5",
        rotation_invariant=True,
        pooling="max",
    ),
    _p1(
        "3.c.3",
        "checkerboard",
        "laws",
        2,
        "mirror",
        kernels="L5S5",
        rotation_invariant=True,
        pooling="max",
        compute_energy=True,
        energy_distance=7,
    ),
    _p1(
        "4.a.1",
        "impulse",
        "gabor",
        2,
        "zero",
        sigma_mm=10.0,
        lambda_mm=4.0,
        gamma=0.5,
        theta=pi / 3.0,
    ),
    _p1(
        "4.a.2",
        "impulse",
        "gabor",
        2,
        "zero",
        sigma_mm=10.0,
        lambda_mm=4.0,
        gamma=0.5,
        rotation_invariant=True,
        delta_theta=pi / 4.0,
        pooling="average",
        average_over_planes=True,
    ),
    _p1(
        "4.b.1",
        "sphere",
        "gabor",
        2,
        "mirror",
        sigma_mm=20.0,
        lambda_mm=8.0,
        gamma=2.5,
        theta=5.0 * pi / 4.0,
    ),
    _p1(
        "4.b.2",
        "sphere",
        "gabor",
        2,
        "mirror",
        sigma_mm=20.0,
        lambda_mm=8.0,
        gamma=2.5,
        rotation_invariant=True,
        delta_theta=pi / 8.0,
        pooling="average",
        average_over_planes=True,
    ),
    _p1(
        "5.a.1",
        "impulse",
        "wavelet",
        3,
        "zero",
        wavelet="db2",
        level=1,
        decomposition="LHL",
    ),
    _p1(
        "5.a.2",
        "impulse",
        "wavelet",
        3,
        "zero",
        wavelet="db2",
        level=1,
        decomposition="LHL",
        rotation_invariant=True,
        pooling="average",
    ),
    _p1(
        "6.a.1",
        "sphere",
        "wavelet",
        3,
        "periodic",
        wavelet="coif1",
        level=1,
        decomposition="HHL",
    ),
    _p1(
        "6.a.2",
        "sphere",
        "wavelet",
        3,
        "periodic",
        wavelet="coif1",
        level=1,
        decomposition="HHL",
        rotation_invariant=True,
        pooling="average",
    ),
    _p1(
        "7.a.1",
        "checkerboard",
        "wavelet",
        3,
        "mirror",
        wavelet="haar",
        level=2,
        decomposition="LLL",
        rotation_invariant=True,
        pooling="average",
    ),
    _p1(
        "7.a.2",
        "checkerboard",
        "wavelet",
        3,
        "mirror",
        wavelet="haar",
        level=2,
        decomposition="HHH",
        rotation_invariant=True,
        pooling="average",
    ),
    _p1("8.a.1", "checkerboard", "simoncelli", 3, "periodic", level=1),
    _p1("8.a.2", "checkerboard", "simoncelli", 3, "periodic", level=2),
    _p1("8.a.3", "checkerboard", "simoncelli", 3, "periodic", level=3),
    _p1(
        "9.a",
        "impulse",
        "riesz_log",
        3,
        "zero",
        sigma_mm=3.0,
        truncate=4.0,
        order=[1, 0, 0],
    ),
    _p1(
        "9.b.1",
        "sphere",
        "riesz_log",
        3,
        "zero",
        sigma_mm=3.0,
        truncate=4.0,
        order=[0, 2, 0],
    ),
    _p1(
        "9.b.2",
        "sphere",
        "riesz_log",
        3,
        "zero",
        sigma_mm=3.0,
        truncate=4.0,
        order=[0, 2, 0],
        tensor_sigma=1.0,
    ),
    _p1(
        "10.a",
        "impulse",
        "riesz_simoncelli",
        3,
        "zero",
        level=1,
        order=[1, 0, 0],
    ),
    _p1(
        "10.b.1",
        "pattern_1",
        "riesz_simoncelli",
        3,
        "nearest",
        level=1,
        order=[0, 2, 0],
    ),
    _p1(
        "10.b.2",
        "pattern_1",
        "riesz_simoncelli",
        3,
        "nearest",
        level=1,
        order=[0, 2, 0],
        tensor_sigma=1.0,
    ),
)
PHASE1_FILTER_SPECS_BY_ID = {spec.test_id: spec for spec in PHASE1_FILTER_SPECS}


def _p2(
    number: int,
    dimension: str,
    filter_name: str,
    **parameters: Any,
) -> Phase2FilterSpec:
    dimensionality = 2 if dimension == "A" else 3
    return Phase2FilterSpec(
        filter_id=f"{number}.{dimension}",
        parameters={
            "filter": filter_name,
            "dimensionality": dimensionality,
            **parameters,
        },
    )


_PHASE2_BY_NUMBER: dict[int, tuple[str, dict[str, Any], dict[str, Any]]] = {
    1: ("none", {}, {}),
    2: ("mean", {"support": 5}, {}),
    3: ("log", {"sigma_mm": 1.5, "truncate": 4.0}, {}),
    4: (
        "laws",
        {
            "kernels": "L5E5",
            "rotation_invariant": True,
            "pooling": "max",
            "compute_energy": True,
            "energy_distance": 7,
        },
        {"kernels": "L5E5E5"},
    ),
    5: (
        "gabor",
        {
            "sigma_mm": 5.0,
            "lambda_mm": 2.0,
            "gamma": 1.5,
            "rotation_invariant": True,
            "delta_theta": pi / 8.0,
            "pooling": "average",
            "average_over_planes": False,
        },
        # Configuration B still uses a 2D Gabor kernel and averages the
        # responses over the three orthogonal planes; it is not a 3D kernel.
        {"dimensionality": 2, "average_over_planes": True},
    ),
    6: (
        "wavelet",
        {
            "wavelet": "db3",
            "level": 1,
            "decomposition": "LH",
            "rotation_invariant": True,
            "pooling": "average",
        },
        {"decomposition": "LLH"},
    ),
    7: (
        "wavelet",
        {
            "wavelet": "db3",
            "level": 2,
            "decomposition": "HH",
            "rotation_invariant": True,
            "pooling": "average",
        },
        {"decomposition": "HHH"},
    ),
    8: ("simoncelli", {"level": 1}, {}),
    9: ("simoncelli", {"level": 2}, {}),
    10: (
        "riesz_simoncelli",
        {"level": 1, "order": [0, 2]},
        {"order": [0, 2, 0]},
    ),
    11: (
        "riesz_simoncelli",
        {"level": 1, "order": [0, 2], "tensor_sigma": 1.0},
        {"order": [0, 2, 0]},
    ),
}


def _phase2_specs() -> tuple[Phase2FilterSpec, ...]:
    output: list[Phase2FilterSpec] = []
    for number in range(1, 12):
        filter_name, common, three_dimensional = _PHASE2_BY_NUMBER[number]
        for dimension in ("A", "B"):
            parameters = dict(common)
            if dimension == "B":
                parameters.update(three_dimensional)
            output.append(_p2(number, dimension, filter_name, **parameters))
    return tuple(output)


PHASE2_FILTER_SPECS = _phase2_specs()
PHASE2_FILTER_SPECS_BY_ID = {spec.filter_id: spec for spec in PHASE2_FILTER_SPECS}


def validate_catalogue() -> None:
    """Fail fast if a later edit makes the reviewed catalogue incomplete."""

    if tuple(spec.test_id for spec in PHASE1_FILTER_SPECS) != IBSI2_PHASE1_TEST_IDS:
        raise RuntimeError("IBSI 2 Phase 1 catalogue is not the exact 36-test surface")
    if tuple(spec.filter_id for spec in PHASE2_FILTER_SPECS) != IBSI2_PHASE2_FILTER_IDS:
        raise RuntimeError(
            "IBSI 2 Phase 2 catalogue is not the exact 22-filter surface"
        )
    for spec in PHASE1_FILTER_SPECS:
        if spec.phantom not in _PHASE1_IMAGE_SHA256:
            raise RuntimeError(f"Unknown Phase 1 phantom: {spec.phantom}")
        if not spec.parameters:
            raise RuntimeError(f"Empty Phase 1 parameters: {spec.test_id}")
    for spec in PHASE2_FILTER_SPECS:
        if not spec.parameters:
            raise RuntimeError(f"Empty Phase 2 parameters: {spec.filter_id}")
        if "boundary" in spec.parameters:
            raise RuntimeError(
                f"Phase 2 {spec.filter_id} must use implementation-selected padding"
            )


def validate_phase1_filter_config(config: Mapping[str, Any], *, test_id: str) -> None:
    """Require byte-semantically equivalent reviewed Phase 1 parameters."""

    expected = PHASE1_FILTER_SPECS_BY_ID[test_id].filter_config()
    if dict(config) != expected:
        raise ValueError(
            f"IBSI 2 Phase 1 {test_id} filter configuration differs from v9"
        )


def validate_phase2_filter_config(config: Mapping[str, Any], *, filter_id: str) -> None:
    """Require byte-semantically equivalent reviewed Phase 2 parameters."""

    expected = PHASE2_FILTER_SPECS_BY_ID[filter_id].filter_config()
    if dict(config) != expected:
        raise ValueError(
            f"IBSI 2 Phase 2 {filter_id} filter configuration differs from v9"
        )


validate_catalogue()


__all__ = [
    "IBSI2_MANUAL_VERSION",
    "IBSI2_PHASE2_BOUNDARY_POLICY",
    "IBSI2_PROTOCOL_REVIEW",
    "IBSI2_PHASE1_SOURCE_MASK_SHA256",
    "PHASE1_FILTER_SPECS",
    "PHASE1_FILTER_SPECS_BY_ID",
    "PHASE2_FILTER_SPECS",
    "PHASE2_FILTER_SPECS_BY_ID",
    "Phase1FilterSpec",
    "Phase2FilterSpec",
    "validate_catalogue",
    "validate_phase1_filter_config",
    "validate_phase2_filter_config",
]

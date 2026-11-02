"""Static adapter capabilities used by planning, validation, and reporting.

The registry describes what the harness requests from each upstream package.  It
does not claim that an implementation is IBSI-conformant; conformance is a
separate, reference-data result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bench.ibsi_families import CALCULATION_FAMILIES, WORKLOAD_GROUPS


SelectionMode = Literal["native", "feature_selection", "post_filter"]


@dataclass(frozen=True)
class AdapterCapabilities:
    name: str
    distribution: str
    families: tuple[str, ...]
    selection_mode: SelectionMode
    notes: str = ""

    def supports(self, family: str) -> bool:
        """Support a family, or a group with at least one supported member."""

        if family in WORKLOAD_GROUPS:
            return bool(self.supported_members(family))
        return family in self.families

    def supported_members(self, workload: str) -> list[str]:
        """Return the families one task of this workload asks the adapter for."""

        members = WORKLOAD_GROUPS.get(workload, (workload,))
        return [family for family in members if family in self.families]


_ALL_IBSI_FAMILIES = tuple(CALCULATION_FAMILIES)

ADAPTERS: dict[str, AdapterCapabilities] = {
    "pictologics": AdapterCapabilities(
        name="pictologics",
        distribution="pictologics",
        families=_ALL_IBSI_FAMILIES,
        selection_mode="native",
    ),
    "pyradiomics": AdapterCapabilities(
        name="pyradiomics",
        distribution="pyradiomics",
        families=(
            "morphology",
            "morphology_shape",
            "intensity",
            "histogram",
            "glcm",
            "glrlm",
            "glszm",
            "ngtdm",
            "ngldm",
        ),
        selection_mode="feature_selection",
        notes="IBSI families are selected through explicit PyRadiomics feature names.",
    ),
    "mirp": AdapterCapabilities(
        name="mirp",
        distribution="mirp",
        families=_ALL_IBSI_FAMILIES,
        selection_mode="native",
    ),
    "medimage": AdapterCapabilities(
        name="medimage",
        distribution="medimage-pkg",
        families=_ALL_IBSI_FAMILIES,
        selection_mode="native",
        notes="The adapter uses MEDimage's family modules because no stable unified API is exposed.",
    ),
    "zrad": AdapterCapabilities(
        name="zrad",
        distribution="z-rad",
        families=_ALL_IBSI_FAMILIES,
        selection_mode="native",
        notes=(
            "Z-Rad 26.8 selects native feature families; canonical morphology "
            "and morphology_spatial request its opt-in morphology_correlation group."
        ),
    ),
}


def get_adapter(name: str) -> AdapterCapabilities:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown adapter {name!r}; choose one of: {choices}") from exc


def validate_families(name: str, families: list[str]) -> tuple[list[str], list[str]]:
    """Return (supported, unsupported) families in canonical order."""

    capabilities = get_adapter(name)
    requested = set(families)
    supported = [
        family
        for family in CALCULATION_FAMILIES
        if family in requested and capabilities.supports(family)
    ]
    unsupported = [
        family
        for family in CALCULATION_FAMILIES
        if family in requested and not capabilities.supports(family)
    ]
    unknown = sorted(requested.difference(CALCULATION_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown IBSI feature families: {', '.join(unknown)}")
    return supported, unsupported

"""Frozen input-representation routing for benchmark feature families.

The public benchmark datasets contain an original image, one mask-specific
IBSI FBN32 texture image, and one mask-specific IVH image. A fair family
comparison selects the same stored representation for every adapter instead of
asking each package to derive a nominally similar discretisation internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from bench.ibsi_families import WORKLOAD_GROUPS


HARMONIZED_INPUT_CONTRACT = "manifest_harmonized"
INTERNAL_INPUT_CONTRACT = "adapter_internal_preprocessing"

RAW_FAMILIES = frozenset(
    {
        "morphology",
        "morphology_shape",
        "morphology_spatial",
        "local_intensity",
        "intensity",
    }
)
FROZEN_FBN_FAMILIES = frozenset(
    {"histogram", "glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"}
)
IVH_FAMILIES = frozenset({"ivh"})


@dataclass(frozen=True)
class RepresentationSelection:
    image_path: str
    image_sha256: str
    representation_id: str
    discretization: str
    bins: int
    bin_width: float
    intensity_min: Optional[float]
    intensity_max: Optional[float]
    configured_levels: Optional[int]
    occupied_levels: Optional[int]
    derivation_sha256: Optional[str]


def select_representation(
    case: Mapping[str, Any],
    family: str,
    *,
    input_contract: str,
    internal_discretization: str,
    internal_bins: int,
    internal_bin_width: float,
    internal_intensity_min: Optional[float],
    internal_intensity_max: Optional[float],
) -> RepresentationSelection:
    """Resolve the immutable image and preprocessing contract for one family."""

    normalized_family = str(family).strip().lower()
    if normalized_family in WORKLOAD_GROUPS:
        # Every member of a group reads the same stored input.
        members = WORKLOAD_GROUPS[normalized_family]
        selections = {
            select_representation(
                case,
                member,
                input_contract=input_contract,
                internal_discretization=internal_discretization,
                internal_bins=internal_bins,
                internal_bin_width=internal_bin_width,
                internal_intensity_min=internal_intensity_min,
                internal_intensity_max=internal_intensity_max,
            )
            for member in members
        }
        if len(selections) != 1:
            raise ValueError(f"group members need different inputs: {family}")
        return selections.pop()
    if input_contract == INTERNAL_INPUT_CONTRACT:
        return RepresentationSelection(
            image_path=str(case["image_abs"]),
            image_sha256=str(case["image_sha256"]),
            representation_id="single_image_internal_preprocessing",
            discretization=internal_discretization,
            bins=int(internal_bins),
            bin_width=float(internal_bin_width),
            intensity_min=internal_intensity_min,
            intensity_max=internal_intensity_max,
            configured_levels=None,
            occupied_levels=None,
            derivation_sha256=None,
        )
    if input_contract != HARMONIZED_INPUT_CONTRACT:
        raise ValueError(f"unknown benchmark input contract: {input_contract!r}")

    if normalized_family in RAW_FAMILIES:
        raw_representation = case.get("raw_representation")
        if isinstance(raw_representation, Mapping):
            representation_id = str(raw_representation.get("id") or "").strip()
        else:
            representation_id = str(raw_representation or "").strip()
        return RepresentationSelection(
            image_path=str(case["image_abs"]),
            image_sha256=str(case["image_sha256"]),
            representation_id=representation_id or "original_continuous_image",
            discretization="raw",
            bins=int(internal_bins),
            bin_width=float(internal_bin_width),
            intensity_min=None,
            intensity_max=None,
            configured_levels=None,
            occupied_levels=None,
            derivation_sha256=None,
        )

    if normalized_family in FROZEN_FBN_FAMILIES:
        texture = case.get("texture_representation")
        if not isinstance(texture, Mapping):
            raise ValueError(
                f"case {case.get('case_id')} lacks texture_representation metadata"
            )
        path = str(case.get("discrete_image_abs") or "").strip()
        digest = str(case.get("discrete_image_sha256") or "").strip().lower()
        if not path or len(digest) != 64:
            raise ValueError(f"case {case.get('case_id')} lacks a bound discrete image")
        configured = int(texture.get("configured_levels") or 0)
        occupied = int(texture.get("occupied_levels") or 0)
        if configured < 2 or occupied < 1 or occupied > configured:
            raise ValueError("invalid configured/occupied discrete grey-level metadata")
        return RepresentationSelection(
            image_path=path,
            image_sha256=digest,
            representation_id=str(texture.get("id") or "").strip()
            or "mask_specific_ibsi_fbn",
            discretization="identity",
            bins=configured,
            bin_width=1.0,
            intensity_min=None,
            intensity_max=None,
            configured_levels=configured,
            occupied_levels=occupied,
            derivation_sha256=(
                str(texture.get("derivation_sha256")).strip().lower()
                if texture.get("derivation_sha256")
                else None
            ),
        )

    if normalized_family in IVH_FAMILIES:
        ivh = case.get("ivh_representation")
        if not isinstance(ivh, Mapping):
            raise ValueError(
                f"case {case.get('case_id')} lacks ivh_representation metadata"
            )
        path = str(case.get("ivh_image_abs") or "").strip()
        digest = str(case.get("ivh_image_sha256") or "").strip().lower()
        if not path or len(digest) != 64:
            raise ValueError(f"case {case.get('case_id')} lacks a bound IVH image")
        configured = int(ivh.get("configured_levels") or 0)
        occupied = int(ivh.get("occupied_levels") or 0)
        if configured < 2 or occupied < 1 or occupied > configured:
            raise ValueError("invalid configured/occupied IVH grey-level metadata")
        return RepresentationSelection(
            image_path=path,
            image_sha256=digest,
            representation_id=str(ivh.get("id") or "").strip()
            or "mask_specific_ibsi_fbs1_ivh_indices",
            discretization="identity",
            bins=configured,
            bin_width=1.0,
            intensity_min=None,
            intensity_max=None,
            configured_levels=configured,
            occupied_levels=occupied,
            derivation_sha256=(
                str(ivh.get("derivation_sha256")).strip().lower()
                if ivh.get("derivation_sha256")
                else None
            ),
        )

    raise ValueError(f"no harmonized input representation is defined for {family!r}")

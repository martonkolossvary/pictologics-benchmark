"""Typed records shared by IBSI importers, evaluators, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


def _optional_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class ReferenceRecord:
    specification: str
    phase: str
    dataset: str
    configuration: str
    profile: str
    in_profile: bool
    aggregation: str
    family: str
    feature_name: str
    feature_tag: str
    semantic_key: str
    ibsi_code: str
    consensus: str
    reference_value: Optional[float]
    tolerance: Optional[float]
    standardized: bool
    source_sheet: str
    source_row: int

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.specification,
            self.phase,
            self.configuration,
            self.aggregation,
            self.feature_tag,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "ReferenceRecord":
        return cls(
            specification=str(row.get("specification", "")),
            phase=str(row.get("phase", "")),
            dataset=str(row.get("dataset", "")),
            configuration=str(row.get("configuration", "")),
            profile=str(row.get("profile", "")),
            in_profile=_bool(row.get("in_profile", False)),
            aggregation=str(row.get("aggregation", "")),
            family=str(row.get("family", "")),
            feature_name=str(row.get("feature_name", "")),
            feature_tag=str(row.get("feature_tag", "")),
            semantic_key=str(row.get("semantic_key", "")),
            ibsi_code=str(row.get("ibsi_code", "")),
            consensus=str(row.get("consensus", "")),
            reference_value=_optional_float(row.get("reference_value")),
            tolerance=_optional_float(row.get("tolerance")),
            standardized=_bool(row.get("standardized", False)),
            source_sheet=str(row.get("source_sheet", "")),
            source_row=int(row.get("source_row", 0)),
        )


@dataclass(frozen=True)
class ComparisonRecord:
    specification: str
    phase: str
    adapter: str
    software_version: str
    configuration: str
    profile: str
    aggregation: str
    family: str
    feature_name: str
    feature_tag: str
    semantic_key: str
    ibsi_code: str
    standardized: bool
    observed_supported: bool
    mapped: bool
    attempted: bool
    finite: bool
    referencable: bool
    evaluated: bool
    passed: Optional[bool]
    status: str
    native_feature_names: str = ""
    value: Optional[float] = None
    reference_value: Optional[float] = None
    tolerance: Optional[float] = None
    raw_abs_error: Optional[float] = None
    comparison_error: Optional[float] = None
    error_tolerance_ratio: Optional[float] = None
    comparison_policy: str = ""
    detail: str = ""
    strict_passed: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

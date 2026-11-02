"""Source-specific numerical comparison policies.

The IBSI 1 workbook compares a *difference* truncated toward zero to three
significant digits at the scale of the reference.  It does not round the
calculated feature value before subtraction.  Raw error is retained alongside
the workbook comparison error so both are auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN, localcontext
from typing import Optional


IBSI1_WORKBOOK_POLICY = "ibsi1_workbook_difference_3_significant_digits"
IBSI2_ABSOLUTE_POLICY = "ibsi2_absolute_error"


@dataclass(frozen=True)
class ToleranceResult:
    raw_abs_error: float
    comparison_error: float
    tolerance: float
    passed: bool
    category: str
    policy: str
    error_tolerance_ratio: Optional[float]
    # The workbook truncates the difference to the reference's three
    # significant digits, which suits rounded references.  For a zero
    # reference it truncates to an integer and passes any |value| < 1, so the
    # strict result compares the raw error with the tolerance there.
    strict_passed: bool


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _ratio(error: float, tolerance: float) -> Optional[float]:
    if tolerance > 0.0:
        return error / tolerance
    if error == 0.0:
        return 0.0
    return None


def _category(error: float, tolerance: float) -> str:
    if error <= tolerance:
        return "match"
    if tolerance > 0.0 and error < 3.0 * tolerance:
        return "partial_match"
    return "no_match"


def ibsi1_workbook_error(reference: float, measured: float) -> float:
    """Reproduce the official workbook's ``ROUNDDOWN(reference-result, ...)``.

    Excel's zero-reference fallback truncates the signed difference to zero
    decimal places.  This behavior is preserved exactly and is distinguishable
    from ``raw_abs_error`` in every output row.
    """

    ref = _finite(reference, "reference")
    value = _finite(measured, "measured")
    try:
        ref_decimal = Decimal(str(ref))
        difference = ref_decimal - Decimal(str(value))
        decimal_places = 0
        if ref != 0.0:
            decimal_places = 2 - math.floor(math.log10(abs(ref)))
        quantum = Decimal(1).scaleb(-decimal_places)
        with localcontext() as context:
            context.prec = max(
                34, len(difference.as_tuple().digits) + abs(decimal_places) + 8
            )
            return float(abs(difference.quantize(quantum, rounding=ROUND_DOWN)))
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError("Unable to apply the IBSI 1 workbook comparison rule") from exc


def compare_ibsi1(
    reference: float, measured: float, tolerance: float
) -> ToleranceResult:
    ref = _finite(reference, "reference")
    value = _finite(measured, "measured")
    tol = _finite(tolerance, "tolerance")
    if tol < 0.0:
        raise ValueError("tolerance must be non-negative")
    raw_error = abs(ref - value)
    comparison_error = ibsi1_workbook_error(ref, value)
    return ToleranceResult(
        raw_abs_error=raw_error,
        comparison_error=comparison_error,
        tolerance=tol,
        passed=comparison_error <= tol,
        category=_category(comparison_error, tol),
        policy=IBSI1_WORKBOOK_POLICY,
        error_tolerance_ratio=_ratio(comparison_error, tol),
        strict_passed=(comparison_error <= tol) if ref != 0.0 else (raw_error <= tol),
    )


def compare_absolute(
    reference: float,
    measured: float,
    tolerance: float,
    *,
    policy: str = IBSI2_ABSOLUTE_POLICY,
) -> ToleranceResult:
    ref = _finite(reference, "reference")
    value = _finite(measured, "measured")
    tol = _finite(tolerance, "tolerance")
    if tol < 0.0:
        raise ValueError("tolerance must be non-negative")
    error = abs(ref - value)
    return ToleranceResult(
        raw_abs_error=error,
        comparison_error=error,
        tolerance=tol,
        passed=error <= tol,
        category=_category(error, tol),
        policy=policy,
        error_tolerance_ratio=_ratio(error, tol),
        strict_passed=error <= tol,
    )

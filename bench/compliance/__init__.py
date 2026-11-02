"""IBSI 1/2 compliance reference, evaluation, execution, and reporting tools.

Compliance is intentionally separate from performance benchmarking.  A timing
result never implies numerical conformance, and a mapped feature name never
implies that the installed package actually calculated a finite value.
"""

from bench.compliance.models import ComparisonRecord, ReferenceRecord

__all__ = ["ComparisonRecord", "ReferenceRecord"]

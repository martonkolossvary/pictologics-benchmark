from __future__ import annotations

from bench.compliance.ibsi2_protocol import (
    IBSI2_MANUAL_VERSION,
    IBSI2_PHASE2_BOUNDARY_POLICY,
    PHASE1_FILTER_SPECS,
    PHASE1_FILTER_SPECS_BY_ID,
    PHASE2_FILTER_SPECS,
    PHASE2_FILTER_SPECS_BY_ID,
    validate_phase1_filter_config,
    validate_phase2_filter_config,
)
from bench.compliance.references import (
    IBSI2_PHASE1_STANDARDIZED_IDS,
    IBSI2_PHASE1_TEST_IDS,
    IBSI2_PHASE2_FILTER_IDS,
)


def test_ibsi2_protocol_catalogue_has_exact_defined_surfaces() -> None:
    assert IBSI2_MANUAL_VERSION == "9"
    assert tuple(spec.test_id for spec in PHASE1_FILTER_SPECS) == IBSI2_PHASE1_TEST_IDS
    assert (
        tuple(spec.filter_id for spec in PHASE2_FILTER_SPECS) == IBSI2_PHASE2_FILTER_IDS
    )
    assert len(IBSI2_PHASE1_STANDARDIZED_IDS) == 33
    assert len(PHASE2_FILTER_SPECS) == 22


def test_reviewed_edge_configurations_are_explicit() -> None:
    assert PHASE1_FILTER_SPECS_BY_ID["1.a.4"].parameters == {
        "filter": "mean",
        "dimensionality": 3,
        "boundary": "mirror",
        "support": 15,
    }
    assert PHASE1_FILTER_SPECS_BY_ID["10.b.1"].parameters["boundary"] == "nearest"
    assert PHASE2_FILTER_SPECS_BY_ID["4.A"].parameters["kernels"] == "L5E5"
    assert PHASE2_FILTER_SPECS_BY_ID["4.B"].parameters["kernels"] == "L5E5E5"
    assert PHASE2_FILTER_SPECS_BY_ID["5.A"].parameters["average_over_planes"] is False
    assert PHASE2_FILTER_SPECS_BY_ID["5.B"].parameters["average_over_planes"] is True
    assert PHASE2_FILTER_SPECS_BY_ID["5.B"].parameters["dimensionality"] == 2
    assert PHASE2_FILTER_SPECS_BY_ID["10.A"].parameters["order"] == [0, 2]
    assert PHASE2_FILTER_SPECS_BY_ID["10.B"].parameters["order"] == [0, 2, 0]
    assert "boundary" not in PHASE2_FILTER_SPECS_BY_ID["8.B"].parameters
    assert (
        PHASE2_FILTER_SPECS_BY_ID["8.B"].filter_config()["boundary_policy"]
        == IBSI2_PHASE2_BOUNDARY_POLICY
    )


def test_exact_config_validators_reject_parameter_drift() -> None:
    phase1 = PHASE1_FILTER_SPECS_BY_ID["2.a"].filter_config()
    validate_phase1_filter_config(phase1, test_id="2.a")
    phase1["parameters"]["sigma_mm"] = 2.0
    try:
        validate_phase1_filter_config(phase1, test_id="2.a")
    except ValueError:
        pass
    else:
        raise AssertionError("Phase 1 parameter drift was accepted")

    phase2 = PHASE2_FILTER_SPECS_BY_ID["3.B"].filter_config()
    validate_phase2_filter_config(phase2, filter_id="3.B")
    phase2["parameters"]["truncate"] = 3.0
    try:
        validate_phase2_filter_config(phase2, filter_id="3.B")
    except ValueError:
        pass
    else:
        raise AssertionError("Phase 2 parameter drift was accepted")

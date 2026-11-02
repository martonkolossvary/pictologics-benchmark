from __future__ import annotations

from bench.benchmark_memory import MemoryPreflightPolicy, evaluate_memory_preflight


def test_preflight_uses_static_and_empirical_upper_estimates() -> None:
    policy = MemoryPreflightPolicy(
        reserve_bytes=0,
        budget_fraction=1.0,
        fixed_overhead_bytes=100,
        safety_factor=1.5,
        input_multiplier=2.0,
    )
    prior = [
        {
            "task_status": "measured",
            "adapter": "pictologics",
            "workload": "family:glcm",
            "guardrail_group": "series",
            "input_uncompressed_bytes": 100,
            "worker_ready_rss_bytes": 200,
            "incremental_calculation_peak_rss_bytes": 300,
        }
    ]
    result = evaluate_memory_preflight(
        adapter="pictologics",
        workload_key="family:glcm",
        guardrail_group="series",
        input_uncompressed_bytes=1000,
        available_bytes=10_000,
        total_bytes=10_000,
        prior_records=prior,
        policy=policy,
    )
    assert result["memory_static_estimate_bytes"] == 2100
    assert result["memory_empirical_estimate_bytes"] == 4720
    assert result["memory_estimate_bytes"] == 4720
    assert result["memory_empirical_observation_count"] == 1
    assert result["memory_preflight_decision"] == "launch"


def test_preflight_skips_when_estimate_exceeds_budget() -> None:
    policy = MemoryPreflightPolicy(
        reserve_bytes=0,
        budget_fraction=1.0,
        user_cap_bytes=500,
        fixed_overhead_bytes=100,
        input_multiplier=2.0,
    )
    result = evaluate_memory_preflight(
        adapter="pictologics",
        workload_key="family:glcm",
        guardrail_group="series",
        input_uncompressed_bytes=1000,
        available_bytes=10_000,
        total_bytes=10_000,
        prior_records=[],
        policy=policy,
    )
    assert result["memory_budget_bytes"] == 500
    assert result["memory_preflight_decision"] == "skip"


def test_budget_ignores_momentary_available_memory() -> None:
    policy = MemoryPreflightPolicy(
        reserve_bytes=0,
        budget_fraction=0.8,
        fixed_overhead_bytes=100,
        input_multiplier=2.0,
    )
    budgets = {
        evaluate_memory_preflight(
            adapter="pictologics",
            workload_key="group_texture",
            guardrail_group="series",
            input_uncompressed_bytes=1000,
            available_bytes=available,
            total_bytes=10_000,
            prior_records=[],
            policy=policy,
        )["memory_budget_bytes"]
        for available in (500, 10_000)
    }
    assert budgets == {8000}


def test_ratio_comes_from_the_largest_earlier_input() -> None:
    policy = MemoryPreflightPolicy(
        reserve_bytes=0,
        budget_fraction=1.0,
        fixed_overhead_bytes=0,
        safety_factor=1.0,
        input_multiplier=0.0,
    )
    prior = [
        {
            "task_status": "measured",
            "adapter": "pictologics",
            "workload": "group_texture",
            "guardrail_group": "series",
            "input_uncompressed_bytes": size,
            "worker_ready_rss_bytes": 100,
            "incremental_calculation_peak_rss_bytes": increment,
        }
        # Allocator noise at the small input gives ratio 50; the large gives 2.
        for size, increment in ((10, 500), (1000, 2000))
    ]
    result = evaluate_memory_preflight(
        adapter="pictologics",
        workload_key="group_texture",
        guardrail_group="series",
        input_uncompressed_bytes=10_000,
        available_bytes=10**9,
        total_bytes=10**9,
        prior_records=prior,
        policy=policy,
    )
    assert result["memory_empirical_estimate_bytes"] == 110 + 20_000


def test_a_larger_budget_fraction_takes_effect() -> None:
    budgets = [
        evaluate_memory_preflight(
            adapter="pictologics",
            workload_key="group_texture",
            guardrail_group="series",
            input_uncompressed_bytes=1000,
            available_bytes=100 * 1024**3,
            total_bytes=100 * 1024**3,
            prior_records=[],
            policy=MemoryPreflightPolicy(budget_fraction=fraction),
        )["memory_budget_bytes"]
        for fraction in (0.8, 0.9)
    ]
    assert budgets == [int(0.8 * 100 * 1024**3), int(0.9 * 100 * 1024**3)]

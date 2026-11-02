"""Conservative, auditable memory preflight for isolated benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


GIB = 1024**3


@dataclass(frozen=True)
class MemoryPreflightPolicy:
    enabled: bool = True
    budget_fraction: float = 0.80
    reserve_bytes: int = 4 * GIB
    user_cap_bytes: int | None = None
    safety_factor: float = 1.50
    fixed_overhead_bytes: int = 512 * 1024**2
    # One conservative feasibility multiplier for every adapter, so a guess
    # never skips one package before another.  It is not a fitted peak-RSS
    # claim.  Empirical observations from earlier ranks can only increase the
    # estimate used for later ranks in the same branch.
    input_multiplier: float = 18.0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.budget_fraction) <= 1.0:
            raise ValueError("memory budget fraction must be in (0, 1]")
        if int(self.reserve_bytes) < 0:
            raise ValueError("memory reserve cannot be negative")
        if self.user_cap_bytes is not None and int(self.user_cap_bytes) <= 0:
            raise ValueError("memory cap must be positive")
        if float(self.safety_factor) < 1.0:
            raise ValueError("memory safety factor must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": "total_memory_static_plus_largest_input",
            "enabled": self.enabled,
            "budget_fraction": self.budget_fraction,
            "reserve_bytes": self.reserve_bytes,
            "user_cap_bytes": self.user_cap_bytes,
            "safety_factor": self.safety_factor,
            "fixed_overhead_bytes": self.fixed_overhead_bytes,
            "input_multiplier": self.input_multiplier,
            "empirical_scope": ["adapter", "workload", "guardrail_group"],
            "interpretation": (
                "launch-safety estimate only; never reported as measured memory"
            ),
        }


def evaluate_memory_preflight(
    *,
    adapter: str,
    workload_key: str,
    guardrail_group: str,
    input_uncompressed_bytes: int,
    available_bytes: int,
    total_bytes: int,
    prior_records: Iterable[Mapping[str, Any]],
    policy: MemoryPreflightPolicy,
) -> dict[str, Any]:
    """Return a complete launch/skip decision without mutating run state."""

    if input_uncompressed_bytes <= 0:
        raise ValueError("task input byte count must be positive")
    # The fraction keeps 20 percent free by default; the fixed reserve protects
    # small hosts.  A fixed 20 percent floor here would hide a larger fraction.
    reserve = int(policy.reserve_bytes)
    # The budget depends on total memory only.  A busy moment on the host then
    # cannot skip or stop a task, and a stop cannot skip a branch by chance.
    budget = max(
        0,
        min(
            int(total_bytes) - reserve,
            int(float(policy.budget_fraction) * total_bytes),
        ),
    )
    if policy.user_cap_bytes is not None:
        budget = min(budget, int(policy.user_cap_bytes))

    static_estimate = int(
        policy.fixed_overhead_bytes
        + float(policy.input_multiplier) * int(input_uncompressed_bytes)
    )
    ready_values: list[int] = []
    # (input bytes, incremental bytes) of the largest earlier input.  A tiny
    # input makes allocator noise look like a large per-byte ratio, so only
    # the observation closest in size to this task sets the ratio.
    largest: tuple[int, int] | None = None
    empirical_n = 0
    for record in prior_records:
        if record.get("task_status") != "measured":
            continue
        if record.get("adapter") != adapter or record.get("workload") != workload_key:
            continue
        if record.get("guardrail_group") != guardrail_group:
            continue
        previous_input = record.get("input_uncompressed_bytes")
        ready = record.get("worker_ready_rss_bytes")
        incremental = record.get("incremental_calculation_peak_rss_bytes")
        try:
            previous_input = int(previous_input)
            ready = int(ready)
            incremental = int(incremental)
        except (TypeError, ValueError):
            continue
        if previous_input <= 0 or ready <= 0 or incremental < 0:
            continue
        empirical_n += 1
        ready_values.append(ready)
        if largest is None or previous_input > largest[0]:
            largest = (previous_input, incremental)
        elif previous_input == largest[0]:
            largest = (previous_input, max(largest[1], incremental))

    empirical_estimate = None
    if largest is not None:
        empirical_estimate = int(
            1.10 * max(ready_values)
            + policy.safety_factor
            * (largest[1] / largest[0])
            * int(input_uncompressed_bytes)
        )
    estimate = max(
        static_estimate,
        empirical_estimate if empirical_estimate is not None else 0,
    )
    launch = not policy.enabled or estimate <= budget
    return {
        "memory_preflight_policy_id": "total_memory_static_plus_largest_input",
        "memory_preflight_enabled": policy.enabled,
        "input_uncompressed_bytes": int(input_uncompressed_bytes),
        "memory_static_estimate_bytes": static_estimate,
        "memory_empirical_estimate_bytes": empirical_estimate,
        "memory_empirical_observation_count": empirical_n,
        "memory_estimate_bytes": estimate,
        "memory_available_bytes": int(available_bytes),
        "memory_total_bytes": int(total_bytes),
        "memory_reserve_bytes": reserve,
        "memory_budget_bytes": budget,
        "memory_preflight_decision": "launch" if launch else "skip",
    }


__all__ = ["GIB", "MemoryPreflightPolicy", "evaluate_memory_preflight"]

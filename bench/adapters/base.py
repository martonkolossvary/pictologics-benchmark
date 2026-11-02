from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable


BENCHMARK_EVENT_PREFIX = "BENCH_EVENT "


def write_json(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def write_event(event: str, **details: Any) -> None:
    """Emit one machine-readable worker lifecycle event outside timed regions."""

    payload = {"event": str(event), **details}
    sys.stdout.write(
        BENCHMARK_EVENT_PREFIX
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    sys.stdout.flush()


def _seconds(start_ns: int, stop_ns: int) -> float:
    return (stop_ns - start_ns) / 1_000_000_000.0


def max_rss_bytes() -> int | None:
    """Return this process's peak resident memory, or None without ``resource``."""

    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return peak if sys.platform == "darwin" else peak * 1024


def run_timed_computation(
    compute_fn: Callable[[Any], Any],
    iterations: int = 3,
    no_warmup: bool = False,
    *,
    prepare_fn: Callable[[], Any] | None = None,
    finalize_fn: Callable[[Any, Any], Any] | None = None,
    event_fn: Callable[..., None] | None = None,
):
    """Measure calculation only, with preparation/finalisation outside the clock.

    ``prepare_fn`` may build package-native ROI objects, validate/cast masks,
    reset caches, or otherwise prepare one fresh calculation state.  The
    returned state is passed to ``compute_fn``.  ``finalize_fn`` converts the
    raw result to the adapter payload after the calculation clock has stopped.
    All three stages are measured separately so the primary calculation scope
    remains narrow without discarding operational evidence.
    """

    import numpy as np

    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("iterations must be an integer >= 1")

    prepare = prepare_fn or (lambda: None)
    finalize = finalize_fn or (lambda raw, _state: raw)
    invoke = (
        (lambda state: compute_fn(state))
        if prepare_fn is not None
        else (lambda _state: compute_fn())
    )
    emit = event_fn or (lambda *_args, **_kwargs: None)

    warmup_prepare_sec = None
    warmup_duration_sec = None
    warmup_cpu_time_sec = None
    warmup_finalize_sec = None
    maxrss_before_warmup = max_rss_bytes()
    maxrss_after_warmup = None
    if not no_warmup:
        if prepare_fn is None:
            state = None
            warmup_prepare_sec = 0.0
        else:
            start = time.perf_counter_ns()
            state = prepare()
            warmup_prepare_sec = _seconds(start, time.perf_counter_ns())
        emit("warmup_start", maxrss_bytes=max_rss_bytes())
        sw = time.perf_counter_ns()
        sc = time.process_time_ns()
        raw = invoke(state)
        ew = time.perf_counter_ns()
        ec = time.process_time_ns()
        warmup_duration_sec = _seconds(sw, ew)
        warmup_cpu_time_sec = _seconds(sc, ec)
        maxrss_after_warmup = max_rss_bytes()
        # The warmup call ends here, so finalization runs outside its limit,
        # as it does for the timed calls.
        emit(
            "warmup_complete",
            calculation_sec=warmup_duration_sec,
            preparation_sec=warmup_prepare_sec,
            maxrss_bytes=maxrss_after_warmup,
        )
        if finalize_fn is None:
            _ = raw
            warmup_finalize_sec = 0.0
        else:
            start = time.perf_counter_ns()
            _ = finalize(raw, state)
            warmup_finalize_sec = _seconds(start, time.perf_counter_ns())

    n_measured = iterations

    durations = []
    cpu_times = []
    preparation_times = []
    finalization_times = []

    res = None
    for index in range(n_measured):
        if prepare_fn is None:
            state = None
            preparation_times.append(0.0)
        else:
            start = time.perf_counter_ns()
            state = prepare()
            preparation_times.append(_seconds(start, time.perf_counter_ns()))
        if index == 0:
            emit("worker_ready", measured_iterations=n_measured)
        emit("calculation_start", iteration=index + 1)
        sw = time.perf_counter_ns()
        sc = time.process_time_ns()
        raw = invoke(state)
        ew = time.perf_counter_ns()
        ec = time.process_time_ns()
        durations.append(_seconds(sw, ew))
        cpu_times.append(_seconds(sc, ec))
        emit(
            "calculation_complete",
            iteration=index + 1,
            calculation_sec=durations[-1],
            cpu_time_sec=cpu_times[-1],
            maxrss_bytes=max_rss_bytes(),
        )
        if finalize_fn is None:
            res = raw
            finalization_times.append(0.0)
        else:
            start = time.perf_counter_ns()
            res = finalize(raw, state)
            finalization_times.append(_seconds(start, time.perf_counter_ns()))

    stats = {
        # Median is the primary observation. Minima remain available as a
        # diagnostic but are too optimistic for adapter comparisons.
        "duration_sec": float(np.median(durations)),
        "duration_min_sec": float(np.min(durations)),
        "duration_mean_sec": float(np.mean(durations)),
        "duration_median_sec": float(np.median(durations)),
        "duration_std_sec": float(np.std(durations)) if len(durations) > 1 else 0.0,
        "duration_max_sec": float(np.max(durations)),
        "cpu_time_sec": float(np.median(cpu_times)),
        "cpu_time_min_sec": float(np.min(cpu_times)),
        "cpu_time_mean_sec": float(np.mean(cpu_times)),
        "cpu_time_median_sec": float(np.median(cpu_times)),
        "cpu_time_std_sec": float(np.std(cpu_times)) if len(cpu_times) > 1 else 0.0,
        "cpu_time_max_sec": float(np.max(cpu_times)),
        "measured_iterations": n_measured,
        "warmup_iterations": 0 if no_warmup else 1,
        "total_iterations": n_measured + (0 if no_warmup else 1),
        "warmup_duration_sec": warmup_duration_sec,
        "warmup_cpu_time_sec": warmup_cpu_time_sec,
        "warmup_preparation_sec": warmup_prepare_sec,
        "warmup_finalization_sec": warmup_finalize_sec,
        "duration_samples_sec": [float(value) for value in durations],
        "cpu_time_samples_sec": [float(value) for value in cpu_times],
        "preparation_samples_sec": [float(value) for value in preparation_times],
        "finalization_samples_sec": [float(value) for value in finalization_times],
        # Peak resident memory is a process high-water mark.  The increase
        # over the warmup is the best per-calculation memory estimate.
        "maxrss_before_warmup_bytes": maxrss_before_warmup,
        "maxrss_after_warmup_bytes": maxrss_after_warmup,
        "maxrss_after_measured_bytes": max_rss_bytes(),
    }
    return res, stats

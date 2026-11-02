from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
from importlib import metadata as importlib_metadata
from io import BytesIO
import json
import math
from pathlib import Path
import platform
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from bench.benchmark_ledger import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from bench.ibsi_mapping import classify_feature


BASELINE_ADAPTER = "pictologics"
MEASURED_TASK_STATUS = "measured"

# Every series differs by color, marker, and line pattern. This keeps the
# performance figures interpretable in grayscale and for common color-vision
# deficiencies.
ReportRecordLoader = Callable[
    [Path], pd.DataFrame | tuple[pd.DataFrame, Mapping[str, Any]]
]


def _report_generator_provenance() -> Dict[str, Any]:
    """Describe the code and rendering environment used for this report."""

    bench_root = Path(__file__).resolve().parent
    sources = {
        path.relative_to(bench_root).as_posix(): sha256_file(path)
        for path in sorted(bench_root.rglob("*.py"))
    }
    source_payload = json.dumps(
        sources,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dependencies = {
        "matplotlib": str(matplotlib.__version__),
        "numpy": str(np.__version__),
        "openpyxl": importlib_metadata.version("openpyxl"),
        "pandas": str(pd.__version__),
    }
    provenance: Dict[str, Any] = {
        "schema_version": 1,
        "source_tree_sha256": hashlib.sha256(source_payload).hexdigest(),
        "python_version": platform.python_version(),
        "dependencies": dependencies,
    }
    provenance_payload = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provenance["provenance_sha256"] = hashlib.sha256(provenance_payload).hexdigest()
    return provenance


def _atomic_write_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write_text(path, frame.to_csv(index=False))


def _truthy_series(series: pd.Series) -> pd.Series:
    """Interpret common CSV boolean encodings."""

    return series.map(
        lambda value: (
            value is True or str(value).strip().lower() in {"1", "true", "yes"}
        )
    )


def timing_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid, uncensored timing observations.

    ``task_status`` is authoritative. Timeout bounds, failures, unsupported
    tasks, and policy skips are never converted into measured durations.
    """

    observations = df.copy()
    if "task_status" not in observations.columns:
        return observations.iloc[0:0].copy()
    status = observations["task_status"].fillna("").astype(str).str.strip().str.lower()
    observations = observations.loc[status == MEASURED_TASK_STATUS].copy()

    if "dropped" in observations.columns:
        observations = observations.loc[~_truthy_series(observations["dropped"])].copy()

    if "duration_sec" not in observations.columns:
        return observations.iloc[0:0].copy()

    observations["duration_sec"] = pd.to_numeric(
        observations["duration_sec"], errors="coerce"
    )
    valid = np.isfinite(observations["duration_sec"]) & (
        observations["duration_sec"] > 0.0
    )
    return observations.loc[valid].copy()


def eligible_records(df: pd.DataFrame) -> pd.DataFrame:
    """Return the scheduled tasks that can give a time: all but unsupported and dropped."""

    frame = df.copy()
    if "task_status" in frame.columns:
        status = frame["task_status"].fillna("").astype(str).str.strip().str.lower()
        frame = frame.loc[status != "unsupported"].copy()
    if "dropped" in frame.columns:
        frame = frame.loc[~_truthy_series(frame["dropped"])].copy()
    return frame


def with_report_workload(df: pd.DataFrame) -> pd.DataFrame:
    """Attach one workload key without pooling distinct feature families."""

    out = df.copy()
    group = pd.Series("all", index=out.index, dtype="object")
    source = pd.Series("default", index=out.index, dtype="object")

    # A more specific field overrides a broader one.
    for column in ("category", "family", "workload"):
        if column not in out.columns:
            continue
        values = out[column].fillna("").astype(str).str.strip()
        present = values.ne("") & values.str.lower().ne("nan")
        group.loc[present] = values.loc[present]
        source.loc[present] = column

    out["_report_workload"] = group
    out["_report_grouping_dimension"] = source
    return out


def _normalise_dataset_kind(value: object) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"synthetic", "synthetic_data"}:
        return "synthetic"
    if token in {"real", "real_world", "realworld", "validation"}:
        return "real_world"
    return token or "unknown"


def _resolve_dataset_kind(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        return _normalise_dataset_kind(explicit)

    for column in ("_dataset_kind", "dataset_kind"):
        if column not in df.columns:
            continue
        values = {
            _normalise_dataset_kind(value)
            for value in df[column].dropna().tolist()
            if str(value).strip()
        }
        if len(values) > 1:
            raise ValueError(
                f"Report input mixes dataset kinds in {column}: {sorted(values)}"
            )
        if values:
            return next(iter(values))

    if "modality" in df.columns:
        modalities = {
            str(value).strip().lower()
            for value in df["modality"].dropna().tolist()
            if str(value).strip()
        }
        if modalities == {"synthetic"}:
            return "synthetic"
        if modalities and "synthetic" not in modalities:
            return "real_world"
    return "unknown"


def _ledger_record_loader(
    input_dir: Path,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Load authoritative records and verify every committed measured payload."""

    from bench.benchmark_ledger import BenchmarkLedger, RunIntegrityError
    from bench.benchmark_models import fingerprint, run_spec_identity

    ledger_path = input_dir / "benchmark.sqlite3"
    with BenchmarkLedger(ledger_path) as ledger:
        records, payloads = ledger.verified_records_and_payloads()
        status_counts = ledger.status_counts()
        run_status = ledger.metadata_value("run_status")
        run_fingerprint = ledger.metadata_value("run_fingerprint")
        run_spec_json = ledger.metadata_value("run_spec_json")

    if not run_spec_json:
        raise RunIntegrityError("benchmark ledger has no run_spec_json")
    try:
        decoded = json.loads(run_spec_json)
    except json.JSONDecodeError as exc:
        raise RunIntegrityError(
            "benchmark ledger contains malformed run_spec_json"
        ) from exc
    if not isinstance(decoded, dict):
        raise RunIntegrityError("benchmark ledger run_spec_json must contain an object")
    run_spec: Dict[str, Any] = decoded
    # The runner fingerprints the run identity, which leaves out the
    # append-only repeat horizon and task-plan hash.
    if (
        not run_fingerprint
        or fingerprint(run_spec_identity(run_spec)) != run_fingerprint
    ):
        raise RunIntegrityError(
            "benchmark ledger run specification differs from its stored fingerprint"
        )

    unfinished_statuses = {"pending", "running", "interrupted"}
    unfinished_n = sum(
        count
        for status, count in status_counts.items()
        if status in unfinished_statuses
    )
    execution_complete = (
        run_status in {"completed", "completed_with_failures"} and unfinished_n == 0
    )
    return pd.DataFrame(records), {
        "record_source": str(ledger_path),
        "source_attested": True,
        "verified_payloads": payloads,
        "verified_payload_count": len(payloads),
        "authoritative_record_count": len(records),
        "status_counts": status_counts,
        "task_count": sum(status_counts.values()),
        "unfinished_task_count": unfinished_n,
        "run_status": run_status or "unknown",
        "execution_complete": execution_complete,
        "run_fingerprint": run_fingerprint,
        "run_spec": run_spec,
    }


def _default_record_loader(
    input_dir: Path,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    ledger_path = input_dir / "benchmark.sqlite3"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"run ledger not found: {ledger_path}")
    return _ledger_record_loader(input_dir)


def load_report_records(
    input_dir: Path | str,
    *,
    record_loader: ReportRecordLoader | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load report rows through an explicit input trust boundary.

    The default loader reads ``benchmark.sqlite3`` and verifies committed
    measured payloads. A custom loader can be injected by
    returning either a DataFrame or ``(DataFrame, metadata)`` and must set
    ``source_attested=True`` only after verifying its own trust boundary.
    """

    directory = Path(input_dir)
    loader = record_loader or _default_record_loader
    loaded = loader(directory)
    loader_metadata: Dict[str, Any] = {}
    if isinstance(loaded, tuple):
        frame, raw_metadata = loaded
        loader_metadata.update(dict(raw_metadata))
    else:
        frame = loaded

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Report record loader must return a pandas DataFrame")

    metadata: Dict[str, Any] = {
        "record_source": (
            str(directory / "summary.csv")
            if record_loader is None
            else getattr(loader, "__name__", loader.__class__.__name__)
        ),
        "source_attested": False,
    }
    metadata.update(loader_metadata)

    run_spec = metadata.get("run_spec")
    if not isinstance(run_spec, dict) or not run_spec:
        run_spec_path = directory / "run_spec.json"
        run_spec = None
        if run_spec_path.is_file():
            try:
                run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid run specification: {run_spec_path}") from exc
            if not isinstance(run_spec, dict):
                raise ValueError(
                    f"Run specification must be a JSON object: {run_spec_path}"
                )
            metadata["run_spec"] = run_spec

    if isinstance(run_spec, dict) and run_spec:
        metadata["run_id"] = run_spec.get("run_id")
        metadata["dataset_kind"] = _normalise_dataset_kind(run_spec.get("dataset_kind"))
        metadata["dataset"] = run_spec.get("dataset")
    else:
        metadata["dataset_kind"] = _resolve_dataset_kind(frame)

    out = frame.copy()
    out["_dataset_kind"] = metadata["dataset_kind"]
    return out, metadata


def _baseline_adapter_from_metadata(metadata: Mapping[str, Any]) -> str:
    run_spec = metadata.get("run_spec")
    if isinstance(run_spec, dict):
        guardrail = run_spec.get("guardrail")
        if isinstance(guardrail, dict):
            baseline = str(guardrail.get("baseline_adapter") or "").strip()
            if baseline:
                return baseline
    return BASELINE_ADAPTER


def _metadata_cell(value: Any) -> Any:
    """Return a stable, spreadsheet-safe representation of protocol metadata."""

    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return value


def _markdown_cell(value: Any) -> str:
    return (
        str(_metadata_cell(value))
        .replace("|", "\\|")
        .replace("\r", "")
        .replace("\n", "<br>")
    )


def _protocol_summary(run_spec: Any) -> pd.DataFrame:
    """Flatten publication-relevant RunSpec fields without local file paths."""

    columns = ["field", "value"]
    if not isinstance(run_spec, Mapping):
        return pd.DataFrame(columns=columns)

    fields = (
        "schema_version",
        "run_id",
        "dataset",
        "dataset_kind",
        "dataset_manifest_schema_version",
        "manifest_sha256",
        "dataset_hashes_verified",
        "dataset_values_inspected",
        "selected_case_ids",
        "adapters",
        "workloads",
        "repeats",
        "aggregation",
        "discretization",
        "bins",
        "bin_width",
        "intensity_min",
        "intensity_max",
        "timing_iterations",
        "family_timings",
        "capture_values",
        "drop_first_repeat",
        "no_warmup",
        "timeout_seconds",
        "keep_going",
        "task_plan_sha256",
        "runtime_profiles_sha256",
        "benchmark_sources_sha256",
        "thread_policy",
        "initialization_policy",
        "guardrail",
        "memory_preflight",
        "endpoint_contract_id",
        "endpoint_contract_sha256",
    )
    rows = [
        {"field": field, "value": _metadata_cell(run_spec.get(field))}
        for field in fields
        if field in run_spec
    ]
    machine = run_spec.get("benchmark_machine")
    if isinstance(machine, Mapping):
        rows.extend(
            {
                "field": f"machine:{field}",
                "value": _metadata_cell(value),
            }
            for field, value in sorted(machine.items())
            if field != "hostname"
        )
    return pd.DataFrame(rows, columns=columns)


def _adapter_environment_summary(run_spec: Any) -> pd.DataFrame:
    """Expose reviewed release and numerical-runtime provenance per adapter."""

    columns = [
        "adapter",
        "distribution",
        "configured_release_version",
        "distribution_metadata_version",
        "python_version",
        "profile_python",
        "numpy_version",
        "installed_package_count",
        "installed_packages_sha256",
        "environment_freeze_sha256",
        "environment_metadata_sha256",
        "numpy_build_configuration",
    ]
    if not isinstance(run_spec, Mapping):
        return pd.DataFrame(columns=columns)
    environments = run_spec.get("adapter_environments")
    if not isinstance(environments, Mapping):
        return pd.DataFrame(columns=columns)

    configured_order = run_spec.get("adapters")
    adapter_names = (
        [str(value) for value in configured_order]
        if isinstance(configured_order, list)
        else sorted(str(value) for value in environments)
    )
    rows: list[Dict[str, Any]] = []
    for adapter in adapter_names:
        snapshot = environments.get(adapter)
        if not isinstance(snapshot, Mapping):
            continue
        packages = snapshot.get("packages")
        package_list = packages if isinstance(packages, list) else []
        package_payload = json.dumps(
            package_list,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        numpy_config = snapshot.get("numpy_config")
        if not isinstance(numpy_config, Mapping):
            numpy_config = {}
        rows.append(
            {
                "adapter": adapter,
                "distribution": snapshot.get("distribution"),
                "configured_release_version": snapshot.get(
                    "configured_release_version"
                ),
                "distribution_metadata_version": snapshot.get("distribution_version"),
                "python_version": snapshot.get("python_version"),
                "profile_python": snapshot.get("profile_python"),
                "numpy_version": numpy_config.get("version"),
                "installed_package_count": len(package_list),
                "installed_packages_sha256": hashlib.sha256(
                    package_payload
                ).hexdigest(),
                "environment_freeze_sha256": snapshot.get("environment_freeze_sha256"),
                "environment_metadata_sha256": snapshot.get(
                    "environment_metadata_sha256"
                ),
                "numpy_build_configuration": numpy_config.get("show_config"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def non_timing_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Return terminal outcomes excluded from performance estimates."""

    outcomes = df.copy()
    if "task_status" not in outcomes.columns:
        return outcomes.iloc[0:0].copy()
    status = outcomes["task_status"].fillna("").astype(str).str.strip().str.lower()
    outcomes = outcomes.loc[status != MEASURED_TASK_STATUS].copy()
    outcomes["_report_status"] = status.loc[outcomes.index].replace("", "unknown")
    return with_report_workload(outcomes)


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{context} is missing required column(s): {', '.join(missing)}"
        )


def _canonical_triplet(
    value: object,
    *,
    integer: bool,
    field: str,
    context: str,
) -> str:
    """Return a stable JSON triplet for grouping ledger and CSV records."""

    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} has an invalid {field} triplet: {value!r}"
            ) from exc
    if isinstance(candidate, np.ndarray):
        candidate = candidate.tolist()
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
        raise ValueError(f"{context} requires {field} as a three-element sequence")

    normalized: list[int | float] = []
    for item in candidate:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{context} has a non-numeric {field} value")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{context} requires positive finite {field} values")
        if integer:
            if not numeric.is_integer():
                raise ValueError(f"{context} requires integer {field} values")
            normalized.append(int(numeric))
        else:
            normalized.append(numeric)
    return json.dumps(normalized, separators=(",", ":"))


def _normalise_real_world_geometry(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Validate and canonicalise the case geometry used for report grouping."""

    required = ["case_id", "modality", "shape", "spacing", "image_voxels"]
    _require_columns(frame, required, context)
    normalized = frame.copy()
    normalized["shape"] = normalized["shape"].map(
        lambda value: _canonical_triplet(
            value,
            integer=True,
            field="shape",
            context=context,
        )
    )
    normalized["spacing"] = normalized["spacing"].map(
        lambda value: _canonical_triplet(
            value,
            integer=False,
            field="spacing",
            context=context,
        )
    )
    case_ids = normalized["case_id"].fillna("").astype(str).str.strip()
    modalities = normalized["modality"].fillna("").astype(str).str.strip()
    if case_ids.eq("").any() or modalities.eq("").any():
        raise ValueError(f"{context} requires non-empty case_id and modality")
    normalized["case_id"] = case_ids
    normalized["modality"] = modalities

    voxels = pd.to_numeric(normalized["image_voxels"], errors="coerce")
    valid_voxels = np.isfinite(voxels) & (voxels > 0) & (voxels % 1 == 0)
    if not valid_voxels.all():
        raise ValueError(f"{context} requires positive integer image_voxels")
    normalized["image_voxels"] = voxels.astype(np.int64)
    shape_products = normalized["shape"].map(lambda value: math.prod(json.loads(value)))
    if not shape_products.eq(normalized["image_voxels"]).all():
        raise ValueError(f"{context} shape does not match image_voxels")

    geometry_columns = [
        "case_id",
        "modality",
        "shape",
        "spacing",
        "image_voxels",
    ]
    geometry_variants = (
        normalized[geometry_columns]
        .drop_duplicates()
        .groupby("case_id", dropna=False)
        .size()
    )
    if (geometry_variants > 1).any():
        raise ValueError(
            f"{context} contains inconsistent geometry for the same case_id"
        )
    return normalized


def _performance_group_columns(
    observations: pd.DataFrame, dataset_kind: str
) -> list[str]:
    columns = ["_report_workload", "adapter"]
    if dataset_kind == "synthetic":
        _require_columns(observations, ["size"], "Synthetic timing report")
        # Pillar 1 subjects are the noise levels; never pool them.
        if (
            "subject_id" in observations.columns
            and observations["subject_id"].notna().any()
        ):
            columns.append("subject_id")
        columns.append("size")
        if "mask_id" in observations.columns:
            columns.append("mask_id")
    elif dataset_kind == "real_world":
        _require_columns(
            observations,
            ["case_id", "modality", "shape", "spacing", "image_voxels"],
            "Real-world timing report",
        )
        columns.extend(["case_id", "modality", "shape", "spacing", "image_voxels"])
    elif "size" in observations.columns:
        columns.append("size")
    elif "case_id" in observations.columns:
        columns.append("case_id")
    return columns


def aggregate_timing_observations(
    df: pd.DataFrame, *, dataset_kind: str | None = None
) -> pd.DataFrame:
    """Summarize absolute runtime per cell, only when every repeat was measured.

    ``df`` holds all task records.  A cell with a censored, failed or skipped
    repeat keeps its counts but gets no runtime or memory summary, because the
    measured repeats alone would give a too-small value.  Synthetic rows
    retain cube edge length and mask. Real-world rows retain modality and the
    true image voxel count; real-world groupings never use a cube edge length.
    """

    observations = with_report_workload(timing_observations(df))
    if observations.empty:
        return pd.DataFrame()
    _require_columns(observations, ["adapter"], "Timing report")

    kind = _resolve_dataset_kind(observations, dataset_kind)
    eligible = with_report_workload(eligible_records(df))
    if kind == "real_world":
        observations = _normalise_real_world_geometry(
            observations, "Real-world timing report"
        )
        eligible = _normalise_real_world_geometry(eligible, "Real-world timing report")
    group_columns = _performance_group_columns(observations, kind)

    memory_columns = (
        "peak_rss_bytes",
        "worker_ready_rss_bytes",
        "calculation_peak_rss_bytes",
        "incremental_calculation_peak_rss_bytes",
    )
    for column in memory_columns:
        if column not in observations.columns:
            observations[column] = np.nan
        observations[column] = pd.to_numeric(observations[column], errors="coerce")

    grouped = (
        observations.groupby(group_columns, dropna=False)
        .agg(
            grouping_dimension=("_report_grouping_dimension", "first"),
            duration_steady_sec=("duration_sec", "median"),
            duration_q1_sec=("duration_sec", lambda values: values.quantile(0.25)),
            duration_q3_sec=("duration_sec", lambda values: values.quantile(0.75)),
            duration_mean_sec=("duration_sec", "mean"),
            duration_std_sec=("duration_sec", "std"),
            peak_rss_mb=(
                "peak_rss_bytes",
                lambda values: (
                    float(values.max()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_rss_mb=(
                "peak_rss_bytes",
                lambda values: (
                    float(values.median()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            calculation_peak_rss_mb=(
                "calculation_peak_rss_bytes",
                lambda values: (
                    float(values.max()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_calculation_peak_rss_mb=(
                "calculation_peak_rss_bytes",
                lambda values: (
                    float(values.median()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_worker_ready_rss_mb=(
                "worker_ready_rss_bytes",
                lambda values: (
                    float(values.median()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            median_incremental_calculation_rss_mb=(
                "incremental_calculation_peak_rss_bytes",
                lambda values: (
                    float(values.median()) / (1024.0 * 1024.0)
                    if values.notna().any()
                    else np.nan
                ),
            ),
            timing_observations=("duration_sec", "count"),
        )
        .reset_index()
    )
    planned = (
        eligible.groupby(group_columns, dropna=False)
        .size()
        .rename("planned_repeats")
        .reset_index()
    )
    grouped = grouped.merge(planned, on=group_columns, how="left").rename(
        columns={"_report_workload": "workload"}
    )
    grouped["complete_repeats"] = grouped["timing_observations"].eq(
        grouped["planned_repeats"]
    )
    summary_columns = [
        column
        for column in grouped.columns
        if column.startswith(("duration_", "median_")) or column.endswith("_mb")
    ]
    grouped.loc[~grouped["complete_repeats"], summary_columns] = np.nan
    grouped.insert(0, "dataset_kind", kind)
    return grouped


def feature_workload_contract(records: pd.DataFrame) -> pd.DataFrame:
    """Summarise planned and observed family-level feature denominators.

    Runtime is deliberately not divided by either denominator.  The table
    instead exposes the exact frozen native-output count beside task coverage
    so readers can interpret unequal package workloads without inventing a
    linear per-feature cost model.
    """

    columns = [
        "adapter",
        "workload",
        "expected_native_outputs",
        "observed_native_outputs_min",
        "observed_native_outputs_max",
        "planned_tasks",
        "measured_tasks",
        "terminal_tasks",
        "measured_fraction_of_planned",
        "runtime_normalization",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)
    frame = with_report_workload(records.copy())
    _require_columns(frame, ["adapter", "_report_workload"], "Feature workload")
    for field in ("expected_feature_count", "feature_count"):
        if field not in frame.columns:
            frame[field] = np.nan
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if "task_status" not in frame.columns:
        success = (
            _truthy_series(frame["success"])
            if "success" in frame.columns
            else pd.Series(False, index=frame.index)
        )
        frame["task_status"] = np.where(success, MEASURED_TASK_STATUS, "unknown")

    rows: list[dict[str, Any]] = []
    for (adapter, workload), group in frame.groupby(
        ["adapter", "_report_workload"], dropna=False, sort=True
    ):
        expected_values = group["expected_feature_count"].dropna().unique()
        if len(expected_values) > 1:
            raise ValueError(
                f"Feature workload has inconsistent frozen counts for "
                f"{adapter}/{workload}"
            )
        measured = group.loc[group["task_status"] == MEASURED_TASK_STATUS]
        observed = measured["feature_count"].dropna()
        terminal = group.loc[
            ~group["task_status"].isin(["pending", "running", "interrupted"])
        ]
        planned_n = int(len(group))
        measured_n = int(len(measured))
        rows.append(
            {
                "adapter": adapter,
                "workload": workload,
                "expected_native_outputs": (
                    int(expected_values[0]) if len(expected_values) else np.nan
                ),
                "observed_native_outputs_min": (
                    int(observed.min()) if not observed.empty else np.nan
                ),
                "observed_native_outputs_max": (
                    int(observed.max()) if not observed.empty else np.nan
                ),
                "planned_tasks": planned_n,
                "measured_tasks": measured_n,
                "terminal_tasks": int(len(terminal)),
                "measured_fraction_of_planned": (
                    measured_n / planned_n if planned_n else np.nan
                ),
                "runtime_normalization": "none",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _duplicate_key_message(frame: pd.DataFrame, keys: list[str], label: str) -> str:
    duplicate = frame.loc[frame.duplicated(keys, keep=False), keys].head(5)
    examples = duplicate.astype(str).agg(" | ".join, axis=1).tolist()
    return (
        f"{label} has duplicate measured rows for exact comparison key "
        f"{tuple(keys)}; examples: {examples}"
    )


def matched_runtime_observations(
    df: pd.DataFrame,
    *,
    baseline_adapter: str = BASELINE_ADAPTER,
) -> pd.DataFrame:
    """Create exact candidate/baseline runtime pairs.

    Rows are matched only on ``(case_id, workload, repeat)`` after filtering
    to measured, non-dropped observations. The ratio is candidate duration
    divided by the configured baseline duration; values above one mean the
    baseline ran faster. Unmatched or censored rows never enter the result.
    Duplicate comparison identities are rejected instead of being multiplied
    by a many-to-many merge.
    """

    observations = with_report_workload(timing_observations(df))
    match_keys = ["case_id", "_report_workload", "repeat"]
    required = [*match_keys, "adapter", "duration_sec"]
    _require_columns(observations, required, "Matched runtime report")

    baseline_name = baseline_adapter.strip().lower()
    adapter_key = observations["adapter"].fillna("").astype(str).str.strip().str.lower()
    baseline = observations.loc[adapter_key == baseline_name].copy()
    candidates = observations.loc[adapter_key != baseline_name].copy()

    if baseline.duplicated(match_keys, keep=False).any():
        raise ValueError(
            _duplicate_key_message(baseline, match_keys, "Baseline adapter")
        )
    candidate_keys = [*match_keys, "adapter"]
    if candidates.duplicated(candidate_keys, keep=False).any():
        raise ValueError(
            _duplicate_key_message(candidates, candidate_keys, "Candidate adapter")
        )

    output_columns = [
        "case_id",
        "workload",
        "repeat",
        "adapter",
        "candidate_duration_sec",
        "baseline_duration_sec",
        "runtime_ratio",
        "baseline_adapter",
        "dataset",
        "modality",
        "subject_id",
        "size",
        "mask_id",
        "mask_label",
        "shape",
        "spacing",
        "image_voxels",
        "mask_voxels",
        "_dataset_kind",
        "candidate_task_id",
        "baseline_task_id",
    ]
    if baseline.empty or candidates.empty:
        return pd.DataFrame(columns=output_columns)

    baseline_columns = [*match_keys, "duration_sec"]
    if "task_id" in baseline.columns:
        baseline_columns.append("task_id")
    baseline_for_join = baseline[baseline_columns].rename(
        columns={
            "duration_sec": "baseline_duration_sec",
            "task_id": "baseline_task_id",
        }
    )

    candidate_frame = candidates.copy()
    candidate_frame = candidate_frame.rename(
        columns={
            "duration_sec": "candidate_duration_sec",
            "task_id": "candidate_task_id",
        }
    )
    matched = candidate_frame.merge(
        baseline_for_join,
        on=match_keys,
        how="inner",
        validate="many_to_one",
    )
    matched["runtime_ratio"] = (
        matched["candidate_duration_sec"] / matched["baseline_duration_sec"]
    )
    matched["baseline_adapter"] = baseline_adapter
    # ``workload`` may already exist in modern summaries. Overwrite it with the
    # normalized report key instead of creating a duplicate column name.
    matched["workload"] = matched["_report_workload"]

    for column in output_columns:
        if column not in matched.columns:
            matched[column] = np.nan
    return (
        matched[output_columns]
        .sort_values(
            ["workload", "adapter", "case_id", "repeat"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def matched_runtime_summary(
    matched: pd.DataFrame,
    *,
    dataset_kind: str | None = None,
    records: pd.DataFrame | None = None,
    baseline_adapter: str = BASELINE_ADAPTER,
) -> pd.DataFrame:
    """Summarize exact runtime pairs and expose the matched sample size.

    With ``records``, a cell also counts its planned pairs: the repeats where
    both adapters were scheduled.  A cell with a missing pair (a censored,
    failed or skipped task of either adapter) keeps its counts but gets no
    ratio summary, because the remaining pairs alone are a biased sample.
    """

    if matched.empty:
        kind = _normalise_dataset_kind(dataset_kind)
        dimensions = ["subject_id", "size", "mask_id"] if kind == "synthetic" else []
        if kind == "real_world":
            dimensions = [
                "case_id",
                "modality",
                "shape",
                "spacing",
                "image_voxels",
            ]
        return pd.DataFrame(
            columns=[
                "dataset_kind",
                "workload",
                "adapter",
                "baseline_adapter",
                *dimensions,
                "matched_n",
                "runtime_ratio_median",
                "runtime_ratio_q1",
                "runtime_ratio_q3",
                "candidate_duration_median_sec",
                "baseline_duration_median_sec",
                "ratio_definition",
            ]
        )
    required = [
        "workload",
        "adapter",
        "candidate_duration_sec",
        "baseline_duration_sec",
        "runtime_ratio",
        "baseline_adapter",
    ]
    _require_columns(matched, required, "Matched runtime summary")
    kind = _resolve_dataset_kind(matched, dataset_kind)
    if kind == "real_world":
        matched = _normalise_real_world_geometry(
            matched, "Real-world matched runtime summary"
        )

    group_columns = ["workload", "adapter", "baseline_adapter"]
    if kind == "synthetic":
        _require_columns(matched, ["size"], "Synthetic matched runtime summary")
        # Pillar 1 subjects are the noise levels; never pool them.
        if "subject_id" in matched.columns and matched["subject_id"].notna().any():
            group_columns.append("subject_id")
        group_columns.append("size")
        if "mask_id" in matched.columns:
            group_columns.append("mask_id")
    elif kind == "real_world":
        _require_columns(
            matched,
            ["case_id", "modality", "shape", "spacing", "image_voxels"],
            "Real-world matched runtime summary",
        )
        group_columns.extend(
            ["case_id", "modality", "shape", "spacing", "image_voxels"]
        )
    elif "size" in matched.columns:
        group_columns.append("size")

    summary = (
        matched.groupby(group_columns, dropna=False)
        .agg(
            matched_n=("runtime_ratio", "count"),
            runtime_ratio_median=("runtime_ratio", "median"),
            runtime_ratio_q1=(
                "runtime_ratio",
                lambda values: values.quantile(0.25),
            ),
            runtime_ratio_q3=(
                "runtime_ratio",
                lambda values: values.quantile(0.75),
            ),
            candidate_duration_median_sec=(
                "candidate_duration_sec",
                "median",
            ),
            baseline_duration_median_sec=(
                "baseline_duration_sec",
                "median",
            ),
        )
        .reset_index()
    )
    if records is not None:
        eligible = with_report_workload(eligible_records(records))
        eligible["workload"] = eligible["_report_workload"]
        if kind == "real_world":
            eligible = _normalise_real_world_geometry(
                eligible, "Real-world matched runtime summary"
            )
        keys = ["case_id", "workload", "repeat"]
        is_baseline = (
            eligible["adapter"].astype(str).str.strip().str.lower()
            == baseline_adapter.strip().lower()
        )
        pairs = eligible.loc[~is_baseline].merge(
            eligible.loc[is_baseline, keys].drop_duplicates(), on=keys, how="inner"
        )
        pairs["baseline_adapter"] = baseline_adapter
        planned = (
            pairs.groupby(group_columns, dropna=False)
            .size()
            .rename("planned_pairs")
            .reset_index()
        )
        summary = summary.merge(planned, on=group_columns, how="left")
        summary["complete_pairs"] = summary["matched_n"].eq(summary["planned_pairs"])
        summary.loc[
            ~summary["complete_pairs"],
            [
                "runtime_ratio_median",
                "runtime_ratio_q1",
                "runtime_ratio_q3",
                "candidate_duration_median_sec",
                "baseline_duration_median_sec",
            ],
        ] = np.nan
    summary.insert(0, "dataset_kind", kind)
    summary["ratio_definition"] = "candidate_duration / baseline_duration"
    return summary


def generate_plots(
    records: pd.DataFrame,
    report_dir: Path,
    *,
    logical_cpus: int | None = None,
) -> list[Dict[str, str]]:
    """Draw the timing figures of ``bench.figures.performance`` for this run."""

    from bench.figures.performance import figures_from_records, prepare_records

    figures = figures_from_records(
        prepare_records(records), report_dir, logical_cpu_count=logical_cpus
    )
    return [
        {
            "artifact": f"{figure['figure_id']}.pdf",
            "format": "PDF, SVG and PNG",
            "alt_text": figure["alt_text"],
        }
        for figure in figures
    ]


def generate_excel_report(
    timing_summary: pd.DataFrame,
    matched: pd.DataFrame,
    matched_summary: pd.DataFrame,
    feature_contract: pd.DataFrame,
    outcomes: pd.DataFrame,
    run_metadata: Mapping[str, Any],
    report_dir: Path,
) -> Path:
    """Write the minimal auditable performance workbook."""

    path = report_dir / "academic_performance_report.xlsx"
    metadata_rows = [
        {"field": "run_id", "value": run_metadata.get("run_id")},
        {"field": "run_status", "value": run_metadata.get("run_status")},
        {
            "field": "execution_complete",
            "value": bool(run_metadata.get("execution_complete")),
        },
        {"field": "task_count", "value": run_metadata.get("task_count")},
        {
            "field": "unfinished_task_count",
            "value": run_metadata.get("unfinished_task_count"),
        },
        {
            "field": "verified_payload_count",
            "value": run_metadata.get("verified_payload_count"),
        },
        {"field": "record_source", "value": run_metadata.get("record_source")},
        {
            "field": "source_attested",
            "value": bool(run_metadata.get("source_attested")),
        },
        {
            "field": "publication_attested",
            "value": bool(run_metadata.get("publication_attested")),
        },
        {
            "field": "run_fingerprint",
            "value": run_metadata.get("run_fingerprint"),
        },
    ]
    report_generator = run_metadata.get("report_generator")
    if isinstance(report_generator, Mapping):
        metadata_rows.extend(
            [
                {
                    "field": "report_generator:provenance_sha256",
                    "value": report_generator.get("provenance_sha256"),
                },
                {
                    "field": "report_generator:source_tree_sha256",
                    "value": report_generator.get("source_tree_sha256"),
                },
                {
                    "field": "report_generator:python_version",
                    "value": report_generator.get("python_version"),
                },
            ]
        )
        dependencies = report_generator.get("dependencies")
        if isinstance(dependencies, Mapping):
            metadata_rows.extend(
                {
                    "field": f"report_generator:dependency:{name}",
                    "value": version,
                }
                for name, version in sorted(dependencies.items())
            )
    for status, count in sorted(dict(run_metadata.get("status_counts") or {}).items()):
        metadata_rows.append({"field": f"status_count:{status}", "value": count})
    protocol = _protocol_summary(run_metadata.get("run_spec"))
    environments = _adapter_environment_summary(run_metadata.get("run_spec"))
    sheets = {
        "Run metadata": pd.DataFrame(metadata_rows),
        "Protocol": protocol,
        "Adapter environments": environments,
        "Measured runtime": timing_summary,
        "Feature workload": feature_contract,
        "Matched pairs": matched,
        "Matched summary": matched_summary,
        "Excluded outcomes": outcomes,
    }
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [
                    "" if cell.value is None else str(cell.value)
                    for cell in column_cells[:200]
                ]
                width = min(48, max(11, max(map(len, values), default=0) + 2))
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    atomic_write_bytes(path, buffer.getvalue())
    return path


def _denominator_names(payload: Mapping[str, Any], name: str) -> Optional[set[str]]:
    containers = [payload.get("feature_denominators")]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        containers.append(metadata.get("feature_denominators"))
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get(name)
        if isinstance(value, list):
            return {str(item) for item in value}
        if isinstance(value, dict):
            return {str(item) for item in value}
    return None


def _summarize_adapter_payloads(
    adapter: str, payloads: list[Mapping[str, Any]]
) -> Dict[str, Any]:
    attempted: set[str] = set()
    finite: set[str] = set()
    supported: set[str] = set()
    referenced: set[str] = set()
    passing: set[str] = set()
    finite_observed = False
    supported_observed = False
    referenced_observed = False
    passing_observed = False

    for payload in payloads:
        benchmark = payload.get("benchmark")
        if isinstance(benchmark, dict) and benchmark.get("status") not in {
            None,
            MEASURED_TASK_STATUS,
        }:
            continue
        features = payload.get("features")
        if isinstance(features, dict):
            for feature in features.get("all", []):
                attempted.add(str(feature))

        values_container = payload.get("values")
        values = (
            values_container.get("all") if isinstance(values_container, dict) else None
        )
        if isinstance(values, dict):
            finite_observed = True
            for feature_name, value in values.items():
                try:
                    if math.isfinite(float(value)):
                        finite.add(str(feature_name))
                except (TypeError, ValueError):
                    continue

        for name, target in (
            ("supported", supported),
            ("referenced", referenced),
            ("passing", passing),
        ):
            names = _denominator_names(payload, name)
            if names is None:
                continue
            target.update(names)
            if name == "supported":
                supported_observed = True
            elif name == "referenced":
                referenced_observed = True
            else:
                passing_observed = True

    from bench.ibsi_mapping import documented_semantic_aliases

    ibsi_codes: set[str] = set()
    for feature_name in attempted:
        code, status = classify_feature(adapter, feature_name)
        if status == "mapped" and code:
            ibsi_codes.add(code)
        ibsi_codes.update(documented_semantic_aliases(adapter, feature_name))

    from bench.ibsi_families import CODE_TO_FAMILY

    family_counts: Dict[str, int] = {}
    for code in ibsi_codes:
        family = CODE_TO_FAMILY.get(code)
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1

    return {
        "denominators": {
            "supported": len(supported) if supported_observed else None,
            "attempted": len(attempted),
            "finite": len(finite) if finite_observed else None,
            "referenced": len(referenced) if referenced_observed else None,
            "passing": len(passing) if passing_observed else None,
        },
        "uniquely_mapped_ibsi_from_attempted": len(ibsi_codes),
        "uniquely_mapped_features_by_family": family_counts,
    }


def summarize_features_from_payloads(
    payloads: list[Mapping[str, Any]], adapters: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Summarize feature denominators from ledger-verified payload objects."""

    by_adapter: Dict[str, list[Mapping[str, Any]]] = {
        adapter: [] for adapter in adapters
    }
    for payload in payloads:
        adapter = str(payload.get("adapter") or "").strip()
        if adapter in by_adapter:
            by_adapter[adapter].append(payload)
    return {
        adapter: _summarize_adapter_payloads(adapter, by_adapter[adapter])
        for adapter in adapters
    }


def feature_agreement(payloads: list[Mapping[str, Any]]) -> pd.DataFrame:
    """Compare the value of each IBSI feature across adapters on the same input.

    One row per case, workload, IBSI code and adapter.  The values do not
    depend on the repeat, so the lowest measured repeat of each adapter is
    used.  The consensus is the median over the adapters that give the code.
    A speed comparison is fair only where the adapters calculate the same
    values, so large differences need a reason before a speed claim.
    """

    from bench.ibsi_families import CODE_TO_FAMILY

    chosen: Dict[tuple[str, str, str], tuple[int, Mapping[str, Any]]] = {}
    for payload in payloads:
        benchmark = payload.get("benchmark")
        record = benchmark.get("record") if isinstance(benchmark, Mapping) else None
        container = payload.get("values")
        values = container.get("all") if isinstance(container, Mapping) else None
        if not isinstance(record, Mapping) or not isinstance(values, Mapping):
            continue
        key = (
            str(record.get("case_id")),
            str(record.get("workload")),
            str(payload.get("adapter")),
        )
        repeat = int(record.get("repeat") or 0)
        if key not in chosen or repeat < chosen[key][0]:
            chosen[key] = (repeat, values)

    rows = []
    for (case_id, workload, adapter), (repeat, values) in chosen.items():
        by_code: Dict[str, list[tuple[str, float]]] = {}
        for name, value in values.items():
            code, status = classify_feature(adapter, str(name))
            if status == "mapped" and code:
                by_code.setdefault(code, []).append((str(name), float(value)))
        for code, items in by_code.items():
            # Two native names for one code cannot give one comparable value.
            ambiguous = len(items) > 1
            rows.append(
                {
                    "case_id": case_id,
                    "workload": workload,
                    "ibsi_code": code,
                    "family": CODE_TO_FAMILY.get(code),
                    "adapter": adapter,
                    "repeat": repeat,
                    "feature_name": "; ".join(name for name, _ in items),
                    "value": np.nan if ambiguous else items[0][1],
                    "ambiguous_mapping": ambiguous,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keys = ["case_id", "workload", "ibsi_code"]
    grouped = frame.groupby(keys)["value"]
    frame["adapters_with_value"] = grouped.transform("count")
    frame["consensus_median"] = grouped.transform("median")
    frame["absolute_difference"] = (frame["value"] - frame["consensus_median"]).abs()
    scale = frame["consensus_median"].abs()
    frame["relative_difference"] = frame["absolute_difference"] / scale.where(scale > 0)
    return frame.sort_values([*keys, "adapter"], kind="stable").reset_index(drop=True)


def feature_agreement_summary(agreement: pd.DataFrame) -> pd.DataFrame:
    """Per workload, adapter and IBSI family: how often the values agree."""

    columns = [
        "workload",
        "adapter",
        "family",
        "compared_values",
        "relative_difference_median",
        "relative_difference_p95",
        "relative_difference_max",
        "share_within_0_1_percent",
        "share_within_1_percent",
    ]
    compared = (
        agreement.loc[
            (agreement["adapters_with_value"] >= 2)
            & agreement["relative_difference"].notna()
        ]
        if not agreement.empty
        else agreement
    )
    if compared.empty:
        return pd.DataFrame(columns=columns)
    difference = compared.groupby(["workload", "adapter", "family"])[
        "relative_difference"
    ]
    summary = difference.agg(
        compared_values="count",
        relative_difference_median="median",
        relative_difference_p95=lambda values: values.quantile(0.95),
        relative_difference_max="max",
        share_within_0_1_percent=lambda values: float((values <= 1e-3).mean()),
        share_within_1_percent=lambda values: float((values <= 1e-2).mean()),
    ).reset_index()
    return summary[columns]


TASK_OUTCOME_COLUMNS = [
    "case_id",
    "dataset",
    "modality",
    "subject_id",
    "size",
    "mask_id",
    "image_voxels",
    "mask_voxels",
    "adapter",
    "workload",
    "repeat",
    "task_status",
    "duration_sec",
    "warmup_duration_sec",
    "censor_lower_bound_sec",
    "stop_phase",
    "per_call_timeout_sec",
    "setup_sec",
    "peak_rss_bytes",
    "worker_maxrss_after_warmup_bytes",
    "memory_observed_bytes",
]


def task_outcome_table(records: pd.DataFrame) -> pd.DataFrame:
    """Return one row per task and outcome; a screen run's feasibility map."""

    frame = with_report_workload(records)
    frame["workload"] = frame["_report_workload"]
    for column in TASK_OUTCOME_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[TASK_OUTCOME_COLUMNS].reset_index(drop=True)


def _diagnostic_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame(
            columns=[
                "case_id",
                "adapter",
                "workload",
                "repeat",
                "status",
                "modality",
                "shape",
                "spacing",
                "image_voxels",
                "size",
                "mask_id",
                "reason",
                "censor_lower_bound_sec",
                "stop_phase",
                "warmup_duration_sec",
                "per_call_timeout_sec",
            ]
        )

    result = pd.DataFrame(index=outcomes.index)
    for column in (
        "case_id",
        "adapter",
        "repeat",
        "modality",
        "shape",
        "spacing",
        "image_voxels",
        "size",
        "mask_id",
        "censor_lower_bound_sec",
        "stop_phase",
        "warmup_duration_sec",
        "per_call_timeout_sec",
    ):
        result[column] = outcomes[column] if column in outcomes.columns else np.nan
    result["workload"] = outcomes["_report_workload"]
    result["status"] = outcomes["_report_status"]

    reason = pd.Series("", index=outcomes.index, dtype="object")
    for column in ("error", "policy_reason"):
        if column not in outcomes.columns:
            continue
        values = outcomes[column].fillna("").astype(str).str.strip()
        reason = reason.where(reason.ne(""), values)
    result["reason"] = reason.replace("", "No diagnostic message recorded")
    return result[
        [
            "case_id",
            "adapter",
            "workload",
            "repeat",
            "status",
            "modality",
            "shape",
            "spacing",
            "image_voxels",
            "size",
            "mask_id",
            "reason",
            "censor_lower_bound_sec",
            "stop_phase",
            "warmup_duration_sec",
            "per_call_timeout_sec",
        ]
    ]


def _write_markdown_summary(
    path: Path,
    *,
    input_dir: Path,
    metadata: Mapping[str, Any],
    measured_n: int,
    matched: pd.DataFrame,
    matched_summary: pd.DataFrame,
    feature_contract: pd.DataFrame,
    diagnostics: pd.DataFrame,
    coverage: Dict[str, Dict[str, Any]],
    figures: list[Dict[str, str]],
) -> None:
    kind = _normalise_dataset_kind(metadata.get("dataset_kind"))
    run_id = metadata.get("run_id") or input_dir.name
    baseline_adapter = str(metadata.get("baseline_adapter") or BASELINE_ADAPTER)
    lines = [
        "# Benchmark performance report",
        "",
        f"- Run: `{run_id}`",
        f"- Dataset kind: `{kind}`",
        f"- Run status: `{metadata.get('run_status', 'unknown')}`",
        f"- Execution complete: "
        f"{'yes' if metadata.get('execution_complete') else 'no'}",
        f"- Planned tasks: {metadata.get('task_count', 'unknown')}",
        f"- Unfinished tasks: {metadata.get('unfinished_task_count', 'unknown')}",
        f"- Measured timing observations: {measured_n}",
        f"- Checksum-verified measured payloads: "
        f"{metadata.get('verified_payload_count', 'unknown')}",
        f"- Comparison baseline adapter: `{baseline_adapter}`",
        f"- Exact candidate/baseline pairs: {len(matched)}",
        f"- Input record source: `{metadata.get('record_source', 'unknown')}`",
        f"- Report generator provenance: "
        f"`{dict(metadata.get('report_generator') or {}).get('provenance_sha256', 'unknown')}`",
        f"- Source attested against ledger payload hashes: "
        f"{'yes' if metadata.get('source_attested') else 'no'}",
        f"- Publication-attested complete report: "
        f"{'yes' if metadata.get('publication_attested') else 'no'}",
        "",
    ]
    status_counts = dict(metadata.get("status_counts") or {})
    if status_counts:
        lines.extend(
            [
                "## Authoritative task status counts",
                "",
                "| Task status | n |",
                "|---|---:|",
            ]
        )
        for status, count in sorted(status_counts.items()):
            lines.append(f"| {status} | {count} |")
        lines.append("")
    if not metadata.get("execution_complete"):
        lines.extend(
            [
                "> **Incomplete run:** this report is an explicitly partial "
                "snapshot and must not be described as a completed benchmark.",
                "",
            ]
        )

    protocol = _protocol_summary(metadata.get("run_spec"))
    if not protocol.empty:
        lines.extend(
            [
                "## Executed protocol",
                "",
                "| Field | Value |",
                "|---|---|",
            ]
        )
        for row in protocol.itertuples(index=False):
            lines.append(
                f"| {_markdown_cell(row.field)} | {_markdown_cell(row.value)} |"
            )
        lines.append("")

    environments = _adapter_environment_summary(metadata.get("run_spec"))
    if not environments.empty:
        lines.extend(
            [
                "## Adapter environments",
                "",
                "| Adapter | Distribution | Reviewed release | Installed metadata | Python | NumPy | Environment record SHA-256 |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in environments.itertuples(index=False):
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        row.adapter,
                        row.distribution,
                        row.configured_release_version,
                        row.distribution_metadata_version,
                        row.python_version,
                        row.numpy_version,
                        row.environment_metadata_sha256,
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "The separate `adapter_environment_summary.csv` retains the "
                "complete NumPy/BLAS build configuration and a digest of the "
                "installed package set, plus the exact recorded `pip freeze` "
                "digest. Reviewed release and installed "
                "distribution metadata are intentionally separate; this is "
                "material for PyRadiomics 3.1.0, whose official tag exposes "
                "upstream metadata 3.0.1a1.",
                "",
            ]
        )

    lines.extend(
        [
            "## Runtime comparison contract",
            "",
            f"Relative runtime is candidate duration divided by {baseline_adapter} "
            "duration. Values above 1 mean the baseline was faster. Each ratio uses "
            "the same `case_id`, workload, and process repeat. Unmatched, timed-out, "
            "failed, unsupported, skipped, and dropped rows are excluded from ratios.",
            "",
            "A runtime or ratio summary of one cell needs every planned repeat. A "
            "cell with a censored, failed or skipped repeat keeps its counts "
            "(`planned_repeats`, `planned_pairs`) but gets no summary value, "
            "because the finished repeats alone would understate the runtime.",
            "",
            "A matched canonical family does not imply that every adapter calculates "
            "an identical list of individual features. These timings represent each "
            "package's reviewed maximum native family surface; they are package "
            "workloads, not a common-feature microbenchmark. Observed feature "
            "denominators are reported separately and must accompany speed claims.",
            "",
            "Adapter timing policies are explicit: package import, NIfTI loading, "
            "mask/resegmentation preparation, stored-representation validation, "
            "discretisation, warm-up, and output normalisation are outside the "
            "timer. Matrix/mesh/neighbourhood construction and feature arithmetic "
            "are inside. Pictologics JIT initialization is outside the measured "
            "region. The Z-Rad local-intensity cache and every cached table of the "
            "Pictologics feature modules are cleared before and after every "
            "calculation.",
            "",
            "Synthetic and real-world performance runs benchmark radiomic feature "
            "extraction only. IBSI 2 filter results establish conformity, not filter "
            "runtime; no filter-performance claim is made.",
            "",
        ]
    )
    if kind == "synthetic":
        lines.extend(
            [
                "Synthetic summaries retain cube edge length and mask identity.",
                "",
            ]
        )
    elif kind == "real_world":
        lines.extend(
            [
                "Real-world summaries retain modality and the true image voxel "
                "count. No cube geometry is inferred from a maximum dimension.",
                "",
            ]
        )

    lines.extend(
        [
            "## Output tables",
            "",
            "- [Measured runtime summary](adapter_performance_table.csv)",
            "- [Frozen and observed feature workloads](feature_workload_contract.csv)",
            "- [Exact matched runtime pairs](matched_runtime_pairs.csv)",
            "- [Matched runtime summary with matched n](matched_runtime_summary.csv)",
            "- [Diagnostic-only excluded outcomes](excluded_outcomes.csv)",
            "- [Executed protocol and machine](protocol_summary.csv)",
            "- [Adapter software and numerical runtimes](adapter_environment_summary.csv)",
            "- [Feature values compared across adapters](feature_agreement.csv)",
            "- [Agreement by adapter and IBSI family](feature_agreement_summary.csv)",
            "- [Every task record, with each timed call](tables/data_task_records.csv)",
            "- [Auditable workbook](academic_performance_report.xlsx)",
            "- [Run-bound artifact checksums](report_manifest.json)",
            "",
        ]
    )
    if not matched_summary.empty:
        lines.extend(
            [
                "The matched summary reports medians and interquartile ranges. "
                "`matched_n` is the number of exact pairs in each displayed cell.",
                "",
            ]
        )

    if not feature_contract.empty:
        lines.extend(
            [
                "## Feature workload denominators",
                "",
                "Runtime is not divided by feature count. Matrix construction "
                "and feature evaluation share costs, so a per-feature quotient "
                "would assert a linear cost model that the benchmark does not "
                "establish. The separate workload table reports frozen native "
                "outputs and measured/planned task coverage beside runtime.",
                "",
            ]
        )

    lines.extend(["## Figures", ""])
    if figures:
        for figure in figures:
            lines.append(
                f"- [{figure['artifact']}]({figure['artifact']}): {figure['alt_text']}"
            )
    else:
        lines.append("- No runtime figure could be generated.")
    lines.append("")

    lines.extend(["## Excluded outcomes", ""])
    if diagnostics.empty:
        lines.append("No non-measured terminal outcomes were recorded.")
    else:
        counts = (
            diagnostics.groupby(["status", "adapter"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        lines.extend(
            [
                "| Status | Adapter | n |",
                "|---|---|---:|",
            ]
        )
        for row in counts.itertuples(index=False):
            lines.append(f"| {row.status} | {row.adapter} | {row.n} |")
    lines.append("")

    if not metadata.get("source_attested"):
        lines.extend(
            [
                "## Input integrity note",
                "",
                "This report used an unattested record source and is not "
                "publication-attested. A ledger-backed run is required to verify "
                "committed payload hashes and task completeness.",
                "",
            ]
        )
    elif not metadata.get("publication_attested"):
        lines.extend(
            [
                "## Publication integrity note",
                "",
                "Ledger records and measured payloads were integrity-checked, "
                "but this report is not publication-attested because execution "
                "is incomplete or dataset hashes were not verified.",
                "",
            ]
        )

    atomic_write_text(path, "\n".join(lines) + "\n")


def _publication_attested(metadata: Mapping[str, Any]) -> bool:
    run_spec = metadata.get("run_spec")
    hashes_verified = (
        bool(run_spec.get("dataset_hashes_verified"))
        if isinstance(run_spec, dict)
        else False
    )
    return bool(
        metadata.get("source_attested")
        and metadata.get("execution_complete")
        and metadata.get("run_fingerprint")
        and hashes_verified
    )


def _write_report_manifest(
    path: Path,
    *,
    output_dir: Path,
    artifacts: list[Path],
    metadata: Mapping[str, Any],
    dataset_kind: str,
    baseline_adapter: str,
) -> None:
    """Bind every report artifact to its run provenance and file digest.

    The manifest itself is excluded because a file cannot contain its own
    stable checksum. It is written last and atomically.
    """

    entries: list[Dict[str, Any]] = []
    unique_paths = sorted(
        {artifact.resolve() for artifact in artifacts},
        key=lambda artifact: str(artifact),
    )
    for artifact in unique_paths:
        if artifact == path.resolve():
            continue
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Report artifact is missing before manifest commit: {artifact}"
            )
        try:
            relative = artifact.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Report artifact is outside the output directory: {artifact}"
            ) from exc
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
        )

    run_spec = metadata.get("run_spec")
    hashes_verified = (
        bool(run_spec.get("dataset_hashes_verified"))
        if isinstance(run_spec, dict)
        else False
    )
    manifest = {
        "schema_version": 1,
        "run_id": metadata.get("run_id"),
        "run_fingerprint": metadata.get("run_fingerprint"),
        "report_generator": dict(metadata.get("report_generator") or {}),
        "run_status": metadata.get("run_status"),
        "execution_complete": bool(metadata.get("execution_complete")),
        "source_attested": bool(metadata.get("source_attested")),
        "publication_attested": _publication_attested(metadata),
        "dataset_hashes_verified": hashes_verified,
        "dataset_kind": dataset_kind,
        "baseline_adapter": baseline_adapter,
        "record_source": metadata.get("record_source"),
        "status_counts": dict(metadata.get("status_counts") or {}),
        "verified_payload_count": metadata.get("verified_payload_count"),
        "artifact_count": len(entries),
        "manifest_self_excluded": True,
        "artifacts": entries,
    }
    atomic_write_json(path, manifest)


def generate_report(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    record_loader: ReportRecordLoader | None = None,
) -> Dict[str, Path]:
    """Generate the performance report without executing benchmark tasks."""

    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    records, metadata = load_report_records(input_path, record_loader=record_loader)
    metadata["report_generator"] = _report_generator_provenance()
    measured = timing_observations(records)
    if measured.empty:
        raise ValueError(
            "No measured, non-dropped timing observations found in report input."
        )
    kind = _resolve_dataset_kind(measured, metadata.get("dataset_kind"))
    baseline_adapter = _baseline_adapter_from_metadata(metadata)
    metadata["baseline_adapter"] = baseline_adapter
    metadata["publication_attested"] = _publication_attested(metadata)

    timing_summary = aggregate_timing_observations(records, dataset_kind=kind)
    matched = matched_runtime_observations(measured, baseline_adapter=baseline_adapter)
    matched_summary = matched_runtime_summary(
        matched,
        dataset_kind=kind,
        records=records,
        baseline_adapter=baseline_adapter,
    )
    diagnostics = _diagnostic_table(non_timing_outcomes(records))
    feature_contract = feature_workload_contract(records)

    table_path = output_path / "adapter_performance_table.csv"
    pair_path = output_path / "matched_runtime_pairs.csv"
    matched_summary_path = output_path / "matched_runtime_summary.csv"
    diagnostic_path = output_path / "excluded_outcomes.csv"
    outcome_path = output_path / "task_outcomes.csv"
    feature_contract_path = output_path / "feature_workload_contract.csv"
    protocol_path = output_path / "protocol_summary.csv"
    environment_path = output_path / "adapter_environment_summary.csv"
    _atomic_write_dataframe_csv(timing_summary, table_path)
    _atomic_write_dataframe_csv(matched, pair_path)
    _atomic_write_dataframe_csv(matched_summary, matched_summary_path)
    _atomic_write_dataframe_csv(diagnostics, diagnostic_path)
    _atomic_write_dataframe_csv(task_outcome_table(records), outcome_path)
    _atomic_write_dataframe_csv(feature_contract, feature_contract_path)
    _atomic_write_dataframe_csv(
        _protocol_summary(metadata.get("run_spec")), protocol_path
    )
    _atomic_write_dataframe_csv(
        _adapter_environment_summary(metadata.get("run_spec")),
        environment_path,
    )

    thread_policy = dict(getattr(metadata.get("run_spec"), "thread_policy", None) or {})
    figures = generate_plots(
        records, output_path, logical_cpus=thread_policy.get("logical_cpus")
    )
    figure_manifest_path = output_path / "figure_manifest.csv"
    _atomic_write_dataframe_csv(
        pd.DataFrame(
            figures,
            columns=["artifact", "format", "alt_text"],
        ),
        figure_manifest_path,
    )

    adapters = sorted(measured["adapter"].dropna().astype(str).unique())
    coverage = summarize_features_from_payloads(
        list(metadata.get("verified_payloads") or []), adapters
    )
    coverage_path = output_path / "feature_coverage_report.json"
    atomic_write_json(coverage_path, coverage)
    agreement = feature_agreement(list(metadata.get("verified_payloads") or []))
    agreement_path = output_path / "feature_agreement.csv"
    agreement_summary_path = output_path / "feature_agreement_summary.csv"
    _atomic_write_dataframe_csv(agreement, agreement_path)
    _atomic_write_dataframe_csv(
        feature_agreement_summary(agreement), agreement_summary_path
    )

    workbook_path = generate_excel_report(
        timing_summary,
        matched,
        matched_summary,
        feature_contract,
        diagnostics,
        metadata,
        output_path,
    )
    summary_path = output_path / "academic_report_summary.md"
    _write_markdown_summary(
        summary_path,
        input_dir=input_path,
        metadata=metadata,
        measured_n=len(measured),
        matched=matched,
        matched_summary=matched_summary,
        feature_contract=feature_contract,
        diagnostics=diagnostics,
        coverage=coverage,
        figures=figures,
    )
    artifact_paths = [
        table_path,
        pair_path,
        matched_summary_path,
        diagnostic_path,
        outcome_path,
        feature_contract_path,
        protocol_path,
        environment_path,
        figure_manifest_path,
        coverage_path,
        agreement_path,
        agreement_summary_path,
        workbook_path,
        summary_path,
    ]
    for figure in figures:
        pdf_path = output_path / figure["artifact"]
        artifact_paths.extend(
            pdf_path.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png", ".json")
        )
    report_manifest_path = output_path / "report_manifest.json"
    _write_report_manifest(
        report_manifest_path,
        output_dir=output_path,
        artifacts=artifact_paths,
        metadata=metadata,
        dataset_kind=kind,
        baseline_adapter=baseline_adapter,
    )
    return {
        "timing_summary": table_path,
        "matched_pairs": pair_path,
        "matched_summary": matched_summary_path,
        "excluded_outcomes": diagnostic_path,
        "task_outcomes": outcome_path,
        "feature_workload_contract": feature_contract_path,
        "protocol_summary": protocol_path,
        "adapter_environments": environment_path,
        "figure_manifest": figure_manifest_path,
        "feature_coverage": coverage_path,
        "feature_agreement": agreement_path,
        "feature_agreement_summary": agreement_summary_path,
        "workbook": workbook_path,
        "markdown_summary": summary_path,
        "report_manifest": report_manifest_path,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench report")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Run directory that contains benchmark.sqlite3",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for report tables and figures",
    )
    args = parser.parse_args(argv)
    outputs = generate_report(args.input_dir, args.output_dir)
    print(f"Report generated: {outputs['markdown_summary'].parent}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

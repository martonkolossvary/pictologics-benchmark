"""Timing and memory figures (Figures 4 to 6, Figures S3 to S8).

Records come straight from each run's SQLite ledger, opened read-only, so the
figures can be drawn while a run is still in progress.  A censored task never
receives a value: a timeout appears as its lower bound with an up arrow, and
every other outcome appears as a letter in the outcome strip.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
from matplotlib.patches import Rectangle

from bench.figures import style

STATUS_CODE = {
    "timed_out_censored": "T",
    "memory_exceeded_censored": "M",
    "skipped_memory_preflight": "P",
    "skipped_policy": "S",
    "unsupported": "U",
    "failed": "F",
    "interrupted": "I",
}
STATUS_NAME = {
    "T": "timed out",
    "M": "memory stop",
    "P": "memory preflight skip",
    "S": "skipped, earlier stop",
    "U": "not implemented",
    "F": "failed",
    "I": "interrupted",
}
PILLARS = ("pillar1", "pillar2", "pillar3")
BOOTSTRAP_SAMPLES = 2000


def load_records(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """All task records of run directories or submitted package pillars, read-only."""

    rows = []
    for run_dir in run_dirs:
        package = Path(run_dir) / "tasks.jsonl.gz"
        if package.is_file():
            lines = gzip.decompress(package.read_bytes()).decode("utf-8").splitlines()
            rows += [
                {**row["record"], "ordinal": row.get("ordinal")}
                for row in map(json.loads, filter(None, lines))
                if row["record"]
            ]
            continue
        database = Path(run_dir) / "benchmark.sqlite3"
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            # The run order lets a later analysis test order effects.
            for record, ordinal, started_at in connection.execute(
                "SELECT record_json, ordinal, started_at FROM tasks "
                "WHERE record_json IS NOT NULL"
            ):
                rows.append(
                    {**json.loads(record), "ordinal": ordinal, "started_at": started_at}
                )
        finally:
            connection.close()
    return prepare_records(pd.DataFrame(rows))


def prepare_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the group, pillar and measured columns that the figures use."""

    if frame.empty:
        return frame
    frame = frame.copy()
    frame["group"] = frame["workload"].astype(str).str.split(":").str[-1]
    dataset = frame["dataset"].astype(str)
    frame["pillar"] = np.select(
        [
            dataset.str.contains("pillar1"),
            dataset.str.contains("pillar2"),
            dataset.str.contains("ibsi2_phase3"),
        ],
        list(PILLARS),
        "other",
    )
    frame["measured"] = frame["task_status"] == "measured"
    return frame


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def cells(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per case, adapter and group: the repeats summarised."""

    keys = ["pillar", "adapter", "group", "case_id"]
    rows = []
    for key, part in frame.groupby(keys, sort=False):
        measured = part.loc[part["measured"], "duration_sec"].astype(float)
        done = part.loc[part["measured"]]
        # The peak memory in use of the whole adapter process tree is the
        # memory that the task needs.  A task too short for one memory sample
        # falls back to its peak RSS.
        in_use = _numeric(done, "host_peak_memory_in_use_bytes")
        memory = in_use.where(in_use > 0, _numeric(done, "peak_rss_bytes"))
        calculation = _numeric(done, "calculation_peak_rss_bytes")
        extra = _numeric(done, "incremental_calculation_peak_rss_bytes")
        cpu = part.loc[part["measured"], "cpu_time_sec"].astype(float)
        complete = len(measured) == len(part)
        others = part.loc[~part["measured"], "task_status"].map(STATUS_CODE).dropna()
        bounds = (
            part.loc[
                part["task_status"] == "timed_out_censored", "censor_lower_bound_sec"
            ]
            .dropna()
            .astype(float)
        )
        first = part.iloc[0]
        rows.append(
            {
                **dict(zip(keys, key)),
                "size": first.get("size"),
                "subject_id": first.get("subject_id"),
                "mask_id": first.get("mask_id"),
                "modality": str(first.get("modality") or "").lower(),
                "R": float(first["mask_voxels"]),
                "T": float(first["image_voxels"]),
                "planned": len(part),
                "n_measured": int(len(measured)),
                "times": tuple(measured),
                "median": float(np.median(measured)) if complete else np.nan,
                "memory_mb": float(np.median(memory)) / 1e6 if complete else np.nan,
                "calculation_peak_mb": float(np.median(calculation)) / 1e6
                if complete
                else np.nan,
                "extra_memory_mb": float(np.median(extra)) / 1e6
                if complete
                else np.nan,
                "cpu_median_sec": float(np.median(cpu)) if complete else np.nan,
                "cpu_ratio": float(np.median(cpu / measured.to_numpy()))
                if complete
                else np.nan,
                "bound": float(bounds.max()) if len(bounds) else np.nan,
                "code": others.mode().iloc[0] if len(others) else "",
                "budget_mb": float(first.get("memory_budget_bytes") or np.nan) / 1e6,
                "timeout": float(first.get("per_call_timeout_sec") or np.nan),
            }
        )
    return pd.DataFrame(rows)


def _major_log_ticks(ax) -> None:
    """Label only powers of ten, so log axes never crowd their tick labels."""

    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == "log":
            axis.set_major_locator(LogLocator(base=10, numticks=6))
            axis.set_minor_locator(
                LogLocator(base=10, subs=np.arange(2, 10), numticks=12)
            )
            axis.set_minor_formatter(NullFormatter())


def _narrow_log_ticks(ax, axis_name: str = "x") -> None:
    """Label 1, 2 and 5 per decade when an axis spans less than two decades."""

    axis = ax.xaxis if axis_name == "x" else ax.yaxis
    low, high = sorted(axis.get_view_interval())
    if low <= 0 or np.log10(high / low) >= 2:
        return
    ticks = [
        m * 10.0**k
        for k in range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
        for m in (1, 2, 5)
    ]
    ticks = [tick for tick in ticks if low <= tick <= high]
    axis.set_ticks(ticks, [f"{tick:g}" for tick in ticks])
    axis.set_minor_formatter(NullFormatter())


def _sparse_decades(ax, axis_name: str = "x", max_labels: int = 4) -> None:
    """Label every k-th power of ten so a narrow log axis keeps its labels apart."""

    axis = ax.xaxis if axis_name == "x" else ax.yaxis
    low, high = sorted(axis.get_view_interval())
    first, last = int(np.ceil(np.log10(low))), int(np.floor(np.log10(high)))
    step = max(1, int(np.ceil((last - first + 1) / max_labels)))
    axis.set_ticks([10.0**k for k in range(first, last + 1, step)])
    axis.set_major_formatter(LogFormatterSciNotation())
    axis.set_minor_formatter(NullFormatter())


def logical_cpus(run_dirs: Iterable[Path]) -> int | None:
    """The logical CPU count recorded in the run specification."""

    for run_dir in run_dirs:
        spec = Path(run_dir) / "run_spec.json"
        if spec.is_file():
            policy = (
                json.loads(spec.read_text(encoding="utf-8")).get("thread_policy") or {}
            )
            if policy.get("logical_cpus"):
                return int(policy["logical_cpus"])
    return None


def _groups_present(frame: pd.DataFrame) -> list[tuple[str, str]]:
    present = set(frame["group"])
    return [(key, label) for key, label in style.GROUPS if key in present]


def _adapters_present(frame: pd.DataFrame) -> list[str]:
    return [key for key in style.ADAPTER_ORDER if key in set(frame["adapter"])]


def _series(
    ax,
    data: pd.DataFrame,
    adapter: str,
    value: str,
    x: str = "R",
) -> None:
    """Medians joined by the adapter's line, repeats as small points, bounds as arrows."""

    adapter_style = style.ADAPTER[adapter]
    data = data.sort_values(x)
    complete = data.loc[data[value].notna()]
    ax.plot(
        complete[x],
        complete[value],
        color=adapter_style.color,
        linestyle=adapter_style.linestyle,
        marker=adapter_style.marker,
        markeredgecolor=style.INK,
        markeredgewidth=0.5,
        linewidth=1.1,
        zorder=4,
    )
    if value == "median":
        for row in data.itertuples():
            if row.n_measured and not np.isfinite(row.median):
                # Some repeats are censored: show the measured ones, no median.
                ax.scatter(
                    [row.R] * row.n_measured,
                    row.times,
                    marker=adapter_style.marker,
                    facecolor=style.WHITE,
                    edgecolor=adapter_style.color,
                    linewidth=0.8,
                    s=14,
                    zorder=5,
                )
        spread = complete.explode("times")
        ax.scatter(
            spread[x],
            spread["times"].astype(float),
            marker=adapter_style.marker,
            s=5,
            color=adapter_style.color,
            alpha=0.6,
            linewidth=0,
            zorder=3,
        )
        censored = data.loc[data["bound"].notna() & data["median"].isna()]
        for row in censored.itertuples():
            ax.scatter(
                [getattr(row, x)],
                [row.bound],
                marker=adapter_style.marker,
                facecolor=style.WHITE,
                edgecolor=adapter_style.color,
                linewidth=0.9,
                s=18,
                zorder=6,
            )
            ax.annotate(
                "",
                (getattr(row, x), row.bound * 2.2),
                xytext=(getattr(row, x), row.bound * 1.15),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": adapter_style.color,
                    "lw": 0.8,
                    "mutation_scale": 6,
                },
                zorder=6,
            )


def _outcome_strip(
    ax,
    data: pd.DataFrame,
    adapters: Sequence[str],
    x: str = "R",
    *,
    show_timeouts: bool = False,
) -> None:
    """One row per adapter with a letter at every size that has no value."""

    for row_index, adapter in enumerate(adapters):
        part = data.loc[data["adapter"] == adapter]
        y = len(adapters) - 1 - row_index
        if len(part) and (part["code"] == "U").all():
            ax.text(
                0.5,
                y,
                "not implemented",
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="center",
            )
            continue
        for row in part.itertuples():
            if row.code and (show_timeouts or row.code != "T"):
                ax.text(getattr(row, x), y, row.code, ha="center", va="center")
    ax.set_ylim(-0.6, len(adapters) - 0.4)
    ax.set_yticks(
        range(len(adapters)), [style.ADAPTER[key].label for key in reversed(adapters)]
    )
    ax.tick_params(axis="both", which="both", length=0, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _grid_axis(ax, data: pd.DataFrame) -> None:
    """Top axis with the grid edges where each size is one case."""

    sizes = data.drop_duplicates("size").set_index("size")["R"]
    if data.groupby("size")["case_id"].nunique().max() != 1:
        return
    for size in (32, 64, 128, 256, 512):
        if size in sizes.index:
            ax.text(
                sizes[size],
                1.0,
                f"{size}³",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
            )


def _legend_axes(
    ax, adapters: Sequence[str], codes: Sequence[str], extra: Sequence[str] = ()
) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=style.ADAPTER[key].color,
            linestyle=style.ADAPTER[key].linestyle,
            marker=style.ADAPTER[key].marker,
            markeredgecolor=style.INK,
            markeredgewidth=0.5,
            label=style.ADAPTER[key].label,
        )
        for key in adapters
    ]
    ax.axis("off")
    ax.legend(handles=handles, loc="upper left", handlelength=3.2, title="Package")
    lines = [
        f"{code}  {STATUS_NAME[code]}"
        for code in codes
        if code in STATUS_NAME and code != "T"
    ] + list(extra)
    lines.insert(0, "open marker with arrow: timeout,\n    drawn at its lower bound")
    ax.text(
        0.0,
        0.42,
        "Outcome strip and marks\n" + "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )


def _scaling_figure(
    summary: pd.DataFrame,
    value: str,
    ylabel: str,
    stem: Path,
    *,
    caption: str,
    alt_text: str,
    reference: tuple[float, str] | None = None,
) -> dict[str, Any]:
    groups = _groups_present(summary)
    adapters = _adapters_present(summary)
    codes = sorted(set(summary["code"]) - {""})
    # Three panels to a row, with the key in the last slot.
    rows = int(np.ceil((len(groups) + 1) / 3))
    with style.figure(style.DOUBLE_COLUMN_MM, 82.5 * rows, font_pt=8.0) as fig:
        grid = fig.add_gridspec(2 * rows, 3, height_ratios=[0.3, 1] * rows)
        positive = summary[value].replace(0, np.nan).dropna()
        bounds = (
            summary["bound"].dropna() if value == "median" else pd.Series(dtype=float)
        )
        values = pd.concat([positive, bounds])
        low = values.min() / 2 if len(values) else 1e-4
        high = max(
            values.max() * 4 if len(values) else 1.0,
            reference[0] * 1.5 if reference else 0,
        )
        main_axes = []
        for index, (group, label) in enumerate(groups + [("key", "")]):
            row, column = divmod(index, 3)
            if group == "key":
                _legend_axes(
                    fig.add_subplot(grid[2 * row : 2 * row + 2, column]),
                    adapters,
                    codes,
                )
                continue
            strip = fig.add_subplot(grid[2 * row, column])
            ax = fig.add_subplot(grid[2 * row + 1, column], sharex=strip)
            data = summary.loc[summary["group"] == group]
            for adapter in adapters:
                _series(ax, data.loc[data["adapter"] == adapter], adapter, value)
            _grid_axis(ax, data)
            _outcome_strip(strip, data, adapters, show_timeouts=value != "median")
            if column:
                strip.set_yticklabels([])
            ax.set_xscale("log")
            ax.set_yscale("log")
            _major_log_ticks(ax)
            ax.set_ylim(low, high)
            if reference:
                style.reference_line(ax, y=reference[0], label=reference[1])
            strip.set_title(label, loc="left")
            ax.set_xlabel("ROI voxels R")
            if column == 0:
                ax.set_ylabel(ylabel)
            main_axes.append(ax)
        return style.save(fig, stem, caption=caption, alt_text=alt_text)


def runtime_scaling(
    summary: pd.DataFrame, output_dir: Path, *, pillar: str = "pillar2"
) -> dict[str, Any]:
    data = summary.loc[summary["pillar"] == pillar]
    timeout = float(data["timeout"].dropna().max())
    return _scaling_figure(
        data,
        "median",
        "Wall time per call (s)",
        output_dir / "figure4_runtime_scaling",
        reference=(timeout, f"{timeout:,.0f} s limit per call"),
        caption=(
            "Runtime scaling on the dense whole-anatomy series (Pillar 2, mask A1). One "
            "panel per feature group; x: ROI voxels R; y: wall time per calculation call. "
            "Large markers: the median of the three process medians, joined by the "
            "package's line (markers and line styles as in the key); small points: the "
            "three process medians. A median is drawn only when all three repeats were "
            "measured. Open markers with an up arrow: a timeout, drawn at its lower bound. "
            "Letters in the strip above each panel give the other outcomes (key). Runtime "
            "is not divided by feature count; group times are not full-extraction times."
        ),
        alt_text=(
            "Five log-log line charts of wall time per call against region of interest "
            "voxels for five packages on the whole-anatomy series, one chart per feature group."
        ),
    )


def memory_scaling(
    summary: pd.DataFrame, output_dir: Path, *, pillar: str = "pillar2"
) -> dict[str, Any]:
    data = summary.loc[summary["pillar"] == pillar].copy()
    budget = float(data["budget_mb"].dropna().max())
    return _scaling_figure(
        data,
        "memory_mb",
        "Peak memory in use (MB)",
        output_dir / "figure6_memory_scaling",
        reference=(budget, f"memory budget {budget / 1000:,.1f} GB"),
        caption=(
            "Memory on the dense whole-anatomy series (Pillar 2, mask A1): the peak "
            "memory in use of the whole adapter process tree during the task, median of "
            "three processes. This is the same measure that the memory stop uses, so it "
            "counts everything that a package needs, including its own input "
            "preparation. The labelled line is the memory budget; letters give memory "
            "stops (M) and preflight skips (P)."
        ),
        alt_text=(
            "Line charts of peak memory in use against region of interest voxels for "
            "five packages, with the memory budget as a line."
        ),
    )


def _geometric_summary(
    values: np.ndarray, rng: np.random.Generator, *, bootstrap: bool = True
) -> tuple[float, float, float]:
    """Geometric mean with a bootstrap interval over independent cases, else the range."""

    logs = np.log(values)
    mean = float(np.exp(logs.mean()))
    if not bootstrap or len(values) < 4:
        return mean, float(values.min()), float(values.max())
    samples = rng.choice(logs, size=(BOOTSTRAP_SAMPLES, len(logs)), replace=True).mean(
        axis=1
    )
    low, high = np.exp(np.percentile(samples, [2.5, 97.5]))
    return mean, float(low), float(high)


def _ratio_selections(
    summary: pd.DataFrame, synthetic_size: int
) -> list[tuple[str, pd.DataFrame, bool]]:
    """The data sets of the ratio figure: (label, cells, independent cases).

    Pillar 1 gives one data set per noise level, because the noise levels are
    paired, not replicates.  The masks of one noise level, and the single A1
    case of Pillar 2, come from one synthetic scene, so they get a range, not
    a bootstrap interval.  Only the Pillar 3 patients are independent cases.
    """

    selections = []
    for pillar in ("pillar1", "pillar2"):
        part = summary.loc[summary["pillar"] == pillar]
        if not len(part):
            continue
        sizes = sorted(set(part["size"].dropna().astype(int)))
        size = (
            synthetic_size
            if synthetic_size in sizes
            else max(s for s in sizes if s <= synthetic_size)
            if any(s <= synthetic_size for s in sizes)
            else sizes[0]
        )
        part = part.loc[part["size"] == size]
        if pillar == "pillar2":
            selections.append((f"Pillar 2, {size}³", part, False))
            continue
        for noise, noise_label in style.NOISE_LEVELS:
            level = part.loc[part["subject_id"] == noise]
            if len(level):
                name = noise_label.split(" (")[0].lower()
                selections.append((f"Pillar 1, {size}³, {name}", level, False))
    for modality in ("ct", "mri", "pet"):
        part = summary.loc[
            (summary["pillar"] == "pillar3") & (summary["modality"] == modality)
        ]
        if len(part):
            selections.append(
                (f"Pillar 3, {style.MODALITY_LABEL[modality]}", part, True)
            )
    return selections


def _case_ratio(
    candidate: float, candidate_bound: float, baseline: float, baseline_bound: float
) -> tuple[float, str]:
    """One case: the ratio of medians, or a bound when one side timed out.

    A package timeout gives a lower bound and a Pictologics timeout gives an
    upper bound, so a censored task of either side counts the same way.
    """

    if np.isfinite(candidate) and np.isfinite(baseline):
        return candidate / baseline, "point"
    if not np.isfinite(candidate) and not np.isfinite(candidate_bound):
        return np.nan, "no_package_value"
    if np.isfinite(baseline):
        return candidate_bound / baseline, "lower_bound"
    if np.isfinite(candidate) and np.isfinite(baseline_bound):
        return candidate / baseline_bound, "upper_bound"
    return np.nan, "not_estimable"


def ratio_cases(summary: pd.DataFrame, *, synthetic_size: int = 128) -> pd.DataFrame:
    """One row per data set, group, package and case: the ratio to Pictologics."""

    rows = []
    for label, part, independent in _ratio_selections(summary, synthetic_size):
        reference = part.loc[part["adapter"] == "pictologics"].set_index(
            ["group", "case_id"]
        )
        for row in part.loc[part["adapter"] != "pictologics"].itertuples():
            if (row.group, row.case_id) not in reference.index:
                continue
            base = reference.loc[(row.group, row.case_id)]
            ratio, kind = _case_ratio(
                row.median, row.bound, base["median"], base["bound"]
            )
            rows.append(
                {
                    "dataset": label,
                    "independent_cases": independent,
                    "group": row.group,
                    "adapter": row.adapter,
                    "case_id": row.case_id,
                    "subject_id": row.subject_id,
                    "mask_id": row.mask_id,
                    "package_median_sec": row.median,
                    "package_bound_sec": row.bound,
                    "package_code": row.code,
                    "pictologics_median_sec": base["median"],
                    "pictologics_bound_sec": base["bound"],
                    "pictologics_code": base["code"],
                    "ratio": ratio,
                    "ratio_kind": kind,
                }
            )
    return pd.DataFrame(rows)


def paired_ratios(summary: pd.DataFrame, *, synthetic_size: int = 128) -> pd.DataFrame:
    """Per data set, group and package: the geometric mean ratio to Pictologics.

    Each case gives one ratio of medians.  A censored package task gives a
    lower bound and a censored Pictologics task an upper bound.  A case where
    both sides, or Pictologics without a time bound, were censored is not
    estimable.
    """

    rng = np.random.default_rng(0)
    cases = ratio_cases(summary, synthetic_size=synthetic_size)
    rows = []
    if cases.empty:
        return pd.DataFrame(rows)
    for (label, group, adapter), part in cases.groupby(
        ["dataset", "group", "adapter"], sort=False
    ):
        kinds = part["ratio_kind"]
        values = part.loc[kinds.isin(["point", "lower_bound", "upper_bound"]), "ratio"]
        lower = int((kinds == "lower_bound").sum())
        upper = int((kinds == "upper_bound").sum())
        codes = part["package_code"].loc[part["package_code"] != ""]
        entry = {
            "dataset": label,
            "group": group,
            "adapter": adapter,
            "n": len(values),
            "planned": len(part),
            "n_lower_bound": lower,
            "n_upper_bound": upper,
            "lower_bound": lower > 0 and upper == 0,
            "upper_bound": upper > 0 and lower == 0,
            "mixed_bounds": lower > 0 and upper > 0,
            "not_estimable": int((kinds == "not_estimable").sum()),
            "interval": "bootstrap_95"
            if bool(part["independent_cases"].iloc[0]) and len(values) >= 4
            else "range",
            "code": codes.mode().iloc[0] if len(codes) else "",
        }
        if len(values):
            mean, low, high = _geometric_summary(
                values.to_numpy(dtype=float),
                rng,
                bootstrap=bool(part["independent_cases"].iloc[0]),
            )
            entry.update({"ratio": mean, "low": low, "high": high})
        rows.append(entry)
    return pd.DataFrame(rows)


def ratio_figure(
    summary: pd.DataFrame, output_dir: Path, *, synthetic_size: int = 128
) -> dict[str, Any]:
    return ratio_dot_plot(
        paired_ratios(summary, synthetic_size=synthetic_size),
        "dataset",
        output_dir / "figure5_runtime_ratio",
        caption=(
            "Runtime relative to Pictologics on the same case. A ratio above 1 means the "
            "package took longer. Each case gives the ratio of the two medians of three "
            "processes. Pillar 1 has one row per noise level; the marker is the geometric "
            "mean over its four masks, and the line is their range, because the masks "
            "share one synthetic scene. Pillar 2 has one case. For Pillar 3 the marker is "
            "the geometric mean over patients, and the line is its 95 percent bootstrap "
            "interval (2,000 resamples); with fewer than four patients the line is the "
            "range. An open marker with ≥ is a lower bound (a package task timed out), "
            "with ≤ an upper bound (a Pictologics task timed out), and without a sign it "
            "mixes both. Letters: U not implemented, T timed out, M memory stop, P "
            "preflight skip, S skipped after an earlier stop, NE not estimable (both "
            "tasks censored, or Pictologics stopped without a time bound). Markers follow "
            "the package key of all figures."
        ),
        alt_text=(
            "Dot plot of runtime relative to Pictologics, on a log scale, for four packages, "
            "five feature groups and up to five data sets, with intervals."
        ),
    )


def ratio_dot_plot(
    table: pd.DataFrame, row_field: str, stem: Path, *, caption: str, alt_text: str
) -> dict[str, Any]:
    """One panel per feature group; one row per package and value of *row_field*."""

    groups = [item for item in style.GROUPS if item[0] in set(table["group"])]
    adapters = [key for key in style.ADAPTER_ORDER if key in set(table["adapter"])]
    datasets = list(dict.fromkeys(table[row_field]))
    rows = [(adapter, dataset) for adapter in adapters for dataset in datasets]
    finite = table[["low", "high", "ratio"]].stack()
    low, high = (finite.min() / 1.6, finite.max() * 1.6) if len(finite) else (0.1, 10)
    with style.figure(style.DOUBLE_COLUMN_MM, 30 + 5.2 * len(rows), font_pt=8.0) as fig:
        axes = fig.subplots(1, len(groups), sharey=True)
        for ax, (group, label) in zip(np.atleast_1d(axes), groups):
            for index, (adapter, dataset) in enumerate(rows):
                y = len(rows) - 1 - index
                entry = table.loc[
                    (table["group"] == group)
                    & (table["adapter"] == adapter)
                    & (table[row_field] == dataset)
                ]
                adapter_style = style.ADAPTER[adapter]
                if entry.empty or not np.isfinite(entry.iloc[0].get("ratio", np.nan)):
                    # The same outcome letters as the strips of Figures 4 and 6.
                    code = entry.iloc[0]["code"] if len(entry) else ""
                    text = (
                        "NE"
                        if len(entry) and entry.iloc[0]["not_estimable"]
                        else (code or "pending")
                    )
                    ax.text(
                        np.sqrt(low * high),
                        y,
                        text,
                        ha="center",
                        va="center",
                        color=style.INK_SECONDARY,
                    )
                    continue
                row = entry.iloc[0]
                bounded = bool(
                    row["lower_bound"]
                    or row.get("upper_bound", False)
                    or row.get("mixed_bounds", False)
                )
                ax.plot(
                    [row["low"], row["high"]],
                    [y, y],
                    color=adapter_style.color,
                    linewidth=1.2,
                )
                ax.scatter(
                    [row["ratio"]],
                    [y],
                    marker=adapter_style.marker,
                    s=22,
                    facecolor=style.WHITE if bounded else adapter_style.color,
                    edgecolor=adapter_style.color if bounded else style.INK,
                    linewidth=0.6,
                    zorder=3,
                )
                sign = (
                    "≥"
                    if row["lower_bound"]
                    else "≤"
                    if row.get("upper_bound", False)
                    else ""
                )
                if sign:
                    ax.annotate(
                        sign,
                        (row["ratio"], y),
                        xytext=(-6, 0),
                        textcoords="offset points",
                        ha="right",
                        va="center",
                    )
            style.reference_line(ax, x=1.0)
            ax.set_xscale("log")
            _major_log_ticks(ax)
            ax.set_xlim(low, high)
            _sparse_decades(ax, "x", max_labels=3)
            # Six panels share one row: a two-word title takes two lines.
            ax.set_title(label.replace(" ", "\n", 1), loc="left")
            ax.set_xlabel("Ratio to\nPictologics")
            for boundary in range(1, len(adapters)):
                ax.axhline(
                    len(rows) - boundary * len(datasets) - 0.5,
                    color="#BDBDBD",
                    linewidth=0.6,
                )
        first = np.atleast_1d(axes)[0]
        first.set_yticks(
            range(len(rows)),
            [f"{style.ADAPTER[a].label}: {d}" for a, d in reversed(rows)],
        )
        first.tick_params(axis="y", length=0)
        return style.save(fig, stem, caption=caption, alt_text=alt_text)


def outcome_map(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Figure S7: the outcome of every adapter, group and size, per pillar."""

    synthetic = frame.loc[frame["pillar"].isin(["pillar1", "pillar2"])]
    blocks = [
        (pillar, synthetic.loc[synthetic["pillar"] == pillar])
        for pillar in ("pillar1", "pillar2")
        if (synthetic["pillar"] == pillar).any()
    ]
    adapters = _adapters_present(frame)
    groups = _groups_present(frame)
    rows = [(adapter, group) for adapter in adapters for group, _ in groups]
    fills = {
        "measured": "#252525",
        "T": "#858585",
        "M": style.WHITE,
        "P": style.WHITE,
        "S": "#D9D9D9",
        "U": style.WHITE,
        "F": "#858585",
    }
    hatches = {"M": "xx", "P": "..", "U": "//", "F": "xx"}
    with style.figure(style.DOUBLE_COLUMN_MM, 30 + 4.4 * len(rows), font_pt=8.0) as fig:
        axes = np.atleast_1d(
            fig.subplots(
                1,
                len(blocks),
                sharey=True,
                width_ratios=[len(set(b["size"])) for _, b in blocks],
            )
        )
        for ax, (pillar, block) in zip(axes, blocks):
            sizes = sorted(set(block["size"].astype(int)))
            for row_index, (adapter, group) in enumerate(rows):
                y = len(rows) - 1 - row_index
                for column, size in enumerate(sizes):
                    part = block.loc[
                        (block["adapter"] == adapter)
                        & (block["group"] == group)
                        & (block["size"] == size)
                    ]
                    if part.empty:
                        continue
                    measured = part["measured"].sum()
                    codes = (
                        part.loc[~part["measured"], "task_status"]
                        .map(STATUS_CODE)
                        .dropna()
                    )
                    code = "measured" if measured == len(part) else codes.mode().iloc[0]
                    ax.add_patch(
                        Rectangle(
                            (column + 0.05, y - 0.42),
                            0.9,
                            0.84,
                            facecolor=fills[code],
                            edgecolor=style.INK,
                            linewidth=0.5,
                            hatch=hatches.get(code),
                        )
                    )
                    if code != "measured":
                        ax.text(
                            column + 0.5,
                            y,
                            code if measured == 0 else f"{code} {measured}/{len(part)}",
                            ha="center",
                            va="center",
                            bbox={
                                "facecolor": style.WHITE,
                                "edgecolor": "none",
                                "pad": 0.4,
                            }
                            if code in hatches
                            else None,
                        )
            ax.set_xlim(0, len(sizes))
            ax.set_xticks(
                [index + 0.5 for index in range(len(sizes))],
                [str(size) for size in sizes],
            )
            ax.set_xlabel("Grid edge (voxels)")
            ax.set_title(
                f"Pillar {pillar[-1]}" + (" (pooled)" if pillar == "pillar1" else ""),
                loc="left",
            )
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_ylim(-0.6, len(rows) - 0.4)
        axes[0].set_yticks(
            range(len(rows)),
            [
                f"{style.ADAPTER[a].label}, {dict(style.GROUPS)[g]}"
                for a, g in reversed(rows)
            ],
        )
        return style.save(
            fig,
            output_dir / "figureS7_outcome_map",
            caption=(
                "Outcome of every package, feature group and grid size; Pillar 1 pools its masks and noise levels. Black: all tasks "
                "measured. Letters: T timed out, M memory stop, P memory preflight skip, S "
                "skipped after an earlier stop in the branch, U not implemented, F failed; a "
                "fraction gives the measured share when some tasks were measured."
            ),
            alt_text="Tile map of benchmark outcomes by package, feature group and grid size.",
        )


def build_performance_figures(
    run_dirs: Sequence[Path], output_dir: Path, *, synthetic_size: int = 128
) -> list[dict[str, Any]]:
    return figures_from_records(
        load_records(run_dirs),
        output_dir,
        synthetic_size=synthetic_size,
        logical_cpu_count=logical_cpus(run_dirs),
    )


def figures_from_records(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    synthetic_size: int = 128,
    logical_cpu_count: int | None = None,
) -> list[dict[str, Any]]:
    """Every timing figure that the records support."""

    if frame.empty or not frame["pillar"].isin(PILLARS).any():
        return []
    summary = cells(frame)
    records = []
    if (summary["pillar"] == "pillar2").any():
        records.append(runtime_scaling(summary, output_dir))
        records.append(memory_scaling(summary, output_dir))
    if (summary["adapter"] == "pictologics").any():
        records.append(ratio_figure(summary, output_dir, synthetic_size=synthetic_size))
    if frame["pillar"].isin(["pillar1", "pillar2"]).any():
        records.append(outcome_map(frame, output_dir))
    for builder in (mask_effect, patient_runtime, ratio_by_size):
        record = builder(summary, output_dir)
        if record is not None:
            records.append(record)
    record = parallel_work(summary, output_dir, logical_cpus=logical_cpu_count)
    if record is not None:
        records.append(record)
    slope_table(summary).to_csv(
        _tables(output_dir) / "tableS5_loglog_slopes.csv", index=False
    )
    # The data behind every figure, so the figures can be drawn again.
    frame.to_csv(_tables(output_dir) / "data_task_records.csv", index=False)
    summary.to_csv(_tables(output_dir) / "data_cells.csv", index=False)
    if (summary["adapter"] == "pictologics").any():
        ratio_cases(summary, synthetic_size=synthetic_size).to_csv(
            _tables(output_dir) / "data_ratio_cases.csv", index=False
        )
        paired_ratios(summary, synthetic_size=synthetic_size).to_csv(
            _tables(output_dir) / "data_ratio_summary.csv", index=False
        )
    record = fixed_costs(frame, output_dir)
    if record is not None:
        records.append(record)
    return records


def _scatter_series(ax, data: pd.DataFrame, adapter: str, x: str, y: str) -> None:
    adapter_style = style.ADAPTER[adapter]
    values = data.loc[data[y].notna()]
    ax.scatter(
        values[x],
        values[y],
        marker=adapter_style.marker,
        s=14,
        facecolor=adapter_style.color,
        edgecolor=style.INK,
        linewidth=0.4,
        alpha=0.8,
        zorder=3,
    )
    censored = (
        data.loc[data[y].isna() & data["bound"].notna()]
        if "bound" in data
        else data.iloc[0:0]
    )
    ax.scatter(
        censored[x],
        censored["bound"],
        marker=adapter_style.marker,
        s=16,
        facecolor=style.WHITE,
        edgecolor=adapter_style.color,
        linewidth=0.8,
        zorder=4,
    )


def _panel_rows(groups, *, ncols: int = 3) -> int:
    """Rows of a grid with one panel per group and one key panel."""

    return int(np.ceil((len(groups) + 1) / ncols))


def _panel_grid(fig, groups, *, ncols: int = 3):
    axes = np.atleast_2d(fig.subplots(_panel_rows(groups, ncols=ncols), ncols))
    return axes.ravel()


def mask_effect(summary: pd.DataFrame, output_dir: Path) -> dict[str, Any] | None:
    """Figure S3: paired M2/M1, M3/M1 and M4/M1 runtime ratios (Pillar 1)."""

    data = summary.loc[(summary["pillar"] == "pillar1") & summary["median"].notna()]
    reference = data.loc[data["mask_id"] == "M1"].set_index(
        ["adapter", "group", "subject_id", "size"]
    )["median"]
    masks = [key for key in ("M2", "M3", "M4") if (data["mask_id"] == key).any()]
    if not masks or reference.empty:
        return None
    groups = _groups_present(data)
    adapters = _adapters_present(data)
    with style.figure(style.DOUBLE_COLUMN_MM, 20 + 42 * len(masks), font_pt=8.0) as fig:
        axes = np.atleast_2d(
            fig.subplots(len(masks), len(groups), sharex=True, sharey=True)
        )
        for row, mask in enumerate(masks):
            part = data.loc[data["mask_id"] == mask]
            for column, (group, label) in enumerate(groups):
                ax = axes[row, column]
                for adapter in adapters:
                    rows = []
                    for item in part.loc[
                        (part["adapter"] == adapter) & (part["group"] == group)
                    ].itertuples():
                        key = (adapter, group, item.subject_id, item.size)
                        if key in reference.index:
                            rows.append((item.size, item.median / reference[key]))
                    if not rows:
                        continue
                    frame = pd.DataFrame(rows, columns=["size", "ratio"])
                    stats = (
                        frame.groupby("size")["ratio"]
                        .agg(lambda v: float(np.exp(np.log(v).mean())))
                        .rename("mean")
                        .to_frame()
                    )
                    stats["low"], stats["high"] = (
                        frame.groupby("size")["ratio"].min(),
                        frame.groupby("size")["ratio"].max(),
                    )
                    adapter_style = style.ADAPTER[adapter]
                    ax.plot(
                        stats.index,
                        stats["mean"],
                        color=adapter_style.color,
                        linestyle=adapter_style.linestyle,
                        marker=adapter_style.marker,
                        markeredgecolor=style.INK,
                        markeredgewidth=0.5,
                    )
                    ax.vlines(
                        stats.index,
                        stats["low"],
                        stats["high"],
                        color=adapter_style.color,
                        linewidth=0.8,
                    )
                style.reference_line(ax, y=1.0)
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                _major_log_ticks(ax)
                if row == 0:
                    ax.set_title(label.replace(" ", "\n", 1), loc="left")
                if column == 0:
                    ax.set_ylabel(f"{mask} / M1 runtime")
                if row == len(masks) - 1:
                    ax.set_xlabel("Grid edge (voxels)")
                    sizes = sorted(set(data["size"].astype(int)))
                    ax.set_xticks(sizes, [str(size) for size in sizes])
                    ax.xaxis.set_minor_formatter(NullFormatter())
        handles = [
            Line2D(
                [],
                [],
                color=style.ADAPTER[k].color,
                linestyle=style.ADAPTER[k].linestyle,
                marker=style.ADAPTER[k].marker,
                markeredgecolor=style.INK,
                markeredgewidth=0.5,
                label=style.ADAPTER[k].label,
            )
            for k in adapters
        ]
        fig.legend(
            handles=handles,
            loc="outside upper center",
            ncol=len(handles),
            handlelength=3,
        )
        return style.save(
            fig,
            output_dir / "figureS3_mask_effect",
            caption=(
                "Runtime of each mask relative to M1 on the same scene, noise level and grid "
                "(Pillar 1). Markers: the geometric mean over the three noise levels; vertical "
                "lines: their range. The masks differ in ROI voxels, bounding box, surface and "
                "components (Figure S1), and M3 also changes the FBN32 bin width, so a ratio "
                "is the effect of the defined mask, not of geometry alone."
            ),
            alt_text="Grid of log-scale line charts of runtime ratios of masks M2, M3 and M4 to mask M1 against grid size, per feature group and package.",
        )


def patient_runtime(summary: pd.DataFrame, output_dir: Path) -> dict[str, Any] | None:
    """Figure S4: runtime per patient against R, per modality and group (Pillar 3)."""

    data = summary.loc[summary["pillar"] == "pillar3"]
    if data.empty:
        return None
    groups = _groups_present(data)
    modalities = [m for m in ("ct", "mri", "pet") if (data["modality"] == m).any()]
    adapters = _adapters_present(data)
    with style.figure(
        style.DOUBLE_COLUMN_MM, 20 + 40 * len(groups), font_pt=8.0
    ) as fig:
        axes = np.atleast_2d(fig.subplots(len(groups), len(modalities), sharey=True))
        for row, (group, label) in enumerate(groups):
            for column, modality in enumerate(modalities):
                ax = axes[row, column]
                part = data.loc[
                    (data["group"] == group) & (data["modality"] == modality)
                ]
                for adapter in adapters:
                    _scatter_series(
                        ax, part.loc[part["adapter"] == adapter], adapter, "R", "median"
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                _major_log_ticks(ax)
                ax.autoscale_view()
                _narrow_log_ticks(ax, "x")
                if row == 0:
                    ax.set_title(style.MODALITY_LABEL[modality], loc="left")
                if column == 0:
                    ax.set_ylabel(f"{label}\nwall time (s)")
                if row == len(groups) - 1:
                    ax.set_xlabel("ROI voxels R")
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                color=style.ADAPTER[k].color,
                marker=style.ADAPTER[k].marker,
                markeredgecolor=style.INK,
                markeredgewidth=0.4,
                label=style.ADAPTER[k].label,
            )
            for k in adapters
        ]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
        return style.save(
            fig,
            output_dir / "figureS4_patient_runtime",
            caption=(
                "Runtime per patient in the IBSI 2 Phase 3 cohort (Pillar 3), against ROI "
                "voxels R. Each point is one patient and modality: the median of three "
                "processes. Open markers are timeouts at their lower bound. The points show "
                "an association across real cases, not a scaling law."
            ),
            alt_text="Grid of log-log scatter plots of runtime per patient against region of interest voxels, per feature group and modality.",
        )


def parallel_work(
    summary: pd.DataFrame, output_dir: Path, *, logical_cpus: int | None = None
) -> dict[str, Any] | None:
    """Figure S5: CPU time divided by wall time against R."""

    data = summary.loc[summary["cpu_ratio"].notna()].copy()
    if data.empty:
        return None
    # The core-count line appears only when the run records the count.  The
    # CPU clock is coarse below 10 ms (15.6 ms steps on Windows), so only the
    # longer calls set the axis; a shorter call above it sits at its top.
    reliable = data.loc[data["median"] >= 0.01, "cpu_ratio"]
    top = max(
        float(logical_cpus or 0), float(reliable.max()) if len(reliable) else 0.0, 2.0
    )
    data["cpu_ratio"] = data["cpu_ratio"].clip(upper=top * 1.08)
    groups = _groups_present(data)
    adapters = _adapters_present(data)
    with style.figure(
        style.DOUBLE_COLUMN_MM, 60 * _panel_rows(groups), font_pt=8.0
    ) as fig:
        axes = _panel_grid(fig, groups)
        for ax, (group, label) in zip(axes, groups):
            part = data.loc[data["group"] == group]
            short = part["median"] < 0.01
            for adapter in adapters:
                adapter_style = style.ADAPTER[adapter]
                rows = part.loc[part["adapter"] == adapter]
                ax.scatter(
                    rows.loc[~short, "R"],
                    rows.loc[~short, "cpu_ratio"],
                    marker=adapter_style.marker,
                    s=14,
                    facecolor=adapter_style.color,
                    edgecolor=style.INK,
                    linewidth=0.4,
                    zorder=3,
                )
                ax.scatter(
                    rows.loc[short, "R"],
                    rows.loc[short, "cpu_ratio"],
                    marker=adapter_style.marker,
                    s=14,
                    facecolor=style.WHITE,
                    edgecolor=adapter_style.color,
                    linewidth=0.7,
                    zorder=3,
                )
            style.reference_line(ax, y=1.0, label="one core")
            if logical_cpus and logical_cpus > 1:
                style.reference_line(
                    ax, y=float(logical_cpus), label=f"{logical_cpus} logical cores"
                )
            ax.set_xscale("log")
            _major_log_ticks(ax)
            ax.set_ylim(0, top * 1.12)
            ax.set_title(label, loc="left")
            ax.set_xlabel("ROI voxels R")
            ax.set_ylabel("CPU time / wall time")
        key = axes[len(groups)]
        key.axis("off")
        handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                color=style.ADAPTER[k].color,
                marker=style.ADAPTER[k].marker,
                markeredgecolor=style.INK,
                markeredgewidth=0.4,
                label=style.ADAPTER[k].label,
            )
            for k in adapters
        ]
        key.legend(handles=handles, loc="upper left", title="Package")
        key.text(
            0.0,
            0.2,
            "open markers: calls below 10 ms,\nwhere the CPU clock is coarse;\n"
            "a higher value sits at the top",
            transform=key.transAxes,
            ha="left",
            va="top",
        )
        for ax in axes[len(groups) + 1 :]:
            ax.axis("off")
        return style.save(
            fig,
            output_dir / "figureS5_parallel_work",
            caption=(
                "Parallel work: CPU time divided by wall time of the calculation calls (median "
                "of three processes), against ROI voxels R, for all cases. A value above 1 means "
                "the call used more than one core. Pictologics runs explicit parallel kernels; "
                "MEDimage and MIRP morphology use threads of their maths libraries. Open markers "
                "are calls shorter than 10 ms, where the CPU clock resolution limits the ratio. "
                "Only the longer calls set the axis; a short call above it sits at its top edge."
            ),
            alt_text="Scatter plots of CPU time divided by wall time against region of interest voxels for five feature groups and five packages.",
        )


def ratio_by_size(summary: pd.DataFrame, output_dir: Path) -> dict[str, Any] | None:
    """Figure S6: runtime relative to Pictologics at every Pillar 2 size."""

    data = summary.loc[(summary["pillar"] == "pillar2") & summary["median"].notna()]
    if data.empty:
        return None
    reference = data.loc[data["adapter"] == "pictologics"].set_index(
        ["group", "case_id"]
    )
    groups = _groups_present(data)
    adapters = [key for key in _adapters_present(data) if key != "pictologics"]
    with style.figure(
        style.DOUBLE_COLUMN_MM, 60 * _panel_rows(groups), font_pt=8.0
    ) as fig:
        axes = _panel_grid(fig, groups)
        for ax, (group, label) in zip(axes, groups):
            part = data.loc[data["group"] == group]
            for adapter in adapters:
                rows = []
                for item in part.loc[part["adapter"] == adapter].itertuples():
                    if (group, item.case_id) not in reference.index:
                        continue
                    base = reference.loc[(group, item.case_id)]
                    pairs = [
                        a / b for a, b in zip(sorted(item.times), sorted(base["times"]))
                    ]
                    rows.append(
                        (item.R, item.median / base["median"], min(pairs), max(pairs))
                    )
                if not rows:
                    continue
                frame = pd.DataFrame(
                    rows, columns=["R", "ratio", "low", "high"]
                ).sort_values("R")
                adapter_style = style.ADAPTER[adapter]
                ax.plot(
                    frame["R"],
                    frame["ratio"],
                    color=adapter_style.color,
                    linestyle=adapter_style.linestyle,
                    marker=adapter_style.marker,
                    markeredgecolor=style.INK,
                    markeredgewidth=0.5,
                )
                ax.vlines(
                    frame["R"],
                    frame["low"],
                    frame["high"],
                    color=adapter_style.color,
                    linewidth=0.8,
                )
            style.reference_line(ax, y=1.0, label="same speed")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _major_log_ticks(ax)
            ax.set_title(label, loc="left")
            ax.set_xlabel("ROI voxels R")
            ax.set_ylabel("Ratio to Pictologics")
        key = axes[len(groups)]
        key.axis("off")
        handles = [
            Line2D(
                [],
                [],
                color=style.ADAPTER[k].color,
                linestyle=style.ADAPTER[k].linestyle,
                marker=style.ADAPTER[k].marker,
                markeredgecolor=style.INK,
                markeredgewidth=0.5,
                label=style.ADAPTER[k].label,
            )
            for k in adapters
        ]
        key.legend(handles=handles, loc="upper left", title="Package", handlelength=3)
        for ax in axes[len(groups) + 1 :]:
            ax.axis("off")
        return style.save(
            fig,
            output_dir / "figureS6_ratio_by_size",
            caption=(
                "Runtime relative to Pictologics at every grid of the whole-anatomy series "
                "(Pillar 2). Markers: the ratio of the two medians of three processes; "
                "vertical lines: the range of the three repeat-wise ratios."
            ),
            alt_text="Log-log line charts of runtime relative to Pictologics against region of interest voxels for four packages and five feature groups.",
        )


def fixed_costs(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any] | None:
    """Figure S8: work outside the timer, per package."""

    measured = frame.loc[frame["measured"]].copy()
    if measured.empty:
        return None
    measured["worker_mb"] = measured["worker_ready_rss_bytes"].astype(float) / 1e6
    panels = (
        ("worker_mb", "Worker memory before the first call (MB)"),
        ("load_sec", "Load time (s)"),
        ("setup_sec", "Setup time before the timer (s)"),
    )
    adapters = _adapters_present(measured)
    rng = np.random.default_rng(0)
    with style.figure(style.DOUBLE_COLUMN_MM, 62, font_pt=8.0) as fig:
        axes = fig.subplots(1, len(panels), sharey=True)
        for ax, (column, label) in zip(axes, panels):
            for index, adapter in enumerate(adapters):
                y = len(adapters) - 1 - index
                values = (
                    measured.loc[measured["adapter"] == adapter, column]
                    .astype(float)
                    .dropna()
                )
                values = values[values > 0]
                adapter_style = style.ADAPTER[adapter]
                ax.scatter(
                    values,
                    y + rng.uniform(-0.22, 0.22, len(values)),
                    marker=adapter_style.marker,
                    s=6,
                    color=adapter_style.color,
                    alpha=0.4,
                    linewidth=0,
                )
                if len(values):
                    ax.scatter(
                        [values.median()],
                        [y],
                        marker=adapter_style.marker,
                        s=34,
                        facecolor=adapter_style.color,
                        edgecolor=style.INK,
                        linewidth=0.6,
                        zorder=3,
                    )
            ax.set_xscale("log")
            _major_log_ticks(ax)
            ax.autoscale_view()
            _narrow_log_ticks(ax)
            ax.set_xlabel(label)
        axes[0].set_yticks(
            range(len(adapters)), [style.ADAPTER[k].label for k in reversed(adapters)]
        )
        axes[0].tick_params(axis="y", length=0)
        return style.save(
            fig,
            output_dir / "figureS8_fixed_costs",
            caption=(
                "Work outside the timer for every measured task: the worker memory before the "
                "first call, the load time from process start to the files being read, and the "
                "setup time from then to the first call (crop and preparation). Small points: "
                "tasks; large markers: the median. These costs are recorded but are not part of "
                "the calculation time."
            ),
            alt_text="Three strip plots of worker memory, load time and setup time per package, on log scales.",
        )


def _tables(output_dir: Path) -> Path:
    path = output_dir / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _censored_slope(
    x: np.ndarray, y: np.ndarray, censored_x: np.ndarray, censored_y: np.ndarray
) -> tuple[float, float]:
    """Slope and standard error of log time on log R; a timeout counts as at least its bound.

    The model is a straight line with normal errors on the log scale.  A
    measured cell adds its density and a timed-out cell the chance of a value
    above its bound (a Tobit model).  The standard error comes from the
    numerical Hessian of the log-likelihood; it is NaN when that is singular.
    """

    from scipy import optimize, stats

    def negative_log_likelihood(theta: np.ndarray) -> float:
        a, b, log_sigma = theta
        # The lower limit on the spread keeps a perfect line from collapsing it.
        sigma = np.exp(max(log_sigma, np.log(1e-3)))
        return -float(
            stats.norm.logpdf(y, a + b * x, sigma).sum()
            + stats.norm.logsf(censored_y, a + b * censored_x, sigma).sum()
        )

    slope, intercept = np.polyfit(x, y, 1)
    spread = max(float(np.std(y - (intercept + slope * x))), 1e-3)
    fit = optimize.minimize(
        negative_log_likelihood,
        [intercept, slope, np.log(spread)],
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000},
    )
    if not fit.success:
        return np.nan, np.nan
    step = 1e-4
    unit = np.eye(3) * step
    hessian = np.array(
        [
            [
                (
                    negative_log_likelihood(fit.x + unit[i] + unit[j])
                    - negative_log_likelihood(fit.x + unit[i] - unit[j])
                    - negative_log_likelihood(fit.x - unit[i] + unit[j])
                    + negative_log_likelihood(fit.x - unit[i] - unit[j])
                )
                / (4 * step**2)
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    try:
        variance = float(np.linalg.inv(hessian)[1, 1])
    except np.linalg.LinAlgError:
        variance = np.nan
    return float(fit.x[1]), float(np.sqrt(variance)) if variance > 0 else np.nan


def slope_table(
    summary: pd.DataFrame, *, pillar: str = "pillar2", minimum_size: int = 64
) -> pd.DataFrame:
    """Table S5: log-log slope of time against R, with a 95 percent interval.

    32^3 is left out as a coarse-grid point.  The primary fit gives every
    package of a group the same sizes: the sizes from ``minimum_size`` where all
    packages of the group have a complete median.  The censored fit uses every
    size from ``minimum_size`` and counts a timeout as at least its bound.  A
    memory stop or a skip has no time bound, so it stays out of both fits.
    """

    from scipy import stats

    rows = []
    data = summary.loc[
        (summary["pillar"] == pillar) & (summary["size"] >= minimum_size)
    ]
    for group, part in data.groupby("group", sort=False):
        complete = part.loc[part["median"].notna()]
        packages = [a for a in style.ADAPTER_ORDER if a in set(complete["adapter"])]
        common = set.intersection(
            *(set(complete.loc[complete["adapter"] == a, "size"]) for a in packages)
        )
        for adapter in packages:
            mine = part.loc[part["adapter"] == adapter]
            observed = mine.loc[mine["median"].notna()]
            timed_out = mine.loc[mine["median"].isna() & mine["bound"].notna()]
            fitted = observed.loc[observed["size"].isin(common)]
            entry = {
                "package": style.ADAPTER[adapter].label,
                "group": style.GROUP_LABEL[group],
                "slope": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "sizes": ", ".join(str(int(size)) for size in sorted(common)),
                "n": len(fitted),
                "slope_censored": np.nan,
                "ci_low_censored": np.nan,
                "ci_high_censored": np.nan,
                "n_censored_fit": len(observed) + len(timed_out),
                "n_timeouts": len(timed_out),
            }
            if len(fitted) >= 3:
                x = np.log10(fitted["R"].to_numpy())
                y = np.log10(fitted["median"].to_numpy())
                fit = stats.linregress(x, y)
                margin = stats.t.ppf(0.975, len(fitted) - 2) * fit.stderr
                entry.update(
                    slope=round(fit.slope, 3),
                    ci_low=round(fit.slope - margin, 3),
                    ci_high=round(fit.slope + margin, 3),
                )
            if len(observed) >= 2 and len(observed) + len(timed_out) >= 3:
                slope, error = _censored_slope(
                    np.log10(observed["R"].to_numpy()),
                    np.log10(observed["median"].to_numpy()),
                    np.log10(timed_out["R"].to_numpy()),
                    np.log10(timed_out["bound"].to_numpy()),
                )
                if np.isfinite(slope):
                    entry.update(
                        slope_censored=round(slope, 3),
                        ci_low_censored=round(slope - 1.96 * error, 3),
                        ci_high_censored=round(slope + 1.96 * error, 3),
                    )
            rows.append(entry)
    return pd.DataFrame(
        rows,
        columns=[
            "package",
            "group",
            "slope",
            "ci_low",
            "ci_high",
            "sizes",
            "n",
            "slope_censored",
            "ci_low_censored",
            "ci_high_censored",
            "n_censored_fit",
            "n_timeouts",
        ],
    )

"""Compare the machines that sent results, without pooling absolute times.

Absolute times depend on the hardware and the operating system, so each
machine gives its own ratios to Pictologics, and the machines sit side by side.
The ratio figure needs a full run of the five libraries.  The hardware figure
shows the Pictologics times of every machine, from a full run or from a
Pictologics-only run, one line for each machine.

Usage: ``python -m bench.figures.community --submissions submissions --output-dir DIR``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from bench.figures import style
from bench.figures.performance import (
    _groups_present,
    _major_log_ticks,
    cells,
    load_records,
    paired_ratios,
    ratio_dot_plot,
)

PILLARS = ("pillar1_morphology", "pillar2_whole_anatomy", "pillar3_ibsi2_phase3")
# Markers and line styles tell the machines apart in black and grey.
MACHINE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
MACHINE_LINES = ("-", "--", ":", "-.")
MACHINE_COLORS = (style.INK, style.INK_SECONDARY)
RUN_KIND = {"all": "full run", "pictologics": "Pictologics-only run"}


def _scope(folder: Path) -> str:
    summary = json.loads((folder / "submission.json").read_text("utf-8"))
    # A package without a scope comes from a full run of the five libraries.
    return str(summary.get("scope", "all"))


def latest_packages(submissions: Path, *, scope: str = "all") -> dict[str, Path]:
    """The newest package of each machine; ``scope="any"`` takes both kinds."""

    latest: dict[str, Path] = {}
    # Folder names sort by date: <date> and <date>-pictologics.
    for summary in sorted(submissions.glob("*/*/submission.json")):
        if scope in ("any", _scope(summary.parent)):
            latest[summary.parent.parent.name] = summary.parent
    return latest


def machine_ratios(submissions: Path) -> pd.DataFrame:
    tables = []
    for machine, folder in latest_packages(submissions).items():
        records = load_records([folder / pillar for pillar in PILLARS])
        table = paired_ratios(cells(records))
        submission = json.loads((folder / "submission.json").read_text("utf-8"))
        table["machine"] = machine
        table["platform_key"] = submission["platform_key"]
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def community_figure(
    submissions: Path, output_dir: Path, *, dataset: str = "Pillar 3, CT"
) -> dict[str, Any] | None:
    table = machine_ratios(submissions)
    if table.empty:
        return None
    table = table.loc[table["dataset"] == dataset]
    return ratio_dot_plot(
        table,
        "machine",
        output_dir / "figure_community_runtime_ratio",
        caption=(
            f"Runtime relative to Pictologics on each submitted machine ({dataset}). "
            "Each machine compares the packages on its own hardware; absolute times "
            "are never pooled across machines. The marker is the geometric mean over "
            "cases of the ratio of medians, and the line is its 95 percent bootstrap "
            "interval. An open marker with ≥ is a lower bound (a package task timed "
            "out), with ≤ an upper bound (a Pictologics task timed out)."
        ),
        alt_text=(
            "Dot plot of runtime relative to Pictologics, on a log scale, with one row "
            "for each package and submitted machine, in five feature-group panels."
        ),
    )


def pictologics_times(submissions: Path) -> pd.DataFrame:
    """The Pictologics cells of the newest package of each machine, any scope."""

    tables = []
    for machine, folder in latest_packages(submissions, scope="any").items():
        records = load_records([folder / pillar for pillar in PILLARS])
        if records.empty:
            continue
        records = records.loc[records["adapter"] == "pictologics"]
        if records.empty:
            continue
        table = cells(records)
        submission = json.loads((folder / "submission.json").read_text("utf-8"))
        host_file = folder / "host_profile.json"
        host = json.loads(host_file.read_text("utf-8")) if host_file.is_file() else {}
        table["machine"] = machine
        table["machine_label"] = host.get("machine_label") or machine
        table["platform_key"] = submission.get("platform_key")
        table["run_kind"] = RUN_KIND.get(_scope(folder), _scope(folder))
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _machine_style(index: int) -> dict[str, Any]:
    return {
        "marker": MACHINE_MARKERS[index % len(MACHINE_MARKERS)],
        "linestyle": MACHINE_LINES[index % len(MACHINE_LINES)],
        "color": MACHINE_COLORS[index % len(MACHINE_COLORS)],
    }


def pictologics_hardware_figure(
    submissions: Path, output_dir: Path, *, pillar: str = "pillar2"
) -> dict[str, Any] | None:
    """Pictologics wall time against ROI size, one line for each machine."""

    table = pictologics_times(submissions)
    if table.empty:
        return None
    table = table.loc[table["pillar"] == pillar]
    if table.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["machine", "machine_label", "platform_key", "run_kind", "group"]
    table[[*columns, "size", "R", "n_measured", "median", "bound", "code"]].sort_values(
        ["machine", "group", "R"]
    ).to_csv(output_dir / "table_community_pictologics_times.csv", index=False)
    machines = (
        table.drop_duplicates("machine")
        .sort_values("machine")[
            ["machine", "machine_label", "platform_key", "run_kind"]
        ]
        .to_dict("records")
    )
    groups = _groups_present(table)
    rows = int(np.ceil(len(groups) / 3))
    # Two lines in the key for each machine: its ID and its hardware.
    key_mm = 8.0 + 8.0 * len(machines)
    with style.figure(style.DOUBLE_COLUMN_MM, 62.0 * rows + key_mm, font_pt=8.0) as fig:
        grid = fig.add_gridspec(rows + 1, 3, height_ratios=[62.0] * rows + [key_mm])
        positive = table["median"].replace(0, np.nan).dropna()
        low = positive.min() / 2 if len(positive) else 1e-4
        high = positive.max() * 4 if len(positive) else 1.0
        for index, (group, label) in enumerate(groups):
            row, column = divmod(index, 3)
            ax = fig.add_subplot(grid[row, column])
            data = table.loc[table["group"] == group]
            for number, machine in enumerate(machines):
                part = data.loc[data["machine"] == machine["machine"]].sort_values("R")
                complete = part.loc[part["median"].notna()]
                ax.plot(
                    complete["R"],
                    complete["median"],
                    markeredgecolor=style.INK,
                    markeredgewidth=0.5,
                    markerfacecolor=style.WHITE if number % 2 else None,
                    linewidth=1.0,
                    markersize=4,
                    **_machine_style(number),
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            _major_log_ticks(ax)
            ax.set_ylim(low, high)
            ax.set_title(label, loc="left")
            ax.set_xlabel("ROI voxels R")
            if column == 0:
                ax.set_ylabel("Wall time per call (s)")
        key = fig.add_subplot(grid[rows, :])
        key.axis("off")
        handles = [
            Line2D(
                [],
                [],
                markeredgecolor=style.INK,
                markeredgewidth=0.5,
                markerfacecolor=style.WHITE if number % 2 else None,
                linewidth=1.0,
                markersize=4,
                label=(
                    f"{machine['machine']} ({machine['platform_key']}, "
                    f"{machine['run_kind']})\n{machine['machine_label']}"
                ),
                **_machine_style(number),
            )
            for number, machine in enumerate(machines)
        ]
        key.legend(
            handles=handles,
            loc="upper left",
            handlelength=3.2,
            frameon=False,
            title="Machine",
            alignment="left",
        )
        return style.save(
            fig,
            output_dir / "figure_community_pictologics_hardware",
            caption=(
                "Pictologics wall time per call on each submitted machine (Pillar 2, "
                "dense whole-anatomy mask), one line for each machine and one panel "
                "for each feature group. A marker is the median of the three process "
                "medians, drawn only when all three repeats were measured. Absolute "
                "times are shown machine by machine and never pooled. A full run times "
                "Pictologics between the other libraries; a Pictologics-only run times "
                "it alone, on the same tasks."
            ),
            alt_text=(
                "Log-log line charts of Pictologics wall time per call against region "
                "of interest voxels, one line for each submitted machine, one chart "
                "for each feature group."
            ),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", default="submissions")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    submissions, output_dir = Path(args.submissions), Path(args.output_dir)
    for record in (
        community_figure(submissions, output_dir),
        pictologics_hardware_figure(submissions, output_dir),
    ):
        print("no submitted packages" if record is None else record["figure_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

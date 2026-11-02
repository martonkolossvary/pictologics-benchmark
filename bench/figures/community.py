"""Compare the machines that sent results, without pooling absolute times.

Absolute times depend on the hardware and the operating system, so each
machine gives its own ratios to Pictologics, and the machines sit side by side.

Usage: ``python -m bench.figures.community --submissions submissions --output-dir DIR``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bench.figures.performance import cells, load_records, paired_ratios, ratio_dot_plot

PILLARS = ("pillar1_morphology", "pillar2_whole_anatomy", "pillar3_ibsi2_phase3")


def latest_packages(submissions: Path) -> dict[str, Path]:
    """The newest package of each machine."""

    latest: dict[str, Path] = {}
    for summary in sorted(submissions.glob("*/*/submission.json")):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", default="submissions")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    record = community_figure(Path(args.submissions), Path(args.output_dir))
    print("no submitted packages" if record is None else record["figure_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

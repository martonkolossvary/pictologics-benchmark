"""Conformance figures (Figure 2, Figure 3, Figure S9) and their tables.

Every value comes from the frozen release tables of IBSI 1, IBSI 2 Phase 1
and IBSI 2 Phase 2.  The figures use fixed denominators, so an adapter that
does not support a check is never shown as a high pass rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from bench.figures import style

IBSI1_FAMILIES = (
    ("morphology", "Morphology"),
    ("local_intensity", "Local\nintensity"),
    ("intensity", "Intensity\nstatistics"),
    ("histogram", "Intensity\nhistogram"),
    ("ivh", "IVH"),
    ("glcm", "GLCM"),
    ("glrlm", "GLRLM"),
    ("glszm", "GLSZM"),
    ("gldzm", "GLDZM"),
    ("ngtdm", "NGTDM"),
    ("ngldm", "NGLDM"),
)
PHASE1_FILTERS = (
    ("mean", "Mean"),
    ("log", "LoG"),
    ("laws", "Laws"),
    ("wavelet", "Wavelet"),
    ("simoncelli", "Simoncelli"),
    ("riesz_log", "Riesz\nLoG"),
    ("riesz_simoncelli", "Riesz\nSimoncelli"),
)
FILTER_NAMES = {
    "none": "none",
    "mean": "mean",
    "log": "LoG",
    "laws": "Laws",
    "gabor": "Gabor",
    "wavelet": "wavelet",
    "simoncelli": "Simoncelli",
}


@dataclass(frozen=True)
class Outcomes:
    adapter: str
    version: str
    passed: int
    failed: int
    not_supported: int
    not_evaluated: int

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.not_supported + self.not_evaluated


@dataclass(frozen=True)
class Cell:
    denominator: int
    evaluated: int
    passed: int
    failed: int


def _flag(values: pd.Series) -> pd.Series:
    """Read a CSV boolean column, whether pandas parsed it or not."""

    return values.astype(str).str.strip().str.lower() == "true"


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    rank = {key: index for index, key in enumerate(style.ADAPTER_ORDER)}
    return frame.assign(_rank=frame["adapter"].map(rank)).sort_values("_rank")


def ibsi1_outcomes(report_dir: Path) -> list[Outcomes]:
    frame = _ordered(pd.read_csv(report_dir / "compliance_summary.csv"))
    rows = []
    for row in frame.itertuples():
        total = int(row.defined_standardized_checks)
        not_evaluated = int(row.missing) + int(row.nonfinite) + int(row.ambiguous)
        rows.append(
            Outcomes(
                row.adapter,
                str(row.software_version),
                int(row.passed),
                int(row.failed),
                total - int(row.passed) - int(row.failed) - not_evaluated,
                not_evaluated,
            )
        )
    return rows


def phase1_outcomes(report_dir: Path) -> list[Outcomes]:
    frame = _ordered(pd.read_csv(report_dir / "phase1_strict_3d_summary.csv"))
    return [
        Outcomes(
            row.adapter,
            str(row.generator_version),
            int(row.passed),
            int(row.failed),
            int(row.standardized_reference_tests) - int(row.evaluated),
            0,
        )
        for row in frame.itertuples()
    ]


def phase2_outcomes(report_dir: Path) -> list[Outcomes]:
    frame = _ordered(pd.read_csv(report_dir / "compliance_phase2_b_summary.csv"))
    return [
        Outcomes(
            row.adapter,
            str(row.software_version),
            int(row.passed_standardized_checks),
            int(row.failed_standardized_checks),
            int(row.standardized_b_checks) - int(row.evaluated_standardized_checks),
            0,
        )
        for row in frame.itertuples()
    ]


def _cells(
    frame: pd.DataFrame,
    column: str,
    keys: Sequence[str],
    denominator: str,
    evaluated: str,
    passed: str,
    failed: str,
) -> dict[tuple[str, str], Cell]:
    cells = {}
    for row in frame.itertuples():
        key = getattr(row, column)
        if key in keys:
            cells[(row.adapter, key)] = Cell(
                int(getattr(row, denominator)),
                int(getattr(row, evaluated)),
                int(getattr(row, passed)),
                int(getattr(row, failed)),
            )
    return cells


def ibsi1_cells(report_dir: Path) -> dict[tuple[str, str], Cell]:
    frame = pd.read_csv(report_dir / "compliance_by_family.csv")
    keys = [key for key, _ in IBSI1_FAMILIES]
    return _cells(
        frame,
        "family",
        keys,
        "standardized_checks",
        "evaluated_checks",
        "passed_checks",
        "failed_checks",
    )


def phase1_cells(report_dir: Path) -> dict[tuple[str, str], Cell]:
    frame = pd.read_csv(report_dir / "phase1_strict_3d_by_filter.csv")
    keys = [key for key, _ in PHASE1_FILTERS]
    return _cells(
        frame,
        "filter_family",
        keys,
        "standardized_reference_tests",
        "evaluated",
        "passed",
        "failed",
    )


def phase2_columns(report_dir: Path) -> list[tuple[str, str]]:
    frame = pd.read_csv(report_dir / "compliance_phase2_b_by_configuration.csv")
    published = frame.loc[_flag(frame["published_configuration"])]
    configurations = published.drop_duplicates("configuration")
    order = sorted(
        configurations["configuration"], key=lambda value: int(value.split(".")[0])
    )
    family = dict(zip(configurations["configuration"], configurations["filter_family"]))
    # The dagger marks 5.B, which averages 2D Gabor responses over three planes.
    return [
        (
            key,
            f"{key}{' †' if family[key] == 'gabor' else ''}\n{FILTER_NAMES.get(family[key], family[key])}",
        )
        for key in order
    ]


def phase2_cells(report_dir: Path) -> dict[tuple[str, str], Cell]:
    frame = pd.read_csv(report_dir / "compliance_phase2_b_by_configuration.csv")
    keys = [key for key, _ in phase2_columns(report_dir)]
    return _cells(
        frame,
        "configuration",
        keys,
        "standardized_reference_checks",
        "evaluated_standardized_checks",
        "passed_standardized_checks",
        "failed_standardized_checks",
    )


def _versions(outcomes: Sequence[Outcomes]) -> dict[str, str]:
    return {row.adapter: row.version for row in outcomes}


def draw_outcome_bars(ax, outcomes: Sequence[Outcomes], *, title: str) -> None:
    """Horizontal stacked bars of fixed-denominator outcomes, one per adapter."""

    total = outcomes[0].total
    if any(row.total != total for row in outcomes):
        raise ValueError("outcome bars need one fixed denominator")
    for index, row in enumerate(outcomes):
        y = len(outcomes) - 1 - index
        left = 0
        notes = []
        for key in ("passed", "failed", "not_supported", "not_evaluated"):
            count = getattr(row, key)
            if not count:
                continue
            outcome = style.OUTCOME[key]
            ax.barh(
                y,
                count,
                left=left,
                height=0.62,
                color=outcome.fill,
                edgecolor=style.INK,
                linewidth=style.OUTLINE_PT,
                hatch=outcome.hatch or None,
            )
            if count / total >= 0.12:
                ax.text(
                    left + count / 2,
                    y,
                    f"{count} {outcome.code}".strip(),
                    ha="center",
                    va="center",
                    color=outcome.text_color,
                    bbox=None
                    if outcome.key == "passed"
                    else {
                        "facecolor": outcome.fill if not outcome.hatch else style.WHITE,
                        "edgecolor": "none",
                        "pad": 0.6,
                    },
                )
            elif key != "passed":
                notes.append(f"{count} {outcome.code}")
            left += count
        ax.text(
            total * 1.01,
            y,
            "  ".join([f"{row.passed}/{total}"] + notes),
            ha="left",
            va="center",
            color=style.INK,
        )
    ax.set_yticks(
        range(len(outcomes)),
        [style.adapter_label(row.adapter, row.version) for row in reversed(outcomes)],
    )
    ax.set_xlim(0, total)
    ax.set_xticks(np.linspace(0, total, 5), ["0", "25", "50", "75", "100"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left")


def outcome_legend(
    fig, *, keys: Sequence[str] = ("passed", "failed", "not_supported")
) -> None:
    handles = [
        Patch(
            facecolor=style.OUTCOME[key].fill,
            edgecolor=style.INK,
            linewidth=style.OUTLINE_PT,
            hatch=style.OUTCOME[key].hatch or None,
            label=style.OUTCOME[key].label
            + (f" ({style.OUTCOME[key].code})" if style.OUTCOME[key].code else ""),
        )
        for key in keys
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))


def _matrix_fill(cell: Cell) -> tuple[str, str, str]:
    """Return fill, hatch and text colour for one matrix cell."""

    if cell.evaluated == 0:
        outcome = style.OUTCOME["not_supported"]
        return outcome.fill, outcome.hatch, style.INK
    share = cell.passed / cell.denominator
    if share == 1.0:
        return "#252525", "", style.WHITE
    if share >= 0.5:
        return "#858585", "", style.INK
    return style.WHITE, "", style.INK


def draw_matrix(
    ax,
    cells: Mapping[tuple[str, str], Cell],
    columns: Sequence[tuple[str, str]],
    versions: Mapping[str, str],
    *,
    title: str,
) -> None:
    """Adapter-by-family matrix with the pass count in each cell."""

    adapters = [key for key in style.ADAPTER_ORDER if key in versions]
    for row_index, adapter in enumerate(adapters):
        for column_index, (key, _) in enumerate(columns):
            cell = cells.get((adapter, key))
            if cell is None:
                continue
            fill, hatch, text_color = _matrix_fill(cell)
            ax.add_patch(
                Rectangle(
                    (column_index + 0.04, row_index + 0.06),
                    0.92,
                    0.88,
                    facecolor=fill,
                    edgecolor=style.INK,
                    linewidth=style.OUTLINE_PT,
                    hatch=hatch or None,
                )
            )
            # The second line names the part that did not pass.
            missing = cell.denominator - cell.evaluated
            notes = [f"{cell.failed} F"] * bool(cell.failed) + [f"{missing} NS"] * bool(
                missing and cell.evaluated
            )
            label = "NS" if cell.evaluated == 0 else str(cell.passed)
            if notes:
                ax.text(
                    column_index + 0.5,
                    row_index + 0.72,
                    ", ".join(notes),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=style.MIN_FONT_PT,
                )
            ax.text(
                column_index + 0.5,
                row_index + (0.36 if notes else 0.5),
                label,
                ha="center",
                va="center",
                color=text_color,
                bbox={"facecolor": style.WHITE, "edgecolor": "none", "pad": 0.8}
                if hatch
                else None,
            )
    denominators = {
        key: next(
            cell.denominator
            for (adapter, column), cell in cells.items()
            if column == key
        )
        for key, _ in columns
    }
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(adapters), 0)
    ax.set_xticks(
        [index + 0.5 for index in range(len(columns))],
        [f"{label}\nn = {denominators[key]}" for key, label in columns],
    )
    ax.xaxis.tick_top()
    ax.set_yticks(
        [index + 0.5 for index in range(len(adapters))],
        [style.adapter_label(adapter, versions[adapter]) for adapter in adapters],
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left")


def matrix_legend(fig) -> None:
    handles = [
        Patch(
            facecolor="#252525",
            edgecolor=style.INK,
            linewidth=style.OUTLINE_PT,
            label="All passed",
        ),
        Patch(
            facecolor="#858585",
            edgecolor=style.INK,
            linewidth=style.OUTLINE_PT,
            label="50 to 99 % passed",
        ),
        Patch(
            facecolor=style.WHITE,
            edgecolor=style.INK,
            linewidth=style.OUTLINE_PT,
            label="Below 50 % passed",
        ),
        Patch(
            facecolor=style.WHITE,
            edgecolor=style.INK,
            linewidth=style.OUTLINE_PT,
            hatch="//",
            label="Not supported (NS)",
        ),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))


def _ratio_rows(report_dir: Path, phase: str) -> pd.DataFrame:
    """Error divided by tolerance for every evaluated check.

    A zero-tolerance check passes only with zero error, so it becomes an exact
    zero, or an infinite ratio when its error is positive; none is dropped.
    """

    if phase == "phase1":
        frame = pd.read_csv(report_dir / "phase1_detail.csv")
        error_column, tolerance_column = "max_abs_error", "voxel_tolerance"
        if "strict_3d_scope" in frame:
            frame = frame.loc[_flag(frame["strict_3d_scope"])]
    else:
        frame = pd.read_csv(report_dir / "compliance_detail.csv")
        error_column, tolerance_column = "comparison_error", "tolerance"
        frame = frame.loc[frame["configuration"].astype(str).str.endswith(".B")]
    if error_column not in frame or tolerance_column not in frame:
        return pd.DataFrame({"adapter": [], "ratio": []})
    frame = frame.loc[_flag(frame["evaluated"])]
    error = frame[error_column].astype(float).to_numpy()
    tolerance = frame[tolerance_column].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            tolerance > 0, error / tolerance, np.where(error == 0, 0.0, np.inf)
        )
    return pd.DataFrame({"adapter": frame["adapter"].to_numpy(), "ratio": ratio})


def _zero_tolerance_count(report_dir: Path, kind: str) -> int:
    if kind == "phase1":
        frame = pd.read_csv(report_dir / "phase1_detail.csv")
        column = "voxel_tolerance"
    else:
        frame = pd.read_csv(report_dir / "compliance_detail.csv")
        column = "tolerance"
    if column not in frame:
        return 0
    evaluated = frame.loc[_flag(frame["evaluated"])]
    return int((evaluated[column].astype(float) == 0).sum())


def draw_ratio_strip(ax_zero, ax, rows: pd.DataFrame) -> None:
    """Error relative to tolerance per adapter; exact zeros counted apart."""

    from matplotlib.ticker import LogLocator, NullLocator

    adapters = [key for key in style.ADAPTER_ORDER if key in set(rows["adapter"])]
    rng = np.random.default_rng(0)
    finite = rows["ratio"].replace(np.inf, np.nan)
    positive = finite.loc[finite > 0]
    # Every nonzero value stays on the axis; no value is clipped to a floor.
    low = 10 ** np.floor(np.log10(positive.min())) if len(positive) else 1e-6
    for index, adapter in enumerate(adapters):
        y = len(adapters) - 1 - index
        values = rows.loc[rows["adapter"] == adapter, "ratio"].to_numpy(dtype=float)
        nonzero = values[(values > 0) & np.isfinite(values)]
        infinite = int(np.isinf(values).sum())
        if infinite:
            # Positive error on a zero tolerance: a failure with no finite ratio.
            ax.annotate(
                f"{infinite} above, zero tolerance",
                (1.0, y),
                xycoords=("axes fraction", "data"),
                xytext=(-2, 0),
                textcoords="offset points",
                ha="right",
                va="center",
            )
        adapter_style = style.ADAPTER[adapter]
        ax.scatter(
            nonzero,
            y + rng.uniform(-0.25, 0.25, size=nonzero.size),
            marker=adapter_style.marker,
            s=12,
            facecolor=adapter_style.color,
            edgecolor=style.INK,
            linewidth=0.4,
            alpha=0.7,
            zorder=3,
        )
        ax_zero.text(0.5, y, str(int((values == 0).sum())), ha="center", va="center")
    ax.set_xscale("log")
    ax.set_xlim(low, 3.0)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(NullLocator())
    style.reference_line(ax, x=1.0, label="tolerance")
    ax.set_ylim(-0.6, len(adapters) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Error relative to tolerance")
    ax_zero.set_xlim(0, 1)
    ax_zero.set_ylim(ax.get_ylim())
    ax_zero.set_xticks([])
    ax_zero.set_title("Exact 0", loc="center")
    ax_zero.set_yticks(
        range(len(adapters)), [style.ADAPTER[key].label for key in reversed(adapters)]
    )
    ax_zero.tick_params(length=0)
    for spine in ax_zero.spines.values():
        spine.set_visible(False)


def plt_size() -> float:
    import matplotlib.pyplot as plt

    return plt.rcParams["font.size"]


def build_accuracy_figures(
    ibsi1_dir: Path, phase1_dir: Path, phase2_dir: Path, output_dir: Path
) -> list[dict[str, Any]]:
    """Write Figures 2, 3 and S9 and the accuracy tables."""

    records = []
    ibsi1, phase1, phase2 = (
        ibsi1_outcomes(ibsi1_dir),
        phase1_outcomes(phase1_dir),
        phase2_outcomes(phase2_dir),
    )
    sources = style.sha256_files(
        [
            ibsi1_dir / "compliance_summary.csv",
            ibsi1_dir / "compliance_by_family.csv",
            phase1_dir / "phase1_strict_3d_summary.csv",
            phase1_dir / "phase1_strict_3d_by_filter.csv",
            phase2_dir / "compliance_phase2_b_summary.csv",
            phase2_dir / "compliance_phase2_b_by_configuration.csv",
        ]
    )

    with style.figure(style.DOUBLE_COLUMN_MM, 118, font_pt=8.5) as fig:
        axes = fig.subplots(3, 1)
        for ax, rows, title, letter in zip(
            axes,
            (ibsi1, phase1, phase2),
            (
                f"IBSI 1 digital phantom: {ibsi1[0].total} standardised checks",
                f"IBSI 2 Phase 1, 3D filters: {phase1[0].total} standardised response maps",
                f"IBSI 2 Phase 2, configuration B: {phase2[0].total} standardised checks",
            ),
            "abc",
        ):
            draw_outcome_bars(ax, rows, title=title)
            style.panel_label(ax, letter, x=-0.02, y=1.02)
        axes[-1].set_xlabel(
            "Share of the standardised checks of each specification (%)"
        )
        outcome_legend(fig)
        records.append(
            style.save(
                fig,
                output_dir / "figure2_conformance_outcomes",
                caption=_figure2_caption(ibsi1, phase1, phase2),
                alt_text=_figure2_alt(ibsi1),
                sources=sources,
            )
        )

    columns2 = phase2_columns(phase2_dir)
    with style.figure(style.DOUBLE_COLUMN_MM, 205, font_pt=8.5) as fig:
        axes = fig.subplots(3, 1, height_ratios=[1, 1, 1])
        for ax, cells, columns, versions, title, letter in zip(
            axes,
            (
                ibsi1_cells(ibsi1_dir),
                phase1_cells(phase1_dir),
                phase2_cells(phase2_dir),
            ),
            (IBSI1_FAMILIES, PHASE1_FILTERS, columns2),
            (_versions(ibsi1), _versions(phase1), _versions(phase2)),
            (
                "IBSI 1: passed standardised checks by feature family",
                "IBSI 2 Phase 1: passed standardised 3D response maps by filter",
                "IBSI 2 Phase 2: passed standardised checks by configuration B",
            ),
            "abc",
        ):
            draw_matrix(ax, cells, columns, versions, title=title)
            style.panel_label(ax, letter, x=-0.02, y=1.30)
        matrix_legend(fig)
        records.append(
            style.save(
                fig,
                output_dir / "figure3_conformance_by_family",
                caption=_figure3_caption(),
                alt_text=_figure3_alt(ibsi1_cells(ibsi1_dir)),
                sources=sources,
            )
        )

    with style.figure(style.DOUBLE_COLUMN_MM, 84, font_pt=8.5) as fig:
        for panel, (directory, phase, title, letter) in zip(
            fig.subfigures(1, 2),
            (
                (phase1_dir, "phase1", "IBSI 2 Phase 1, 3D response maps", "a"),
                (phase2_dir, "phase2", "IBSI 2 Phase 2, configuration B", "b"),
            ),
        ):
            ax_zero, ax = panel.subplots(1, 2, width_ratios=[0.16, 1])
            draw_ratio_strip(ax_zero, ax, _ratio_rows(directory, phase))
            panel.suptitle(title, x=0.06, ha="left", fontsize=plt_size())
            panel.text(
                0.01,
                0.995,
                letter,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=plt_size() + 2,
            )
        records.append(
            style.save(
                fig,
                output_dir / "figureS9_error_relative_to_tolerance",
                caption=(
                    "Error relative to tolerance for every evaluated check. Phase 1: the "
                    "largest voxel error divided by the voxel tolerance (1 percent of "
                    "the reference range). Phase 2: the absolute error divided by the "
                    "published tolerance. Each point is one check; markers follow the "
                    "adapter key of all figures. Exact zeros cannot sit on a log axis "
                    "and are counted in the left column. The line at 1 is the tolerance; "
                    "a check passes at or below it."
                ),
                alt_text=(
                    "Strip plots of error relative to tolerance on a log scale for five "
                    "packages in IBSI 2 Phase 1 and Phase 2. Every evaluated check lies "
                    "below the tolerance line."
                ),
                sources=sources,
            )
        )
    _write_tables(ibsi1_dir, phase1_dir, phase2_dir, output_dir)
    return records


def _figure2_caption(ibsi1, phase1, phase2) -> str:
    return (
        "Conformance outcomes on fixed denominators. Each bar is one package; its "
        "length is the share of all standardised checks of that specification. Dark "
        "segments passed, mid-grey cross-hatched segments (F) failed, white "
        "diagonal-hatched segments (NS) are checks the package does not support. "
        "The text at the bar end gives passed / total and the small segments. "
        f"a, IBSI 1 digital phantom, {ibsi1[0].total} standardised checks (3D merged "
        f"profile). b, IBSI 2 Phase 1, {phase1[0].total} standardised 3D response "
        f"maps. c, IBSI 2 Phase 2, {phase2[0].total} standardised configuration-B "
        "checks. Packages are listed with their versions."
    )


def _figure2_alt(ibsi1) -> str:
    by = {row.adapter: row for row in ibsi1}
    best = [
        style.ADAPTER[key].label for key, row in by.items() if row.passed == row.total
    ]
    pyrad = by.get("pyradiomics")
    text = (
        "Stacked bar charts of conformance outcomes for five software packages in IBSI 1 "
        f"and IBSI 2. {' and '.join(best)} pass all {ibsi1[0].total} IBSI 1 checks"
    )
    if pyrad is not None:
        text += f"; PyRadiomics passes {pyrad.passed} and does not support {pyrad.not_supported}"
    return text + "."


def _figure3_caption() -> str:
    return (
        "Conformance by feature family. Each cell gives the number of passed "
        "standardised checks; the column header gives the fixed denominator n. Black "
        "cells: all checks passed. Mid-grey cells: 50 to 99 percent passed. White cells: "
        "below 50 percent passed. Hatched cells (NS): not supported. A failure count is "
        "shown as n F. a, IBSI 1 Table 2 feature families. b, IBSI 2 Phase 1 3D filter "
        "families. c, IBSI 2 Phase 2 published configuration-B filters; 5.B (dagger) "
        "averages native 2D Gabor responses over three orthogonal planes, as IBSI "
        "specifies."
    )


def _figure3_alt(cells: Mapping[tuple[str, str], Cell]) -> str:
    failures = [
        f"{style.ADAPTER[adapter].label} fails {cell.failed} {dict(IBSI1_FAMILIES)[family].replace(chr(10), ' ').lower()}"
        for (adapter, family), cell in cells.items()
        if cell.failed
    ]
    return (
        "Three matrices of passed checks by feature family and package. Most cells show "
        "that all checks passed; " + "; ".join(failures) + " checks."
    )


def _write_tables(
    ibsi1_dir: Path, phase1_dir: Path, phase2_dir: Path, output_dir: Path
) -> None:
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(ibsi1_dir / "compliance_detail.csv")
    failures = detail.loc[detail["status"] == "fail"]
    failures = _ordered(failures)[
        [
            "adapter",
            "feature_name",
            "ibsi_code",
            "value",
            "reference_value",
            "tolerance",
            "comparison_error",
        ]
    ]
    failures.assign(
        adapter=failures["adapter"].map(lambda key: style.ADAPTER[key].label)
    ).to_csv(tables / "tableS3_ibsi1_failures.csv", index=False)
    summary = pd.read_csv(ibsi1_dir / "compliance_summary.csv")
    coverage = _ordered(summary)[
        [
            "adapter",
            "software_version",
            "defined_features",
            "supported_features",
            "defined_benchmark_instances",
            "finite_benchmark_instances",
            "defined_standardized_checks",
            "evaluated",
            "passed",
        ]
    ]
    coverage.assign(
        adapter=coverage["adapter"].map(lambda key: style.ADAPTER[key].label)
    ).to_csv(tables / "tableS1_ibsi1_coverage.csv", index=False)
    # Table S4: the secondary designs.  "No reference" stays apart from "not
    # supported": a configuration without a published reference cannot pass.
    rows = []
    for row in _ordered(pd.read_csv(phase1_dir / "phase1_summary.csv")).itertuples():
        rows.append(
            [
                style.ADAPTER[row.adapter].label,
                "Phase 1, all 36 tests",
                row.defined_filter_tests,
                row.native_supported_filter_tests,
                row.standardized_reference_tests,
                row.evaluated,
                row.passed,
            ]
        )
    configuration = pd.read_csv(phase2_dir / "compliance_by_configuration.csv")
    number = (
        configuration["configuration"].astype(str).str.split(".").str[0].astype(int)
    )
    configuration["scope"] = np.where(
        number >= 10,
        "Phase 2, configurations 10 and 11 (no reference)",
        np.where(
            configuration["protocol_variant"] == "A",
            "Phase 2, A pathway",
            "Phase 2, B pathway",
        ),
    )
    for (adapter, scope), part in _ordered(configuration).groupby(
        ["adapter", "scope"], sort=False
    ):
        rows.append(
            [
                style.ADAPTER[adapter].label,
                scope,
                int(part["defined_checks"].sum()),
                int(_flag(part["native_supported"]).sum()),
                int(part["standardized_reference_checks"].sum()),
                int(part["evaluated_standardized_checks"].sum()),
                int(part["passed_standardized_checks"].sum()),
            ]
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "package",
            "scope",
            "defined",
            "supported",
            "standardised",
            "evaluated",
            "passed",
        ],
    )
    order = {
        "Phase 1, all 36 tests": 0,
        "Phase 2, A pathway": 1,
        "Phase 2, B pathway": 2,
    }
    package_rank = {
        style.ADAPTER[key].label: index for index, key in enumerate(style.ADAPTER_ORDER)
    }
    table = table.sort_values(
        by=["package", "scope"],
        key=lambda column: (
            column.map(package_rank)
            if column.name == "package"
            else column.map(lambda v: order.get(v, 3))
        ),
    )
    table.to_csv(tables / "tableS4_secondary_designs.csv", index=False)


def release_figures(report_dir: Path, kind: str) -> list[dict[str, Any]]:
    """Single-specification figures for one release report.

    ``kind`` is ``ibsi1``, ``phase1`` or ``phase2``.  The report must already
    have written its CSV tables to ``report_dir``.
    """

    if kind == "phase1":
        outcomes, cells, columns = (
            phase1_outcomes(report_dir),
            phase1_cells(report_dir),
            PHASE1_FILTERS,
        )
        unit, name = "standardised 3D response maps", "IBSI 2 Phase 1"
    elif kind == "phase2":
        outcomes, cells, columns = (
            phase2_outcomes(report_dir),
            phase2_cells(report_dir),
            phase2_columns(report_dir),
        )
        unit, name = "standardised configuration-B checks", "IBSI 2 Phase 2"
    else:
        outcomes, cells, columns = (
            ibsi1_outcomes(report_dir),
            ibsi1_cells(report_dir),
            IBSI1_FAMILIES,
        )
        unit, name = "standardised checks", "IBSI 1"
    columns = [
        column for column in columns if any(key[1] == column[0] for key in cells)
    ]
    scope = (
        " IBSI 1 defines 172 Table 2 definitions and 174 workbook instances; the 169 "
        "instances with standardised tolerances form the denominator."
        if kind == "ibsi1"
        else ""
    )
    records = []
    with style.figure(style.DOUBLE_COLUMN_MM, 52, font_pt=8.5) as fig:
        ax = fig.subplots()
        draw_outcome_bars(ax, outcomes, title=f"{name}: {outcomes[0].total} {unit}")
        ax.set_xlabel("Share of the standardised checks (%)")
        outcome_legend(fig)
        records.append(
            style.save(
                fig,
                report_dir / "figure_conformance_outcomes",
                caption=(
                    f"{name} outcomes on the fixed denominator of {outcomes[0].total} {unit}. "
                    "Dark segments passed, mid-grey cross-hatched segments (F) failed, white "
                    "diagonal-hatched segments (NS) are not supported. The bar-end text gives "
                    "passed / total and the small segments." + scope
                ),
                alt_text=(
                    f"Stacked bar chart of {name} outcomes for {len(outcomes)} packages: "
                    + "; ".join(
                        f"{style.ADAPTER[row.adapter].label} passes {row.passed} of {row.total}"
                        for row in outcomes
                    )
                    + "."
                ),
            )
        )
    with style.figure(style.DOUBLE_COLUMN_MM, 80, font_pt=8.5) as fig:
        ax = fig.subplots()
        draw_matrix(
            ax,
            cells,
            columns,
            _versions(outcomes),
            title=f"{name}: passed {unit} by family",
        )
        matrix_legend(fig)
        records.append(
            style.save(
                fig,
                report_dir / "figure_conformance_by_family",
                caption=_figure3_caption().split(" a, IBSI 1")[0]
                + (
                    " 5.B (dagger) averages native 2D Gabor responses over three "
                    "orthogonal planes, as IBSI specifies."
                    if kind == "phase2"
                    else ""
                )
                + scope,
                alt_text=f"Matrix of passed {name} checks by family and package."
                + scope,
            )
        )
    ratios = _ratio_rows(report_dir, kind) if kind in {"phase1", "phase2"} else None
    if ratios is not None and len(ratios):
        above = int((ratios["ratio"] > 1).sum())
        zero_tolerance = _zero_tolerance_count(report_dir, kind)
        with style.figure(style.DOUBLE_COLUMN_MM / 2 + 10, 80, font_pt=8.5) as fig:
            ax_zero, ax = fig.subplots(1, 2, width_ratios=[0.18, 1])
            draw_ratio_strip(ax_zero, ax, ratios)
            records.append(
                style.save(
                    fig,
                    report_dir / "figure_error_relative_to_tolerance",
                    caption=(
                        f"{name} error relative to tolerance for every evaluated check, on a "
                        "log scale. "
                        + (
                            "Phase 1 uses the largest voxel error and a voxel tolerance of "
                            "one percent of the reference-map range. "
                            if kind == "phase1"
                            else "Phase 2 uses the absolute error and the published tolerance. "
                        )
                        + "Exact zeros are counted in the left column. A zero-tolerance "
                        "check passes only with zero error: "
                        f"{zero_tolerance} evaluated zero-tolerance row(s) appear as exact "
                        "zeros or, with a positive error, as a count above the line. The "
                        "line at 1 is the tolerance."
                    ),
                    alt_text=(
                        f"Strip plot of {name} error relative to tolerance for each package; "
                        f"{above} of {len(ratios)} evaluated checks lie above the tolerance line."
                    ),
                )
            )
    return [
        {
            "id": record["figure_id"],
            "files": [*record["files"], f"{record['figure_id']}.json"],
            "caption": record["caption"],
            "alt_text": record["alt_text"],
        }
        for record in records
    ]

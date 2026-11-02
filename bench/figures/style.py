"""Shared style for every manuscript figure.

Every figure takes its names, order, colours, markers, fonts and sizes from
this module.  ``docs/figures.md`` describes the choices.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

MM = 1.0 / 25.4
DOUBLE_COLUMN_MM = 183.0
MIN_FONT_PT = 8.0
FONT = "Arial"
PNG_DPI = 600

INK = "#000000"
INK_SECONDARY = "#404040"
REFERENCE_LINE = "#404040"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class AdapterStyle:
    key: str
    label: str
    color: str
    marker: str
    linestyle: Any
    hatch: str


# Contrast on white: 5.19, 3.09, 3.42, 6.09 and 6.37 to 1.  The worst
# colour-blind pair (MIRP, PyRadiomics) also differs in marker and line.
ADAPTERS = (
    AdapterStyle("pictologics", "Pictologics", "#0072B2", "o", "-", ""),
    AdapterStyle("pyradiomics", "PyRadiomics", "#EE6677", "s", (0, (4, 2)), "//"),
    AdapterStyle("mirp", "MIRP", "#009E73", "^", (0, (1, 1.5)), "\\\\"),
    AdapterStyle("medimage", "MEDimage", "#AA3377", "D", (0, (5, 1.5, 1, 1.5)), "xx"),
    AdapterStyle("zrad", "Z-Rad", "#8C510A", "P", (0, (9, 2)), ".."),
)
ADAPTER = {style.key: style for style in ADAPTERS}
ADAPTER_ORDER = tuple(style.key for style in ADAPTERS)

GROUPS = (
    ("group_intensity", "Intensity"),
    ("group_local_intensity", "Local intensity"),
    ("group_morphology", "Morphology"),
    ("group_spatial_morphology", "Spatial morphology"),
    ("group_texture", "Texture"),
    ("group_ivh", "IVH"),
)
GROUP_LABEL = dict(GROUPS)
NOISE_LEVELS = (
    ("low_noise", "Low noise (12 HU)"),
    ("reference_noise", "Reference noise (20 HU)"),
    ("high_noise", "High noise (30 HU)"),
)
NOISE_LABEL = dict(NOISE_LEVELS)
MASK_LABEL = {
    "M1": "M1 whole tumour",
    "M2": "M2 doughnut",
    "M3": "M3 spiculated",
    "M4": "M4 multifocal",
    "A1": "A1 whole anatomy",
}
MODALITY_LABEL = {"ct": "CT", "mri": "MRI", "pet": "PET"}


@dataclass(frozen=True)
class OutcomeStyle:
    key: str
    label: str
    code: str
    fill: str
    hatch: str
    hatch_color: str
    text_color: str


OUTCOMES = (
    OutcomeStyle("passed", "Passed", "", "#252525", "", "#252525", WHITE),
    OutcomeStyle("failed", "Failed", "F", "#858585", "xx", INK, INK),
    OutcomeStyle("not_supported", "Not supported", "NS", WHITE, "//", "#737373", INK),
    OutcomeStyle("not_evaluated", "Not evaluated", "NE", "#D9D9D9", "", "#D9D9D9", INK),
)
OUTCOME = {style.key: style for style in OUTCOMES}
OUTLINE_PT = 0.8


def adapter_label(key: str, version: str | None = None) -> str:
    label = ADAPTER[key].label
    return f"{label} {version}" if version else label


def contrast_ratio(first: str, second: str) -> float:
    """WCAG 2 contrast ratio of two hex colours."""

    def luminance(colour: str) -> float:
        value = colour.lstrip("#")
        channels = [int(value[index : index + 2], 16) / 255 for index in (0, 2, 4)]
        linear = [
            channel / 12.92
            if channel <= 0.03928
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def require_font() -> str:
    """Return the Arial font file; fail instead of falling back silently."""

    try:
        return font_manager.findfont(FONT, fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(f"figures require the {FONT} font") from exc


def rc_params(font_pt: float = 8.0) -> dict[str, Any]:
    if font_pt < MIN_FONT_PT:
        raise ValueError(f"figure text must be at least {MIN_FONT_PT} pt")
    return {
        "font.family": FONT,
        "font.size": font_pt,
        "axes.titlesize": font_pt,
        "axes.labelsize": font_pt,
        "xtick.labelsize": font_pt,
        "ytick.labelsize": font_pt,
        "legend.fontsize": font_pt,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "hatch.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        # Editable text in SVG and embedded TrueType in PDF.
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "svg.hashsalt": "pictologics-benchmark",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 2 * MM,
        "figure.constrained_layout.w_pad": 2 * MM,
    }


@contextmanager
def figure(
    width_mm: float, height_mm: float, *, font_pt: float = 8.0, **kwargs: Any
) -> Iterator[plt.Figure]:
    """Create one figure at its final print size."""

    require_font()
    with plt.rc_context(rc_params(font_pt)):
        fig = plt.figure(figsize=(width_mm * MM, height_mm * MM), **kwargs)
        try:
            yield fig
        finally:
            plt.close(fig)


def panel_label(ax: plt.Axes, letter: str, *, x: float = -0.02, y: float = 1.0) -> None:
    """Draw a bold lowercase panel letter just outside the top-left corner."""

    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
        fontsize=max(MIN_FONT_PT + 2, plt.rcParams["font.size"] + 2),
        color=INK,
    )


def reference_line(
    ax: plt.Axes,
    *,
    x: float | None = None,
    y: float | None = None,
    label: str | None = None,
) -> None:
    """Draw a labelled reference line that never looks like a series."""

    if x is not None:
        ax.axvline(x, color=REFERENCE_LINE, linewidth=0.8, zorder=1)
    if y is not None:
        ax.axhline(y, color=REFERENCE_LINE, linewidth=0.8, zorder=1)
    if label:
        if x is not None:
            ax.annotate(
                label,
                (x, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                ha="left",
                va="top",
                color=INK_SECONDARY,
            )
        else:
            # Below the line at the right, clear of labels above the axes.
            ax.annotate(
                label,
                (1.0, y),
                xycoords=("axes fraction", "data"),
                xytext=(-2, -2),
                textcoords="offset points",
                ha="right",
                va="top",
                color=INK_SECONDARY,
            )


def save(
    fig: plt.Figure,
    stem: Path,
    *,
    caption: str,
    alt_text: str,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write PDF, SVG and 600 dpi PNG plus a sidecar with caption and alt text."""

    from bench.figures.checks import check_figure

    problems = check_figure(fig)
    if problems:
        raise ValueError(f"{stem.name} fails the figure checks: " + "; ".join(problems))
    stem.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix, options in (
        (
            "pdf",
            {"metadata": {"CreationDate": None, "Creator": None, "Producer": None}},
        ),
        ("svg", {"metadata": {"Date": None, "Creator": None}}),
        ("png", {"dpi": PNG_DPI, "metadata": {"Software": None}}),
    ):
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, **options)
        files.append(path.name)
    width, height = (value / MM for value in fig.get_size_inches())
    record = {
        "figure_id": stem.name,
        "files": files,
        "size_mm": [round(width, 1), round(height, 1)],
        "caption": caption,
        "alt_text": alt_text,
        "sources": dict(sources or {}),
    }
    (stem.parent / f"{stem.name}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def sha256_files(paths: Sequence[Path]) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

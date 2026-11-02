"""Automatic layout checks and colour-vision renders for figures."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from matplotlib.text import Text

from bench.figures.style import MIN_FONT_PT

# Machado, Oliveira and Fernandes (2009), severity 1.0, applied in linear RGB.
CVD_MATRICES = {
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "tritanopia": np.array(
        [
            [1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900],
        ]
    ),
}


def _undrawn_tick_labels(fig) -> set[int]:
    """Tick labels outside the axis range keep stale text but are not drawn."""

    hidden: set[int] = set()
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            slack = 1e-9 * max(abs(low), abs(high), 1e-300)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                location = tick.get_loc()
                if location is None or not low - slack <= location <= high + slack:
                    hidden.update({id(tick.label1), id(tick.label2)})
    return hidden


def _visible_texts(fig) -> list[Text]:
    hidden = _undrawn_tick_labels(fig)
    return [
        artist
        for artist in fig.findobj(Text)
        if artist.get_visible()
        and artist.get_text().strip()
        and id(artist) not in hidden
    ]


def check_figure(fig, *, min_font_pt: float = MIN_FONT_PT) -> list[str]:
    """Return text that is too small, leaves the figure or overlaps other text."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    problems: list[str] = []
    boxes = []
    for text in _visible_texts(fig):
        label = text.get_text().strip().replace("\n", " ")[:40]
        if text.get_fontsize() < min_font_pt - 1e-6:
            problems.append(f"text below {min_font_pt} pt: {label!r}")
        box = text.get_window_extent(renderer)
        if (
            box.x0 < bounds.x0 - 1
            or box.y0 < bounds.y0 - 1
            or box.x1 > bounds.x1 + 1
            or box.y1 > bounds.y1 + 1
        ):
            problems.append(f"text leaves the figure: {label!r}")
        boxes.append((label, box))
    for (first, a), (second, b) in combinations(boxes, 2):
        overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
        overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
        # One pixel of touching is allowed; real overlaps are larger.
        if overlap_x > 1 and overlap_y > 1:
            problems.append(f"text overlaps: {first!r} and {second!r}")
    return problems


def _to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)


def accessibility_views(png: Path, output_dir: Path) -> list[Path]:
    """Write greyscale and simulated colour-blind versions of a PNG for review."""

    import matplotlib.image as mpimg

    image = mpimg.imread(png)[..., :3].astype(float)
    linear = _to_linear(image)
    views = {
        "greyscale": np.repeat(
            (linear @ np.array([0.2126, 0.7152, 0.0722]))[..., None], 3, axis=2
        )
    }
    for name, matrix in CVD_MATRICES.items():
        views[name] = linear @ matrix.T
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, view in views.items():
        path = output_dir / f"{png.stem}_{name}.png"
        mpimg.imsave(path, _to_srgb(view))
        written.append(path)
    return written

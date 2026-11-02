from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench.figures import style
from bench.figures.checks import accessibility_views, check_figure


class FigureStyleTests(unittest.TestCase):
    def test_every_adapter_has_its_own_marker_line_and_hatch(self) -> None:
        for field in ("marker", "linestyle", "hatch", "color"):
            values = [str(getattr(adapter, field)) for adapter in style.ADAPTERS]
            self.assertEqual(len(values), len(set(values)), field)

    def test_adapter_colours_reach_three_to_one_on_white(self) -> None:
        for adapter in style.ADAPTERS:
            self.assertGreaterEqual(
                style.contrast_ratio(adapter.color, style.WHITE), 3.0, adapter.key
            )

    def test_outcome_text_and_neighbouring_fills_pass_contrast(self) -> None:
        for outcome in style.OUTCOMES:
            self.assertGreaterEqual(
                style.contrast_ratio(outcome.text_color, outcome.fill), 4.5, outcome.key
            )
        passed, failed = style.OUTCOME["passed"].fill, style.OUTCOME["failed"].fill
        self.assertGreaterEqual(style.contrast_ratio(passed, failed), 3.0)
        self.assertGreaterEqual(style.contrast_ratio(failed, style.WHITE), 3.0)

    def test_text_below_the_minimum_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least"):
            style.rc_params(7.0)

    def test_checks_find_small_and_overlapping_text(self) -> None:
        with style.figure(89, 40) as fig:
            ax = fig.subplots()
            ax.text(0.5, 0.5, "first label")
            ax.text(0.5, 0.5, "second label")
            ax.text(0.1, 0.1, "tiny", fontsize=6)
            problems = check_figure(fig)
        self.assertTrue(any("overlaps" in problem for problem in problems))
        self.assertTrue(any("below" in problem for problem in problems))

    def test_save_writes_three_formats_and_a_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stem = Path(temporary) / "figure_test"
            with style.figure(89, 40) as fig:
                ax = fig.subplots()
                ax.plot([1, 2], [1, 2], color=style.ADAPTER["pictologics"].color)
                record = style.save(fig, stem, caption="Caption.", alt_text="Alt text.")
            self.assertEqual(record["size_mm"], [89.0, 40.0])
            for suffix in ("pdf", "svg", "png", "json"):
                self.assertTrue(stem.with_suffix("." + suffix).is_file())
            views = accessibility_views(
                stem.with_suffix(".png"), Path(temporary) / "views"
            )
            self.assertEqual(len(views), 4)


if __name__ == "__main__":
    unittest.main()

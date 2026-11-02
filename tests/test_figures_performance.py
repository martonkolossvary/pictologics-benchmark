from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench.figures import performance


def _record(
    adapter: str,
    size: int,
    repeat: int,
    *,
    status: str = "measured",
    duration: float = 1.0,
) -> dict:
    return {
        "task_status": status,
        "adapter": adapter,
        "workload": "family:group_texture",
        "dataset": "pictologics_pillar2_a1",
        "case_id": f"p2_a1_n{size:03d}",
        "subject_id": "reference",
        "mask_id": "A1",
        "modality": "synthetic",
        "size": size,
        "repeat": repeat,
        "mask_voxels": size**3 // 2,
        "image_voxels": size**3,
        "duration_sec": duration if status == "measured" else None,
        "cpu_time_sec": duration if status == "measured" else None,
        "incremental_calculation_peak_rss_bytes": 2e6 if status == "measured" else None,
        "host_peak_memory_in_use_bytes": 3e8 + size**3
        if status == "measured"
        else None,
        "peak_rss_bytes": 2.5e8 + size**3 if status == "measured" else None,
        "worker_ready_rss_bytes": 2e8,
        "load_sec": 1.0,
        "setup_sec": 0.01,
        "censor_lower_bound_sec": 130.0 if status == "timed_out_censored" else None,
        "memory_budget_bytes": 4e10,
        "per_call_timeout_sec": 120.0,
    }


def _ledger(directory: Path, records: list[dict]) -> Path:
    directory.mkdir(parents=True)
    connection = sqlite3.connect(directory / "benchmark.sqlite3")
    connection.execute(
        "CREATE TABLE tasks (ordinal INTEGER, started_at REAL, record_json TEXT)"
    )
    connection.executemany(
        "INSERT INTO tasks VALUES (?, ?, ?)",
        [
            (index, float(index), json.dumps(record))
            for index, record in enumerate(records)
        ],
    )
    connection.commit()
    connection.close()
    return directory


class PerformanceFigureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        records = []
        for size in (32, 64, 128):
            for repeat in (1, 2, 3):
                records.append(
                    _record("pictologics", size, repeat, duration=0.01 * size)
                )
                slow = "timed_out_censored" if size == 128 else "measured"
                records.append(
                    _record("mirp", size, repeat, status=slow, duration=0.05 * size)
                )
        self.run_dir = _ledger(self.root / "run", records)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_a_timeout_never_gets_a_value(self) -> None:
        summary = performance.cells(performance.load_records([self.run_dir]))
        censored = summary.loc[
            (summary["adapter"] == "mirp") & (summary["size"] == 128)
        ].iloc[0]
        self.assertTrue(np.isnan(censored["median"]))
        self.assertEqual(censored["bound"], 130.0)
        self.assertEqual(censored["code"], "T")

    def test_censored_pairs_make_the_ratio_a_lower_bound(self) -> None:
        summary = performance.cells(performance.load_records([self.run_dir]))
        table = performance.paired_ratios(summary, synthetic_size=128)
        row = table.loc[table["adapter"] == "mirp"].iloc[0]
        self.assertTrue(row["lower_bound"])
        self.assertAlmostEqual(row["ratio"], 130.0 / (0.01 * 128))

    def test_a_pictologics_timeout_makes_the_ratio_an_upper_bound(self) -> None:
        records = []
        for repeat in (1, 2, 3):
            records.append(
                _record("pictologics", 128, repeat, status="timed_out_censored")
            )
            records.append(_record("mirp", 128, repeat, duration=50.0))
        frame = performance.prepare_records(performance.pd.DataFrame(records))
        table = performance.paired_ratios(performance.cells(frame))
        row = table.loc[table["adapter"] == "mirp"].iloc[0]
        self.assertTrue(row["upper_bound"])
        self.assertFalse(row["lower_bound"])
        self.assertEqual(row["not_estimable"], 0)
        self.assertAlmostEqual(row["ratio"], 50.0 / 130.0)

    def test_pillar1_noise_levels_are_never_pooled(self) -> None:
        records = []
        for noise, factor in (("low_noise", 1.0), ("high_noise", 2.0)):
            for mask in ("M1", "M2", "M3", "M4"):
                for adapter, scale in (("pictologics", 1.0), ("mirp", factor)):
                    for repeat in (1, 2, 3):
                        record = _record(adapter, 128, repeat, duration=scale)
                        record.update(
                            dataset="pictologics_pillar1",
                            subject_id=noise,
                            mask_id=mask,
                            case_id=f"p1_{noise}_{mask}_n128",
                        )
                        records.append(record)
        frame = performance.prepare_records(performance.pd.DataFrame(records))
        table = performance.paired_ratios(performance.cells(frame)).set_index("dataset")
        self.assertEqual(
            set(table.index),
            {"Pillar 1, 128³, low noise", "Pillar 1, 128³, high noise"},
        )
        self.assertAlmostEqual(table.loc["Pillar 1, 128³, high noise", "ratio"], 2.0)
        # The masks share one scene, so the interval is their range.
        self.assertEqual(table.loc["Pillar 1, 128³, low noise", "interval"], "range")

    def test_slopes_use_common_sizes_and_count_timeouts(self) -> None:
        records = []
        for size in (64, 96, 128, 192, 256):
            for repeat in (1, 2, 3):
                records.append(
                    _record("pictologics", size, repeat, duration=1e-6 * size**3)
                )
                slow = "timed_out_censored" if size == 256 else "measured"
                record = _record(
                    "mirp", size, repeat, status=slow, duration=1e-5 * size**3
                )
                if slow != "measured":
                    record["censor_lower_bound_sec"] = 200.0
                records.append(record)
        frame = performance.prepare_records(performance.pd.DataFrame(records))
        table = performance.slope_table(performance.cells(frame)).set_index("package")
        self.assertEqual(table.loc["Pictologics", "sizes"], "64, 96, 128, 192")
        self.assertEqual(table.loc["Pictologics", "n"], 4)
        self.assertAlmostEqual(table.loc["MIRP", "slope"], 1.0, places=2)
        self.assertEqual(table.loc["MIRP", "n_timeouts"], 1)
        self.assertTrue(np.isfinite(table.loc["MIRP", "slope_censored"]))

    def test_every_figure_fits_all_timed_groups(self) -> None:
        from bench.figures import style

        records = []
        for group, _ in style.GROUPS:
            for size in (32, 64, 128):
                for repeat in (1, 2, 3):
                    for adapter, factor in (("pictologics", 0.01), ("mirp", 0.05)):
                        record = _record(adapter, size, repeat, duration=factor * size)
                        record["workload"] = f"family:{group}"
                        records.append(record)
        run_dir = _ledger(self.root / "all_groups", records)
        figures = performance.build_performance_figures(
            [run_dir], self.root / "all_groups_figures"
        )
        self.assertIn("figure4_runtime_scaling", {f["figure_id"] for f in figures})

    def test_every_figure_passes_the_layout_checks(self) -> None:
        records = performance.build_performance_figures(
            [self.run_dir], self.root / "figures"
        )
        names = {record["figure_id"] for record in records}
        self.assertIn("figure4_runtime_scaling", names)
        self.assertIn("figure5_runtime_ratio", names)
        self.assertIn("figureS7_outcome_map", names)
        for table in (
            "tableS5_loglog_slopes.csv",
            "data_task_records.csv",
            "data_cells.csv",
            "data_ratio_cases.csv",
            "data_ratio_summary.csv",
        ):
            self.assertTrue((self.root / "figures" / "tables" / table).is_file())


if __name__ == "__main__":
    unittest.main()


class CommunityFigureTests(unittest.TestCase):
    def test_machines_sit_side_by_side_from_their_packages(self) -> None:
        import gzip

        from bench.figures import community

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for machine, speed in (("macos-a-1111", 1.0), ("linux-b-2222", 2.0)):
                folder = root / "submissions" / machine / "2026-11-01"
                records = [
                    _record(adapter, size, repeat, duration=factor * speed * size)
                    for size in (32, 64)
                    for repeat in (1, 2, 3)
                    for adapter, factor in (("pictologics", 0.01), ("mirp", 0.05))
                ]
                for pillar in community.PILLARS:
                    (folder / pillar).mkdir(parents=True)
                    lines = [json.dumps({"record": record}) for record in records]
                    (folder / pillar / "tasks.jsonl.gz").write_bytes(
                        gzip.compress(("\n".join(lines) + "\n").encode())
                    )
                (folder / "submission.json").write_text(
                    json.dumps({"platform_key": "Linux-x86_64"}), encoding="utf-8"
                )
            table = community.machine_ratios(root / "submissions")
            self.assertEqual(set(table["machine"]), {"macos-a-1111", "linux-b-2222"})
            # The same relative speed on both machines, whatever the absolute time.
            ratios = table.loc[table["dataset"].str.startswith("Pillar 2"), "ratio"]
            np.testing.assert_allclose(ratios.to_numpy(), 5.0)
            record = community.community_figure(
                root / "submissions", root / "figures", dataset="Pillar 2, 64³"
            )
            self.assertTrue((root / "figures" / f"{record['figure_id']}.pdf").is_file())

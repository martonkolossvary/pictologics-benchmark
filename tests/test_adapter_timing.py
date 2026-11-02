from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

from bench.adapters.base import run_timed_computation
from bench.adapters.protocol import (
    add_common_arguments,
    make_payload,
    run_adapter_timing,
    timing_contract_metadata,
)


class AdapterTimingTests(unittest.TestCase):
    def test_primary_duration_is_median_not_optimistic_minimum(self) -> None:
        # Three measured calls: 1 s, 9 s, 5 s. Each call consumes a start/end.
        wall = iter(
            [
                0,
                1_000_000_000,
                2_000_000_000,
                11_000_000_000,
                20_000_000_000,
                25_000_000_000,
            ]
        )
        cpu = iter(
            [0, 100_000_000, 200_000_000, 1_100_000_000, 2_000_000_000, 2_500_000_000]
        )
        with (
            patch("time.perf_counter_ns", side_effect=lambda: next(wall)),
            patch("time.process_time_ns", side_effect=lambda: next(cpu)),
        ):
            result, stats = run_timed_computation(
                lambda: "ok", iterations=3, no_warmup=True
            )
        self.assertEqual(result, "ok")
        self.assertEqual(stats["duration_min_sec"], 1.0)
        self.assertEqual(stats["duration_sec"], 5.0)
        self.assertEqual(stats["duration_median_sec"], 5.0)
        self.assertEqual(stats["duration_samples_sec"], [1.0, 9.0, 5.0])

    def test_one_warmup_is_discarded(self) -> None:
        calls = []

        def compute():
            calls.append(len(calls))
            return calls[-1]

        _, stats = run_timed_computation(compute, iterations=2, no_warmup=False)
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(stats["warmup_iterations"], 1)
        self.assertEqual(stats["measured_iterations"], 2)
        self.assertEqual(stats["total_iterations"], 3)
        self.assertIsNotNone(stats["warmup_duration_sec"])
        self.assertEqual(len(stats["preparation_samples_sec"]), 2)
        self.assertEqual(len(stats["finalization_samples_sec"]), 2)

    def test_warmup_call_ends_before_finalization(self) -> None:
        events = []

        def finalize(raw, _state):
            events.append("finalize")
            return raw

        run_timed_computation(
            lambda: "ok",
            iterations=1,
            finalize_fn=finalize,
            event_fn=lambda name, **_details: events.append(name),
        )
        self.assertEqual(events[:3], ["warmup_start", "warmup_complete", "finalize"])

    def test_preparation_and_finalization_are_outside_calculation_clock(self) -> None:
        calls = []
        wall = iter(
            [
                0,
                4_000_000_000,
                10_000_000_000,
                12_000_000_000,
                20_000_000_000,
                28_000_000_000,
            ]
        )
        cpu = iter([0, 500_000_000])

        def prepare():
            calls.append("prepare")
            return "state"

        def compute(state):
            calls.append(("compute", state))
            return "raw"

        def finalize(raw, state):
            calls.append(("finalize", raw, state))
            return "result"

        with (
            patch("time.perf_counter_ns", side_effect=lambda: next(wall)),
            patch("time.process_time_ns", side_effect=lambda: next(cpu)),
        ):
            result, stats = run_timed_computation(
                compute,
                iterations=1,
                no_warmup=True,
                prepare_fn=prepare,
                finalize_fn=finalize,
            )
        self.assertEqual(result, "result")
        self.assertEqual(stats["preparation_samples_sec"], [4.0])
        self.assertEqual(stats["duration_samples_sec"], [2.0])
        self.assertEqual(stats["finalization_samples_sec"], [8.0])
        self.assertEqual(
            calls,
            ["prepare", ("compute", "state"), ("finalize", "raw", "state")],
        )

    def test_invalid_iterations_are_rejected(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_timed_computation(lambda: None, iterations=value)

    def test_common_cli_exposes_warmup_and_optional_input_bindings(self) -> None:
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        image_digest = "A" * 64
        mask_digest = "b" * 64
        args = parser.parse_args(
            [
                "--image",
                "image.nii.gz",
                "--mask",
                "mask.nii.gz",
                "--image-sha256",
                image_digest,
                "--mask-sha256",
                mask_digest,
                "--no-warmup",
            ]
        )
        self.assertTrue(args.no_warmup)
        self.assertEqual(args.image_sha256, image_digest.lower())
        self.assertEqual(args.mask_sha256, mask_digest)

    def test_shared_timing_wrapper_forwards_no_warmup(self) -> None:
        def compute():
            return "ok"

        with patch(
            "bench.adapters.base.run_timed_computation",
            return_value=("ok", {"measured_iterations": 3}),
        ) as timed:
            result = run_adapter_timing(
                compute,
                iterations=3,
                no_warmup=True,
            )
        self.assertEqual(result[0], "ok")
        timed.assert_called_once_with(
            compute,
            iterations=3,
            no_warmup=True,
            event_fn=unittest.mock.ANY,
        )

    def test_timed_payload_declares_scope_and_input_provenance(self) -> None:
        image_digest = "a" * 64
        mask_digest = "b" * 64
        payload = make_payload(
            adapter="pictologics",
            feature_names=["feature"],
            timing={"measured_iterations": 2},
            image_sha256=image_digest,
            mask_sha256=mask_digest,
            modality="CT",
        )
        self.assertEqual(
            payload["metadata"]["timing_contract"],
            timing_contract_metadata(),
        )
        self.assertEqual(
            payload["metadata"]["input"],
            {
                "image_sha256": image_digest,
                "source_image_sha256": image_digest,
                "mask_sha256": mask_digest,
                "modality": "CT",
                "input_contract": "adapter_internal_preprocessing",
                "representation_id": "single_image_internal_preprocessing",
                "representation_derivation_sha256": None,
                "configured_levels": None,
                "occupied_levels": None,
            },
        )


if __name__ == "__main__":
    unittest.main()

    def test_pictologics_feature_caches_are_cleared(self) -> None:
        import sys
        import types
        from functools import lru_cache

        from bench.adapters import pictologics_adapter

        @lru_cache(maxsize=4)
        def offsets(radius: float) -> float:
            return radius

        @lru_cache(maxsize=None)
        def directions(planar: bool) -> bool:
            return planar

        intensity = types.ModuleType("pictologics.features.intensity")
        intensity._sphere_offsets_for_radius = offsets
        texture = types.ModuleType("pictologics.features.texture")
        texture._directions = directions
        morphology = types.ModuleType("pictologics.features.morphology")
        features = types.ModuleType("pictologics.features")
        features.intensity = intensity
        features.texture = texture
        features.morphology = morphology
        package = types.ModuleType("pictologics")
        package.features = features
        modules = {
            "pictologics": package,
            "pictologics.features": features,
            "pictologics.features.intensity": intensity,
            "pictologics.features.texture": texture,
            "pictologics.features.morphology": morphology,
        }
        with patch.dict(sys.modules, modules):
            offsets(1.0)
            directions(True)
            cleared = pictologics_adapter._clear_feature_caches()
        self.assertEqual(offsets.cache_info().currsize, 0)
        self.assertEqual(directions.cache_info().currsize, 0)
        self.assertEqual(
            cleared,
            [
                "pictologics.features.intensity._sphere_offsets_for_radius",
                "pictologics.features.texture._directions",
            ],
        )

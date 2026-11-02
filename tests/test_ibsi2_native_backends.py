from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from bench.compliance import ibsi2_native_backends as native


class TestNativeParameterContract(unittest.TestCase):
    def test_mean_is_canonicalized_and_defaults_to_ibsi_mirror(self) -> None:
        self.assertEqual(
            native.normalize_parameters(
                {"filter": "MEAN", "dimensionality": 3, "support": 5}
            ),
            {
                "filter": "mean",
                "dimensionality": 3,
                "boundary": "mirror",
                "support": 5,
            },
        )

    def test_laws_energy_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "compute_energy"):
            native.normalize_parameters(
                {
                    "filter": "laws",
                    "dimensionality": 3,
                    "boundary": "zero",
                    "kernels": "L5E5E5",
                    "energy_distance": 7,
                }
            )

        params = native.normalize_parameters(
            {
                "filter": "laws",
                "dimensionality": 3,
                "boundary": "mirror",
                "kernels": "L5E5E5",
                "rotation_invariant": True,
                "pooling": "max",
                "compute_energy": True,
                "energy_distance": 7,
            }
        )
        self.assertTrue(params["compute_energy"])
        self.assertEqual(params["energy_distance"], 7)

    def test_rotation_step_must_divide_full_circle(self) -> None:
        with self.assertRaisesRegex(ValueError, r"divide 2\*pi"):
            native.normalize_parameters(
                {
                    "filter": "gabor",
                    "dimensionality": 2,
                    "sigma_mm": 2.0,
                    "lambda_mm": 3.0,
                    "rotation_invariant": True,
                    "delta_theta": 0.7,
                }
            )

    def test_unknown_or_inapplicable_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not applicable"):
            native.normalize_parameters(
                {
                    "filter": "mean",
                    "dimensionality": 3,
                    "support": 3,
                    "sigma_mm": 1.0,
                }
            )

    def test_simoncelli_defaults_to_periodic(self) -> None:
        params = native.normalize_parameters(
            {"filter": "simoncelli", "dimensionality": 3, "level": 2}
        )
        self.assertEqual(params["boundary"], "periodic")

    def test_phase2_implementation_selected_boundary_is_adapter_aware(self) -> None:
        params = native.normalize_parameters(
            {"filter": "simoncelli", "dimensionality": 3, "level": 1},
            adapter="mirp",
        )
        self.assertEqual(params["boundary"], "periodic")

    def test_identity_filter_has_no_invented_boundary(self) -> None:
        params = native.normalize_parameters(
            {"filter": "none", "dimensionality": 3},
            adapter="pictologics",
        )
        self.assertEqual(params, {"filter": "none", "dimensionality": 3})


class TestNativeDispatchAndCli(unittest.TestCase):
    def test_apply_dispatches_normalized_mapping_and_writes_output(self) -> None:
        captured = {}

        def fake_backend(source, destination, params):
            captured.update(params)
            destination.write_bytes(b"response")
            return {"geometry_preserved": True}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.nii.gz"
            output = root / "output.nii.gz"
            source.write_bytes(b"input")
            with mock.patch.dict(native._BACKENDS, {"pictologics": fake_backend}):
                result = native.apply_native_filter(
                    "pictologics",
                    source,
                    output,
                    {"filter": "mean", "dimensionality": 3, "support": 3},
                )

        self.assertEqual(captured["boundary"], "mirror")
        self.assertEqual(result["adapter"], "pictologics")
        self.assertTrue(result["geometry_preserved"])
        self.assertEqual(
            result["boundary_execution"],
            {
                "policy": "implementation_selected",
                "selected": "mirror",
                "effective": "mirror",
                "implementation": "as_specified",
            },
        )

    def test_apply_identity_records_boundary_as_not_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.nii.gz"
            output = root / "output.nii.gz"
            source.write_bytes(b"identity")
            result = native.apply_native_filter(
                "pictologics",
                source,
                output,
                {"filter": "none", "dimensionality": 3},
            )
            self.assertEqual(output.read_bytes(), b"identity")

        self.assertEqual(
            result["parameters"],
            {"filter": "none", "dimensionality": 3},
        )
        self.assertEqual(
            result["boundary_execution"],
            {
                "policy": "not_applicable",
                "selected": None,
                "effective": None,
                "implementation": "not_applicable",
            },
        )

    def test_pictologics_051_forwards_nonperiodic_riesz_boundary(self) -> None:
        captured = {}

        def riesz_simoncelli(image, **kwargs):
            captured.update(kwargs)
            return np.zeros_like(image)

        capability = types.SimpleNamespace(
            effective_boundary="as_specified_via_padding"
        )
        fake_filters = types.SimpleNamespace(
            riesz_simoncelli=riesz_simoncelli,
            get_filter_capabilities=lambda name: capability,
            CAPABILITIES_SCHEMA_VERSION="1.1.0",
        )
        fake_package = types.ModuleType("pictologics")
        fake_package.filters = fake_filters
        params = native.normalize_parameters(
            {
                "filter": "riesz_simoncelli",
                "dimensionality": 3,
                "boundary": "nearest",
                "level": 1,
                "order": [0, 2, 0],
            }
        )
        with (
            mock.patch.dict(sys.modules, {"pictologics": fake_package}),
            mock.patch.object(
                native,
                "_load_nibabel",
                return_value=(mock.Mock(), np.zeros((3, 3, 3)), (1.0, 1.0, 1.0)),
            ),
            mock.patch.object(native, "_save_nibabel"),
            mock.patch.object(
                native,
                "asdict",
                return_value={
                    "effective_boundary": "as_specified_via_padding",
                },
            ),
        ):
            metadata = native._pictologics_backend(
                Path("input.nii.gz"), Path("output.nii.gz"), params
            )

        self.assertEqual(captured["boundary"], "NEAREST")
        self.assertEqual(
            metadata["boundary_implementation"], "as_specified_via_padding"
        )

    def test_cli_parameter_file_accepts_reviewed_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "filter.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "phase": "phase1",
                        "parameters": {
                            "filter": "mean",
                            "dimensionality": 3,
                            "boundary": "zero",
                            "support": 5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(parameters_file=str(config), parameters_json=None)
            params = native._load_cli_parameters(args)
        self.assertEqual(params["filter"], "mean")
        self.assertEqual(params["support"], 5)

    def test_cli_unsupported_result_has_machine_readable_evidence(self) -> None:
        error_stream = io.StringIO()
        failure = native.UnsupportedNativeFilter(
            "pyradiomics", "log", "truncate is not selectable"
        )
        with mock.patch.object(native, "apply_native_filter", side_effect=failure):
            with contextlib.redirect_stderr(error_stream):
                status = native.main(
                    [
                        "--adapter",
                        "pyradiomics",
                        "--input",
                        "input.nii.gz",
                        "--output",
                        "output.nii.gz",
                        "--parameters-json",
                        '{"filter":"log","dimensionality":3,"sigma_mm":1.0}',
                    ]
                )
        payload = json.loads(error_stream.getvalue())
        self.assertEqual(status, 2)
        self.assertEqual(payload["status"], "unsupported_native_filter")
        self.assertTrue(payload["evidence"])

    def test_mirp_rejects_nonperiodic_simoncelli_before_execution(self) -> None:
        params = native.normalize_parameters(
            {
                "filter": "simoncelli",
                "dimensionality": 3,
                "boundary": "nearest",
                "level": 1,
            }
        )
        with self.assertRaises(native.UnsupportedNativeFilter):
            native._mirp_kwargs(params)

    def test_mirp_rejects_slice_wise_riesz_before_native_index_error(self) -> None:
        params = native.normalize_parameters(
            {
                "filter": "riesz_simoncelli",
                "dimensionality": 2,
                "level": 1,
                "order": [0, 2],
            },
            adapter="mirp",
        )
        with self.assertRaisesRegex(
            native.UnsupportedNativeFilter,
            "cannot execute its public Riesz filters slice-wise",
        ):
            native._mirp_kwargs(params)

    def test_mirp_maps_ibsi_riesz_axis_order_without_enabling_steering(self) -> None:
        params = native.normalize_parameters(
            {
                "filter": "riesz_log",
                "dimensionality": 3,
                "boundary": "zero",
                "sigma_mm": 3.0,
                "truncate": 4.0,
                "order": [1, 0, 0],
            }
        )
        kwargs = native._mirp_kwargs(params)
        self.assertEqual(kwargs["filter_kernels"], "riesz_log")
        self.assertEqual(kwargs["riesz_filter_order"], [0, 0, 1])
        self.assertEqual(kwargs["riesz_filter_tensor_sigma"], 1.0)
        self.assertNotIn("ibsi_compliant", kwargs)

    def test_mirp_non_cubic_riesz_grid_failure_is_reported_as_unsupported(self) -> None:
        params = native.normalize_parameters(
            {
                "filter": "riesz_log",
                "dimensionality": 3,
                "boundary": "zero",
                "sigma_mm": 3.0,
                "truncate": 4.0,
                "order": [1, 0, 0],
            },
            adapter="mirp",
        )
        mirp_module = types.ModuleType("mirp")
        mirp_module.extract_images = mock.Mock(
            side_effect=ValueError(
                "operands could not be broadcast together with shapes "
                "(180,197,200) (197,180,200)"
            )
        )
        with mock.patch.dict(sys.modules, {"mirp": mirp_module}):
            with self.assertRaisesRegex(
                native.UnsupportedNativeFilter,
                "frequency-grid axes do not match",
            ):
                native._mirp_backend(
                    Path("input.nii.gz"), Path("output.nii.gz"), params
                )

    def test_mirp_unrelated_riesz_value_error_is_not_hidden(self) -> None:
        params = native.normalize_parameters(
            {
                "filter": "riesz_log",
                "dimensionality": 3,
                "boundary": "zero",
                "sigma_mm": 3.0,
                "truncate": 4.0,
                "order": [1, 0, 0],
            },
            adapter="mirp",
        )
        mirp_module = types.ModuleType("mirp")
        mirp_module.extract_images = mock.Mock(
            side_effect=ValueError("unrelated failure")
        )
        with mock.patch.dict(sys.modules, {"mirp": mirp_module}):
            with self.assertRaisesRegex(ValueError, "unrelated failure"):
                native._mirp_backend(
                    Path("input.nii.gz"), Path("output.nii.gz"), params
                )

    def test_pictologics_does_not_invent_slice_wise_mean_support(self) -> None:
        params = native.normalize_parameters(
            {"filter": "mean", "dimensionality": 2, "support": 15}
        )
        with self.assertRaisesRegex(
            native.UnsupportedNativeFilter, "documented mean-filter API"
        ):
            native._validate_pictologics_static(params)

    def test_pictologics_rejects_unsupported_sum_pooling_cleanly(self) -> None:
        params = native.normalize_parameters(
            {
                "filter": "laws",
                "dimensionality": 3,
                "kernels": "L5E5E5",
                "rotation_invariant": True,
                "pooling": "sum",
            }
        )
        with self.assertRaisesRegex(native.UnsupportedNativeFilter, "not sum"):
            native._validate_pictologics_static(params)

    def test_zrad_zero_padded_laws_energy_is_rejected(self) -> None:
        params = native.normalize_parameters(
            {
                "filter": "laws",
                "dimensionality": 3,
                "boundary": "zero",
                "kernels": "L5E5E5",
                "compute_energy": True,
                "energy_distance": 7,
            }
        )
        with self.assertRaisesRegex(
            native.UnsupportedNativeFilter, "hard-codes reflect"
        ):
            native._validate_zrad_static(params)

    def test_native_default_runs_only_for_boundary_or_truncation_limits(self) -> None:
        laws = native.normalize_parameters(
            {
                "filter": "laws",
                "dimensionality": 3,
                "boundary": "zero",
                "kernels": "L5E5E5",
                "compute_energy": True,
                "energy_distance": 7,
            }
        )
        with self.assertRaises(native.UncontrolledNativeParameter):
            native._validate_zrad_static(laws)
        record = native._validate_zrad_static({**laws, "native_default": True})
        self.assertEqual(record["uncontrolled"], ["energy_boundary"])

        log = native.normalize_parameters(
            {"filter": "log", "dimensionality": 3, "sigma_mm": 1.5}
        )
        with self.assertRaises(native.UncontrolledNativeParameter):
            native._validate_pyradiomics_static(log)
        record = native._validate_pyradiomics_static({**log, "native_default": True})
        self.assertEqual(record["uncontrolled"], ["truncate", "boundary"])

        # A 2D LoG or a rotation-invariant wavelet is a different operation,
        # so a native-default run never replaces it.
        for params in (
            {"filter": "log", "dimensionality": 2, "sigma_mm": 1.5},
            {
                "filter": "wavelet",
                "dimensionality": 3,
                "wavelet": "db3",
                "decomposition": "LLH",
                "level": 1,
                "rotation_invariant": True,
                "pooling": "average",
            },
        ):
            normalized = {
                **native.normalize_parameters(params, adapter="pyradiomics"),
                "native_default": True,
            }
            with self.subTest(params=params):
                with self.assertRaises(native.UnsupportedNativeFilter) as caught:
                    native._validate_pyradiomics_static(normalized)
                self.assertNotIsInstance(
                    caught.exception, native.UncontrolledNativeParameter
                )

    def test_zrad_gabor_uses_the_full_ibsi_plus_minus_seven_sigma_support(self) -> None:
        params = native.normalize_parameters(
            {
                "filter": "gabor",
                "dimensionality": 2,
                "boundary": "zero",
                "sigma_mm": 10.0,
                "lambda_mm": 4.0,
                "gamma": 0.5,
                "theta": math.pi / 3.0,
            }
        )

        kwargs = native._zrad_gabor_kwargs(
            params,
            spacing=(2.0, 2.0, 2.0),
            boundary="constant",
        )

        self.assertEqual(kwargs["n_stds"], 14.0)
        self.assertEqual(kwargs["res_mm"], 2.0)
        self.assertAlmostEqual(kwargs["theta"], math.pi / 3.0)

    def test_medimage_maps_ibsi_gabor_angle_to_native_axis_handedness(self) -> None:
        captured = {}

        class FakeGabor:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def convolve(self, images, *, orthogonal_rot, pooling_method):
                return np.zeros_like(images)

        params = native.normalize_parameters(
            {
                "filter": "gabor",
                "dimensionality": 2,
                "boundary": "zero",
                "sigma_mm": 10.0,
                "lambda_mm": 4.0,
                "gamma": 0.5,
                "theta": math.pi / 3.0,
            }
        )
        with (
            mock.patch.object(
                native,
                "_load_nibabel",
                return_value=(mock.Mock(), np.zeros((3, 3, 3)), (2.0, 2.0, 2.0)),
            ),
            mock.patch.object(
                native,
                "_medimage_filter_module",
                return_value=mock.Mock(Gabor=FakeGabor),
            ),
            mock.patch.object(native, "_save_nibabel"),
        ):
            native._medimage_backend(
                Path("input.nii.gz"), Path("output.nii.gz"), params
            )

        self.assertAlmostEqual(captured["theta"], -math.pi / 3.0)

    def test_medimage_laws_conjugates_the_native_axis_reflections(self) -> None:
        image = np.arange(24, dtype=float).reshape(2, 3, 4)
        captured = {}

        def apply_laws(input_image, **kwargs):
            captured["input"] = np.asarray(input_image)
            captured["kwargs"] = kwargs
            return np.asarray(input_image) + 10.0

        def save_nibabel(reference, response, destination):
            captured["response"] = np.asarray(response)

        params = native.normalize_parameters(
            {
                "filter": "laws",
                "dimensionality": 3,
                "boundary": "zero",
                "kernels": "E5L5S5",
            }
        )
        with (
            mock.patch.object(
                native,
                "_load_nibabel",
                return_value=(mock.Mock(), image, (2.0, 2.0, 2.0)),
            ),
            mock.patch.object(
                native,
                "_medimage_filter_module",
                return_value=mock.Mock(apply_laws=apply_laws),
            ),
            mock.patch.object(native, "_save_nibabel", side_effect=save_nibabel),
        ):
            metadata = native._medimage_backend(
                Path("input.nii.gz"),
                Path("output.nii.gz"),
                params,
            )

        np.testing.assert_array_equal(
            captured["input"],
            np.flip(image, axis=(0, 1)),
        )
        np.testing.assert_array_equal(captured["response"], image + 10.0)
        self.assertEqual(
            metadata["coordinate_frame_correction"],
            "conjugated_reflection_axes_0_1",
        )

    def test_medimage_2d_wavelet_uses_one_native_call_per_axial_slice(self) -> None:
        constructor = {}
        calls = []

        class FakeWavelet:
            def __init__(self, **kwargs):
                constructor.update(kwargs)

            def convolve(self, images, *, _filter, level):
                calls.append((images.copy(), _filter, level))
                return images + len(calls)

        image = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        params = native.normalize_parameters(
            {
                "filter": "wavelet",
                "dimensionality": 2,
                "boundary": "mirror",
                "wavelet": "db3",
                "level": 1,
                "decomposition": "LH",
                "rotation_invariant": False,
                "pooling": "average",
            }
        )
        save = mock.Mock()
        with (
            mock.patch.object(
                native,
                "_load_nibabel",
                return_value=(mock.Mock(), image, (1.0, 1.0, 3.0)),
            ),
            mock.patch.object(
                native,
                "_medimage_filter_module",
                return_value=mock.Mock(Wavelet=FakeWavelet),
            ),
            mock.patch.object(native, "_save_nibabel", save),
        ):
            metadata = native._medimage_backend(
                Path("input.nii.gz"), Path("output.nii.gz"), params
            )

        self.assertEqual(constructor["ndims"], 2)
        self.assertIs(constructor["rot_invariance"], False)
        self.assertEqual(len(calls), image.shape[2])
        for index, (native_input, decomposition, level) in enumerate(calls):
            self.assertEqual(native_input.shape, (1, *image.shape[:2]))
            np.testing.assert_array_equal(native_input[0], image[:, :, index])
            self.assertEqual(decomposition, "LH")
            self.assertEqual(level, 1)
        saved_response = save.call_args.args[1]
        self.assertEqual(saved_response.shape, image.shape)
        for index in range(image.shape[2]):
            np.testing.assert_array_equal(
                saved_response[:, :, index], image[:, :, index] + index + 1
            )
        self.assertEqual(metadata["shape"], list(image.shape))

    def test_medimage_level1_2d_wavelet_uses_four_quarter_turns(self) -> None:
        constructor = {}
        calls = []

        class FakeWavelet:
            def __init__(self, **kwargs):
                constructor.update(kwargs)

            def convolve(self, images, *, _filter, level):
                calls.append(images.copy())
                return images + len(calls)

        image = np.arange(12, dtype=np.float64).reshape(2, 3, 2)
        params = native.normalize_parameters(
            {
                "filter": "wavelet",
                "dimensionality": 2,
                "boundary": "mirror",
                "wavelet": "db3",
                "level": 1,
                "decomposition": "LH",
                "rotation_invariant": True,
                "pooling": "average",
            }
        )
        save = mock.Mock()
        with (
            mock.patch.object(
                native,
                "_load_nibabel",
                return_value=(mock.Mock(), image, (1.0, 1.0, 3.0)),
            ),
            mock.patch.object(
                native,
                "_medimage_filter_module",
                return_value=mock.Mock(Wavelet=FakeWavelet),
            ),
            mock.patch.object(native, "_save_nibabel", save),
        ):
            metadata = native._medimage_backend(
                Path("input.nii.gz"), Path("output.nii.gz"), params
            )

        self.assertIs(constructor["rot_invariance"], False)
        self.assertEqual(len(calls), 4 * image.shape[2])
        for slice_index in range(image.shape[2]):
            image_slice = image[:, :, slice_index]
            for quarter_turns in range(4):
                native_input = calls[4 * slice_index + quarter_turns]
                np.testing.assert_array_equal(
                    native_input[0],
                    np.rot90(image_slice, quarter_turns),
                )
        saved_response = save.call_args.args[1]
        np.testing.assert_allclose(saved_response[:, :, 0], image[:, :, 0] + 2.5)
        np.testing.assert_allclose(saved_response[:, :, 1], image[:, :, 1] + 6.5)
        self.assertEqual(
            metadata["rotation_orchestration"],
            "four_quarter_turn_native_dwt_average",
        )


if __name__ == "__main__":
    unittest.main()

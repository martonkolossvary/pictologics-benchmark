from __future__ import annotations

import argparse
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from bench.adapters import (
    medimage_adapter,
    mirp_adapter,
    pictologics_adapter,
    pyradiomics_adapter,
)
from bench.adapters.protocol import (
    add_common_arguments,
    parse_intensity_range,
    requested_families,
    roi_crop,
)


class CommonRangeTests(unittest.TestCase):
    def _parse(self, *extra: str):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        return parser.parse_args(["--image", "image", "--mask", "mask", *extra])

    def test_common_range_accepts_only_a_finite_strictly_ordered_pair(self) -> None:
        args = self._parse("--intensity-min", "-10", "--intensity-max", "20")
        self.assertEqual(parse_intensity_range(args), (-10.0, 20.0))

        invalid = [
            ("--intensity-min", "-10"),
            ("--intensity-min", "-10", "--intensity-max", "inf"),
            ("--intensity-min", "2", "--intensity-max", "2"),
            ("--intensity-min", "3", "--intensity-max", "2"),
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_intensity_range(self._parse(*values))

    def test_common_protocol_accepts_identity_and_explicit_aggregation(self) -> None:
        args = self._parse("--discretization", "identity", "--aggregation", "3d_merge")
        self.assertEqual(args.discretization, "identity")
        self.assertEqual(args.aggregation, "3d_merge")

    def test_common_protocol_accepts_raw_first_order_extraction(self) -> None:
        args = self._parse("--discretization", "raw", "--aggregation", "3d_merge")
        self.assertEqual(args.discretization, "raw")
        self.assertEqual(args.aggregation, "3d_merge")

    def test_common_protocol_defaults_to_required_3d_merged_aggregation(self) -> None:
        import argparse

        from bench.adapters.protocol import add_common_arguments

        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args(["--image", "image.nii", "--mask", "mask.nii"])
        self.assertEqual(args.aggregation, "3d_merge")

    def test_common_protocol_rejects_invalid_discretization_parameters(self) -> None:
        for extra in (
            ("--bins", "0"),
            ("--bin-width", "nan"),
            ("--bin-width", "inf"),
            ("--bin-width", "0"),
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                requested_families(
                    "pictologics",
                    self._parse("--families", "intensity", *extra),
                )


class RoiCropTests(unittest.TestCase):
    def test_crop_keeps_one_voxel_or_the_local_intensity_sphere(self) -> None:
        mask = np.zeros((40, 40, 20), dtype=np.uint8)
        mask[10:12, 20:25, 5:6] = 1

        slices, record = roi_crop(mask, (1.0, 2.0, 4.0), ["glcm"])
        self.assertEqual(slices, (slice(9, 13), slice(19, 26), slice(4, 7)))
        self.assertEqual(record["padding_voxels"], [1, 1, 1])

        slices, record = roi_crop(mask, (1.0, 2.0, 4.0), ["local_intensity"])
        self.assertEqual(record["padding_voxels"], [7, 4, 2])
        self.assertEqual(slices, (slice(3, 19), slice(16, 29), slice(3, 8)))
        self.assertEqual(record["cropped_shape"], [16, 13, 5])

    def test_crop_rejects_an_empty_mask(self) -> None:
        with self.assertRaises(ValueError):
            roi_crop(np.zeros((3, 3, 3)), (1.0, 1.0, 1.0), ["glcm"])


class PyRadiomicsAggregationTests(unittest.TestCase):
    def test_native_family_selection_exposes_complete_eligible_3d_surface(self) -> None:
        registry = {
            "shape": {
                "MeshVolume": False,
                "Maximum2DDiameterColumn": False,
                "Maximum2DDiameterRow": False,
                "Maximum2DDiameterSlice": False,
                "Compactness1": True,
                "UnapprovedDeprecatedShape": True,
            },
            "shape2D": {"PixelSurface": False},
            "firstorder": {
                "Mean": False,
                "Entropy": False,
                "Uniformity": False,
                "TotalEnergy": False,
                "StandardDeviation": True,
            },
            "glcm": {
                "JointEnergy": False,
                "MCC": False,
                "Dissimilarity": True,
            },
            "gldm": {
                "DependenceEntropy": False,
                "DependencePercentage": True,
                "GrayLevelNonUniformityNormalized": True,
            },
        }

        selection = pyradiomics_adapter._native_family_selection(
            ["morphology", "intensity", "histogram", "glcm", "ngldm"],
            registry,
        )

        self.assertEqual(
            selection["shape"],
            [
                "Compactness1",
                "Maximum2DDiameterColumn",
                "Maximum2DDiameterRow",
                "Maximum2DDiameterSlice",
                "MeshVolume",
            ],
        )
        self.assertNotIn("shape2D", selection)
        self.assertEqual(
            selection["firstorder"],
            ["Entropy", "Mean", "TotalEnergy", "Uniformity"],
        )
        self.assertEqual(selection["glcm"], ["JointEnergy", "MCC"])
        self.assertEqual(selection["gldm"], ["DependenceEntropy"])

    def test_native_firstorder_surface_is_partitioned_without_duplicates(self) -> None:
        registry = {
            "firstorder": {
                "Mean": False,
                "Entropy": False,
                "Uniformity": False,
                "TotalEnergy": False,
            }
        }

        intensity = pyradiomics_adapter._native_family_selection(
            ["intensity"], registry
        )
        histogram = pyradiomics_adapter._native_family_selection(
            ["histogram"], registry
        )

        self.assertEqual(intensity, {"firstorder": ["Mean", "TotalEnergy"]})
        self.assertEqual(histogram, {"firstorder": ["Entropy", "Uniformity"]})
        self.assertFalse(
            set(intensity["firstorder"]).intersection(histogram["firstorder"])
        )

    def test_native_selection_fails_when_pinned_class_is_missing(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError, "lacks required feature classes: glcm"
        ):
            pyradiomics_adapter._native_family_selection(["glcm"], {})

    def test_3d_merge_uses_documented_unit_weighted_matrix_sum(self) -> None:
        self.assertEqual(
            pyradiomics_adapter._aggregation_settings("3d_merge", ["glcm", "glrlm"]),
            {"weightingNorm": "no_weighting"},
        )

    def test_native_average_does_not_enable_matrix_sum(self) -> None:
        self.assertEqual(
            pyradiomics_adapter._aggregation_settings("3d_average", ["glcm"]),
            {},
        )

    def test_nondirectional_family_does_not_receive_weighting_setting(self) -> None:
        self.assertEqual(
            pyradiomics_adapter._aggregation_settings("3d_merge", ["glszm"]),
            {},
        )

    def test_calculated_values_require_one_finite_scalar_each(self) -> None:
        valid = pyradiomics_adapter._finite_calculated_values(
            {
                "diagnostics_Image-original_Hash": "ignored",
                "original_firstorder_Mean": np.asarray([3.0]),
                "original_firstorder_Kurtosis": 5.0,
            }
        )
        self.assertEqual(valid["original_firstorder_Mean"], 3.0)
        self.assertEqual(valid["original_firstorder_Kurtosis"], 2.0)
        for value in (np.nan, [1.0, 2.0], "1.0", "not-a-number", True):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    RuntimeError,
                    "finite scalar|numeric scalar",
                ),
            ):
                pyradiomics_adapter._finite_calculated_values(
                    {"original_firstorder_Mean": value}
                )

    def test_calculated_feature_names_are_trimmed_before_validation(self) -> None:
        values = pyradiomics_adapter._finite_calculated_values(
            {" original_firstorder_Mean ": 3.0}
        )
        self.assertEqual(values, {"original_firstorder_Mean": 3.0})

        for payload in (
            {"   ": 1.0},
            {
                "original_firstorder_Mean": 1.0,
                " original_firstorder_Mean ": 2.0,
            },
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaisesRegex(RuntimeError, "empty or duplicate feature name"),
            ):
                pyradiomics_adapter._finite_calculated_values(payload)

    def test_timed_run_honors_common_no_warmup(self) -> None:
        events = []
        radiomics = types.ModuleType("radiomics")
        radiomics.__path__ = []
        simple_itk = types.ModuleType("SimpleITK")

        class FeatureClass:
            @classmethod
            def getFeatureNames(cls):
                return {"Mean": False}

        class Image:
            def __init__(self, path):
                self.path = path

            def __gt__(self, value):
                events.append(("binarize", self.path, value))
                return self

            def GetSpacing(self):
                return (1.0, 1.0, 1.0)

        def read_image(path):
            events.append(("read_image", path))
            return Image(path)

        def cast(image, pixel_type):
            events.append(("cast", image.path, pixel_type))
            return image

        radiomics.getFeatureClasses = lambda: {"firstorder": FeatureClass}
        simple_itk.ReadImage = read_image
        simple_itk.Cast = cast
        simple_itk.sitkUInt8 = "uint8"
        simple_itk.GetArrayViewFromImage = lambda image: np.ones((2, 2, 2))
        modules = mock.patch.dict(
            sys.modules,
            {
                "radiomics": radiomics,
                "SimpleITK": simple_itk,
            },
        )

        def prepare(**kwargs):
            events.append(("prepare", kwargs["selection"]))

            def calculate():
                events.append(("execute",))
                return {"original_firstorder_Mean": 3.0}

            return calculate, lambda raw, _state=None: raw

        with (
            modules,
            mock.patch.object(
                pyradiomics_adapter,
                "_prepare_calculation_only",
                side_effect=prepare,
            ),
            mock.patch.object(pyradiomics_adapter, "write_json") as write_json,
        ):
            result = pyradiomics_adapter.main(
                [
                    "--image",
                    "image.nii.gz",
                    "--mask",
                    "mask.nii.gz",
                    "--families",
                    "intensity",
                    "--timed",
                    "--iterations",
                    "3",
                    "--no-warmup",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(sum(event[0] == "read_image" for event in events), 2)
        self.assertEqual(sum(event[0] == "cast" for event in events), 1)
        self.assertEqual(sum(event[0] == "prepare" for event in events), 1)
        self.assertEqual(sum(event[0] == "execute" for event in events), 3)
        payload = write_json.call_args.args[0]
        self.assertEqual(payload["timing"]["measured_iterations"], 3)
        self.assertEqual(payload["timing"]["warmup_iterations"], 0)
        self.assertEqual(
            payload["metadata"]["feature_selection"],
            {
                "contract": "complete_native_3d_family_surface",
                "dimension": "3d",
                "shape2d_class_enabled": False,
                "selected_by_class": {"firstorder": ["Mean"]},
            },
        )

    def test_shape_mesh_construction_is_inside_the_calculation_clock(self) -> None:
        events = []
        radiomics = types.ModuleType("radiomics")
        radiomics.__path__ = []
        imageoperations = types.ModuleType("radiomics.imageoperations")
        imageoperations.checkMask = lambda image, mask, **kwargs: ("box", None)
        imageoperations.cropToTumorMask = lambda image, mask, box, padDistance: (
            image,
            mask,
        )
        radiomics.imageoperations = imageoperations

        def feature_class(class_name):
            class FeatureClass:
                def __init__(self, image, mask, **kwargs):
                    events.append(("construct", class_name))

                def enableFeatureByName(self, name):
                    pass

                def execute(self):
                    events.append(("execute", class_name))
                    return {"Value": 1.0}

            return FeatureClass

        radiomics_module = SimpleNamespace(
            getFeatureClasses=lambda: {
                "shape": feature_class("shape"),
                "glcm": feature_class("glcm"),
            }
        )
        with mock.patch.dict(
            sys.modules,
            {"radiomics": radiomics, "radiomics.imageoperations": imageoperations},
        ):
            calculate, _ = pyradiomics_adapter._prepare_calculation_only(
                image="image",
                mask="mask",
                selection={"shape": ["MeshVolume"], "glcm": ["Contrast"]},
                settings={"label": 1},
                radiomics_module=radiomics_module,
            )
            self.assertEqual(events, [("construct", "glcm")])
            calculate()
            calculate()

        self.assertEqual(
            events[1:],
            [
                ("construct", "shape"),
                ("execute", "shape"),
                ("execute", "glcm"),
                ("construct", "shape"),
                ("execute", "shape"),
                ("execute", "glcm"),
            ],
        )


def _prepare_and_compute_medimage(
    *,
    families,
    modules,
    image_full,
    binary_mask,
    spacing,
    intensity_type,
    discretization,
    bins,
    bin_width,
    intensity_range,
):
    """Prepare the MEDimage volumes and calculate the families, as the adapter does."""

    vol_raw, vol_quant, vol_ivh, wd, intensity_mask = (
        medimage_adapter._prepare_roi_volumes(
            image_full,
            binary_mask,
            discretization=discretization,
            bins=bins,
            bin_width=bin_width,
            intensity_range=intensity_range,
            apply_discretization=bool(
                set(families).intersection(medimage_adapter._DISCRETIZED_FAMILIES)
            ),
        )
    )
    return medimage_adapter._compute_features(
        families=families,
        modules=modules,
        image_full=image_full,
        vol_raw=vol_raw,
        vol_quant=vol_quant,
        vol_ivh=vol_ivh,
        morphology_mask_bool=binary_mask,
        intensity_mask_bool=intensity_mask,
        spacing=spacing,
        intensity_type=intensity_type,
        wd=wd,
        discretization=discretization,
        intensity_range=intensity_range,
    )


def fake_pictologics_modules():
    calls = []
    package = types.ModuleType("pictologics")
    package.__path__ = []
    preprocessing = types.ModuleType("pictologics.preprocessing")
    features = types.ModuleType("pictologics.features")
    loader = types.ModuleType("pictologics.loader")
    warmup = types.ModuleType("pictologics.warmup")

    class Image:
        def __init__(
            self,
            array,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=None,
            modality="ct",
        ):
            self.array = np.asarray(array)
            self.spacing = spacing
            self.origin = origin
            self.direction = direction
            self.modality = modality

    raw_image = Image(np.arange(8, dtype=float).reshape(2, 2, 2))
    raw_mask = Image(np.ones((2, 2, 2), dtype=np.uint8))

    def resegment_mask(image, mask, **kwargs):
        calls.append(("resegment_mask", kwargs))
        return Image(mask.array.copy())

    def discretise_image(image, **kwargs):
        calls.append(("discretise_image", kwargs))
        return Image(np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2))

    def apply_mask(image, mask):
        return image.array[mask.array > 0]

    def ivh(values, **kwargs):
        calls.append(("calculate_ivh_features", kwargs))
        return {"ivh_feature": 1.0}

    preprocessing.resegment_mask = resegment_mask
    preprocessing.discretise_image = discretise_image
    preprocessing.apply_mask = apply_mask
    features.calculate_ivh_features = ivh
    for name in (
        "calculate_all_texture_matrices",
        "calculate_glcm_features",
        "calculate_gldzm_features",
        "calculate_glrlm_features",
        "calculate_glszm_features",
        "calculate_intensity_features",
        "calculate_intensity_histogram_features",
        "calculate_local_intensity_features",
        "calculate_morphology_features",
        "calculate_ngldm_features",
        "calculate_ngtdm_features",
        "calculate_spatial_intensity_features",
    ):
        setattr(features, name, lambda *args, **kwargs: {})

    loader.Image = Image
    load_calls = iter((raw_image, raw_mask))
    loader.load_image = lambda path: next(load_calls)
    warmup.warmup_jit = lambda: calls.append(("warmup_jit", {}))
    package.get_num_threads = lambda: 1
    family_modules = {
        name: types.ModuleType(f"pictologics.features.{name}")
        for name in ("intensity", "morphology", "texture")
    }
    for name, module in family_modules.items():
        setattr(features, name, module)
    modules = {
        "pictologics": package,
        "pictologics.preprocessing": preprocessing,
        "pictologics.features": features,
        "pictologics.loader": loader,
        "pictologics.warmup": warmup,
        **{f"pictologics.features.{n}": m for n, m in family_modules.items()},
    }
    return mock.patch.dict(sys.modules, modules), calls, raw_image, raw_mask


class PictologicsGroupTests(unittest.TestCase):
    def test_texture_group_builds_all_matrices_in_one_pass(self) -> None:
        modules, calls, image, mask = fake_pictologics_modules()
        passes = []
        families = ["histogram", "glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"]

        def all_matrices(*args, **kwargs):
            passes.append({k: v for k, v in kwargs.items() if k.startswith("calc_")})
            keys = ("glcm", "glrlm", "glszm", "gldzm", "ngtdm_s", "ngtdm_n", "ngldm")
            return {key: None for key in keys}

        with modules:
            from pictologics import features, preprocessing

            features.calculate_all_texture_matrices = all_matrices
            image.array = np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2)
            calculate, _ = pictologics_adapter._prepare_calculation_only(
                image=image,
                mask=mask,
                families=families,
                discretization="identity",
                bins=32,
                bin_width=1.0,
                intensity_range=None,
                feature_modules=(preprocessing, features),
            )
            calculate()
        self.assertEqual(len(passes), 1)
        self.assertTrue(all(passes[0].values()))
        self.assertEqual(len(passes[0]), 6)


class PictologicsValueValidationTests(unittest.TestCase):
    def test_calculated_values_require_one_finite_scalar_each(self) -> None:
        self.assertEqual(
            pictologics_adapter._finite_feature_mapping({"mean": np.asarray([3.0])}),
            {"mean": 3.0},
        )
        for value in (np.inf, [], [1.0, 2.0], "1.0", "not-a-number", True):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    RuntimeError,
                    "finite scalar|numeric scalar",
                ),
            ):
                pictologics_adapter._finite_feature_mapping({"mean": value})

    def test_calculated_feature_names_are_nonempty_and_unique_after_normalization(
        self,
    ) -> None:
        for values in ({"   ": 1.0}, {1: 1.0, "1": 2.0}):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(RuntimeError, "empty or duplicate"),
            ):
                pictologics_adapter._finite_feature_mapping(values)

    def test_cross_family_feature_names_cannot_be_overwritten(self) -> None:
        target = {"shared": 1.0}
        with self.assertRaisesRegex(RuntimeError, "across families"):
            pictologics_adapter._merge_feature_mapping(target, {"shared": 2.0})


class PictologicsRangeTests(unittest.TestCase):
    def test_timed_run_honors_common_no_warmup(self) -> None:
        modules, calls, _, _ = fake_pictologics_modules()
        image_digest = "a" * 64
        mask_digest = "b" * 64
        with (
            mock.patch.dict(
                os.environ,
                {"PICTOLOGICS_DISABLE_WARMUP": "0"},
            ),
            modules,
            mock.patch.object(
                pictologics_adapter,
                "_prepare_calculation_only",
                side_effect=lambda **kwargs: (
                    lambda: [{"intensity_feature": 1.0}],
                    lambda raw, _state=None: dict(raw[0]),
                ),
            ) as prepare,
            mock.patch.object(pictologics_adapter, "write_json") as write_json,
        ):
            result = pictologics_adapter.main(
                [
                    "--image",
                    "image.nii.gz",
                    "--mask",
                    "mask.nii.gz",
                    "--image-sha256",
                    image_digest,
                    "--mask-sha256",
                    mask_digest,
                    "--families",
                    "intensity",
                    "--timed",
                    "--iterations",
                    "3",
                    "--no-warmup",
                ]
            )
            effective_warmup_environment = os.environ.get("PICTOLOGICS_DISABLE_WARMUP")

        self.assertEqual(result, 0)
        self.assertEqual(effective_warmup_environment, "1")
        self.assertEqual(prepare.call_count, 1)
        # --no-warmup drops the discarded call; the package JIT warmup of a
        # plain import still runs once, before the files are read.
        self.assertEqual(sum(event[0] == "warmup_jit" for event in calls), 1)
        payload = write_json.call_args.args[0]
        self.assertEqual(payload["timing"]["measured_iterations"], 3)
        self.assertEqual(payload["timing"]["warmup_iterations"], 0)
        self.assertEqual(
            payload["metadata"]["input"]["image_sha256"],
            image_digest,
        )
        initialization = payload["metadata"]["package_initialization"]
        self.assertTrue(initialization["jit_warmup_performed"])
        self.assertTrue(initialization["outside_measured_region"])
        self.assertEqual(initialization["threads"], 1)

    def test_identity_discretization_preserves_positive_integer_levels(self) -> None:
        modules, _, image, mask = fake_pictologics_modules()
        image.array = np.array([[[-10.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]])
        mask.array = np.ones_like(image.array, dtype=np.uint8)
        mask.array[0, 0, 0] = 0
        with modules:
            discrete = pictologics_adapter._identity_discrete_image(image, mask)
        self.assertEqual(discrete.array[0, 0, 0], 0)
        np.testing.assert_array_equal(
            discrete.array[mask.array > 0], image.array[mask.array > 0]
        )

        image.array[0, 0, 1] = 1.25
        with modules, self.assertRaisesRegex(ValueError, "positive-integer"):
            pictologics_adapter._identity_discrete_image(image, mask)

    def test_fbs_uses_resegmentation_anchor_and_physical_ivh_axis(self) -> None:
        modules, calls, image, mask = fake_pictologics_modules()
        with modules:
            values = pictologics_adapter._compute_features(
                image=image,
                mask=mask,
                families=["ivh"],
                discretization="fbs",
                bins=32,
                bin_width=25.0,
                intensity_range=(-1000.0, 400.0),
            )

        self.assertEqual(values, {"ivh_feature": 1.0})
        discretise = next(value for name, value in calls if name == "discretise_image")
        self.assertEqual(discretise["method"], "FBS")
        self.assertEqual(discretise["min_val"], -1000.0)
        self.assertEqual(discretise["max_val"], 400.0)
        self.assertEqual(discretise["bin_width"], 25.0)
        ivh = next(value for name, value in calls if name == "calculate_ivh_features")
        self.assertEqual(
            ivh,
            {
                "bin_width": 25.0,
                "min_val": -1000.0,
                "max_val": 400.0,
                "target_range_min": -1000.0,
                "target_range_max": 400.0,
            },
        )

    def test_fbn_preserves_observed_roi_discretisation_and_unit_ivh_steps(self) -> None:
        modules, calls, image, mask = fake_pictologics_modules()
        with modules:
            pictologics_adapter._compute_features(
                image=image,
                mask=mask,
                families=["ivh"],
                discretization="fbn",
                bins=16,
                bin_width=3.0,
                intensity_range=(-1000.0, 400.0),
            )

        discretise = next(value for name, value in calls if name == "discretise_image")
        self.assertEqual(discretise["method"], "FBN")
        self.assertEqual(discretise["n_bins"], 16)
        self.assertNotIn("min_val", discretise)
        self.assertNotIn("max_val", discretise)
        ivh = next(value for name, value in calls if name == "calculate_ivh_features")
        self.assertEqual(ivh, {"bin_width": 1.0})

    def test_morphology_receives_the_resegmented_intensity_mask(self) -> None:
        modules, calls, image, mask = fake_pictologics_modules()
        with modules:
            from pictologics import features

            def morphology(mask_arg, **kwargs):
                calls.append(
                    (
                        "calculate_morphology_features",
                        {"mask": mask_arg, **kwargs},
                    )
                )
                return {"morphology_feature": 1.0}

            features.calculate_morphology_features = morphology
            values = pictologics_adapter._compute_features(
                image=image,
                mask=mask,
                families=["morphology"],
                discretization="fbn",
                bins=16,
                bin_width=3.0,
                intensity_range=(2.0, 7.0),
            )

        self.assertEqual(values, {"morphology_feature": 1.0})
        call = next(
            value for name, value in calls if name == "calculate_morphology_features"
        )
        self.assertIs(call["mask"], mask)
        self.assertIs(call["image"], image)
        self.assertIsNot(call["intensity_mask"], mask)
        np.testing.assert_array_equal(
            call["intensity_mask"].array,
            mask.array,
        )

    def test_histogram_receives_full_discrete_grey_level_range(self) -> None:
        modules, calls, image, mask = fake_pictologics_modules()
        with modules:
            from pictologics import features

            def histogram(values, **kwargs):
                calls.append(("calculate_intensity_histogram_features", kwargs))
                return {"histogram_feature": 1.0}

            features.calculate_intensity_histogram_features = histogram
            values = pictologics_adapter._compute_features(
                image=image,
                mask=mask,
                families=["histogram"],
                discretization="fbn",
                bins=16,
                bin_width=3.0,
                intensity_range=None,
            )

        self.assertEqual(values, {"histogram_feature": 1.0})
        histogram_call = next(
            value
            for name, value in calls
            if name == "calculate_intensity_histogram_features"
        )
        self.assertEqual(histogram_call, {"n_bins": 16})

    def test_main_records_effective_range_and_anchor(self) -> None:
        modules, _, _, _ = fake_pictologics_modules()
        with (
            modules,
            mock.patch.object(
                pictologics_adapter,
                "_compute_features",
                return_value={"ivh_feature": 1.0},
            ),
            mock.patch.object(pictologics_adapter, "write_json") as write_json,
        ):
            result = pictologics_adapter.main(
                [
                    "--image",
                    "image.nii.gz",
                    "--mask",
                    "mask.nii.gz",
                    "--families",
                    "ivh",
                    "--discretization",
                    "fbs",
                    "--bin-width",
                    "25",
                    "--intensity-min",
                    "-1000",
                    "--intensity-max",
                    "400",
                    "--no-warmup",
                ]
            )

        self.assertEqual(result, 0)
        preprocessing = write_json.call_args.args[0]["metadata"]["preprocessing"]
        self.assertEqual(preprocessing["intensity_range"], [-1000.0, 400.0])
        self.assertEqual(preprocessing["fbs_anchor"], -1000.0)


class FakeTable:
    def __init__(self, values):
        self._values = values
        self.columns = list(values)
        self.empty = False
        self.shape = (1, len(values))
        self.iloc = self

    def __getitem__(self, index):
        if index == 0:
            return self._values
        raise IndexError(index)


class FakeMirpImage:
    image_spacing = (1.0, 1.0, 1.0)

    def __init__(self):
        self.crops = []
        self.grid = np.zeros((1, 1, 1))

    def crop(self, **bounds):
        self.crops.append(bounds)

    def get_voxel_grid(self):
        return self.grid

    def set_voxel_grid(self, voxel_grid):
        self.grid = voxel_grid


class FakeMirpMask(FakeMirpImage):
    def __init__(self):
        super().__init__()
        grid = np.zeros((6, 6, 6), dtype=bool)
        grid[2:4, 3, 3:5] = True
        self.roi = SimpleNamespace(get_voxel_grid=lambda: grid)


def fake_mirp_settings_modules():
    events = []
    mirp = types.ModuleType("mirp")
    mirp.__path__ = []
    settings_package = types.ModuleType("mirp.settings")
    settings_package.__path__ = []
    feature_module = types.ModuleType("mirp.settings.feature_parameters")
    generic_module = types.ModuleType("mirp.settings.generic")
    resegmentation_module = types.ModuleType("mirp.settings.resegmentation_parameters")

    class FeatureExtractionSettingsClass:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("feature_settings", kwargs))

    class ResegmentationSettingsClass:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("resegmentation_settings", kwargs))

    class SettingsClass:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("settings", kwargs))

    feature_module.FeatureExtractionSettingsClass = FeatureExtractionSettingsClass
    resegmentation_module.ResegmentationSettingsClass = ResegmentationSettingsClass
    generic_module.SettingsClass = SettingsClass
    modules = {
        "mirp": mirp,
        "mirp.settings": settings_package,
        "mirp.settings.feature_parameters": feature_module,
        "mirp.settings.generic": generic_module,
        "mirp.settings.resegmentation_parameters": resegmentation_module,
    }
    return mock.patch.dict(sys.modules, modules), mirp, events


class MirpPublicApiTests(unittest.TestCase):
    def test_benchmark_only_modalities_use_generic_without_losing_provenance(
        self,
    ) -> None:
        cases = {
            "CT": "CT",
            "MRI": "MRI",
            "PET": "PET",
            "synthetic": "ct",
            "other": None,
        }
        for benchmark_modality, expected_mirp_modality in cases.items():
            modules, mirp, events = fake_mirp_settings_modules()

            def extract_images(**kwargs):
                events.append(("extract_images", kwargs))
                return [([FakeMirpImage()], [FakeMirpMask()])]

            def extract_features(**kwargs):
                return [FakeTable({"sample_name": "case", "stat_mean": 3.0})]

            mirp.extract_images = extract_images
            mirp.extract_features = extract_features
            with (
                self.subTest(modality=benchmark_modality),
                modules,
                mock.patch.object(mirp_adapter, "write_json") as write_json,
            ):
                result = mirp_adapter.main(
                    [
                        "--image",
                        "image.nii.gz",
                        "--mask",
                        "mask.nii.gz",
                        "--modality",
                        benchmark_modality,
                        "--families",
                        "intensity",
                    ]
                )

                self.assertEqual(result, 0)
                load_kwargs = next(
                    value for name, value in events if name == "extract_images"
                )
                self.assertEqual(
                    load_kwargs.get("image_modality"), expected_mirp_modality
                )
                payload = write_json.call_args.args[0]
                self.assertEqual(
                    payload["metadata"]["input"]["modality"], benchmark_modality
                )
                self.assertEqual(
                    payload["metadata"]["modality_bridge"],
                    {
                        "benchmark": benchmark_modality,
                        "effective_mirp": expected_mirp_modality,
                    },
                )
                feature_settings = next(
                    value for name, value in events if name == "feature_settings"
                )
                self.assertFalse(feature_settings["ibsi_compliant"])
                self.assertEqual(
                    payload["metadata"]["feature_selection"],
                    {
                        "contract": "complete_native_3d_family_surface",
                        "mirp_ibsi_compliant_filter": False,
                    },
                )
                self.assertIn(
                    "internal_copy_inside_timing",
                    payload["metadata"]["input_execution_scope"],
                )

    def test_code_selected_compliance_retains_mirp_ibsi_filter(self) -> None:
        modules, mirp, events = fake_mirp_settings_modules()
        mirp.extract_images = lambda **kwargs: [([FakeMirpImage()], [FakeMirpMask()])]
        mirp.extract_features = lambda **kwargs: [
            FakeTable({"sample_name": "case", "stat_mean": 3.0})
        ]
        with (
            modules,
            mock.patch.object(
                mirp_adapter,
                "_table_to_values",
                return_value={"stat_mean": 3.0},
            ),
            mock.patch(
                "bench.ibsi_mapping.mirp_families_for_codes",
                return_value=["statistics"],
            ),
            mock.patch(
                "bench.ibsi_mapping.classify_feature",
                return_value=("Q4LE", "mapped"),
            ),
            mock.patch.object(mirp_adapter, "write_json") as write_json,
        ):
            result = mirp_adapter.main(
                [
                    "--image",
                    "image.nii.gz",
                    "--mask",
                    "mask.nii.gz",
                    "--families",
                    "intensity",
                    "--include-ibsi-codes",
                    "Q4LE",
                ]
            )

        self.assertEqual(result, 0)
        feature_settings = next(
            value for name, value in events if name == "feature_settings"
        )
        self.assertTrue(feature_settings["ibsi_compliant"])
        self.assertEqual(
            write_json.call_args.args[0]["metadata"]["feature_selection"],
            {
                "contract": "strict_ibsi_compliant_code_selection",
                "mirp_ibsi_compliant_filter": True,
            },
        )

    def test_timed_run_honors_common_no_warmup(self) -> None:
        modules, mirp, events = fake_mirp_settings_modules()
        native_image = FakeMirpImage()
        native_mask = FakeMirpMask()

        def extract_images(**kwargs):
            events.append(("extract_images", kwargs))
            return [([native_image], [native_mask])]

        def extract_features(**kwargs):
            events.append(("extract_features", kwargs))
            return [FakeTable({"sample_name": "case", "stat_mean": 3.0})]

        mirp.extract_images = extract_images
        mirp.extract_features = extract_features

        def prepare(**kwargs):
            events.append(("prepare", tuple(kwargs["families"])))

            def calculate():
                events.append(("calculate", tuple(kwargs["families"])))
                return [object()]

            def finalize(_features, _state=None):
                events.append(("finalize", tuple(kwargs["families"])))
                return {"stat_mean": 3.0}

            return calculate, finalize

        with (
            modules,
            mock.patch.object(
                mirp_adapter,
                "_prepare_calculation_only",
                side_effect=prepare,
            ),
            mock.patch.object(mirp_adapter, "write_json") as write_json,
        ):
            result = mirp_adapter.main(
                [
                    "--image",
                    "image.nii.gz",
                    "--mask",
                    "mask.nii.gz",
                    "--families",
                    "intensity",
                    "--timed",
                    "--iterations",
                    "3",
                    "--no-warmup",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            sum(event[0] == "extract_images" for event in events),
            1,
        )
        self.assertEqual(
            sum(event[0] == "prepare" for event in events),
            1,
        )
        self.assertEqual(
            sum(event[0] == "calculate" for event in events),
            3,
        )
        self.assertEqual(sum(event[0] == "extract_features" for event in events), 0)
        payload = write_json.call_args.args[0]
        self.assertEqual(payload["timing"]["measured_iterations"], 3)
        self.assertEqual(payload["timing"]["warmup_iterations"], 0)
        # The ROI box (z 2-3, y 3, x 3-4) plus one voxel on every side.
        bounds = {"ind_ext_z": [1, 4], "ind_ext_y": [2, 4], "ind_ext_x": [2, 5]}
        self.assertEqual(native_image.crops, [bounds])
        self.assertEqual(native_mask.crops, [bounds])
        self.assertEqual(payload["metadata"]["roi_crop"]["cropped_shape"], [4, 3, 4])

    def test_identity_and_directional_aggregation_are_explicit_settings(self) -> None:
        modules, _, events = fake_mirp_settings_modules()
        with modules:
            mirp_adapter._build_settings(
                families=["glcm", "glrlm", "ivh"],
                discretization="identity",
                bins=0,
                bin_width=float("nan"),
                intensity_range=None,
                aggregation="3d_average",
                identity_ivh_bins=6,
            )
        feature_settings = next(
            value for name, value in events if name == "feature_settings"
        )
        self.assertEqual(feature_settings["base_discretisation_method"], "none")
        self.assertEqual(
            feature_settings["ivh_discretisation_method"], "fixed_bin_number"
        )
        self.assertEqual(feature_settings["ivh_discretisation_n_bins"], 6)
        self.assertNotIn("base_discretisation_n_bins", feature_settings)
        self.assertNotIn("base_discretisation_bin_width", feature_settings)
        self.assertEqual(feature_settings["glcm_spatial_method"], "3d_average")
        self.assertEqual(feature_settings["glrlm_spatial_method"], "3d_average")

    def test_two_and_a_half_d_profile_uses_exact_ibsi_spatial_methods(self) -> None:
        modules, _, events = fake_mirp_settings_modules()
        with modules:
            mirp_adapter._build_settings(
                families=["glcm", "glszm", "ngldm"],
                discretization="fbn",
                bins=32,
                bin_width=1.0,
                intensity_range=None,
                aggregation="2.5d_direction_merge",
            )
        feature_settings = next(
            value for name, value in events if name == "feature_settings"
        )
        self.assertEqual(
            feature_settings["glcm_spatial_method"], "2.5d_direction_merge"
        )
        self.assertEqual(feature_settings["glszm_spatial_method"], "2.5d")
        self.assertEqual(feature_settings["ngldm_spatial_method"], "2.5d")

    def test_public_native_preload_keeps_paths_outside_feature_repeats(self) -> None:
        modules, mirp, events = fake_mirp_settings_modules()
        native_image = object()
        native_mask = object()

        def extract_images(**kwargs):
            events.append(("extract_images", kwargs))
            return [([native_image], [native_mask])]

        def extract_features(**kwargs):
            events.append(("extract_features", kwargs))
            return [
                FakeTable(
                    {
                        "sample_name": "case",
                        "image_voxel_size_x": 1.0,
                        "stat_mean": 3.0,
                    }
                )
            ]

        mirp.extract_images = extract_images
        mirp.extract_features = extract_features
        with modules:
            image, mask = mirp_adapter._load_native_pair(
                mirp,
                image_path="image.nii.gz",
                mask_path="mask.nii.gz",
                modality="ct",
            )
            settings = mirp_adapter._build_settings(
                families=["statistics"],
                discretization="fbs",
                bins=32,
                bin_width=25.0,
                intensity_range=(-1000.0, 400.0),
            )
            values = mirp_adapter._extract_public(
                mirp,
                image=image,
                mask=mask,
                settings=settings,
                families=["statistics"],
            )

        self.assertEqual(values, {"stat_mean": 3.0})
        load_kwargs = next(value for name, value in events if name == "extract_images")
        self.assertEqual(load_kwargs["image_export_format"], "native")
        extraction = next(value for name, value in events if name == "extract_features")
        self.assertIs(extraction["image"], native_image)
        self.assertIs(extraction["mask"], native_mask)
        self.assertNotIsInstance(extraction["image"], str)
        resegmentation = next(
            value for name, value in events if name == "resegmentation_settings"
        )
        self.assertEqual(
            resegmentation["resegmentation_intensity_range"], [-1000.0, 400.0]
        )
        feature_settings = next(
            value for name, value in events if name == "feature_settings"
        )
        self.assertEqual(
            feature_settings["base_discretisation_method"], "fixed_bin_size"
        )
        self.assertEqual(feature_settings["base_discretisation_bin_width"], 25.0)

    def test_public_extraction_rejects_zero_features(self) -> None:
        mirp = SimpleNamespace(
            extract_features=lambda **kwargs: [
                FakeTable({"sample_name": "case", "image_voxel_size_x": 1.0})
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "zero finite"):
            mirp_adapter._extract_public(
                mirp,
                image=object(),
                mask=object(),
                settings=object(),
                families=["statistics"],
            )

    def test_feature_table_rejects_invalid_calculated_values(self) -> None:
        invalid_values = {
            "text": "3.0",
            "none": None,
            "positive infinity": float("inf"),
            "negative infinity": float("-inf"),
            "nan": float("nan"),
            "boolean": True,
        }
        for label, raw in invalid_values.items():
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(RuntimeError, "nonnumeric|non-finite"),
            ):
                mirp_adapter._table_to_values(
                    FakeTable(
                        {
                            "sample_name": "case",
                            "image_arbitrary_provenance": object(),
                            "stat_mean": raw,
                        }
                    ),
                    families=["statistics"],
                )

    def test_feature_table_ignores_only_known_provenance_columns(self) -> None:
        values = mirp_adapter._table_to_values(
            FakeTable(
                {
                    "sample_name": "case",
                    "image_arbitrary_provenance": object(),
                    "stat_mean": np.float32(3.0),
                }
            ),
            families=["statistics"],
        )
        self.assertEqual(values, {"stat_mean": 3.0})

    def test_feature_table_rejects_empty_or_string_colliding_columns(self) -> None:
        for table in (
            FakeTable({"   ": 1.0}),
            FakeTable({1: 1.0, "1": 2.0}),
        ):
            with (
                self.subTest(columns=table.columns),
                self.assertRaisesRegex(RuntimeError, "empty or duplicate"),
            ):
                mirp_adapter._table_to_values(table, families=["statistics"])


class MedimageRangeTests(unittest.TestCase):
    def test_timed_run_repeats_preprocessing_without_warmup(
        self,
    ) -> None:
        image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        mask = np.ones_like(image, dtype=np.uint8)
        stats = SimpleNamespace(extract_all=mock.Mock(return_value={"Fstat_mean": 3.5}))
        image_digest = "a" * 64
        mask_digest = "b" * 64

        with (
            mock.patch.object(
                medimage_adapter,
                "_load_nifti",
                side_effect=[
                    (image, (1.0, 1.0, 1.0)),
                    (mask, (1.0, 1.0, 1.0)),
                ],
            ),
            mock.patch.object(
                medimage_adapter,
                "_medimage_modules",
                return_value={"stats": stats},
            ),
            mock.patch.object(
                medimage_adapter,
                "_prepare_roi_volumes",
                wraps=medimage_adapter._prepare_roi_volumes,
            ) as prepare,
            mock.patch.object(medimage_adapter, "write_json") as write_json,
        ):
            result = medimage_adapter.main(
                [
                    "--image",
                    "image.nii.gz",
                    "--mask",
                    "mask.nii.gz",
                    "--image-sha256",
                    image_digest,
                    "--mask-sha256",
                    mask_digest,
                    "--modality",
                    "CT",
                    "--families",
                    "intensity",
                    "--timed",
                    "--iterations",
                    "3",
                    "--no-warmup",
                ]
            )

        self.assertEqual(result, 0)
        # ROI construction is calculation input preparation and occurs once,
        # outside all selected-family timing calls.
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(stats.extract_all.call_count, 3)
        for call in prepare.call_args_list:
            self.assertEqual(call.args[1].dtype, np.dtype(bool))
            self.assertIs(call.kwargs["apply_discretization"], False)

        payload = write_json.call_args.args[0]
        self.assertEqual(payload["timing"]["measured_iterations"], 3)
        self.assertEqual(payload["timing"]["warmup_iterations"], 0)
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
        self.assertFalse(
            payload["metadata"]["timing_contract"]["includes_required_preprocessing"]
        )

    def test_continuous_families_resegment_without_discretizing(self) -> None:
        image = np.array([[[-20.0, -5.5], [5.5, 20.0]]], dtype=np.float32)
        mask = np.ones_like(image, dtype=bool)
        module_keys = {
            "morphology": "morph",
            "local_intensity": "local_intensity",
            "intensity": "stats",
        }

        for family, module_key in module_keys.items():
            module = SimpleNamespace(
                extract_all=mock.Mock(return_value={f"{family}_value": 1.0})
            )
            with self.subTest(family=family):
                values = _prepare_and_compute_medimage(
                    families=[family],
                    modules={module_key: module},
                    image_full=image,
                    binary_mask=mask,
                    spacing=(1.0, 1.0, 1.0),
                    intensity_type="definite",
                    # These noninteger values would be rejected if identity
                    # discretization were incorrectly run.
                    discretization="identity",
                    bins=0,
                    bin_width=float("nan"),
                    intensity_range=(-10.0, 10.0),
                )

            self.assertEqual(values, {f"{family}_value": 1.0})
            call_kwargs = module.extract_all.call_args.kwargs
            if family == "local_intensity":
                self.assertEqual(int(np.sum(call_kwargs["roi_obj"])), 2)
            elif family == "morphology":
                self.assertEqual(int(np.sum(call_kwargs["mask_int"])), 2)
                self.assertEqual(int(np.sum(~np.isnan(call_kwargs["vol"]))), 2)
            else:
                self.assertEqual(int(np.sum(~np.isnan(call_kwargs["vol"]))), 2)

    def test_discretized_families_still_quantize(self) -> None:
        image = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
        mask = np.ones_like(image, dtype=bool)
        module_keys = {
            "histogram": "intensity_histogram",
            "ivh": "int_vol_hist",
            "glcm": "glcm",
            "glrlm": "glrlm",
            "glszm": "glszm",
            "gldzm": "gldzm",
            "ngtdm": "ngtdm",
            "ngldm": "ngldm",
        }
        for family, module_key in module_keys.items():
            module = SimpleNamespace(
                extract_all=mock.Mock(return_value={f"{family}_value": 4.5})
            )
            with self.subTest(family=family):
                values = _prepare_and_compute_medimage(
                    families=[family],
                    modules={module_key: module},
                    image_full=image,
                    binary_mask=mask,
                    spacing=(1.0, 1.0, 1.0),
                    intensity_type="definite",
                    discretization="fbn",
                    bins=4,
                    bin_width=float("nan"),
                    intensity_range=None,
                )

            self.assertEqual(values, {f"{family}_value": 4.5})
            extraction_kwargs = module.extract_all.call_args.kwargs
            quantized = extraction_kwargs.get(
                "vol",
                extraction_kwargs.get("vol_int"),
            )
            np.testing.assert_array_equal(
                np.unique(quantized),
                [1.0, 2.0, 3.0, 4.0],
            )

    def test_feature_normalization_accepts_only_finite_singletons(self) -> None:
        accepted = {
            "python_scalar": 1.5,
            "numpy_scalar": np.float32(2.5),
            "singleton_list": [3.5],
            "singleton_array": np.array([[4.5]], dtype=np.float64),
        }
        self.assertEqual(
            medimage_adapter._normalize_features(accepted),
            {
                "python_scalar": 1.5,
                "numpy_scalar": 2.5,
                "singleton_list": 3.5,
                "singleton_array": 4.5,
            },
        )

        invalid_values = {
            "empty list": [],
            "empty array": np.array([], dtype=float),
            "multi-element list": [1.0, 2.0],
            "multi-element array": np.array([1.0, 2.0]),
            "text": "1.0",
            "none": None,
            "positive infinity": float("inf"),
            "negative infinity": float("-inf"),
            "nan": float("nan"),
            "boolean": True,
        }
        for label, raw in invalid_values.items():
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, "Invalid MEDimage feature"),
            ):
                medimage_adapter._normalize_features({"bad": raw})

    def test_feature_normalization_rejects_empty_or_string_colliding_names(
        self,
    ) -> None:
        for values in ({"   ": 1.0}, {1: 1.0, "1": 2.0}):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, "empty or duplicate"),
            ):
                medimage_adapter._normalize_features(values)

    def test_identity_discretization_preserves_the_integer_grey_level_grid(
        self,
    ) -> None:
        image = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
        mask = np.ones_like(image, dtype=np.uint8)
        vol_raw, vol_quant, vol_ivh, width, _ = medimage_adapter._prepare_roi_volumes(
            image,
            mask,
            discretization="identity",
            bins=0,
            bin_width=float("nan"),
            intensity_range=None,
        )
        self.assertEqual(width, 1.0)
        np.testing.assert_array_equal(vol_quant, vol_raw)
        np.testing.assert_array_equal(vol_ivh, vol_raw)

    def test_fbs_resegments_and_produces_physical_ivh_bin_centres(self) -> None:
        image = np.array([[[-20.0, -5.0], [5.0, 20.0]]], dtype=np.float32)
        mask = np.ones_like(image, dtype=np.uint8)
        vol_raw, vol_quant, vol_ivh, wd, intensity_mask = (
            medimage_adapter._prepare_roi_volumes(
                image,
                mask,
                discretization="fbs",
                bins=32,
                bin_width=10.0,
                intensity_range=(-10.0, 10.0),
            )
        )

        self.assertEqual(wd, 10.0)
        self.assertEqual(int(np.sum(intensity_mask)), 2)
        np.testing.assert_array_equal(vol_quant[~np.isnan(vol_quant)], [1.0, 2.0])
        np.testing.assert_array_equal(vol_ivh[~np.isnan(vol_ivh)], [-5.0, 5.0])
        self.assertEqual(int(np.sum(~np.isnan(vol_raw))), 2)

    def test_fbn_resegments_but_retains_unit_ivh_steps(self) -> None:
        image = np.array([[[-20.0, -5.0], [5.0, 20.0]]], dtype=np.float32)
        mask = np.ones_like(image, dtype=np.uint8)
        _, vol_quant, vol_ivh, wd, intensity_mask = (
            medimage_adapter._prepare_roi_volumes(
                image,
                mask,
                discretization="fbn",
                bins=4,
                bin_width=10.0,
                intensity_range=(-10.0, 10.0),
            )
        )
        self.assertEqual(wd, 1.0)
        self.assertEqual(int(np.sum(intensity_mask)), 2)
        np.testing.assert_array_equal(vol_quant, vol_ivh)

    def test_family_zero_result_is_not_reported_as_success(self) -> None:
        module = SimpleNamespace(extract_all=lambda **kwargs: {})
        array = np.ones((1, 1, 1), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "returned zero finite features"):
            medimage_adapter._compute_features(
                families=["intensity"],
                modules={"stats": module},
                image_full=array,
                vol_raw=array,
                vol_quant=array,
                vol_ivh=array,
                morphology_mask_bool=array.astype(bool),
                intensity_mask_bool=array.astype(bool),
                spacing=(1.0, 1.0, 1.0),
                intensity_type="definite",
                wd=1.0,
                discretization="fbn",
                intensity_range=None,
            )

    def test_cross_family_feature_names_cannot_be_overwritten(self) -> None:
        array = np.ones((1, 1, 1), dtype=np.float32)
        with (
            mock.patch.object(
                medimage_adapter,
                "_extract_family",
                return_value={"shared": 1.0},
            ),
            self.assertRaisesRegex(RuntimeError, "across families"),
        ):
            medimage_adapter._compute_features(
                families=["intensity", "histogram"],
                modules={},
                image_full=array,
                vol_raw=array,
                vol_quant=array,
                vol_ivh=array,
                morphology_mask_bool=array.astype(bool),
                intensity_mask_bool=array.astype(bool),
                spacing=(1.0, 1.0, 1.0),
                intensity_type="definite",
                wd=1.0,
                discretization="fbn",
                intensity_range=None,
            )

    def test_local_intensity_receives_full_image_and_separate_roi(self) -> None:
        image = np.array([[[7.0, 2.0]]], dtype=np.float32)
        roi = np.array([[[0, 1]]], dtype=bool)
        vol_raw = np.where(roi, image, np.nan)
        captured = {}

        def extract_all(**kwargs):
            captured.update(kwargs)
            return {"Floc_peak_local": 2.0}

        values = medimage_adapter._extract_family(
            family="local_intensity",
            modules={"local_intensity": SimpleNamespace(extract_all=extract_all)},
            image_full=image,
            vol_raw=vol_raw,
            vol_quant=vol_raw,
            vol_ivh=vol_raw,
            morphology_mask_bool=roi,
            intensity_mask_bool=roi,
            spacing=(1.0, 1.0, 1.0),
            intensity_type="definite",
            wd=1.0,
            discretization="identity",
            intensity_range=None,
        )

        self.assertEqual(values, {"Floc_peak_local": 2.0})
        np.testing.assert_array_equal(captured["img_obj"], image)
        np.testing.assert_array_equal(captured["roi_obj"], roi.astype(np.uint8))

    def test_ngtdm_explicitly_disables_distance_correction(self) -> None:
        array = np.ones((1, 1, 1), dtype=np.float32)
        captured = {}

        def extract_all(**kwargs):
            captured.update(kwargs)
            return {"Fngt_coarseness": 1.0}

        values = medimage_adapter._extract_family(
            family="ngtdm",
            modules={"ngtdm": SimpleNamespace(extract_all=extract_all)},
            image_full=array,
            vol_raw=array,
            vol_quant=array,
            vol_ivh=array,
            morphology_mask_bool=array.astype(bool),
            intensity_mask_bool=array.astype(bool),
            spacing=(1.0, 1.0, 1.0),
            intensity_type="definite",
            wd=1.0,
            discretization="identity",
            intensity_range=None,
        )

        self.assertEqual(values, {"Fngt_coarseness": 1.0})
        self.assertIs(captured["dist_correction"], False)


if __name__ == "__main__":
    unittest.main()

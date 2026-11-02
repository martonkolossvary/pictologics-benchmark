from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from bench.adapters import zrad_adapter
from bench.adapters.registry import get_adapter
from bench.ibsi_mapping import map_pyradiomics


def fake_zrad_modules(result_factory=None):
    """Return mocked Z-Rad 26.6 modules and a shared call ledger."""

    events = []
    zrad = types.ModuleType("zrad")
    zrad.__path__ = []
    image_module = types.ModuleType("zrad.image")
    preprocessing_module = types.ModuleType("zrad.preprocessing")
    radiomics_module = types.ModuleType("zrad.radiomics")
    radiomics_module.__path__ = []
    intensity_module = types.ModuleType("zrad.radiomics.intensity")
    intensity_module._LOCAL_MEANS_CACHE = {}

    class Image:
        def __init__(self, label):
            self.label = label
            self.array = np.ones((4, 4, 4), dtype=np.uint8)
            self.spacing = (1.0, 1.0, 1.0)

        @classmethod
        def from_nifti(cls, image_path):
            events.append(("from_nifti", image_path))
            return cls("image")

        @classmethod
        def from_nifti_mask(cls, mask_path, reference):
            events.append(("from_nifti_mask", mask_path, reference))
            return cls("mask")

    class RoiData:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.intensity_mask = kwargs.get("intensity_mask")
            self.intensity_range = kwargs.get("intensity_range")
            self.texture_discretized_image = kwargs.get("texture_discretized_image")
            self.ivh_intensity_image = kwargs.get("ivh_intensity_image")
            events.append(("roi_data", kwargs))

    class IntensityMaskBuilder:
        def apply(self, roi_data):
            events.append(("intensity_mask",))
            roi_data.intensity_mask = object()
            return roi_data

    class Resegmenter:
        def __init__(self, **kwargs):
            events.append(("resegment_init", kwargs))
            self.intensity_range = kwargs["intensity_range"]

        def apply(self, roi_data):
            events.append(("resegment_apply", self.intensity_range))
            roi_data.intensity_range = self.intensity_range
            return roi_data

    class TextureDiscretizer:
        def __init__(self, **kwargs):
            events.append(("texture_init", kwargs))

        def apply(self, roi_data):
            events.append(("texture_apply",))
            roi_data.texture_discretized_image = object()
            return roi_data

    class IVHIntensityDiscretizer:
        def __init__(self, method, **kwargs):
            events.append(("ivh_init", method, kwargs))

        def apply(self, roi_data):
            events.append(("ivh_apply",))
            roi_data.ivh_intensity_image = object()
            return roi_data

    class RoiCropper:
        def __init__(self, padding):
            self.padding = padding

        def apply(self, roi_data):
            events.append(("roi_crop", self.padding))
            return roi_data

    class Radiomics:
        def __init__(self, **kwargs):
            events.append(("radiomics_init", kwargs))

        def extract_features(self, **kwargs):
            events.append(("extract_features", kwargs))
            if "local_intensity" in kwargs["families"]:
                if not intensity_module._LOCAL_MEANS_CACHE:
                    events.append(("local_intensity_convolution",))
                    intensity_module._LOCAL_MEANS_CACHE["image"] = object()
            if result_factory is None:
                return {family + "_feature": 1.0 for family in kwargs["families"]}
            if callable(result_factory):
                return result_factory(kwargs)
            return result_factory

    image_module.Image = Image
    preprocessing_module.RoiData = RoiData
    preprocessing_module.IntensityMaskBuilder = IntensityMaskBuilder
    preprocessing_module.Resegmenter = Resegmenter
    preprocessing_module.RoiCropper = RoiCropper
    preprocessing_module.TextureDiscretizer = TextureDiscretizer
    preprocessing_module.IVHIntensityDiscretizer = IVHIntensityDiscretizer
    radiomics_module.Radiomics = Radiomics

    modules = {
        "zrad": zrad,
        "zrad.image": image_module,
        "zrad.preprocessing": preprocessing_module,
        "zrad.radiomics": radiomics_module,
        "zrad.radiomics.intensity": intensity_module,
    }
    return mock.patch.dict(sys.modules, modules), events, Image


class ZRadAdapterTests(unittest.TestCase):
    def test_registry_declares_native_selection(self) -> None:
        capabilities = get_adapter("zrad")
        self.assertEqual(capabilities.selection_mode, "native")

    def test_public_nifti_loaders_preserve_reference_alignment(self) -> None:
        modules, events, _ = fake_zrad_modules()
        with modules:
            image, mask = zrad_adapter._load_nifti_pair(
                Path("image.nii.gz"), Path("mask.nii.gz")
            )

        self.assertEqual(events[0], ("from_nifti", "image.nii.gz"))
        self.assertEqual(events[1], ("from_nifti_mask", "mask.nii.gz", image))
        self.assertEqual(mask.label, "mask")

    def test_fbn_preprocessing_uses_both_discretizers_and_resegmenter(self) -> None:
        modules, events, Image = fake_zrad_modules()
        with modules:
            roi_data = zrad_adapter._prepare_roi_data(
                image=Image("image"),
                mask=Image("mask"),
                families=["histogram", "ivh"],
                discretization="fbn",
                bins=64,
                bin_width=2.5,
                intensity_range=(-1000.0, 400.0),
            )

        self.assertIn(("resegment_init", {"intensity_range": (-1000.0, 400.0)}), events)
        self.assertIn(("texture_init", {"number_of_bins": 64}), events)
        self.assertIn(("ivh_init", "fixed_bin_number", {"number_of_bins": 64}), events)
        self.assertIsNotNone(roi_data.texture_discretized_image)
        self.assertIsNotNone(roi_data.ivh_intensity_image)

    def test_fbs_preprocessing_uses_bin_width_and_requires_anchor(self) -> None:
        modules, events, Image = fake_zrad_modules()
        with modules:
            zrad_adapter._prepare_roi_data(
                image=Image("image"),
                mask=Image("mask"),
                families=["glcm", "ivh"],
                discretization="fbs",
                bins=32,
                bin_width=25.0,
                intensity_range=(-1024.0, 3096.0),
            )

        self.assertIn(("texture_init", {"bin_size": 25.0}), events)
        self.assertIn(("ivh_init", "fixed_bin_size", {"bin_size": 25.0}), events)

        modules, _, Image = fake_zrad_modules()
        with modules, self.assertRaisesRegex(ValueError, "explicit"):
            zrad_adapter._prepare_roi_data(
                image=Image("image"),
                mask=Image("mask"),
                families=["glcm"],
                discretization="fbs",
                bins=32,
                bin_width=25.0,
                intensity_range=None,
            )

    def test_identity_preprocessing_uses_validated_unit_width_equivalence(self) -> None:
        modules, events, _ = fake_zrad_modules()
        image = types.SimpleNamespace(
            array=np.arange(1, 9, dtype=float).reshape(2, 2, 2)
        )
        mask = types.SimpleNamespace(
            array=np.ones((2, 2, 2), dtype=np.uint8), spacing=(1.0, 1.0, 1.0)
        )
        with modules:
            roi_data = zrad_adapter._prepare_roi_data(
                image=image,
                mask=mask,
                families=["glcm", "ivh"],
                discretization="identity",
                bins=0,
                bin_width=float("nan"),
                intensity_range=None,
            )
        self.assertIn(("texture_init", {"bin_size": 1.0}), events)
        self.assertIn(("ivh_init", "direct", {}), events)
        # The crop runs first, so the discretisers see only the ROI box.
        self.assertEqual(events[1], ("roi_crop", [1, 1, 1]))
        self.assertLess(
            events.index(("roi_crop", [1, 1, 1])), events.index(("ivh_apply",))
        )
        self.assertEqual(roi_data.intensity_range, (1.0, 8.0))

    def test_morphology_selects_native_correlation_without_post_filtering(self) -> None:
        modules, events, _ = fake_zrad_modules(
            {"morph_volume": 2.0, "morph_moran_i": 0.25}
        )
        with modules:
            names, values = zrad_adapter._compute_zrad_features(
                roi_data=object(),
                families=["morphology"],
                aggr_dim="3D",
                aggr_method="AVER",
            )

        extraction = [event for event in events if event[0] == "extract_features"]
        self.assertEqual(len(extraction), 1)
        self.assertEqual(
            extraction[0][1]["families"],
            ["morphology", "morphology_correlation"],
        )
        self.assertFalse(extraction[0][1]["include_metadata"])
        self.assertEqual(names, ["morph_volume", "morph_moran_i"])
        self.assertEqual(values["morph_moran_i"], 0.25)

    def test_main_emits_native_selection_and_preprocessing_metadata(self) -> None:
        modules, _, _ = fake_zrad_modules({"morph_volume": 2.0, "morph_moran_i": 0.25})
        with modules, mock.patch.object(zrad_adapter, "write_json") as write_json:
            return_code = zrad_adapter.main(
                [
                    "--image",
                    "image.nii.gz",
                    "--mask",
                    "mask.nii.gz",
                    "--families",
                    "morphology",
                    "--include-values",
                ]
            )

        self.assertEqual(return_code, 0)
        payload = write_json.call_args.args[0]
        self.assertEqual(payload["selection"]["mode"], "native")
        self.assertEqual(payload["selection"]["requested_families"], ["morphology"])
        self.assertEqual(
            payload["metadata"]["native_families"],
            ["morphology", "morphology_correlation"],
        )
        self.assertEqual(
            payload["metadata"]["timing_execution_scope"], "native_selected_families"
        )
        self.assertEqual(
            payload["metadata"]["local_intensity_cache_policy"],
            "not_applicable",
        )
        self.assertEqual(payload["values"]["all"]["morph_moran_i"], 0.25)

    def test_timed_main_repeats_preprocessing_and_honors_no_warmup(self) -> None:
        modules, events, _ = fake_zrad_modules({"glcm_feature": 1.0})
        image_digest = "a" * 64
        mask_digest = "b" * 64
        with modules, mock.patch.object(zrad_adapter, "write_json") as write_json:
            return_code = zrad_adapter.main(
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
                    "glcm",
                    "--timed",
                    "--iterations",
                    "3",
                    "--no-warmup",
                ]
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(sum(event[0] == "from_nifti" for event in events), 1)
        self.assertEqual(sum(event[0] == "from_nifti_mask" for event in events), 1)
        # The selected-family RoiData is prepared once outside the calculation
        # timer.  The preparation builds one RoiData for the crop and one after it.
        self.assertEqual(sum(event[0] == "roi_data" for event in events), 2)
        self.assertEqual(sum(event[0] == "intensity_mask" for event in events), 1)
        self.assertEqual(sum(event[0] == "texture_apply" for event in events), 1)
        self.assertEqual(sum(event[0] == "extract_features" for event in events), 3)
        self.assertEqual(sum(event[0] == "radiomics_init" for event in events), 1)

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
        self.assertEqual(
            payload["metadata"]["timing_contract"]["scope"],
            "prepared_family_inputs_to_radiomic_calculations",
        )
        self.assertFalse(
            payload["metadata"]["timing_contract"]["includes_required_preprocessing"]
        )

    def test_local_intensity_cache_is_cold_for_every_measured_iteration(self) -> None:
        modules, events, _ = fake_zrad_modules({"loc_peak_glob": 1.0})
        with modules, mock.patch.object(zrad_adapter, "write_json") as write_json:
            self.assertEqual(
                zrad_adapter.main(
                    [
                        "--image",
                        "image.nii.gz",
                        "--mask",
                        "mask.nii.gz",
                        "--families",
                        "local_intensity",
                        "--timed",
                        "--iterations",
                        "3",
                        "--no-warmup",
                    ]
                ),
                0,
            )
            cache = sys.modules["zrad.radiomics.intensity"]._LOCAL_MEANS_CACHE
            self.assertEqual(cache, {})
            self.assertEqual(
                write_json.call_args.args[0]["metadata"][
                    "local_intensity_cache_policy"
                ],
                "cleared_before_and_after_each_calculation",
            )

        self.assertEqual(
            sum(event[0] == "local_intensity_convolution" for event in events),
            3,
        )

    def test_main_maps_ibsi_two_and_a_half_d_profile_to_native_controls(self) -> None:
        modules, events, _ = fake_zrad_modules()
        with modules, mock.patch.object(zrad_adapter, "write_json") as write_json:
            return_code = zrad_adapter.main(
                [
                    "--image",
                    "image.nii.gz",
                    "--mask",
                    "mask.nii.gz",
                    "--families",
                    "glcm",
                    "--aggregation",
                    "2.5d_direction_merge",
                    "--include-values",
                ]
            )

        self.assertEqual(return_code, 0)
        native_settings = next(
            event[1] for event in events if event[0] == "radiomics_init"
        )
        self.assertEqual(native_settings["aggr_dim"], "2.5D")
        self.assertEqual(native_settings["aggr_method"], "AVER")
        metadata = write_json.call_args.args[0]["metadata"]["aggregation"]
        self.assertEqual(metadata["effective_directional"], "2.5d_direction_merge")
        self.assertEqual(metadata["native_dimension"], "2.5D")

    def test_empty_native_result_is_a_structured_failure(self) -> None:
        modules, _, _ = fake_zrad_modules({})
        with modules, self.assertRaises(zrad_adapter.ZRadExtractionError) as caught:
            zrad_adapter._compute_zrad_features(
                roi_data=object(),
                families=["ngtdm"],
                aggr_dim="3D",
                aggr_method="MERG",
            )

        self.assertEqual(caught.exception.phase, "validate_result")
        self.assertEqual(caught.exception.families, ("ngtdm",))
        self.assertIn("zero features", str(caught.exception))

    def test_invalid_native_scalar_is_a_structured_failure(self) -> None:
        for raw_value in (
            np.nan,
            [1.0, 2.0],
            "1.0",
            "not-a-number",
            True,
            np.bool_(True),
        ):
            modules, _, _ = fake_zrad_modules({"ngtdm_feature": raw_value})
            with (
                modules,
                self.subTest(raw_value=raw_value),
                self.assertRaises(zrad_adapter.ZRadExtractionError) as caught,
            ):
                zrad_adapter._compute_zrad_features(
                    roi_data=object(),
                    families=["ngtdm"],
                    aggr_dim="3D",
                    aggr_method="MERG",
                )
            self.assertEqual(caught.exception.phase, "validate_result")
            self.assertIn("non-finite or non-scalar", str(caught.exception))

    def test_native_feature_names_are_nonempty_and_unique_after_normalization(
        self,
    ) -> None:
        for raw in ({"   ": 1.0}, {1: 1.0, "1": 2.0}):
            modules, _, _ = fake_zrad_modules(raw)
            with (
                modules,
                self.subTest(raw=raw),
                self.assertRaises(zrad_adapter.ZRadExtractionError) as caught,
            ):
                zrad_adapter._compute_zrad_features(
                    roi_data=object(),
                    families=["ngtdm"],
                    aggr_dim="3D",
                    aggr_method="MERG",
                )
            self.assertEqual(caught.exception.phase, "validate_result")
            self.assertIn("empty or duplicate", str(caught.exception))

    def test_include_code_filter_uses_existing_ibsi_mapping(self) -> None:
        names, values = zrad_adapter._filter_by_ibsi_codes(
            ["stat_mean", "stat_var"],
            {"stat_mean": 3.0, "stat_var": 2.0},
            ["Q4LE"],
        )
        self.assertEqual(names, ["stat_mean"])
        self.assertEqual(values, {"stat_mean": 3.0})

    def test_pyradiomics_non_emitting_dependence_percentage_is_not_mapped(self) -> None:
        self.assertIsNone(map_pyradiomics("original_gldm_DependencePercentage"))


if __name__ == "__main__":
    unittest.main()

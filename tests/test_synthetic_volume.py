from __future__ import annotations

import numpy as np
from scipy import ndimage

from bench.synthetic_volume import HU_PROFILES, VolumeConfig, _axis_centres
from bench.synthetic_volume import (
    COARSEST_EDGE,
    COARSE_GUARD_VOXELS,
    DENSE_BODY_RADII_MM,
    LARGE_SPICULE_SEGMENTS,
    _body_and_soft_support,
    _voxel_noise,
    evaluate_synthetic_volume,
    generate_synthetic_volume,
    measure_satellite_component_voxels,
    measure_spicule_external_voxels,
    measure_volume_scale,
)


def test_synthetic_volume_is_deterministic_and_true_3d() -> None:
    config = VolumeConfig(edge=64, fov_mm=256.0, slab_depth=7)
    forward = generate_synthetic_volume(config)
    reverse = generate_synthetic_volume(config, slab_order="reverse")
    np.testing.assert_array_equal(forward.image, reverse.image)
    np.testing.assert_array_equal(forward.labels, reverse.labels)
    for mask_id in forward.masks:
        np.testing.assert_array_equal(forward.masks[mask_id], reverse.masks[mask_id])
    metrics = evaluate_synthetic_volume(forward)
    assert metrics.identical_adjacent_image_slices == 0
    assert metrics.distinct_image_slices == 64
    # Four empty guard slices share one all-background label plane; every
    # anatomy-bearing slice remains distinct.
    assert metrics.distinct_label_slices == 61
    assert metrics.masks["A1"].occupied_z_slices == 60
    assert metrics.masks["A1"].minimum_face_margin_voxels >= 1


def test_voxel_noise_keeps_geometry_and_slab_order_invariance() -> None:
    plain = generate_synthetic_volume(VolumeConfig(edge=48, slab_depth=5))
    config = VolumeConfig(edge=48, slab_depth=5, hu_profile="reference_noise")
    forward = generate_synthetic_volume(config)
    reverse = generate_synthetic_volume(config, slab_order="reverse")
    np.testing.assert_array_equal(forward.image, reverse.image)
    for mask_id in plain.masks:
        np.testing.assert_array_equal(forward.masks[mask_id], plain.masks[mask_id])
    difference = forward.image.astype(float) - plain.image.astype(float)
    inside = plain.masks["A1"] > 0
    assert 15.0 < float(np.std(difference[inside])) < 25.0
    assert forward.metadata["voxel_noise_hu"] == 20.0
    # Profiles without noise keys have no voxel noise in their bytes or metadata.
    assert _voxel_noise(config, HU_PROFILES["reference"]) is None
    assert "voxel_noise_hu" not in plain.metadata


def test_large_masks_preserve_bone_and_expected_relations() -> None:
    bundle = generate_synthetic_volume(
        VolumeConfig(edge=128, fov_mm=256.0, slab_depth=11)
    )
    metrics = evaluate_synthetic_volume(bundle)
    assert len(LARGE_SPICULE_SEGMENTS) == 18
    assert np.all(bundle.masks["M2"] <= bundle.masks["M1"])
    assert np.all(bundle.masks["M1"] <= bundle.masks["M3"])
    assert np.all(bundle.masks["M1"] <= bundle.masks["M4"])
    for mask_id in ("M1", "M2", "M3", "M4"):
        assert metrics.masks[mask_id].bone_overlap_voxels == 0
    assert 0.095 <= metrics.masks["M1"].image_fraction <= 0.098
    assert 0.16 <= metrics.masks["M1"].body_fraction <= 0.18
    assert 0.86 <= metrics.masks["M1"].occupied_z_slices / 128 <= 0.88
    assert metrics.masks["M1"].occupied_z_mm == 222.0
    assert metrics.masks["M1"].within_3p5mm_of_bone_voxels > 0
    assert metrics.masks["M1"].components_26 == 1
    assert metrics.masks["M3"].components_26 == 1
    satellite_components = ndimage.label(
        bundle.labels == 10, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )[1]
    assert satellite_components == 6
    assert metrics.masks["M2"].cavities_background_6 == 1
    assert metrics.masks["M4"].components_26 == 7
    assert 0.56 <= metrics.masks["A1"].image_fraction <= 0.58
    assert metrics.masks["A1"].occupied_z_slices == 120
    assert metrics.masks["A1"].face_margins_voxels == (6, 2, 4, 2, 4, 4)
    assert np.array_equal(bundle.masks["A1"] != 0, bundle.labels != 0)


def test_streaming_scale_records_full_xyz_and_workload() -> None:
    small = measure_volume_scale(32)
    reference = measure_volume_scale(128)
    assert small["image_voxels"] == 32**3
    assert reference["image_voxels"] == 128**3
    assert reference["a1_z_slices"] == 120
    assert 0.095 <= float(reference["m1_image_fraction"]) <= 0.098
    assert 0.56 <= float(reference["a1_image_fraction"]) <= 0.58
    assert int(reference["m1_voxels"]) > 200_000
    assert tuple(small["a1_face_margins_voxels"]) == (1, 1, 1, 1, 1, 1)
    assert min(tuple(reference["a1_face_margins_voxels"])) >= 1


def test_coarse_grid_face_guard_is_contractual() -> None:
    assert COARSEST_EDGE == 32
    assert COARSE_GUARD_VOXELS == 1
    bundle = generate_synthetic_volume(VolumeConfig(edge=COARSEST_EDGE))
    a1 = bundle.masks["A1"] != 0
    assert not np.any(a1[0])
    assert not np.any(a1[-1])
    assert not np.any(a1[:, 0])
    assert not np.any(a1[:, -1])
    assert not np.any(a1[:, :, 0])
    assert not np.any(a1[:, :, -1])


def test_dense_body_is_maximal_on_declared_quarter_mm_search_grid() -> None:
    config = VolumeConfig(edge=COARSEST_EDGE)
    axis = _axis_centres(config)
    x = axis[:, None, None]
    y = axis[None, :, None]
    z = axis[None, None, :]
    baseline, _soft, _bone = _body_and_soft_support(x, y, z)
    assert not any(
        (
            np.any(baseline[0]),
            np.any(baseline[-1]),
            np.any(baseline[:, 0]),
            np.any(baseline[:, -1]),
            np.any(baseline[:, :, 0]),
            np.any(baseline[:, :, -1]),
        )
    )
    for candidate in (
        (DENSE_BODY_RADII_MM[0] + 0.25, DENSE_BODY_RADII_MM[1]),
        (DENSE_BODY_RADII_MM[0], DENSE_BODY_RADII_MM[1] + 0.25),
    ):
        expanded, _soft, _bone = _body_and_soft_support(
            x, y, z, body_radii_mm=candidate
        )
        assert any(
            (
                np.any(expanded[0]),
                np.any(expanded[-1]),
                np.any(expanded[:, 0]),
                np.any(expanded[:, -1]),
                np.any(expanded[:, :, 0]),
                np.any(expanded[:, :, -1]),
            )
        )


def test_all_tumour_topologies_are_stable_from_coarsest_grid() -> None:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    for edge in (32, 48, 64, 96, 128):
        bundle = generate_synthetic_volume(VolumeConfig(edge=edge, slab_depth=7))
        metrics = evaluate_synthetic_volume(bundle)
        assert metrics.masks["M1"].components_26 == 1
        assert metrics.masks["M2"].components_26 == 1
        assert metrics.masks["M2"].cavities_background_6 == 1
        assert metrics.masks["M3"].components_26 == 1
        assert metrics.masks["M3"].cavities_background_6 == 0
        assert metrics.masks["M4"].components_26 == 7
        satellites = ndimage.label(bundle.labels == 10, structure=structure)[1]
        assert satellites == 6
        spicule_counts = measure_spicule_external_voxels(bundle)
        assert len(spicule_counts) == 18
        assert min(spicule_counts) > 0
        satellite_counts = measure_satellite_component_voxels(bundle)
        assert len(satellite_counts) == 6
        assert min(satellite_counts) > 0
        for mask_id in ("M1", "M2", "M3", "M4"):
            assert metrics.masks[mask_id].bone_overlap_voxels == 0

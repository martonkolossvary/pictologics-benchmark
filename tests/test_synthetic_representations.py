from __future__ import annotations

import numpy as np
import pytest

from bench.synthetic_volume import VolumeConfig, generate_synthetic_volume
from bench.synthetic_representations import (
    RepresentationError,
    compile_mask_specific_fbn,
    compile_mask_specific_fbs,
    texture_workload,
)


def test_ibsi_fbn_hand_oracle_and_maximum_rule() -> None:
    image = np.asarray([0.0, 2.0, 4.0, 7.0, 10.0]).reshape(5, 1, 1)
    mask = np.asarray([0, 1, 1, 1, 1]).reshape(5, 1, 1)
    result = compile_mask_specific_fbn(image, mask, levels=4)
    assert result.array[:, 0, 0].tolist() == [0, 1, 2, 3, 4]
    assert result.configured_levels == 4
    assert result.occupied_levels == 4
    assert result.roi_min == 2.0
    assert result.roi_max == 10.0


def test_fbn_partners_are_mask_specific_and_positive_in_roi() -> None:
    bundle = generate_synthetic_volume(VolumeConfig(edge=64, fov_mm=256.0))
    partners = {
        name: compile_mask_specific_fbn(bundle.image, mask, levels=32)
        for name, mask in bundle.masks.items()
    }
    for name, result in partners.items():
        roi = bundle.masks[name] != 0
        assert np.all(result.array[~roi] == 0)
        assert int(np.min(result.array[roi])) >= 1
        assert int(np.max(result.array[roi])) <= 32
        assert result.configured_levels == 32
        assert 1 <= result.occupied_levels <= 32
    assert not np.array_equal(partners["M1"].array, partners["M3"].array)


def test_fbn_fails_closed_for_invalid_inputs() -> None:
    with pytest.raises(RepresentationError, match="constant"):
        compile_mask_specific_fbn(np.ones((3, 3, 3)), np.ones((3, 3, 3)))
    with pytest.raises(RepresentationError, match="shapes differ"):
        compile_mask_specific_fbn(np.ones((3, 3, 3)), np.ones((3, 3, 2)))
    invalid_mask = np.ones((3, 3, 3), dtype=np.uint8)
    invalid_mask[0, 0, 0] = 2
    with pytest.raises(RepresentationError, match="canonical binary"):
        compile_mask_specific_fbn(np.arange(27).reshape(3, 3, 3), invalid_mask)


def test_ibsi_fbs_hand_oracle_and_uint_selection() -> None:
    image = np.asarray([-10.0, -9.1, -8.9, 245.1]).reshape(4, 1, 1)
    mask = np.ones((4, 1, 1), dtype=np.uint8)
    result = compile_mask_specific_fbs(image, mask, bin_width=1.0, anchor=-10.0)
    assert result.array[:, 0, 0].tolist() == [1, 1, 2, 256]
    assert result.array.dtype == np.dtype(np.uint16)
    assert result.configured_levels == 256
    assert result.occupied_levels == 3
    assert result.roi_min == -10.0
    assert result.roi_max == 245.1


def test_fbs_fails_closed_for_invalid_inputs() -> None:
    image = np.arange(8, dtype=float).reshape(2, 2, 2)
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    with pytest.raises(RepresentationError, match="no greater"):
        compile_mask_specific_fbs(image, mask, anchor=1.0)
    mask[0, 0, 0] = 2
    with pytest.raises(RepresentationError, match="canonical binary"):
        compile_mask_specific_fbs(image, mask)


def test_texture_workload_counts_26_connected_zones_and_boundary() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    discrete = np.zeros_like(mask)
    discrete[1:4, 1:4, 1:4] = 1
    # One diagonal corner voxel of level 2 forms its own zone.
    discrete[1, 1, 1] = 2
    workload = texture_workload(discrete, mask)
    assert workload["zone_count_26"] == 2
    assert workload["mean_voxels_per_zone"] == 13.5
    # Only the centre voxel of the 3x3x3 ROI has all six neighbours inside.
    assert workload["boundary_voxels_6"] == 26

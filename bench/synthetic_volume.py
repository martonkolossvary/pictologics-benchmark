"""Deterministic synthetic CT volumes for Pillar 1 and Pillar 2.

The generator builds a thigh-like anatomy with a soft-tissue tumour and its
registered mask catalogue (M1 to M4, and the whole-anatomy mask A1).  Geometry
is fixed in physical coordinates across the resolution ladder and keeps a
complete background-voxel guard on all six faces at the coarsest ``32^3`` grid.
Array axes and affine columns are ``x, y, z`` in RAS+ physical space.  Only
output arrays are volume-sized; temporaries are bounded by ``slab_depth``.  The
module never imports or invokes a radiomics adapter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
from scipy import ndimage


# The profiles share geometry and the same normalised texture field.  Only the
# HU calibration and texture/noise amplitudes differ.  Values are engineering
# choices for a CT-inspired benchmark, not claims about scanner protocols.
HU_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "low_contrast": {
        "display_label": "low contrast",
        "fat_hu": -82.0,
        "muscle_hu": (45.0, 49.0, 42.0, 47.0),
        "muscle_texture_hu": 8.0,
        "fascia_hu": -18.0,
        "fascia_texture_hu": 5.0,
        "fat_texture_hu": 6.0,
        "cortex_hu": 860.0,
        "cortex_texture_hu": 30.0,
        "marrow_hu": 120.0,
        "marrow_texture_hu": 16.0,
        "viable_hu": 39.0,
        "viable_texture_hu": 15.0,
        "necrotic_hu": 16.0,
        "necrotic_texture_hu": 5.0,
        "satellite_hu": 41.0,
        "satellite_texture_hu": 13.0,
        "scanner_noise_hu": 2.0,
    },
    "reference": {
        "display_label": "reference",
        "fat_hu": -92.0,
        "muscle_hu": (47.0, 53.0, 43.0, 50.0),
        "muscle_texture_hu": 10.0,
        "fascia_hu": -24.0,
        "fascia_texture_hu": 6.0,
        "fat_texture_hu": 7.0,
        "cortex_hu": 920.0,
        "cortex_texture_hu": 38.0,
        "marrow_hu": 135.0,
        "marrow_texture_hu": 20.0,
        "viable_hu": 44.0,
        "viable_texture_hu": 22.0,
        "necrotic_hu": 8.0,
        "necrotic_texture_hu": 6.0,
        "satellite_hu": 46.0,
        "satellite_texture_hu": 18.0,
        "scanner_noise_hu": 3.0,
    },
    "high_contrast": {
        "display_label": "higher contrast",
        "fat_hu": -105.0,
        "muscle_hu": (49.0, 56.0, 44.0, 53.0),
        "muscle_texture_hu": 12.0,
        "fascia_hu": -31.0,
        "fascia_texture_hu": 7.0,
        "fat_texture_hu": 8.0,
        "cortex_hu": 980.0,
        "cortex_texture_hu": 45.0,
        "marrow_hu": 145.0,
        "marrow_texture_hu": 24.0,
        "viable_hu": 68.0,
        "viable_texture_hu": 26.0,
        "necrotic_hu": 12.0,
        "necrotic_texture_hu": 8.0,
        "satellite_hu": 70.0,
        "satellite_texture_hu": 22.0,
        "scanner_noise_hu": 4.5,
    },
}

# The noise profiles share the reference tissue values and differ only in
# independent per-voxel noise blurred by a scanner-like point-spread function.
# Smooth texture fields alone give zones that grow with grid size; per-voxel
# noise keeps the texture workload per voxel comparable across sizes.
for _name, _label, _noise_hu in (
    ("low_noise", "low noise", 12.0),
    ("reference_noise", "reference noise", 20.0),
    ("high_noise", "high noise", 30.0),
):
    HU_PROFILES[_name] = {
        **HU_PROFILES["reference"],
        "display_label": _label,
        "voxel_noise_hu": _noise_hu,
        "voxel_noise_psf_voxels": 0.8,
    }


@dataclass(frozen=True)
class VolumeConfig:
    """Physical and numerical controls for one synthetic volume."""

    edge: int = 128
    fov_mm: float = 256.0
    seed: int = 20260809
    slab_depth: int = 8
    hu_profile: str = "reference"

    def __post_init__(self) -> None:
        if self.edge < 24:
            raise ValueError("edge must be at least 24 for inspectable anatomy")
        if not np.isfinite(self.fov_mm) or self.fov_mm <= 0:
            raise ValueError("fov_mm must be finite and positive")
        if self.slab_depth < 1:
            raise ValueError("slab_depth must be positive")
        if self.hu_profile not in HU_PROFILES:
            raise ValueError(
                f"hu_profile must be one of {tuple(HU_PROFILES)}, "
                f"received {self.hu_profile!r}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.edge, self.edge, self.edge)

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        spacing = float(self.fov_mm) / float(self.edge)
        return (spacing, spacing, spacing)


def _affine(config: VolumeConfig) -> np.ndarray:
    spacing = config.spacing_mm[0]
    first_center = -0.5 * config.fov_mm + 0.5 * spacing
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing
    affine[1, 1] = spacing
    affine[2, 2] = spacing
    affine[:3, 3] = first_center
    return affine


def _axis_centres(config: VolumeConfig) -> np.ndarray:
    spacing = config.spacing_mm[0]
    return (
        -0.5 * config.fov_mm
        + (np.arange(config.edge, dtype=np.float32) + 0.5) * spacing
    )


def _phase(seed: int, name: str, term: int) -> float:
    digest = hashlib.sha256(f"{seed}:{name}:{term}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return float(2.0 * np.pi * fraction)


def _unit_interval(seed: int, name: str, term: int, field: str) -> float:
    digest = hashlib.sha256(f"{seed}:{name}:{term}:{field}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _texture(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    seed: int,
    name: str,
    correlation_mm: float,
) -> np.ndarray:
    """Deterministic isotropic random-wave field, invariant to slab traversal.

    Directions, wavelengths, phases, and weights are all derived independently
    from SHA-256.  Sixteen modes are enough to suppress the conspicuous diagonal
    banding produced by the original four hand-selected directions while keeping
    a 192-cube volume practical.
    """

    mode_count = 16
    out = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), dtype=np.float32)
    weights: list[float] = []
    modes: list[tuple[float, float, float, float, float]] = []
    for term in range(mode_count):
        # Uniform on S2: azimuth is uniform and cos(polar angle) is uniform.
        azimuth = 2.0 * np.pi * _unit_interval(seed, name, term, "azimuth")
        dz = 2.0 * _unit_interval(seed, name, term, "polar") - 1.0
        radial = np.sqrt(max(0.0, 1.0 - dz * dz))
        dx = radial * np.cos(azimuth)
        dy = radial * np.sin(azimuth)
        # Log-uniform wavelengths prevent a single visible carrier frequency.
        scale = float(
            np.exp(
                np.log(0.62)
                + _unit_interval(seed, name, term, "scale")
                * (np.log(1.75) - np.log(0.62))
            )
        )
        weight = 0.65 + 0.70 * _unit_interval(seed, name, term, "weight")
        weights.append(weight)
        modes.append((dx, dy, dz, scale, _phase(seed, name, term)))
    normalizer = float(np.sqrt(0.5 * sum(weight * weight for weight in weights)))
    for weight, (dx, dy, dz, scale, phase) in zip(weights, modes):
        k = np.float32(2.0 * np.pi * scale / correlation_mm)
        argument = k * (dx * x + dy * y + dz * z) + phase
        out += np.float32(weight) * np.sin(argument, dtype=np.float32)
    out /= np.float32(normalizer)
    return out


def _ellipsoid_score(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    q = (
        ((x - center[0]) / radii[0]) ** 2
        + ((y - center[1]) / radii[1]) ** 2
        + ((z - center[2]) / radii[2]) ** 2
    )
    return np.asarray(1.0 - q, dtype=np.float32)


def _tapered_spicule(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    base_radius_mm: float,
    tip_radius_mm: float,
) -> np.ndarray:
    """Return a bounded 3D cone-like projection joined to the lesion surface."""

    vx, vy, vz = (end[index] - start[index] for index in range(3))
    denominator = vx * vx + vy * vy + vz * vz
    raw_t = (
        (x - start[0]) * vx + (y - start[1]) * vy + (z - start[2]) * vz
    ) / denominator
    t = np.clip(raw_t, 0.0, 1.0)
    distance2 = (
        (x - (start[0] + t * vx)) ** 2
        + (y - (start[1] + t * vy)) ** 2
        + (z - (start[2] + t * vz)) ** 2
    )
    radius = base_radius_mm + t * (tip_radius_mm - base_radius_mm)
    return np.asarray((raw_t >= 0.0) & (raw_t <= 1.0) & (distance2 <= radius**2))


def _anatomy_slab(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    seed: int,
    hu_profile: Mapping[str, Any],
    body_radii_mm: tuple[float, float] = (82.0, 70.0),
    body_taper_fraction: float = 0.11,
    body_z_half_extent_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render asymmetric thigh anatomy before lesion overlays.

    The optional geometry arguments let the volumetric benchmark use the same
    tissue renderer with a denser, hashable anatomical support.  Defaults keep
    the reference geometry unchanged.  ``body_z_half_extent_mm`` is a
    physical support guard, not a display crop.
    """

    body_rx_mm, body_ry_mm = (float(value) for value in body_radii_mm)
    if body_rx_mm <= 0 or body_ry_mm <= 0:
        raise ValueError("body radii must be positive")
    if not 0 <= body_taper_fraction < 1:
        raise ValueError("body_taper_fraction must be in [0, 1)")
    if body_z_half_extent_mm is not None and body_z_half_extent_mm <= 0:
        raise ValueError("body_z_half_extent_mm must be positive when supplied")

    z_fraction = z / np.float32(128.0)
    center_x = 2.5 + 2.0 * np.sin(z / 67.0)
    center_y = 1.0 - 1.5 * np.cos(z / 74.0)
    taper = 1.0 - float(body_taper_fraction) * z_fraction**2
    rx = body_rx_mm * taper
    ry = body_ry_mm * taper
    dx = x - center_x
    dy = y - center_y
    theta = np.arctan2(dy / ry, dx / rx)
    outer_radius = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    outer_limit = (
        1.0
        + 0.035 * np.sin(theta + 0.45)
        + 0.024 * np.sin(3.0 * theta + z / 45.0)
        + 0.010 * np.cos(5.0 * theta - z / 61.0)
    )
    body = outer_radius <= outer_limit
    if body_z_half_extent_mm is not None:
        body &= np.abs(z) < float(body_z_half_extent_mm)

    muscle_dx = x - (center_x + 1.5)
    muscle_dy = y - (center_y - 1.5)
    muscle_rx = np.maximum(rx - 13.0, 20.0)
    muscle_ry = np.maximum(ry - 12.0, 20.0)
    muscle_radius = np.sqrt((muscle_dx / muscle_rx) ** 2 + (muscle_dy / muscle_ry) ** 2)
    muscle = body & (muscle_radius <= 1.0 + 0.018 * np.cos(4.0 * theta - z / 53.0))

    image = np.full(np.broadcast_shapes(x.shape, y.shape, z.shape), -1000.0, np.float32)
    labels = np.zeros(image.shape, dtype=np.uint8)
    image[body] = float(hu_profile["fat_hu"])
    labels[body] = 1

    # Four unequal grouped-muscle regions.  The anterior group is the containing
    # muscle bed; three bounded, off-centre elliptical groups are embedded in it.
    # This avoids the visually artificial four-way Voronoi/pinwheel junction.
    scale_x = body_rx_mm / 82.0
    scale_y = body_ry_mm / 70.0
    anterior = (
        (muscle_dx - 5.0 * scale_x - scale_x * np.sin(z / 57.0)) / (39.0 * scale_x)
    ) ** 2 + ((muscle_dy + 31.0 * scale_y) / (24.0 * scale_y)) ** 2
    medial = (
        (muscle_dx - 35.0 * scale_x - 3.0 * scale_x * np.sin(z / 29.0))
        / (23.0 * scale_x)
    ) ** 2 + (
        (muscle_dy - 5.0 * scale_y - 1.2 * scale_y * np.cos(z / 51.0))
        / (37.0 * scale_y)
    ) ** 2
    posterior = (
        (muscle_dx + 1.0 * scale_x - 3.0 * scale_x * np.sin(z / 33.0))
        / (42.0 * scale_x)
    ) ** 2 + ((muscle_dy - 32.0 * scale_y) / (25.0 * scale_y)) ** 2
    lateral = (
        (muscle_dx + 35.0 * scale_x - 3.0 * scale_x * np.cos(z / 31.0))
        / (26.0 * scale_x)
    ) ** 2 + (
        (muscle_dy + 2.0 * scale_y - scale_y * np.cos(z / 69.0)) / (36.0 * scale_y)
    ) ** 2
    # An anterior oval establishes a non-radial upper group.  Outside it, label
    # 2 remains a modest connective/background muscle bed between groups.
    compartment = np.full(np.broadcast_shapes(x.shape, y.shape, z.shape), 2, np.uint8)
    anterior_group = anterior <= 1.20
    medial_group = medial <= 1.05
    posterior_group = posterior <= 1.08
    lateral_group = lateral <= 1.05
    # Resolve the small peripheral overlaps locally by the lowest elliptical
    # score; no global power diagram or shared four-way junction is created.
    bounded_scores = np.stack(
        np.broadcast_arrays(medial, posterior, lateral), axis=0
    ).astype(np.float32, copy=False)
    bounded_active = np.stack(
        np.broadcast_arrays(medial_group, posterior_group, lateral_group), axis=0
    )
    bounded_scores = np.where(bounded_active, bounded_scores, np.inf)
    bounded_choice = np.argmin(bounded_scores, axis=0)
    any_bounded = np.any(bounded_active, axis=0)
    bounded_codes = np.asarray([3, 4, 5], dtype=np.uint8)
    compartment[any_bounded] = bounded_codes[bounded_choice[any_bounded]]
    compartment[anterior_group & ~any_bounded] = 2

    # Short, curved fascial arcs follow selected outer portions of the grouped
    # ellipses.  They are spatially windowed in x/y/z and therefore cannot form
    # full-length dark planes in either longitudinal view.
    medial_arc = (
        (np.abs(np.sqrt(medial) - 1.0) < 0.055)
        & (muscle_dx > 12.0 * scale_x)
        & (muscle_dy > -24.0 * scale_y)
        & (muscle_dy < 31.0 * scale_y)
        & (np.abs(z + 12.0) < 32.0)
    )
    posterior_arc = (
        (np.abs(np.sqrt(posterior) - 1.0) < 0.055)
        & (muscle_dy > 13.0 * scale_y)
        & (muscle_dx > -24.0 * scale_x)
        & (muscle_dx < 28.0 * scale_x)
        & (np.abs(z - 18.0) < 30.0)
    )
    lateral_arc = (
        (np.abs(np.sqrt(lateral) - 1.0) < 0.055)
        & (muscle_dx < -13.0 * scale_x)
        & (muscle_dy > -27.0 * scale_y)
        & (muscle_dy < 27.0 * scale_y)
        & (np.abs(z + 20.0) < 27.0)
    )
    fascia = muscle & (medial_arc | posterior_arc | lateral_arc)
    muscle_texture = _texture(x, y, z, seed=seed, name="muscle", correlation_mm=13.0)
    muscle_hu = tuple(float(value) for value in hu_profile["muscle_hu"])
    compartment_hu = np.choose(
        compartment,
        [-1000.0, float(hu_profile["fat_hu"]), *muscle_hu],
    ).astype(np.float32)
    image[muscle] = (
        compartment_hu + float(hu_profile["muscle_texture_hu"]) * muscle_texture
    )[muscle]
    labels[muscle] = compartment[muscle]

    fascia_texture = _texture(x, y, z, seed=seed, name="fascia", correlation_mm=11.0)
    image[fascia] = (
        float(hu_profile["fascia_hu"])
        + float(hu_profile["fascia_texture_hu"]) * fascia_texture
    )[fascia]
    labels[fascia] = 11

    fat_texture = _texture(x, y, z, seed=seed, name="fat", correlation_mm=18.0)
    fat = body & ~muscle
    image[fat] += (float(hu_profile["fat_texture_hu"]) * fat_texture)[fat]

    # Eccentric femur with marrow and a dense cortical annulus.
    femur_x = -19.0 + 1.4 * np.sin(z / 58.0)
    femur_y = 8.0 + 1.0 * np.cos(z / 61.0)
    femur_r2 = (x - femur_x) ** 2 + (y - femur_y) ** 2
    cortex = body & (femur_r2 <= 14.5**2)
    marrow = body & (femur_r2 <= 8.5**2)
    bone_texture = _texture(x, y, z, seed=seed, name="bone", correlation_mm=8.0)
    image[cortex] = (
        float(hu_profile["cortex_hu"])
        + float(hu_profile["cortex_texture_hu"]) * bone_texture
    )[cortex]
    labels[cortex] = 6
    image[marrow] = (
        float(hu_profile["marrow_hu"])
        + float(hu_profile["marrow_texture_hu"]) * bone_texture
    )[marrow]
    labels[marrow] = 7

    return image, labels


def _slab_indices(
    config: VolumeConfig, order: Literal["forward", "reverse"]
) -> list[tuple[int, int]]:
    slabs = [
        (start, min(config.edge, start + config.slab_depth))
        for start in range(0, config.edge, config.slab_depth)
    ]
    if order == "reverse":
        slabs.reverse()
    elif order != "forward":
        raise ValueError("slab_order must be 'forward' or 'reverse'")
    return slabs


def _retain_component_overlapping_primary(
    candidate: np.ndarray,
    primary: np.ndarray,
) -> np.ndarray:
    """Discard discretely detached M3 fragments after voxelisation.

    The analytic capsules begin inside the expanded primary, but a thin oblique
    capsule can alias into detached voxels on a coarse grid.  Keeping the single
    26-connected component that overlaps M1 makes the topology contractual, not
    an accidental property of a particular spacing.  A uint8 work array bounds
    the additional memory to one byte per output voxel.
    """

    from scipy import ndimage

    component_labels = np.empty(candidate.shape, dtype=np.uint8)
    count = ndimage.label(
        candidate != 0,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
        output=component_labels,
    )
    if count <= 1:
        return np.asarray(candidate, dtype=np.uint8)
    overlapping = component_labels[primary != 0]
    overlapping = overlapping[overlapping != 0]
    if overlapping.size == 0:
        raise RuntimeError("M3 has no component overlapping the primary tumour")
    identifiers, counts = np.unique(overlapping, return_counts=True)
    keep = int(identifiers[int(np.argmax(counts))])
    return np.asarray(component_labels == keep, dtype=np.uint8)


MASK_IDS = ("M1", "M2", "M3", "M4", "A1")


REFERENCE_FOV_MM = 256.0


COARSEST_EDGE = 32


COARSE_GUARD_VOXELS = 1


# Maximal radii on a 0.25-mm deterministic search grid for the frozen curved,
# lobulated body model below.  Increasing either radius by 0.25 mm activates a
# face voxel at 32^3; this pair leaves exactly one background voxel on all six
# faces while maximising the observed coarse-grid A1 voxel count.
DENSE_BODY_RADII_MM = (110.75, 117.5)


DENSE_BODY_TAPER_FRACTION = 0.04


DENSE_BODY_Z_HALF_EXTENT_MM = 120.0


@dataclass(frozen=True)
class SyntheticVolume:
    """One synthetic volume and its registered mask catalogue."""

    image: np.ndarray
    labels: np.ndarray
    masks: Mapping[str, np.ndarray]
    affine: np.ndarray
    spacing_mm: tuple[float, float, float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MaskMetric:
    mask_id: str
    voxels: int
    volume_ml: float
    image_fraction: float
    body_fraction: float
    bbox_shape_xyz: tuple[int, int, int]
    face_margins_voxels: tuple[int, int, int, int, int, int]
    minimum_face_margin_voxels: int
    occupied_z_slices: int
    occupied_z_mm: float
    components_26: int
    cavities_background_6: int
    bone_overlap_voxels: int
    within_1p5mm_of_bone_voxels: int
    within_3p5mm_of_bone_voxels: int


@dataclass(frozen=True)
class VolumeMetrics:
    shape_xyz: tuple[int, int, int]
    spacing_mm_xyz: tuple[float, float, float]
    image_voxels: int
    body_voxels: int
    body_fraction: float
    masks: Mapping[str, MaskMetric]
    adjacent_slice_r_z: float
    adjacent_slice_pairs: int
    identical_adjacent_image_slices: int
    distinct_image_slices: int
    distinct_label_slices: int


def _warped_xy(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map world coordinates to a visibly curved and gently twisting limb."""

    angle = 0.11 * np.sin(z / 61.0) + 0.075 * z / 128.0
    cosine = np.cos(angle)
    sine = np.sin(angle)
    shift_x = 5.5 * np.sin(z / 52.0) + 1.8 * z / 128.0
    shift_y = 3.8 * np.sin(z / 43.0 + 0.55)
    centered_x = x - shift_x
    centered_y = y - shift_y
    return (
        cosine * centered_x + sine * centered_y,
        -sine * centered_x + cosine * centered_y,
    )


def _body_and_soft_support(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    body_radii_mm: tuple[float, float] = DENSE_BODY_RADII_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body, non-bone soft tissue, and the warped femoral cylinder."""

    xw, yw = _warped_xy(x, y, z)
    z_fraction = z / np.float32(128.0)
    center_x = 2.5 + 2.0 * np.sin(z / 67.0)
    center_y = 1.0 - 1.5 * np.cos(z / 74.0)
    taper = 1.0 - DENSE_BODY_TAPER_FRACTION * z_fraction**2
    rx = float(body_radii_mm[0]) * taper
    ry = float(body_radii_mm[1]) * taper
    dx = xw - center_x
    dy = yw - center_y
    theta = np.arctan2(dy / ry, dx / rx)
    outer_radius = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    outer_limit = (
        1.0
        + 0.035 * np.sin(theta + 0.45)
        + 0.024 * np.sin(3.0 * theta + z / 45.0)
        + 0.010 * np.cos(5.0 * theta - z / 61.0)
    )
    body = (outer_radius <= outer_limit) & (np.abs(z) < DENSE_BODY_Z_HALF_EXTENT_MM)
    femur_x = -19.0 + 1.4 * np.sin(z / 58.0)
    femur_y = 8.0 + 1.0 * np.cos(z / 61.0)
    femur_r2 = (xw - femur_x) ** 2 + (yw - femur_y) ** 2
    bone = body & (femur_r2 <= 14.5**2)
    return np.asarray(body), np.asarray(body & ~bone), np.asarray(bone)


def _large_primary_score(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Curved lobulated peri-femoral lesion in physical coordinates."""

    center_x = 42.0 + 7.0 * np.sin(z / 52.0)
    center_y = -7.0 + 5.0 * np.sin(z / 43.0 + 0.70)
    dx = x - center_x
    dy = y - center_y
    theta = np.arctan2(dy / 58.0, dx / 65.0)
    radial = np.sqrt((dx / 65.0) ** 2 + (dy / 58.0) ** 2 + (z / 104.0) ** 2)
    boundary = (
        1.0
        + 0.045 * np.sin(3.0 * theta + z / 39.0)
        + 0.025 * np.cos(5.0 * theta - z / 47.0)
        + 0.015 * np.sin(2.0 * theta - z / 31.0)
    )
    return np.asarray(boundary - radial, dtype=np.float32)


def _large_core(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    center_x = 50.0 + 4.0 * np.sin((z - 5.0) / 55.0)
    center_y = -11.0 + 3.0 * np.sin(z / 47.0 + 0.4)
    score = (
        1.0
        - ((x - center_x) / 36.0) ** 2
        - ((y - center_y) / 29.0) ** 2
        - ((z - 4.0) / 57.0) ** 2
    )
    return np.asarray(score >= 0)


def _spicule_segments() -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float], float, float],
    ...,
]:
    """Return eighteen deterministic, spatially distributed 3D projections."""

    segments = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    center = np.asarray((42.0, -7.0, 0.0), dtype=float)
    radii = np.asarray((65.0, 58.0, 104.0), dtype=float)
    for index in range(18):
        unit_z = 1.0 - 2.0 * (index + 0.5) / 18.0
        unit_xy = np.sqrt(max(0.0, 1.0 - unit_z**2))
        phi = index * golden_angle + 0.37
        direction = np.asarray(
            (unit_xy * np.cos(phi), unit_xy * np.sin(phi), unit_z),
            dtype=float,
        )
        start = center + 0.74 * radii * direction
        end = center + radii * direction + (30.0 + 4.0 * (index % 4)) * direction
        segments.append(
            (
                tuple(float(value) for value in start),
                tuple(float(value) for value in end),
                9.0,
                2.8,
            )
        )
    return tuple(segments)


LARGE_SPICULE_SEGMENTS = _spicule_segments()


def _large_lesion_fields(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    soft_tissue: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score = _large_primary_score(x, y, z)
    primary = (score >= 0) & soft_tissue
    core = _large_core(x, y, z) & primary
    spicules = np.zeros(np.broadcast_shapes(x.shape, y.shape, z.shape), dtype=bool)
    for start, end, base_radius, tip_radius in LARGE_SPICULE_SEGMENTS:
        spicules |= _tapered_spicule(
            x,
            y,
            z,
            start,
            end,
            base_radius,
            tip_radius,
        )
    spiculated = (primary | spicules) & soft_tissue

    satellite_specs = (
        ((-62.0, -57.0, 70.0), (10.0, 9.0, 12.0)),
        ((-58.0, 62.0, 68.0), (10.0, 9.0, 12.0)),
        ((-62.0, -60.0, -67.0), (10.0, 9.0, 12.0)),
        ((-58.0, 66.0, -65.0), (10.0, 9.0, 12.0)),
        ((0.0, 86.0, 0.0), (10.0, 8.0, 11.0)),
        ((0.0, -92.0, 0.0), (10.0, 8.0, 11.0)),
    )
    satellites = np.zeros_like(primary)
    for center, radii in satellite_specs:
        satellites |= _ellipsoid_score(x, y, z, center, radii) >= 0
    satellites &= soft_tissue & ~primary
    return primary, core, spiculated, satellites


def _voxel_noise(config: VolumeConfig, profile: Mapping[str, Any]) -> np.ndarray | None:
    """Return the profile's blurred per-voxel noise volume, or None.

    One seeded volume is drawn for the whole grid, so slab order cannot change
    it.  The blur is in voxel units and the result is rescaled to the target
    standard deviation.
    """

    noise_hu = float(profile.get("voxel_noise_hu", 0.0))
    if noise_hu <= 0.0:
        return None
    rng = np.random.default_rng([int(config.seed), int(config.edge)])
    noise = rng.standard_normal(config.shape, dtype=np.float32)
    noise = ndimage.gaussian_filter(
        noise, sigma=float(profile["voxel_noise_psf_voxels"])
    )
    noise *= np.float32(noise_hu / float(noise.std()))
    return noise


def generate_synthetic_volume(
    config: VolumeConfig = VolumeConfig(),
    *,
    slab_order: Literal["forward", "reverse"] = "forward",
) -> SyntheticVolume:
    """Generate actual review arrays without invoking benchmark adapters."""

    if config.edge < COARSEST_EDGE:
        raise ValueError(
            f"volumetric geometry requires edge >= {COARSEST_EDGE}; "
            "smaller grids cannot preserve its one-voxel face guard"
        )
    if not np.isclose(config.fov_mm, REFERENCE_FOV_MM):
        raise ValueError(
            f"volumetric geometry is frozen to a {REFERENCE_FOV_MM:g}-mm field"
        )

    image = np.empty(config.shape, dtype=np.int16)
    labels = np.empty(config.shape, dtype=np.uint8)
    masks = {name: np.zeros(config.shape, dtype=np.uint8) for name in MASK_IDS}
    profile = HU_PROFILES[config.hu_profile]
    voxel_noise = _voxel_noise(config, profile)
    axis = _axis_centres(config)
    x = axis[:, None, None]
    y = axis[None, :, None]

    for z_start, z_stop in _slab_indices(config, slab_order):
        z = axis[None, None, z_start:z_stop]
        xw, yw = _warped_xy(x, y, z)
        slab_image, slab_labels = _anatomy_slab(
            xw,
            yw,
            z,
            seed=config.seed,
            hu_profile=profile,
            body_radii_mm=DENSE_BODY_RADII_MM,
            body_taper_fraction=DENSE_BODY_TAPER_FRACTION,
            body_z_half_extent_mm=DENSE_BODY_Z_HALF_EXTENT_MM,
        )
        body, soft_tissue, _bone = _body_and_soft_support(x, y, z)
        primary, core, spiculated, satellites = _large_lesion_fields(
            x, y, z, soft_tissue
        )
        viable = primary & ~core
        tumour_texture = _texture(
            x, y, z, seed=config.seed, name="large_tumour", correlation_mm=12.0
        )
        slab_image[viable] = (
            float(profile["viable_hu"])
            + float(profile["viable_texture_hu"]) * tumour_texture
        )[viable]
        slab_image[core] = (
            float(profile["necrotic_hu"])
            + float(profile["necrotic_texture_hu"]) * tumour_texture
        )[core]
        slab_image[satellites] = (
            float(profile["satellite_hu"])
            + float(profile["satellite_texture_hu"]) * tumour_texture
        )[satellites]
        slab_labels[viable] = 8
        slab_labels[core] = 9
        slab_labels[satellites] = 10
        scanner = _texture(
            x, y, z, seed=config.seed, name="large_scanner", correlation_mm=3.3
        )
        slab_image += float(profile["scanner_noise_hu"]) * scanner
        if voxel_noise is not None:
            slab_image += voxel_noise[:, :, z_start:z_stop]

        image[:, :, z_start:z_stop] = np.rint(
            np.clip(slab_image, -1024.0, 2000.0)
        ).astype(np.int16)
        labels[:, :, z_start:z_stop] = slab_labels
        masks["M1"][:, :, z_start:z_stop] = primary
        masks["M2"][:, :, z_start:z_stop] = viable
        masks["M3"][:, :, z_start:z_stop] = spiculated
        masks["M4"][:, :, z_start:z_stop] = primary | satellites
        masks["A1"][:, :, z_start:z_stop] = body

    masks["M3"] = _retain_component_overlapping_primary(masks["M3"], masks["M1"])
    a1_margins = _face_margins_voxels(masks["A1"] != 0)
    if min(a1_margins) < COARSE_GUARD_VOXELS:
        raise RuntimeError(
            "whole-anatomy support violates the one-voxel image-face guard: "
            f"observed margins={a1_margins}"
        )
    metadata = {
        "generator": "synthetic_volumetric_phantom",
        "generator_version": 1,
        "status": "production_pillar1_geometry_v1",
        "shape_xyz": list(config.shape),
        "spacing_mm_xyz": list(config.spacing_mm),
        "fov_mm_xyz": [config.fov_mm] * 3,
        "coordinate_system": "RAS+",
        "array_axis_order": "xyz",
        "hu_profile_id": config.hu_profile,
        # Present only for noise profiles, so older metadata stays unchanged.
        **(
            {
                "voxel_noise_hu": float(profile["voxel_noise_hu"]),
                "voxel_noise_psf_voxels": float(profile["voxel_noise_psf_voxels"]),
            }
            if voxel_noise is not None
            else {}
        ),
        "seed": config.seed,
        "geometry_contract": {
            "reference_fov_mm": REFERENCE_FOV_MM,
            "coarsest_edge": COARSEST_EDGE,
            "coarse_guard_voxels_all_faces": COARSE_GUARD_VOXELS,
            "body_radii_mm_xy": list(DENSE_BODY_RADII_MM),
            "maximization_search_step_mm": 0.25,
            "maximality_witness": (
                "increasing either transverse radius by 0.25 mm activates "
                "an N=32 face voxel"
            ),
            "body_taper_fraction": DENSE_BODY_TAPER_FRACTION,
            "body_z_half_extent_mm": DENSE_BODY_Z_HALF_EXTENT_MM,
            "observed_a1_face_margins_voxels": list(a1_margins),
        },
        "mask_definitions": {
            "M1": "dense curved peri-femoral whole tumour including necrosis",
            "M2": "viable shell excluding the image-visible necrotic cavity",
            "M3": "M1 with eighteen attached tapered three-dimensional spicules",
            "M4": "M1 plus six image-visible satellite foci",
            "A1": "dense whole anatomy including bone and pathology",
        },
        "spicule_count": len(LARGE_SPICULE_SEGMENTS),
        "satellite_count": 6,
    }
    return SyntheticVolume(
        image=image,
        labels=labels,
        masks=masks,
        affine=_affine(config),
        spacing_mm=config.spacing_mm,
        metadata=metadata,
    )


def _slice_digest_count(volume: np.ndarray) -> int:
    return len(
        {
            np.ascontiguousarray(volume[:, :, index]).tobytes()
            for index in range(volume.shape[2])
        }
    )


def _face_margins_voxels(mask: np.ndarray) -> tuple[int, int, int, int, int, int]:
    """Return low/high background margins for x, y, and z image faces."""

    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("cannot measure image-face margins of an empty mask")
    low = np.min(coordinates, axis=0)
    high = np.max(coordinates, axis=0)
    shape = np.asarray(mask.shape, dtype=int)
    margins = (
        int(low[0]),
        int(shape[0] - 1 - high[0]),
        int(low[1]),
        int(shape[1] - 1 - high[1]),
        int(low[2]),
        int(shape[2] - 1 - high[2]),
    )
    return margins


def _cavity_count(mask: np.ndarray) -> int:
    background = ~mask
    labels, count = ndimage.label(
        background,
        structure=ndimage.generate_binary_structure(3, 1),
    )
    exterior = np.unique(
        np.concatenate(
            (
                labels[0, :, :].ravel(),
                labels[-1, :, :].ravel(),
                labels[:, 0, :].ravel(),
                labels[:, -1, :].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            )
        )
    )
    return int(sum(identifier not in exterior for identifier in range(1, count + 1)))


def evaluate_synthetic_volume(
    bundle: SyntheticVolume,
) -> VolumeMetrics:
    """Measure topology, Z support, burden, and longitudinal correlation."""

    image = np.asarray(bundle.image)
    labels = np.asarray(bundle.labels)
    spacing = tuple(float(value) for value in bundle.spacing_mm)
    image_voxels = int(image.size)
    body = np.asarray(bundle.masks["A1"]) != 0
    body_voxels = int(np.count_nonzero(body))
    bone = np.isin(labels, (6, 7))
    distance_to_bone = ndimage.distance_transform_edt(~bone, sampling=spacing)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    records: dict[str, MaskMetric] = {}
    voxel_volume_mm3 = float(np.prod(spacing))
    for mask_id, source in bundle.masks.items():
        mask = np.asarray(source) != 0
        coordinates = np.argwhere(mask)
        if coordinates.size == 0:
            raise ValueError(f"synthetic mask {mask_id} is empty")
        low = np.min(coordinates, axis=0)
        high = np.max(coordinates, axis=0)
        z_slices = int(high[2] - low[2] + 1)
        _, component_count = ndimage.label(mask, structure=structure)
        is_tumour = mask_id != "A1"
        face_margins = _face_margins_voxels(mask)
        records[mask_id] = MaskMetric(
            mask_id=mask_id,
            voxels=int(coordinates.shape[0]),
            volume_ml=float(coordinates.shape[0] * voxel_volume_mm3 / 1000.0),
            image_fraction=float(coordinates.shape[0] / image_voxels),
            body_fraction=float(coordinates.shape[0] / body_voxels),
            bbox_shape_xyz=tuple(int(value) for value in high - low + 1),
            face_margins_voxels=face_margins,
            minimum_face_margin_voxels=min(face_margins),
            occupied_z_slices=z_slices,
            occupied_z_mm=float(z_slices * spacing[2]),
            components_26=int(component_count),
            cavities_background_6=_cavity_count(mask),
            bone_overlap_voxels=(
                int(np.count_nonzero(mask & bone))
                if is_tumour
                else int(np.count_nonzero(mask & bone))
            ),
            within_1p5mm_of_bone_voxels=(
                int(np.count_nonzero(mask & (distance_to_bone <= 1.5)))
                if is_tumour
                else 0
            ),
            within_3p5mm_of_bone_voxels=(
                int(np.count_nonzero(mask & (distance_to_bone <= 3.5)))
                if is_tumour
                else 0
            ),
        )

    m1 = np.asarray(bundle.masks["M1"]) != 0
    pair_mask = m1[:, :, :-1] & m1[:, :, 1:]
    first = image[:, :, :-1][pair_mask].astype(np.float64)
    second = image[:, :, 1:][pair_mask].astype(np.float64)
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        raise ValueError("adjacent-slice correlation is undefined")
    identical = sum(
        np.array_equal(image[:, :, index], image[:, :, index + 1])
        for index in range(image.shape[2] - 1)
    )
    return VolumeMetrics(
        shape_xyz=tuple(int(value) for value in image.shape),
        spacing_mm_xyz=spacing,
        image_voxels=image_voxels,
        body_voxels=body_voxels,
        body_fraction=float(body_voxels / image_voxels),
        masks=records,
        adjacent_slice_r_z=float(np.corrcoef(first, second)[0, 1]),
        adjacent_slice_pairs=int(first.size),
        identical_adjacent_image_slices=int(identical),
        distinct_image_slices=_slice_digest_count(image),
        distinct_label_slices=_slice_digest_count(labels),
    )


def measure_spicule_external_voxels(
    bundle: SyntheticVolume,
) -> tuple[int, ...]:
    """Count actual retained exterior voxels contributed by every M3 branch.

    Counts are deliberately measured after voxelisation, body clipping, bone
    exclusion, and final connected-component retention. A positive value means
    the declared analytic branch is represented outside M1 on that grid.
    """

    shape = tuple(int(value) for value in bundle.image.shape)
    if len(set(shape)) != 1:
        raise ValueError("spicule audit requires a cubic grid")
    axis = np.arange(shape[0], dtype=np.float32) * float(bundle.affine[0, 0]) + float(
        bundle.affine[0, 3]
    )
    x = axis[:, None, None]
    y = axis[None, :, None]
    z = axis[None, None, :]
    _body, soft_tissue, _bone = _body_and_soft_support(x, y, z)
    retained_exterior = (np.asarray(bundle.masks["M3"]) != 0) & ~(
        np.asarray(bundle.masks["M1"]) != 0
    )
    return tuple(
        int(
            np.count_nonzero(
                _tapered_spicule(
                    x,
                    y,
                    z,
                    start,
                    end,
                    base_radius,
                    tip_radius,
                )
                & soft_tissue
                & retained_exterior
            )
        )
        for start, end, base_radius, tip_radius in LARGE_SPICULE_SEGMENTS
    )


def measure_satellite_component_voxels(
    bundle: SyntheticVolume,
) -> tuple[int, ...]:
    """Return sorted actual voxel counts for the six M4-only components."""

    satellites = (np.asarray(bundle.masks["M4"]) != 0) & ~(
        np.asarray(bundle.masks["M1"]) != 0
    )
    labels, count = ndimage.label(
        satellites, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    return tuple(
        sorted(
            int(np.count_nonzero(labels == identifier))
            for identifier in range(1, count + 1)
        )
    )


def measure_volume_scale(
    edge: int,
    *,
    fov_mm: float = 256.0,
    slab_depth: int = 8,
) -> Mapping[str, int | float]:
    """Measure M1/A1 scale without materialising a full image volume."""

    if edge < COARSEST_EDGE:
        raise ValueError(f"edge must be >= {COARSEST_EDGE}")
    if not np.isclose(fov_mm, REFERENCE_FOV_MM):
        raise ValueError(f"fov_mm must equal {REFERENCE_FOV_MM:g}")

    config = VolumeConfig(edge=edge, fov_mm=fov_mm, slab_depth=slab_depth)
    axis = _axis_centres(config)
    x = axis[:, None, None]
    y = axis[None, :, None]
    m1_count = 0
    a1_count = 0
    m1_z = 0
    a1_z = 0
    m1_low = np.full(3, edge, dtype=int)
    m1_high = np.full(3, -1, dtype=int)
    a1_low = np.full(3, edge, dtype=int)
    a1_high = np.full(3, -1, dtype=int)
    for z_start, z_stop in _slab_indices(config, "forward"):
        z = axis[None, None, z_start:z_stop]
        body, soft, _bone = _body_and_soft_support(x, y, z)
        primary = (_large_primary_score(x, y, z) >= 0) & soft
        m1_count += int(np.count_nonzero(primary))
        a1_count += int(np.count_nonzero(body))
        m1_z += int(np.count_nonzero(np.any(primary, axis=(0, 1))))
        a1_z += int(np.count_nonzero(np.any(body, axis=(0, 1))))
        for mask, low, high in (
            (primary, m1_low, m1_high),
            (body, a1_low, a1_high),
        ):
            coordinates = np.argwhere(mask)
            if coordinates.size:
                coordinates[:, 2] += z_start
                low[:] = np.minimum(low, np.min(coordinates, axis=0))
                high[:] = np.maximum(high, np.max(coordinates, axis=0))

    def margins(low: np.ndarray, high: np.ndarray) -> tuple[int, ...]:
        return (
            int(low[0]),
            int(edge - 1 - high[0]),
            int(low[1]),
            int(edge - 1 - high[1]),
            int(low[2]),
            int(edge - 1 - high[2]),
        )

    a1_margins = margins(a1_low, a1_high)
    if min(a1_margins) < COARSE_GUARD_VOXELS:
        raise RuntimeError(f"A1 violates the face guard at edge={edge}: {a1_margins}")
    return {
        "edge": int(edge),
        "spacing_mm": float(fov_mm / edge),
        "image_voxels": int(edge**3),
        "m1_voxels": m1_count,
        "m1_image_fraction": float(m1_count / edge**3),
        "m1_z_slices": m1_z,
        "m1_z_mm": float(m1_z * fov_mm / edge),
        "m1_face_margins_voxels": margins(m1_low, m1_high),
        "a1_voxels": a1_count,
        "a1_image_fraction": float(a1_count / edge**3),
        "a1_z_slices": a1_z,
        "a1_z_mm": float(a1_z * fov_mm / edge),
        "a1_face_margins_voxels": a1_margins,
    }

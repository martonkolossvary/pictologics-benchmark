"""Study-design figures (Figure 1, Figures S1 and S2) from the frozen workspace.

Every image comes from the hash-checked NIfTI files of ``data/benchmark``
and every number from its manifests, never from regenerated arrays.  Images use
the radiological convention: the patient's right is on the left of the image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from bench.figures import style

CT_WINDOW = (-160.0, 240.0)  # level 40 HU, width 400 HU
MASK_ORDER = ("M1", "M2", "M3", "M4")
NOISE_ORDER = ("low_noise", "reference_noise", "high_noise")
EXAMPLE_PATIENT = "STS_001"
CROP_MARGIN_MM = 12.0


@dataclass(frozen=True)
class Volume:
    data: np.ndarray
    spacing: tuple[float, float, float]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (
            0.0,
            self.data.shape[0] * self.spacing[0],
            0.0,
            self.data.shape[1] * self.spacing[1],
        )


def load(path: Path) -> Volume:
    image = nib.as_closest_canonical(nib.load(str(path)))
    return Volume(
        np.asanyarray(image.dataobj),
        tuple(float(v) for v in image.header.get_zooms()[:3]),
    )


def manifest_cases(dataset_dir: Path) -> dict[str, Mapping[str, Any]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return {case["case_id"]: case for case in manifest["cases"]}


def axial_index(mask: np.ndarray) -> int:
    """The nonempty axial slice nearest the mask centroid, with the largest area."""

    area = np.count_nonzero(mask, axis=(0, 1))
    centre = float(np.average(np.arange(mask.shape[2]), weights=area))
    candidates = np.flatnonzero(area)
    distance = np.abs(candidates - centre)
    nearest = candidates[distance <= distance.min() + 1.0]
    return int(nearest[np.argmax(area[nearest])])


def crop_limits(
    mask2d: np.ndarray, spacing: Sequence[float], margin_mm: float
) -> tuple[float, float, float, float]:
    x, y = np.nonzero(mask2d)
    return (
        max(0.0, x.min() * spacing[0] - margin_mm),
        min(mask2d.shape[0] * spacing[0], (x.max() + 1) * spacing[0] + margin_mm),
        max(0.0, y.min() * spacing[1] - margin_mm),
        min(mask2d.shape[1] * spacing[1], (y.max() + 1) * spacing[1] + margin_mm),
    )


def show_slice(
    ax,
    volume: Volume,
    z: int,
    window: tuple[float, float],
    limits=None,
    cmap: str = "gray",
) -> None:
    ax.imshow(
        volume.data[:, :, z].T,
        origin="lower",
        cmap=cmap,
        vmin=window[0],
        vmax=window[1],
        extent=volume.extent,
        interpolation="nearest",
    )
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.invert_xaxis()  # radiological convention
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)


def outline(ax, mask: Volume, z: int, *, linestyle: Any = "-") -> None:
    """A white-under-black outline that shows on dark and bright tissue."""

    grid_x = (np.arange(mask.data.shape[0]) + 0.5) * mask.spacing[0]
    grid_y = (np.arange(mask.data.shape[1]) + 0.5) * mask.spacing[1]
    plane = (mask.data[:, :, z] > 0).T.astype(float)
    ax.contour(grid_x, grid_y, plane, levels=[0.5], colors=style.WHITE, linewidths=2.2)
    ax.contour(
        grid_x,
        grid_y,
        plane,
        levels=[0.5],
        colors=style.INK,
        linewidths=1.0,
        linestyles=[linestyle],
    )


def scale_bar(ax, length_mm: float) -> None:
    left, right = sorted(ax.get_xlim())
    bottom, top = sorted(ax.get_ylim())
    x = right - 0.06 * (right - left)  # the x axis is inverted
    y = bottom + 0.08 * (top - bottom)
    ax.add_patch(
        Rectangle(
            (x - length_mm, y - 0.012 * (top - bottom)),
            length_mm,
            0.024 * (top - bottom),
            facecolor=style.INK,
            edgecolor=style.WHITE,
            linewidth=0.8,
            zorder=5,
        )
    )
    ax.text(
        x - length_mm / 2,
        y + 0.05 * (top - bottom),
        f"{length_mm:g} mm",
        ha="center",
        va="bottom",
        bbox={"facecolor": style.WHITE, "edgecolor": "none", "pad": 0.8},
    )


def _letter(subfig, letter: str, title: str) -> None:
    """Bold panel letter and its title at the top left, a fixed distance apart."""

    from matplotlib.transforms import offset_copy

    # The title is a layout-aware suptitle, so the panels below it make room.
    width_pt = subfig.bbox.width / subfig.figure.dpi * 72
    subfig.suptitle(title, x=13 / width_pt, ha="left", fontsize=style_font(subfig))
    at = offset_copy(
        subfig.transSubfigure, fig=subfig.figure, x=1, y=-1, units="points"
    )
    subfig.text(
        0,
        1,
        letter,
        transform=at,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=style_font(subfig) + 2,
    )


def style_font(subfig) -> float:
    import matplotlib.pyplot as plt

    return plt.rcParams["font.size"]


def _row_masks(subfig, root: Path) -> None:
    cases = manifest_cases(root / "pillar1")
    base = "p1_reference_noise_{}_n256"
    image = load(root / "pillar1" / cases[base.format("m1")]["image_path"])
    masks = {
        key: load(root / "pillar1" / cases[base.format(key.lower())]["mask_path"])
        for key in MASK_ORDER
    }
    # The slice where M2 to M4 differ most from M1 shows the cavity, spicules
    # and satellites together.
    reference = masks["M1"].data > 0
    difference = sum(
        np.count_nonzero((masks[key].data > 0) != reference, axis=(0, 1))
        for key in MASK_ORDER[1:]
    )
    z = int(np.argmax(difference))
    union = np.any([mask.data[:, :, z] > 0 for mask in masks.values()], axis=0)
    limits = crop_limits(union, image.spacing, CROP_MARGIN_MM)
    axes = subfig.subplots(1, 4)
    for ax, key in zip(axes, MASK_ORDER):
        show_slice(ax, image, z, CT_WINDOW, limits)
        outline(ax, masks[key], z)
        ax.set_title(style.MASK_LABEL[key])
    scale_bar(axes[0], 20.0)
    _letter(
        subfig,
        "a",
        "Pillar 1: four tumour masks on one synthetic CT scene (256³ grid, reference noise)",
    )


def _row_noise_and_anatomy(subfig, root: Path) -> None:
    left, right = subfig.subfigures(1, 2, width_ratios=[3, 1.25])
    cases = manifest_cases(root / "pillar1")
    mask = load(root / "pillar1" / cases["p1_reference_noise_m1_n256"]["mask_path"])
    z = axial_index(mask.data)
    limits = crop_limits(mask.data[:, :, z] > 0, mask.spacing, CROP_MARGIN_MM)
    axes = left.subplots(1, 3)
    for ax, noise in zip(axes, NOISE_ORDER):
        image = load(root / "pillar1" / cases[f"p1_{noise}_m1_n256"]["image_path"])
        show_slice(ax, image, z, CT_WINDOW, limits)
        outline(ax, mask, z)
        ax.set_title(style.NOISE_LABEL[noise])
    _letter(left, "b", "Pillar 1: three noise levels of one seeded field")
    anatomy = manifest_cases(root / "pillar2_a1")["p2_a1_n256"]
    image = load(root / "pillar2_a1" / anatomy["image_path"])
    a1 = load(root / "pillar2_a1" / anatomy["mask_path"])
    ax = right.subplots()
    show_slice(ax, image, z, CT_WINDOW)
    outline(ax, a1, z)
    scale_bar(ax, 50.0)
    _letter(right, "c", "Pillar 2: whole-anatomy mask A1")


def _row_patient_and_inputs(subfig, root: Path) -> None:
    left, right = subfig.subfigures(1, 2, width_ratios=[1, 1])
    cases = manifest_cases(root / "ibsi2_phase3")
    axes = left.subplots(1, 3)
    for ax, modality in zip(axes, ("ct", "mri", "pet")):
        case = cases[f"{EXAMPLE_PATIENT}_{modality}"]
        image = load(root / "ibsi2_phase3" / case["image_path"])
        mask = load(root / "ibsi2_phase3" / case["mask_path"])
        z = axial_index(mask.data)
        limits = crop_limits(mask.data[:, :, z] > 0, mask.spacing, 25.0)
        if modality == "ct":
            window = CT_WINDOW
        else:
            region = image.data[:, :, z]
            window = (float(np.percentile(region, 1)), float(np.percentile(region, 99)))
        show_slice(ax, image, z, window, limits)
        outline(ax, mask, z)
        ax.set_title(style.MODALITY_LABEL[modality])
        if modality == "ct":
            scale_bar(ax, 20.0)
    _letter(
        left,
        "d",
        f"Pillar 3: one patient ({EXAMPLE_PATIENT.replace('_', ' ')}) in three modalities",
    )

    p1 = manifest_cases(root / "pillar1")["p1_reference_noise_m1_n256"]
    image = load(root / "pillar1" / p1["image_path"])
    mask = load(root / "pillar1" / p1["mask_path"])
    discrete = load(root / "pillar1" / p1["discrete_image_path"])
    ivh = load(root / "pillar1" / p1["ivh_image_path"])
    z = axial_index(mask.data)
    limits = crop_limits(mask.data[:, :, z] > 0, mask.spacing, CROP_MARGIN_MM)
    axes = right.subplots(1, 3)
    show_slice(axes[0], image, z, CT_WINDOW, limits)
    outline(axes[0], mask, z)
    axes[0].set_title("Raw image")
    bar = right.colorbar(
        axes[0].images[0], ax=axes[0], location="bottom", shrink=0.9, pad=0.02
    )
    bar.set_ticks(
        list(CT_WINDOW), labels=[f"{CT_WINDOW[0]:.0f}", f"{CT_WINDOW[1]:.0f}"]
    )
    bar.set_label("HU (window)")
    for ax, volume, title, label in (
        (axes[1], discrete, "FBN32 levels", "level"),
        (axes[2], ivh, "IVH grid (FBS1)", "bin"),
    ):
        values = np.ma.masked_where(mask.data == 0, volume.data.astype(float))
        roi = values.compressed()
        shown = Volume(values, volume.spacing)
        ax.imshow(
            shown.data[:, :, z].T,
            origin="lower",
            cmap=_blues(),
            vmin=float(roi.min()),
            vmax=float(roi.max()),
            extent=shown.extent,
            interpolation="nearest",
        )
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        outline(ax, mask, z)
        ax.set_title(title)
        bar = right.colorbar(
            ax.images[0], ax=ax, location="bottom", shrink=0.9, pad=0.02
        )
        bar.set_ticks([float(roi.min()), float(roi.max())])
        bar.set_ticklabels([f"{roi.min():.0f}", f"{roi.max():.0f}"])
        bar.set_label(label)
    _letter(
        right, "e", "Three stored inputs of one case, the same bytes for every package"
    )


def _blues():
    from matplotlib import colormaps

    cmap = colormaps["Blues"].copy()
    cmap.set_bad(style.WHITE)
    return cmap


def _row_grids_and_protocol(subfig, root: Path, inventory: Mapping[str, Any]) -> None:
    left, right = subfig.subfigures(1, 2, width_ratios=[1, 2.1])
    ax = left.subplots()
    p1 = manifest_cases(root / "pillar1").values()
    p2 = manifest_cases(root / "pillar2_a1").values()
    m1 = sorted(
        (case["size"], case["image_voxels"], case["mask_voxels"])
        for case in p1
        if case["case_id"].startswith("p1_reference_noise_m1_")
    )
    a1 = sorted((case["size"], case["mask_voxels"]) for case in p2)
    sizes = [row[0] for row in m1]
    for values, marker, linestyle, label in (
        ([row[1] for row in m1], "s", "-", "image voxels T"),
        ([row[1] for row in a1], "o", (0, (4, 2)), "R, A1 whole anatomy"),
        ([row[2] for row in m1], "^", (0, (1, 1.5)), "R, M1 whole tumour"),
    ):
        ax.plot(
            sizes,
            values,
            marker=marker,
            linestyle=linestyle,
            color=style.INK,
            markerfacecolor=style.WHITE,
            markersize=3.5,
            label=label,
        )
    ax.legend(loc="lower right", handlelength=2.4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([32, 64, 128, 256, 512], ["32", "64", "128", "256", "512"])
    ax.minorticks_off()
    ax.set_xlabel("Grid edge (voxels)")
    ax.set_ylabel("Voxels")
    _letter(left, "f", "Ten grid sizes")

    ax = right.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    steps = (
        ("New\nprocess", False),
        ("Read\nfiles", False),
        ("Crop to\nROI box", False),
        ("Masks,\ndiscretise", False),
        ("1 warmup\n(untimed)", False),
        ("3 timed\ncalls", True),
        ("Median of\n3 calls", False),
    )
    width, gap, y = 11.8, 2.4, 20
    for index, (text, timed) in enumerate(steps):
        x = 1 + index * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                11,
                boxstyle="round,pad=0.3,rounding_size=1.2",
                facecolor=style.WHITE,
                edgecolor=style.INK,
                linewidth=2.0 if timed else 0.8,
            )
        )
        ax.text(x + width / 2, y + 5.5, text, ha="center", va="center")
        if index:
            ax.add_patch(
                FancyArrowPatch(
                    (x - gap + 0.2, y + 5.5),
                    (x - 0.2, y + 5.5),
                    arrowstyle="-|>",
                    mutation_scale=6,
                    color=style.INK,
                    linewidth=0.8,
                )
            )
    first = 1
    last_outside = 1 + 4 * (width + gap) + width
    ax.add_patch(
        Rectangle(
            (first, 11),
            last_outside - first,
            6,
            facecolor=style.WHITE,
            edgecolor=style.INK,
            linewidth=0.6,
            hatch="//",
        )
    )
    ax.text(
        (first + last_outside) / 2,
        14,
        "outside the timer",
        ha="center",
        va="center",
        bbox={"facecolor": style.WHITE, "edgecolor": "none", "pad": 0.8},
    )
    timed_x = 1 + 5 * (width + gap)
    ax.add_patch(
        Rectangle(
            (timed_x, 11),
            width,
            6,
            facecolor="#252525",
            edgecolor=style.INK,
            linewidth=0.6,
        )
    )
    ax.text(
        timed_x + width / 2, 14, "timed", ha="center", va="center", color=style.WHITE
    )
    note = ax.text(
        1,
        4,
        "Timed: ROI voxel extraction and feature calculation, native threads,\n"
        f"at most {inventory['per_call_timeout_seconds']:,.0f} s per call. Three fresh processes per task;\n"
        "reported value: the median of the three process medians.",
        ha="left",
        va="center",
    )
    note.set_in_layout(False)
    _letter(right, "g", "Timing of one task")


def _row_inventory(subfig, inventory: Mapping[str, Any]) -> None:
    ax = subfig.subplots()
    ax.axis("off")
    totals = inventory["totals"]
    pillars = inventory["pillars"]
    text = (
        f"Timing: {totals['case_count']} cases (Pillar 1: {pillars['pillar1_morphology']['case_count']}, "
        f"Pillar 2: {pillars['pillar2_whole_anatomy']['case_count']}, Pillar 3: "
        f"{pillars['pillar3_ibsi2_phase3']['case_count']}) × {inventory['adapter_count']} packages × "
        f"{inventory['family_count']} feature groups × {inventory['fresh_process_repeats']} processes = "
        f"{totals['scheduled_task_records']:,} tasks\n"
        f"({totals['eligible_calculation_tasks']:,} eligible, {totals['preempted_unsupported_tasks']:,} "
        "not supported by PyRadiomics). Accuracy: the IBSI 1 digital phantom and IBSI 2 Phase 1 and "
        "Phase 2,\nrun through the same group calls that the benchmark times."
    )
    ax.text(0.0, 0.5, text, ha="left", va="center", transform=ax.transAxes)
    _letter(subfig, "h", "Inventory")


def build_design_figure(workspace_root: Path, output_dir: Path) -> dict[str, Any]:
    """Write Figure 1 from the frozen workspace."""

    workspace = json.loads(
        (workspace_root / "workspace_manifest.json").read_text(encoding="utf-8")
    )
    inventory = {**workspace["task_inventory"], **workspace["launch_policy"]}
    with style.figure(style.DOUBLE_COLUMN_MM, 236, font_pt=8.0) as fig:
        rows = fig.subfigures(5, 1, height_ratios=[50, 50, 52, 52, 18])
        _row_masks(rows[0], workspace_root)
        _row_noise_and_anatomy(rows[1], workspace_root)
        _row_patient_and_inputs(rows[2], workspace_root)
        _row_grids_and_protocol(rows[3], workspace_root, inventory)
        _row_inventory(rows[4], inventory)
        return style.save(
            fig,
            output_dir / "figure1_study_design",
            caption=_figure1_caption(),
            alt_text=(
                "Study design in eight panels: four synthetic tumour masks, three noise "
                "levels and a whole-anatomy mask on one synthetic CT scene, one patient in "
                "CT, MRI and PET, the three stored inputs of a case, the ten grid sizes, "
                "the timing steps and the task inventory."
            ),
            sources=style.sha256_files([workspace_root / "workspace_manifest.json"]),
        )


def _figure1_caption() -> str:
    return (
        "Study design. a, Pillar 1 masks on the reference-noise scene at the 256³ grid "
        "(1 mm voxels): M1 whole tumour, M2 doughnut with a necrotic cavity, M3 M1 with "
        "18 spicules, M4 M1 with six satellite foci. b, the same crop at the three noise "
        "levels (12, 20 and 30 HU, one seeded field blurred by a 0.8-voxel kernel). c, "
        "Pillar 2 whole-anatomy mask A1. d, Pillar 3 example patient STS 001 in CT, MRI "
        "and PET. e, the three stored inputs of one case: the raw image for intensity and "
        "morphology, the mask-specific FBN32 image for texture and histogram, and the IVH "
        "grid (FBS1 for CT and synthetic cases, FBN1000 for MRI and PET); colour bars show "
        "the range inside the ROI. f, image voxels T and ROI voxels R over the ten grids, "
        "from 32³ to 512³ over a fixed 256 mm field. g, the timing of one task. h, the task "
        "inventory. Outlines are white under black so that they show on dark and bright "
        "tissue. CT window: level 40 HU, width 400 HU; MRI and PET: 1st to 99th percentile "
        "of the slice. Images use the radiological convention (patient right on the image "
        "left)."
    )


def show_plane(ax, volume: Volume, axis: int, index: int, window, limits=None) -> None:
    """Axial (axis 2), coronal (axis 1) or sagittal (axis 0) slice in millimetres.

    Axial and coronal views put the patient's right on the left; the sagittal
    view puts anterior on the left.
    """

    if axis == 2:
        plane, spacing = (
            volume.data[:, :, index],
            (volume.spacing[0], volume.spacing[1]),
        )
    elif axis == 1:
        plane, spacing = (
            volume.data[:, index, :],
            (volume.spacing[0], volume.spacing[2]),
        )
    else:
        plane, spacing = (
            volume.data[index, :, :],
            (volume.spacing[1], volume.spacing[2]),
        )
    extent = (0.0, plane.shape[0] * spacing[0], 0.0, plane.shape[1] * spacing[1])
    ax.imshow(
        plane.T,
        origin="lower",
        cmap="gray",
        vmin=window[0],
        vmax=window[1],
        extent=extent,
        interpolation="nearest",
    )
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def plane_outline(ax, mask: Volume, axis: int, index: int) -> None:
    if axis == 2:
        plane, spacing = mask.data[:, :, index], (mask.spacing[0], mask.spacing[1])
    elif axis == 1:
        plane, spacing = mask.data[:, index, :], (mask.spacing[0], mask.spacing[2])
    else:
        plane, spacing = mask.data[index, :, :], (mask.spacing[1], mask.spacing[2])
    grid_x = (np.arange(plane.shape[0]) + 0.5) * spacing[0]
    grid_y = (np.arange(plane.shape[1]) + 0.5) * spacing[1]
    values = (plane > 0).T.astype(float)
    ax.contour(grid_x, grid_y, values, levels=[0.5], colors=style.WHITE, linewidths=2.0)
    ax.contour(grid_x, grid_y, values, levels=[0.5], colors=style.INK, linewidths=0.9)


def _plane_limits(mask: Volume, axis: int, index: int, margin_mm: float):
    if axis == 2:
        plane, spacing = mask.data[:, :, index], (mask.spacing[0], mask.spacing[1])
    elif axis == 1:
        plane, spacing = mask.data[:, index, :], (mask.spacing[0], mask.spacing[2])
    else:
        plane, spacing = mask.data[index, :, :], (mask.spacing[1], mask.spacing[2])
    return crop_limits(plane > 0, spacing, margin_mm)


def _centre_index(mask: np.ndarray, axis: int) -> int:
    other = tuple(index for index in range(3) if index != axis)
    area = np.count_nonzero(mask, axis=other)
    centre = int(round(float(np.average(np.arange(mask.shape[axis]), weights=area))))
    return (
        centre
        if area[centre]
        else int(np.flatnonzero(area)[np.argmin(np.abs(np.flatnonzero(area) - centre))])
    )


def build_mask_supplement(workspace_root: Path, output_dir: Path) -> dict[str, Any]:
    """Figure S1: the five masks in three planes with their measured descriptors."""

    p1 = manifest_cases(workspace_root / "pillar1")
    p2 = manifest_cases(workspace_root / "pillar2_a1")
    image = load(
        workspace_root / "pillar1" / p1["p1_reference_noise_m1_n256"]["image_path"]
    )
    rows = [
        (key, p1[f"p1_reference_noise_{key.lower()}_n256"], "pillar1")
        for key in MASK_ORDER
    ]
    rows.append(("A1", p2["p2_a1_n256"], "pillar2_a1"))
    with style.figure(style.DOUBLE_COLUMN_MM, 225, font_pt=8.0) as fig:
        grid = fig.subplots(len(rows), 4, width_ratios=[1, 1, 1, 1.05])
        for row_index, (key, case, dataset) in enumerate(rows):
            mask = load(workspace_root / dataset / case["mask_path"])
            for column, (axis, name) in enumerate(
                ((2, "Axial"), (1, "Coronal"), (0, "Sagittal"))
            ):
                ax = grid[row_index, column]
                index = _centre_index(mask.data > 0, axis)
                show_plane(
                    ax,
                    image,
                    axis,
                    index,
                    CT_WINDOW,
                    _plane_limits(mask, axis, index, 10.0),
                )
                plane_outline(ax, mask, axis, index)
                if row_index == 0:
                    ax.set_title(name)
                if column == 0:
                    ax.set_ylabel(style.MASK_LABEL[key])
                if row_index == 0 and column == 0:
                    scale_bar(ax, 20.0)
            text_ax = grid[row_index, 3]
            text_ax.axis("off")
            box = case["bbox_shape_xyz"]
            text_ax.text(
                0.02,
                0.5,
                f"R = {case['mask_voxels']:,} voxels\n"
                f"Bounding box {box[0]} × {box[1]} × {box[2]}\n"
                f"Components: {case['components_26']}\n"
                f"Cavities: {case['cavities_background_6']}\n"
                f"Slices with ROI: {case['occupied_z_slices']}",
                ha="left",
                va="center",
                transform=text_ax.transAxes,
            )
        return style.save(
            fig,
            output_dir / "figureS1_mask_set",
            caption=(
                "The five masks on the reference-noise scene at the 256³ grid (1 mm "
                "voxels), in axial, coronal and sagittal slices through each mask centre. "
                "The text gives the measured descriptors from the frozen manifests: ROI "
                "voxels R, the bounding box in voxels (x × y × z), 26-connected "
                "components, enclosed background cavities and the number of slices with "
                "ROI. CT window: level 40 HU, width 400 HU. Axial and coronal views put "
                "the patient's right on the left; sagittal views put anterior on the left."
            ),
            alt_text=(
                "Grid of CT slices in three planes for five masks, from the whole tumour to "
                "the whole anatomy, each with its outline and its voxel count."
            ),
            sources=style.sha256_files(
                [
                    workspace_root / "pillar1" / "manifest.json",
                    workspace_root / "pillar2_a1" / "manifest.json",
                ]
            ),
        )


def build_noise_supplement(workspace_root: Path, output_dir: Path) -> dict[str, Any]:
    """Figure S2: the noise levels across grids, and zone size against grid."""

    p1 = manifest_cases(workspace_root / "pillar1")
    p3 = manifest_cases(workspace_root / "ibsi2_phase3")
    edges = (32, 64, 128, 256, 512)
    reference_mask = load(
        workspace_root / "pillar1" / p1["p1_reference_noise_m1_n256"]["mask_path"]
    )
    z_mm = (axial_index(reference_mask.data) + 0.5) * reference_mask.spacing[2]
    limits = crop_limits(
        reference_mask.data[:, :, axial_index(reference_mask.data)] > 0,
        reference_mask.spacing,
        CROP_MARGIN_MM,
    )
    with style.figure(style.DOUBLE_COLUMN_MM, 190, font_pt=8.0) as fig:
        top, bottom = fig.subfigures(2, 1, height_ratios=[1.35, 1])
        axes = top.subplots(len(NOISE_ORDER), len(edges))
        for row, noise in enumerate(NOISE_ORDER):
            for column, edge in enumerate(edges):
                case = p1[f"p1_{noise}_m1_n{edge:03d}"]
                image = load(workspace_root / "pillar1" / case["image_path"])
                z = min(image.data.shape[2] - 1, int(z_mm / image.spacing[2]))
                ax = axes[row, column]
                show_slice(ax, image, z, CT_WINDOW, limits)
                if row == 0:
                    ax.set_title(f"{edge}³ ({256 / edge:.2g} mm)")
                if column == 0:
                    ax.set_ylabel(style.NOISE_LABEL[noise])
        _letter(top, "a", "The same physical crop at five of the ten grids")
        ax = bottom.subplots()
        clinical = sorted(
            case["texture_workload"]["mean_voxels_per_zone"] for case in p3.values()
        )
        # Weibull (exclusive) quantiles, the method of the README figures.
        q1, median, q3 = np.percentile(clinical, [25, 50, 75], method="weibull")
        ax.axhspan(
            q1,
            q3,
            facecolor=style.WHITE,
            edgecolor="#737373",
            hatch="/",
            linewidth=0.6,
            zorder=0,
        )
        ax.axhline(median, color=style.REFERENCE_LINE, linewidth=0.8)
        ax.annotate(
            f"clinical cohort: median {median:.1f}, interquartile range {q1:.1f} to {q3:.1f}",
            (0.01, q3),
            xycoords=("axes fraction", "data"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="left",
            va="bottom",
            bbox={"facecolor": style.WHITE, "edgecolor": "none", "pad": 0.6},
        )
        sizes = sorted({case["size"] for case in p1.values()})
        for noise, marker, linestyle in zip(
            NOISE_ORDER, ("o", "s", "^"), ("-", (0, (4, 2)), (0, (1, 1.5)))
        ):
            values = [
                p1[f"p1_{noise}_m1_n{size:03d}"]["texture_workload"][
                    "mean_voxels_per_zone"
                ]
                for size in sizes
            ]
            ax.plot(
                sizes,
                values,
                marker=marker,
                linestyle=linestyle,
                color=style.INK,
                markerfacecolor=style.WHITE,
                label=style.NOISE_LABEL[noise],
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(list(edges), [str(edge) for edge in edges])
        ax.minorticks_off()
        ax.set_yticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
        ax.set_xlabel("Grid edge (voxels)")
        ax.set_ylabel("Voxels per FBN32 zone (M1)")
        ax.legend(loc="lower right")
        _letter(bottom, "b", "Texture zone size against grid size")
        return style.save(
            fig,
            output_dir / "figureS2_noise_levels",
            caption=(
                "a, the reference-noise M1 crop at five of the ten grids for the three "
                "noise levels (12, 20 and 30 HU, one seeded field at three amplitudes, "
                "blurred by a 0.8-voxel kernel). The same physical region and CT window "
                "(level 40 HU, width 400 HU) are used in every image. b, mean voxels per "
                "26-connected FBN32 zone inside M1 at every grid, from the manifests; the "
                "hatched band and the line give the interquartile range and the median of "
                "the 153 clinical cases."
            ),
            alt_text=(
                "Grid of synthetic CT crops at five resolutions and three noise levels, and a "
                "log-log line chart of texture zone size against grid size with the "
                "clinical range as a band."
            ),
            sources=style.sha256_files(
                [
                    workspace_root / "pillar1" / "manifest.json",
                    workspace_root / "ibsi2_phase3" / "manifest.json",
                ]
            ),
        )

# Benchmark design

The benchmark times five radiomics adapters on three input pillars. Each
pillar answers a different question. Analyse each pillar on its own before you
make a combined summary. The timing rules are in
[`timing_method.md`](timing_method.md), and the timed feature groups are in
[`feature_contract.md`](feature_contract.md).

| Pillar | Input | Cases | Question |
| --- | --- | ---: | --- |
| 1. Morphology | One synthetic scene: 3 noise levels × 4 masks (M1 to M4) × 10 cubic grids | 120 | How does runtime change with lesion shape and grid size in one controlled image? |
| 2. Whole anatomy | The same scene at `reference_noise`: mask A1 × 10 cubic grids | 10 | How does runtime scale when most of the image is inside the ROI? |
| 3. Clinical cohort | IBSI 2 Phase 3 soft-tissue sarcoma (STS) cohort: 51 subjects × CT, MRI and PET | 153 | How much do runtime and memory change across real patients, modalities and matrices? |

Pillars 1 and 2 are controlled series. Along one series, only the grid
changes. Pillar 3 is a fixed cohort. It has no cube size and no scaling
ladder.

## Synthetic scene

[`bench/synthetic_volume.py`](../bench/synthetic_volume.py) builds the scene
for Pillars 1 and 2. It never imports a radiomics adapter.

- The field of view is a fixed 256 mm cube in RAS+ space. The grid edge N
  sets the voxel spacing to 256/N mm on all three axes.
- The anatomy is like a thigh: fat, four muscle groups, short fascial arcs,
  and a femur with cortex and marrow.
- A lobulated soft-tissue tumour lies next to the femur. It has a viable rim,
  a necrotic core and six satellite foci.
- Smooth random-wave texture fields give each tissue its texture. The fixed
  seed 20260809 makes every field the same at each build.
- The geometry is in millimetres. Thus every grid shows the same anatomy with
  a different sampling.
- The generator works in slabs of 8 slices, so only the output arrays use
  full-volume memory.

The ten grids come from [`configs/benchmark/pillar1.json`](../configs/benchmark/pillar1.json)
and [`configs/benchmark/pillar2.json`](../configs/benchmark/pillar2.json).
Image voxels rise 4,096-fold from 32³ to 512³.

| Grid N | Spacing (mm) | Image voxels | M1 ROI voxels | A1 ROI voxels |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 8.000 | 32,768 | 3,153 | 18,694 |
| 48 | 5.333 | 110,592 | 10,665 | 61,885 |
| 64 | 4.000 | 262,144 | 25,239 | 149,922 |
| 96 | 2.667 | 884,736 | 85,229 | 505,876 |
| 128 | 2.000 | 2,097,152 | 202,091 | 1,199,081 |
| 192 | 1.333 | 7,077,888 | 681,796 | 4,047,382 |
| 256 | 1.000 | 16,777,216 | 1,615,927 | 9,593,656 |
| 320 | 0.800 | 32,768,000 | 3,156,397 | 18,737,013 |
| 384 | 0.667 | 56,623,104 | 5,454,028 | 32,377,220 |
| 512 | 0.500 | 134,217,728 | 12,928,347 | 76,745,513 |

### Noise levels

The three noise levels share the reference tissue values. They differ only in
per-voxel noise. For each grid, the generator makes the noise in four steps:

1. It draws one seeded Gaussian noise volume for the grid.
2. It blurs the volume with a Gaussian kernel of 0.8 voxel. This kernel
   stands in for the scanner point-spread function.
3. It scales the volume to a standard deviation of 12 HU (`low_noise`),
   20 HU (`reference_noise`) or 30 HU (`high_noise`).
4. It adds the volume to the tissue values, clips the sum to −1,024 to
   2,000 HU and rounds it to `int16`.

The three levels use one noise volume at three amplitudes. Thus they form
pairs by design. They are not independent replicates. Without per-voxel noise,
the smooth texture fields give texture zones that grow strongly with grid
size. The noise keeps the texture work per voxel in the range of the clinical
cases.

## Pillar 1: morphology phantoms

[`bench/pillar1_dataset.py`](../bench/pillar1_dataset.py) builds 3 noise
levels × 4 masks × 10 grids = 120 cases. The masks come from the scene
geometry, so they are the same at all three noise levels. The builder stops if
a mask changes with the noise level.

| Mask | Definition | ROI voxels at 512³ | Bounding box at 512³ | 26-connected components | Enclosed cavities |
| --- | --- | ---: | --- | ---: | ---: |
| M1 | Whole tumour, necrosis included | 12,928,347 | 274 × 238 × 446 | 1 | 0 |
| M2 | Viable shell; the necrotic core is a cavity | 10,934,318 | 274 × 238 × 446 | 1 | 1 |
| M3 | M1 plus 18 attached tapered 3D spicules | 13,307,267 | 350 × 390 × 480 | 1 | 0 |
| M4 | M1 plus six satellite foci | 13,132,139 | 367 × 388 × 446 | 7 | 0 |

The component and cavity counts are the same on all ten grids.

- The four masks at one noise level and grid share the same raw image bytes.
  They form a matched block.
- The masks differ in ROI voxels, bounding box, surface and components. Thus
  a mask contrast is the effect of one defined morphology, not a pure shape
  effect at a fixed ROI size.
- M2 and M4 follow structures in the image: the necrotic core and the
  satellites have their own HU values.
- The M3 spicules have no HU values of their own. They sample the fat and
  muscle next to the tumour. M3 is a boundary stress mask, not a painted
  spiculated tumour.
- The spicules reach fat, so the M3 ROI minimum is lower, the FBN32 bins are
  wider and the zones are larger. For `low_noise` at 512³, the ROI minimum is
  −145 HU for M3 and −80 HU for M1.
- At 512³, M3 has about 1.4 to 1.5 times the voxels per zone of M1. An M3/M1
  texture ratio thus includes this quantization effect.
- The generator keeps only the M3 component that overlaps M1. Thus a coarse
  grid cannot add detached fragments.
- At 32³ (8 mm voxels), the grid shows the spicule tips (radius 2.8 mm) and
  the cavity wall only coarsely. Treat this grid as a coarse overhead point.

Each case stores its measured texture workload under `texture_workload`:

- `zone_count_26`: the number of 26-connected zones of one grey level;
- `mean_voxels_per_zone`: the ROI voxels divided by the zone count;
- `boundary_voxels_6`: the ROI voxels with a 6-neighbour outside the ROI.

Over all Pillar 1 cases, the mean is 1.9 to 64.6 voxels per zone. Below 96³
it is 1.9 to 5.1. For M1 at 512³ it is 43.7, 22.0 and 14.7 for the low,
reference and high noise levels. Zones grow with grid size, so a texture
analysis must use the stored zone count, not only the grid.

## Pillar 2: whole-anatomy A1

[`bench/pillar2_dataset.py`](../bench/pillar2_dataset.py) builds one A1 case
on each of the ten grids.

- A1 covers all anatomy that is not air: fat, muscle, fascia, cortex, marrow
  and the tumour.
- The body radii are the largest values, on a 0.25 mm search grid, that keep
  one background voxel on all six image faces at 32³. The generator fails if
  this guard breaks.
- A1 holds 56.0 percent of the image voxels at 48³ and 57.0 to 57.2 percent
  at all other grids.
- The image is the `reference_noise` scene. Its bytes are equal to the
  Pillar 1 `reference_noise` image of the same grid. Pillar 2 keeps its own
  copy and its own manifest.
- A1 is a dense workload, not a fifth tumour shape. Do not pool it with M1
  to M4.
- From 32³ to 512³, the occupied FBN32 levels rise from 17 to 22 of 32. The
  mean voxels per zone rise from 37 to about 126.

## Pillar 3: IBSI 2 Phase 3 cohort

The IBSI describes Phase 3 as a multi-modality cohort of 51 patients with
soft-tissue sarcoma, for filter-response reproducibility
([IBSI 2](https://theibsi.github.io/ibsi2/)). The files come from The Cancer
Imaging Archive ([IBSI datasets](https://theibsi.github.io/datasets/),
[TCIA DOI](https://doi.org/10.7937/TCIA.2019.b7o0uq20)). This benchmark uses
the cohort for performance only. It makes no IBSI 2 Phase 3 compliance claim.

[`scripts/bootstrap_reproducibility_inputs.py`](../scripts/bootstrap_reproducibility_inputs.py)
fetches the cohort from the `theibsi/data_sets` repository at the commit in
[`reproducibility/inputs/manifest.json`](../reproducibility/inputs/manifest.json).
It checks every byte count and SHA-256 and writes a source manifest in
`data/ibsi2_validation`. [`bench/ibsi2_phase3_dataset.py`](../bench/ibsi2_phase3_dataset.py)
then prepares the workspace copy:

1. It finds each `STS_NNN_<modality>_image.nii.gz` file and its mask.
2. It requires 51 subjects, each with CT, MRI and PET.
3. It checks the file list and every SHA-256 against the source manifest.
4. It copies each image and mask byte for byte and checks the hash of the
   copy.
5. It reads the geometry and the ROI values from the copied files.
6. It writes the FBN32 image and the IVH grid of each case.
7. It writes `preparation.json`, `manifest.json` and `cases.csv`.

The preparation never resamples, crops, resegments or rescales a file. It
gives no case a cube size. The `size` field holds the largest image edge and
serves only to select cases. The manifest keeps the IBSI and TCIA links. Keep
this attribution, the links and the bound checksums in every public bundle.

| Modality | Cases | Image voxels, median (range) | ROI voxels, median (range) | ROI fraction, median (range) | ROI volume in mL, median (range) | Mean voxels per zone, median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CT | 51 | 2,279,290 (856,528 to 7,106,608) | 113,615 (5,564 to 748,026) | 4.52% (0.65 to 13.23%) | 354 (17.4 to 2,333) | 8.6 |
| MRI | 51 | 1,386,000 (103,125 to 6,405,298) | 113,615 (3,856 to 748,026) | 8.82% (0.67 to 22.83%) | 354 (17.3 to 2,333) | 27.0 |
| PET | 51 | 104,017 (36,848 to 387,828) | 4,762 (239 to 30,711) | 4.47% (0.65 to 12.30%) | 353 (17.2 to 2,339) | 4.6 |

- All 153 images are `float32` and all masks are `uint8` with values 0 and 1.
  All images have LPS orientation.
- Each of the 153 cases has its own matrix shape.
- Image voxels span about 193-fold and ROI voxels about 3,130-fold.
- 50 subjects have one 26-connected lesion component. STS_044 has two, in all
  three modalities. No mask has an enclosed background cavity.
- The lesion spans 15 to 82 axial slices (median 36).
- Over all 153 cases, the mean voxels per zone has a median of 9.6
  (interquartile range 4.8 to 26.9, range 1.5 to 218).

The three masks of one subject outline the same lesion at different samplings.
Within a subject, the ratio of the largest to the smallest physical mask
volume has a median of 1.002 and a maximum of 1.018. Thus the subject is a
pairing key, and the three cases of a subject are not independent. Matrix
size, spacing, modality, crop extent, lesion volume and grey levels change
together in this cohort.

## Stored representations

Every case has one binary mask and three images. Each timed group reads
exactly one of them, through
[`bench/benchmark_representations.py`](../bench/benchmark_representations.py).

| Representation | Timed groups | Rule | Data type |
| --- | --- | --- | --- |
| Raw image | `group_intensity`, `group_local_intensity`, `group_morphology`, `group_spatial_morphology` | Synthetic: the generated HU (`original_hu_int16`). Clinical: the released values. | `int16` or `float32` |
| FBN32 image (`mask_specific_ibsi_fbn32`) | `group_texture` | IBSI fixed bin number, 32 levels over the exact ROI minimum and maximum | `uint8` |
| IVH grid FBS1 (`mask_specific_ibsi_fbs1_ivh_indices`) | `group_ivh` for synthetic and CT cases | IBSI fixed bin size, width 1, anchor at the ROI minimum | `uint8` or `uint16` |
| IVH grid FBN1000 (`mask_specific_ibsi_fbn1000_ivh_indices`) | `group_ivh` for MRI and PET cases | IBSI fixed bin number, 1,000 levels over the exact ROI | `uint16` |

[`bench/synthetic_representations.py`](../bench/synthetic_representations.py)
compiles these grids for all three pillars. In the rules below, x is an ROI
value, a is the ROI minimum and b is the ROI maximum.

- The FBN32 level is `min(32, floor(32 (x − a) / (b − a)) + 1)`. The ROI
  maximum always gets level 32.
- The FBN1000 level uses the same rule with 1,000 levels.
- The FBS1 index is `floor(x − a) + 1`.

These rules apply to all grids:

- Every grid holds levels from 1 upwards inside the ROI and 0 outside it.
- The compiler runs once per case, before any timing. All five adapters read
  the same bytes and apply no re-binning.
- FBN depends on the ROI, so each mask has its own FBN32 image. One global
  discretised image cannot replace it.
- A constant ROI fails, because FBN has no definition for it.
- IVH never reads the FBN32 image. The FBS1 record keeps the map back to HU:
  value = anchor + (index − 1) × width.
- Each case records the configured levels (32, 1,000, or the largest FBS1
  index) and the occupied levels G inside the ROI. G can be below the
  configured count.
- Each case also records the ROI minimum and maximum of every grid.

## Build and validation

[`scripts/prepare_workspace.py`](../scripts/prepare_workspace.py) builds or
resumes all three pillars in `data/benchmark` and then validates them through
[`bench/benchmark_workspace.py`](../bench/benchmark_workspace.py). With
`--validate-only`, it validates the files on disk again. No step runs a
radiomics calculation.

[`bench/dataset_manifest.py`](../bench/dataset_manifest.py) checks every
pillar:

- Every file matches the byte count and SHA-256 in the manifest.
- Image and mask have the same 3D shape, spacing and affine. Two active
  transforms (qform and sform) must agree.
- Each mask holds only 0 and 1 and has at least one ROI voxel.
- Every image voxel is finite, because local intensity reads voxels outside
  the ROI.
- The declared shape, voxel counts, orientation, spacing and affine match the
  files.
- A synthetic case has the modality `synthetic` and a cubic shape whose edge
  equals its size.
- The generation or preparation record matches its SHA-256 in the manifest.

The pillar builders add these checks:

- Pillar 1: the full noise × mask × grid set exists, the file role counts
  match, and the raw images are `int16`. Every FBN32 image and FBS1 grid
  matches a fresh compile byte for byte, with the same occupied levels and ROI
  range.
- Pillar 2: every FBN32 image and FBS1 grid matches a fresh compile, and A1
  keeps at least one background voxel on every face.
- Pillar 3: all 51 CT, MRI and PET blocks are complete, and each image and
  mask hash equals its source hash. Every FBN32 image, FBS1 grid and FBN1000
  grid matches a fresh compile.

The generation and preparation records hold the SHA-256 of the builder source
files. A resume fails when the profile, the source manifest or the builder
code differs.

The last step writes `data/benchmark/workspace_manifest.json` with the status
`inputs_validated_calculations_not_started`. It binds the three dataset
manifests, the three timing contracts, the workspace source files and the task
inventory. [`scripts/launch_benchmark.py`](../scripts/launch_benchmark.py)
refuses to start when a workspace source file or a dataset manifest differs.

## Case and task counts

A task is one adapter, one case, one timed group and one repeat. Each case
gets 5 adapters × 6 groups × 3 repeats = 90 scheduled records. PyRadiomics
has no `group_local_intensity`, no `group_spatial_morphology` and no
`group_ivh`. Thus 9 records of each case are unsupported, and the controller
records them without a calculation.

| Pillar | Cases | Stored files | Scheduled records | Eligible tasks | Unsupported records |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. Morphology | 120 | 310 | 10,800 | 9,720 | 1,080 |
| 2. Whole anatomy | 10 | 40 | 900 | 810 | 90 |
| 3. Clinical cohort | 153 | 612 | 13,770 | 12,393 | 1,377 |
| Total | 283 | 962 | 25,470 | 22,923 | 2,547 |

Pillar 1 stores 30 raw images, 40 masks, 120 FBN32 images and 120 FBS1 grids.
Pillar 2 stores 10 of each. Pillar 3 stores 153 images, 153 masks, 153 FBN32
images, 51 FBS1 grids and 102 FBN1000 grids. If every eligible task finishes,
the main run makes 22,923 discarded warmup calls and 68,769 measured calls.

## What each pillar lets a reader conclude

Pillar 1:

- It gives paired M2/M1, M3/M1 and M4/M1 runtime ratios at each grid and noise
  level, within one adapter and group.
- It gives runtime and memory against the measured ROI voxels, bounding box,
  boundary voxels, components, occupied levels and zone count.
- It shows how the mask effect changes with the grid.
- It cannot separate shape from ROI size. Its noise levels form pairs, not
  replicates.

Pillar 2:

- It gives the cleanest dense scaling curve: image voxels rise 4,096-fold with
  about 57 percent of them in the ROI.
- Model the scaling against the measured A1 ROI voxels and bounding box, not
  against an assumed fraction.
- It shows one synthetic scene. It does not show clinical variation.

Pillar 3:

- It gives the spread of runtime and memory across real patients, for each
  modality, and paired CT, MRI and PET comparisons within each subject.
- It shows associations with image voxels, ROI voxels, bounding box, spacing
  and grey levels. These values change together, so a cohort regression is
  not a scaling law.
- Keep the three cases of a subject together in any model or resampling.
- Each case is its own skip branch. A result from one patient never censors
  another patient.

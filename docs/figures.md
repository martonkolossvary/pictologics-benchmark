# Manuscript figures and tables

[`bench/figures`](../bench/figures) draws every manuscript figure and every
supplementary table. It reads finished outputs and never runs a library. It
opens the run ledgers read-only, so it also works during a timing run.

## Build the figures

```bash
poetry run python -m bench.figures.build \
  --output-dir <result-root>/<machine-id>/figures \
  --compliance-root <result-root>/<machine-id>/compliance \
  --workspace-root data/benchmark \
  --runs <result-root>/<machine-id>/pillar1_morphology \
         <result-root>/<machine-id>/pillar2_whole_anatomy \
         <result-root>/<machine-id>/pillar3_ibsi2_phase3
```

[`bench/figures/build.py`](../bench/figures/build.py) has these options:

| Option | Default | Use |
| --- | --- | --- |
| `--output-dir` | none (required) | Folder for the figures, tables and renders. |
| `--compliance-root` | `results/compliance` | Folder with `ibsi1`, `ibsi2-phase1` and `ibsi2-phase2`. The build reads the `report` folder of each. |
| `--workspace-root` | `data/benchmark` | Frozen benchmark workspace for Figures 1, S1 and S2. |
| `--contract` | `configs/benchmark/main.json` | Benchmark contract for Table S2. |
| `--runs` | none | Run folders of the three pillars. Without them, the build draws no timing figure. |
| `--synthetic-size` | `128` | Grid edge of the Pillar 1 and Pillar 2 rows in Figure 5. |

The build always needs the three compliance reports and the workspace. A
timing figure appears only when the run records hold its data. For example,
Figures 4, 6 and S6 need Pillar 2, S3 needs Pillar 1, and S4 needs Pillar 3.

The one-command runner [`bench/reproduce/stages.py`](../bench/reproduce/stages.py)
runs the build in its "Draw the figures" stage:

- It writes to `<result-root>/<machine-id>/figures`.
- A quick run writes to `<result-root>/<machine-id>/quick/figures` and uses `--synthetic-size 64`.
- When Arial is missing, the stage draws no figure and prints a note.

Each run report of `bench.cli report` also draws the timing figures of its
run. [`bench/report.py`](../bench/report.py) calls the same functions.

The build writes these files:

```text
figures/
  <figure-id>.pdf  <figure-id>.svg  <figure-id>.png  <figure-id>.json
  tables/tableS1_ibsi1_coverage.csv ... tables/tableS5_loglog_slopes.csv
  tables/data_task_records.csv   every task record, with each timed call
  tables/data_cells.csv          one row per library, group and case: repeats, medians, memory, bounds
  tables/data_ratio_cases.csv    the Figure 5 ratio of every case, and its kind
  tables/data_ratio_summary.csv  the Figure 5 markers and lines
  accessibility/<figure-id>_greyscale.png
  accessibility/<figure-id>_deuteranopia.png
  accessibility/<figure-id>_protanopia.png
  accessibility/<figure-id>_tritanopia.png
  figures_manifest.json      the sidecar records of all figures
```

## Main figures

| Figure | File stem | Size (mm) | Data |
| --- | --- | --- | --- |
| 1 | `figure1_study_design` | 183 × 236 | Workspace: `workspace_manifest.json`, and the manifests and NIfTI files of `pillar1`, `pillar2_a1` and `ibsi2_phase3` |
| 2 | `figure2_conformance_outcomes` | 183 × 118 | `ibsi1/report/compliance_summary.csv`, `ibsi2-phase1/report/phase1_strict_3d_summary.csv`, `ibsi2-phase2/report/compliance_phase2_b_summary.csv` |
| 3 | `figure3_conformance_by_family` | 183 × 205 | `ibsi1/report/compliance_by_family.csv`, `ibsi2-phase1/report/phase1_strict_3d_by_filter.csv`, `ibsi2-phase2/report/compliance_phase2_b_by_configuration.csv` |
| 4 | `figure4_runtime_scaling` | 183 × 165 | Pillar 2 run records |
| 5 | `figure5_runtime_ratio` | 183 wide; 30 mm plus 5.2 mm for each row | Run records of all three pillars |
| 6 | `figure6_memory_scaling` | 183 × 165 | Pillar 2 run records |

[`design.py`](../bench/figures/design.py) draws Figure 1.
[`accuracy.py`](../bench/figures/accuracy.py) draws Figures 2 and 3.
[`performance.py`](../bench/figures/performance.py) draws Figures 4 to 6.

### Figure 1: study design

- **a.** The four tumour masks M1 to M4 on one synthetic CT scene, at the 256³ grid with reference noise.
- **b.** The M1 crop at the three noise levels.
- **c.** The Pillar 2 whole-anatomy mask A1.
- **d.** Patient STS 001 of Pillar 3 in CT, MRI and PET.
- **e.** The three stored inputs of one case: the raw image, the FBN32 levels and the IVH grid.
- **f.** Image voxels T and ROI voxels R against the grid edge, from 32 to 512.
- **g.** The timing steps of one task.
- **h.** The task inventory from `workspace_manifest.json`.

The CT window is level 40 HU and width 400 HU. MRI and PET use the 1st to
99th percentile of the slice. Images use the radiological convention.

### Figure 2: conformance outcomes

- Three panels of horizontal stacked bars, with one bar for each library.
- **a.** IBSI 1, 169 standardized checks. **b.** IBSI 2 Phase 1, 24 standardized 3D response maps. **c.** IBSI 2 Phase 2, 161 standardized configuration-B checks.
- Each bar goes from 0 to 100% of the fixed denominator.
- The segments are passed, failed (F), not supported (NS) and not evaluated (NE). NE counts missing, non-finite and ambiguous IBSI 1 values.
- A segment of 12% or more shows its count inside. The text at the bar end gives passed / total and the counts of small segments.
- The row labels give the library name and version.

### Figure 3: conformance by family

- Three matrices. **a.** IBSI 1, 11 Table 2 families. **b.** Phase 1, seven 3D filter families. **c.** Phase 2, the nine published B configurations.
- Each cell shows the passed count. A second line gives the failures ("n F") and the unsupported checks ("n NS").
- Each column header gives the family and its denominator, for example "n = 25".
- In panel c, a dagger marks `5.B`. This configuration averages 2D Gabor responses over three orthogonal planes.

### Figure 4: runtime scaling

- Pillar 2, mask A1. Six group panels and one key panel.
- x: ROI voxels R, log scale. y: wall time per call, log scale, with the same range in every panel.
- Large marker: the median of the three process medians, joined by the library line. Small points: the three process medians.
- The median appears only when all three repeats are measured. Otherwise, open markers show the measured repeats.
- A timeout appears as an open marker at its lower bound, with an up arrow.
- A strip above each panel gives a letter for each other outcome. "Not implemented" fills the strip row of a library without the group.
- A labelled line marks the time limit per call. A label row above the axes gives the grid edge.

### Figure 5: runtime relative to Pictologics

- One panel for each group. One row for each library and data set.
- Data sets: Pillar 1 at the synthetic size, one row for each noise level; Pillar 2 at the synthetic size; and Pillar 3 CT, MRI and PET. The noise levels are paired, not replicates, so the figure never pools them.
- Each case gives the ratio of the library median to the Pictologics median.
- Marker: the geometric mean of the ratios.
- Line for Pillar 1 and Pillar 2: the range over the cases. The four masks of one noise level share one synthetic scene, so they are not independent cases.
- Line for Pillar 3: the 95% bootstrap interval over patients, from 2,000 resamples. With fewer than four patients, the line is the range.
- A timed-out case of the library adds its lower bound. The marker is then open, with "≥".
- A timed-out Pictologics case adds an upper bound. The marker is then open, with "≤". Thus a timeout counts the same way on both sides.
- A marker with both kinds of bound is open, without a sign.
- A case is not estimable (NE) when both tasks were censored, or when Pictologics stopped without a time bound (a memory stop). A row without a value shows its outcome letter.
- `data_ratio_cases.csv` gives the ratio and its kind (`point`, `lower_bound`, `upper_bound`, `not_estimable`, `no_package_value`) for every case.
- A reference line marks a ratio of 1.

### Figure 6: peak memory

- The same layout as Figure 4.
- y: the peak memory in use of the whole adapter process tree during the task, the median of three processes, in MB. This is the measure of the memory stop. A task too short for one memory sample uses its peak RSS.
- Each package prepares a different amount of data before its first call. Thus the increase over the worker-ready memory is not comparable across packages. `data_cells.csv` keeps it as `extra_memory_mb`, beside `calculation_peak_mb`.
- A labelled line marks the memory budget. The strip gives timeouts (T), memory stops (M) and preflight skips (P).

## Supplementary figures

| Figure | File stem | Size (mm) | Content | Data |
| --- | --- | --- | --- | --- |
| S1 | `figureS1_mask_set` | 183 × 225 | The five masks M1 to M4 and A1 at the 256³ grid, in axial, coronal and sagittal slices. Text gives R, bounding box, components, cavities and slices with ROI. | `pillar1` and `pillar2_a1` manifests and NIfTI files |
| S2 | `figureS2_noise_levels` | 183 × 190 | a: the M1 crop at five grids and three noise levels. b: mean voxels per FBN32 zone in M1 against grid edge. A hatched band and a line give the interquartile range and the median of the 153 clinical cases. | `texture_workload` in the `pillar1` and `ibsi2_phase3` manifests |
| S3 | `figureS3_mask_effect` | 183 wide; 20 mm plus 42 mm for each mask | Runtime of M2, M3 and M4 relative to M1 on the same scene, noise level and grid. Marker: geometric mean over the noise levels. Vertical line: range. | Pillar 1 records |
| S4 | `figureS4_patient_runtime` | 183 wide; 20 mm plus 40 mm for each group | Runtime per patient against R, for each group and modality. Points only. Open markers are timeouts at their lower bound. | Pillar 3 records |
| S5 | `figureS5_parallel_work` | 183 × 120 | CPU time divided by wall time against R. Labelled lines mark one core and the logical core count. Open markers are calls below 10 ms. Only longer calls set the axis, so a short call above it sits at its top edge. | All records; core count from `run_spec.json` |
| S6 | `figureS6_ratio_by_size` | 183 × 120 | Runtime relative to Pictologics at every Pillar 2 grid. Vertical line: range of the three repeat ratios. | Pillar 2 records |
| S7 | `figureS7_outcome_map` | 183 wide; 30 mm plus 4.4 mm for each row | One tile for each library, group and grid edge, in Pillar 1 (pooled) and Pillar 2. | Pillar 1 and Pillar 2 records |
| S8 | `figureS8_fixed_costs` | 183 × 62 | Worker memory before the first call, load time and setup time. Small points: tasks. Large marker: median. | All measured tasks |
| S9 | `figureS9_error_relative_to_tolerance` | 183 × 84 | Error divided by tolerance for every evaluated check. a: Phase 1 3D maps. b: Phase 2 configuration B. | `ibsi2-phase1/report/phase1_detail.csv`, `ibsi2-phase2/report/compliance_detail.csv` |

[`design.py`](../bench/figures/design.py) draws S1 and S2.
[`performance.py`](../bench/figures/performance.py) draws S3 to S8.
[`accuracy.py`](../bench/figures/accuracy.py) draws S9.

Figure S9 uses a log x axis with a line at 1. Exact zeros cannot sit on a log
axis, so a column at the left counts them. A zero-tolerance check with a
positive error has no finite ratio, so the panel counts it at the right. No
value is clipped to a floor.

## Supplementary tables

| Table | File in `tables/` | Content | Data |
| --- | --- | --- | --- |
| S1 | `tableS1_ibsi1_coverage.csv` | IBSI 1 coverage for each library: version, finite definitions of 172, finite instances of 174, the 169 checks, evaluated and passed. | `ibsi1/report/compliance_summary.csv` |
| S2 | `tableS2_native_outputs_per_group.csv` | Native output count for each library and timed group. Runtime is never divided by these counts. | `adapter_inventories` in [`configs/benchmark/main.json`](../configs/benchmark/main.json) |
| S3 | `tableS3_ibsi1_failures.csv` | Every failed IBSI 1 check: library, feature, IBSI code, value, reference, tolerance and comparison error. | `ibsi1/report/compliance_detail.csv` |
| S4 | `tableS4_secondary_designs.csv` | The secondary designs: all 36 Phase 1 tests, the Phase 2 A pathway, the B pathway, and configurations 10 and 11. Columns: defined, supported, standardized, evaluated, passed. | `ibsi2-phase1/report/phase1_summary.csv`, `ibsi2-phase2/report/compliance_by_configuration.csv` |
| S5 | `tableS5_loglog_slopes.csv` | The slope b of log time against log R for each library and group, with a 95% interval: one fit on the sizes common to all libraries, and one fit that counts timeouts. | Pillar 2 records |

- Table S4 keeps "no reference" apart from "not supported". Configurations 10 and 11 have no published reference, so they cannot pass.
- Table S5 leaves out 32³, a coarse-grid point.
- The primary fit (`slope`) uses, for every library of a group, the same grids: the grids from 64³ where all libraries of the group have a complete median. A fit needs three grids or more. The interval uses the t distribution.
- The censored fit (`slope_censored`) uses every grid from 64³. A timeout counts as at least its time bound (a Tobit model on log time). Its interval is ± 1.96 standard errors from the numerical Hessian.
- A memory stop or a skip has no time bound, so it stays out of both fits. `n_timeouts` gives the timeouts in the censored fit.

## Community figure

[`bench/figures/community.py`](../bench/figures/community.py) draws
`figure_community_runtime_ratio`. The build command does not run it. Run it
when machines send their results:

```bash
poetry run python -m bench.figures.community \
  --submissions submissions --output-dir <folder>
```

- It reads the newest package of each machine in `submissions/<machine-id>/<date>/`.
- Each machine gives its own ratios to Pictologics, on its own hardware. The figure never pools absolute times across machines.
- It uses Pillar 3 CT and the Figure 5 layout, with one row for each library and machine.
- It prints "no submitted packages" when the folder holds no package.

## Compliance report figures

Each compliance report draws its own figures with the panel functions of
[`accuracy.py`](../bench/figures/accuracy.py). Thus each figure type has one
source.

| File stem | Size (mm) | Content |
| --- | --- | --- |
| `figure_conformance_outcomes` | 183 × 52 | The Figure 2 bars for one specification |
| `figure_conformance_by_family` | 183 × 80 | The Figure 3 matrix for one specification |
| `figure_error_relative_to_tolerance` | 101.5 × 80 | The Figure S9 strip plot, for Phase 1 and Phase 2 only |

## Shared style

[`bench/figures/style.py`](../bench/figures/style.py) holds every shared
choice. No figure sets its own colours, names, fonts or sizes.

### Text, size and files

- **Font.** Arial. `require_font` stops the build when Arial is missing, so no figure changes its font silently.
- **Text size.** At least 8 pt at final size. `rc_params` rejects a smaller size. The conformance figures use 8.5 pt. The other figures use 8 pt.
- **Panel letters.** Bold lowercase letters, 2 pt larger than the body text. No other text is bold.
- **Width.** The style defines 89 mm for one column and 183 mm for two columns. Every manuscript figure uses 183 mm. The height fits the content.
- **Axes.** Lines of 0.8 pt, no top or right spine, white background.
- **Log axes.** Most log axes label only powers of ten. A narrow axis also labels 2 and 5, and a grid axis labels the grid edges.
- **Text colour.** Black, or dark grey `#404040` (10.4:1 on white). Text never uses a library colour.
- **Images.** Each image caption gives the window. Outlines are white under black, so they show on dark and bright tissue.

`style.save` writes each figure in four files:

1. a PDF with embedded TrueType fonts;
2. an SVG with editable text;
3. a PNG at 600 dpi;
4. a JSON sidecar with the figure ID, the file names, the size in mm, the caption, the alt text and the input checksums.

The save removes creation dates and software names from the file metadata.
Figures 1 to 3, S1, S2 and S9 record the SHA-256 of their input tables or
manifests. The timing figures read the run records and record no input
checksum.

### Libraries

| Order | Library | Colour | Contrast on white | Marker | Line | Hatch |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | Pictologics | `#0072B2` | 5.19:1 | circle | solid | none |
| 2 | PyRadiomics | `#EE6677` | 3.09:1 | square | dashed | `//` |
| 3 | MIRP | `#009E73` | 3.42:1 | triangle | dotted | `\\` |
| 4 | MEDimage | `#AA3377` | 6.09:1 | diamond | dash-dot | `xx` |
| 5 | Z-Rad | `#8C510A` | 6.37:1 | filled plus | long dash | `..` |

Every figure uses this order and these names.

### Outcomes

Outcome fills never use a library colour. Every fill has a black outline of
0.8 pt.

| Outcome | Code | Fill | Pattern | Text on fill |
| --- | --- | --- | --- | --- |
| Passed | none | `#252525` | none | white, 15.3:1 |
| Failed | F | `#858585` | `xx` in black | black, 5.7:1 |
| Not supported | NS | white | `//` in `#737373` | black |
| Not evaluated | NE | `#D9D9D9` | none | black, 14.9:1 |

The passed and failed fills differ by 4.15:1. The matrices of Figure 3 and of
the reports use four steps:

| Cell | Fill | Text |
| --- | --- | --- |
| All checks passed | `#252525` | white |
| 50 to 99% passed | `#858585` | black |
| Below 50% passed | white | black |
| Not supported (no check evaluated) | white with `//` hatch | black "NS" |

The timing figures use letters for an outcome without a value:

| Letter | Meaning |
| --- | --- |
| T | timed out |
| M | memory stop |
| P | memory preflight skip |
| S | skipped because of a stop in the same branch |
| U | not implemented |
| F | failed |
| NE | not estimable (Figure 5): the Pictologics task was censored |

No censored task receives a value. The tiles of Figure S7 also have their own
fill and hatch for each letter.

### Reference lines and regions

- Reference lines are 0.8 pt solid `#404040`. They never use a library line style.
- A direct text label names each reference line, except the line at 1 in Figures 5 and S3.
- A region is a hatched fill with an outline, never a tint alone. An example is the clinical band in Figure S2.

### Names

- Groups: Intensity, Morphology, Spatial morphology, Texture, IVH, in this order.
- Noise levels: Low noise (12 HU), Reference noise (20 HU), High noise (30 HU).
- Masks: M1 whole tumour, M2 doughnut, M3 spiculated, M4 multifocal, A1 whole anatomy.
- Modalities: CT, MRI, PET.
- Variables: T for image voxels, R for ROI voxels.

## Automatic checks

`style.save` calls `check_figure` from
[`bench/figures/checks.py`](../bench/figures/checks.py) before it writes a
file. A problem stops the save with an error. The check finds:

1. text smaller than 8 pt;
2. text that goes more than 1 pixel outside the figure;
3. text that overlaps other text by more than 1 pixel in both directions.

The check ignores tick labels outside the axis range, because they are not
drawn. It does not test text against data marks or cells. Look for these
problems on the renders.

[`tests/test_figure_style.py`](../tests/test_figure_style.py) tests the style:

- each library has its own marker, line style, hatch and colour;
- each library colour has a contrast of 3:1 or more on white;
- each outcome text has a contrast of 4.5:1 or more on its fill;
- the passed and failed fills differ by 3:1 or more, and so do the failed fill and white;
- `rc_params` rejects text of 7 pt;
- `check_figure` finds small text and overlapping text;
- `save` writes the four files, and `accessibility_views` writes the four renders.

## Accessibility

After the build draws all figures, `accessibility_views` writes four renders
of each PNG into `accessibility/`:

- a greyscale render, from the relative luminance in linear RGB;
- simulated deuteranopia, protanopia and tritanopia, from the matrices of Machado, Oliveira and Fernandes (2009) at severity 1.0.

Use the renders to make sure that each library and each outcome can still be
named.

- **Libraries.** Identity never depends on colour alone. Each library has its own marker, line style and hatch. The bars and matrices also name it in the row label.
- **Greyscale.** Some library colours are almost equal in greyscale, down to 1.05:1. The other cues carry the identity.
- **Outcomes.** Each outcome has its own tone, pattern and letter code.
- **Captions.** A caption names each series by its marker, line, pattern or letter, never by its colour alone. For example, the Figure 2 caption names "mid-grey cross-hatched segments (F)".
- **Alt text.** The sidecar of each figure holds its alt text.

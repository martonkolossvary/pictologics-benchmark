# IBSI conformance checks

This repository checks five radiomics libraries against two IBSI standards.
IBSI 1 checks feature values on the digital phantom. IBSI 2 Phase 1 checks
filter response maps, and IBSI 2 Phase 2 checks feature values on response maps.

The checks have no timer and run apart from the timing benchmark. They save
every result with a SHA-256 checksum and resume after a stop. The results
compare software. They are not an IBSI certification.

## Libraries

The profiles in [`configs/adapters`](../configs/adapters) set the library
versions. All results in this document are for these versions.

| Library | Distribution | Version | Python |
| --- | --- | ---: | ---: |
| Pictologics | `pictologics` | 1.0.0 | 3.12 |
| PyRadiomics | `pyradiomics` | 3.1.0 | 3.11 |
| MIRP | `mirp` | 2.7.0 | 3.12 |
| MEDimage | `medimage-pkg` | 0.9.8 | 3.10 |
| Z-Rad | `z-rad` | 26.8.0 | 3.12 |

The PyRadiomics 3.1.0 release reports the metadata version `3.0.1a1`. The
reports show the release version, and the mapping audit keeps the metadata
version.

## Reference inputs

[`reproducibility/inputs/manifest.json`](../reproducibility/inputs/manifest.json)
pins every input file by path, byte count and SHA-256.
[`bench/compliance/references.py`](../bench/compliance/references.py) holds
the same checksums. A compliance command stops when a checksum does not match.

| Input | Source | SHA-256 |
| --- | --- | --- |
| IBSI 1 workbook | <https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx> | `3af20577b580f2161bd546e147338ff226abb1676488e2906f2ce908b7985710` |
| IBSI 1 phantom image | `theibsi/data_sets` | `83773ac2a288aa93cf819a98eaf301c18d35d5876182ec9c18733184e2b7a83a` |
| IBSI 1 phantom mask | `theibsi/data_sets` | `3032d340944b577b83bc559ea7ae9a6034b84633a840eff39570ec17aa1d6281` |
| IBSI 2 Phase 1 phantoms (nine images) | `theibsi/data_sets` | one value for each image in `references.py` |
| IBSI 2 Phase 1 phantom masks (one identical file) | `theibsi/data_sets` | `f169b2fced49fe23eee6c09914d9ad07e0aba835b46f2c74caa99a609996fc16` |
| IBSI 2 Phase 1 reference maps (33) | `theibsi/ibsi_2_reference_data` | one value for each map in `references.py` |
| IBSI 2 Phase 2 PAT1 CT image | `theibsi/data_sets` | `909129fdf99d5f5fd17bc568b1834d31edce8663ea6746c6020720bfc1d7bbcb` |
| IBSI 2 Phase 2 PAT1 CT mask | `theibsi/data_sets` | `0e9243e473b1255c60e6460df465d0b07b65436d6e0c691fe3d17bca0594ae78` |
| IBSI 2 Phase 2 reference values | `theibsi/ibsi_2_reference_data` | `6d1a4e7b297e5e9d79ce54471e81f6f84e202fafe0322df1258ac834ce95dd45` |

The pinned repository commits are:

- `theibsi/data_sets`: `6da96021bc91faf4c0cb7fd7fa56a4225d2064a8`;
- `theibsi/ibsi_2_reference_data`: `5404579fb3e0d17e8db421f0e82d64ce2432ec03`;
- `theibsi/ibsi_2_data_analysis`: `fde70ca61819799f673c07dca776deea065d2a7d`.

The code records the analysis commit and the SHA-256 of its Phase 2 script,
`results/analysis_with_tolerance.R`. The IBSI 2 protocol follows the
[IBSI 2 reference manual, version 9](https://arxiv.org/html/2006.05470v9).

To get and check the inputs, run these two commands. The second command
changes nothing.

```bash
poetry run python scripts/bootstrap_reproducibility_inputs.py
poetry run python scripts/bootstrap_reproducibility_inputs.py --verify-only
```

## Commands

Every compliance command is a subcommand of [`bench/cli.py`](../bench/cli.py):
`poetry run python -m bench.cli compliance <subcommand>`.

| Subcommand | Action | Code |
| --- | --- | --- |
| `import-ibsi1` | Reads the six-sheet workbook. Writes `ibsi1_references.csv` and `ibsi1_reference_manifest.json`. | [`references.py`](../bench/compliance/references.py) |
| `validate-ibsi2-phase1` | Checks the 33 reference maps against their checksums. Writes `reference_manifest.json`. | [`references.py`](../bench/compliance/references.py) |
| `import-ibsi2-phase2` | Reads the Phase 2 reference CSV. Writes `ibsi2_phase2_references.csv` and `ibsi2_phase2_reference_manifest.json`. | [`references.py`](../bench/compliance/references.py) |
| `run-ibsi1` | Calculates and compares the IBSI 1 features of every library. | [`run.py`](../bench/compliance/run.py), [`evaluate.py`](../bench/compliance/evaluate.py) |
| `generate-ibsi2-candidates` | Makes the Phase 1 and Phase 2 response maps of every library. | [`ibsi2_candidates.py`](../bench/compliance/ibsi2_candidates.py), [`ibsi2_native_backends.py`](../bench/compliance/ibsi2_native_backends.py) |
| `evaluate-ibsi2-phase1` | Compares the Phase 1 response maps with the reference maps. | [`evaluate.py`](../bench/compliance/evaluate.py) |
| `run-ibsi2-phase2` | Calculates the 18 statistics on each Phase 2 response map and compares them. | [`run.py`](../bench/compliance/run.py) |
| `report` | Writes the tables and figures again from a saved `comparisons.csv`. | [`report.py`](../bench/compliance/report.py) |

The candidate generator stops at the first library that fails a filter. The
MEDimage Gabor filter of Phase 2 configuration 5.B holds several 2.5 GiB
arrays, so it needs more than 16 GiB of memory. On a computer with less
memory, the one-command runner reports the failure, keeps the IBSI 1 results
and runs the timing benchmark. The next run tries IBSI 2 again. On Linux, the
runner limits each IBSI 2 process to three quarters of the memory
(`RLIMIT_AS`). Without this limit, Linux stops processes without an error when
memory runs out, and can stop the whole computer.

- `--adapters` selects libraries. The default is all five.
- `--resume` continues `run-ibsi1`, `generate-ibsi2-candidates` and `run-ibsi2-phase2` after a stop.
- `evaluate-ibsi2-phase1` has no resume. It needs an empty output folder.
- `--timeout` limits one calculation call in seconds. The candidate generator uses 1,800 s by default.
- `--no-report` skips the report of `run-ibsi1` and `run-ibsi2-phase2`.
- The importers accept only the reviewed checksums. Use `--allow-unknown-hash` for development only, never for published results.

### Run the checks

The one-command runner [`bench/reproduce/stages.py`](../bench/reproduce/stages.py)
imports the references in its "Import the IBSI reference values" stage. Its
"Check IBSI conformance" stage runs the checks. It records each stage as
complete and skips it on the next run. It runs the Phase 1 evaluation only
when `ibsi2-phase1/report` does not exist.

To run the same steps by hand, set `ROOT` to `<result-root>/<machine-id>/compliance`.
Then do these steps:

1. Import the references.

   ```bash
   poetry run python -m bench.cli compliance import-ibsi1 \
     --workbook data/downloads/IBSI-1-submission-table.xlsx \
     --output-dir data/ibsi1/references
   poetry run python -m bench.cli compliance validate-ibsi2-phase1 \
     --reference-dir data/ibsi2/references/phase1/maps \
     --manifest-out data/ibsi2/references/phase1/reference_manifest.json
   poetry run python -m bench.cli compliance import-ibsi2-phase2 \
     --csv data/ibsi2/references/phase2/source/reference_values.csv \
     --output-dir data/ibsi2/references/phase2
   ```

2. Run IBSI 1.

   ```bash
   poetry run python -m bench.cli compliance run-ibsi1 \
     --image data/ibsi1/digital_phantom/image/phantom.nii.gz \
     --mask data/ibsi1/digital_phantom/mask/mask.nii.gz \
     --references data/ibsi1/references/ibsi1_references.csv \
     --reference-manifest data/ibsi1/references/ibsi1_reference_manifest.json \
     --output-dir "$ROOT/ibsi1" --resume
   ```

3. Make the IBSI 2 response maps.

   ```bash
   poetry run python -m bench.cli compliance generate-ibsi2-candidates \
     --output-dir "$ROOT/ibsi2-candidates" --phases phase1,phase2 --resume
   ```

4. Evaluate Phase 1.

   ```bash
   poetry run python -m bench.cli compliance evaluate-ibsi2-phase1 \
     --reference-manifest data/ibsi2/references/phase1/reference_manifest.json \
     --reference-dir data/ibsi2/references/phase1/maps \
     --candidate-manifest "$ROOT/ibsi2-candidates/ibsi2_phase1_candidates.json" \
     --output-dir "$ROOT/ibsi2-phase1"
   ```

5. Run Phase 2.

   ```bash
   poetry run python -m bench.cli compliance run-ibsi2-phase2 \
     --candidate-manifest "$ROOT/ibsi2-candidates/ibsi2_phase2_candidates.json" \
     --references data/ibsi2/references/phase2/ibsi2_phase2_references.csv \
     --reference-manifest data/ibsi2/references/phase2/ibsi2_phase2_reference_manifest.json \
     --output-dir "$ROOT/ibsi2-phase2" --resume
   ```

### Safe resume

- `run-ibsi1` and `run-ibsi2-phase2` write `run_manifest.json` with a fingerprint.
- The fingerprint covers the controller source, the input and reference checksums, the library profiles, the environment freezes and the task list.
- A resume stops when the fingerprint changes.
- A resume uses a completed payload again only when its checksum and protocol version match.
- The candidate generator keeps the same rule in `generation_state.json`. Its fingerprint covers the generator source, the Phase 2 preprocessing files, the source manifest and the library environments.

## IBSI 1: digital phantom

### Rules

- **Input.** The run uses only the official digital phantom and mask. It stops when a checksum differs.
- **Discretisation.** The phantom has integer grey levels. The run uses them unchanged (identity discretisation).
- **Aggregation.** The run requires `3d_merge`. GLCM and GLRLM use one merged 3D matrix. GLSZM, GLDZM, NGTDM and NGLDM use their single 3D matrix.
- **No substitute.** The controller rejects every other aggregation. When a library cannot calculate 3D merged values for a group, that group stays unsupported.
- **PyRadiomics setting.** The PyRadiomics default `weightingNorm=None` averages features over directions. The adapter sets `weightingNorm="no_weighting"`, which merges the matrices before the calculation.
- **Code path.** The run calls the timed group calculation of each library once, without a warmup. Thus accuracy and timing use the same calls.
- **Profile.** The import keeps all six workbook sheets. The run uses only the `digital_phantom_3d_merged` profile. CT rows never enter the run, the denominators, the tables or the figures.

PyRadiomics reports kurtosis without the IBSI offset of -3, as its
documentation states. The adapter subtracts 3 after the timer stops, when it
converts the raw result. Without this offset, PyRadiomics passes 106 of 108
evaluated checks, not 107.

### Denominators

| Table 2 family | Definitions | Workbook instances | Standardized checks |
| --- | ---: | ---: | ---: |
| Morphology | 29 | 29 | 25 |
| Local intensity | 2 | 2 | 2 |
| Intensity statistics | 18 | 18 | 18 |
| Intensity histogram | 23 | 23 | 23 |
| Intensity-volume histogram (IVH) | 5 | 7 | 6 |
| GLCM | 25 | 25 | 25 |
| GLRLM | 16 | 16 | 16 |
| GLSZM | 16 | 16 | 16 |
| GLDZM | 16 | 16 | 16 |
| NGTDM | 5 | 5 | 5 |
| NGLDM | 17 | 17 | 17 |
| **Total** | **172** | **174** | **169** |

- **172 definitions.** IBSI Table 2 defines 172 features.
- **174 workbook instances.** The workbook calculates two IVH definitions at 10% and at 90%. These are volume at intensity fraction and intensity at volume fraction.
- **169 standardized checks.** Five definitions have no consensus value: four OMBB and MVEE density features and the IVH area under the curve. Thus 172 - 5 + 2 = 169.

A definition counts as covered when every instance of it has a finite value.
Both the 10% and the 90% instance must be finite for the two IVH definitions.
A value comes from a named native output or from an exact identity that the
library documents. An exact identity keeps its source output name and does not
count as a separate calculation.

### Comparison rule and the strict column

[`bench/compliance/tolerance.py`](../bench/compliance/tolerance.py) copies the
workbook rule. The workbook truncates the difference between reference and
value to three significant digits of the reference. A check passes when this
error is at most the tolerance. Every row also keeps the raw absolute error.

The summary adds a strict column. It equals the workbook result, except when
the reference value is zero. Then the workbook truncates the difference to an
integer and passes any value between -1 and 1. The strict column compares the
raw absolute error with the tolerance instead.

### Results

| Library | Definitions covered / 172 | Finite instances / 174 | Distinct source outputs | Exact identities | Passed / evaluated | Passed / 169 | Strict passed / evaluated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pictologics 1.0.0 | 172 | 174 | 174 | 0 | 169/169 | 169/169 | 169/169 |
| PyRadiomics 3.1.0 | 108 | 108 | 105 | 3 | 107/108 | 107/169 | 107/108 |
| MIRP 2.7.0 | 172 | 174 | 174 | 0 | 169/169 | 169/169 | 169/169 |
| MEDimage 0.9.8 | 171 | 173 | 173 | 0 | 168/168 | 168/169 | 168/168 |
| Z-Rad 26.8.0 | 167 | 169 | 169 | 0 | 165/169 | 165/169 | 165/169 |

- Every task completed without an execution error. Every supported standardized check gave a finite value.
- The strict counts equal the workbook counts for every library.
- PyRadiomics covers 108 definitions: 105 named native outputs and three exact identities. Its one failure is morphology least axis length. [PyRadiomics coverage](feature_contract.md#pyradiomics-coverage) gives the reasons for the 64 missing definitions.
- MEDimage does not give NGLDM dependence count percentage. All its other definitions are covered.
- Z-Rad fails global intensity peak and three intensity histogram gradient features. Its peak kernel uses zero boundary values. Its gradient calculation drops absent grey levels.

## IBSI 2 Phase 1: response maps

### Design

- Phase 1 defines 36 filter tests. IBSI publishes standardized response maps for 33 of them. Tests `9.b.2`, `10.a` and `10.b.2` have no map.
- The primary result uses the 27 tests with an officially 3D kernel. Of these, 24 have a reference map.
- The nine 2D tests are `1.b.1`, `2.c`, `3.c.1` to `3.c.3` and `4.a.1` to `4.b.2`. They include all four Gabor tests. They are in the secondary 36-test result only.
- The 27 3D tests cover seven filter families: mean, LoG, Laws, wavelet, Simoncelli, Riesz-LoG and Riesz-Simoncelli.
- Each library filters the official phantoms with its own code. The boundary conditions are the explicit values in Table 6.1 of the manual.
- Native support means that the library's own backend ran the exact reviewed parameters. Every library and test gets a support record with a reason and evidence.
- A supplied map alone never counts as native support. For the configured versions, every library supplied a map for each natively supported test.

### All-voxel rule

```text
tolerance = 0.01 × (maximum of reference map - minimum of reference map)
pass      = |candidate - reference| <= tolerance at every voxel
```

- The range comes from the reference map only. A candidate cannot make its own tolerance larger.
- The check first compares the map shape and the NIfTI affine. A geometry mismatch or a non-finite voxel is a failure.
- A reference map with zero range passes only with zero error at every voxel.
- The detail table keeps the maximum absolute error, the tolerance, the failing voxel count and fraction, and the map ranges.

Some libraries use other conventions. The backends translate them and record
the reason in the map metadata. MEDimage Laws responses are conjugated into the
IBSI coordinate frame. Z-Rad gets a Gabor width of 14 standard deviations,
because its API reads this value as a full width.

### Results: 27 tests with a 3D kernel

| Library | Native / 27 | Evaluated / 24 | Passed / evaluated | Passed / 24 |
| --- | ---: | ---: | ---: | ---: |
| Pictologics 1.0.0 | 25/27 | 24/24 | 24/24 | 24/24 |
| PyRadiomics 3.1.0 | 1/27 | 1/24 | 1/1 | 1/24 |
| MIRP 2.7.0 | 24/27 | 23/24 | 23/23 | 23/24 |
| MEDimage 0.9.8 | 18/27 | 18/24 | 18/18 | 18/24 |
| Z-Rad 26.8.0 | 17/27 | 17/24 | 17/17 | 17/24 |

### Results: all 36 tests

| Library | Native / 36 | Evaluated / 33 | Passed / evaluated | Passed / 33 |
| --- | ---: | ---: | ---: | ---: |
| Pictologics 1.0.0 | 29/36 | 28/33 | 28/28 | 28/33 |
| PyRadiomics 3.1.0 | 1/36 | 1/33 | 1/1 | 1/33 |
| MIRP 2.7.0 | 33/36 | 32/33 | 32/32 | 32/33 |
| MEDimage 0.9.8 | 27/36 | 27/33 | 27/27 | 27/33 |
| Z-Rad 26.8.0 | 26/36 | 26/33 | 26/26 | 26/33 |

- Every evaluated map passed. The differences between libraries come from native filter support, not from failures.
- Pictologics does not implement structure-tensor steering. Tests `9.b.2` and `10.b.2` need it, and neither has a reference map.
- The 36-test result includes 2D kernels. Do not use it as a 3D result.

## IBSI 2 Phase 2: feature values on response maps

### Design

| Scope | Configurations | Designed checks | Published rows | Standardized checks |
| --- | ---: | ---: | ---: | ---: |
| Configuration B | 11 | 198 | 162 | 161 |
| Configurations A and B | 22 | 396 | 324 | 323 |

- Phase 2 defines 22 configurations, `1.A` to `11.B`, and 18 first-order statistics.
- The official table publishes `1.A` to `9.B`. The row `8.B/stat_qcod` has no consensus value.
- Thus 73 designed checks have no standardized tolerance: the 72 cells of `10.A`, `10.B`, `11.A` and `11.B`, and `8.B/stat_qcod`.
- The primary result uses configuration B. The full A and B design is the secondary result.
- Configurations `1.A` and `1.B` apply no filter. They test only the statistics.

### Input and preprocessing

Phase 2 uses the official PAT1 lung CT phantom and its mask. One common step
makes the A and B inputs in the Pictologics environment. Then every library
filters the same inputs with its own code. The files
[`configs/compliance/ibsi2_phase2_preprocessing_A.json`](../configs/compliance/ibsi2_phase2_preprocessing_A.json)
and
[`configs/compliance/ibsi2_phase2_preprocessing_B.json`](../configs/compliance/ibsi2_phase2_preprocessing_B.json)
hold the steps.

- **A (slice-wise 2D).** No interpolation. Resegmentation to -1000 to 400 HU.
- **B (volumetric 3D).** Resampling to 1 mm isotropic voxels with tricubic image interpolation. Trilinear mask interpolation with a 0.5 threshold. Rounding to the nearest integer. Resegmentation to -1000 to 400 HU.
- **A and B.** No crop. The resegmentation range applies to the unfiltered image. The response map is not discretised. The statistics use the complete 3D ROI.
- **5.B.** This configuration belongs to the B workflow. IBSI defines it as 2D Gabor responses, averaged over three orthogonal planes. Never call it a 3D Gabor kernel.

Phase 2 lets each implementation select its padding. The generator records
the selected and the effective boundary for each library and configuration.
For example, Pictologics calculates Simoncelli `8.B` and `9.B` with its native
periodic FFT.

Each library calculates its own 18 statistics on its own response map. A check
passes when `|value - reference| <= tolerance`, with the published tolerance.
The comparison rounds nothing.

Phase 2 checks the native filters and statistics as components, after common
preprocessing. It does not check the image import, the resampling or the full
workflow of each library.

### Results: configuration B

| Library | Native / 11 designed | Native / 9 published | Finite / 162 | Passed / evaluated | Passed / 161 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pictologics 1.0.0 | 10/11 | 9/9 | 162/162 | 161/161 | 161/161 |
| PyRadiomics 3.1.0 | 1/11 | 1/9 | 15/162 | 15/15 | 15/161 |
| MIRP 2.7.0 | 9/11 | 9/9 | 162/162 | 161/161 | 161/161 |
| MEDimage 0.9.8 | 7/11 | 7/9 | 126/162 | 126/126 | 126/161 |
| Z-Rad 26.8.0 | 7/11 | 7/9 | 126/162 | 126/126 | 126/161 |

### Results: configurations A and B

| Library | Native / 22 | Finite / 396 | Finite / 324 | Passed / evaluated | Passed / 323 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pictologics 1.0.0 | 12/22 | 216/396 | 198/324 | 197/197 | 197/323 |
| PyRadiomics 3.1.0 | 2/22 | 30/396 | 30/324 | 30/30 | 30/323 |
| MIRP 2.7.0 | 18/22 | 324/396 | 324/324 | 323/323 | 323/323 |
| MEDimage 0.9.8 | 14/22 | 252/396 | 252/324 | 252/252 | 252/323 |
| Z-Rad 26.8.0 | 14/22 | 252/396 | 252/324 | 252/252 | 252/323 |

- Every evaluated statistic passed. The differences between libraries come from native filter and statistic support.
- Pictologics supports `1.B` to `10.B`. Configuration `11.B` needs structure-tensor steering, which Pictologics does not implement.
- The PyRadiomics identity configurations give 15 of the 18 statistics in each path. Thus PyRadiomics has 30 finite cells, not 36.
- Pictologics has no general slice-wise 2D workflow. Do not mix its A rows into a 3D result.

## Native-default class

Some libraries have a filter but no setting for one of its controls. An exact
run then marks the test unsupported. The native-default class runs the same
filter once more with the library's own setting for that control. It is a
separate class: it never changes the native-support results above.

A test qualifies only when the exact run fails on a boundary or truncation
control alone. These controls change the response near the image edge or in
the kernel tails. A limit that changes the operation itself never qualifies:
pooling, rotation invariance, steering, 2D against 3D, or a voxel-size limit.

| Library | Control without a setting | Setting that the library uses | Tests |
| --- | --- | --- | --- |
| PyRadiomics | LoG truncation and boundary | SimpleITK recursive Gaussian | 3D LoG |
| PyRadiomics | Wavelet boundary | Periodic | Wavelet with a zero, nearest or mirror boundary |
| MIRP | Simoncelli and Riesz-Simoncelli boundary | Periodic (FFT without padding) | Tests with a non-periodic boundary |
| Z-Rad | Boundary of the Laws energy step | Mirror | Laws energy with a non-mirror boundary |

Pictologics and MEDimage have no qualifying test.

- `generate-ibsi2-candidates` writes the native-default maps to
  `maps/phase1_native_default/` and `maps/phase2_native_default/`, and their
  manifests to `ibsi2_phase1_native_default_candidates.json` and
  `ibsi2_phase2_native_default_candidates.json`.
- Each map records the uncontrolled controls and the boundary that the library
  used. A candidate manifest never mixes the two classes.
- The runner evaluates them with the same rules into `ibsi2-phase1-native-default`
  and `ibsi2-phase2-native-default`.

### Results: Phase 1, native default

All seven qualifying tests have a 3D kernel.

| Library | Test | Uncontrolled control | Result |
| --- | --- | --- | --- |
| PyRadiomics | `2.a` LoG, σ 3 mm, zero boundary | Truncation, boundary | Fail: 0.2% of voxels, maximum error 8.2 |
| PyRadiomics | `2.b` LoG, σ 5 mm, mirror boundary | Truncation, boundary | Fail: all voxels, maximum error 137 |
| PyRadiomics | `5.a.1` wavelet, zero boundary | Boundary | Pass |
| MIRP | `10.a` Riesz-Simoncelli, zero boundary | Boundary | No reference map |
| MIRP | `10.b.1` Riesz-Simoncelli, nearest boundary | Boundary | Pass |
| MIRP | `10.b.2` steered Riesz-Simoncelli | Boundary | No reference map |
| Z-Rad | `3.a.3` Laws energy, zero boundary | Energy boundary | Pass |

In Phase 2, only PyRadiomics `3.B` (3D LoG, mirror boundary) qualifies. It
gives 15 finite statistics of 18. 2 of the 15 pass and 13 fail.

## What the denominators mean

The reports keep these quantities apart:

- **Native support.** The library's own code gives the value or runs the filter with the exact reviewed parameters.
- **Finite coverage.** The library returns a finite value. The denominator is the full design: 172 definitions, 174 instances, 396 Phase 2 cells.
- **Evaluated.** A finite, mapped value has a reference value and tolerance, and the code compares them.
- **Conditional accuracy.** Passed divided by evaluated. It shows how often a supported value is correct.
- **Overall standardized success.** Passed divided by a fixed denominator. The denominators are 169 (IBSI 1), 24 and 33 (Phase 1), and 161 and 323 (Phase 2).

An unsupported, missing, non-finite, ambiguous or failed value stays in the
fixed denominator and never counts as a pass. A value without a standardized
reference shows execution only. It never counts as a pass or a failure.

## Output layout

The runner writes the results to `<result-root>/<machine-id>/compliance`:

```text
compliance/
  ibsi1/
    run_manifest.json         fixed run specification and fingerprint
    run_state.json            task states for a resume
    raw/<library>__<group>.json   one checksummed payload for each library and group
    comparisons.csv           one row for each library and workbook instance (870 rows)
    mapping_audit.json        native names, exact identities, unmapped names
    result_manifest.json      row count, failed tasks, table checksum
    report/
  ibsi2-candidates/
    bundle_manifest.json
    generation_state.json
    ibsi2_phase1_candidates.json   maps, support records and provenance for Phase 1
    ibsi2_phase2_candidates.json   maps, support records and provenance for Phase 2
    ibsi2_phase1_native_default_candidates.json   the native-default class, Phase 1
    ibsi2_phase2_native_default_candidates.json   the native-default class, Phase 2
    source/ licenses/ environments/ generator/ configs/
    preprocessed/phase2/      A and B inputs and their metadata
    maps/phase1/<library>/<test>.nii.gz
    maps/phase2/<library>/<configuration>.nii.gz
    maps/phase1_native_default/  maps/phase2_native_default/
    logs/
  ibsi2-phase1/
    phase1_results.json       one row for each library and test (180 rows)
    result_manifest.json
    report/
  ibsi2-phase2/
    run_manifest.json  run_state.json  raw/  mapping_audit.json  result_manifest.json
    comparisons.csv           one row for each library and designed check (1,980 rows)
    report/
  ibsi2-phase1-native-default/  ibsi2-phase2-native-default/
                              the same files for the native-default class
```

| Report folder | Files |
| --- | --- |
| `ibsi1/report` | `compliance_detail.csv`, `compliance_summary.csv`, `compliance_by_family.csv`, `compliance_summary.md`, `compliance_tables.xlsx`, `data_dictionary.csv`, `figure_captions.md`, `report_manifest.json`, `release_manifest.json` |
| `ibsi2-phase2/report` | the `ibsi1/report` files, plus `compliance_by_configuration.csv`, `compliance_phase2_b_summary.csv` and `compliance_phase2_b_by_configuration.csv` |
| `ibsi2-phase1/report` | `phase1_detail.csv`, `phase1_summary.csv`, `phase1_strict_3d_summary.csv`, `phase1_strict_3d_by_filter.csv`, `phase1_summary.md`, `phase1_tables.xlsx`, `data_dictionary.csv`, `phase1_figure_captions.md`, `phase1_report_manifest.json`, `release_manifest.json` |

- `release_manifest.json` gives the SHA-256 and the size of every report file.
- `data_dictionary.csv` defines every column of every table.
- Each report also draws `figure_conformance_outcomes` and `figure_conformance_by_family`.
- The Phase 1 and Phase 2 reports also draw `figure_error_relative_to_tolerance`.
- Each figure has PDF, SVG and PNG files and a JSON sidecar. [`figures.md`](figures.md) describes the figure style.

To draw the IBSI 1 or Phase 2 tables and figures again, run `report` on the
saved rows. This step runs no library.

```bash
poetry run python -m bench.cli compliance report \
  --comparisons "$ROOT/ibsi1/comparisons.csv" \
  --output-dir <new-report-folder>
```

A change to inputs, mappings, aggregation, filters or tolerances needs a new
run in a new output folder.

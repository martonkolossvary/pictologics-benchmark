# Pictologics radiomics benchmark

This repository compares five Python radiomics libraries: Pictologics,
PyRadiomics, MIRP, MEDimage and Z-Rad. It checks their conformance to the
Image Biomarker Standardisation Initiative (IBSI 1 and IBSI 2). It also
measures their calculation time and memory on three sets of inputs.

One command runs the whole benchmark on your computer. You can then send your
results back to this repository by pull request.

## Run the benchmark

You need:

| Item | Requirement |
| --- | --- |
| System | macOS on Apple Silicon, Linux x86_64 or Windows x86_64 |
| Tools | git and a C compiler ([docs/platforms.md](docs/platforms.md) lists one for each system) |
| Disk | 10 GiB for the repository folder, 8 GiB for the results (3 GiB for the quick run) |
| Memory | 16 GiB or more; with less memory, more large tasks stop at the memory limit. IBSI 2 conformance needs more than 16 GiB; with less, the command reports it and runs the timing benchmark |
| Time | About 2 hours for the quick run; several days for the full run |
| Folder | A local folder that is not in a cloud-sync folder (OneDrive, Dropbox, iCloud, Google Drive) |

Start the benchmark (macOS and Linux):

```bash
git clone https://github.com/martonkolossvary/pictologics-benchmark.git
cd pictologics-benchmark
./reproduce.sh
```

On Windows, open PowerShell in the repository folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1
```

Windows blocks scripts by default. `Bypass` lifts this block for this one run
only, and it changes no setting of the computer. On Windows, use this command
wherever this README writes `./reproduce.sh`, with the same options after it.

To check your computer first, add `--quick`. The quick run takes about two
hours. Its results are not comparable with the full run, and the command never
sends them.

The command does these steps:

1. It installs a private, pinned copy of uv in `.tools/`, the pinned Pythons and
   Poetry ([`reproducibility/toolchain.txt`](reproducibility/toolchain.txt)).
2. It checks your computer and the release files against
   [`reproducibility/release-lock.json`](reproducibility/release-lock.json).
3. It downloads the public inputs and checks every byte.
4. It builds one isolated environment for each library from its lock.
5. It builds the three benchmark datasets (about 25 minutes).
6. It checks the IBSI conformance of each library.
7. It records your hardware in a host profile.
8. It runs the timing benchmark.
9. It writes the reports and the figures.
10. It builds a result package and asks if you want to send it.

On the first run, the command asks for a result folder and a machine ID. Do not
use a personal name in the machine ID.

### Stop and resume

- To stop, press Ctrl-C once and wait for the message. A closed terminal
  window also stops the run cleanly.
- To resume, run the same command again. Finished steps are skipped, and the
  timing run continues from its last finished task.
- Until the run ends, do not update the operating system, do not move the
  repository folder, and do not run `git pull`. A resume refuses each of these
  changes, because the run binds the kernel release, the paths and the code.
- Keep the computer on mains power. Before the first task and after each
  task, the run checks the power source and power saving. If they change, the
  run stops, and the next run does the last task again.

`./reproduce.sh --status` shows the progress. `./reproduce.sh --help` lists all
options.

## Results

The results go to `<result folder>/<machine ID>/`:

| Folder | Content |
| --- | --- |
| `compliance/` | IBSI 1 and IBSI 2 conformance tables and figures |
| `pillar1_morphology/`, `pillar2_whole_anatomy/`, `pillar3_ibsi2_phase3/` | The run ledger, the task records and a `report/` folder for each pillar |
| `figures/` | The manuscript figures and tables |
| `submissions/<machine ID>/<date>/` | The result package |
| `quick/` | The same folders for the quick run |

## Send your results

At the end of the full run, the command asks if it can send the result
package. It opens a pull request with the GitHub CLI, or it prints the steps to
do it by hand. The package holds no paths, user names or host names. An
automatic check verifies each package.
[docs/submitting_results.md](docs/submitting_results.md) gives the details.

## What the benchmark measures

| Library | Version | Python |
| --- | --- | ---: |
| Pictologics | 1.0.1 | 3.12 |
| PyRadiomics | 3.1.0 | 3.11 |
| MIRP | 2.7.0 | 3.12 |
| MEDimage | 0.9.8 | 3.10 |
| Z-Rad | 26.8.0 | 3.12 |

The three input pillars are:

- **Pillar 1:** synthetic phantoms with four tumour masks, three noise levels
  and grid sizes from 32³ to 512³.
- **Pillar 2:** one dense whole-anatomy mask on the same grid sizes.
- **Pillar 3:** the IBSI 2 Phase 3 soft-tissue sarcoma cohort, 51 patients
  with CT, MRI and PET.

The timing rules are:

- Each task times one library, one case and one feature group in a fresh
  process: one untimed warmup call and three timed calls. Each task runs three
  times.
- The six feature groups are intensity, local intensity, morphology, spatial
  morphology (Moran's I and Geary's C), texture and the intensity-volume
  histogram. Each group runs in one library call, so a library can share work
  between its families.
- The five libraries run in a balanced order (a Williams design), so no
  library always runs first or after the same library.
- All libraries read the same stored inputs: the raw image, one mask-specific
  32-bin texture image and one intensity-volume histogram grid.
- File reading, mask preparation and discretisation stay outside the timer.
  Matrix, mesh and neighbourhood construction stay inside it.
- Each library uses its native threading.
- Each calculation call has a limit of 1,800 s. A task also stops when its
  memory reaches the memory budget of the computer. A stopped task gets no
  time value, and larger sizes in the same branch are skipped.

[docs/benchmark_design.md](docs/benchmark_design.md),
[docs/timing_method.md](docs/timing_method.md) and
[docs/feature_contract.md](docs/feature_contract.md) give the full method.

## Documentation

| Document | Content |
| --- | --- |
| [docs/benchmark_design.md](docs/benchmark_design.md) | The three input pillars |
| [docs/timing_method.md](docs/timing_method.md) | The timing boundary, limits, censoring and resume |
| [docs/feature_contract.md](docs/feature_contract.md) | Feature groups and the feature surface of each library |
| [docs/ibsi_compliance.md](docs/ibsi_compliance.md) | IBSI 1 and IBSI 2 conformance |
| [docs/figures.md](docs/figures.md) | The figures and tables |
| [docs/platforms.md](docs/platforms.md) | Supported systems and their prerequisites |
| [docs/submitting_results.md](docs/submitting_results.md) | The result package and the pull request |
| [docs/maintainer_guide.md](docs/maintainer_guide.md) | Release lock, library updates and submissions |

## Repository layout

| Path | Content |
| --- | --- |
| [`bench/`](bench) | The benchmark controller, dataset builders and reports |
| [`bench/adapters/`](bench/adapters) | One adapter for each library |
| [`bench/compliance/`](bench/compliance) | IBSI conformance checks |
| [`bench/figures/`](bench/figures) | Figures and tables |
| [`bench/reproduce/`](bench/reproduce) | The one-command runner |
| [`bench/submission/`](bench/submission) | The result package, its check and the pull request |
| [`configs/`](configs) | Library profiles, benchmark contracts and host profiles |
| [`reproducibility/`](reproducibility) | Input checksums, environment locks, the feature surface and the release lock |
| [`scripts/`](scripts) | The launcher, the workspace builder and maintainer tools |
| [`submissions/`](submissions) | Result packages that users send |
| [`tests/`](tests) | Unit and integration tests |

## Data and licences

The source code of this repository uses the Apache License 2.0, the same
licence as Pictologics. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

This licence does not cover external software or data. The input download
keeps their licence texts. At the pinned revisions:

- IBSI 1 digital phantom: CC BY 4.0.
- IBSI 2 digital phantom: CC BY 4.0.
- IBSI 2 PAT1 CT phantom: CC BY-NC 3.0. You must not use it for commercial
  work.
- IBSI soft-tissue sarcoma cohort: CC BY 3.0, with the citations in its
  official README.
- IBSI 2 reference maps and values: CC0 1.0.
- Each library and each dependency: its own licence.

## Citation

Cite this repository with the data in [`CITATION.cff`](CITATION.cff).

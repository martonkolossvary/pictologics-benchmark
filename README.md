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
| Tools | git and a C compiler: the Xcode Command Line Tools (macOS), `gcc` (Linux) or the Microsoft C++ Build Tools with "Desktop development with C++" (Windows). [docs/platforms.md](docs/platforms.md) gives the install commands |
| To send results (optional) | The GitHub CLI (`gh`), signed in, and a git `user.name` and `user.email`. [docs/submitting_results.md](docs/submitting_results.md#before-the-run) gives the commands |
| Disk | 10 GiB for the repository folder, 8 GiB for the results (3 GiB for the quick run) |
| Memory | 16 GiB or more; with less memory, more large tasks stop at the memory limit. IBSI 2 conformance needs more than 16 GiB; with less, the command reports it and runs the timing benchmark |
| Time | About 2 hours for the quick run; several days for the full run |
| Folder | A local folder that is not in a cloud-sync folder (OneDrive, Dropbox, iCloud, Google Drive) |

Install all prerequisites before the first run. Do any restart that an
installer asks for before the first run, because a restart stops the run.

Start the benchmark (macOS and Linux):

```bash
git clone https://github.com/martonkolossvary/pictologics-benchmark.git
cd pictologics-benchmark
./reproduce.sh
```

Start the benchmark (Windows, in PowerShell):

```powershell
cd $env:USERPROFILE
git clone https://github.com/martonkolossvary/pictologics-benchmark.git
cd pictologics-benchmark
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1
```

Open this PowerShell window from the Start menu, or start Windows Terminal by
itself. Do not use a terminal that another app opened, for example the
Claude desktop app. [docs/platforms.md](docs/platforms.md#terminals-on-windows)
gives the reason.

On Windows, the repository path must have 80 characters or fewer. Clone into
`%USERPROFILE%` only when it is not in OneDrive. Otherwise, start with
`cd C:\` to clone into `C:\pictologics-benchmark`.

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

The command asks these questions, in this order. Each question shows its
default in brackets, and Return accepts the default.

1. `Result folder`: the folder for the results. Only the first run asks.
2. `Machine ID (no personal names)`: a suggested ID. Only the first run asks.
   Do not use a personal name.
3. `Type CALCULATE to start`: this question comes after a checklist, before
   the timing benchmark starts or resumes. The default is `CALCULATE`. Any
   other answer prints "Not started." and stops the command with exit code 1.
4. `Send the results to the benchmark repository now? [y/N]`: this question
   comes at the end of a full run. Type `y` to send the results.

Steps 1 to 7 run before question 3, so stay near the computer until then.
The dataset build alone takes about 25 minutes.

### Stop and resume

- To stop, press Ctrl-C once and wait for the message. A closed terminal
  window also stops the run cleanly.
- To resume, run the same command again. Finished steps are skipped, and the
  timing run continues from its last finished task.
- Until the run ends, do not update the operating system, do not move the
  repository folder, and do not run `git pull`. A resume refuses each of these
  changes, because the run binds the kernel release, the paths and the code.
- On Windows, pause Windows Update before the start: Settings > Windows
  Update > Pause updates, with the longest period. On Windows, the kernel
  release is only the major version (for example `11`), so a resume does not
  see a monthly update.
- Keep the computer on mains power. Before the first task and after each
  task, the run checks the power source and power saving. If they change, the
  run stops, and the next run does the last task again.

`./reproduce.sh --status` shows the progress. `./reproduce.sh --help` lists all
options. On Windows, use
`powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 --status` and
`powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 --help`.

Each start of `reproduce.sh` or `reproduce.ps1` first checks uv, the Pythons
and the Poetry environment again. While a run goes on, use the controller
Python directly instead. This command only reads the progress files, so it
does not disturb the run:

```bash
.venv/bin/python -m bench.reproduce --status
```

On Windows: `.venv\Scripts\python.exe -m bench.reproduce --status`.

The command ends with one of these exit codes:

| Code | Meaning |
| --- | --- |
| 0 | The benchmark is complete. |
| 1 | A stage stopped, or the answer to `Type CALCULATE to start` was not `CALCULATE`. The message says what to do. A change of the power state also gives 1: correct the power state, then run the same command again. |
| 130 | Ctrl-C stopped the command. Run the same command again to resume. |

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

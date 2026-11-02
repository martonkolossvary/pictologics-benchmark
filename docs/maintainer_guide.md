# Maintainer guide

This guide is for the repository owner. Run each command from the repository folder with the controller Python: `.venv/bin/python` on macOS and Linux, `.venv\Scripts\python.exe` on Windows. `reproduce.sh` and `reproduce.ps1` make this environment. Put `.tools/` first on the `PATH`, so that the scripts find the pinned uv:

```bash
export PATH="$PWD/.tools:$PATH"
```

## Refresh the release lock

`reproducibility/release-lock.json` holds the SHA-256 of each bound file and of the built workspace. [`bench/release_lock.py`](../bench/release_lock.py) writes and checks it.

The bound files are:

- `poetry.lock` and `pyproject.toml`
- `configs/adapters/*.yaml`, `configs/benchmark/*.json`, `configs/benchmark/hosts/*.json` and `configs/compliance/*.json`
- `reproducibility/toolchain.txt`, `reproducibility/inputs/manifest.json`, `reproducibility/contracts/*` and `reproducibility/environment-locks/*/*.txt`
- `bench/*.py`, `bench/adapters/*.py` and `bench/compliance/*.py`
- `scripts/launch_benchmark.py`, `scripts/prepare_workspace.py` and `scripts/bootstrap_reproducibility_inputs.py`

For the workspace, the lock holds the SHA-256 of `data/benchmark/workspace_manifest.json`. It also holds the SHA-256 of the manifests of `pillar1`, `pillar2_a1` and `ibsi2_phase3`, and the scheduled task count of each pillar.

| Command | Result |
| --- | --- |
| `python -m bench.release_lock --check` | Compares the bound files with the lock, and the workspace when `data/benchmark` exists. Exits with 1 on a difference. |
| `python -m bench.release_lock --write` | Writes the lock again. Stops when `data/benchmark` does not exist. |

The runner checks the bound files on each start, and the workspace after it builds the workspace. The stage records of a machine hold a checksum of the lock. A new lock thus makes the runner do each recorded stage again.

A change to a bound file makes a new release. To make the release, do these steps:

1. Make the change.
2. Build or check `data/benchmark` with [`scripts/prepare_workspace.py`](../scripts/prepare_workspace.py). On Linux, set `NPY_DISABLE_CPU_FEATURES` to the value of `NUMPY_AVX512_OFF` in [`bench/reproduce/stages.py`](../bench/reproduce/stages.py) first.
3. Run `python -m bench.release_lock --write`.
4. Run `python -m bench.release_lock --check`.
5. [Accept the release](#accept-a-release).
6. Commit the change, the lock and `submissions/accepted_releases.json` together.

## Regenerate environment locks

[`scripts/compile_environment_locks.py`](../scripts/compile_environment_locks.py) resolves the Linux and Windows locks of each adapter from its macOS lock. It uses `uv pip compile` for the targets `x86_64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`.

- Each package that exists on the target keeps its version from the macOS lock.
- The script resolves no package that is newer than `CUTOFF` in the script. `PACKAGE_CUTOFFS` gives one package a later date: its new release, while every other package keeps `CUTOFF`.
- `pip`, `setuptools` and `wheel` resolve together with the adapter, so the locks also hold their dependencies.
- The script drops a macOS pin only when that version has no wheel for the target. It then writes a comment line at the top of the lock, for example `# pyqt5-qt5: the reference version has no Windows-AMD64 wheel`.

Without `--write`, the script only prints the package count of each lock. With `--write`, it writes the Linux and Windows locks. It also writes the `environment_locks` block of each [`configs/adapters/*.yaml`](../configs/adapters), with both checksums for all three platforms.

The locks and the adapter profiles are bound files. After a new lock, [refresh the release lock](#refresh-the-release-lock).

## Swap an adapter version

The Pictologics 1.0.1 release used this procedure. Use the same steps for each other adapter.

1. Update [`configs/adapters/pictologics.yaml`](../configs/adapters/pictologics.yaml) and the macOS lock [`reproducibility/environment-locks/darwin-arm64/pictologics.txt`](../reproducibility/environment-locks/darwin-arm64/pictologics.txt).
   - In the profile, update `version`, `requirement`, `upstream`, `verified_latest_stable`, `source_commit` and `notes`.
   - Build a clean Pictologics environment on the Mac. Write its `pip freeze --all` list to the macOS lock. The environment check compares the two lists.
2. Run `python scripts/compile_environment_locks.py --write`.
   - Before you run it, add the new package and the date of its release to `PACKAGE_CUTOFFS` in the script. Resolve the macOS lock with the same dates (`uv pip install --exclude-newer <CUTOFF> --exclude-newer-package <package>=<date>`), so that only the swapped package and its own new dependencies change.
3. Update `release_version` of `pictologics` in [`configs/benchmark/main.json`](../configs/benchmark/main.json), [`screen.json`](../configs/benchmark/screen.json) and [`screen_one_library_thread.json`](../configs/benchmark/screen_one_library_thread.json).
   - When the feature counts change, also update `native_output_count` and `family_output_counts`.
4. Rebuild and verify the Pictologics environment:

   ```bash
   python -m bench.cli env create --profiles pictologics --force
   python -m bench.cli env verify --profiles pictologics
   ```

   For Linux and Windows, run the `environments` job of the `ci` workflow.
5. Run the feature-surface audit:

   ```bash
   python scripts/audit_adapter_feature_surface.py \
     --dataset-dir data/benchmark/pillar1 \
     --case-id p1_reference_noise_m1_n032 \
     --output-dir <new folder>
   ```

   The audit fails when an adapter reports a version that differs from its profile. It also fails when a name in [`reproducibility/contracts/adapter_feature_surface.csv`](../reproducibility/contracts/adapter_feature_surface.csv) is missing.
6. Rebuild the workspace manifest: `python scripts/prepare_workspace.py --output-root data/benchmark --validate-only`. The manifest holds the checksums of the three contracts.
7. Rerun the Pictologics conformance. Use the four commands of the `compliance` stage in [`bench/reproduce/stages.py`](../bench/reproduce/stages.py), with `--adapters pictologics` and new output folders. The conformance commands do not resume after a source change.
8. Run `python -m bench.release_lock --write`.
9. Add the new release-lock checksum to [`submissions/accepted_releases.json`](../submissions/accepted_releases.json).

## Accept a release

[`bench/submission/validate.py`](../bench/submission/validate.py) accepts a package only when its release-lock checksum is in `submissions/accepted_releases.json`. The package stage of the runner uses the same check. To accept a release, do these steps:

1. Calculate the checksum of the lock:

   ```bash
   python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('reproducibility/release-lock.json').read_bytes()).hexdigest())"
   ```

2. Add the checksum to the `release_lock_sha256` list.
3. Keep the checksums of the other accepted releases in the list. The packages of each listed release continue to pass.
4. Commit the file together with the lock.

`submissions/accepted_releases.json` is not a bound file, so this change does not change the release lock. The check counts the tasks of a package only when the package comes from the current release.

## Set the repository address

[`bench/submission/submit.py`](../bench/submission/submit.py) reads the GitHub address of the benchmark repository from `project.urls.Repository` in [`pyproject.toml`](../pyproject.toml). Without this address, the runner prints the manual steps and opens no pull request. `pyproject.toml` is a bound file, so a new address makes a new release.

## Review and merge a submitted package

1. Make sure that the `submission-check` workflow passes.
2. Open `submission.json` in the package.
3. Make sure that the machine ID holds no personal name.
4. Compare the platform key and the hardware with the host profile.
5. Read the status counts of each pillar. Look for failed tasks and for times that seem wrong. The check rejects only impossible times.
6. Merge the pull request.
7. [Build the community figure](#build-the-community-figure) again.

To run the check on your computer, use `python -I bench/submission/validate.py . submissions/<machine ID>/<date>`.

## Build the community figure

```bash
python -m bench.figures.community --submissions submissions --output-dir <folder>
```

[`bench/figures/community.py`](../bench/figures/community.py) uses the newest package of each machine. Each machine gives its own runtime ratios to Pictologics, and the figure shows the machines side by side. It never pools absolute times. The default data are Pillar 3, CT. The output is `figure_community_runtime_ratio` in the output folder. The figure needs the Arial font.

## Run the feasibility screens

Run `./reproduce.sh --screens` (on Windows, `powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 --screens`). The clone keeps this setting in `.reproduce/settings.json`. `--quick` ignores it.

The runner runs two screens before the main run:

| Screen | Contract | Folder |
| --- | --- | --- |
| Feasibility screen: one warmup, one timed call, one repeat, 600 s for each call | [`configs/benchmark/screen.json`](../configs/benchmark/screen.json) | `<machine folder>/screen/` |
| The same screen, with one thread for the maths libraries; Numba and ITK keep their native threads | [`configs/benchmark/screen_one_library_thread.json`](../configs/benchmark/screen_one_library_thread.json) | `<machine folder>/screen_one_library_thread/` |

The runner writes no reports for the screens. To report a screen pillar, run `python -m bench.cli report --input-dir <screen pillar folder> --output-dir <folder>`. The report writes `task_outcomes.csv`, with one row for each task.

## CI jobs

| Workflow | Job | Starts on | What it does |
| --- | --- | --- | --- |
| [`ci.yml`](../.github/workflows/ci.yml) | `controller` | each push and pull request, except one that changes only result packages | On macOS, Ubuntu and Windows: `poetry sync`, `poetry check`, Ruff lint and format check, `bench.release_lock --check`, `pytest` |
| `ci.yml` | `scripts` | each push and pull request, except one that changes only result packages | ShellCheck on `reproduce.sh`, PSScriptAnalyzer on `reproduce.ps1` |
| `ci.yml` | `environments` | manual start, with `environments` on | On macOS, Ubuntu and Windows: builds and verifies the five adapter environments from their locks |
| `ci.yml` | `quick-run` | manual start, with `quick_run` on | On Ubuntu and Windows: `reproduce --quick --yes --no-submit`, then keeps the reports and the figures as artifacts |
| [`submission.yml`](../.github/workflows/submission.yml) | `check` | each pull request that changes `submissions/` | Checks each result package (see [Submitting results](submitting_results.md#the-automatic-check)) |

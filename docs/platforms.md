# Platforms

The benchmark runs on three platforms. The runner ([`reproduce.sh`](../reproduce.sh) or [`reproduce.ps1`](../reproduce.ps1)) stops on every other platform.

The runner starts the launcher ([`scripts/launch_benchmark.py`](../scripts/launch_benchmark.py)). The launcher starts the controller ([`bench/run.py`](../bench/run.py)) for each pillar. The controller runs the adapters.

| Platform | Platform key | Lock folder |
| --- | --- | --- |
| macOS on Apple Silicon | `Darwin-arm64` | `darwin-arm64` |
| Linux on x86_64 | `Linux-x86_64` | `linux-x86_64` |
| Windows on x86_64 | `Windows-AMD64` | `windows-amd64` |

The platform key is `platform.system()` and `platform.machine()`, joined by a hyphen (`native_platform_key` in [`bench/env.py`](../bench/env.py)).

The benchmark does not support Linux on ARM. Z-Rad requires PyQt5, and PyQt5 has no wheel for Linux on ARM.

## Prerequisites

The first stage of the runner checks this computer. It lists every problem that it finds and then stops (`prerequisites` in [`bench/reproduce/stages.py`](../bench/reproduce/stages.py)).

| Item | macOS | Linux | Windows |
| --- | --- | --- | --- |
| git | required; the Xcode Command Line Tools include it | required, for example `sudo apt install git` | required, for example `winget install --id Git.Git -e` |
| C compiler | Xcode Command Line Tools: `xcode-select --install` | `cc` or `gcc`, for example `sudo apt install build-essential` | Microsoft C++ Build Tools, with "Desktop development with C++" (see below) |
| Repository path | any length | any length | at most 80 characters |

PyRadiomics installs from its source code at a git commit. For this reason, every platform needs git and a C compiler. On macOS and Linux, `reproduce.sh` uses `curl` to download uv.

On Windows, this command installs the Build Tools with the C++ workload:

```powershell
winget install --id Microsoft.VisualStudio.BuildTools -e --override "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

- `Microsoft.VisualStudio.BuildTools` is the Build Tools 2026. The CI builds PyRadiomics with Visual Studio 2026 (the GitHub image `windows-2025-vs2026`).
- The runner accepts any version that has the component `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`. It finds the component with `vswhere.exe`.
- The install needs administrator rights (a UAC prompt). The installer can ask for a restart. Restart before the first run, because a restart stops the run.

After an install, open a new PowerShell window, so that the `PATH` is current.

Windows blocks PowerShell scripts by default (the execution policy). Start the runner with `powershell -ExecutionPolicy Bypass -File .\reproduce.ps1`, and put the options after it. `Bypass` holds only for this one process. It changes no setting of the computer.

Free disk space:

- The repository folder needs 10 GiB.
- The result folder needs 8 GiB, or 3 GiB for a run with `--quick`.
- uv keeps the three Pythons and its package cache in its own user folders, outside the repository. `.tools/uv python dir` and `.tools/uv cache dir` print the two folders. On Windows, the defaults are `%APPDATA%\uv\data\python` and `%LOCALAPPDATA%\uv\cache`. On the test Mac, after several runs, the Pythons used 0.2 GiB and the cache 4.9 GiB. The runner does not check this space.

Memory: 16 GiB or more. With less memory, more large tasks stop at the memory limit. IBSI 2 conformance needs more than 16 GiB. With less, the runner reports it and runs the timing benchmark. The runner does not check the memory before the start.

Use local folders. The runner stops when the repository or the result folder is in a cloud-sync folder. The check finds OneDrive, Dropbox, Google Drive, iCloud Drive and `Library/CloudStorage` folders by their path. It also finds the folders that the `OneDrive`, `Dropbox` and `GoogleDrive` environment variables name (`_known_sync_root` in [`scripts/launch_benchmark.py`](../scripts/launch_benchmark.py)).

On Windows, the launcher also stops when the longest result path is more than 240 characters. Some native libraries do not support long paths.

The runner asks for the result folder on the first run. The default is `~/pictologics-benchmark-results` on macOS and Linux, and `%USERPROFILE%\AppData\Local\pictologics-benchmark-results` on Windows. It is not the repository folder, so a clone in the home folder keeps its results apart.

[`.gitattributes`](../.gitattributes) keeps LF line endings in every text file on every platform. Thus the checksums of text files are the same on all platforms.

On Linux, numpy uses AVX-512 math routines (fast CPU math code) when the CPU has them. These routines round some results differently, and the synthetic images then miss the release checksums. The runner thus sets `NPY_DISABLE_CPU_FEATURES` for the workspace build on Linux only. The timed adapter processes keep the routines.

## What the runner installs

`reproduce.sh` (macOS and Linux) and `reproduce.ps1` (Windows) do these steps:

1. Install uv into `.tools/` in the repository, at the version in [`reproducibility/toolchain.txt`](../reproducibility/toolchain.txt). The script replaces this uv only when its version differs. Another uv on the computer stays as it is.
2. Put `.tools/` first on the `PATH`.
3. Install the three Pythons in `toolchain.txt` with `uv python install`. uv keeps them in its own user folder.
4. Run Poetry, at the version in `toolchain.txt`, through `uv tool run`.
5. Make the controller environment `.venv/` with the first Python in `toolchain.txt` (3.12), and fill it with `poetry sync`.
6. Start `python -m bench.reproduce` with the arguments of the command.

## Environment locks

Each adapter has one environment lock for each platform, in [`reproducibility/environment-locks/<platform>/`](../reproducibility/environment-locks). The `environment_locks` block in [`configs/adapters/*.yaml`](../configs/adapters) holds two checksums for each lock. `sha256` covers the bytes of the file. `freeze_sha256` covers the sorted package list.

To build an adapter environment, [`bench/env.py`](../bench/env.py) does these steps:

1. Find the Python for the adapter (see [Interpreter discovery](#interpreter-discovery)).
2. Make the environment in `.venvs/adapters/<adapter>/` with `uv venv --seed`.
3. Find the lock for the platform key of this computer, and check its two checksums.
4. Install the lock with `uv pip install --requirement`.
5. Run `pip check` and the smoke imports of the adapter profile.
6. Compare `pip freeze --all` with the lock. A difference stops the build.
7. Write `environment.json` with the package freeze, the Python version and the lock record.

On each start, the runner verifies every environment against its `environment.json`. It rebuilds an environment that fails this check.

The Linux and Windows locks come from the macOS lock (see the [maintainer guide](maintainer_guide.md#regenerate-environment-locks)). A comment line at the top of a lock names each package whose macOS version has no wheel for that platform.

## Interpreter discovery

MEDimage uses Python 3.10 and PyRadiomics uses Python 3.11. MIRP, Pictologics and Z-Rad use Python 3.12. `_find_python` in [`bench/env.py`](../bench/env.py) looks for each Python in this order:

1. The controller Python, when its major and minor version match.
2. `uv python find --managed-python <version>`: a Python that uv installed.
3. `uv python find <version>`: any Python that uv finds.
4. On Windows, the Python launcher: `py -<version>`.
5. `python<version>` on the `PATH`, after a version check.

When no Python matches, the build stops with a message.

## Power and sleep

On the first run, the runner measures this computer and writes `host_profile.json` into the machine folder ([`bench/reproduce/host_profile.py`](../bench/reproduce/host_profile.py)). The profile requires mains power and low-power mode off. It also requires a sleep-prevention assertion when the computer can hold one. `--host-profile` gives a frozen profile to use instead.

[`bench/host_state.py`](../bench/host_state.py) reads the power state with a different probe on each platform:

| Check | macOS | Linux | Windows |
| --- | --- | --- | --- |
| Power source | `pmset -g batt` | `/sys/class/power_supply`: a `Mains` supply is online, or the computer has no battery | `GetSystemPowerStatus`: AC line on, or no battery |
| Low-power mode | `pmset -g custom`, AC block: `lowpowermode 1` or `powermode 1` | `powerprofilesctl get` gives `power-saver`; without `powerprofilesctl`, the check reads "off" | `GetSystemPowerStatus`: battery saver on |
| Sleep assertion | `pmset -g assertions`: `PreventSystemSleep` or `PreventUserIdleSystemSleep` above 0 | `systemd-inhibit --list` shows a lock that blocks sleep | the hold of the controller process |
| Controller sleep guard | `caffeinate -dimsu -w <pid>` | `systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block` | `SetThreadExecutionState` |

On Linux, the profile requires the sleep assertion only when a test lock with `systemd-inhibit` works. A Linux server without logind cannot block sleep, and it does not sleep.

The checks occur in this sequence:

1. The launcher checks the power source and low-power mode before it starts the controller. It does not check the sleep assertion, because the controller holds its own guard.
2. The controller holds the sleep guard for the full task loop (`prevent_sleep` in [`bench/run.py`](../bench/run.py)).
3. Before the first task and after each task, the controller reads the power state again and compares it with the host profile. The check after a task also clears the next task, so the run reads the power state once for each task.
4. On a difference, the controller records a `runtime_state_stop` event and exits with code 75. A difference after a task marks that task `interrupted`, whatever its result, so the resume runs it again.
5. The runner tells you to correct the power state and to run the same command again.

## Process trees

Each task runs its adapter in a new process. [`bench/run.py`](../bench/run.py) controls this process and all its child processes as one tree:

- On macOS and Linux, the controller starts the adapter with `start_new_session=True`. The adapter and its children thus form their own process group. To stop them, the controller sends `SIGTERM` to the group, and `SIGKILL` after the grace period.
- On Windows, the controller starts the adapter with `CREATE_NEW_PROCESS_GROUP`. It then puts the adapter into a kill-on-close Job Object (`WindowsJob` in [`bench/process_tree.py`](../bench/process_tree.py)). When the controller exits for any reason, Windows closes the job and ends every process in it.
- Before a run, the controller stops stray adapters. These are adapter processes that use the inputs folder of this run, for example after a `kill -9` of a controller. The controller records the event `stray_adapters_stopped`.
- For the memory stop, the controller adds the memory of all processes in the tree. macOS gives `phys_footprint`, Linux gives `VmRSS` plus `VmSwap`, and Windows gives private bytes.

## Fonts for the figures

The figures use Arial. The figures stage looks for Arial first. When Arial is missing, the stage does not draw the figures. The runner then prints an install hint at the end. On Linux, the hint is `sudo apt install ttf-mscorefonts-installer`. Windows and macOS include Arial. When Arial is missing there, the hint is "install the Arial font".

## Times from different systems

Absolute times depend on the hardware and the operating system. Thus the times of each system are a separate stratum. Never pool absolute times from different systems. Each result package records its platform key and hardware in `submission.json`. The community figure ([`bench/figures/community.py`](../bench/figures/community.py)) gives each machine its own ratios to Pictologics and shows the machines side by side.

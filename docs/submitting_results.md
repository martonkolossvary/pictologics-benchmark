# Submitting results

After a complete main run, the runner builds a result package. It then offers to send the package to the benchmark repository by pull request. The runner never builds or sends a package for a quick run (`--quick`).

## Before the run

The send comes several days after the start. Thus prepare the GitHub CLI and git before the run:

1. Install the GitHub CLI (`gh`). On Windows, use `winget install --id GitHub.cli -e`. The MSI installs for all users and needs administrator rights (a UAC prompt). On macOS and Linux, see <https://cli.github.com>.
2. On Windows, open a new PowerShell window, so that the `PATH` is current.
3. Sign in to GitHub. On Windows, use a PowerShell window from the Start menu, not a terminal that another app opened ([platforms.md](platforms.md#terminals-on-windows)). The first command opens the browser:
   ```
   gh auth login --web --git-protocol https
   gh auth setup-git
   gh auth status
   ```
4. Set the git identity. The submit stage commits the package in a temporary clone, and git needs a name and an email for this commit:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
   GitHub's private noreply address works as the email. GitHub shows it under Settings > Emails.

Without a name and an email, the commit fails with "Author identity unknown". The runner then prints the git error at the end and sends nothing. Set the identity, and run the command again with `--submit`.

## The package and submit stages

The last two stages of the runner are `package` and `submit` ([`bench/reproduce/stages.py`](../bench/reproduce/stages.py)). They do these steps:

1. The package stage builds the package. When this machine has a package already, the stage uses the newest one.
2. The package stage checks the package with [`bench/submission/validate.py`](../bench/submission/validate.py), the same check that runs on a pull request. A failed check stops the runner and lists the problems.
3. The submit stage asks: "Send the results to the benchmark repository now? [y/N]".

The package holds no reports and no figures. Thus, in a main run, a failed `reports` or `figures` stage does not stop the package and submit stages. The runner names the failure at the end, and the next run of the same command tries the failed stage again. A quick run stops on such a failure, so that the CI quick run finds it.

These options control the submit stage:

| Option | Result |
| --- | --- |
| `--submit` | The runner sends the package and does not ask. |
| `--no-submit` | The runner does not send the package and does not ask. |
| `--yes` without `--submit` | The runner does not send the package. |

To send the package later, run the same command again with `--submit`. The runner does not repeat the finished benchmark stages.

```bash
./reproduce.sh --submit
```

On Windows: `powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 --submit`.

A clone that ran with `--quick` keeps that setting in `.reproduce/settings.json`. Use a new clone for a main run.

## What the package holds

[`bench/submission/package.py`](../bench/submission/package.py) writes the package to `<result folder>/<machine ID>/submissions/<machine ID>/<UTC date>/`. A machine can make one package on each UTC date. The three pillars must have the run status `completed` or `completed_with_failures`.

| File | Content |
| --- | --- |
| `host_profile.json` | Machine ID, label, CPU model, hardware counts and the required power state |
| `<pillar>/tasks.jsonl.gz` | One line for each task: task ID, order, status, attempt, payload checksum and the ledger record of the task |
| `<pillar>/run_spec.json` | The run specification of the pillar |
| `environment/<adapter>.json` | The `environment.json` of each adapter environment, with its full package freeze |
| `compliance/<suite>/*.csv.gz` | The IBSI conformance tables of `ibsi1`, `ibsi2-phase1` and `ibsi2-phase2`, and of the native-default class (`ibsi2-phase1-native-default`, `ibsi2-phase2-native-default`) |
| `submission.json` | Machine ID, date, release-lock checksum, commit, platform key, hardware, the run status and status counts of each pillar, replacement counts, and file checksums and sizes |
| `SHA256SUMS` | The SHA-256 of each file in the package |

The pillars are `pillar1_morphology`, `pillar2_whole_anatomy` and `pillar3_ibsi2_phase3`.

The package leaves out the staged input copies, the raw payload files, the SQLite ledger and the run lock. It also leaves out the reports and the figures. From the error output of an adapter (`adapter_stderr`), it keeps the last 4,096 characters.

## How the package removes personal data

The package builder does these steps:

1. Replace the paths of this computer with tokens: `{RESULT_ROOT}`, `{REPO}`, `{UV_PYTHON}`, `{TMP}` and `{HOME}`. It replaces the native form, the POSIX form and the escaped form of each path. The longest path goes first, so a deeper path keeps its own token.
2. Replace the user name with `{USER}`, and the host names with `{HOST}`. A name must be a full word to match, in upper or lower case. Names of one or two characters stay.
3. Count the replacements of each token into `scrubbed_values` in `submission.json`.
4. Write all files, then read each file again and search it for each path and name.
5. When the search finds one, stop with "personal data remains" and delete the partial package.

The host profile holds no serial number, hardware UUID, host name or account name. The machine ID that the runner suggests comes from the operating system, the CPU model and a random part. It never uses the host name. Do not put a personal name in a machine ID.

## Size limit

The package must be 45 MiB or smaller. A larger package stops the build, and the check also rejects it.

## Send the package by pull request

The submit stage uses the GitHub CLI (`gh`). It needs `gh` on the `PATH`, a `gh` login ([Before the run](#before-the-run)), and the GitHub address of the benchmark repository in `project.urls.Repository` of [`pyproject.toml`](../pyproject.toml). When these are ready, [`bench/submission/submit.py`](../bench/submission/submit.py) does these steps:

1. Fork the benchmark repository to your GitHub account. GitHub lets no account fork its own repository, so for the owner of the benchmark repository the next steps use the repository itself.
2. Clone the fork into a temporary folder.
3. Make the branch `results-<machine ID>-<date>`.
4. Copy the package to `submissions/<machine ID>/<date>/`, commit it and push the branch.
5. Open the pull request "Benchmark results from \<machine ID\> (\<date\>)".

The runner prints the address of the pull request at the end. To see the pull request and the result of its check later:

```
gh pr list --repo martonkolossvary/pictologics-benchmark
gh pr checks <number> --repo martonkolossvary/pictologics-benchmark
```

When one of these three is missing, the runner prints the manual steps:

1. Fork the benchmark repository on GitHub.
2. Copy the package folder to `submissions/<machine ID>/<date>/` in your fork.
3. Commit the folder.
4. Open a pull request to the benchmark repository.

## The automatic check

[`.github/workflows/submission.yml`](../.github/workflows/submission.yml) runs on each pull request that changes `submissions/`. It does these steps:

1. Check out the base branch and the pull request into two separate folders. The validator and the list of accepted releases come from the base branch. The files of the pull request are only data.
2. Fail when the pull request changes a file outside `submissions/<machine ID>/<YYYY-MM-DD>/`.
3. Run `python -I base/bench/submission/validate.py base pr/<folder>` on each package folder.

The validator checks these items:

- The folder names are a valid machine ID and a date.
- The package is 45 MiB or smaller.
- `SHA256SUMS` lists all files of the package and only those files, and each checksum matches.
- `submission.json` has schema version 1, and its machine ID is the folder name.
- The release-lock checksum is in [`submissions/accepted_releases.json`](../submissions/accepted_releases.json).
- The host profile holds no private field, and its hardware is the hardware in `submission.json`.
- Each task has a known status. Each measured task has a payload checksum and plausible times.
- The status counts and the run status of each pillar agree with `submission.json`.
- For the current release, each pillar holds all tasks that the release lock schedules.
- For the current release, each task record names its library and feature group, and it comes from the main contract ([`configs/benchmark/main.json`](../configs/benchmark/main.json)). A screen or a quick run thus fails the check.
- For the current release, the package holds one environment record for each library lock of its platform. Each environment holds exactly the packages of its lock.
- No file holds a personal path (`/Users/<name>`, `/home/<name>`, `C:\Users\<name>`) or an email address.

To run the same check on your computer, use the controller Python in the repository folder:

- macOS and Linux: `.venv/bin/python -I bench/submission/validate.py . <package folder>`
- Windows: `.venv\Scripts\python.exe -I bench\submission\validate.py . <package folder>`

On Windows, a plain `python` can be the Microsoft Store placeholder, which runs no script.

## Known limit

The checksums prove that a package is complete and consistent. They do not prove that its timings are genuine. The time check rejects only impossible values: a wall time of zero or less, or a CPU time above wall time × (logical CPUs + 1).

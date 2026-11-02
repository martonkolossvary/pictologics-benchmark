# Contributing

## Send results

Run the benchmark with `./reproduce.sh` (macOS and Linux) or `powershell -ExecutionPolicy Bypass -File .\reproduce.ps1` (Windows). After a complete main run, the runner offers to send a result package by pull request. [docs/submitting_results.md](docs/submitting_results.md) tells what the package holds and how the check works. [docs/platforms.md](docs/platforms.md) gives the prerequisites for each platform.

## Report a problem

Open an issue in the GitHub repository. Include these items:

- The platform key, for example `Darwin-arm64`, `Linux-x86_64` or `Windows-AMD64`.
- The commit: the output of `git rev-parse HEAD`.
- The title of the stage that stopped, and the error message.
- The output of `./reproduce.sh --status` or `powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 --status`.

Remove your user name, host name and private paths from the text before you send it.

## Change the code

A change to a bound file needs a refresh of the release lock, and it makes a new release. [`bench/release_lock.py`](bench/release_lock.py) lists the bound files in `BOUND_PATTERNS`. The [maintainer guide](docs/maintainer_guide.md#refresh-the-release-lock) gives the procedure.

Before you open a pull request, run the checks of the `controller` job in [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
poetry check
poetry run ruff check .
poetry run ruff format --check .
poetry run python -m bench.release_lock --check
poetry run pytest -q
```

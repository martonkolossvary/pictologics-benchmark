"""Send a result package to the benchmark repository as a pull request."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def repository() -> str | None:
    """Return ``owner/name`` of the public repository from ``pyproject.toml``."""

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    url = project.get("project", {}).get("urls", {}).get("Repository", "")
    match = re.match(r"https://github\.com/([^/]+/[^/]+?)(?:\.git)?/?$", url)
    return match.group(1) if match else None


def _gh(*arguments: str, cwd: Path | None = None) -> str:
    process = subprocess.run(
        ["gh", *arguments], cwd=cwd, capture_output=True, text=True
    )
    if process.returncode != 0:
        raise RuntimeError(f"gh {' '.join(arguments[:2])}: {process.stderr.strip()}")
    return process.stdout.strip()


def _git(*arguments: str, cwd: Path) -> None:
    subprocess.run(["git", *arguments], cwd=cwd, check=True)


def manual_steps(package: Path, target: str | None) -> str:
    destination = f"submissions/{package.parent.name}/{package.name}"
    where = f"https://github.com/{target}" if target else "the benchmark repository"
    return (
        "To send the results by hand:\n"
        f"  1. Fork {where} on GitHub.\n"
        f"  2. Copy the folder {package}\n"
        f"     to {destination} in your fork, and commit it.\n"
        "  3. Open a pull request to the benchmark repository.\n"
        "A check then verifies the package."
    )


def submit(package: Path) -> str:
    """Open the pull request; return its URL.  Raise with manual steps otherwise."""

    target = repository()
    ready = (
        target is not None
        and shutil.which("gh") is not None
        and subprocess.run(["gh", "auth", "status"], capture_output=True).returncode
        == 0
    )
    if not ready:
        raise RuntimeError(
            "The GitHub CLI (gh) is missing or not logged in.\n"
            + manual_steps(package, target)
        )
    machine_id, date = package.parent.name, package.name
    login = _gh("api", "user", "--jq", ".login")
    owner, name = target.split("/")
    branch = f"results-{machine_id}-{date}"
    with tempfile.TemporaryDirectory() as temporary:
        # GitHub lets no account fork its own repository, so the owner
        # pushes the branch to the repository itself.
        if login.lower() != owner.lower():
            _gh("repo", "fork", target, "--clone=false")
        fork = Path(temporary) / "fork"
        _gh(
            "repo",
            "clone",
            f"{login}/{name}",
            str(fork),
            "--",
            "--depth",
            "1",
        )
        _git("checkout", "-b", branch, cwd=fork)
        destination = fork / "submissions" / machine_id / date
        shutil.copytree(package, destination)
        _git("add", "submissions", cwd=fork)
        _git("commit", "-m", f"Add benchmark results from {machine_id}", cwd=fork)
        _git("push", "--set-upstream", "origin", branch, cwd=fork)
        return _gh(
            "pr", "create",
            "--repo", target,
            "--head", f"{login}:{branch}",
            "--title", f"Benchmark results from {machine_id} ({date})",
            "--body",
            "Result package made by reproduce. The submission check verifies it.",
        )  # fmt: skip

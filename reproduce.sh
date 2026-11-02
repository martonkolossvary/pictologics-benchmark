#!/usr/bin/env bash
# Run the whole benchmark on this computer with one command (macOS and Linux).
# Run the same command again to resume after a stop.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
value() { sed -n "s/^$1=//p" reproducibility/toolchain.txt; }

# A private, pinned uv in .tools/ leaves any other uv on this computer alone.
UV="$PWD/.tools/uv"
if [[ ! -x "$UV" || "$("$UV" --version | cut -d' ' -f2)" != "$(value uv)" ]]; then
    echo "Installing uv $(value uv) into .tools/"
    curl -LsSf "https://astral.sh/uv/$(value uv)/install.sh" |
        env UV_UNMANAGED_INSTALL="$PWD/.tools" sh
fi
export PATH="$PWD/.tools:$PATH"

# shellcheck disable=SC2046
"$UV" python install $(value python)
poetry() { "$UV" tool run --from "poetry==$(value poetry)" poetry "$@"; }
export POETRY_VIRTUALENVS_IN_PROJECT=true
poetry env use "$("$UV" python find --managed-python "$(value python | head -n 1)")" >/dev/null
poetry sync --quiet
exec .venv/bin/python -m bench.reproduce "$@"

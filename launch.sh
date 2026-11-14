#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Veyra's virtual environment is missing. Run ./setup.sh first." >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"

if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ ! -x /usr/bin/sandbox-exec ]]; then
        echo "macOS sandbox-exec is unavailable; Veyra will not launch without network isolation." >&2
        exit 1
    fi
    # Inherited by the tracking worker too. Setup/downloads run separately.
    exec /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' \
        "$PYTHON" -m veyra "$@"
fi

echo "Note: Veyra v0.2 is tested on macOS; this platform has no launcher network sandbox." >&2
exec "$PYTHON" -m veyra "$@"

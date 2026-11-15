#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv-research/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Run ./setup-research.sh to install the optional research environment." >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export TORCH_HOME="$PROJECT_DIR/.cache/torch"
export PYTORCH_ENABLE_MPS_FALLBACK=0
if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ ! -x /usr/bin/sandbox-exec ]]; then
        echo "Network isolation unavailable; refusing research launch." >&2
        exit 1
    fi
    exec /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' "$PYTHON" -m veyra3d "$@"
fi
exec "$PYTHON" -m veyra3d "$@"

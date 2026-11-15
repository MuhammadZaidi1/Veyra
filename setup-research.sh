#!/bin/bash
# Separate optional environment; never installs into the live baseline.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python || ! -x .tools/bin/uv ]]; then
    echo "Run ./setup.sh first to provide the project-local Python and uv." >&2
    exit 1
fi
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.python"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
if [[ ! -e .venv-research ]]; then
    .tools/bin/uv venv --seed --python .venv/bin/python .venv-research
elif [[ ! -x .venv-research/bin/python ]] || ! .venv-research/bin/python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    echo "Existing .venv-research is not a Python 3.12 environment; move it aside and retry." >&2
    exit 1
fi
.tools/bin/uv pip install --python .venv-research/bin/python -r requirements-research-macos.lock.txt
echo "Research setup complete. Run ./research.sh doctor or ./research.sh smoke. No training started."

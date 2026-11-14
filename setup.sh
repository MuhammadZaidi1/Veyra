#!/bin/bash
# All tools, Python runtimes, dependencies, and models stay inside this project.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.python"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"

if [[ ! -x "$PROJECT_DIR/.tools/bin/uv" ]]; then
    if [[ -e "$PROJECT_DIR/.tools" && ! -x "$PROJECT_DIR/.tools/bin/python" ]]; then
        echo "Existing .tools is not a Veyra bootstrap environment. Move it aside and retry." >&2
        exit 1
    fi
    if [[ ! -x "$PROJECT_DIR/.tools/bin/python" ]]; then
        python3 -m venv "$PROJECT_DIR/.tools"
    fi
    "$PROJECT_DIR/.tools/bin/python" -m pip install 'uv==0.13.8'
fi

UV="$PROJECT_DIR/.tools/bin/uv"
"$UV" python install --no-bin 3.12
if [[ -e "$PROJECT_DIR/.venv" ]]; then
    if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]] || ! "$PROJECT_DIR/.venv/bin/python" -c \
        'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
        echo "Existing .venv is not Python 3.12. Move it aside, then rerun ./setup.sh." >&2
        exit 1
    fi
else
    "$UV" venv --seed --managed-python --python 3.12 "$PROJECT_DIR/.venv"
fi
"$UV" pip install --python "$PROJECT_DIR/.venv/bin/python" -r requirements.lock.txt
"$PROJECT_DIR/.venv/bin/python" scripts/download_model.py

echo "Setup complete. Launch with: ./launch.sh"

#!/usr/bin/env bash
# ------------------------------------------------------------
# run.sh — set up and launch notebook 26 in its isolated env.
#
# Usage:
#   ./envs/notebook26/run.sh            # launch Jupyter Lab on the notebook
#   ./envs/notebook26/run.sh --execute  # run the notebook headlessly (CI / smoke)
#   ./envs/notebook26/run.sh --shell    # just drop into the activated venv shell
# ------------------------------------------------------------
set -euo pipefail

# Resolve repo root from script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${SCRIPT_DIR}"
NOTEBOOK="${REPO_ROOT}/26_CodeAgent_Analisis_Inteligente_GithubIssues.ipynb"
KERNEL_NAME="notebook26-codeact"
KERNEL_DISPLAY="Python 3.12 (notebook26 codeact)"

MODE="${1:-launch}"

# Short-circuit help so it doesn't trigger a sync.
case "${MODE}" in
    -h|--help|help)
        sed -n '2,12p' "${BASH_SOURCE[0]}"
        exit 0
        ;;
esac

# 1) Ensure uv is available.
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not found in PATH. Install from https://docs.astral.sh/uv/" >&2
    exit 1
fi

# 2) Sync the isolated environment (creates .venv if missing, idempotent).
echo "==> Syncing isolated env at ${ENV_DIR}"
(cd "${ENV_DIR}" && uv sync --python 3.12)

VENV_PY="${ENV_DIR}/.venv/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
    echo "ERROR: venv python not found at ${VENV_PY}" >&2
    exit 1
fi

# 3) Register the Jupyter kernel (idempotent).
if ! jupyter kernelspec list 2>/dev/null | grep -q "${KERNEL_NAME}"; then
    echo "==> Registering Jupyter kernel: ${KERNEL_NAME}"
    "${VENV_PY}" -m ipykernel install --user \
        --name "${KERNEL_NAME}" \
        --display-name "${KERNEL_DISPLAY}"
else
    echo "==> Kernel ${KERNEL_NAME} already registered."
fi

# 4) Ensure the root .env exists so ANTHROPIC_API_KEY can be loaded.
if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    echo "WARN: ${REPO_ROOT}/.env not found. Copy .env.example and set ANTHROPIC_API_KEY." >&2
fi

# 5) Activate the venv for the chosen mode.
# shellcheck disable=SC1091
source "${ENV_DIR}/.venv/bin/activate"

case "${MODE}" in
    --shell)
        echo "==> Activated venv. Type 'exit' to leave."
        exec "${SHELL:-/bin/bash}"
        ;;

    --execute)
        echo "==> Executing notebook headlessly with kernel ${KERNEL_NAME}"
        # Install jupyter into the venv if missing (only needed for --execute).
        if ! "${VENV_PY}" -c "import nbconvert" 2>/dev/null; then
            (cd "${ENV_DIR}" && uv add jupyter nbconvert)
        fi
        cd "${REPO_ROOT}"
        "${VENV_PY}" -m jupyter nbconvert \
            --to notebook --execute --inplace \
            --ExecutePreprocessor.kernel_name="${KERNEL_NAME}" \
            "${NOTEBOOK}"
        ;;

    launch|"")
        echo "==> Launching Jupyter Lab on notebook 26"
        # Pick a jupyter launcher. Prefer one that has 'lab' available.
        JUPYTER_BIN=""
        for cand in \
            "${REPO_ROOT}/.venv/bin/jupyter" \
            "$(command -v jupyter-lab || true)" \
            "$(command -v jupyter || true)"; do
            if [[ -n "${cand}" && -x "${cand}" ]]; then
                JUPYTER_BIN="${cand}"
                break
            fi
        done
        if [[ -z "${JUPYTER_BIN}" ]]; then
            echo "ERROR: No jupyter binary found. Install jupyter in any venv or system-wide." >&2
            exit 1
        fi
        echo "    using: ${JUPYTER_BIN}"
        cd "${REPO_ROOT}"
        # Try 'lab', fall back to 'notebook'.
        if "${JUPYTER_BIN}" lab --version >/dev/null 2>&1; then
            exec "${JUPYTER_BIN}" lab "${NOTEBOOK}"
        else
            exec "${JUPYTER_BIN}" notebook "${NOTEBOOK}"
        fi
        ;;

    *)
        echo "Unknown mode: ${MODE}" >&2
        echo "Use --help for options." >&2
        exit 2
        ;;
esac

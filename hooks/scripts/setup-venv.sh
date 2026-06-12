#!/usr/bin/env bash
# setup-venv.sh — idempotent, hash-gated deps-only venv installer for the kb plugin.
#
# Called as a SessionStart hook (runs before env-probe and context hooks).
# Fast no-op when requirements.txt has not changed.
#
# Venv location (Phase 1 decision, kb-fzd.1):
#   Primary:  ${CLAUDE_PLUGIN_DATA}/venv     — survives ROOT updates
#   Fallback: ~/.cache/kb/plugin-venv        — when CLAUDE_PLUGIN_DATA is unset
#
# Tool invocation convention (Phase 3, kb-fzd.3):
#   ${KB_VENV_PYTHON} ${CLAUDE_PLUGIN_ROOT}/kb.py <command>
# where KB_VENV_PYTHON = <venv>/bin/python (resolved below).
#
# Hash gate: sha256 of requirements.txt stored as <venv>/.requirements-hash.
# Rebuild triggered on: first run, requirements.txt change, or missing python binary.

set -euo pipefail

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
REQUIREMENTS="${PLUGIN_ROOT}/requirements.txt"

# Resolve venv path (DATA-or-fallback)
if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
    VENV_DIR="${CLAUDE_PLUGIN_DATA}/venv"
else
    VENV_DIR="${HOME}/.cache/kb/plugin-venv"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
HASH_FILE="${VENV_DIR}/.requirements-hash"

# Compute current requirements hash
if ! command -v sha256sum &>/dev/null && command -v shasum &>/dev/null; then
    CURRENT_HASH=$(shasum -a 256 "${REQUIREMENTS}" | awk '{print $1}')
else
    CURRENT_HASH=$(sha256sum "${REQUIREMENTS}" | awk '{print $1}')
fi

# Fast no-op check: venv exists + python binary present + hash unchanged
if [ -f "${VENV_PYTHON}" ] && [ -f "${HASH_FILE}" ]; then
    STORED_HASH=$(cat "${HASH_FILE}")
    if [ "${CURRENT_HASH}" = "${STORED_HASH}" ]; then
        # Venv is up-to-date; emit no output (SessionStart must be non-blocking)
        exit 0
    fi
fi

# Build or rebuild the venv
# Find a suitable python3 interpreter (prefer python3.11+)
PY_BIN=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "${candidate}" &>/dev/null; then
        PY_BIN="${candidate}"
        break
    fi
done

if [ -z "${PY_BIN}" ]; then
    echo "kb-plugin setup-venv: ERROR — no python3 found in PATH" >&2
    exit 1
fi

echo "kb-plugin setup-venv: building deps venv at ${VENV_DIR}" >&2
mkdir -p "$(dirname "${VENV_DIR}")"

# Remove stale venv on hash change (dim/model changes need clean slate for sqlite-vec)
if [ -d "${VENV_DIR}" ]; then
    rm -rf "${VENV_DIR}"
fi

"${PY_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${REQUIREMENTS}"

# Store the hash so the next run is a no-op
echo "${CURRENT_HASH}" > "${HASH_FILE}"

echo "kb-plugin setup-venv: venv ready (${VENV_PYTHON})" >&2

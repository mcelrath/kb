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
PLUGIN_ROOT="${PLUGIN_ROOT%/}"  # normalize: a trailing slash else trips the root-changed re-install every session
REQUIREMENTS="${PLUGIN_ROOT}/requirements.txt"

# Resolve venv path (DATA-or-fallback)
if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
    VENV_DIR="${CLAUDE_PLUGIN_DATA}/venv"
else
    VENV_DIR="${HOME}/.cache/kb/plugin-venv"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
HASH_FILE="${VENV_DIR}/.requirements-hash"

# Install / upgrade ~/.local/bin/<name> wrappers (stable: reference the venv path,
# NOT the versioned CLAUDE_PLUGIN_ROOT). The plugin OWNS these wrappers so the
# `kb`/`kbt` commands always resolve to the plugin-managed venv — a stale
# hand-written wrapper (e.g. one hardcoding a dev .venv path) is backed up once
# and replaced. Defined early + called on BOTH the fast path and a full rebuild
# so an already-built venv still gets its wrapper reconciled.
ensure_wrappers() {
    local _LOCAL_BIN="${HOME}/.local/bin"
    [ -d "${_LOCAL_BIN}" ] || return 0
    case ":${PATH}:" in
        *":${_LOCAL_BIN}:"*) ;;
        *) echo "kb-plugin setup-venv: NOTE — ${_LOCAL_BIN} is not on PATH; add it to your shell profile so \`kb\`/\`kbt\` resolve by name." >&2 ;;
    esac
    local name wrapper venv_tool
    for name in kb kbt; do
        wrapper="${_LOCAL_BIN}/${name}"
        venv_tool="${VENV_DIR}/bin/${name}"
        # Already points at our venv tool? leave it.
        if [ -f "${wrapper}" ] && grep -qF "${venv_tool}" "${wrapper}" 2>/dev/null; then
            continue
        fi
        # A different wrapper exists — back it up once before taking ownership.
        if [ -f "${wrapper}" ] && [ ! -f "${wrapper}.pre-kb-plugin" ]; then
            cp -p "${wrapper}" "${wrapper}.pre-kb-plugin"
            echo "kb-plugin setup-venv: backed up existing ${wrapper} -> ${wrapper}.pre-kb-plugin" >&2
        fi
        cat > "${wrapper}" << WRAPPER_EOF
#!/usr/bin/env bash
# ${name} wrapper — written by kb-plugin setup-venv.sh
# Stable: references the plugin venv, not the versioned CLAUDE_PLUGIN_ROOT.
# kb config defaults (embedding/llm URLs) come from kb/config.py — no env needed.
exec "${venv_tool}" "\$@"
WRAPPER_EOF
        chmod +x "${wrapper}"
        echo "kb-plugin setup-venv: installed ${wrapper}" >&2
    done
}

# Compute current requirements hash
if ! command -v sha256sum &>/dev/null && command -v shasum &>/dev/null; then
    CURRENT_HASH=$(shasum -a 256 "${REQUIREMENTS}" | awk '{print $1}')
else
    CURRENT_HASH=$(sha256sum "${REQUIREMENTS}" | awk '{print $1}')
fi

# Fast no-op check: venv exists + python binary present + hash unchanged
ROOT_FILE="${VENV_DIR}/.kb-src-root"
if [ -f "${VENV_PYTHON}" ] && [ -f "${HASH_FILE}" ]; then
    STORED_HASH=$(cat "${HASH_FILE}")
    if [ "${CURRENT_HASH}" = "${STORED_HASH}" ]; then
        # Requirements unchanged — but re-run editable install if the plugin root
        # changed (e.g. plugin updated to a new versioned directory).
        STORED_ROOT=""
        [ -f "${ROOT_FILE}" ] && STORED_ROOT=$(cat "${ROOT_FILE}")
        if [ "${STORED_ROOT}" != "${PLUGIN_ROOT}" ]; then
            echo "kb-plugin setup-venv: plugin root changed (${STORED_ROOT} -> ${PLUGIN_ROOT}); re-installing editable package" >&2
            "${VENV_DIR}/bin/pip" install --quiet -e "${PLUGIN_ROOT}"
            echo "${PLUGIN_ROOT}" > "${ROOT_FILE}"
        fi
        # Venv is up-to-date — still reconcile the wrappers (cheap, idempotent),
        # then exit quietly (SessionStart must be non-blocking).
        ensure_wrappers
        exit 0
    fi
fi

# Build or rebuild the venv.
# Interpreter selection: kb supports Python 3.11+ (PEP-563 portability fix landed
# in kb-9kr.2 — offending modules with builtin-name shadows in class bodies now
# carry "from __future__ import annotations" so eager evaluation on 3.11-3.13 is
# safe). Prefer 3.14+ when available (PEP-649 lazy eval is faster), but 3.11-3.13
# are fully supported.
PY_BIN=""
for candidate in python3.14 python3.15 python3.16 python3.13 python3.12 python3.11 python3 python; do
    if command -v "${candidate}" &>/dev/null; then
        ver=$("${candidate}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
        case "${ver}" in
            3.11|3.12|3.13|3.14|3.15|3.16) PY_BIN="${candidate}"; break ;;
        esac
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

# Install kb as an editable package so <venv>/bin/kb exists and is importable.
# Record the installed root so we can detect plugin-version changes on future runs.
# (ROOT_FILE already declared above for the fast-path check.)
"${VENV_DIR}/bin/pip" install --quiet -e "${PLUGIN_ROOT}"
echo "${PLUGIN_ROOT}" > "${ROOT_FILE}"

ensure_wrappers

echo "kb-plugin setup-venv: venv ready (${VENV_PYTHON})" >&2

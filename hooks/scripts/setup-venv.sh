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
        # Venv is up-to-date; emit no output (SessionStart must be non-blocking)
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

# Install ~/.local/bin/kb wrapper (stable: references venv path, not versioned root).
_install_wrapper() {
    local local_bin="${HOME}/.local/bin"
    local wrapper="${local_bin}/kb"
    local venv_kb="${VENV_DIR}/bin/kb"

    [ -d "${local_bin}" ] || return 0   # ~/.local/bin absent — skip silently

    # Check ~/.local/bin is on PATH (advisory only; don't fail).
    case ":${PATH}:" in
        *":${local_bin}:"*) ;;
        *)
            echo "kb-plugin setup-venv: NOTE — ${local_bin} is not on PATH; add it to your shell profile so \`kb\` resolves by name." >&2
            ;;
    esac

    # If a wrapper already exists and points at the same venv kb, leave it.
    if [ -f "${wrapper}" ]; then
        if grep -qF "${venv_kb}" "${wrapper}" 2>/dev/null; then
            return 0   # already correct
        fi
        # Check it's ours (not some unrelated kb binary) before overwriting.
        if ! grep -qF "kb-plugin" "${wrapper}" 2>/dev/null && ! grep -qF "${HOME}/.cache/kb" "${wrapper}" 2>/dev/null && ! grep -qF "CLAUDE_PLUGIN_DATA" "${wrapper}" 2>/dev/null; then
            echo "kb-plugin setup-venv: ${wrapper} exists and does not look like a kb-plugin wrapper — skipping." >&2
            return 0
        fi
    fi

    cat > "${wrapper}" << WRAPPER_EOF
#!/usr/bin/env bash
# kb wrapper — written by kb-plugin setup-venv.sh
# Stable: references the plugin venv, not the versioned CLAUDE_PLUGIN_ROOT.
exec "${venv_kb}" "\$@"
WRAPPER_EOF
    chmod +x "${wrapper}"
    echo "kb-plugin setup-venv: installed ${wrapper}" >&2
}
_install_wrapper

echo "kb-plugin setup-venv: venv ready (${VENV_PYTHON})" >&2

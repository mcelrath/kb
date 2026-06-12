#!/usr/bin/env bash
# venv-path.sh — resolve the kb plugin venv location.
#
# Convention (Phase 3, kb-fzd.3):
#   Primary:  ${CLAUDE_PLUGIN_DATA}/venv     (stable across ROOT updates)
#   Fallback: ~/.cache/kb/plugin-venv        (fixed path when DATA is unset)
#
# Usage (source or subshell):
#   source "$(dirname "$0")/venv-path.sh"
#   # then use: $KB_VENV_DIR and $KB_VENV_PYTHON
#
# Or inline:
#   KB_VENV_PYTHON=$(bash "${CLAUDE_PLUGIN_ROOT}/hooks/scripts/lib/venv-path.sh" --python)
#
# When called with --python, echoes the resolved python path and exits.
# When sourced (no args), exports KB_VENV_DIR and KB_VENV_PYTHON.

_kb_resolve_venv() {
    if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
        echo "${CLAUDE_PLUGIN_DATA}/venv"
    else
        echo "${HOME}/.cache/kb/plugin-venv"
    fi
}

if [ "${1:-}" = "--python" ]; then
    echo "$(_kb_resolve_venv)/bin/python"
else
    export KB_VENV_DIR
    KB_VENV_DIR="$(_kb_resolve_venv)"
    export KB_VENV_PYTHON="${KB_VENV_DIR}/bin/python"
fi

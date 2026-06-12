#!/bin/bash
# claude-env.sh — derive CLAUDE_DIR from hook location (plugin-local copy).
#
# In the plugin context, CLAUDE_DIR is not the plugin root — it is the user's
# ~/.claude config dir (where sessions/ and state/ live).  The plugin root is
# CLAUDE_PLUGIN_ROOT; hooks needing the user's config dir use CLAUDE_DIR.
#
# Derivation: hooks live at ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/; going up
# three levels (scripts/ -> hooks/ -> plugin-root/) does NOT give ~/.claude.
# So we fall back to the env var or the standard location.
#
# Usage: source "$(dirname "$0")/../lib/claude-env.sh"
# Then use $CLAUDE_DIR instead of $HOME/.claude

if [[ -z "$CLAUDE_DIR" ]]; then
    # Prefer an env var if the harness exports one
    CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
fi

# Validate; fall back to HOME-relative if not a real dir
if [[ ! -d "$CLAUDE_DIR" ]]; then
    CLAUDE_DIR="$HOME/.claude"
fi

# Local LLM server (llama.cpp / vLLM)
# Override with LLM_HOST env var if server is on a different machine
LLM_HOST="${LLM_HOST:-localhost}"
LLM_PORT="${LLM_PORT:-8014}"
LLM_URL="${LLM_URL:-http://${LLM_HOST}:${LLM_PORT}/v1/chat/completions}"
KB_LLM_URL="${KB_LLM_URL:-http://${LLM_HOST}:${LLM_PORT}/completion}"

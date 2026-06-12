#!/usr/bin/env bash
# Phase-1 ENV PROBE: emit all CLAUDE_PLUGIN_* env vars as additionalContext.
# This trivial hook is the Phase-1 gate — it tells us which plugin env vars
# Claude Code actually exports, so the venv location can be decided.
# Replace this with the real kb-context.sh in Phase 4.

# Collect all CLAUDE_PLUGIN_* vars
PLUGIN_VARS=""
while IFS='=' read -r key value; do
    case "$key" in
        CLAUDE_PLUGIN_*)
            PLUGIN_VARS="${PLUGIN_VARS}  ${key}=${value}\n"
            ;;
    esac
done < <(env)

if [ -z "$PLUGIN_VARS" ]; then
    PLUGIN_VARS="  (none exported)\n"
fi

# Also explicitly probe the two vars the plan cares about
ROOT_VALUE="${CLAUDE_PLUGIN_ROOT:-(not set)}"
DATA_VALUE="${CLAUDE_PLUGIN_DATA:-(not set)}"

cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "KB PLUGIN ENV PROBE RESULT:\n  CLAUDE_PLUGIN_ROOT=${ROOT_VALUE}\n  CLAUDE_PLUGIN_DATA=${DATA_VALUE}\n\nAll CLAUDE_PLUGIN_* vars:\n$(printf '%b' "${PLUGIN_VARS}")\n(Phase-1 probe hook — will be replaced with kb-context.sh in Phase 4)"
  }
}
EOF

exit 0

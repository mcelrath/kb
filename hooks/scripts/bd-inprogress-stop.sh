#!/bin/bash
# Stop hook: block a bridge agent from stopping while it has claimed in-progress bd work.
#
# Fires ONLY for sessions whose bridge identity is a known persona (pip, tip,
# emmy, carl, archie, kb-dev, ...). User/main sessions with no bridge identity
# are not affected.
#
# Mechanism: resolve the session's bridge persona; run `bd list --status=in_progress
# --assignee=<persona>`; if any items exist, block with exit 2 and echo the queue.
#
# Loop guard: uses stop_hook_active (same as block-unprompted-deferral.sh) — fires
# once, then allows the stop so the agent isn't trapped forever. One warning is enough.
#
# Why this matters: the IDLE RULE in persona files ("idle legitimate ONLY when bd
# in_progress shows nothing claimed by you") is instruction-level prose that the Stop
# event does not enforce. This hook converts it to enforcement.

[ -x "$HOME/.agent-bridge/bridge" ] || exit 0

INPUT=$(cat 2>/dev/null)

# Loop guard: if stop hook already fired for this stop attempt, let it through.
STOP_HOOK_ACTIVE=$(echo "$INPUT" | python3 -c "
import sys, json
v = json.load(sys.stdin).get('stop_hook_active', False)
print('1' if (v is True or (isinstance(v, str) and v.lower() == 'true')) else '0')
" 2>/dev/null)
[ "$STOP_HOOK_ACTIVE" = "1" ] && exit 0

SESSION_ID=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
AGENTS_FILE="$HOME/.agent-bridge/agents.json"
[ -f "$AGENTS_FILE" ] || exit 0

# Resolve bridge persona for this session.
# Priority: AGENT_ID env → session_id lookup.
PERSONA=""
if [ -n "${AGENT_ID:-}" ]; then
    PERSONA="$AGENT_ID"
elif [ -n "$SESSION_ID" ]; then
    PERSONA=$(python3 -c "
import json, sys
agents = json.load(open('$AGENTS_FILE'))
for a in agents.get('agents', []):
    if a.get('session_id') == '$SESSION_ID':
        print(a.get('id', ''))
        break
" 2>/dev/null)
fi

# Only fire for known bridge personas — not for the main user session.
case "${PERSONA:-}" in
    pip|tip|emmy|carl|archie|kb-dev|qwen|victor|pip2|pip3|emmy-emitter) : ;;
    *) exit 0 ;;
esac

# Check for claimed in-progress work.
IN_PROGRESS=$(bd list --status=in_progress --assignee="$PERSONA" 2>/dev/null)
# bd outputs "No issues found" when empty — test for actual issue rows (lines starting with id characters)
echo "$IN_PROGRESS" | grep -qE '^[a-z0-9]' || exit 0

cat >&2 <<EOF
BD_INPROGRESS_BLOCKED: $PERSONA has claimed in-progress bd work. Do not stop silently.

Your claimed queue:
$IN_PROGRESS

Per your IDLE RULE: idle is legitimate ONLY when 'bd list --status=in_progress'
shows nothing claimed by you. You have claimed work — finish it, or if blocked:
  1. State the blocker on the bridge (bridge send archie "blocked on X because Y")
  2. Set an explicit defer on each item if genuinely gated:
       bd update <id> --notes "defer: <reason>"
  Then you may stop.

This hook fires once. The next stop attempt will be allowed.
EOF
exit 2

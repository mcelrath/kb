#!/bin/bash
# Stop hook: block a bridge agent from stopping while it has claimed in-progress kbt work.
#
# Fires ONLY for sessions whose bridge identity is a known persona (pip, tip,
# emmy, carl, archie, kb-dev, ...). User/main sessions with no bridge identity
# are not affected.
#
# Mechanism: resolve the session's bridge persona; run `kbt list --status=in_progress
# --assignee=<persona> --json`; if any items exist, block with exit 2 and echo the queue.
#
# Loop guard (TWO layers): (1) stop_hook_active fast-path, same as
# block-unprompted-deferral.sh; (2) a DURABLE per-session disk marker. Layer 2 exists
# because some harnesses re-wake the agent into a FULL NEW TURN on exit-2, so the next
# Stop is a fresh event with stop_hook_active=false — layer 1 alone then never trips and
# the agent loops until its context is exhausted (observed: a physics persona burned its
# whole context this way). The disk marker blocks AT MOST ONCE per session regardless.
#
# Why this matters: the IDLE RULE in persona files ("idle legitimate ONLY when kbt
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

# Check for claimed in-progress work. Use --json + a count (kbt human output is
# bracketed `[id] (status) title`, so a "line starts with id char" test is wrong).
COUNT=$(kbt list --status=in_progress --assignee="$PERSONA" --json 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(len(d) if isinstance(d, list) else 0)
except Exception:
    print(0)
" 2>/dev/null)
[ "${COUNT:-0}" = "0" ] && exit 0

# Durable loop guard: block AT MOST ONCE per session, independent of stop_hook_active.
# A harness that re-wakes the agent into a fresh turn resets stop_hook_active to false
# each time; this disk marker survives that and breaks the infinite Stop loop.
STATE_DIR="${CLAUDE_STATE_DIR:-$HOME/.claude/state}"
mkdir -p "$STATE_DIR" 2>/dev/null
GUARD_KEY="${SESSION_ID:-$PERSONA}"
MARK="$STATE_DIR/${GUARD_KEY//[^A-Za-z0-9_-]/_}-kbt-inprogress-blocked"
[ -f "$MARK" ] && exit 0
: > "$MARK" 2>/dev/null

IN_PROGRESS=$(kbt list --status=in_progress --assignee="$PERSONA" 2>/dev/null)

cat >&2 <<EOF
KBT_INPROGRESS_BLOCKED: $PERSONA has claimed in-progress kbt work. Do not stop silently.

Your claimed queue:
$IN_PROGRESS

Per your IDLE RULE: idle is legitimate ONLY when 'kbt list --status=in_progress'
shows nothing claimed by you. You have claimed work — finish it, or if blocked:
  1. State the blocker on the bridge (bridge send archie "blocked on X because Y")
  2. Set an explicit defer on each item if genuinely gated:
       kbt update <id> --notes "defer: <reason>"
  Then you may stop.

This hook fires once. The next stop attempt will be allowed.
EOF
exit 2

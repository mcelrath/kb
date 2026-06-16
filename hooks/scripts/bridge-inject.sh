#!/bin/bash
# Bridge message INJECTOR (kb-2os.3) — reads unread via the kb-SERVER, not the
# jsonl `bridge recv`. Wired on PreToolUse + UserPromptSubmit: drains unread peer
# messages for the current agent (GET /bridge/messages, cursor-advanced by
# _bridge_inject_fetch.py) and injects the bodies as additionalContext — same
# delivery model as kb findings surfacing, no extra `recv` tool call.
#   PreToolUse — inject via exit-0 additionalContext JSON (NEVER exit 2: that
#     blocks the call AND cancels the parallel tool batch).
#   UserPromptSubmit — inject via stdout (becomes a system-reminder).
# The kb-server is the transport API; the per-session cursor lives in
# $KB_STATE_DIR/<session>-bridge-injected so a message injects exactly once.
set -u
BASE="${KB_SERVER_URL:-http://127.0.0.1:8765}"
HERE="$(dirname "$(readlink -f "$0")")"

INPUT=$(cat 2>/dev/null)
EVENT=$(printf '%s' "$INPUT" | python3 -c "import sys,json;print(json.load(sys.stdin).get('hook_event_name',''))" 2>/dev/null)
SESSION_ID=$(printf '%s' "$INPUT" | python3 -c "import sys,json;print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
[ -z "$SESSION_ID" ] && exit 0

# Identity: resolve session_id -> agent id via the bridge registry (agents.json
# is the registry, not the message transport; reading it is fine).
AGENTS="$HOME/.agent-bridge/agents.json"
ID=""
[ -f "$AGENTS" ] && ID=$(jq -r --arg sid "$SESSION_ID" '.agents[] | select(.session_id == $sid) | .id' "$AGENTS" 2>/dev/null | head -n1)
case "$ID" in ""|"("*|"null") exit 0 ;; esac

# Refresh this agent's LIVENESS mtime so `bridge agents` reports it ONLINE while
# it is actively running turns. `bridge:_agent_status` derives liveness from the
# mtime of ~/.agent-bridge/<id>.cursor (fresh <120s = online). The old
# bridge-watcher-check.sh refreshed it on every PreToolUse/UserPromptSubmit, but
# injection migrated to the kb-server and nothing touched the file — so every
# active agent read 'offline:stale' despite delivering/receiving every turn.
# This hook fires on the SAME events, so it is the correct place to restore it.
# `touch` bumps mtime (preserves content for an existing legacy cursor; an empty
# create is harmless since reads now go through the kb-server cursor, not this file).
touch "$HOME/.agent-bridge/${ID}.cursor" 2>/dev/null || true

# NOTE (kb-ee7): no watcher teardown here anymore. The watcher is now an asyncRewake
# Stop hook (bridge-watch-rewake.sh) launched/owned by the harness; its flock keeps a
# single instance, and an exit-2 while ACTIVE just queues to the next turn (harmless).
# While working, THIS per-turn injection is the delivery path regardless.

# Fetch unread via the kb-server (cursor-tracked, injects once).
UNREAD=$(KB_SERVER_URL="$BASE" python3 "$HERE/_bridge_inject_fetch.py" "$ID" "$SESSION_ID" 2>/dev/null)
[ -z "$UNREAD" ] && exit 0

if [ "$EVENT" = "PreToolUse" ]; then
    printf 'BRIDGE_UPDATE (new peer messages):\n%s\n(end bridge messages)' "$UNREAD" \
        | python3 -c "import sys,json; print(json.dumps({'hookSpecificOutput':{'hookEventName':'PreToolUse','additionalContext':sys.stdin.read()}}))"
    exit 0
fi
echo "BRIDGE_UPDATE (new peer messages since last user prompt):"
echo "$UNREAD"
echo "(end bridge messages)"
exit 0

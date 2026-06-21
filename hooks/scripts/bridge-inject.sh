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

# Refresh this agent's LIVENESS each turn AND record WHICH session holds the id.
# CRITICAL (archie #5949 / #5951): the legacy `bridge` binary uses ~/.agent-bridge/<id>.cursor
# as a NUMERIC recv cursor (jq --argjson). The earlier kb-72f717.3 change overwrote that
# file's CONTENT with this session_id (a UUID) — which crashed `bridge recv`
# (invalid JSON to --argjson) and clobbered the recv position every turn. Split the two:
#   <id>.kbsession : holds THIS session_id (content) for persona collision detection,
#                    mtime = kb per-turn liveness. kb-owned; the binary ignores it.
#   <id>.cursor    : the binary's numeric recv cursor — we only `touch` it (bump mtime
#                    for `_agent_status` online/offline) and NEVER overwrite its content.
printf '%s' "$SESSION_ID" > "$HOME/.agent-bridge/${ID}.kbsession" 2>/dev/null || true
touch "$HOME/.agent-bridge/${ID}.cursor" 2>/dev/null || true

# NOTE (kb-ee7): no watcher teardown here anymore. The watcher is now an asyncRewake
# Stop hook (bridge-watch-rewake.sh) launched/owned by the harness; its flock keeps a
# single instance, and an exit-2 while ACTIVE just queues to the next turn (harmless).
# While working, THIS per-turn injection is the delivery path regardless.

# Fetch + FORMAT unread via the kb-server (cursor-tracked, injects once). The helper
# emits the harness-appropriate hook output itself: Claude gets a JSON envelope carrying
# BOTH a user-visible systemMessage ("📨 bridge: <sender> — <subject>") and the model
# additionalContext; goose gets raw bodies on UserPromptSubmit (emit_collect) and nothing
# on PreToolUse (emit_blocking has no context channel). Harness is detected here.
CLAUDE=0
{ [ -n "${CLAUDE_PLUGIN_ROOT:-}" ] || [ -n "${CLAUDE_SESSION_ID:-}" ]; } && CLAUDE=1
OUT=$(KB_SERVER_URL="$BASE" python3 "$HERE/_bridge_inject_fetch.py" "$ID" "$SESSION_ID" "$EVENT" "$CLAUDE" 2>/dev/null)
[ -z "$OUT" ] && exit 0
printf '%s' "$OUT"
exit 0

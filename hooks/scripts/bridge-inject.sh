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
# `bridge:_agent_status` derives online/offline from the mtime of
# ~/.agent-bridge/<id>.cursor (fresh <120s = online) — writing the file bumps mtime,
# so presence still works (filename UNCHANGED — the binary reads exactly `<id>.cursor`).
# kb-72f717.3 F1: write THIS session_id as the cursor CONTENT (not a bare `touch`), so
# persona identity resolution can tell whether a DIFFERENT live session currently holds
# this bridge id (content=other session_id + fresh mtime ⇒ a live collision). A renamed
# per-session file (<id>.<sid>.cursor) would break `_agent_status` (no glob) — content is
# the only variant that preserves presence AND makes liveness per-session-decidable.
printf '%s' "$SESSION_ID" > "$HOME/.agent-bridge/${ID}.cursor" 2>/dev/null || true

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

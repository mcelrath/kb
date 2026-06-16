#!/bin/bash
# SessionStart hook (kb-2os.3): restore bridge registration for the kb-bridge.
#
# The MESSAGE transport is now the kb-server (inject via GET /bridge/messages,
# send via POST /bridge/send, wake via the SSE watcher). The REGISTRY (agents.json:
# announce/whoami/identity) stays on the bridge binary — it's the agent directory,
# not the message channel, and there is no kb-server registry-write endpoint.
#
# Invariants at session start (SIDE EFFECTS ONLY — this is a SessionStart hook, whose
# stdout is injected on Claude but NOT on goose): (1) agent announced (session_id
# refreshed), (2) authoritative session pin written. The BRIDGE PROTOCOL instruction
# text moved to kb-instructions.sh (UserPromptSubmit, once-per-session) so it injects on
# BOTH harnesses (kb-d0m). Draining is handled by bridge-inject (kb-server) per turn.
BRIDGE="$HOME/.agent-bridge/bridge"
[[ ! -x "$BRIDGE" ]] && exit 0
HERE="$(dirname "$(readlink -f "$0")")"

# Resolve agent id: persona pin (authoritative) -> whoami (theft-guarded).
AGENT_ID=""
# Pin dir = git-root/.claude/.persona — SAME location /persona and session-persona.sh
# use (was $PWD before; git-root is correct when cwd is a subdir).
PIN_DIR="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")/.claude/.persona"
if [[ -n "${CLAUDE_SESSION_ID:-}" ]]; then
    PIN_FILE="$PIN_DIR/session-$CLAUDE_SESSION_ID"
    [[ -f "$PIN_FILE" ]] && AGENT_ID=$(tr -d '[:space:]' < "$PIN_FILE" 2>/dev/null)
fi
if [[ -z "$AGENT_ID" ]]; then
    AGENT_ID=$("$BRIDGE" whoami 2>/dev/null | grep "^Effective identity:" | awk '{print $3}')
    if [[ -n "$AGENT_ID" && -n "${CLAUDE_SESSION_ID:-}" && -f "$HOME/.agent-bridge/agents.json" ]]; then
        REG_SID=$(python3 -c "
import json
try:
    d=json.load(open('$HOME/.agent-bridge/agents.json'))
    a=next((x for x in d.get('agents',[]) if x.get('id')=='$AGENT_ID'),{})
    print(a.get('session_id',''))
except: pass" 2>/dev/null)
        SHORT_SID="${CLAUDE_SESSION_ID:0:8}"
        if [[ -n "$REG_SID" && "$REG_SID" != "$SHORT_SID" && "$REG_SID" != "$CLAUDE_SESSION_ID" ]]; then
            echo "BRIDGE RESUME: whoami '$AGENT_ID' belongs to session $REG_SID (mine: $SHORT_SID). REFUSING cwd-fallback. Run /persona to pin, then announce."
            exit 0
        fi
    fi
fi
[[ -z "$AGENT_ID" ]] && exit 0

# --- QUALIFY-ON-CONFLICT (kb-72f717.3) — replaces the blind --steal flapping engine ---
# This SessionStart re-announce is the AUTOMATIC path and NEVER --steals. (Explicit takeover
# lives at the action site: the /persona command and goose POST /bridge/identity do
# `announce --steal` themselves.) If a DIFFERENT, LIVE session currently holds this bridge id
# — detected via the per-session CONTENT cursor (~/.agent-bridge/<id>.cursor holds the
# holder's session_id; mtime <120s = live, written each turn by bridge-inject) — we QUALIFY
# to <id>#<shortsid> and repin instead of stealing (kb-86r D1 lower-session-id-keeps-bare
# convergence, D2 no-reclaim). A stale/dead holder -> keep the bare name (D3 refined: qualify
# only against a LIVE distinct session). BARE_ID is preserved for the registry role lookup
# (a freshly-qualified id has no registry entry yet).
BARE_ID="$AGENT_ID"
SHORT_SID="${CLAUDE_SESSION_ID:0:8}"
_cursor="$HOME/.agent-bridge/${BARE_ID}.cursor"
_hold=$(cat "$_cursor" 2>/dev/null | tr -d '[:space:]')
_age=$(( $(date +%s) - $(stat -c %Y "$_cursor" 2>/dev/null || echo 0) ))
if [[ -n "$_hold" && -n "${CLAUDE_SESSION_ID:-}" && "$_hold" != "$CLAUDE_SESSION_ID" && "$_age" -lt 120 ]]; then
    AGENT_ID="${BARE_ID}#${SHORT_SID}"
    echo "BRIDGE RESUME: bare id '$BARE_ID' is held by a LIVE session (${_hold:0:8}, ${_age}s ago) — qualified to '$AGENT_ID' (no steal; kb-86r). Run /persona explicitly to take it over."
fi

# Persist the (possibly-qualified) id as the authoritative session pin (kb-86r — stops
# resolution falling through to the last-writer registry).
if [[ -n "${CLAUDE_SESSION_ID:-}" ]]; then
    mkdir -p "$PIN_DIR" 2>/dev/null \
        && printf '%s\n' "$AGENT_ID" > "$PIN_DIR/session-$CLAUDE_SESSION_ID" 2>/dev/null
fi

# Re-announce (registry refresh after compaction) — NEVER --steal here. Role is looked up by
# the BARE id (the qualified id has no prior entry); the announce registers the resolved id.
AGENTS_JSON="$HOME/.agent-bridge/agents.json"
if [[ -f "$AGENTS_JSON" ]]; then
    read -r ROLE FOCUS OFFERING < <(python3 -c "
import json
try:
    d=json.load(open('$AGENTS_JSON'))
    a=next((x for x in d.get('agents',[]) if x.get('id')=='$BARE_ID'),{})
    print(a.get('role','-'), a.get('focus','-'), a.get('offering','-'))
except: print('- - -')" 2>/dev/null)
    [[ "$ROLE" != "-" && -n "$ROLE" ]] && "$BRIDGE" announce --id "$AGENT_ID" --role "$ROLE" \
        --focus "resumed after compaction" --offering "${OFFERING#-}" \
        --directed "checking the bridge for missed messages" </dev/null 2>/dev/null
fi

echo "BRIDGE RESUME [$AGENT_ID]: announced. Idle reachability is automatic (asyncRewake Stop hook) — no watcher to launch."

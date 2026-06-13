#!/bin/bash
# SessionStart hook (kb-2os.3): restore bridge registration for the kb-bridge.
#
# The MESSAGE transport is now the kb-server (inject via GET /bridge/messages,
# send via POST /bridge/send, wake via the SSE watcher). The REGISTRY (agents.json:
# announce/whoami/identity) stays on the bridge binary — it's the agent directory,
# not the message channel, and there is no kb-server registry-write endpoint.
#
# Invariants at session start: (1) agent announced (session_id refreshed),
# (2) the BRIDGE PROTOCOL injected (now points at the kb SSE watcher), (3) a single
# kb SSE watcher running (launched by the agent via run_in_background; the Stop
# launcher reminds). Draining is handled by bridge-inject (kb-server) on the first
# tool call — no recv here.
BRIDGE="$HOME/.agent-bridge/bridge"
[[ ! -x "$BRIDGE" ]] && exit 0
HERE="$(dirname "$(readlink -f "$0")")"

# BRIDGE PROTOCOL — main session only (SessionStart does not fire for sub-agents).
cat <<BRIDGEDOC
BRIDGE PROTOCOL (main session only — sub-agents never run bridge commands):
- Peer messages are AUTO-INJECTED at every tool call + user prompt via the kb-SERVER
  (GET /bridge/messages, cursor-tracked) — you do NOT need the watcher while WORKING.
- The kb SSE WATCHER is for IDLE reachability ONLY. Launch it as its OWN Bash call with
  BOTH run_in_background:true AND timeout:600000 (REQUIRED — without the timeout param the
  bg task is capped and the watcher dies as 'failed' exit 144):
      bash "$HERE/kb-bridge-watch.sh" <your-id>
  It holds the SSE ~9 min then EXITS CLEANLY (a quiet timeout, EMPTY output) — relaunch
  ONE when that completion fires. On a real peer message it prints 'BRIDGE_WAKE <json>'
  and EXITS (the task-exit notification wakes you) → relaunch ONE. Empty output ⇒ quiet
  timeout; BRIDGE_WAKE ⇒ real message; either way relaunch exactly ONE.
  It connects to the kb-server SSE and starts at the tail (only NEW messages wake).
  It IGNORES event:announce frames (peers joining — non-actionable, surfaced by
  injection instead), so it wakes ONLY for real directed/actionable messages. It is
  torn down automatically when you go active (a user prompt) and relaunched at the
  next Stop, so it is alive — and can only wake — while you are IDLE.
  Relaunch at session start, after compaction, and after a real wake — NOT every turn.
  Detect liveness by the kb-bridge-watch.sh process; keep exactly ONE live.
- SEND via the kb-server: POST http://127.0.0.1:8765/bridge/send
  {"from":<id>,"to":<id|list>,"subject":..,"body":..,"reply_to":..,"needs_reply":..}.
  (The ~/.agent-bridge/bridge binary still works for registry ops: announce/whoami/agents.)
- Owed replies (inbound --needs-reply you haven't answered) are surfaced at Stop from the
  kb-server feed; close one by sending a reply with reply_to=<id>.
BRIDGEDOC

# Resolve agent id: persona pin (authoritative) -> whoami (theft-guarded).
AGENT_ID=""
if [[ -n "${CLAUDE_SESSION_ID:-}" ]]; then
    PIN_FILE="$PWD/.claude/.persona/session-$CLAUDE_SESSION_ID"
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

# Re-announce (registry op) to refresh session_id after compaction.
AGENTS_JSON="$HOME/.agent-bridge/agents.json"
if [[ -f "$AGENTS_JSON" ]]; then
    read -r ROLE FOCUS OFFERING < <(python3 -c "
import json
try:
    d=json.load(open('$AGENTS_JSON'))
    a=next((x for x in d.get('agents',[]) if x.get('id')=='$AGENT_ID'),{})
    print(a.get('role','-'), a.get('focus','-'), a.get('offering','-'))
except: print('- - -')" 2>/dev/null)
    STEAL_FLAG=""; [[ -n "$PIN_FILE" && -f "$PIN_FILE" ]] && STEAL_FLAG="--steal"
    [[ "$ROLE" != "-" && -n "$ROLE" ]] && "$BRIDGE" announce --id "$AGENT_ID" --role "$ROLE" \
        --focus "resumed after compaction" --offering "${OFFERING#-}" \
        --directed "checking the bridge for missed messages" $STEAL_FLAG </dev/null 2>/dev/null

fi

echo "BRIDGE RESUME [$AGENT_ID]: announced. Launch the kb SSE watcher via run_in_background:true on first tool call (Stop launcher reminds): bash $HERE/kb-bridge-watch.sh $AGENT_ID"

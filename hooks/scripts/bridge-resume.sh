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
  run_in_background:true and NO timeout parameter — OMIT timeout entirely. An omitted
  timeout runs it UNBOUNDED, so it holds for DAYS. Do NOT pass timeout:N — that caps the
  bg task at N (<=10min) and the harness kills it ('failed' exit 144); that cap-kill was
  the old 144 bug, NOT the SSE.
      kb bridge watch <your-id>
  It holds the SSE for days until a REAL peer message — then it prints 'BRIDGE_WAKE <json>'
  and EXITS (the task-exit notification wakes you) → relaunch ONE. It also exits (empty) if
  the kb-server closes the connection (e.g. a server restart) → relaunch ONE. Relaunch is
  rare (only on a wake or a server restart), NOT a ~9-min cycle.
  It connects to the kb-server SSE and starts at the tail (only NEW messages wake).
  It IGNORES event:announce frames (peers joining — non-actionable, surfaced by
  injection instead), so it wakes ONLY for real directed/actionable messages. It is
  torn down automatically when you go active (a user prompt) and relaunched at the
  next Stop, so it is alive — and can only wake — while you are IDLE.
  Relaunch at session start, after compaction, and after a real wake — NOT every turn.
  Detect liveness by the kb-bridge-watch.sh process; keep exactly ONE live.
- SEND via `kb bridge send <to> "<subject>" --needs-reply` (body on stdin or --body).
  This is the canonical send path.
- JOIN/announce via `kb bridge announce …`; drain explicitly with `kb bridge recv` (rarely
  needed — peer messages auto-inject every turn).
- Owed replies (inbound --needs-reply you haven't answered) are surfaced at Stop from the
  kb-server feed; close one with `kb bridge send <peer> "<re>" --reply <id>`.
BRIDGEDOC

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

# kb-86r: persist the resolved identity as the AUTHORITATIVE session pin if none
# exists yet. Reaching here means a CLEAN resolve — the whoami theft-guard above
# already REFUSED (exit 0) any registry entry whose session_id isn't ours — so we
# never pin a stolen identity. Without this, an agent that was announced but never
# ran /persona has no pin, and resolution falls through to the last-writer-wins
# registry: the identity-collision root cause (kb-86r). Writing the pin makes THIS
# session's identity stable regardless of later registry stomps.
if [[ -n "${CLAUDE_SESSION_ID:-}" && ! -f "$PIN_DIR/session-$CLAUDE_SESSION_ID" ]]; then
    mkdir -p "$PIN_DIR" 2>/dev/null \
        && printf '%s\n' "$AGENT_ID" > "$PIN_DIR/session-$CLAUDE_SESSION_ID" 2>/dev/null \
        && echo "BRIDGE RESUME [$AGENT_ID]: wrote authoritative session pin (kb-86r — resolution no longer falls back to the last-writer registry)."
fi

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

echo "BRIDGE RESUME [$AGENT_ID]: announced. Launch the kb SSE watcher via run_in_background:true on first tool call (Stop launcher reminds): kb bridge watch $AGENT_ID"

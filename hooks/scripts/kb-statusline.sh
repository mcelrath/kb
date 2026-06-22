#!/bin/bash
# kb plugin status-line SEGMENT (opt-in). Reads the Claude Code status-line stdin
# JSON and emits ONLY the kb-specific segment, for composition into your own status
# line (Claude Code allows just ONE statusLine command, so a plugin cannot register
# its own — it provides a segment you call). Wire it from your status line script:
#
#   kb_seg=$(printf '%s' "$input" | /path/to/kb-statusline.sh)
#   [[ -n "$kb_seg" ]] && line1_parts+=("$kb_seg")
#
# CHEAP BY CONTRACT: local file reads only — NO kb/KB construction, network, or DB
# (a status line re-renders constantly; it must never block on the embedding server
# or build a KnowledgeBase). Shows:
#   🎭 <persona>   the adopted persona / bridge id (kb plugin pin)
#   🛑 kb-embed down   when the embedding server is unreachable (cached probe)
set -u

input=$(cat 2>/dev/null)
[[ -z "$input" ]] && exit 0

sid=$(printf '%s' "$input" | jq -r '.session_id // ""' 2>/dev/null)
cwd=$(printf '%s' "$input" | jq -r '.cwd // .workspace.current_dir // ""' 2>/dev/null)

CYAN='\033[0;36m'; RED='\033[0;31m'; RESET='\033[0m'
parts=()

# Persona / bridge id for this session. Resolve in two cheap local steps:
#   1. the kb plugin persona pin <git-root>/.claude/.persona/session-<id> (a /persona
#      adoption), then
#   2. the bridge registry ~/.agent-bridge/agents.json by session_id (covers an id
#      claimed via `bridge announce`/`--from` without an explicit /persona pin).
# Both are local file reads. Empty when this session has no bridge identity.
persona=""
if [[ -n "$sid" && -n "$cwd" ]]; then
    proot=$(cd "$cwd" 2>/dev/null && git rev-parse --show-toplevel </dev/null 2>/dev/null)
    pin="${proot:-$cwd}/.claude/.persona/session-${sid}"
    [[ -f "$pin" ]] && persona=$(tr -d '[:space:]' < "$pin" 2>/dev/null)
fi
if [[ -z "$persona" && -n "$sid" && -f "$HOME/.agent-bridge/agents.json" ]]; then
    persona=$(jq -r --arg s "$sid" '.agents[] | select(.session_id == $s) | .id' \
              "$HOME/.agent-bridge/agents.json" 2>/dev/null | head -1)
    [[ "$persona" == "null" ]] && persona=""
fi
[[ -n "$persona" ]] && parts+=("${CYAN}🎭 ${persona}${RESET}")

# Embedding-server health: read ash_health's 60s cache ("emb=0,llm=0"); do NOT probe.
# emb=1 => the embedding server is down => kb search + surfacing are BLIND.
cache="/tmp/.kbinfra_health_cache"
if [[ -f "$cache" ]]; then
    c=$(cat "$cache" 2>/dev/null)
    [[ "$c" == *"emb=1"* ]] && parts+=("${RED}🛑 kb-embed down${RESET}")
fi

(( ${#parts[@]} )) && printf '%b' "${parts[*]}"
exit 0

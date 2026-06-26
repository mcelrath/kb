#!/bin/bash
# /persona helper — list personas, or adopt one, in ONE invocation.
# Does everything the SKILL used to ask the agent to improvise (dir-resolution, validation,
# session pin, bridge announce --steal, and emitting the COMPOSED role), so /persona costs a
# single tool call instead of a fan-out of ls/find/cat. The kb plugin owns the persona subsystem.
#
# Usage:  persona.sh             # list
#         persona.sh <name>      # adopt (name or name-suffix; base = before first '-')
set -u

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Persona dir: git root -> cwd -> home (matches session-persona.sh resolution).
PERSONA_DIR=""
GITROOT=$(git rev-parse --show-toplevel 2>/dev/null)
for d in "$GITROOT/.claude/agents/personas" "$PWD/.claude/agents/personas" "$HOME/.claude/agents/personas"; do
    [ -n "$d" ] && [ -d "$d" ] && { PERSONA_DIR="$d"; break; }
done

# _fm <file> <key>: print one frontmatter value (single real parser, no name-regex).
_fm() {
    python3 - "$1" "$2" <<'PY'
import sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
key = sys.argv[2]
fm = {}
if t.startswith("---"):
    e = t.find("---", 3)
    if e != -1:
        for ln in t[3:e].splitlines():
            if ":" in ln:
                k, _, v = ln.partition(":")
                fm[k.strip()] = v.strip().strip('"').strip("'")
print(fm.get(key, ""))
PY
}

# An augmentation layer (carries applies_to:) is not independently adoptable — hide from lists.
_is_augmentation() { [ -n "$(_fm "$1" applies_to)" ]; }

if [ -z "$PERSONA_DIR" ]; then
    echo "No personas directory (looked: git-root/.claude/agents/personas, cwd, ~/.claude). This project defines no personas."
    exit 0
fi

NAME="${1:-}"

# ---- LIST ----
if [ -z "$NAME" ]; then
    echo "Available personas (dir: $PERSONA_DIR):"
    found=0
    for f in "$PERSONA_DIR"/*.md; do
        [ -e "$f" ] || continue
        _is_augmentation "$f" && continue
        echo "  $(basename "$f" .md) — $(_fm "$f" description)"
        found=1
    done
    [ "$found" -eq 0 ] && echo "  (none)"
    echo "Adopt with: /persona <name>"
    exit 0
fi

# ---- ADOPT ----
NAME=$(printf '%s' "$NAME" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
BASE="${NAME%%-*}"
PERSONA_FILE="$PERSONA_DIR/$BASE.md"
if [ ! -f "$PERSONA_FILE" ]; then
    echo "Unknown persona '$BASE'. Valid:"
    for f in "$PERSONA_DIR"/*.md; do
        [ -e "$f" ] || continue
        _is_augmentation "$f" && continue
        echo "  $(basename "$f" .md)"
    done
    exit 1
fi

FULL_ID="$NAME"
ROLE="$(_fm "$PERSONA_FILE" role)"
[ -z "$ROLE" ] && ROLE="$(_fm "$PERSONA_FILE" archetype)"
[ -z "$ROLE" ] && ROLE="$BASE"
OFFERING="$(_fm "$PERSONA_FILE" offering)"
[ -z "$OFFERING" ] && OFFERING="$(_fm "$PERSONA_FILE" description)"

# Pin where session-persona.sh reads it: dirname(dirname(PERSONA_DIR))/.persona == .claude/.persona
MARKER_DIR="$(dirname "$(dirname "$PERSONA_DIR")")/.persona"
SID="${CLAUDE_SESSION_ID:-unknown}"
mkdir -p "$MARKER_DIR" && printf '%s\n' "$FULL_ID" > "$MARKER_DIR/session-${SID}"

# Explicit adoption = takeover: --steal (the SessionStart re-announce never steals).
BR="$HOME/.agent-bridge/bridge"
if [ -x "$BR" ]; then
    "$BR" announce --id "$FULL_ID" --role "$ROLE" --focus "adopted via /persona" \
        --offering "$OFFERING" --directed "ping $FULL_ID" --steal </dev/null >/dev/null 2>&1 \
        && echo "pinned + claimed bridge id '$FULL_ID' (--steal)"
fi

echo "ACTIVE PERSONA: $BASE (bridge id $FULL_ID). Re-loads after compaction via the SessionStart hook."
echo "----- composed operating role (adopt this in full) -----"
python3 "$PLUGIN_ROOT/hooks/scripts/lib/persona_compose.py" "$PERSONA_FILE" "$PLUGIN_ROOT" "$PERSONA_DIR" 2>/dev/null || cat "$PERSONA_FILE"
echo "----- end -----"

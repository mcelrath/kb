#!/bin/bash
# SessionStart hook (kb plugin; moved from the global ~/.claude/hooks/session/ — kb-72f717.1):
# inject this agent's PERSONA into context at every session start / resume / compact, so the
# persona survives compaction (a one-shot /slash command would be summarized away; this
# re-injects). kb now OWNS the persona+bridge subsystem (the persona machinery + bridge
# identity resolution all ship in the plugin).
#
# Projects supply their own persona files at <project>/.claude/agents/personas/<name>.md;
# ~/.claude/agents/personas/ is a cross-project fallback. The persona name IS the bridge id —
# the suffix after the first '-' is stripped (tip-mathlib -> tip). No id<->name mapping.
#
# Identity resolves by THIS session's own identity ONLY (AGENT_ID env or session_id), never
# the cwd fallback: multiple sessions share a cwd, and a cwd-based resolution would hand them
# all a neighbor's persona. A session that cannot identify itself gets NO persona (safe).

BRIDGE="$HOME/.agent-bridge/bridge"
[[ ! -x "$BRIDGE" ]] && exit 0

# Plugin root (for the persona composer + archetype dir). Env first, else 2 levels up
# from this script (hooks/scripts/) — matches kb-error-extract.sh:7 / kb-precompact.sh:6.
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# session_id: env, else from the hook-input JSON on stdin.
SID="${CLAUDE_SESSION_ID:-}"
if [[ -z "$SID" && ! -t 0 ]]; then
    SID=$(python3 -c "import sys,json; print(json.load(sys.stdin).get('session_id',''))" 2>/dev/null)
fi

WHOAMI=$("$BRIDGE" whoami 2>/dev/null)
_res() { printf '%s\n' "$WHOAMI" | sed -n "s/^[[:space:]]*by $1:[[:space:]]*//p" | head -1; }
AGENT_ID=$(_res "AGENT_ID env")
[[ "$AGENT_ID" == "(no match)" || -z "$AGENT_ID" ]] && AGENT_ID=$(_res "session_id")
[[ "$AGENT_ID" == "(no match)" ]] && AGENT_ID=""
# A session-id pin is self-identifying and must still load; bail only if BOTH absent.
[[ -z "$AGENT_ID" && -z "$SID" ]] && exit 0

# Locate the persona dir: git-root (handles subdir cwd), then cwd, then global.
PERSONA_DIR=""
GITROOT=$(git rev-parse --show-toplevel 2>/dev/null)
for d in "$GITROOT/.claude/agents/personas" "$PWD/.claude/agents/personas" "$HOME/.claude/agents/personas"; do
    [[ -n "$d" && -d "$d" ]] && { PERSONA_DIR="$d"; break; }
done
[[ -z "$PERSONA_DIR" ]] && exit 0
MARKER_DIR="$(dirname "$(dirname "$PERSONA_DIR")")/.persona"   # .claude/.persona

# 1a. session-id pin (highest priority — set by /persona; survives compaction).
PERSONA=""
if [[ -n "$SID" && -f "$MARKER_DIR/session-$SID" ]]; then
    PERSONA=$(tr -d '[:space:]' < "$MARKER_DIR/session-$SID")
fi
# 1b. explicit agent-id override pin.
if [[ -z "$PERSONA" && -n "$AGENT_ID" && -f "$MARKER_DIR/$AGENT_ID" ]]; then
    PERSONA=$(tr -d '[:space:]' < "$MARKER_DIR/$AGENT_ID")
fi
# 2. the bridge agent-id IS the persona name (no mapping layer).
[[ -z "$PERSONA" && -n "$AGENT_ID" ]] && PERSONA="$AGENT_ID"
[[ -z "$PERSONA" ]] && exit 0

# Base name = persona/bridge-id with the first-hyphen suffix stripped
# (tip-mathlib -> tip); the base determines which persona file loads.
BASE="${PERSONA%%-*}"
PERSONA_FILE="$PERSONA_DIR/$BASE.md"
[[ ! -f "$PERSONA_FILE" ]] && PERSONA_FILE="$PERSONA_DIR/$PERSONA.md"
[[ ! -f "$PERSONA_FILE" ]] && exit 0

BRIDGE_ID="${AGENT_ID:-$PERSONA}"
echo "ACTIVE PERSONA: $BASE (agent $BRIDGE_ID). This is your BINDING operating role for the session — it persists across compaction because this SessionStart hook re-injects it. Read any files it references in full and adopt it now:"
echo "---"
# Compose archetype (L1) + augmentation (L2) + instance body per the persona's frontmatter.
# Fail-safe: any composer error falls back to the flat persona file, so a parser bug can
# never blank a session's persona (same discipline as the SID parse above).
COMPOSED=$(python3 "$PLUGIN_ROOT/hooks/scripts/lib/persona_compose.py" \
    "$PERSONA_FILE" "$PLUGIN_ROOT" "$PERSONA_DIR" 2>/dev/null)
if [[ $? -eq 0 && -n "$COMPOSED" ]]; then
    printf '%s\n' "$COMPOSED"
else
    cat "$PERSONA_FILE"
fi
echo "---"
echo "(To switch: /persona <name>. To list: /persona)"
echo ""
echo "RESUME PROTOCOL (compact/resume/clear — execute IN ORDER before touching the work queue):"
echo "  1. kb bridge recv $BRIDGE_ID        # drain YOUR mail FIRST — do not touch the tracker before this"
echo "  2. kbt list --status in_progress --assignee $BRIDGE_ID   # claimed work FIRST"
echo "  3. git -C <your-repo> log --oneline -5                  # sync the view"
echo "  4. ONLY IF 1-3 empty/clear: kbt ready                   # unclaimed = FALLBACK, not default"

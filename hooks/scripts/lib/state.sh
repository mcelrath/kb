# Shared session-state root for shell hooks. Source this; it exports STATE_DIR.
#
# Persistent (reboot-surviving) replacement for the old /tmp/claude-kb-state
# tmpfs root (kb-h3b). Must agree with the python side: lib/_state.py resolves
# STATE_DIR identically (CLAUDE_STATE_DIR override, else ~/.claude/state).
#
# CLAUDE_STATE_DIR lets the test harness redirect the root to a throwaway dir
# (mirrors test_wake_filter.sh using a throwaway $HOME). GC runs at SessionStart
# against the resolved root, so the override cannot be used to dodge cleanup of
# the real root — it only redirects the overriding process's own state.
#
# Usage:  source "$(dirname "$0")/../lib/state.sh"   # then use "$STATE_DIR/..."
export STATE_DIR="${CLAUDE_STATE_DIR:-$HOME/.claude/state}"
mkdir -p "$STATE_DIR" 2>/dev/null || true

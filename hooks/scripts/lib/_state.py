"""Shared session-state helpers for kb hooks.

Locates the claude-kb-state session file using:
  1. $CLAUDE_SESSION_ID env var — fastest; set by session-init.sh, available
     in PreToolUse/PostToolUse hook environment.
  2. PPID walk up /proc/{pid}/status PPid: chain — fallback for SubagentStop
     and any other context where the env var is not inherited.
"""
import json
import os

# Persistent (reboot-surviving) session-state root (kb-h3b). Was
# /tmp/claude-kb-state (tmpfs, wiped on reboot — lost sub-TTL state like the
# bridge owed-deferred log). CLAUDE_STATE_DIR override is for test isolation;
# must agree with the shell side (hooks/lib/state.sh).
STATE_DIR = os.environ.get('CLAUDE_STATE_DIR') or os.path.expanduser('~/.claude/state')
_MAX_WALK = 8


def kb_project_for_path(fpath: str) -> str | None:
    """Resolve the KB project name for a file by walking up to the nearest
    `.claude/kb-project.json` ({"kb_project": "<name>"}). Generic replacement
    for the old hardcoded path->project map (kb-bp4 P6). Returns None if no
    project config is found up-tree."""
    if not fpath:
        return None
    d = os.path.dirname(os.path.abspath(fpath))
    prev = None
    while d and d != prev:
        cfg = os.path.join(d, '.claude', 'kb-project.json')
        if os.path.isfile(cfg):
            try:
                with open(cfg) as fh:
                    return json.load(fh).get('kb_project') or None
            except Exception:
                return None
        prev, d = d, os.path.dirname(d)
    return None


def get_session_id() -> str | None:
    """Return the current Claude session ID, or None if unavailable."""
    # Prefer env var — O(1), works for all normal PreToolUse/PostToolUse hooks
    sid = os.environ.get('CLAUDE_SESSION_ID', '').strip()
    if sid:
        return sid

    # PPID walk — for SubagentStop and other contexts where env var is absent
    try:
        pid = os.getpid()
        for _ in range(_MAX_WALK):
            try:
                with open(f'/proc/{pid}/status') as fh:
                    ppid = None
                    for line in fh:
                        if line.startswith('PPid:'):
                            ppid = int(line.split()[1])
                            break
                if ppid is None:
                    break
                pid = ppid
            except OSError:
                break
            session_file = os.path.join(STATE_DIR, f'session-{pid}')
            if os.path.exists(session_file):
                with open(session_file) as fh:
                    return fh.read().strip() or None
    except Exception:
        pass

    return None


def state_path(suffix: str) -> str | None:
    """Return $STATE_DIR/<session_id>-<suffix>, or None if session unknown."""
    sid = get_session_id()
    if not sid:
        return None
    return os.path.join(STATE_DIR, f'{sid}-{suffix}')

"""Shared session-state helpers for kb hooks.

Locates the claude-kb-state session file using:
  1. $CLAUDE_SESSION_ID env var — fastest; set by session-init.sh, available
     in PreToolUse/PostToolUse hook environment.
  2. PPID walk up /proc/{pid}/status PPid: chain — fallback for SubagentStop
     and any other context where the env var is not inherited.
"""
import os

STATE_DIR = '/tmp/claude-kb-state'
_MAX_WALK = 8


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

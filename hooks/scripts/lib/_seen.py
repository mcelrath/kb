"""Session-scoped advisory dedup for kb hooks.

Uses $STATE_DIR/<session_id>-hook-seen (one key per line, fcntl-locked).
filter_unseen() atomically reads existing keys, filters candidates, appends
new ones, and returns the new-only subset.

Key conventions (callers must use these prefixes):
  sym:{name}        — python_symbols CANONICAL advisory (NOT retired — always surface)
  notation:{sym}    — notations table advisory

Keys NOT deduplicated (context-specific, worth re-surfacing):
  frac/kb-value hits, LEAN theorems, ALREADY-PROVEN, LAKE-ERROR

Falls back to returning all keys when session state is unavailable.
"""
import fcntl
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _state import STATE_DIR, state_path  # noqa: E402


def filter_unseen(keys: list[str]) -> list[str]:
    """Return only keys not yet surfaced this session; atomically marks them seen.

    Thread/process safe via fcntl.LOCK_EX on the seen file.
    Empty or unavailable state → returns all keys (safe degradation).
    """
    if not keys:
        return []

    path = state_path('hook-seen')
    if path is None:
        return keys  # no session state — pass everything through

    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(path, 'a+') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            already = set(fh.read().splitlines())
            new_keys = [k for k in keys if k not in already]
            if new_keys:
                fh.seek(0, 2)  # seek to end
                fh.write('\n'.join(new_keys) + '\n')
            return new_keys
    except Exception:
        return keys  # on any error, pass everything through

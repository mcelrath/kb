"""Write-retry helper for the SQLite layer.

`PRAGMA busy_timeout` makes SQLite *wait* on a locked resource, but it can still
surface `OperationalError: database is locked/busy` when contention outlasts the
timeout or when a lock upgrade would deadlock. Under many concurrent writers
(agents + hooks + the server all hitting one db) that happens. This decorator
makes a write operation BLOCK — retrying with exponential backoff until it lands
or a generous ceiling is hit — instead of failing the caller.

Usage: decorate a repository WRITE method (one that does its execute(s) + commit
and is safe to re-run from the top — a failed attempt committed nothing). On a
lock error the method's pending transaction is rolled back (via the instance's
`.conn`) before the next attempt, so retries start clean.
"""
from __future__ import annotations

import functools
import random
import sqlite3
import time
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

# Generous ceiling: block up to this long across all retries before giving up.
# Far longer than any legitimate single-statement write lock; a failure past
# this is a real problem (deadlock / wedged writer), not transient contention.
_MAX_WAIT_S = 120.0
_BASE_S = 0.02
_CAP_S = 1.0


def _is_lock_error(exc: sqlite3.OperationalError) -> bool:
    m = str(exc).lower()
    return "locked" in m or "busy" in m


def retry_on_locked(fn: _F) -> _F:
    """Block-and-retry a write method on `database is locked/busy` (exp backoff)."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if not _is_lock_error(exc) or (time.monotonic() - start) >= _MAX_WAIT_S:
                    raise
                # Roll back the failed attempt's pending transaction so the retry
                # starts clean (the method re-runs its statements from the top).
                conn = getattr(args[0], "conn", None) if args else None
                if conn is not None:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                delay = min(_CAP_S, _BASE_S * (2 ** attempt)) + random.uniform(0, _BASE_S)
                time.sleep(delay)
                attempt += 1
    return wrapper  # type: ignore[return-value]

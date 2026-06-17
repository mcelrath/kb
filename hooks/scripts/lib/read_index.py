"""Session-scoped read-index: what THIS session has actually read vs. only had mentioned.

Backs the read-before-modify/claim gates (Phase C). Same substrate as _seen.py — a
single fcntl-locked ledger file at $STATE_DIR/<session_id>-read-index, one prefixed
key per line:

  read:{abspath}     — a file whose CONTENT entered this session via a Read tool result
  mfile:{abspath}    — a file MENTIONED (sub-agent report / kb retrieval) but not (yet) read
  msym:{name}        — a symbol mentioned but not read

"read" beats "mentioned": is_read() only consults read: keys. The ledger is per-session,
so a sub-agent reading X does NOT mark the dispatcher's session as having read X — which is
exactly the failure mode the gate catches (dispatcher acts on a sub-agent's claim about an
unread file).

Fail-open everywhere: if session state is unavailable, treat nothing as tracked (callers
must not block on a None/empty result — see gate logic in Phase C).
"""
import fcntl
import os

from _state import STATE_DIR, state_path  # noqa: E402


def _norm(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path)) if path else ""


def _append(keys: list[str]) -> None:
    """Append keys to the ledger (deduped against existing), fcntl-locked. Fail-open."""
    if not keys:
        return
    path = state_path('read-index')
    if path is None:
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(path, 'a+') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            already = set(fh.read().splitlines())
            new = [k for k in keys if k not in already]
            if new:
                fh.seek(0, 2)
                fh.write('\n'.join(new) + '\n')
    except Exception:
        pass


def _load() -> set[str]:
    path = state_path('read-index')
    if path is None:
        return set()
    try:
        with open(path) as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            return set(fh.read().splitlines())
    except Exception:
        return set()


def mark_read(paths: list[str], syms: list[str] | None = None) -> None:
    keys = [f'read:{_norm(p)}' for p in paths if p]
    if syms:
        keys += [f'read-sym:{s}' for s in syms if s]
    _append(keys)


def mark_mentioned_files(paths: list[str]) -> None:
    _append([f'mfile:{_norm(p)}' for p in paths if p])


def mark_mentioned_syms(names: list[str]) -> None:
    _append([f'msym:{s}' for s in names if s])


def is_read(path: str) -> bool:
    return f'read:{_norm(path)}' in _load()


def unread_files(paths: list[str]) -> list[str]:
    """Return the subset of paths NOT marked read this session (normalized, deduped)."""
    keys = _load()
    out, seen = [], set()
    for p in paths:
        n = _norm(p)
        if n and n not in seen and f'read:{n}' not in keys:
            seen.add(n)
            out.append(n)
    return out


def mentioned_unread_files() -> list[str]:
    """Files mentioned this session that were never read (mfile: minus read:)."""
    keys = _load()
    read = {k[len('read:'):] for k in keys if k.startswith('read:')}
    return [k[len('mfile:'):] for k in keys if k.startswith('mfile:') and k[len('mfile:'):] not in read]

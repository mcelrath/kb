#!/usr/bin/env python3
"""G2: open-issues-surface — PreToolUse on Agent/Bash(bridge send).

When an agent is dispatched or a bridge message is sent, surface relevant
issues from the kb issues table as context advisories:
  [OPEN-ISSUE: <id> (P<pri>) — <title>]     open / in_progress issues
  [RESOLVED-ISSUE: <id> — <title>]          recently closed (prior art)

Uses the kb SQLite issues table directly — no bd subprocess.
Isolation: project-column filter via kb_project_for_path, same mechanism as
  compose_time_check.py. The .beads/-dir gate is gone; project column is the
  cross-repo isolation boundary.
Search: vector similarity (embed prompt -> issues_vec); falls back to FTS5
  when the embedding server is down (kb-zma degradation — never crashes/blocks).
Fires PreToolUse/Task (Agent) and PreToolUse/Bash (bridge send only).
Advisory only (exit 0 always).
"""
import sys
import json
import os
import re
import sqlite3
import struct
import urllib.request

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'lib'))
from _seen import filter_unseen  # noqa: E402
from _state import kb_project_for_path  # noqa: E402
from _db import kb_db_path  # noqa: E402
try:
    from ash_health import ash_down
except Exception:
    def ash_down(): return False  # type: ignore[misc]

_MAX_SURFACE = 4           # max open issues to surface per hook call
_MAX_CLOSED = 2            # max recently-closed issues to surface
_RECENTLY_CLOSED_DAYS = 14  # surface closed issues up to this many days old
_VECTOR_K = 15             # KNN candidates before project/status filter


def _db_path() -> str:
    return kb_db_path()  # shared resolver (kb-05n)


def _current_project() -> str | None:
    """Resolve the KB project for the current working directory."""
    cwd = os.environ.get('CLAUDE_PROJECT_DIR', '') or os.getcwd()
    return kb_project_for_path(os.path.join(cwd, '_probe'))


def _embed_prompt(text: str) -> bytes | None:
    """Call the embedding server; return packed float32 blob or None on failure."""
    url = os.environ.get('KB_EMBEDDING_URL', 'http://ash:8081/embedding')
    fmt = os.environ.get('KB_EMBEDDING_FORMAT', 'llamacpp')
    timeout = int(os.environ.get('KB_EMBED_TIMEOUT', '5'))
    try:
        if fmt == 'openai':
            model = os.environ.get('KB_EMBEDDING_MODEL', 'text-embedding-ada-002')
            payload = json.dumps({'input': [text], 'model': model}).encode()
            req = urllib.request.Request(url, data=payload,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            vec = body['data'][0]['embedding']
        else:
            # llamacpp format
            payload = json.dumps({'content': text}).encode()
            req = urllib.request.Request(url, data=payload,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            vec = body.get('embedding')
        if not vec:
            return None
        return struct.pack(f'{len(vec)}f', *vec)
    except Exception:
        return None


def _search_vector(conn: sqlite3.Connection, blob: bytes,
                   project: str | None) -> list[dict]:
    """KNN search over issues_vec; join to issues for metadata."""
    try:
        rows = conn.execute(
            """SELECT v.id, v.distance
               FROM issues_vec v
               JOIN issues i ON i.id = v.id
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (blob, _VECTOR_K),
        ).fetchall()
    except Exception:
        return []

    results = []
    for iid, dist in rows:
        row = conn.execute(
            """SELECT id, status, priority, title, project, closed_at
               FROM issues WHERE id = ?""",
            (iid,),
        ).fetchone()
        if not row:
            continue
        rid, status, priority, title, proj, closed_at = row
        # Project isolation: skip if BOTH sides have a project and they differ.
        # When project col is NULL (unset), allow through — created before
        # per-project scoping was enforced; belongs to the current context.
        if proj is not None and project is not None and proj != project:
            continue
        similarity = round(1 - (dist ** 2) / 2, 4)
        results.append({
            'id': rid, 'status': status, 'priority': priority,
            'title': title, 'project': proj, 'closed_at': closed_at,
            'similarity': similarity,
        })
    return results


def _search_fts(conn: sqlite3.Connection, text: str, project: str | None) -> list[dict]:
    """FTS5 fallback; project-isolated via issues JOIN (project col or NULL)."""
    tokens = list({
        m.group(1).lower()
        for m in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]{4,})\b', text)
    })
    if not tokens:
        return []
    query_str = ' OR '.join(tokens[:20])
    try:
        if project:
            rows = conn.execute(
                """SELECT i.id, i.status, i.priority, i.title, i.project, i.closed_at
                   FROM issues_fts
                   JOIN issues i ON i.rowid = issues_fts.rowid
                   WHERE issues_fts MATCH ?
                     AND (i.project = ? OR i.project IS NULL)
                   ORDER BY bm25(issues_fts)
                   LIMIT ?""",
                (query_str, project, _VECTOR_K),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT i.id, i.status, i.priority, i.title, i.project, i.closed_at
                   FROM issues_fts
                   JOIN issues i ON i.rowid = issues_fts.rowid
                   WHERE issues_fts MATCH ?
                   ORDER BY bm25(issues_fts)
                   LIMIT ?""",
                (query_str, _VECTOR_K),
            ).fetchall()
    except Exception:
        return []

    return [
        {'id': r[0], 'status': r[1], 'priority': r[2],
         'title': r[3], 'project': r[4], 'closed_at': r[5], 'similarity': None}
        for r in rows
    ]


def _is_recent_closed(closed_at: str | None) -> bool:
    """Return True if closed_at is within RECENTLY_CLOSED_DAYS of now."""
    if not closed_at:
        return False
    try:
        from datetime import datetime, timedelta
        dt = datetime.fromisoformat(closed_at.replace('Z', ''))
        return dt >= datetime.utcnow() - timedelta(days=_RECENTLY_CLOSED_DAYS)
    except Exception:
        return False


def _extract_prompt(tool_name: str, ti: dict) -> str:
    """Extract the text to scan from the tool input."""
    if tool_name in ('Task', 'Agent'):
        return ti.get('prompt', '')
    if tool_name == 'Bash':
        cmd = ti.get('command', '')
        if 'bridge send' not in cmd:
            return ''
        parts = []
        m = re.search(r"<<\s*'?EOF'?\s*\n(.+?)(?:\nEOF\b|\Z)", cmd, re.DOTALL)
        if m:
            parts.append(m.group(1))
        m2 = re.search(r'bridge send\s+\S+\s+"([^"]+)"', cmd)
        if m2:
            parts.append(m2.group(1))
        return '\n'.join(parts)
    return ''


def main() -> None:
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', '')
    ti = data.get('tool_input', {})

    prompt_text = _extract_prompt(tool_name, ti)
    if not prompt_text or len(prompt_text) < 20:
        sys.exit(0)

    db = _db_path()
    if not os.path.exists(db):
        sys.exit(0)

    project = _current_project()

    try:
        conn = sqlite3.connect(db, timeout=3)

        # Verify issues table exists and has rows
        try:
            total = conn.execute('SELECT COUNT(*) FROM issues').fetchone()[0]
        except Exception:
            conn.close()
            sys.exit(0)
        if total == 0:
            conn.close()
            sys.exit(0)

        # Vector search with FTS fallback (kb-zma pattern)
        candidates: list[dict] = []
        if not ash_down():
            blob = _embed_prompt(prompt_text[:2000])
            if blob:
                candidates = _search_vector(conn, blob, project)

        if not candidates:
            # Embedding unavailable or returned nothing — degrade to FTS
            candidates = _search_fts(conn, prompt_text, project)

        conn.close()
    except Exception:
        sys.exit(0)

    if not candidates:
        sys.exit(0)

    # Partition: open/in_progress vs recently-closed (prior art)
    open_issues = [c for c in candidates if c['status'] in ('open', 'in_progress')]
    closed_issues = [c for c in candidates
                     if c['status'] == 'closed' and _is_recent_closed(c.get('closed_at'))]

    open_issues.sort(key=lambda x: (x.get('priority') if x.get('priority') is not None else 99))
    closed_issues.sort(key=lambda x: x.get('closed_at') or '', reverse=True)

    all_items = (
        [(i, 'open') for i in open_issues[:_MAX_SURFACE]] +
        [(i, 'closed') for i in closed_issues[:_MAX_CLOSED]]
    )
    if not all_items:
        sys.exit(0)

    # Dedup via session-scoped seen set; key prefix 'issue:' (distinct from old 'bd:')
    keys = [f'issue:{item["id"]}' for item, _ in all_items]
    new_key_set = set(filter_unseen(keys))

    lines: list[str] = []
    for item, kind in all_items:
        key = f'issue:{item["id"]}'
        if key not in new_key_set:
            continue
        iid = item['id']
        title = (item.get('title') or '?')[:70]
        pri = item.get('priority')
        if kind == 'open':
            pri_str = f' (P{pri})' if pri is not None else ''
            lines.append(f'[OPEN-ISSUE: {iid}{pri_str} — {title}]')
        else:
            lines.append(f'[RESOLVED-ISSUE: {iid} — {title}]')

    if lines:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": "\n".join(lines),
            }
        }))

    sys.exit(0)


if __name__ == '__main__':
    main()

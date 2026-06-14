#!/usr/bin/env python3
"""Patch compose_time_check.py to add query_issues function."""
import re

path = '/home/mcelrath/.claude/hooks/kb/compose_time_check.py'
content = open(path).read()

new_fn = r'''

_RECENTLY_CLOSED_DAYS_CTC = 14   # compose_time_check: surface closed issues up to N days old
_ISSUES_FTS_K = 12


def _is_recent_closed_ctc(closed_at: str | None) -> bool:
    """Return True if closed_at is within _RECENTLY_CLOSED_DAYS_CTC of now."""
    if not closed_at:
        return False
    try:
        from datetime import datetime, timedelta
        dt = datetime.fromisoformat(closed_at.replace('Z', ''))
        return dt >= datetime.utcnow() - timedelta(days=_RECENTLY_CLOSED_DAYS_CTC)
    except Exception:
        return False


def query_issues(conn: sqlite3.Connection, prompt_text: str,
                 project: str | None = None) -> list[str]:
    """Surface [OPEN-ISSUE] and [RESOLVED-ISSUE] advisories from the kb issues table.

    Uses FTS5 token search — no embedding call needed at compute-intent time.
    Project-column isolation mirrors query_db: (project = ? OR project IS NULL).
    Advisory only; returns at most 3 lines.
    """
    advisories: list[str] = []
    try:
        conn.execute('SELECT 1 FROM issues LIMIT 1')
    except Exception:
        return advisories

    tokens = list({
        m.group(1).lower()
        for m in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]{4,})\b', prompt_text)
    })
    if not tokens:
        return advisories

    query_str = ' OR '.join(tokens[:20])
    try:
        if project:
            rows = conn.execute(
                """SELECT i.id, i.status, i.priority, i.title, i.closed_at
                   FROM issues_fts
                   JOIN issues i ON i.rowid = issues_fts.rowid
                   WHERE issues_fts MATCH ?
                     AND (i.project = ? OR i.project IS NULL)
                   ORDER BY bm25(issues_fts)
                   LIMIT ?""",
                (query_str, project, _ISSUES_FTS_K),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT i.id, i.status, i.priority, i.title, i.closed_at
                   FROM issues_fts
                   JOIN issues i ON i.rowid = issues_fts.rowid
                   WHERE issues_fts MATCH ?
                   ORDER BY bm25(issues_fts)
                   LIMIT ?""",
                (query_str, _ISSUES_FTS_K),
            ).fetchall()
    except Exception:
        return advisories

    candidates: list[tuple[str, str]] = []   # (key, advisory_line)
    for iid, status, priority, title, closed_at in rows:
        title_short = (title or '?')[:70]
        if status in ('open', 'in_progress'):
            pri_str = f' (P{priority})' if priority is not None else ''
            candidates.append(
                (f'issue:{iid}',
                 f'[OPEN-ISSUE: {iid}{pri_str} — {title_short}]')
            )
        elif status == 'closed' and _is_recent_closed_ctc(closed_at):
            candidates.append(
                (f'issue:{iid}',
                 f'[RESOLVED-ISSUE: {iid} — {title_short}]')
            )

    if not candidates:
        return advisories

    new_keys = set(filter_unseen([k for k, _ in candidates]))
    for key, line in candidates[:3]:
        if key in new_keys:
            advisories.append(line)

    return advisories

'''

# Insert just before def main()
insert_point = content.index('\ndef main() -> None:')
new_content = content[:insert_point] + new_fn + content[insert_point:]

# Add query_issues call inside main(), after query_contracts line
old_call = '        advisories += query_contracts(conn, tokens, project=project, raw_text=prompt_text)'
new_call = old_call + '\n        advisories += query_issues(conn, prompt_text, project=project)'
assert old_call in new_content, "query_contracts call not found"
new_content = new_content.replace(old_call, new_call, 1)

open(path, 'w').write(new_content)
print(f'Done, lines: {len(new_content.splitlines())}')
assert 'query_issues' in new_content
assert 'OPEN-ISSUE' in new_content
assert 'RESOLVED-ISSUE' in new_content
assert new_call in new_content
print('All assertions OK')

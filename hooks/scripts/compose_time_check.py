#!/usr/bin/env python3
"""L2: compose-time prior-art check.

Fires PreToolUse on Agent dispatches and bridge-send Bash calls.
Scans the outgoing prompt/message for symbols/quantities already in the KB
and surfaces [ALREADY-CODIFIED: ...] advisories BEFORE the dispatch.
Never blocks (exit 0 always). Advisory only.
"""
import sys
import json
import os
import re
import sqlite3

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'lib'))
from _seen import filter_unseen  # noqa: E402
from _state import kb_project_for_path  # noqa: E402
from _db import kb_db_path  # noqa: E402
from compose_tokens import extract_candidate_tokens, extract_fractions, parse_prompt_text  # noqa: E402
try:
    from ash_health import embedding_down, STOP_LINE
except Exception:
    def embedding_down(): return False
    STOP_LINE = ''


def _project_from_cwd() -> str | None:
    """Detect the current KB project from CLAUDE_PROJECT_DIR (or cwd) via the
    nearest .claude/kb-project.json (kb-bp4 P6 — no hardcoded path map)."""
    cwd = os.environ.get('CLAUDE_PROJECT_DIR', '') or os.getcwd()
    return kb_project_for_path(os.path.join(cwd, '_probe'))


def query_db(conn: sqlite3.Connection, tokens: list[str], fracs: list[str],
             project: str | None = None) -> list[str]:
    """Query symbols, notations, and findings for matches. Returns advisory lines."""
    advisories = []
    seen_names: set[str] = set()

    if not tokens and not fracs:
        return advisories

    # --- symbols exact name match — project-scoped to prevent cross-project FPs ---
    placeholders = ','.join('?' * len(tokens))
    if project:
        rows = conn.execute(
            f'SELECT name, kind, status, module, file, line, redirect_to '
            f'FROM symbols WHERE name IN ({placeholders}) AND project=? LIMIT 20',
            tokens + [project],
        ).fetchall()
    else:
        rows = conn.execute(
            f'SELECT name, kind, status, module, file, line, redirect_to '
            f'FROM symbols WHERE name IN ({placeholders}) LIMIT 20',
            tokens,
        ).fetchall()
    canonical_candidates: list[tuple[str, str]] = []
    for name, kind, status, module, fpath, line, redirect_to in rows:
        if name in seen_names:
            continue
        seen_names.add(name)
        mod_str = f'{module}.{name}' if module else name
        loc = f'{os.path.basename(fpath or "")}:{line}' if fpath else '?'
        if status == 'canonical':
            canonical_candidates.append(
                (f'sym:{name}', f'[ALREADY-CODIFIED: {mod_str} ({loc}) — canonical]')
            )
        elif status == 'public':
            # public is not deduplicated — not in the sym: namespace
            advisories.append(
                f'[ALREADY-CODIFIED: {mod_str} ({loc}) — public function/constant]'
            )
        elif status == 'retired':
            # RETIRED never deduplicated
            redir = f' → use {redirect_to}' if redirect_to else ''
            advisories.append(f'[ALREADY-CODIFIED: {name} RETIRED{redir}]')

    if canonical_candidates:
        new_keys = filter_unseen([k for k, _ in canonical_candidates])
        new_key_set = set(new_keys)
        advisories.extend(line for k, line in canonical_candidates if k in new_key_set)

    # --- notations exact symbol match (skip generic-fallback rows) — project-scoped ---
    _not_base = (
        f"SELECT current_symbol, meaning FROM notations "
        f"WHERE current_symbol IN ({placeholders}) "
        f"AND meaning IS NOT NULL AND meaning != '' AND meaning != '?' "
        f"AND (meaning_source IS NULL OR meaning_source != 'generic-fallback')"
    )
    if project:
        rows2 = conn.execute(
            _not_base + " AND (project IS NULL OR project=?) LIMIT 10",
            tokens + [project],
        ).fetchall()
    else:
        rows2 = conn.execute(_not_base + " LIMIT 10", tokens).fetchall()
    notation_candidates: list[tuple[str, str]] = []
    for sym, meaning in rows2:
        if sym in seen_names:
            continue
        seen_names.add(sym)
        notation_candidates.append(
            (f'notation:{sym}', f'[ALREADY-CODIFIED: notation {sym} = {(meaning or "?")[:60]}]')
        )

    if notation_candidates:
        new_keys = filter_unseen([k for k, _ in notation_candidates])
        new_key_set = set(new_keys)
        advisories.extend(line for k, line in notation_candidates if k in new_key_set)

    # --- findings: search for exact fractions / small constants — project-scoped ---
    # Skip entirely when project unknown: unscoped hits are cross-project FPs.
    # Rarity gate: skip fractions appearing in >= 5 entries (arithmetic furniture).
    if not project:
        return advisories
    _FRAC_RARITY_THRESHOLD = 5
    seen_fids: set[str] = set()  # dedup across frac iterations
    for frac in fracs[:5]:
        if project:
            count = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE content LIKE ? AND project=?",
                (f'%{frac}%', project),
            ).fetchone()[0]
        else:
            count = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE content LIKE ?",
                (f'%{frac}%',),
            ).fetchone()[0]
        if count >= _FRAC_RARITY_THRESHOLD:
            continue  # too common — arithmetic furniture, not a notable quantity
        if project:
            rows3 = conn.execute(
                "SELECT id, summary FROM findings WHERE content LIKE ? AND project=? LIMIT 2",
                (f'%{frac}%', project),
            ).fetchall()
        else:
            rows3 = conn.execute(
                "SELECT id, summary FROM findings WHERE content LIKE ? LIMIT 2",
                (f'%{frac}%',),
            ).fetchall()
        for fid, summary in rows3:
            if not fid or fid in seen_fids:
                continue
            if not summary or not summary.strip():
                continue  # unactionable — empty summary
            seen_fids.add(fid)
            short_id = fid[:20]
            preview = summary.strip()[:80]
            advisories.append(
                f'[ALREADY-CODIFIED: value {frac} in KB entry {short_id}: {preview}]'
            )

    return advisories


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


def main() -> None:
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', '')
    ti = data.get('tool_input', {})

    prompt_text = parse_prompt_text(tool_name, ti)
    if not prompt_text or len(prompt_text) < 20:
        sys.exit(0)

    if embedding_down() and STOP_LINE:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "additionalContext": STOP_LINE}}))
        sys.exit(0)

    db = kb_db_path()
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        tokens = extract_candidate_tokens(prompt_text)
        fracs = extract_fractions(prompt_text)
        project = _project_from_cwd()
        advisories = query_issues(conn, prompt_text, project=project)
        advisories += query_db(conn, tokens, fracs, project=project)
        conn.close()
        if advisories:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": "\n".join(advisories),
                }
            }))
    except Exception:
        pass  # never block on failure

    sys.exit(0)


if __name__ == '__main__':
    main()

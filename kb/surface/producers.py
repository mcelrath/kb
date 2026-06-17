"""Pure surfacing producers for kb surface modes.

Each produce_* function takes an input (text, path, msg id) and returns an
Injection dataclass.  The `context` field is the EXACT additionalContext
string the corresponding hook emits (pre-dedup — the seen-gate stays hook-side).

No side effects: no filter_unseen, no bridge recv, no advisory log writes,
no exit-2.  The embed network call + DB reads are allowed (they are reads).

Composed facades used:
  produce_prompt    -> kb.search(query, limit=8, project=None) [findings]
  produce_analysis  -> kb.search(query[:600], limit=8)        [findings]
  produce_symbols   -> kb.conn (symbols + notations)   [direct SQL]
  produce_open_issues -> kb._issues.search / conn (issues_vec + issues_fts)
  produce_bridge    -> kb._bridge.search / conn (bridge_messages)
"""

from __future__ import annotations

import ast
import json
import os
import re
import sqlite3
import struct
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Injection result type
# ---------------------------------------------------------------------------

@dataclass
class Injection:
    producer: str
    context: str        # the exact additionalContext string (pre-dedup)
    hits: list[dict]    # structured hits for --json rendering
    fired: bool         # False when nothing met the threshold / gate


# ---------------------------------------------------------------------------
# Constants mirrored from hooks
# ---------------------------------------------------------------------------

_PROMPT_SIM_FLOOR = 0.42
_PROMPT_MAX = 3
_PROMPT_MIN_LEN = 25

_ANALYSIS_SIM_FLOOR = 0.62
_ANALYSIS_MAX = 2
_ANALYSIS_MIN_LEN = 200

_SYMBOL_MIN_LEN = 3
_SYMBOL_MAX_ADVISORIES = 12

_ISSUE_MAX_SURFACE = 4
_ISSUE_MAX_CLOSED = 2
_ISSUE_RECENTLY_CLOSED_DAYS = 14
_ISSUE_VECTOR_K = 15

# Reimplementation-intent regex from kb-analysis-surface.py (verbatim)
INTENT_RX = re.compile(
    r"(?i)("
    r"\bi['']?ll\s+(?:add|create|build|implement|write|wire|introduce|scaffold)\b|"
    r"\blet me\s+(?:add|create|build|implement|write|wire|introduce)\b|"
    r"\bgoing to\s+(?:add|create|build|implement|write|wire)\b|"
    r"\bi['']?ll\s+write\s+a\b|"
    r"\bnew\s+(?:function|module|hook|class|script|endpoint|helper|command|method|repository|table)\b|"
    r"\bimplement(?:ing)?\s+(?:a|an|the)\b|"
    r"\bcreate\s+(?:a|an|the)\s+\w+|"
    r"\badd\s+(?:a|an|the)\s+\w+\s+(?:function|method|module|hook|endpoint|command|class|helper|table)\b|"
    r"\bwrite\s+(?:a|an|the)\s+\w+\s+(?:function|module|script|hook|helper)\b"
    r")"
)

_SCAN_EXTENSIONS = {
    '.lean', '.py', '.tex', '.md', '.txt', '.output', '.json',
    '.rs', '.ts', '.tsx', '',
}
_CODE_EXTENSIONS = {'.py', '.rs', '.ts', '.tsx'}


# ---------------------------------------------------------------------------
# Token extraction helpers (from symbol_surface.py, verbatim logic)
# ---------------------------------------------------------------------------

def _extract_from_python(source: str) -> list[str]:
    import warnings
    candidates: set[str] = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return _extract_from_text(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if len(node.id) >= _SYMBOL_MIN_LEN:
                candidates.add(node.id)
        elif isinstance(node, ast.Attribute):
            if len(node.attr) >= _SYMBOL_MIN_LEN:
                candidates.add(node.attr)
    return list(candidates)


def _extract_from_text(text: str) -> list[str]:
    candidates: set[str] = set()
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
        tok = m.group(1)
        if len(tok) >= _SYMBOL_MIN_LEN:
            candidates.add(tok)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b', text):
        candidates.add(m.group(1))
    for m in re.finditer(r'\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b', text):
        tok = m.group(1)
        if len(tok) >= _SYMBOL_MIN_LEN:
            candidates.add(tok)
    for ch in re.findall(r'[α-ωΑ-Ω]', text):
        candidates.add(ch)
    return list(candidates)


def _extract_fractions(text: str) -> list[str]:
    return re.findall(r'\b(\d{1,4}/\d{1,4})\b', text)


# ---------------------------------------------------------------------------
# Issue embedding helper — preserves llamacpp|openai format branching
# from open_issues_surface.py verbatim
# ---------------------------------------------------------------------------

def _embed_for_issues(text: str) -> bytes | None:
    """Call embedding server; return packed float32 blob or None.

    Preserves the llamacpp | openai format branching from open_issues_surface.py.
    """
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


# ---------------------------------------------------------------------------
# produce_prompt
# ---------------------------------------------------------------------------

def produce_prompt(
    text: str,
    *,
    kb: Any,
    limit: int = 8,
    min_sim: float = _PROMPT_SIM_FLOOR,
) -> Injection:
    """Semantic kb search on a user prompt.

    Replicates kb-prompt-surface.py's compute + format logic exactly.
    Returns pre-dedup hits (no filter_unseen applied).
    """
    if len(text.strip()) < _PROMPT_MIN_LEN:
        return Injection(producer='prompt', context='', hits=[], fired=False)

    query = text[:600]
    try:
        results = kb.search(query, limit=limit)
    except Exception:
        return Injection(producer='prompt', context='', hits=[], fired=False)

    if not isinstance(results, list):
        return Injection(producer='prompt', context='', hits=[], fired=False)

    hits = []
    for rec in results:
        try:
            sim = float(rec.get('similarity') or 0)
        except (TypeError, ValueError):
            sim = 0.0
        if sim < min_sim:
            continue
        hits.append((sim, rec))
    if not hits:
        return Injection(producer='prompt', context='', hits=[], fired=False)
    hits.sort(key=lambda x: -x[0])

    lines = []
    for sim, rec in hits:
        if len(lines) >= _PROMPT_MAX:
            break
        rid = rec.get('id', '?')
        proj = rec.get('project', '?')
        summ = (rec.get('summary') or rec.get('content') or '')[:80]
        lines.append(f'[KB ~{sim:.2f} {rid} ({proj}): {summ}]')

    if not lines:
        return Injection(producer='prompt', context='', hits=[], fired=False)

    header = ('Possibly-relevant prior findings (semantic match to your prompt — '
              '`kb get <id>` to read before reimplementing):')
    context = header + '\n' + '\n'.join(lines)
    structured = [
        {'sim': sim, 'id': rec.get('id'), 'project': rec.get('project'),
         'summary': (rec.get('summary') or rec.get('content') or '')[:80]}
        for sim, rec in hits[:_PROMPT_MAX]
    ]
    return Injection(producer='prompt', context=context, hits=structured, fired=True)


# ---------------------------------------------------------------------------
# produce_analysis
# ---------------------------------------------------------------------------

def _intent_window(text: str) -> str | None:
    """Return focused query around the reimplementation-intent match."""
    m = INTENT_RX.search(text)
    if not m:
        return None
    start = max(0, m.start() - 40)
    end = min(len(text), m.end() + 280)
    return text[start:end].strip()


def produce_analysis(
    text: str,
    *,
    kb: Any,
    limit: int = 8,
    min_sim: float = _ANALYSIS_SIM_FLOOR,
) -> Injection:
    """Reimplementation-intent gate + near-duplicate prior-art surface.

    Replicates kb-analysis-surface.py's compute + format logic.
    No _already_seen check (that's hook-side dedup), no log write, no exit-2.
    Returns fired=False if INTENT_RX does not match.
    """
    if len(text) < _ANALYSIS_MIN_LEN:
        return Injection(producer='analysis', context='', hits=[], fired=False)

    query = _intent_window(text)
    if not query:
        return Injection(producer='analysis', context='', hits=[], fired=False)

    try:
        results = kb.search(query[:600], limit=limit)
    except Exception:
        return Injection(producer='analysis', context='', hits=[], fired=False)

    if not isinstance(results, list):
        return Injection(producer='analysis', context='', hits=[], fired=False)

    cands = []
    for rec in results:
        try:
            sim = float(rec.get('similarity') or 0)
        except (TypeError, ValueError):
            sim = 0.0
        if sim < min_sim:
            continue
        cands.append((sim, rec))
    if not cands:
        return Injection(producer='analysis', context='', hits=[], fired=False)
    cands.sort(key=lambda x: -x[0])

    lines = []
    for sim, rec in cands:
        if len(lines) >= _ANALYSIS_MAX:
            break
        rid = rec.get('id', '?')
        proj = rec.get('project', '?')
        summ = (rec.get('summary') or rec.get('content') or '')[:90]
        lines.append(f'[KB ~{sim:.2f} {rid} ({proj}): {summ}]')

    if not lines:
        return Injection(producer='analysis', context='', hits=[], fired=False)

    body = (
        "PRIOR ART — your analysis proposes building something the kb may already "
        "have. Before implementing, `kb get <id>` and REUSE if it covers your plan "
        "(per CLAUDE.md: search + reuse before any new function):\n"
        + '\n'.join(lines)
    )
    structured = [
        {'sim': sim, 'id': rec.get('id'), 'project': rec.get('project'),
         'summary': (rec.get('summary') or rec.get('content') or '')[:90]}
        for sim, rec in cands[:_ANALYSIS_MAX]
    ]
    return Injection(producer='analysis', context=body, hits=structured, fired=True)


# ---------------------------------------------------------------------------
# produce_symbols
# ---------------------------------------------------------------------------

def produce_symbols(
    *,
    file_path: str | None = None,
    text: str | None = None,
    project: str | None = None,
    kb: Any,
) -> Injection:
    """RETIRED/NOTATION surface for a file or text content.

    Replicates symbol_surface.py's compute + query_symbols logic.
    Reads symbols + notations tables directly via kb.conn.

    filter_unseen is NOT applied here (hook-side); returns all advisories.
    """
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in _SCAN_EXTENSIONS:
            return Injection(producer='symbols', context='', hits=[], fired=False)
        if not os.path.isfile(file_path):
            return Injection(producer='symbols', context='', hits=[], fired=False)
        try:
            with open(file_path, encoding='utf-8', errors='replace') as fh:
                content = fh.read()
        except OSError:
            return Injection(producer='symbols', context='', hits=[], fired=False)
    elif text:
        content = text
        ext = ''
    else:
        return Injection(producer='symbols', context='', hits=[], fired=False)

    if not content or len(content) < 50:
        return Injection(producer='symbols', context='', hits=[], fired=False)

    if ext == '.py':
        tokens = _extract_from_python(content)
        fracs: list[str] = []
    elif ext in _CODE_EXTENSIONS:
        tokens = _extract_from_text(content)
        fracs = []
    else:
        tokens = _extract_from_text(content)
        fracs = _extract_fractions(content)

    if not tokens and not fracs:
        return Injection(producer='symbols', context='', hits=[], fired=False)

    try:
        advisories = _query_symbols(kb.conn, tokens, fracs, project=project)
    except Exception:
        return Injection(producer='symbols', context='', hits=[], fired=False)

    if not advisories:
        return Injection(producer='symbols', context='', hits=[], fired=False)

    context = '\n'.join(advisories)
    hits = [{'advisory': a} for a in advisories]
    return Injection(producer='symbols', context=context, hits=hits, fired=True)


def _query_symbols(
    conn: sqlite3.Connection,
    tokens: list[str],
    fracs: list[str],
    project: str | None = None,
) -> list[str]:
    """Replicate symbol_surface.py query_symbols exactly.

    Note: filter_unseen for notation_candidates is NOT applied here (hook-side).
    All notation candidates that pass the project/meaning filter are included.
    """
    if not tokens and not fracs:
        return []

    advisories: list[str] = []
    seen: set[str] = set()
    ph = ','.join('?' * len(tokens)) if tokens else ''

    # symbols — retired only (canonical suppressed on Read per hook logic)
    if tokens:
        if project:
            rows = conn.execute(
                f'SELECT name, kind, status, module, file, line, redirect_to '
                f'FROM symbols WHERE name IN ({ph}) AND project=? LIMIT 40',
                tokens + [project],
            ).fetchall()
        else:
            rows = conn.execute(
                f'SELECT name, kind, status, module, file, line, redirect_to '
                f'FROM symbols WHERE name IN ({ph}) LIMIT 40',
                tokens,
            ).fetchall()
        for name, kind, status, module, fpath, line, redirect_to in rows:
            key = f'sym:{name}'
            if key in seen:
                continue
            seen.add(key)
            if status == 'retired':
                redir = f' → {redirect_to}' if redirect_to else ''
                advisories.append(f'[RETIRED: {name}{redir}]')

    # notations — skip generic-fallback rows; project-scoped
    if tokens:
        _not_base = (
            f"SELECT current_symbol, meaning FROM notations "
            f"WHERE current_symbol IN ({ph}) "
            f"AND meaning IS NOT NULL "
            f"AND (meaning_source IS NULL OR meaning_source != 'generic-fallback')"
        )
        if project:
            rows2 = conn.execute(
                _not_base + " AND (project IS NULL OR project=?) LIMIT 10",
                tokens + [project],
            ).fetchall()
        else:
            rows2 = conn.execute(_not_base + " LIMIT 10", tokens).fetchall()
        for sym, meaning in rows2:
            key = f'notation:{sym}'
            if key in seen:
                continue
            seen.add(key)
            advisories.append(f'[NOTATION: {sym} = {(meaning or "?")[:60]}]')

    # fractions — project-scoped, rarity-gated
    if not project:
        return advisories[:_SYMBOL_MAX_ADVISORIES]

    _FRAC_RARITY = 5
    seen_fids: set[str] = set()
    for frac in fracs[:3]:
        count = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE content LIKE ? AND project=?",
            (f'%{frac}%', project),
        ).fetchone()[0]
        if count >= _FRAC_RARITY:
            continue
        rows3 = conn.execute(
            "SELECT id, summary FROM findings WHERE content LIKE ? AND project=? LIMIT 2",
            (f'%{frac}%', project),
        ).fetchall()
        for fid, summary in rows3:
            if not fid or fid in seen_fids:
                continue
            if not summary or not summary.strip():
                continue
            seen_fids.add(fid)
            key = f'frac:{frac}:{fid}'
            if key in seen:
                continue
            seen.add(key)
            preview = summary.strip()[:70]
            short_id = fid[:20]
            advisories.append(f'[KB-VALUE: {frac} — {short_id}: {preview}]')

    return advisories[:_SYMBOL_MAX_ADVISORIES]


# ---------------------------------------------------------------------------
# produce_open_issues
# ---------------------------------------------------------------------------

def _is_recent_closed(closed_at: str | None) -> bool:
    if not closed_at:
        return False
    try:
        from datetime import datetime, timedelta
        dt = datetime.fromisoformat(closed_at.replace('Z', ''))
        return dt >= datetime.utcnow() - timedelta(days=_ISSUE_RECENTLY_CLOSED_DAYS)
    except Exception:
        return False


def _issues_search_fts(conn: sqlite3.Connection, text: str,
                        project: str | None) -> list[dict]:
    """FTS5 fallback — verbatim from open_issues_surface.py."""
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
                (query_str, project, _ISSUE_VECTOR_K),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT i.id, i.status, i.priority, i.title, i.project, i.closed_at
                   FROM issues_fts
                   JOIN issues i ON i.rowid = issues_fts.rowid
                   WHERE issues_fts MATCH ?
                   ORDER BY bm25(issues_fts)
                   LIMIT ?""",
                (query_str, _ISSUE_VECTOR_K),
            ).fetchall()
    except Exception:
        return []
    return [
        {'id': r[0], 'status': r[1], 'priority': r[2],
         'title': r[3], 'project': r[4], 'closed_at': r[5], 'similarity': None}
        for r in rows
    ]


def _issues_search_vector(conn: sqlite3.Connection, blob: bytes,
                           project: str | None) -> list[dict]:
    """KNN search over issues_vec — verbatim from open_issues_surface.py."""
    try:
        rows = conn.execute(
            """SELECT v.id, v.distance
               FROM issues_vec v
               JOIN issues i ON i.id = v.id
               WHERE v.embedding MATCH ? AND k = ?
               ORDER BY v.distance""",
            (blob, _ISSUE_VECTOR_K),
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
        if proj is not None and project is not None and proj != project:
            continue
        similarity = round(1 - (dist ** 2) / 2, 4)
        results.append({
            'id': rid, 'status': status, 'priority': priority,
            'title': title, 'project': proj, 'closed_at': closed_at,
            'similarity': similarity,
        })
    return results


def produce_open_issues(
    text: str,
    *,
    kb: Any,
    project: str | None = None,
) -> Injection:
    """Surface relevant open/recently-closed issues.

    Replicates open_issues_surface.py compute + format logic.
    Preserves llamacpp|openai embed-format branching via _embed_for_issues.
    No filter_unseen (hook-side).
    """
    if not text or len(text) < 20:
        return Injection(producer='open_issues', context='', hits=[], fired=False)

    conn = kb.conn

    try:
        total = conn.execute('SELECT COUNT(*) FROM issues').fetchone()[0]
    except Exception:
        return Injection(producer='open_issues', context='', hits=[], fired=False)
    if total == 0:
        return Injection(producer='open_issues', context='', hits=[], fired=False)

    candidates: list[dict] = []
    blob = _embed_for_issues(text[:2000])
    if blob:
        candidates = _issues_search_vector(conn, blob, project)
    if not candidates:
        candidates = _issues_search_fts(conn, text, project)

    if not candidates:
        return Injection(producer='open_issues', context='', hits=[], fired=False)

    open_issues = [c for c in candidates if c['status'] in ('open', 'in_progress')]
    closed_issues = [c for c in candidates
                     if c['status'] == 'closed' and _is_recent_closed(c.get('closed_at'))]

    open_issues.sort(key=lambda x: (x.get('priority') if x.get('priority') is not None else 99))
    closed_issues.sort(key=lambda x: x.get('closed_at') or '', reverse=True)

    all_items = (
        [(i, 'open') for i in open_issues[:_ISSUE_MAX_SURFACE]] +
        [(i, 'closed') for i in closed_issues[:_ISSUE_MAX_CLOSED]]
    )
    if not all_items:
        return Injection(producer='open_issues', context='', hits=[], fired=False)

    lines: list[str] = []
    for item, kind in all_items:
        iid = item['id']
        title = (item.get('title') or '?')[:70]
        pri = item.get('priority')
        if kind == 'open':
            pri_str = f' (P{pri})' if pri is not None else ''
            lines.append(f'[OPEN-ISSUE: {iid}{pri_str} — {title}]')
        else:
            lines.append(f'[RESOLVED-ISSUE: {iid} — {title}]')

    if not lines:
        return Injection(producer='open_issues', context='', hits=[], fired=False)

    context = '\n'.join(lines)
    hits = [
        {'id': item['id'], 'status': item['status'], 'kind': kind,
         'priority': item.get('priority'), 'title': item.get('title')}
        for item, kind in all_items
    ]
    return Injection(producer='open_issues', context=context, hits=hits, fired=True)


# ---------------------------------------------------------------------------
# produce_bridge
# ---------------------------------------------------------------------------

def produce_bridge(
    *,
    msg_id: int | None = None,
    msg_text: str | None = None,
    kb: Any,
    limit: int = 5,
    min_sim: float = 0.35,
) -> Injection:
    """Surface bridge messages relevant to an already-fetched message.

    ID resolution: if msg_id is given, looks up body+subject from
    bridge_messages table (NEVER calls `bridge recv`).
    Then runs kb._bridge.search() on the resolved text.

    If msg_text is given directly, searches on that text.
    """
    query_text = msg_text

    if msg_id is not None and query_text is None:
        # Resolve id against bridge_messages table
        try:
            row = kb.conn.execute(
                "SELECT subject, body FROM bridge_messages WHERE id = ?",
                (int(msg_id),),
            ).fetchone()
            if row:
                subject, body = row
                query_text = f"{subject or ''}\n{body or ''}".strip()
        except Exception:
            pass

    if not query_text or len(query_text.strip()) < 20:
        return Injection(producer='bridge', context='', hits=[], fired=False)

    try:
        results = kb._bridge.search(query_text, limit=limit)
    except Exception:
        return Injection(producer='bridge', context='', hits=[], fired=False)

    if not results:
        return Injection(producer='bridge', context='', hits=[], fired=False)

    hits = [r for r in results if r.get('similarity', 0) >= min_sim]
    if not hits:
        return Injection(producer='bridge', context='', hits=[], fired=False)

    lines = []
    for r in hits:
        mid = r.get('id', '?')
        sender = r.get('sender', '?')
        sim = r.get('similarity', 0.0)
        preview = (r.get('subject') or r.get('body') or '')[:60]
        lines.append(f'[BRIDGE ~{sim:.2f} #{mid} {sender}: {preview}]')

    context = '\n'.join(lines)
    return Injection(producer='bridge', context=context, hits=hits, fired=True)

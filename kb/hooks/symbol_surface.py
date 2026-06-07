#!/usr/bin/env python3
"""A1: symbol-surface — after reading any file, surface CANONICAL/RETIRED status
for python_symbols and notations mentioned in the file content.

Fires PostToolUse/Read. Advisory only (exit 0 always).

Extraction strategy by file type:
  .py   — ast.walk() over parsed AST: Name + Attribute nodes only.
          Zero false positives from comments/strings. No regex.
  other — regex fallback for unstructured text (bridge output, .md, .lean, .tex).
          Accepts more noise; good enough for symbol names in prose.
"""
import sys
import json
import os
import ast
import re
import sqlite3
import warnings

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _seen import filter_unseen  # noqa: E402

_SCAN_EXTENSIONS = {
    '.lean', '.py', '.tex', '.md', '.txt', '.output', '.json',
    '',  # extensionless (bridge output, etc.)
}

_MIN_SYMBOL_LEN = 3
_MAX_ADVISORIES = 12


# ---------------------------------------------------------------------------
# Python extraction — AST-based, no false positives from comments/strings
# ---------------------------------------------------------------------------

def extract_from_python(source: str) -> list[str]:
    """Parse Python source and return every referenced name via AST walk.

    Collects:
      Name nodes       — bare identifiers (variables, functions, constants)
      Attribute nodes  — the attribute part of dotted access (obj.attr -> attr)

    Ignores everything inside string literals and comments by construction.
    """
    candidates: set[str] = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        # Fall back to regex on unparseable files
        return extract_from_text(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            if len(name) >= _MIN_SYMBOL_LEN:
                candidates.add(name)
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if len(attr) >= _MIN_SYMBOL_LEN:
                candidates.add(attr)

    return list(candidates)


# ---------------------------------------------------------------------------
# Generic text extraction — regex fallback for unstructured content
# ---------------------------------------------------------------------------

def extract_from_text(text: str) -> list[str]:
    """Extract symbol candidates from arbitrary text via regex.

    Used for .lean, .tex, .md, bridge output, and Python files that fail to parse.
    Accepts some noise (tokens from comments/strings); that's acceptable for prose.
    """
    candidates: set[str] = set()

    # snake_case
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
        tok = m.group(1)
        if len(tok) >= _MIN_SYMBOL_LEN:
            candidates.add(tok)

    # CamelCase / PascalCase
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b', text):
        candidates.add(m.group(1))

    # Mixed-case with underscores: Z_species, W_of_J, Q_EM_w, T_3_L
    for m in re.finditer(r'\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b', text):
        tok = m.group(1)
        if len(tok) >= _MIN_SYMBOL_LEN:
            candidates.add(tok)

    # Greek letters
    for ch in re.findall(r'[α-ωΑ-Ω]', text):
        candidates.add(ch)

    return list(candidates)


def extract_fractions(text: str) -> list[str]:
    return re.findall(r'\b(\d{1,4}/\d{1,4})\b', text)


# ---------------------------------------------------------------------------
# Project detection
# ---------------------------------------------------------------------------

_HOME = os.path.expanduser('~')
_PATH_TO_PROJECT: list[tuple[str, str]] = [
    (os.path.join(_HOME, 'Physics'), 'algebraic-genesis'),
    (os.path.join(_HOME, 'Projects', 'ai', 'kb'), 'knowledge-base'),
]


def _project_from_path(fpath: str) -> str | None:
    """Return the KB project name for a given file path, or None if unknown."""
    for prefix, project in _PATH_TO_PROJECT:
        if fpath.startswith(prefix):
            return project
    return None


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def query_symbols(
    conn: sqlite3.Connection,
    tokens: list[str],
    fracs: list[str],
    project: str | None = None,
) -> list[str]:
    if not tokens and not fracs:
        return []

    advisories = []
    seen: set[str] = set()

    ph = ','.join('?' * len(tokens))

    # python_symbols exact name match — project-scoped to prevent cross-project FPs
    if project:
        rows = conn.execute(
            f'SELECT name, kind, status, module, file, line, redirect_to '
            f'FROM python_symbols WHERE name IN ({ph}) AND project=? LIMIT 40',
            tokens + [project],
        ).fetchall()
    else:
        rows = conn.execute(
            f'SELECT name, kind, status, module, file, line, redirect_to '
            f'FROM python_symbols WHERE name IN ({ph}) LIMIT 40',
            tokens,
        ).fetchall()
    # Collect CANONICAL candidates for cross-hook dedup; RETIRED always surfaces
    canonical_candidates: list[tuple[str, str]] = []  # (dedup_key, advisory_line)
    for name, kind, status, module, fpath, line, redirect_to in rows:
        key = f'sym:{name}'
        if key in seen:
            continue
        seen.add(key)
        mod_str = f'{module}.{name}' if module else name
        loc = f'{os.path.basename(fpath or "")}:{line}' if fpath else '?'
        if status == 'canonical':
            canonical_candidates.append((key, f'[CANONICAL: {mod_str} ({loc})]'))
        elif status == 'retired':
            # RETIRED is never deduplicated — always a correctness hazard
            redir = f' → {redirect_to}' if redirect_to else ''
            advisories.append(f'[RETIRED: {name}{redir}]')

    # Cross-hook dedup for CANONICAL hits
    if canonical_candidates:
        new_keys = filter_unseen([k for k, _ in canonical_candidates])
        new_key_set = set(new_keys)
        advisories.extend(line for k, line in canonical_candidates if k in new_key_set)

    # notations — skip generic-fallback rows; project-scoped
    _not_base = (
        f"SELECT current_symbol, meaning FROM notations "
        f"WHERE current_symbol IN ({ph}) "
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
        key = f'notation:{sym}'
        if key in seen:
            continue
        seen.add(key)
        notation_candidates.append((key, f'[NOTATION: {sym} = {(meaning or "?")[:60]}]'))

    if notation_candidates:
        new_keys = filter_unseen([k for k, _ in notation_candidates])
        new_key_set = set(new_keys)
        advisories.extend(line for k, line in notation_candidates if k in new_key_set)

    # findings with matching fractions — project-scoped, rarity-gated
    # Skip entirely when project unknown: unscoped fraction hits are always cross-project FPs
    # (vLLM/AWQ entries matching physics fracs, etc.)
    if not project:
        return advisories[:_MAX_ADVISORIES]
    _FRAC_RARITY = 5
    seen_fids: set[str] = set()
    for frac in fracs[:3]:
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
        if count >= _FRAC_RARITY:
            continue
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
                continue
            seen_fids.add(fid)
            key = f'frac:{frac}:{fid}'
            if key in seen:
                continue
            seen.add(key)
            preview = summary.strip()[:70]
            short_id = fid[:20]
            advisories.append(f'[KB-VALUE: {frac} — {short_id}: {preview}]')

    return advisories[:_MAX_ADVISORIES]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    data = json.load(sys.stdin)
    if data.get('tool_name') != 'Read':
        sys.exit(0)

    fpath = (data.get('tool_input') or {}).get('file_path', '')
    ext = os.path.splitext(fpath)[1].lower()
    if ext not in _SCAN_EXTENSIONS:
        sys.exit(0)

    home = os.path.expanduser('~')
    allowed_prefixes = (
        os.path.join(home, 'Physics'),
        os.path.join(home, 'Projects', 'ai', 'kb'),
        '/tmp/claude',
        '/tmp/agent-',
    )
    if fpath and not any(fpath.startswith(p) for p in allowed_prefixes):
        sys.exit(0)

    if not fpath or not os.path.isfile(fpath):
        sys.exit(0)
    try:
        with open(fpath, encoding='utf-8', errors='replace') as fh:
            content = fh.read()
    except OSError:
        sys.exit(0)

    if not content or len(content) < 50:
        sys.exit(0)

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        project = _project_from_path(fpath)
        if ext == '.py':
            tokens = extract_from_python(content)
            fracs = []  # fractions in Python source aren't math notation
        else:
            tokens = extract_from_text(content)
            fracs = extract_fractions(content)
        advisories = query_symbols(conn, tokens, fracs, project=project)
        conn.close()
        if advisories:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "\n".join(advisories),
                }
            }))
    except Exception:
        pass

    sys.exit(0)


if __name__ == '__main__':
    main()

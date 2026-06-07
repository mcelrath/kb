#!/usr/bin/env python3
"""A1: symbol-surface — after reading any file, surface CANONICAL/RETIRED status
for python_symbols and notations mentioned in the file content.

Fires PostToolUse/Read. Advisory only (exit 0 always).

Use case: archie reads a bridge message containing "sigma_gap_equation" or "G=17/24"
and this hook surfaces [CANONICAL: ...] before archie re-derives it.
"""
import sys
import json
import os
import re
import sqlite3


# File extensions that carry source-code / bridge-message content worth scanning.
# Excludes large binary / data files.
_SCAN_EXTENSIONS = {
    '.lean', '.py', '.tex', '.md', '.txt', '.output', '.json',
    '',  # extensionless files (bridge output, etc.)
}

# Minimum symbol length to avoid flooding on 1-2 char tokens.
_MIN_SYMBOL_LEN = 3

# Max advisories before truncating (avoid wall of text on huge files).
_MAX_ADVISORIES = 12


def extract_symbol_candidates(text: str) -> list[str]:
    """Extract symbol/function name candidates from arbitrary text."""
    candidates: set[str] = set()

    # snake_case names: likely function/constant references
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
        tok = m.group(1)
        if len(tok) >= _MIN_SYMBOL_LEN:
            candidates.add(tok)

    # CamelCase / PascalCase
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b', text):
        candidates.add(m.group(1))

    # Mixed-case identifiers with underscores: Z_species, W_of_J, S_eff, Q_EM_w
    for m in re.finditer(r'\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b', text):
        tok = m.group(1)
        if len(tok) >= 3:
            candidates.add(tok)

    # ALL_CAPS constants (G, G0, G_eff, VIERBEIN_FACTOR, etc.) — physics convention
    for m in re.finditer(r'\b([A-Z][A-Z0-9_]{0,30})\b', text):
        tok = m.group(1)
        if len(tok) >= 2:
            candidates.add(tok)

    # Greek letters
    for ch in re.findall(r'[α-ωΑ-Ω]', text):
        candidates.add(ch)

    return list(candidates)


def extract_fractions(text: str) -> list[str]:
    return re.findall(r'\b(\d{1,4}/\d{1,4})\b', text)


def query_symbols(
    conn: sqlite3.Connection,
    tokens: list[str],
    fracs: list[str],
) -> list[str]:
    if not tokens and not fracs:
        return []

    advisories = []
    seen: set[str] = set()

    # python_symbols exact match
    ph = ','.join('?' * len(tokens))
    rows = conn.execute(
        f'SELECT name, kind, status, module, file, line, redirect_to '
        f'FROM python_symbols WHERE name IN ({ph}) LIMIT 30',
        tokens,
    ).fetchall()
    for name, kind, status, module, fpath, line, redirect_to in rows:
        key = f'sym:{name}'
        if key in seen:
            continue
        seen.add(key)
        mod_str = f'{module}.{name}' if module else name
        loc = f'{os.path.basename(fpath or "")}:{line}' if fpath else '?'
        if status == 'canonical':
            advisories.append(f'[CANONICAL: {mod_str} ({loc})]')
        elif status == 'retired':
            redir = f' → {redirect_to}' if redirect_to else ''
            advisories.append(f'[RETIRED: {name}{redir}]')

    # notations exact symbol match — skip generic-fallback rows (project-blind meanings)
    rows2 = conn.execute(
        f"SELECT current_symbol, meaning FROM notations "
        f"WHERE current_symbol IN ({ph}) AND (meaning_source IS NULL OR meaning_source != 'generic-fallback') "
        f"LIMIT 10",
        tokens,
    ).fetchall()
    for sym, meaning in rows2:
        key = f'not:{sym}'
        if key in seen:
            continue
        seen.add(key)
        advisories.append(f'[NOTATION: {sym} = {(meaning or "?")[:60]}]')

    # findings: exact fractions (small set — max 3)
    for frac in fracs[:3]:
        rows3 = conn.execute(
            "SELECT id, summary FROM findings WHERE content LIKE ? LIMIT 2",
            (f'%{frac}%',),
        ).fetchall()
        for fid, summary in rows3:
            key = f'frac:{frac}:{fid}'
            if key in seen:
                continue
            seen.add(key)
            preview = (summary or '?')[:70]
            short_id = fid[:20] if fid else '?'
            advisories.append(f'[KB-VALUE: {frac} — {short_id}: {preview}]')

    return advisories[:_MAX_ADVISORIES]


def main() -> None:
    data = json.load(sys.stdin)
    if data.get('tool_name') != 'Read':
        sys.exit(0)

    fpath = (data.get('tool_input') or {}).get('file_path', '')
    ext = os.path.splitext(fpath)[1].lower()
    if ext not in _SCAN_EXTENSIONS:
        sys.exit(0)

    # Only scan files from known project roots + bridge output dirs
    # to avoid false hits on unrelated system files.
    home = os.path.expanduser('~')
    allowed_prefixes = (
        os.path.join(home, 'Physics'),
        os.path.join(home, 'Projects', 'ai', 'kb'),
        '/tmp/claude',     # bridge watcher output: /tmp/claude-1000/... or /tmp/claude-*/
        '/tmp/agent-',
    )
    if fpath and not any(fpath.startswith(p) for p in allowed_prefixes):
        sys.exit(0)

    # Read the file content directly from disk.
    # (PostToolUse tool_response format varies; reading from disk is reliable.)
    if not fpath or not os.path.isfile(fpath):
        sys.exit(0)
    try:
        with open(fpath, encoding='utf-8', errors='replace') as fh:
            content = fh.read(1 << 20)  # scan up to 1 MB (covers all project files)
    except OSError:
        sys.exit(0)

    if not content or len(content) < 50:
        sys.exit(0)

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        tokens = extract_symbol_candidates(content)
        fracs = extract_fractions(content)
        advisories = query_symbols(conn, tokens, fracs)
        conn.close()
        for line in advisories:
            print(line, file=sys.stderr)
    except Exception:
        pass

    sys.exit(0)


if __name__ == '__main__':
    main()

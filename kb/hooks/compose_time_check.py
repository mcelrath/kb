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


def extract_candidate_tokens(text: str) -> list[str]:
    """Extract candidate symbol/quantity tokens from prompt text."""
    candidates: set[str] = set()

    # Verb-phrase extraction: "compute/derive/implement/prove/find X"
    verb_re = re.compile(
        r'\b(?:compute|derive|implement|prove|find|calculate|determine|evaluate|check)\s+'
        r'([A-Za-z_][A-Za-z0-9_]{2,})',
        re.IGNORECASE,
    )
    for m in verb_re.finditer(text):
        candidates.add(m.group(1))

    # Explicit "= value" constants: "G = 17/24", "alpha = 3/64", "c = 3"
    eq_re = re.compile(r'\b([A-Za-z_]\w*)\s*=\s*[\d/\.]+')
    for m in eq_re.finditer(text):
        tok = m.group(1)
        if len(tok) >= 2:
            candidates.add(tok)

    # Exact fractional values that may be KB-indexed: "17/24", "3/64"
    frac_re = re.compile(r'\b(\d{1,4}/\d{1,4})\b')
    for m in frac_re.finditer(text):
        candidates.add(m.group(1))

    # snake_case identifiers (likely function/variable names from codebase)
    snake_re = re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b')
    for m in snake_re.finditer(text):
        tok = m.group(1)
        # Skip common stop-words / short tokens
        if len(tok) >= 6 and tok not in {
            'the_user', 'for_the', 'in_the', 'to_the', 'of_the', 'with_the',
        }:
            candidates.add(tok)

    # CamelCase identifiers
    camel_re = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b')
    for m in camel_re.finditer(text):
        candidates.add(m.group(1))

    # Mixed-case with underscores: Z_species, W_of_J, S_eff, Q_EM_w, T_3_L
    mixed_re = re.compile(r'\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b')
    for m in mixed_re.finditer(text):
        tok = m.group(1)
        if len(tok) >= 3:
            candidates.add(tok)

    # ALL_CAPS constants (≥3 chars)
    upper_re = re.compile(r'\b([A-Z][A-Z0-9_]{2,})\b')
    for m in upper_re.finditer(text):
        candidates.add(m.group(1))

    # Greek letters (unicode range)
    greek_re = re.compile(r'[α-ωΑ-Ω]')
    for ch in greek_re.findall(text):
        candidates.add(ch)

    return list(candidates)


def extract_fractions(text: str) -> list[str]:
    """Extract 'N/D' style exact fractions."""
    return re.findall(r'\b\d{1,4}/\d{1,4}\b', text)


def query_db(conn: sqlite3.Connection, tokens: list[str], fracs: list[str]) -> list[str]:
    """Query python_symbols, notations, and findings for matches. Returns advisory lines."""
    advisories = []
    seen_names: set[str] = set()

    if not tokens and not fracs:
        return advisories

    # --- python_symbols exact name match ---
    placeholders = ','.join('?' * len(tokens))
    rows = conn.execute(
        f'SELECT name, kind, status, module, file, line, redirect_to '
        f'FROM python_symbols WHERE name IN ({placeholders}) LIMIT 20',
        tokens,
    ).fetchall()
    for name, kind, status, module, fpath, line, redirect_to in rows:
        if name in seen_names:
            continue
        seen_names.add(name)
        mod_str = f'{module}.{name}' if module else name
        loc = f'{os.path.basename(fpath or "")}:{line}' if fpath else '?'
        if status == 'canonical':
            advisories.append(
                f'[ALREADY-CODIFIED: {mod_str} ({loc}) — canonical, status=canonical]'
            )
        elif status == 'public':
            advisories.append(
                f'[ALREADY-CODIFIED: {mod_str} ({loc}) — public function/constant]'
            )
        elif status == 'retired':
            redir = f' → use {redirect_to}' if redirect_to else ''
            advisories.append(
                f'[ALREADY-CODIFIED: {name} RETIRED{redir}]'
            )

    # --- notations exact symbol match ---
    rows2 = conn.execute(
        f'SELECT current_symbol, meaning FROM notations WHERE current_symbol IN ({placeholders}) LIMIT 10',
        tokens,
    ).fetchall()
    for sym, meaning in rows2:
        if sym in seen_names:
            continue
        seen_names.add(sym)
        advisories.append(f'[ALREADY-CODIFIED: notation {sym} = {(meaning or "?")[:60]}]')

    # --- findings: search for exact fractions / small constants ---
    for frac in fracs[:5]:
        rows3 = conn.execute(
            "SELECT id, summary FROM findings WHERE content LIKE ? LIMIT 2",
            (f'%{frac}%',),
        ).fetchall()
        for fid, summary in rows3:
            short_id = fid[:20] if fid else '?'
            preview = (summary or '?')[:80]
            advisories.append(
                f'[ALREADY-CODIFIED: value {frac} in KB entry {short_id}: {preview}]'
            )

    return advisories


def main() -> None:
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', '')
    ti = data.get('tool_input', {})

    # Determine the text to scan
    prompt_text = ''
    if tool_name == 'Agent':
        prompt_text = ti.get('prompt', '')
    elif tool_name == 'Bash':
        cmd = ti.get('command', '')
        if 'bridge send' not in cmd:
            sys.exit(0)
        # Extract heredoc body from bridge send command
        m = re.search(r"<<'?EOF'?\n(.+?)(?:\nEOF|\Z)", cmd, re.DOTALL)
        if m:
            prompt_text = m.group(1)
        else:
            # Try to extract the message after --message flag or quoted arg
            m2 = re.search(r'bridge send\s+\S+\s+"([^"]+)"', cmd)
            if m2:
                prompt_text = m2.group(1)
    else:
        sys.exit(0)

    if not prompt_text or len(prompt_text) < 20:
        sys.exit(0)

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        tokens = extract_candidate_tokens(prompt_text)
        fracs = extract_fractions(prompt_text)
        advisories = query_db(conn, tokens, fracs)
        conn.close()
        for line in advisories:
            print(line, file=sys.stderr)
    except Exception:
        pass  # never block on failure

    sys.exit(0)


if __name__ == '__main__':
    main()

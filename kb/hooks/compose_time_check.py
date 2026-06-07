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
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _seen import filter_unseen  # noqa: E402


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


_HOME = os.path.expanduser('~')
_PATH_TO_PROJECT: list[tuple[str, str]] = [
    (os.path.join(_HOME, 'Physics'), 'algebraic-genesis'),
    (os.path.join(_HOME, 'Projects', 'ai', 'kb'), 'knowledge-base'),
]


def _project_from_cwd() -> str | None:
    """Detect the current KB project from CLAUDE_PROJECT_DIR env var."""
    cwd = os.environ.get('CLAUDE_PROJECT_DIR', '') or os.getcwd()
    for prefix, project in _PATH_TO_PROJECT:
        if cwd.startswith(prefix):
            return project
    return None


def query_db(conn: sqlite3.Connection, tokens: list[str], fracs: list[str],
             project: str | None = None) -> list[str]:
    """Query python_symbols, notations, and findings for matches. Returns advisory lines."""
    advisories = []
    seen_names: set[str] = set()

    if not tokens and not fracs:
        return advisories

    # --- python_symbols exact name match — project-scoped to prevent cross-project FPs ---
    placeholders = ','.join('?' * len(tokens))
    if project:
        rows = conn.execute(
            f'SELECT name, kind, status, module, file, line, redirect_to '
            f'FROM python_symbols WHERE name IN ({placeholders}) AND project=? LIMIT 20',
            tokens + [project],
        ).fetchall()
    else:
        rows = conn.execute(
            f'SELECT name, kind, status, module, file, line, redirect_to '
            f'FROM python_symbols WHERE name IN ({placeholders}) LIMIT 20',
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
        # Extract all text worth scanning: heredoc body + subject string
        parts = []
        # Heredoc body (handles << 'EOF', << EOF, <<'EOF')
        m = re.search(r"<<\s*'?EOF'?\s*\n(.+?)(?:\nEOF\b|\Z)", cmd, re.DOTALL)
        if m:
            parts.append(m.group(1))
        # Subject string (quoted arg, may have flags between it and EOF)
        m2 = re.search(r'bridge send\s+\S+\s+"([^"]+)"', cmd)
        if m2:
            parts.append(m2.group(1))
        prompt_text = '\n'.join(parts)
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
        project = _project_from_cwd()
        advisories = query_db(conn, tokens, fracs, project=project)
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

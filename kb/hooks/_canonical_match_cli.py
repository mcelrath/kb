#!/usr/bin/env python3
"""CLI: reads plain text from stdin, prints CANONICAL/RETIRED hits to stdout.

Used by archie-report-ingest.sh to surface canonical symbols in agent reports.
Same two-tier matching as check_symbols.py but accepts raw text (not tool-event
JSON). CANONICAL hits participate in the cross-hook session-scoped seen-set.
RETIRED hits are always printed regardless of prior surface.

Tokenization: uses extract_from_text() from symbol_surface — consistent
_MIN_SYMBOL_LEN=3 filter, handles snake_case + CamelCase + mixed + Greek.
"""
import sys
import os
import re
import sqlite3

# Inline the minimal tokenizer from symbol_surface to avoid circular import
_MIN_SYMBOL_LEN = 3


def _extract_tokens(text: str) -> list[str]:
    candidates: set[str] = set()
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
        tok = m.group(1)
        if len(tok) >= _MIN_SYMBOL_LEN:
            candidates.add(tok)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b', text):
        candidates.add(m.group(1))
    for m in re.finditer(r'\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b', text):
        tok = m.group(1)
        if len(tok) >= _MIN_SYMBOL_LEN:
            candidates.add(tok)
    for ch in re.findall(r'[α-ωΑ-Ω]', text):
        candidates.add(ch)
    return list(candidates)


def main() -> None:
    text = sys.stdin.read()
    if not text.strip():
        return

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        return

    try:
        conn = sqlite3.connect(db, timeout=5)
        rows = conn.execute(
            "SELECT name, module, status, redirect_to FROM python_symbols "
            "WHERE status IN ('canonical','retired')"
        ).fetchall()
        conn.close()
    except Exception:
        return

    symbols = {name: (module, status, redir) for name, module, status, redir in rows}
    if not symbols:
        return

    tokens = _extract_tokens(text)

    # Collect hits — CANONICAL deduped cross-hook, RETIRED always shown
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _seen import filter_unseen

    canonical_hits: list[tuple[str, str]] = []
    retired_lines: list[str] = []

    seen_toks: set[str] = set()
    for tok in tokens:
        if tok in seen_toks:
            continue
        # Tier 1: exact token match
        if tok in symbols:
            seen_toks.add(tok)
            module, status, redir = symbols[tok]
            if status == 'retired':
                suf = f' → {redir}' if redir else ''
                retired_lines.append(f'[RETIRED exact: {module}.{tok}{suf}]')
            else:
                canonical_hits.append((f'sym:{tok}', f'[CANONICAL exact: {module}.{tok}]'))
            continue
        # Tier 2: snake_case/camelCase leading component (>=6 chars)
        parts = re.split(r'_|(?<=[a-z])(?=[A-Z])', tok)
        if len(parts) > 1:
            lead = parts[0]
            if len(lead) >= 6 and lead in symbols:
                seen_toks.add(lead)
                module, status, redir = symbols[lead]
                if status == 'retired':
                    suf = f' → {redir}' if redir else ''
                    retired_lines.append(f"[RETIRED component: {module}.{lead}{suf} — '{tok}']")
                else:
                    canonical_hits.append((
                        f'sym:{lead}',
                        f"[CANONICAL component: {module}.{lead} — '{tok}' contains this name]",
                    ))

    # Dedup CANONICAL against session-scoped seen-set
    if canonical_hits:
        new_keys = set(filter_unseen([k for k, _ in canonical_hits]))
        for k, line in canonical_hits:
            if k in new_keys:
                print(line)

    for line in sorted(set(retired_lines)):
        print(line)


if __name__ == '__main__':
    main()

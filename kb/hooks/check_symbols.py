#!/usr/bin/env python3
"""H1: canonical-symbol-check — PreToolUse/Edit+Write on cl44/ Python files.

Checks function/class names being written against python_symbols DB:
  - RETIRED: blocks (exit 2) — stderr reaches agent as hook feedback
  - CANONICAL: advisory (exit 0) — stdout JSON additionalContext (stderr discarded on exit 0)
"""
import sys
import ast
import json
import sqlite3
import os
import re

from ._seen import filter_unseen

d = json.load(sys.stdin)
ti = d.get('tool_input', {})
file_path = ti.get('file_path', '')

if not any(seg in file_path for seg in ['/cl44/', '/clifford_common/', '/cl11/', '/cl22/']):
    sys.exit(0)

content = ti.get('new_string', ti.get('content', ''))
if not content:
    sys.exit(0)

try:
    tree = ast.parse(content)
    names = [n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
except SyntaxError:
    names = re.findall(r'^(?:def|class)\s+(\w+)', content, re.MULTILINE)

if not names:
    sys.exit(0)

db = os.path.expanduser('~/.cache/kb/knowledge.db')
if not os.path.exists(db):
    sys.exit(0)

try:
    conn = sqlite3.connect(db, timeout=5)
    block = False
    canonical_candidates: list[tuple[str, str]] = []  # (dedup_key, advisory_line)
    for name in set(names):
        rows = conn.execute(
            'SELECT status, module, file, line, redirect_to FROM python_symbols WHERE name=? LIMIT 3',
            (name,)
        ).fetchall()
        for status, module, fpath, line, redirect_to in rows:
            if status == 'retired':
                # PreToolUse blocking: stderr is shown as hook feedback on exit 2; never deduplicated
                print(f'[RETIRED: {name} → use {redirect_to or "?"} instead]', file=sys.stderr)
                block = True
            elif status == 'canonical':
                canonical_candidates.append((
                    f'sym:{name}',
                    f'[CANONICAL: {module}.{name} ({fpath}:{line}) — is this what you are implementing?]',
                ))
    conn.close()

    # Cross-hook dedup: suppress CANONICAL advisories already surfaced this session
    if canonical_candidates and not block:
        new_keys = set(filter_unseen([k for k, _ in canonical_candidates]))
        canonical_lines = [ln for k, ln in canonical_candidates if k in new_keys]
    else:
        canonical_lines = [ln for _, ln in canonical_candidates]

    if canonical_lines and not block:
        # Advisory on exit 0: must use stdout JSON — stderr is discarded by harness
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": "\n".join(canonical_lines),
            }
        }))

    sys.exit(2 if block else 0)
except Exception:
    sys.exit(0)

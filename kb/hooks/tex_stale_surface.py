#!/usr/bin/env python3
"""H4: tex-stale-surface — after reading a .tex file, surface stale python_refs.

PostToolUse/Read on .tex files (carl / claude repo).

Categories:
  DELETED        — python file no longer exists on disk (advisory on Read)
  MISSING-CONST  — file exists but symbol is a module constant, not ingested yet
  NEEDS-INGEST   — file exists but symbol not in python_symbols (run kb ingest python)
  IMPRECISE-REF  — directory ref or no ::name (annotation should use module.py::name)

PreToolUse/Edit on .tex files: category DELETED escalates to blocking (exit 2)
when new_string still contains the deleted ref — perpetuating a dead citation.
"""
import sys
import json
import sqlite3
import os
import re
from pathlib import Path

d = json.load(sys.stdin)
tool_name = d.get('tool_name', '')
ti = d.get('tool_input', {})
fpath = ti.get('file_path', '')

is_read = tool_name == 'Read' and fpath.endswith('.tex')
is_edit = tool_name in ('Edit', 'Write') and fpath.endswith('.tex')

if not (is_read or is_edit):
    sys.exit(0)

db = os.path.expanduser('~/.cache/kb/knowledge.db')
if not os.path.exists(db):
    sys.exit(0)

# For Edit/Write, check if new_string still contains any deleted refs
new_string = ti.get('new_string', ti.get('content', '')) if is_edit else ''

_REF_RE = re.compile(r'^[\w./:-]+(?:::[\w.]+)?')

def classify_ref(ref: str, conn) -> tuple[str, str]:
    """Return (category, detail) for a python_ref token."""
    if ref.endswith('/') or '::' not in ref and '.' not in ref.split('/')[-1]:
        return 'IMPRECISE-REF', 'directory or bare name — use module.py::function_name'

    if '::' in ref:
        file_part, name = ref.split('::', 1)
    else:
        file_part, name = ref, None

    # Check if symbol exists in python_symbols
    if name:
        row = conn.execute(
            'SELECT id FROM python_symbols WHERE name=? AND file LIKE ? LIMIT 1',
            (name.strip(), f'%{file_part.strip()}')
        ).fetchone()
    else:
        row = conn.execute(
            'SELECT id FROM python_symbols WHERE file LIKE ? LIMIT 1',
            (f'%{file_part.strip()}',)
        ).fetchone()

    if row:
        return '', ''  # present, no warning

    # Not in DB — distinguish why
    # Find the file on disk (search under Physics/)
    phys = Path.home() / 'Physics'
    candidates = list(phys.glob(f'**/{file_part.strip()}')) if file_part else []
    file_exists = bool(candidates)

    if not file_exists:
        return 'DELETED', f'{file_part} not found on disk'

    if name:
        # File exists — check if name is an AnnAssign constant
        try:
            import ast
            src = candidates[0].read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(src)
            for node in tree.body:
                if isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name) and node.target.id == name:
                        return 'MISSING-CONST', f'{name} is a module constant not yet ingested (run: kb ingest python)'
        except Exception:
            pass
        return 'NEEDS-INGEST', f'{name} not in python_symbols — run: kb ingest python'
    return 'NEEDS-INGEST', f'{file_part} not ingested — run: kb ingest python'


try:
    conn = sqlite3.connect(db, timeout=5)

    rows = conn.execute(
        "SELECT line, section_title, python_refs FROM tex_annotations WHERE file=? AND python_refs != '[]'",
        (fpath,)
    ).fetchall()

    blocking = False
    advisory_lines = []
    for line, section, refs_json in rows:
        refs = json.loads(refs_json or '[]')
        for ref in refs:
            cat, detail = classify_ref(ref, conn)
            if not cat:
                continue
            loc = f'{os.path.basename(fpath)}:{line}'
            if section:
                loc += f' §{section[:40]}'
            tag = f'[TEX-{cat}: {ref} at {loc} — {detail}]'

            if cat == 'DELETED' and is_edit and ref in new_string:
                # Blocking: stderr + exit 2 (the only channel that blocks)
                print(tag, file=sys.stderr)
                print(f'[BLOCKING: Edit would perpetuate dead citation to {ref} — delete annotation or file a bd cleanup item]',
                      file=sys.stderr)
                blocking = True
            else:
                advisory_lines.append(tag)

    conn.close()

    if advisory_lines and not blocking:
        # Advisory on exit 0 — must use stdout JSON; stderr is discarded
        event = 'PostToolUse' if is_read else 'PreToolUse'
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": "\n".join(advisory_lines),
            }
        }))

    sys.exit(2 if blocking else 0)
except Exception:
    sys.exit(0)

#!/usr/bin/env python3
"""H2: lean-py-surface — after reading a .lean file, surface Python modules that cite it."""
import sys
import json
import sqlite3
import os

d = json.load(sys.stdin)
if d.get('tool_name') != 'Read':
    sys.exit(0)

fpath = (d.get('tool_input') or {}).get('file_path', '')
if not fpath.endswith('.lean'):
    sys.exit(0)

basename = os.path.basename(fpath)
db = os.path.expanduser('~/.cache/kb/knowledge.db')
if not os.path.exists(db):
    sys.exit(0)

try:
    conn = sqlite3.connect(db, timeout=5)
    rows = conn.execute(
        "SELECT DISTINCT module FROM python_symbols WHERE lean_citations LIKE ? AND lean_citations != '[]'",
        (f'%{basename}%',)
    ).fetchall()
    conn.close()
    lines = [f'[PY-CITES-THIS: {module} — cites {basename}]' for (module,) in rows]
    if lines:
        # PostToolUse hooks must emit JSON to stdout; stderr on exit-0 is discarded.
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "\n".join(lines),
            }
        }))
except Exception:
    pass

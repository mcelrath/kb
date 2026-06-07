#!/usr/bin/env python3
"""H5: python-tex-backref — after reading a .py file, surface TeX sections that cite it.

PostToolUse/Read on .py files (tip / secular-constraints repo).
Symmetric to lean_py_surface.py (H3).
"""
import sys
import json
import sqlite3
import os

d = json.load(sys.stdin)
if d.get('tool_name') != 'Read':
    sys.exit(0)

fpath = (d.get('tool_input') or {}).get('file_path', '')
if not fpath.endswith('.py'):
    sys.exit(0)

# Only fire for Physics repo files — tex annotations reference Physics code, not kb/
_physics_dir = os.path.join(os.path.expanduser('~'), 'Physics')
if not fpath.startswith(_physics_dir):
    sys.exit(0)

basename = os.path.basename(fpath)
db = os.path.expanduser('~/.cache/kb/knowledge.db')
if not os.path.exists(db):
    sys.exit(0)

try:
    conn = sqlite3.connect(db, timeout=5)
    rows = conn.execute(
        "SELECT file, line, section_title, python_refs FROM tex_annotations "
        "WHERE python_refs LIKE ? AND python_refs != '[]'",
        (f'%{basename}%',)
    ).fetchall()
    conn.close()
    lines = []
    for tex_file, line, section, _ in rows:
        loc = f'{os.path.basename(tex_file)}:{line}'
        sec = f' §{section[:50]}' if section else ''
        lines.append(f'[TEX-CITES-THIS: {loc}{sec}]')
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

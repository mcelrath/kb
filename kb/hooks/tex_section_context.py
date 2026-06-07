#!/usr/bin/env python3
"""H3: tex-section-context — before editing a .tex file, show annotation context."""
import sys
import json
import sqlite3
import os
import re

d = json.load(sys.stdin)
ti = d.get('tool_input', {})
fpath = ti.get('file_path', '')
if not fpath.endswith('.tex'):
    sys.exit(0)

content = ti.get('new_string', ti.get('content', ''))
if not content:
    sys.exit(0)

labels = re.findall(r'\\label\{([^}]+)\}', content)
if not labels:
    sys.exit(0)

db = os.path.expanduser('~/.cache/kb/knowledge.db')
if not os.path.exists(db):
    sys.exit(0)

try:
    conn = sqlite3.connect(db, timeout=5)
    for label in labels:
        rows = conn.execute(
            'SELECT python_refs, lean_refs FROM tex_annotations WHERE section_label=? LIMIT 1',
            (label,)
        ).fetchall()
        for python_refs, lean_refs in rows:
            py = ', '.join(json.loads(python_refs or '[]'))
            lean = ', '.join(json.loads(lean_refs or '[]'))
            parts = []
            if py:
                parts.append(f'py:{py}')
            if lean:
                parts.append(f'lean:{lean}')
            if parts:
                print(f'[TEX-CONTEXT: {label} annotates {" + ".join(parts)}]', file=sys.stderr)
    conn.close()
except Exception:
    pass

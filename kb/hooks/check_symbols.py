#!/usr/bin/env python3
"""H1: canonical-symbol-check — reads hook JSON from stdin, checks python_symbols."""
import sys
import ast
import json
import sqlite3
import os
import re

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
    for name in set(names):
        rows = conn.execute(
            'SELECT status, module, file, line, redirect_to FROM python_symbols WHERE name=? LIMIT 3',
            (name,)
        ).fetchall()
        for status, module, fpath, line, redirect_to in rows:
            if status == 'retired':
                print(f'[RETIRED: {name} → use {redirect_to or "?"} instead]', file=sys.stderr)
                block = True
            elif status == 'canonical':
                print(f'[CANONICAL: {module}.{name} ({fpath}:{line}) — is this what you are implementing?]',
                      file=sys.stderr)
    conn.close()
    sys.exit(2 if block else 0)
except Exception:
    sys.exit(0)

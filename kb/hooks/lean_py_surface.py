#!/usr/bin/env python3
"""H2: lean-py-surface — after reading a .lean file, surface:
  1. Python modules that explicitly cite it (via lean_citations field)
  2. KB findings that mention it (supersede / refute / scope-guard language)
"""
import sys
import json
import re
import sqlite3
import os

_MAX_FINDINGS = 3

d = json.load(sys.stdin)
if d.get('tool_name') != 'Read':
    sys.exit(0)

fpath = (d.get('tool_input') or {}).get('file_path', '')
if not fpath.endswith('.lean'):
    sys.exit(0)

basename = os.path.basename(fpath)
stem = os.path.splitext(basename)[0]

db = os.path.expanduser('~/.cache/kb/knowledge.db')
if not os.path.exists(db):
    sys.exit(0)

lines = []

try:
    conn = sqlite3.connect(db, timeout=5)

    # 1. Python modules that cite this .lean file
    rows = conn.execute(
        "SELECT DISTINCT module FROM python_symbols WHERE lean_citations LIKE ? AND lean_citations != '[]'",
        (f'%{basename}%',)
    ).fetchall()
    for (module,) in rows:
        lines.append(f'[PY-CITES-THIS: {module} — cites {basename}]')

    # 2. KB findings mentioning this file (look for supersede/refute/scope language)
    finding_rows = conn.execute(
        "SELECT id, summary, content FROM findings "
        "WHERE content LIKE ? LIMIT 10",
        (f'%{stem}%',),
    ).fetchall()

    supersede_kw = re.compile(
        r'\b(supersed|refut|stale|retired|scope.guard|obsolete|replaced|archived|do not use|deprecated)\w*\b',
        re.IGNORECASE,
    ) if finding_rows else None

    # Only surface plain KB-MENTION (non-supersede) if the finding also contains
    # action-relevant vocabulary — scope guards, blockers, or warnings.
    action_kw = re.compile(
        r'\b(block|warn|caution|must not|do not|avoid|danger|hazard|before|prerequisite|requires?)\b',
        re.IGNORECASE,
    )

    shown = 0
    for fid, summary, content in finding_rows:
        if shown >= _MAX_FINDINGS:
            break
        content_str = content or ''
        if supersede_kw and supersede_kw.search(content_str):
            label = 'SUPERSEDED/REFUTED'
        elif action_kw.search(content_str):
            label = 'KB-MENTION'
        else:
            continue  # plain mention without action vocabulary — skip (noise per archie #4474)
        short_id = fid[:20] if fid else '?'
        preview = (summary or content_str[:60])[:70]
        lines.append(f'[{label}: {short_id} — {preview}]')
        shown += 1

    conn.close()

    if lines:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "\n".join(lines),
            }
        }))
except Exception:
    pass

# re needed for supersede_kw — import at top next time, but this works

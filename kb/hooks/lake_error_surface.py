#!/usr/bin/env python3
"""PostToolUse/Bash on lake build commands: surface nearest indexed theorem names for errors.

When lake build fails with "Unknown identifier X" or similar, queries the
lean_theorems index to find X (renamed? in which file?).

Advisory only (exit 0 always). Per tip's spec (#4246).
"""
import sys
import json
import os
import re
import sqlite3

_MAX = 3

# Patterns for lake error lines worth diagnosing
_UNKNOWN_ID_RE = re.compile(r'unknown identifier[\'"]?\s+[\'"]?([A-Za-z_][A-Za-z0-9_\'\.]+)', re.IGNORECASE)
_UNKNOWN_CONST_RE = re.compile(r'unknown constant[\'"]?\s+[\'"]?([A-Za-z_][A-Za-z0-9_\'\.]+)', re.IGNORECASE)
_DECL_NOT_FOUND_RE = re.compile(r'declaration[\'"]?\s+[\'"]?([A-Za-z_][A-Za-z0-9_\'\.]+)[\'"]?\s+not found', re.IGNORECASE)


def extract_error_names(output: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for pat in (_UNKNOWN_ID_RE, _UNKNOWN_CONST_RE, _DECL_NOT_FOUND_RE):
        for m in pat.finditer(output):
            name = m.group(1).strip("'\"")
            # Take just the final component of dotted names for FTS
            short = name.rsplit('.', 1)[-1]
            for candidate in (name, short):
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    names.append(candidate)
    return names[:10]


def lookup_names(conn: sqlite3.Connection, names: list[str]) -> list[str]:
    lines = []
    seen: set[str] = set()
    total = 0

    for name in names:
        # Exact match first
        rows = conn.execute(
            "SELECT lt.name, lt.file FROM lean_theorems lt "
            "WHERE lt.name = ? OR lt.lean_name LIKE ? LIMIT 3",
            (name, f'%.{name}'),
        ).fetchall()

        # FTS fallback
        if not rows:
            try:
                rows = conn.execute(
                    "SELECT lt.name, lt.file FROM lean_theorems lt "
                    "JOIN lean_theorems_fts ON lt.rowid = lean_theorems_fts.rowid "
                    "WHERE lean_theorems_fts MATCH ? LIMIT 3",
                    (name,),
                ).fetchall()
            except Exception:
                pass

        for thm_name, thm_file in rows:
            key = f'{thm_name}:{thm_file}'
            if key in seen:
                continue
            seen.add(key)
            total += 1
            if total <= _MAX:
                basename = os.path.basename(thm_file or '')
                lines.append(f'[LAKE-ERROR {name!r}: found indexed as {thm_name} in {basename}]')

    if total == 0 and names:
        lines.append(f'[LAKE-ERROR: {names[0]!r} not found in 109K theorem index — new declaration or typo]')
    elif total > _MAX:
        lines.append(f'[LAKE-ERROR: +{total - _MAX} more matches — use lean-search for full lookup]')

    return lines


def main() -> None:
    data = json.load(sys.stdin)
    if data.get('tool_name') != 'Bash':
        sys.exit(0)

    cmd = (data.get('tool_input') or {}).get('command', '')
    if 'lake' not in cmd:
        sys.exit(0)

    # Get tool output — PostToolUse has tool_response
    output = ''
    tr = data.get('tool_response') or {}
    if isinstance(tr, str):
        output = tr
    elif isinstance(tr, dict):
        output = tr.get('output') or tr.get('stdout') or tr.get('content') or ''
    elif isinstance(tr, list):
        for item in tr:
            if isinstance(item, dict):
                output += item.get('text') or item.get('content') or ''

    if not output or 'error:' not in output.lower():
        sys.exit(0)

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        names = extract_error_names(output)
        lines = lookup_names(conn, names)
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

    sys.exit(0)


if __name__ == '__main__':
    main()

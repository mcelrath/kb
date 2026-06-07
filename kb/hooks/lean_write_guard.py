#!/usr/bin/env python3
"""PreToolUse/Write+Edit on .lean files: name-collision check against indexed theorems.

Before writing a Lean declaration, checks if a theorem with a similar name already
exists in the 109K-entry lean_theorems index. Prevents re-derivation of indexed results.

Advisory only (exit 0 always). Per tip's spec (#4246).
Capped at 3 matches + count of suppressed.
"""
import sys
import json
import os
import re
import sqlite3

_MAX = 3

# Lean declaration starters
_DECL_RE = re.compile(
    r'^\s*(?:theorem|lemma|def|noncomputable def|abbrev|instance|structure|class)\s+'
    r'([A-Za-z_][A-Za-z0-9_\'\.]*)',
    re.MULTILINE,
)


def extract_decl_names(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(1) for m in _DECL_RE.finditer(text)))


def check_collisions(conn: sqlite3.Connection, names: list[str]) -> list[str]:
    if not names:
        return []

    lines = []
    seen: set[str] = set()
    total_hits = 0

    for name in names[:10]:  # check first 10 declarations max
        # Exact name match
        rows = conn.execute(
            "SELECT lt.name, lt.file, lt.declaration "
            "FROM lean_theorems lt "
            "WHERE lt.name = ? OR lt.lean_name LIKE ? "
            "LIMIT 5",
            (name, f'%.{name}'),
        ).fetchall()

        # FTS match on name
        if not rows:
            try:
                rows = conn.execute(
                    "SELECT lt.name, lt.file, lt.declaration "
                    "FROM lean_theorems lt "
                    "JOIN lean_theorems_fts ON lt.rowid = lean_theorems_fts.rowid "
                    "WHERE lean_theorems_fts MATCH ? "
                    "LIMIT 5",
                    (f'name:{name}',),
                ).fetchall()
            except Exception:
                pass

        for thm_name, thm_file, decl in rows:
            key = f'{thm_name}:{thm_file}'
            if key in seen:
                continue
            seen.add(key)
            total_hits += 1
            if total_hits <= _MAX:
                basename = os.path.basename(thm_file or '')
                kind = decl.split()[0] if decl else '?'
                lines.append(f'[ALREADY-PROVEN: {thm_name} ({basename}) — {kind}]')

    if total_hits > _MAX:
        lines.append(f'[ALREADY-PROVEN: +{total_hits - _MAX} more similar declarations indexed]')

    return lines


def main() -> None:
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', '')
    if tool_name not in ('Write', 'Edit'):
        sys.exit(0)

    ti = data.get('tool_input') or {}
    fpath = ti.get('file_path', '')
    if not fpath.endswith('.lean'):
        sys.exit(0)

    # Get content: Write has 'content', Edit has 'new_string'
    text = ti.get('content') or ti.get('new_string') or ''
    if not text:
        sys.exit(0)

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        names = extract_decl_names(text)
        lines = check_collisions(conn, names)
        conn.close()
        if lines:
            # PreToolUse advisory on exit 0 — must use stdout JSON; stderr is discarded
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": "\n".join(lines),
                }
            }))
    except Exception:
        pass

    sys.exit(0)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""PostToolUse/Read on .py files: surface Lean theorems about the same objects.

For each Python module read, finds related Lean theorems via:
  1. lean_citations from python_symbols in that module (explicit cross-references)
  2. FTS search on the module stem name + key symbol names (indirect coverage)

Staleness: if indexed_at_commit differs from current proofs HEAD, marks STALE.
Capped at 5 results + count of suppressed others per tip's spec (#4246).
"""
import sys
import json
import os
import re
import sqlite3
import subprocess

_MAX = 5
_PROJECT_PROOFS = os.path.expanduser('~/Physics/claude')


def current_proofs_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', _PROJECT_PROOFS, 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ''


def surface_lean_for_py(conn: sqlite3.Connection, fpath: str, head_commit: str) -> list[str]:
    """Return advisory lines for Lean theorems related to this .py file."""
    # Derive module name from path: cl44/generating_functional.py -> generating_functional
    stem = os.path.splitext(os.path.basename(fpath))[0]

    # Step 1: collect lean files explicitly cited by symbols in this module
    # Module in python_symbols is dotted: cl44.generating_functional
    rows = conn.execute(
        "SELECT lean_citations FROM python_symbols "
        "WHERE (module LIKE ? OR file LIKE ?) AND lean_citations != '[]' AND lean_citations != ''",
        (f'%.{stem}', f'%{stem}.py'),
    ).fetchall()

    lean_files: set[str] = set()
    for (citations_json,) in rows:
        try:
            for ref in json.loads(citations_json):
                # ref format: "File.lean::TheoremName" or "File.lean"
                lean_file = ref.split('::')[0].strip()
                if lean_file:
                    lean_files.add(os.path.basename(lean_file))
        except (json.JSONDecodeError, AttributeError):
            pass

    results: list[tuple] = []  # (name, file, declaration, indexed_at_commit)

    # Step 2: query theorems in explicitly cited .lean files
    if lean_files:
        ph = ','.join('?' * len(lean_files))
        file_rows = conn.execute(
            f"SELECT name, file, declaration, indexed_at_commit FROM lean_theorems "
            f"WHERE project='algebraic-genesis' "
            f"AND ({' OR '.join('file LIKE ?' for _ in lean_files)}) "
            f"LIMIT 20",
            [f'%{f}' for f in lean_files],
        ).fetchall()
        results.extend(file_rows)

    # Step 3: FTS search on module stem to catch indirect references
    try:
        fts_rows = conn.execute(
            "SELECT lt.name, lt.file, lt.declaration, lt.indexed_at_commit "
            "FROM lean_theorems lt "
            "JOIN lean_theorems_fts ON lt.rowid = lean_theorems_fts.rowid "
            "WHERE lean_theorems_fts MATCH ? AND lt.project='algebraic-genesis' "
            "LIMIT 10",
            (stem,),
        ).fetchall()
        seen_names = {r[0] for r in results}
        for row in fts_rows:
            if row[0] not in seen_names:
                results.append(row)
    except Exception:
        pass

    if not results:
        return []

    lines = []
    for name, thm_file, decl, commit in results[:_MAX]:
        basename = os.path.basename(thm_file or '')
        stale = ''
        if head_commit and commit and commit != head_commit:
            stale = f' [STALE: indexed@{commit}, current@{head_commit}]'
        elif not commit:
            stale = ' [commit unknown]'
        decl_kind = decl.split()[0] if decl else '?'
        lines.append(f'[LEAN: {name} ({basename}) {decl_kind}{stale}]')

    suppressed = len(results) - _MAX
    if suppressed > 0:
        lines.append(f'[LEAN: +{suppressed} more theorems related to {stem} — use loogle/lean-search for full list]')

    return lines


def main() -> None:
    data = json.load(sys.stdin)
    if data.get('tool_name') != 'Read':
        sys.exit(0)

    fpath = (data.get('tool_input') or {}).get('file_path', '')
    if not fpath.endswith('.py'):
        sys.exit(0)

    home = os.path.expanduser('~')
    if not fpath.startswith(os.path.join(home, 'Physics')):
        sys.exit(0)

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        head = current_proofs_commit()
        lines = surface_lean_for_py(conn, fpath, head)
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

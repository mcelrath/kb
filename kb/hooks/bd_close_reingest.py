#!/usr/bin/env python3
"""PostToolUse/Bash hook: when a bd close completes, check if any lean_contracts
rows are data_blocked_on that bd-id, and if so, re-run ingest to clear advisories.

Never blocks (exit 0 always). Advisory + side-effect only.
"""
import sys
import json
import os
import re
import sqlite3
import subprocess


_BD_CLOSE_RE = re.compile(
    r'\bbd\s+close\s+((?:[a-z]+-[a-z0-9]+\s*)+)',
    re.IGNORECASE,
)
_BD_UPDATE_CLOSED_RE = re.compile(
    r'\bbd\s+update\s+([a-z]+-[a-z0-9]+).*?--status[= ]closed',
    re.IGNORECASE | re.DOTALL,
)


def extract_closed_ids(command: str) -> list[str]:
    ids: set[str] = set()
    for m in _BD_CLOSE_RE.finditer(command):
        for part in m.group(1).split():
            if re.fullmatch(r'[a-z]+-[a-z0-9]+', part, re.IGNORECASE):
                ids.add(part.lower())
    for m in _BD_UPDATE_CLOSED_RE.finditer(command):
        ids.add(m.group(1).lower())
    return list(ids)


def find_blocked_files(conn: sqlite3.Connection, bd_ids: list[str]) -> list[str]:
    placeholders = ','.join('?' * len(bd_ids))
    try:
        rows = conn.execute(
            f'SELECT DISTINCT file FROM lean_contracts '
            f'WHERE data_blocked_on IN ({placeholders})',
            bd_ids,
        ).fetchall()
    except Exception:
        return []
    return [r[0] for r in rows if r[0]]


def main() -> None:
    data = json.load(sys.stdin)
    if data.get('tool_name') != 'Bash':
        sys.exit(0)

    command = data.get('tool_input', {}).get('command', '')
    if 'bd' not in command:
        sys.exit(0)

    # Only fire when the command succeeded
    result = data.get('tool_result', {})
    exit_code = result.get('exitCode', result.get('exit_code', 0))
    if exit_code not in (0, None, '0'):
        sys.exit(0)

    closed_ids = extract_closed_ids(command)
    if not closed_ids:
        sys.exit(0)

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        blocked_files = find_blocked_files(conn, closed_ids)
        conn.close()
    except Exception:
        sys.exit(0)

    if not blocked_files:
        sys.exit(0)

    # Re-ingest proofs to clear data_blocked_on advisories
    ingest_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '..', 'scripts', 'ingest_lean_contracts.py',
    )
    ingest_script = os.path.normpath(ingest_script)
    proofs_dir = os.path.expanduser('~/Physics/claude/proofs')

    n_files = len(blocked_files)
    basenames = ', '.join(os.path.basename(f) for f in blocked_files[:3])
    if n_files > 3:
        basenames += f' (+{n_files - 3} more)'
    print(
        f'[BD-CLOSE-REINGEST] {", ".join(closed_ids)} closed; '
        f'{n_files} lean_contracts file(s) were data_blocked_on it: {basenames}. '
        f'Re-ingesting proofs/ to clear advisories…'
    )

    if os.path.exists(ingest_script) and os.path.isdir(proofs_dir):
        try:
            r = subprocess.run(
                [sys.executable, ingest_script, '--proofs-dir', proofs_dir],
                capture_output=True, text=True, timeout=60,
            )
            last_line = (r.stdout.strip().splitlines() or [''])[-1]
            print(f'[BD-CLOSE-REINGEST] ingest done: {last_line}')
        except Exception as e:
            print(f'[BD-CLOSE-REINGEST] ingest failed: {e}')

    sys.exit(0)


if __name__ == '__main__':
    main()

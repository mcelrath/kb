#!/usr/bin/env python3
"""
Ingest open sorry-contracts from proofs/ into the KB lean_contracts table.

Uses lean-audit --json to find sorries, then reads the source to extract the
enclosing theorem/def name and statement. Inserts into lean_contracts so that
compose_time_check and symbol_surface can surface '[SORRY-CONTRACT WAITING: ...]'
advisories when dispatch text mentions matching terms.

Usage:
    python3 scripts/ingest_lean_contracts.py [--proofs-dir DIR] [--dry-run]
"""

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from kb import KnowledgeBase, DEFAULT_DB_PATH

# --- Schema ------------------------------------------------------------------

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS lean_contracts (
    id TEXT PRIMARY KEY,
    file TEXT NOT NULL,
    line INTEGER NOT NULL,
    decl_name TEXT,           -- enclosing theorem/def/lemma name
    namespace TEXT,           -- Lean namespace prefix
    statement TEXT,           -- theorem statement text (up to 300 chars)
    sorry_text TEXT,          -- the sorry line itself
    kind TEXT DEFAULT 'sorry', -- sorry | axiom | true_stub
    project TEXT,
    indexed_at_commit TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lc_file ON lean_contracts(file);
CREATE INDEX IF NOT EXISTS idx_lc_project ON lean_contracts(project);
CREATE INDEX IF NOT EXISTS idx_lc_decl ON lean_contracts(decl_name);
CREATE VIRTUAL TABLE IF NOT EXISTS lean_contracts_fts USING fts5(
    decl_name, namespace, statement, sorry_text,
    content='lean_contracts',
    content_rowid='rowid'
);
"""


# --- Source parsing ----------------------------------------------------------

_DECL_RE = re.compile(
    r'^(?:private\s+|protected\s+)?'
    r'(?:theorem|lemma|def|noncomputable def|abbrev)\s+'
    r'([A-Za-z_\'][A-Za-z0-9_\'\.]*)',
    re.MULTILINE,
)
_NS_RE = re.compile(r'^namespace\s+(\S+)', re.MULTILINE)
_END_NS_RE = re.compile(r'^end\s+(\S+)', re.MULTILINE)


def _extract_decl_at_line(source: str, sorry_line: int) -> tuple[str | None, str | None]:
    """Return (decl_name, statement_text) for the declaration enclosing sorry_line."""
    lines = source.splitlines()
    # Walk backwards from sorry_line to find the nearest theorem/lemma/def
    for i in range(min(sorry_line - 1, len(lines) - 1), -1, -1):
        m = _DECL_RE.match(lines[i])
        if m:
            decl_name = m.group(1)
            # Collect statement: from this line until ':= by' or ':= do' or 'where'
            stmt_lines = []
            for j in range(i, min(i + 20, len(lines))):
                stmt_lines.append(lines[j])
                joined = ' '.join(stmt_lines)
                if re.search(r':=\s*(by|do)\b|where\b', joined):
                    break
            statement = ' '.join(stmt_lines)[:400].strip()
            return decl_name, statement
    return None, None


def _current_namespace(source: str, sorry_line: int) -> str | None:
    """Return the innermost open namespace at sorry_line."""
    lines = source.splitlines()
    ns_stack: list[str] = []
    for i, line in enumerate(lines):
        if i >= sorry_line:
            break
        m = _NS_RE.match(line)
        if m:
            ns_stack.append(m.group(1))
            continue
        m2 = _END_NS_RE.match(line)
        if m2 and ns_stack and ns_stack[-1] == m2.group(1):
            ns_stack.pop()
    return '.'.join(ns_stack) if ns_stack else None


# --- Main ingestion ----------------------------------------------------------

def ingest(proofs_dir: Path, kb: KnowledgeBase, project: str, dry_run: bool) -> dict:
    """Run lean-audit --json over proofs_dir and insert sorry-contracts."""
    # Get current git commit for indexed_at_commit
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=proofs_dir,
        ).stdout.strip()
    except Exception:
        commit = None

    # Run lean-audit --json
    try:
        result = subprocess.run(
            ['lean-audit', str(proofs_dir), '--json', '--no-deep'],
            capture_output=True, text=True, timeout=120,
        )
        audit = json.loads(result.stdout)
    except Exception as e:
        print(f'lean-audit failed: {e}', file=sys.stderr)
        return {'inserted': 0, 'skipped': 0, 'errors': 1}

    now = datetime.now().isoformat()
    inserted = skipped = purged = 0

    # Ensure table exists (even in dry-run, so the SELECT below works)
    kb.conn.executescript(CREATE_TABLE)
    kb.conn.commit()

    # Purge stale rows: files moved to archive/, deleted, or outside proofs_dir.
    # lean-audit only returns files it found; rows for moved/deleted files linger.
    if not dry_run:
        existing = kb.conn.execute('SELECT id, file FROM lean_contracts WHERE project=?',
                                   (project,)).fetchall()
        for row_id, row_file in existing:
            p = Path(row_file) if row_file else None
            gone = p is None or not p.exists() or '/archive/' in row_file
            if gone:
                kb.conn.execute('DELETE FROM lean_contracts WHERE id=?', (row_id,))
                purged += 1
        if purged:
            kb.conn.commit()

    for fpath, fdata in audit.items():
        sorry_lines = fdata.get('sorry_lines', []) + fdata.get('true_stub_lines', [])
        if not sorry_lines:
            continue

        try:
            source = Path(fpath).read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue

        for entry in sorry_lines:
            lineno = entry.get('line', 0)
            sorry_text = entry.get('text', '')
            kind = entry.get('kind', 'sorry')

            decl_name, statement = _extract_decl_at_line(source, lineno)
            namespace = _current_namespace(source, lineno)

            contract_id = f'lc-{Path(fpath).stem}-{lineno}'

            # Check if already present and unchanged
            existing = kb.conn.execute(
                'SELECT id, sorry_text FROM lean_contracts WHERE id=?',
                (contract_id,),
            ).fetchone()
            if existing and existing[1] == sorry_text:
                skipped += 1
                continue

            if dry_run:
                print(f'  {fpath}:{lineno} [{kind}] {decl_name or "?"} — {(statement or "")[:80]}')
                inserted += 1
                continue

            kb.conn.execute("""
                INSERT OR REPLACE INTO lean_contracts
                  (id, file, line, decl_name, namespace, statement, sorry_text,
                   kind, project, indexed_at_commit, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                contract_id, fpath, lineno, decl_name, namespace,
                statement, sorry_text, kind, project, commit, now, now,
            ))
            inserted += 1

        kb.conn.commit()

    return {'inserted': inserted, 'skipped': skipped, 'purged': purged, 'errors': 0}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Ingest sorry-contracts from proofs/ into lean_contracts table')
    parser.add_argument('--proofs-dir', type=Path,
                        default=Path.home() / 'Physics' / 'claude' / 'proofs',
                        help='Directory to audit (default: ~/Physics/claude/proofs)')
    parser.add_argument('--project', default='algebraic-genesis')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--db', type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    kb = KnowledgeBase(db_path=args.db)
    proofs_dir = args.proofs_dir.expanduser().resolve()

    print(f'Auditing {proofs_dir} (dry_run={args.dry_run})', file=sys.stderr)
    stats = ingest(proofs_dir, kb, args.project, args.dry_run)
    print(f"Inserted: {stats['inserted']}  Skipped(unchanged): {stats['skipped']}  Purged: {stats['purged']}  Errors: {stats['errors']}")


if __name__ == '__main__':
    main()

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
    updated_at TEXT NOT NULL,
    file_status TEXT,         -- T1: -- LEAN-STATUS: <marker> in module docstring
    contract_awaiting TEXT,   -- T2: -- CONTRACT: <description> above decl
    discharge_target TEXT     -- emission: -- DISCHARGES: <FullyQualified.Name> above decl
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

# Migration: columns added after initial schema (for existing installs)
_MIGRATE_COLS = [
    ('file_status', 'TEXT'),
    ('contract_awaiting', 'TEXT'),
    ('discharge_target', 'TEXT'),
]


# --- Source parsing ----------------------------------------------------------

_LEAN_STATUS_RE = re.compile(r'--\s*LEAN-STATUS:\s*(.+)')
_CONTRACT_RE = re.compile(r'--\s*CONTRACT:\s*(.+)')
_DISCHARGES_RE = re.compile(r'--\s*DISCHARGES:\s*(.+)')
_SORRY_CONTRACT_INLINE_RE = re.compile(r'--\s*SORRY-CONTRACT:\s*(.+)')


def _parse_file_status(source: str) -> str | None:
    """Scan first 60 lines for -- LEAN-STATUS: marker (module docstring annotation)."""
    for line in source.splitlines()[:60]:
        m = _LEAN_STATUS_RE.search(line)
        if m:
            return m.group(1).strip()
    return None


def _parse_decl_annotation(source: str, sorry_line: int,
                            pattern: re.Pattern) -> str | None:
    """Scan comment lines immediately before the enclosing decl for a marker pattern."""
    lines = source.splitlines()
    # Walk backwards to find the enclosing decl
    decl_line_idx = None
    for i in range(min(sorry_line - 1, len(lines) - 1), -1, -1):
        if _DECL_RE.match(lines[i]):
            decl_line_idx = i
            break
    if decl_line_idx is None:
        return None
    # Scan up to 8 comment lines before the decl
    start = max(0, decl_line_idx - 8)
    for i in range(decl_line_idx - 1, start - 1, -1):
        stripped = lines[i].strip()
        m = pattern.search(stripped)
        if m:
            return m.group(1).strip()
        # Stop at blank line or non-comment code
        if stripped and not stripped.startswith('--') and not stripped.startswith('/-') \
                and not stripped.startswith('@'):
            break
    return None


def _parse_inline_sorry_contract(source: str, sorry_line: int) -> str | None:
    """Check for informal -- SORRY-CONTRACT: marker on the sorry line itself or above it."""
    lines = source.splitlines()
    # Check the sorry line itself and the 3 lines above it
    idx = sorry_line - 1
    for i in range(idx, max(-1, idx - 4), -1):
        if 0 <= i < len(lines):
            m = _SORRY_CONTRACT_INLINE_RE.search(lines[i])
            if m:
                return m.group(1).strip()
    return None


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

    # Migrate: add new columns if missing (existing installs pre-dating AXIOM-CONTRACT)
    existing_cols = {row[1] for row in kb.conn.execute('PRAGMA table_info(lean_contracts)')}
    for col_name, col_type in _MIGRATE_COLS:
        if col_name not in existing_cols:
            kb.conn.execute(f'ALTER TABLE lean_contracts ADD COLUMN {col_name} {col_type}')
    kb.conn.commit()

    # Build the set of live contract IDs from the current audit output.
    # Any DB row not in this set is either discharged (sorry removed in-place),
    # moved to archive/, or deleted — all stale.
    live_ids: set[str] = set()
    for fpath, fdata in audit.items():
        sorry_lines = fdata.get('sorry_lines', []) + fdata.get('true_stub_lines', [])
        for entry in sorry_lines:
            lineno = entry.get('line', 0)
            live_ids.add(f'lc-{Path(fpath).stem}-{lineno}')

    # Purge stale rows: discharged in-place, moved to archive/, or deleted.
    if not dry_run:
        existing = kb.conn.execute('SELECT id, file FROM lean_contracts WHERE project=?',
                                   (project,)).fetchall()
        for row_id, row_file in existing:
            p = Path(row_file) if row_file else None
            gone = (p is None or not p.exists() or '/archive/' in row_file
                    or row_id not in live_ids)
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

        file_status = _parse_file_status(source)

        for entry in sorry_lines:
            lineno = entry.get('line', 0)
            sorry_text = entry.get('text', '')
            kind = entry.get('kind', 'sorry')

            decl_name, statement = _extract_decl_at_line(source, lineno)
            namespace = _current_namespace(source, lineno)

            # T2 annotations: per-decl markers in comments above the declaration
            contract_awaiting = _parse_decl_annotation(source, lineno, _CONTRACT_RE)
            discharge_target = _parse_decl_annotation(source, lineno, _DISCHARGES_RE)
            # Fall back to informal inline marker if no formal T2 found
            if not contract_awaiting and not discharge_target:
                inline = _parse_inline_sorry_contract(source, lineno)
                if inline:
                    contract_awaiting = inline

            contract_id = f'lc-{Path(fpath).stem}-{lineno}'

            # Check if already present and all fields unchanged
            existing = kb.conn.execute(
                'SELECT sorry_text, file_status, contract_awaiting, discharge_target '
                'FROM lean_contracts WHERE id=?',
                (contract_id,),
            ).fetchone()
            if existing and existing[0] == sorry_text \
                    and existing[1] == file_status \
                    and existing[2] == contract_awaiting \
                    and existing[3] == discharge_target:
                skipped += 1
                continue

            if dry_run:
                ann = ''
                if file_status:
                    ann += f' [LEAN-STATUS: {file_status}]'
                if discharge_target:
                    ann += f' [DISCHARGES: {discharge_target}]'
                elif contract_awaiting:
                    ann += f' [CONTRACT: {contract_awaiting[:60]}]'
                print(f'  {fpath}:{lineno} [{kind}] {decl_name or "?"}{ann} — {(statement or "")[:60]}')
                inserted += 1
                continue

            kb.conn.execute("""
                INSERT OR REPLACE INTO lean_contracts
                  (id, file, line, decl_name, namespace, statement, sorry_text,
                   kind, project, indexed_at_commit, created_at, updated_at,
                   file_status, contract_awaiting, discharge_target)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                contract_id, fpath, lineno, decl_name, namespace,
                statement, sorry_text, kind, project, commit, now, now,
                file_status, contract_awaiting, discharge_target,
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

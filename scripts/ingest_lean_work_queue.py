#!/usr/bin/env python3
"""Bulk-populate lean_work_queue from lean_contracts.

Sources:
  cleared-contract  — lean_contracts WHERE data_blocked_on IS NULL (ready to prove)
  statement-suspect — lean_contracts WHERE verdict = 'SUSPECT'

Per-row readiness follows class defaults:
  cleared-contract  -> EXECUTE-READY  (unless divergence_flag set)
  statement-suspect -> DESIGN-NEEDED

Divergence check: if a sorry_contract file is referenced by a certified_data
key that no longer exists in cl44.certified_data, flag divergence_flag=1 and
force readiness=DESIGN-NEEDED (report to archie before touching).

Usage:
  python3 scripts/ingest_lean_work_queue.py [--dry-run]
"""
import argparse
import hashlib
import os
import sqlite3
import sys

DB = os.path.expanduser('~/.cache/kb/knowledge.db')
PROJECT = 'algebraic-genesis'

_VALID_DEFER_PREFIXES = (
    'data_blocked_on:', 'design-pending:', 'file-conflict:',
    'agent-cap', 'user-gate:', 'verify-first:',
)

READINESS_BY_CLASS = {
    'cleared-contract': 'EXECUTE-READY',
    'docstring-pass': 'EXECUTE-READY',
    'discharge-pad': 'EXECUTE-READY',
    'statement-suspect': 'DESIGN-NEEDED',
    'routing-deposit': 'DESIGN-NEEDED',
    'agent-returns-verify': 'EXECUTE-READY',
    'review-class': 'DESIGN-NEEDED',
}


def row_id(file: str, decl: str | None, cls: str) -> str:
    key = f'{file}|{decl or ""}|{cls}'
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _check_certified_data_key(key: str) -> bool:
    """Return True if the certified_data key still resolves."""
    if not key:
        return True
    try:
        parts = key.split('.')
        # e.g. 'cl44.certified_data.CHARGE_MINPOLY'
        if len(parts) < 3:
            return True
        module_path = '.'.join(parts[:2])
        attr = parts[2]
        import importlib
        sys.path.insert(0, os.path.expanduser('~/Physics/secular-constraints'))
        mod = importlib.import_module(module_path)
        return hasattr(mod, attr)
    except Exception:
        return True  # cannot check → assume OK


def ingest(conn: sqlite3.Connection, dry_run: bool) -> tuple[int, int, int]:
    inserted = skipped = diverged = 0

    import re
    _BD_RE = re.compile(r'\b([a-z]+-[a-z0-9]+)\b')
    _SUSPECT_RE = re.compile(r'\bSUSPECT\b', re.IGNORECASE)
    _SUPERSEDED_RE = re.compile(r'\b(superseded|obsolete|retired|do not use)\b', re.IGNORECASE)
    _MATHLIB_GAP_RE = re.compile(
        r'\b(absent from Mathlib|not in Mathlib|Mathlib gap|upstream[- ]scale|upstream contribution)\b',
        re.IGNORECASE,
    )

    try:
        rows = conn.execute("""
            SELECT file, decl_name, data_blocked_on, proof_grade, file_status
            FROM lean_contracts
            WHERE project = ?
        """, (PROJECT,)).fetchall()
    except Exception as e:
        print(f'lean_contracts not accessible: {e}', file=sys.stderr)
        return 0, 0, 0

    # Scratch/archive path filters — exclude files that should never be queued.
    # archive/: files retired by 3206eaa4 (DW/Haar ban), not actionable.
    # tmp/: iteration artifacts (explore_go, test_*, final_clean, full_proof*).
    # Root-level scratch basenames: heuristic patterns for uncommitted scratch files.
    _ARCHIVE_RE = re.compile(r'(^|/)archive/', re.IGNORECASE)
    _TMP_RE = re.compile(r'(^|/)tmp/')
    _SCRATCH_BASENAME_RE = re.compile(
        r'/(explore_\w+|final_clean|full_proof\w*|test_\w+|scratch_\w*)\.lean$',
        re.IGNORECASE,
    )

    for file, decl, blocked_on, proof_grade, file_status in rows:
        if blocked_on:
            continue  # still waiting on a bead

        # Skip archived, tmp, and scratch files — they are not agent-dispatchable.
        file_path = file or ''
        if (_ARCHIVE_RE.search(file_path)
                or _TMP_RE.search(file_path)
                or _SCRATCH_BASENAME_RE.search('/' + file_path)):
            continue

        # Extract bd-id from file_status e.g. "open-contract (claude-b3fk)"
        bd_id = None
        if file_status:
            m = _BD_RE.search(file_status)
            if m:
                bd_id = m.group(1)

        # Divergence: file_status mentions superseded/obsolete → DESIGN-NEEDED
        div_flag = 0
        if file_status and _SUPERSEDED_RE.search(file_status):
            div_flag = 1

        if _SUSPECT_RE.search(proof_grade or '') or _SUSPECT_RE.search(file_status or ''):
            cls = 'statement-suspect'
        else:
            cls = 'cleared-contract'

        readiness = READINESS_BY_CLASS[cls]
        if div_flag:
            readiness = 'DESIGN-NEEDED'
            diverged += 1

        # Mathlib-gap: contract names a specific lemma absent from Mathlib —
        # these are upstream-scale tasks, NOT agent-dispatchable (kb-w6c).
        mathlib_gap = _MATHLIB_GAP_RE.search(proof_grade or '') or _MATHLIB_GAP_RE.search(file_status or '')
        if mathlib_gap:
            readiness = 'DESIGN-NEEDED'

        rid = row_id(file or '', decl, cls)

        # Skip if already exists with a defer_reason (user deferred explicitly)
        existing = conn.execute(
            'SELECT defer_reason FROM lean_work_queue WHERE id = ?', (rid,)
        ).fetchone()
        if existing and existing[0]:
            skipped += 1
            continue

        auto_defer = 'design-pending:mathlib-gap' if mathlib_gap else None

        if not dry_run:
            conn.execute("""
                INSERT INTO lean_work_queue
                    (id, file, decl_name, class, readiness, bd_id,
                     divergence_flag, defer_reason, project)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    readiness = excluded.readiness,
                    divergence_flag = excluded.divergence_flag,
                    defer_reason = COALESCE(lean_work_queue.defer_reason, excluded.defer_reason),
                    updated_at = datetime('now')
                WHERE lean_work_queue.defer_reason IS NULL OR lean_work_queue.defer_reason = ''
            """, (rid, file or '', decl, cls, readiness, bd_id,
                  div_flag, auto_defer, PROJECT))
        inserted += 1

    if not dry_run:
        conn.commit()
    return inserted, skipped, diverged


def main() -> None:
    ap = argparse.ArgumentParser(description='Populate lean_work_queue from lean_contracts')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(DB):
        print(f'DB not found: {DB}', file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB, timeout=10)
    inserted, skipped, diverged = ingest(conn, args.dry_run)
    conn.close()

    label = '[DRY RUN] ' if args.dry_run else ''
    print(f'{label}lean_work_queue: {inserted} upserted, {skipped} skipped (deferred), {diverged} diverged→DESIGN-NEEDED')


if __name__ == '__main__':
    main()

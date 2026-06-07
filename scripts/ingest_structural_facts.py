#!/usr/bin/env python3
"""
Ingest algebraic structural facts from cl44.certified_data into the KB
structural_facts table.

Single source = certified_data (code, importable, CI-ratcheted, versioned
with the operators it describes). This script reads the existing certified_data
tables and any ALGEBRA_RELATIONS dict once emmy lands it, then upserts rows
into structural_facts. No values are stored in the KB — only pointers and
exact result strings for compose-time hook surfacing.

Usage:
    python3 scripts/ingest_structural_facts.py [--dry-run] [--secular-constraints DIR]
"""

import argparse
import hashlib
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from kb.core.schema import init_schema

DEFAULT_DB = os.path.expanduser('~/.cache/kb/knowledge.db')
DEFAULT_SC = os.path.expanduser('~/Physics/secular-constraints')
PROJECT = 'algebraic-genesis'
NOW = datetime.now(timezone.utc).isoformat()


def _uid(key: str) -> str:
    return 'sf-' + hashlib.sha256(key.encode()).hexdigest()[:12]


def _upsert(conn: sqlite3.Connection, row: dict, dry_run: bool) -> str:
    uid = _uid(f"{row['lhs_operator']}:{row.get('rhs_operator','')}:{row['relation_type']}:{row['result_exact']}")
    if dry_run:
        print(f"  [DRY] {uid}  {row['relation_type']}  {row['lhs_operator']} / {row.get('rhs_operator','-')}  = {row['result_exact']}")
        return uid
    conn.execute("""
        INSERT INTO structural_facts
            (id, relation_type, lhs_operator, rhs_operator, result_exact, negative,
             certified_data_key, lean_thm, project, notes, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            result_exact=excluded.result_exact,
            certified_data_key=excluded.certified_data_key,
            lean_thm=excluded.lean_thm,
            notes=excluded.notes,
            updated_at=excluded.updated_at
    """, (
        uid,
        row['relation_type'],
        row['lhs_operator'],
        row.get('rhs_operator'),
        row['result_exact'],
        int(row.get('negative', False)),
        row.get('certified_data_key'),
        row.get('lean_thm'),
        PROJECT,
        row.get('notes'),
        NOW, NOW,
    ))
    return uid


def load_certified_data(sc_dir: str):
    """Import cl44.certified_data from the secular-constraints repo."""
    sys.path.insert(0, sc_dir)
    try:
        import cl44.certified_data as cd
        return cd
    except ImportError as e:
        print(f"ERROR: cannot import cl44.certified_data from {sc_dir}: {e}", file=sys.stderr)
        sys.exit(1)


def extract_rows(cd) -> list[dict]:
    rows = []

    # ── CHARGE_MINPOLY: eigenvalue relation for Q_EM ──────────────────────
    rows.append({
        'relation_type': 'eigenvalue',
        'lhs_operator': 'Q_EM',
        'rhs_operator': None,
        'result_exact': 'eigenvalues {0:4, ±1/3:2, ±2/3:2, ±1:2} (16-dim)',
        'certified_data_key': 'cl44.certified_data.CHARGE_MINPOLY',
        'lean_thm': None,
        'notes': 'minpoly x(x^2-1/9)(x^2-4/9)(x^2-1); grade: exact-sympy',
    })
    rows.append({
        'relation_type': 'charpoly',
        'lhs_operator': 'Q_EM',
        'rhs_operator': None,
        'result_exact': 'x*(x^2-1/9)*(x^2-4/9)*(x^2-1)',
        'certified_data_key': 'cl44.certified_data.CHARGE_MINPOLY["polynomial_factors"]',
        'lean_thm': None,
    })

    # ── CHARGE_TRACE_MOMENTS ─────────────────────────────────────────────
    from fractions import Fraction
    moments = getattr(cd, 'CHARGE_TRACE_MOMENTS', {})
    for k, v in moments.items():
        if not isinstance(k, int):
            continue
        rows.append({
            'relation_type': 'trace',
            'lhs_operator': f'Q_EM^{k}',
            'rhs_operator': None,
            'result_exact': str(v),
            'certified_data_key': f'cl44.certified_data.CHARGE_TRACE_MOMENTS[{k}]',
            'lean_thm': None,
            'notes': 'Tr(Q^k) exact-sympy',
        })

    # ── TREE_YUKAWA_SPECTRUM ──────────────────────────────────────────────
    spec = getattr(cd, 'TREE_YUKAWA_SPECTRUM', {})
    spectrum_str = ', '.join(
        f'{v}×{k}' for k, v in spec.items() if isinstance(k, Fraction)
    )
    if spectrum_str:
        rows.append({
            'relation_type': 'eigenvalue',
            'lhs_operator': 'M_Yukawa',
            'rhs_operator': None,
            'result_exact': f'tree-level m_f^2 spectrum: {spectrum_str} (16-dim per pairing)',
            'certified_data_key': 'cl44.certified_data.TREE_YUKAWA_SPECTRUM',
            'lean_thm': None,
        })

    # ── KINETIC_BOUND_EXACT ───────────────────────────────────────────────
    kin = getattr(cd, 'KINETIC_BOUND_EXACT', {})
    if kin:
        rows.append({
            'relation_type': 'identity',
            'lhs_operator': 'gamma^0',
            'rhs_operator': None,
            'result_exact': 'svmin(gamma^0)^2 = 3/4 exact; kin(b) = 3*(b+1)/16',
            'certified_data_key': 'cl44.certified_data.KINETIC_BOUND_EXACT',
            'lean_thm': None,
        })

    # ── GRAM_TRACE_EXACT ──────────────────────────────────────────────────
    gram = getattr(cd, 'GRAM_TRACE_EXACT', {})
    if gram:
        rows.append({
            'relation_type': 'trace',
            'lhs_operator': 'M_odd^T M_odd',
            'rhs_operator': None,
            'result_exact': 'Tr = 3846.79... (fit-P, K-exact pending)',
            'certified_data_key': 'cl44.certified_data.GRAM_TRACE_EXACT',
            'lean_thm': None,
        })

    # ── K_GRAM_OPERATOR_TABLE: operator disambiguation ────────────────────
    k_gram = getattr(cd, 'K_GRAM_OPERATOR_TABLE', {})
    for key, entry in (k_gram.items() if isinstance(k_gram, dict) else []):
        if not isinstance(entry, dict):
            continue
        name = entry.get('operator', key)
        tr = entry.get('trace_float', entry.get('trace', ''))
        if name and tr:
            rows.append({
                'relation_type': 'trace',
                'lhs_operator': name,
                'rhs_operator': None,
                'result_exact': f'Gram Tr(M^T M) = {tr}',
                'certified_data_key': f'cl44.certified_data.K_GRAM_OPERATOR_TABLE["{key}"]',
                'lean_thm': None,
                'notes': entry.get('notes', 'operator disambiguation table'),
            })

    # ── ALGEBRA_RELATIONS (emmy #4450 — may not exist yet) ───────────────
    algebra = getattr(cd, 'ALGEBRA_RELATIONS', None)
    if algebra:
        for key, rel in algebra.items():
            if not isinstance(rel, dict):
                continue
            rows.append({
                'relation_type': rel.get('type', 'identity'),
                'lhs_operator': rel.get('lhs', key),
                'rhs_operator': rel.get('rhs'),
                'result_exact': str(rel.get('result', rel.get('value', 'UNKNOWN'))),
                'negative': rel.get('negative', False),
                'certified_data_key': f'cl44.certified_data.ALGEBRA_RELATIONS["{key}"]',
                'lean_thm': rel.get('lean_thm'),
                'notes': rel.get('notes'),
            })
        print(f"  ingested {len(algebra)} ALGEBRA_RELATIONS rows")
    else:
        print("  NOTE: ALGEBRA_RELATIONS not yet in certified_data (emmy #4450 pending) — skipping")

    # ── Pip #4452 commutator results (hardcoded until ALGEBRA_RELATIONS lands) ──
    # These are exact numerical results, committed to bridge log, should be in certified_data.
    # Adding as structural facts now; will be superseded when ALGEBRA_RELATIONS table lands.
    rows += [
        {
            'relation_type': 'anticommutator',
            'lhs_operator': 'K',
            'rhs_operator': 'M_odd',
            'result_exact': 'NONZERO — ||KM+MK||_F = 109.6; K-parity does NOT apply to M_odd sector charpolys',
            'negative': True,
            'certified_data_key': None,
            'lean_thm': None,
            'notes': 'pip bridge #4452; exact float, weight frame; until ALGEBRA_RELATIONS lands',
        },
        {
            'relation_type': 'identity',
            'lhs_operator': 'N',
            'rhs_operator': 'M_odd',
            'result_exact': '(NM)^T(NM) = (3/4) M^T M EXACT per sector (N=gamma_tilde_0, N^T N = 3/4 I); sector charpolys of NM cost nothing given M^T M charpolys',
            'negative': False,
            'certified_data_key': None,
            'lean_thm': None,
            'notes': 'pip bridge #4452; 3/4-Dirac identity; C-conjugation: Q=+q and Q=-q sectors share identical M^T M eigenvalues',
        },
        {
            'relation_type': 'commutator',
            'lhs_operator': 'J_C',
            'rhs_operator': 'M_odd',
            'result_exact': 'NEITHER commutator nor anticommutator — ||{J_C,M_odd}||_F=92, ||[J_C,M_odd]||_F=83; no Kramers constraint',
            'negative': True,
            'certified_data_key': None,
            'lean_thm': None,
            'notes': 'pip bridge #4452; until ALGEBRA_RELATIONS lands',
        },
    ]

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--secular-constraints', default=DEFAULT_SC)
    ap.add_argument('--db', default=DEFAULT_DB)
    args = ap.parse_args()

    cd = load_certified_data(args.secular_constraints)
    rows = extract_rows(cd)

    conn = sqlite3.connect(args.db, timeout=10)
    init_schema(conn, embedding_dim=4096)

    print(f"Ingesting {len(rows)} structural fact rows{'  [DRY RUN]' if args.dry_run else ''}...")
    n = 0
    for row in rows:
        _upsert(conn, row, args.dry_run)
        n += 1

    if not args.dry_run:
        conn.commit()
    conn.close()
    print(f"structural_facts: {n} rows upserted.")


if __name__ == '__main__':
    main()

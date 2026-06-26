"""Physics-specific compose-time advisories: contracts, structural facts, route-to-tip.

Used by the secular-constraints project hook (physics-compose-check.py).
All three functions are NO-OPS on databases that lack the physics tables.
"""
import os
import re
import sqlite3

from _seen import filter_unseen  # noqa: E402


_RELATION_PATTERNS = [
    re.compile(r'\[([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\]'),
    re.compile(r'\{([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\}'),
    re.compile(r'\beigenvalues?\s+of\s+([A-Za-z_]\w*)', re.I),
    re.compile(r'\bspectrum\s+of\s+([A-Za-z_]\w*)', re.I),
    re.compile(r'\bcharpoly\s+of\s+([A-Za-z_]\w*)', re.I),
    re.compile(r'\bTr\s*[\(\[]([A-Za-z_]\w*)', re.I),
    re.compile(r'\btrace\s+of\s+([A-Za-z_]\w*)', re.I),
    re.compile(r'\bcommutes?\s+with\b', re.I),
    re.compile(r'\banticommutes?\b', re.I),
    re.compile(r'\brecompute\b', re.I),
]

_PROOF_VOCAB_RE = re.compile(
    r'\b(prove|theorem|lemma|sorry|discharge|lean.prover|\.lean\b|proof_by|apply\s+Lean|'
    r'lean\s+proof|sorry.contract|tactic|mathlib)\b',
    re.IGNORECASE,
)


def _contract_tokens(text: str) -> list[str]:
    toks: set[str] = set()
    for m in re.finditer(r'\b([A-Za-z][A-Za-z0-9]{4,})\b', text):
        toks.add(m.group(1).lower())
    for m in re.finditer(r'\b([A-Z][a-z0-9]+)\b', text):
        tok = m.group(1).lower()
        if len(tok) >= 5:
            toks.add(tok)
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
        for part in m.group(1).split('_'):
            if len(part) >= 5:
                toks.add(part)
    extras: set[str] = set()
    for tok in toks:
        if tok.endswith('s') and len(tok) > 6:
            extras.add(tok[:-1])
        if tok.endswith('es') and len(tok) > 7:
            extras.add(tok[:-2])
    toks |= extras
    return list(toks)


def query_contracts(conn: sqlite3.Connection, tokens: list[str],
                    project: str | None = None, raw_text: str = '') -> list[str]:
    """Surface open sorry-contracts whose decl_name or statement matches tokens."""
    all_tokens = list(set(tokens) | set(_contract_tokens(raw_text))) if raw_text else tokens
    if not all_tokens:
        return []
    try:
        conn.execute('SELECT 1 FROM lean_contracts LIMIT 1')
    except Exception:
        return []

    contract_hits: dict[str, int] = {}
    contract_meta: dict[str, tuple] = {}
    for tok in all_tokens:
        if len(tok) < 5:
            continue
        if project:
            rows = conn.execute(
                "SELECT id, file, line, decl_name, file_status, discharge_target, contract_awaiting, "
                "proof_grade, data_blocked_on "
                "FROM lean_contracts "
                "WHERE (decl_name LIKE ? OR statement LIKE ?) AND project=? "
                "AND file NOT LIKE '%/archive/%' LIMIT 3",
                (f'%{tok}%', f'%{tok}%', project),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, file, line, decl_name, file_status, discharge_target, contract_awaiting, "
                "proof_grade, data_blocked_on "
                "FROM lean_contracts "
                "WHERE (decl_name LIKE ? OR statement LIKE ?) "
                "AND file NOT LIKE '%/archive/%' LIMIT 3",
                (f'%{tok}%', f'%{tok}%'),
            ).fetchall()
        for cid, fpath, line, decl_name, file_status, discharge_target, contract_awaiting, proof_grade, data_blocked_on in rows:
            contract_hits[cid] = contract_hits.get(cid, 0) + 1
            if cid not in contract_meta:
                contract_meta[cid] = (fpath, line, decl_name, file_status, discharge_target, contract_awaiting, proof_grade, data_blocked_on)

    def _is_relevant(fpath, decl_name, file_status, text_lower):
        if fpath:
            base = os.path.basename(fpath or '').replace('.lean', '').lower()
            if len(base) >= 6 and base in text_lower:
                return True
        if decl_name and len(decl_name) >= 6 and decl_name.lower() in text_lower:
            return True
        if file_status:
            bd_m = re.search(r'((?:claude|secular-constraints)-[a-z0-9]+(?:\.[0-9]+)?)', file_status)
            if bd_m and bd_m.group(1).lower() in text_lower:
                return True
        return False

    text_lower = raw_text.lower()

    contract_candidates: list[tuple[str, str]] = []
    for cid, hits in contract_hits.items():
        if hits < 2:
            continue
        fpath, line, decl_name, file_status, discharge_target, contract_awaiting, proof_grade, data_blocked_on = contract_meta[cid]
        if not _is_relevant(fpath, decl_name, file_status, text_lower):
            continue
        basename = os.path.basename(fpath or '')
        name_str = decl_name or '?'
        if data_blocked_on:
            suffix = f' | DATA-BLOCKED (no discharge until {data_blocked_on} lands; do NOT route to prover)'
        elif discharge_target:
            suffix = f' | DISCHARGES: {discharge_target}'
        elif contract_awaiting:
            suffix = f' | CONTRACT: {contract_awaiting[:70]}'
        elif file_status:
            fs_token = file_status.split()[0] if file_status else ''
            if fs_token == 'contract-skeleton':
                suffix = f' | SKELETON (statements are placeholders — repair statements first, do NOT attempt discharge): {file_status[:60]}'
            elif fs_token == 'statement-suspect':
                suffix = f' | SUSPECT (lean-audit flagged vacuity — route to REVIEW, not discharge): {file_status[:60]}'
            else:
                suffix = f' | CONTRACT-FILE: {file_status[:60]}'
        else:
            suffix = ''
        contract_candidates.append(
            (f'lc:{cid}', f'[SORRY-CONTRACT WAITING: {basename}:{line} — {name_str}{suffix}]')
        )

    advisories: list[str] = []
    if contract_candidates:
        new_keys = set(filter_unseen([k for k, _ in contract_candidates]))
        advisories.extend(line for k, line in contract_candidates if k in new_keys)
    return advisories[:5]


def query_structural_facts(conn: sqlite3.Connection, text: str) -> list[str]:
    """Surface structural-fact advisories when relation-shaped text + cataloged operators appear."""
    try:
        conn.execute('SELECT 1 FROM structural_facts LIMIT 1')
    except Exception:
        return []

    has_relation = any(p.search(text) for p in _RELATION_PATTERNS)
    if not has_relation:
        return []

    if re.search(r'certified_data|STRUCTURAL.FACT|ALGEBRA_RELATIONS', text, re.IGNORECASE):
        return []

    known_ops = set()
    rows = conn.execute(
        'SELECT DISTINCT lhs_operator, rhs_operator FROM structural_facts'
    ).fetchall()
    for lhs, rhs in rows:
        for part in re.split(r'[/\s]+', lhs or ''):
            if len(part) >= 3:
                known_ops.add(part)
        if rhs:
            for part in re.split(r'[/\s]+', rhs):
                if len(part) >= 3:
                    known_ops.add(part)

    matched_ops: set[str] = set()
    for op in known_ops:
        if re.search(r'\b' + re.escape(op) + r'\b', text):
            matched_ops.add(op)

    if not matched_ops:
        return []

    advisories: list[str] = []
    seen_ids: set[str] = set()
    for op in matched_ops:
        sf_rows = conn.execute(
            "SELECT id, relation_type, lhs_operator, rhs_operator, result_exact, "
            "       negative, certified_data_key, lean_thm, notes "
            "FROM structural_facts "
            "WHERE lhs_operator LIKE ? OR rhs_operator LIKE ? LIMIT 4",
            (f'%{op}%', f'%{op}%'),
        ).fetchall()
        for sf_id, rtype, lhs, rhs, result, negative, cd_key, lean_thm, notes in sf_rows:
            if sf_id in seen_ids:
                continue
            seen_ids.add(sf_id)
            lhs_str = lhs or ''
            rhs_str = rhs or ''
            if rhs_str:
                pair = (f'{{{lhs_str},{rhs_str}}}' if rtype == 'anticommutator'
                        else f'[{lhs_str},{rhs_str}]' if rtype == 'commutator'
                        else f'{lhs_str}/{rhs_str}')
            else:
                pair = lhs_str
            neg_tag = ' (NEGATIVE RESULT)' if negative else ''
            src = cd_key or lean_thm or 'certified_data'
            result_short = result[:120] if result else '?'
            line = (f'[STRUCTURAL-FACT{neg_tag}: {rtype}({pair}) = {result_short} '
                    f'({src}) — DO NOT RECOMPUTE; cite certified_data]')
            if notes and len(notes) < 80:
                line += f' note: {notes}'
            advisories.append(line)

    return advisories[:6]


def query_route_to_tip(conn: sqlite3.Connection, tool_name: str, ti: dict,
                       prompt_text: str) -> list[str]:
    """Advisory: if non-lean-prover agent dispatch contains proof vocabulary, route to tip."""
    if tool_name not in ('Task', 'Agent'):
        return []
    try:
        conn.execute('SELECT 1 FROM lean_work_queue LIMIT 1')
    except Exception:
        return []
    subagent_type = ti.get('subagent_type', '') or ''
    if 'lean' in subagent_type.lower():
        return []
    if not _PROOF_VOCAB_RE.search(prompt_text):
        return []
    return ['[ROUTE-TO-TIP: dispatch contains proof-writing vocabulary; tip owns proof work. '
            'File a routing-deposit in lean_work_queue instead of implementing inline. '
            'If tip is offline, file a kbt task with class=proof-work.]']

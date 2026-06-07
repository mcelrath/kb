#!/usr/bin/env python3
"""L2: compose-time prior-art check.

Fires PreToolUse on Agent dispatches and bridge-send Bash calls.
Scans the outgoing prompt/message for symbols/quantities already in the KB
and surfaces [ALREADY-CODIFIED: ...] advisories BEFORE the dispatch.
Never blocks (exit 0 always). Advisory only.
"""
import sys
import json
import os
import re
import sqlite3

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _seen import filter_unseen  # noqa: E402
try:
    sys.path.insert(0, os.path.expanduser('~/.claude/hooks/lib'))
    from ash_health import ash_down, STOP_LINE
except Exception:
    def ash_down(): return False
    STOP_LINE = ''


def extract_candidate_tokens(text: str) -> list[str]:
    """Extract candidate symbol/quantity tokens from prompt text."""
    candidates: set[str] = set()

    # Verb-phrase extraction: "compute/derive/implement/prove/find X"
    verb_re = re.compile(
        r'\b(?:compute|derive|implement|prove|find|calculate|determine|evaluate|check)\s+'
        r'([A-Za-z_][A-Za-z0-9_]{2,})',
        re.IGNORECASE,
    )
    for m in verb_re.finditer(text):
        candidates.add(m.group(1))

    # Explicit "= value" constants: "G = 17/24", "alpha = 3/64", "c = 3"
    eq_re = re.compile(r'\b([A-Za-z_]\w*)\s*=\s*[\d/\.]+')
    for m in eq_re.finditer(text):
        tok = m.group(1)
        if len(tok) >= 2:
            candidates.add(tok)

    # Exact fractional values that may be KB-indexed: "17/24", "3/64"
    frac_re = re.compile(r'\b(\d{1,4}/\d{1,4})\b')
    for m in frac_re.finditer(text):
        candidates.add(m.group(1))

    # snake_case identifiers (likely function/variable names from codebase)
    snake_re = re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+){1,})\b')
    for m in snake_re.finditer(text):
        tok = m.group(1)
        # Skip common stop-words / short tokens
        if len(tok) >= 6 and tok not in {
            'the_user', 'for_the', 'in_the', 'to_the', 'of_the', 'with_the',
        }:
            candidates.add(tok)

    # CamelCase identifiers
    camel_re = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)\b')
    for m in camel_re.finditer(text):
        candidates.add(m.group(1))

    # Mixed-case with underscores: Z_species, W_of_J, S_eff, Q_EM_w, T_3_L
    mixed_re = re.compile(r'\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b')
    for m in mixed_re.finditer(text):
        tok = m.group(1)
        if len(tok) >= 3:
            candidates.add(tok)

    # ALL_CAPS constants (≥3 chars)
    upper_re = re.compile(r'\b([A-Z][A-Z0-9_]{2,})\b')
    for m in upper_re.finditer(text):
        candidates.add(m.group(1))

    # Greek letters (unicode range)
    greek_re = re.compile(r'[α-ωΑ-Ω]')
    for ch in greek_re.findall(text):
        candidates.add(ch)

    return list(candidates)


_FRAC_CONTEXT_RE = re.compile(
    r'(?:'
    r'[A-Za-zα-ωΑ-Ω_]\w*\s*=\s*\d{1,4}/\d{1,4}'   # X = N/D (assignment context)
    r'|\d{1,4}/\d{1,4}\s*[A-Za-zα-ωΑ-Ω_]'           # N/D followed by a unit/operator
    r'|\b(?:value|exact|result|equals?|is)\s+\d{1,4}/\d{1,4}'  # value 17/24
    r')',
    re.IGNORECASE,
)


def extract_fractions(text: str) -> list[str]:
    """Extract 'N/D' style exact fractions that appear in an operator/value context.

    Bare 'items 10/11' style list ranges are excluded — require an adjacent
    operator name, assignment, or explicit value keyword.
    """
    if not _FRAC_CONTEXT_RE.search(text):
        return []
    return re.findall(r'\b\d{1,4}/\d{1,4}\b', text)


_HOME = os.path.expanduser('~')
_PATH_TO_PROJECT: list[tuple[str, str]] = [
    (os.path.join(_HOME, 'Physics'), 'algebraic-genesis'),
    (os.path.join(_HOME, 'Projects', 'ai', 'kb'), 'knowledge-base'),
]


def _project_from_cwd() -> str | None:
    """Detect the current KB project from CLAUDE_PROJECT_DIR env var."""
    cwd = os.environ.get('CLAUDE_PROJECT_DIR', '') or os.getcwd()
    for prefix, project in _PATH_TO_PROJECT:
        if cwd.startswith(prefix):
            return project
    return None


def query_db(conn: sqlite3.Connection, tokens: list[str], fracs: list[str],
             project: str | None = None) -> list[str]:
    """Query python_symbols, notations, and findings for matches. Returns advisory lines."""
    advisories = []
    seen_names: set[str] = set()

    if not tokens and not fracs:
        return advisories

    # --- python_symbols exact name match — project-scoped to prevent cross-project FPs ---
    placeholders = ','.join('?' * len(tokens))
    if project:
        rows = conn.execute(
            f'SELECT name, kind, status, module, file, line, redirect_to '
            f'FROM python_symbols WHERE name IN ({placeholders}) AND project=? LIMIT 20',
            tokens + [project],
        ).fetchall()
    else:
        rows = conn.execute(
            f'SELECT name, kind, status, module, file, line, redirect_to '
            f'FROM python_symbols WHERE name IN ({placeholders}) LIMIT 20',
            tokens,
        ).fetchall()
    canonical_candidates: list[tuple[str, str]] = []
    for name, kind, status, module, fpath, line, redirect_to in rows:
        if name in seen_names:
            continue
        seen_names.add(name)
        mod_str = f'{module}.{name}' if module else name
        loc = f'{os.path.basename(fpath or "")}:{line}' if fpath else '?'
        if status == 'canonical':
            canonical_candidates.append(
                (f'sym:{name}', f'[ALREADY-CODIFIED: {mod_str} ({loc}) — canonical]')
            )
        elif status == 'public':
            # public is not deduplicated — not in the sym: namespace
            advisories.append(
                f'[ALREADY-CODIFIED: {mod_str} ({loc}) — public function/constant]'
            )
        elif status == 'retired':
            # RETIRED never deduplicated
            redir = f' → use {redirect_to}' if redirect_to else ''
            advisories.append(f'[ALREADY-CODIFIED: {name} RETIRED{redir}]')

    if canonical_candidates:
        new_keys = filter_unseen([k for k, _ in canonical_candidates])
        new_key_set = set(new_keys)
        advisories.extend(line for k, line in canonical_candidates if k in new_key_set)

    # --- notations exact symbol match (skip generic-fallback rows) — project-scoped ---
    _not_base = (
        f"SELECT current_symbol, meaning FROM notations "
        f"WHERE current_symbol IN ({placeholders}) "
        f"AND meaning IS NOT NULL AND meaning != '' AND meaning != '?' "
        f"AND (meaning_source IS NULL OR meaning_source != 'generic-fallback')"
    )
    if project:
        rows2 = conn.execute(
            _not_base + " AND (project IS NULL OR project=?) LIMIT 10",
            tokens + [project],
        ).fetchall()
    else:
        rows2 = conn.execute(_not_base + " LIMIT 10", tokens).fetchall()
    notation_candidates: list[tuple[str, str]] = []
    for sym, meaning in rows2:
        if sym in seen_names:
            continue
        seen_names.add(sym)
        notation_candidates.append(
            (f'notation:{sym}', f'[ALREADY-CODIFIED: notation {sym} = {(meaning or "?")[:60]}]')
        )

    if notation_candidates:
        new_keys = filter_unseen([k for k, _ in notation_candidates])
        new_key_set = set(new_keys)
        advisories.extend(line for k, line in notation_candidates if k in new_key_set)

    # --- findings: search for exact fractions / small constants — project-scoped ---
    # Skip entirely when project unknown: unscoped hits are cross-project FPs.
    # Rarity gate: skip fractions appearing in >= 5 entries (arithmetic furniture).
    if not project:
        return advisories
    _FRAC_RARITY_THRESHOLD = 5
    seen_fids: set[str] = set()  # dedup across frac iterations
    for frac in fracs[:5]:
        if project:
            count = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE content LIKE ? AND project=?",
                (f'%{frac}%', project),
            ).fetchone()[0]
        else:
            count = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE content LIKE ?",
                (f'%{frac}%',),
            ).fetchone()[0]
        if count >= _FRAC_RARITY_THRESHOLD:
            continue  # too common — arithmetic furniture, not a notable quantity
        if project:
            rows3 = conn.execute(
                "SELECT id, summary FROM findings WHERE content LIKE ? AND project=? LIMIT 2",
                (f'%{frac}%', project),
            ).fetchall()
        else:
            rows3 = conn.execute(
                "SELECT id, summary FROM findings WHERE content LIKE ? LIMIT 2",
                (f'%{frac}%',),
            ).fetchall()
        for fid, summary in rows3:
            if not fid or fid in seen_fids:
                continue
            if not summary or not summary.strip():
                continue  # unactionable — empty summary
            seen_fids.add(fid)
            short_id = fid[:20]
            preview = summary.strip()[:80]
            advisories.append(
                f'[ALREADY-CODIFIED: value {frac} in KB entry {short_id}: {preview}]'
            )

    return advisories


def _contract_tokens(text: str) -> list[str]:
    """Extract tokens suitable for sorry-contract matching (less strict than dispatch tokenizer)."""
    toks: set[str] = set()
    # All word-like tokens >= 5 chars (catches 'charpoly', 'irreducible', etc.)
    for m in re.finditer(r'\b([A-Za-z][A-Za-z0-9]{4,})\b', text):
        toks.add(m.group(1).lower())
    # CamelCase components: split 'ChargedSectorKCharpolys' -> ['charged', 'sector', 'charpolys']
    for m in re.finditer(r'\b([A-Z][a-z0-9]+)\b', text):
        tok = m.group(1).lower()
        if len(tok) >= 5:
            toks.add(tok)
    # snake_case components
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
        for part in m.group(1).split('_'):
            if len(part) >= 5:
                toks.add(part)
    # Light stemming: add singular form for common plurals (charpolys -> charpoly)
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
    # Use raw_text for better tokenization if available
    all_tokens = list(set(tokens) | set(_contract_tokens(raw_text))) if raw_text else tokens
    if not all_tokens:
        return []
    # Check table exists
    try:
        conn.execute('SELECT 1 FROM lean_contracts LIMIT 1')
    except Exception:
        return []

    # Track how many distinct tokens match each contract; require >= 2 to surface.
    # A single common token ("mass", "spectrum") matches too many unrelated contracts.
    contract_hits: dict[str, int] = {}       # cid -> distinct token hit count
    contract_meta: dict[str, tuple] = {}     # cid -> (fpath, line, decl_name, file_status, discharge_target, contract_awaiting, proof_grade, data_blocked_on)
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

    # Relevance gate: require the text explicitly mentions EITHER:
    #   (a) the file basename (e.g. 'ChargedSectorKCharpolys'), OR
    #   (b) the owning bd-id (e.g. 'claude-gyb.4'), OR
    #   (c) decl_name verbatim (exact CamelCase match).
    # This prevents contracts from firing on every git commit / bridge send
    # that shares common tokens like "mass", "spectrum", "charpoly".
    def _is_relevant(fpath: str | None, decl_name: str | None,
                     file_status: str | None, text_lower: str) -> bool:
        if fpath:
            base = os.path.basename(fpath or '').replace('.lean', '').lower()
            if len(base) >= 6 and base in text_lower:
                return True
        if decl_name and len(decl_name) >= 6 and decl_name.lower() in text_lower:
            return True
        # bd-id from file_status, e.g. 'open-contract (claude-gyb.4)'
        if file_status:
            bd_m = re.search(r'((?:claude|secular-constraints)-[a-z0-9]+(?:\.[0-9]+)?)', file_status)
            if bd_m and bd_m.group(1).lower() in text_lower:
                return True
        return False

    text_lower = raw_text.lower()

    # Only surface contracts with >= 2 distinct token hits (noise filter)
    contract_candidates: list[tuple[str, str]] = []
    for cid, hits in contract_hits.items():
        if hits < 2:
            continue
        fpath, line, decl_name, file_status, discharge_target, contract_awaiting, proof_grade, data_blocked_on = contract_meta[cid]
        if not _is_relevant(fpath, decl_name, file_status, text_lower):
            continue
        basename = os.path.basename(fpath or '')
        name_str = decl_name or '?'
        # Build suffix. Priority order:
        #   1. data_blocked_on: suppress discharge; show blocked-on bd-id
        #   2. DISCHARGES target
        #   3. CONTRACT awaiting
        #   4. file_status token
        # file_status semantics:
        #   open-contract      → statements trustworthy; discharge is appropriate (if not data-blocked)
        #   contract-skeleton  → PLACEHOLDERS; repair statements first, route to owning bd-id
        #   statement-suspect  → lean-audit vacuity flag; route to REVIEW, not discharge or repair
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


_RELATION_PATTERNS = [
    re.compile(r'\[([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\]'),   # [A, B] commutator
    re.compile(r'\{([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\}'),   # {A, B} anticommutator
    re.compile(r'\beigenvalues?\s+of\s+([A-Za-z_]\w*)', re.I),
    re.compile(r'\bspectrum\s+of\s+([A-Za-z_]\w*)', re.I),
    re.compile(r'\bcharpoly\s+of\s+([A-Za-z_]\w*)', re.I),
    re.compile(r'\bTr\s*[\(\[]([A-Za-z_]\w*)', re.I),
    re.compile(r'\btrace\s+of\s+([A-Za-z_]\w*)', re.I),
    re.compile(r'\bcommutes?\s+with\b', re.I),
    re.compile(r'\banticommutes?\b', re.I),
    re.compile(r'\brecompute\b', re.I),
]

_RECOMPUTE_RE = re.compile(r'\brecompute\b', re.I)


def query_structural_facts(conn: sqlite3.Connection, text: str) -> list[str]:
    """Surface structural-fact advisories when relation-shaped text + cataloged operators appear.

    Fires when:
      - Relation-shaped patterns ([A,B], {A,B}, 'commutes with', 'eigenvalues of', etc.)
        appear AND at least one known operator name from structural_facts is found in text.
      - OR 'recompute' appears with a known operator name (explicit recompute intent).
    Never blocks; advisory only.
    """
    try:
        conn.execute('SELECT 1 FROM structural_facts LIMIT 1')
    except Exception:
        return []

    # Check whether any relation-shaped pattern fires
    has_relation = any(p.search(text) for p in _RELATION_PATTERNS)
    if not has_relation:
        return []

    # Echo suppression: if text is already citing certified_data (the source),
    # the agent knows — don't quote the registry back at them.
    if re.search(r'certified_data|STRUCTURAL.FACT|ALGEBRA_RELATIONS', text, re.IGNORECASE):
        return []

    # Load catalog of known operators (lhs + rhs)
    known_ops = set()
    rows = conn.execute(
        'SELECT DISTINCT lhs_operator, rhs_operator FROM structural_facts'
    ).fetchall()
    for lhs, rhs in rows:
        # Split on '/' or whitespace in composite names like 'shift_matrix_sq_48 / M_full_48'
        for part in re.split(r'[/\s]+', lhs or ''):
            if len(part) >= 3:
                known_ops.add(part)
        if rhs:
            for part in re.split(r'[/\s]+', rhs):
                if len(part) >= 3:
                    known_ops.add(part)

    # Find which known operators appear in the text
    matched_ops: set[str] = set()
    for op in known_ops:
        # Require word-boundary match; avoid matching 'M_odd' inside 'M_odd_gram' via token check
        if re.search(r'\b' + re.escape(op) + r'\b', text):
            matched_ops.add(op)

    if not matched_ops:
        return []

    # Query structural_facts for all entries whose lhs or rhs matches a found operator
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
                pair = f'{{{lhs_str},{rhs_str}}}' if rtype == 'anticommutator' else f'[{lhs_str},{rhs_str}]' if rtype == 'commutator' else f'{lhs_str}/{rhs_str}'
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


_PROOF_VOCAB_RE = re.compile(
    r'\b(prove|theorem|lemma|sorry|discharge|lean.prover|\.lean\b|proof_by|apply\s+Lean|'
    r'lean\s+proof|sorry.contract|tactic|mathlib)\b',
    re.IGNORECASE,
)


def query_route_to_tip(tool_name: str, ti: dict, prompt_text: str) -> list[str]:
    """Advisory: if non-lean-prover agent dispatch contains proof vocabulary, route to tip."""
    if tool_name != 'Agent':
        return []
    subagent_type = ti.get('subagent_type', '') or ''
    if 'lean' in subagent_type.lower():
        return []  # already going to a lean agent
    if not _PROOF_VOCAB_RE.search(prompt_text):
        return []
    return ['[ROUTE-TO-TIP: dispatch contains proof-writing vocabulary; tip owns proof work. '
            'File a routing-deposit in lean_work_queue instead of implementing inline. '
            'If tip is offline, file a bd task with class=proof-work.]']


def main() -> None:
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', '')
    ti = data.get('tool_input', {})

    # Determine the text to scan
    prompt_text = ''
    if tool_name == 'Agent':
        prompt_text = ti.get('prompt', '')
    elif tool_name == 'Bash':
        cmd = ti.get('command', '')
        if 'bridge send' not in cmd:
            sys.exit(0)
        # Extract all text worth scanning: heredoc body + subject string
        parts = []
        # Heredoc body (handles << 'EOF', << EOF, <<'EOF')
        m = re.search(r"<<\s*'?EOF'?\s*\n(.+?)(?:\nEOF\b|\Z)", cmd, re.DOTALL)
        if m:
            parts.append(m.group(1))
        # Subject string (quoted arg, may have flags between it and EOF)
        m2 = re.search(r'bridge send\s+\S+\s+"([^"]+)"', cmd)
        if m2:
            parts.append(m2.group(1))
        prompt_text = '\n'.join(parts)
    else:
        sys.exit(0)

    if not prompt_text or len(prompt_text) < 20:
        sys.exit(0)

    # EMBEDDING-DOWN gate: ash:8081 down => semantic retrieval is BLIND. Surface a
    # hard STOP at compute/dispatch time so the agent does not forge ahead blind.
    if ash_down() and STOP_LINE:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "additionalContext": STOP_LINE}}))
        sys.exit(0)

    db = os.path.expanduser('~/.cache/kb/knowledge.db')
    if not os.path.exists(db):
        sys.exit(0)

    try:
        conn = sqlite3.connect(db, timeout=3)
        tokens = extract_candidate_tokens(prompt_text)
        fracs = extract_fractions(prompt_text)
        project = _project_from_cwd()
        advisories = query_db(conn, tokens, fracs, project=project)
        advisories += query_contracts(conn, tokens, project=project, raw_text=prompt_text)
        advisories += query_structural_facts(conn, prompt_text)
        conn.close()
        advisories += query_route_to_tip(tool_name, ti, prompt_text)
        if advisories:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": "\n".join(advisories),
                }
            }))
    except Exception:
        pass  # never block on failure

    sys.exit(0)


if __name__ == '__main__':
    main()

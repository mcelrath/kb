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


def extract_fractions(text: str) -> list[str]:
    """Extract 'N/D' style exact fractions."""
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
        f"AND meaning IS NOT NULL "
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

    # Only surface contracts with >= 2 distinct token hits (noise filter)
    contract_candidates: list[tuple[str, str]] = []
    for cid, hits in contract_hits.items():
        if hits < 2:
            continue
        fpath, line, decl_name, file_status, discharge_target, contract_awaiting, proof_grade, data_blocked_on = contract_meta[cid]
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
        conn.close()
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

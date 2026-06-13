"""CLI handlers and validators for lean commands: lean-verify, queue-defer.

Also contains _validate_lean_tags() — extracted from main() per R4.
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Tag vocabulary validator (was _validate_lean_tags inside main())
# ---------------------------------------------------------------------------

def validate_lean_tags(tags: list[str] | None, content: str, evidence: str | None) -> list[str]:
    """Return list of validation error strings (empty = OK).

    Vocabulary (one primary status tag per entry):
      lean:proven              0 sorry; pure-kernel axioms only (native_decide allowed-but-surfaced);
                               requires: TheoremName (file:line), commit hash, axiom-set, reviewer+date
      lean:proven-conditional  Proven glue; open hypotheses must be listed
      lean:scope-guarded       Proven for narrower object than name suggests; scope+exclusion stated
      lean:contracted          Named sorry-contract/Prop slot; discharge route + bd-id required
      lean:axiom               Load-bearing axiom; "what closes it" required
      lean:build-blocked       Math done, olean can't complete (OOM/timeout); RSS datum + bd-id required
      lean:emitted-unverified  Landing-pad literals, provenance-gated, no verified build yet
      lean:vacuity-suspect     lean-audit flag; mandatory-read (no structural add-time requirement)
      lean:refuted             False or route proven-closed; refutation pointer required
      lean:superseded          Replaced; successor kb-id or TheoremName required
    """
    if not tags:
        return []
    errors = []
    combined = (content + " " + (evidence or "")).lower()
    raw = content + " " + (evidence or "")

    if "lean:proven" in tags:
        has_file_line = bool(re.search(r'\S+\.lean:\d+', raw))
        has_commit = bool(re.search(r'\b[0-9a-f]{7,40}\b', raw))
        has_axiom = bool(re.search(
            r'axiom[-_\s]?set|axioms?:|propext|classical\.choice|quot\.sound|nonstd.axiom|native_decide',
            combined
        ))
        has_reviewer = bool(re.search(
            r'\b(reviewer|reviewed|non.?vacuity|non_vacuity)\b', combined
        ))
        missing = []
        if not has_file_line:
            missing.append("file:line (e.g. Proofs/Foo.lean:42)")
        if not has_commit:
            missing.append("commit hash (≥7 hex chars)")
        if not has_axiom:
            missing.append("axiom-set (list axioms; native_decide must appear if used)")
        if not has_reviewer:
            missing.append("non-vacuity reviewer + date")
        if missing:
            errors.append(
                f"lean:proven requires: {', '.join(missing)}. "
                "Add missing fields or downgrade to lean:contracted."
            )

    if "lean:proven-conditional" in tags:
        has_hypotheses = bool(re.search(
            r'\b(given|assuming|hypothesis|hypotheses|conditional on|requires|open hyp)\b',
            combined
        ))
        if not has_hypotheses:
            errors.append(
                "lean:proven-conditional requires named open hypotheses "
                "(e.g. 'proven given DRI + IsHBLine')."
            )

    if "lean:scope-guarded" in tags:
        has_exclusion = bool(re.search(
            r'\b(not|does not|doesn\'t|only|narrow|excludes?|discharge|scope)\b', combined
        ))
        if not has_exclusion:
            errors.append(
                "lean:scope-guarded requires explicit scope+exclusion statement "
                "(e.g. 'proves E_channel only, NOT E_DRI')."
            )

    if "lean:contracted" in tags:
        has_route = bool(re.search(r'\bbd[-_]?\w+|\bsc#\d+|discharge|route\b', combined))
        if not has_route:
            errors.append(
                "lean:contracted requires a discharge route and bd-id "
                "(e.g. 'discharge route: sorry-contract SC1; bd sc#1234')."
            )

    if "lean:axiom" in tags:
        has_closure = bool(re.search(r'\b(close[sd]?|discharge[sd]?|would|to close|open)\b', combined))
        if not has_closure:
            errors.append(
                "lean:axiom requires stating what would close it "
                "(e.g. 'closed by: proof of X')."
            )

    if "lean:build-blocked" in tags:
        has_datum = bool(re.search(r'\b(oom|rss|timeout|mem|gb|mb)\b', combined))
        has_bd = bool(re.search(r'\bbd[-_]?\w+|\bsc#\d+', combined))
        missing = []
        if not has_datum:
            missing.append("timeout/RSS datum (e.g. 'OOM at 48 GB')")
        if not has_bd:
            missing.append("refactor bd-id")
        if missing:
            errors.append(
                f"lean:build-blocked requires: {', '.join(missing)}."
            )

    if "lean:refuted" in tags and not evidence:
        errors.append(
            "lean:refuted requires evidence (-e) pointing to the "
            "refutation proof or counterexample."
        )

    if "lean:superseded" in tags:
        has_successor = bool(re.search(
            r'kb-\d{8}-\w+|successor|replaced by|see\s+\S+\.lean', combined
        ))
        if not has_successor:
            errors.append(
                "lean:superseded requires naming the successor "
                "(kb-id or TheoremName)."
            )

    return errors


# ---------------------------------------------------------------------------
# lean-verify
# ---------------------------------------------------------------------------

def run_lean_verify(kb, args) -> None:
    import subprocess as _sp

    finding = kb.get(args.id)
    if not finding:
        print(f"Finding not found: {args.id}")
        sys.exit(1)

    tags = finding.get("tags") or []
    lean_tags = [t for t in tags if t.startswith("lean:")]
    if not lean_tags:
        print(f"No lean: tags on {args.id}; nothing to verify.")
        sys.exit(0)

    combined_text = finding["content"] + " " + (finding["evidence"] or "")
    file_line_match = re.search(r'(\S+\.lean):(\d+)', combined_text)
    if not file_line_match:
        print(f"No file:line reference found in {args.id}.")
        print(f"  Content preview: {finding['content'][:200]}")
        print("  lean-verify requires 'path/to/File.lean:N' in the entry.")
        sys.exit(1)

    lean_rel = file_line_match.group(1)
    cited_line = int(file_line_match.group(2))

    search_roots = [
        Path.home() / "Physics/claude/proofs",
        Path.home() / "Physics/secular-constraints",
        Path.home() / "Physics/mathlib4",
    ]
    if args.search_path:
        search_roots = [Path(p) for p in args.search_path] + search_roots

    lean_file: Path | None = None
    for root in search_roots:
        candidate = root / lean_rel
        if candidate.exists():
            lean_file = candidate
            break
        stem = Path(lean_rel).name
        for match in root.rglob(stem):
            if match.suffix == ".lean":
                lean_file = match
                break
        if lean_file:
            break

    if lean_file is None:
        print(f"DRIFT-UNKNOWN: could not locate {lean_rel} in known roots.")
        print("  Pass --search-path DIR to extend the search.")
        sys.exit(2)

    lean_audit = Path.home() / ".local/bin/lean-audit"
    if not lean_audit.exists():
        lean_audit = Path("/usr/local/bin/lean-audit")
    if not lean_audit.exists():
        print("lean-audit not found; install or check PATH.")
        sys.exit(1)

    proc = _sp.run(
        [str(lean_audit), str(lean_file), "--json"],
        capture_output=True, text=True
    )
    if proc.returncode not in (0, 1):
        print(f"lean-audit error: {proc.stderr[:400]}")
        sys.exit(1)

    try:
        audit = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"lean-audit output not JSON:\n{proc.stdout[:400]}")
        sys.exit(1)

    file_key = str(lean_file)
    entry = audit.get(file_key) or next((v for v in audit.values()), None)

    drift_issues = []
    if entry:
        sorry_count = entry.get("sorry_count", 0)
        nonstd = entry.get("nonstd_axioms", [])
        build_status = entry.get("status", "unknown")

        if "lean:proven" in lean_tags:
            if sorry_count > 0:
                drift_issues.append(
                    f"DRIFT: lean:proven entry now has {sorry_count} sorry in {lean_file.name}"
                )
            if build_status not in ("clean", "CLEAN"):
                drift_issues.append(
                    f"DRIFT: lean-audit status is '{build_status}' (expected CLEAN)"
                )

        if drift_issues:
            print(f"DRIFT DETECTED on {args.id}:")
            for d in drift_issues:
                print(f"  {d}")
            print(f"  File: {lean_file}")
            print(f"  Cited line: {cited_line}")
            print(f"  Tags: {', '.join(lean_tags)}")
            print("  Action: use 'kb correct' to update the entry or fix the proof.")
            sys.exit(3)
        else:
            print(f"CLEAN: {args.id}")
            print(f"  File: {lean_file}")
            print(f"  lean-audit: status={build_status} sorry={sorry_count}")
            if nonstd:
                print(f"  nonstd-axioms (allowed): {', '.join(nonstd)}")
            print(f"  Tags: {', '.join(lean_tags)}")
    else:
        print(f"DRIFT-UNKNOWN: lean-audit returned no entry for {lean_file.name}")
        print(f"  Raw output: {proc.stdout[:400]}")
        sys.exit(2)

    for lt in lean_tags:
        theorem_name = lt[len("lean:"):] if lt.startswith("lean:") else lt
        if not theorem_name or theorem_name in ("proven", "sorry", "partial"):
            continue
        rows = kb.conn.execute(
            "SELECT id, statement, statement_pure FROM lean_theorems WHERE lean_name LIKE ?",
            (f"%{theorem_name}%",),
        ).fetchall()
        if not rows:
            print(f"[NOT INDEXED: {theorem_name} not in lean_theorems — run kb ingest lean]")
        else:
            stored_stmt = rows[0][2] or rows[0][1]
            if entry:
                declarations = entry.get("declarations", [])
                for decl in declarations:
                    decl_name = decl.get("name", "") if isinstance(decl, dict) else str(decl)
                    if theorem_name in decl_name:
                        decl_stmt = decl.get("statement", "") if isinstance(decl, dict) else ""
                        if decl_stmt and stored_stmt and decl_stmt.strip() != stored_stmt.strip():
                            print(f"[DRIFT: lean_theorems.statement differs from current source for {theorem_name} — re-ingest needed]")
                        break


# ---------------------------------------------------------------------------
# queue-defer
# ---------------------------------------------------------------------------

def run_queue_defer(kb, args) -> None:
    import os.path as _op

    _VALID_DEFER_PREFIXES = (
        "data_blocked_on:", "design-pending:", "file-conflict:",
        "agent-cap", "user-gate:", "verify-first:",
    )

    if args.list or not args.row_id:
        rows = kb.list_deferred_queue_rows(limit=50)
        if not rows:
            print("lean_work_queue: no deferred rows")
        else:
            print(f"lean_work_queue: {len(rows)} deferred row(s)")
            for rid, file, decl, cls, readiness, defer_reason, defer_detail, ts in rows:
                fname = _op.basename(file or "?")
                detail_str = f" ({defer_detail})" if defer_detail else ""
                print(f"  {rid[:10]}  {readiness} {cls}: {fname}::{decl or '(file-level)'}  reason={defer_reason}{detail_str}  [{ts}]")
        sys.exit(0)

    row_id = args.row_id
    existing = kb.get_queue_row(row_id)
    if not existing:
        print(f"queue-defer: row '{row_id}' not found in lean_work_queue", file=sys.stderr)
        sys.exit(1)

    if args.reason is None:
        kb.clear_defer_reason(row_id)
        print(f"queue-defer: cleared defer on {row_id[:10]} (row re-activated)")
        sys.exit(0)

    reason = args.reason
    detail = args.detail or ""

    valid = any(reason == p or reason.startswith(p) for p in _VALID_DEFER_PREFIXES)
    if not valid:
        valid_list = ", ".join(_VALID_DEFER_PREFIXES)
        print(
            f"queue-defer: invalid reason '{reason}'. Valid prefixes: {valid_list}",
            file=sys.stderr,
        )
        sys.exit(1)

    kb.set_defer_reason(row_id, reason, detail or None)
    _, cls, readiness, _ = existing
    print(f"queue-defer: deferred {row_id[:10]} ({readiness} {cls}) — reason: {reason}" + (f" ({detail})" if detail else ""))
    sys.exit(0)

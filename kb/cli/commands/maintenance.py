"""CLI handlers for maintenance commands: review, refresh, ask, questions, related."""

import sys

import kb.cli.output as output
from kb.cli.output import _fmt_one_line, format_finding


# ---------------------------------------------------------------------------
# Refresh helpers (moved from kb.py)
# ---------------------------------------------------------------------------

def _run_refresh(kb, rows, dry_run: bool, commit_every: int = 1, label: str = "refresh"):
    """Core loop: summarize + retag + reembed each finding row.

    Embedding (ash:8081) and LLM (tardis:9510) run concurrently per row.
    All I/O completes before any DB write opens. Each row is written with its
    own BEGIN IMMEDIATE/COMMIT via kb.update_finding_refresh() — lock held for
    microseconds, not seconds.

    rows: list of (id, project, content, evidence)
    Returns (ok, fail) counts.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    from kb.validation import serialize_f32, l2_normalize

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    ok = fail = 0
    total = len(rows)
    t0 = _time.time()

    bar = _tqdm(total=total, desc=label, unit="row",
                dynamic_ncols=True) if _tqdm and not dry_run else None
    if bar:
        bar.set_postfix(ok=0, fail=0)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for fid, fproject, content, evidence in rows:
                embed_text = content + (" " + evidence if evidence else "")

                if dry_run:
                    summary = kb._analyzer.generate_summary(content, evidence)
                    print(f"[DRY] {fid} ({fproject}): {summary}")
                    ok += 1
                    if bar:
                        bar.set_postfix(ok=ok, fail=fail)
                        bar.update(1)
                    continue

                existing_tags = kb._fetch_existing_tags(fproject)

                def _embed(t=embed_text):
                    return kb._embedding._embed_remote(t, max_retries=5, base_delay=1.5)

                def _llm(c=content, e=evidence, et=existing_tags):
                    s = kb._analyzer.generate_summary(c, e)
                    t = kb._analyzer.suggest_tags(c, et)
                    return s, t

                # Both network calls run concurrently; no DB lock held.
                embed_fut = pool.submit(_embed)
                llm_fut   = pool.submit(_llm)

                embedding = None
                try:
                    embedding = serialize_f32(l2_normalize(embed_fut.result()))
                except Exception as e:
                    (bar.write if bar else print)(f"  EMBED FAIL {fid}: {e}")

                summary = tags = None
                try:
                    summary, tags = llm_fut.result()
                except Exception as e:
                    (bar.write if bar else print)(f"  LLM FAIL {fid}: {e}")

                if summary and len(summary) >= 10:
                    kb.update_finding_refresh(fid, summary, tags, embedding)
                    ok += 1
                else:
                    fail += 1
                    (bar.write if bar else print)(
                        f"  FAIL {fid} ({fproject}): {(content or '')[:60]!r}"
                    )

                if bar:
                    bar.set_postfix(ok=ok, fail=fail)
                    bar.update(1)

    except KeyboardInterrupt:
        if bar:
            bar.write(f"\nInterrupted at {ok+fail}/{total}")
            bar.close()
        else:
            print(f"\nInterrupted at {ok+fail}/{total}")
        elapsed = _time.time() - t0
        print(f"{label}: ok={ok} fail={fail} (interrupted after {elapsed/60:.1f}m)")
        return ok, fail

    if bar:
        bar.close()
    elapsed = _time.time() - t0
    print(f"{label}: ok={ok} fail={fail} total={total} elapsed={elapsed/60:.1f}m")
    return ok, fail


def _fetch_refresh_rows(kb, ids=None, project=None, all_rows=False, limit=0):
    """Build the findings row list for refresh/retag/resummarize."""
    return kb.fetch_refresh_rows(ids=ids, project=project, all_rows=all_rows, limit=limit)


def _backfill_statement_pure(kb, project=None, limit=None, workers=8, dry_run=False):
    """Backfill statement_pure for lean theorems using the KB's LLM client."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from kb.entities.theorems import TheoremRepository
    theorems = TheoremRepository(kb.conn, kb.embedding)
    rows = theorems.fetch_missing_statement_pure(project=project, limit=limit)
    print(f"  theorem backfill: {len(rows)} without statement_pure")
    if not rows:
        return {"updated": 0, "failed": 0}

    if dry_run:
        for tid, lean_name, stmt in rows[:3]:
            print(f"  [DRY] {lean_name}: {stmt[:80]}")
        return {"updated": 0, "failed": 0}

    PROMPT = (
        "Restate this Lean 4 theorem in pure mathematical language. "
        "No Lean syntax, no type annotations. Standard math notation. "
        "One sentence, under 30 words.\n\nLean:\n{statement}\n\nMath:"
    )

    def restate_one(row):
        tid, lean_name, statement = row
        result = kb._analyzer.llm_client.complete(
            PROMPT.format(statement=statement[:600]),
            max_tokens=80, temperature=0.1, timeout=30,
        )
        if result:
            result = result.strip().strip('"').strip("'")
        return tid, lean_name, result or None

    updated = failed = 0
    t0 = _time.time()
    conn = kb.conn

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(restate_one, row): row for row in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            tid, lean_name, pure = fut.result()
            if pure:
                theorems.set_statement_pure(tid, pure)
                updated += 1
            else:
                failed += 1
            if i % 100 == 0:
                conn.commit()
                elapsed = _time.time() - t0
                rate = i / max(elapsed, 0.001)
                print(f"  {i}/{len(rows)} done  {rate:.1f}/s", flush=True)

    conn.commit()

    # Re-embed updated theorems
    if updated > 0:
        print(f"  re-embedding {updated} theorems...")
        updated_rows = conn.execute(
            "SELECT id, statement_pure FROM lean_theorems "
            "WHERE statement_pure IS NOT NULL AND statement_pure != ''"
        ).fetchall()
        for j, (tid, pure) in enumerate(updated_rows):
            theorems.reembed_statement_pure(tid, pure)
            if j % 100 == 0:
                conn.commit()
        conn.commit()

    elapsed = _time.time() - t0
    print(f"  theorem backfill done: updated={updated} failed={failed} elapsed={elapsed:.0f}s")
    return {"updated": updated, "failed": failed}


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def run_review(kb, args) -> None:
    result = kb.review_queue(project=args.project, limit=args.limit)
    any_issues = False
    for category, items in result.items():
        if items:
            any_issues = True
            header = f"\n{output.c(category.upper(), 'bold')} {output.c(f'({len(items)})', 'cyan')}:"
            print(header)
            for item in items:
                p_label = f"({item['project']})" if item.get('project') else ""
                proj = f" {output.c(p_label, 'dim')}" if p_label else ""
                row = f"  {output.c(item['id'], 'yellow')}{proj}: {item.get('content', '')[:60]}..."
                print(output.fit_line(row))
    if not any_issues:
        print("No findings need attention.")


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------

def run_refresh(kb, args) -> None:
    targets = args.targets or []
    explicit_ids = [t for t in targets if t.startswith("kb-")]
    project_args = [t for t in targets if not t.startswith("kb-")]
    if len(project_args) > 1:
        print(f"Error: only one project name allowed, got: {project_args}")
        sys.exit(1)
    project = args.project or (project_args[0] if project_args else None)
    rows = _fetch_refresh_rows(
        kb,
        ids=explicit_ids or None,
        project=project,
        all_rows=args.all,
        limit=args.limit,
    )
    print(f"{output.c('refresh:', 'bold')} {output.c(str(len(rows)), 'cyan')} findings "
          f"(project={project or 'ALL'}, all={args.all}, dry={args.dry_run})"
          f"\n  (Ctrl+C safe: each row committed immediately after I/O completes)")
    _run_refresh(kb, rows, dry_run=args.dry_run, commit_every=1)
    if args.theorems:
        _backfill_statement_pure(
            kb, project=project,
            workers=args.theorem_workers, dry_run=args.dry_run,
        )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

def run_ask(kb, args) -> None:
    result = kb.ask(question=args.question, project=args.project, limit=args.limit)
    # Answer body is multi-line prose — never truncate; just colorize the header.
    print(output.c(f"Answer: {args.question}", 'bold'))
    print(result['answer'])
    if result.get('sources'):
        print(f"\n{output.c('Sources:', 'bold')}")
        for s in result['sources']:
            print(f"  - {s}")


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------

def run_questions(kb, args) -> None:
    questions = kb.generate_open_questions(
        project=args.project,
        limit=args.limit,
        input_limit=args.input,
        query=args.query,
    )
    if not questions:
        print("No questions generated (try adding more findings or a search query).")
    else:
        seed = f'"{args.query}"' if args.query else "recent findings"
        print(f"{output.c('Open questions', 'bold')} from {seed}:\n")
        for i, q in enumerate(questions, 1):
            print(f"{output.c(str(i) + '.', 'cyan')} {q.get('question', '')}")
            if q.get('why'):
                print(f"   {output.c('Why:', 'dim')} {q['why']}")
            if q.get('related_ids'):
                print(f"   {output.c('See:', 'dim')} {', '.join(q['related_ids'][:3])}")
            print()


# ---------------------------------------------------------------------------
# related
# ---------------------------------------------------------------------------

def run_related(kb, args) -> None:
    results = kb.related(finding_id=args.id, limit=args.limit)
    if not results:
        print("No related findings.")
    elif args.long:
        for f in results:
            print(format_finding(f))
            print()
    else:
        for f in results:
            print(_fmt_one_line(f))

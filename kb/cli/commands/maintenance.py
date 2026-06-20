"""CLI handlers for maintenance commands: review, refresh, ask, questions, related."""

import sys

import kb.cli.output as output


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

def run_refresh(kb, args, fetch_refresh_rows_fn, run_refresh_fn, backfill_statement_pure_fn) -> None:
    targets = args.targets or []
    explicit_ids = [t for t in targets if t.startswith("kb-")]
    project_args = [t for t in targets if not t.startswith("kb-")]
    if len(project_args) > 1:
        print(f"Error: only one project name allowed, got: {project_args}")
        sys.exit(1)
    project = args.project or (project_args[0] if project_args else None)
    rows = fetch_refresh_rows_fn(
        kb,
        ids=explicit_ids or None,
        project=project,
        all_rows=args.all,
        limit=args.limit,
    )
    print(f"{output.c('refresh:', 'bold')} {output.c(str(len(rows)), 'cyan')} findings "
          f"(project={project or 'ALL'}, all={args.all}, dry={args.dry_run})"
          f"\n  (Ctrl+C safe: each row committed immediately after I/O completes)")
    run_refresh_fn(kb, rows, dry_run=args.dry_run, commit_every=1)
    if args.theorems:
        backfill_statement_pure_fn(
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

def run_related(kb, args, fmt_one_line_fn, format_finding_fn) -> None:
    results = kb.related(finding_id=args.id, limit=args.limit)
    if not results:
        print("No related findings.")
    elif args.long:
        for f in results:
            print(format_finding_fn(f))
            print()
    else:
        for f in results:
            print(fmt_one_line_fn(f))

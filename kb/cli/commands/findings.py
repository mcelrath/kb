"""CLI handlers for findings commands: add, search, list, get, correct, delete."""

import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

def run_add(kb, args, add_content: str, queue_async_add_fn) -> None:
    """Handle `kb add`. add_content and lean validation already done in main()."""
    import os as _os

    _saved_retries = _os.environ.get("KB_EMBED_MAX_RETRIES")
    _os.environ["KB_EMBED_MAX_RETRIES"] = "1"
    try:
        result = kb.add(
            content=add_content,
            finding_type=args.type,
            project=args.project,
            sprint=args.sprint,
            tags=args.tags,
            evidence=args.evidence,
            summary=args.summary,
            check_duplicate=not args.no_duplicate_check,
            auto_tag=not args.no_auto_tag,
        )
    except Exception as e:
        err = str(e)
        if any(kw in err for kw in ("Remote embedding", "Connection refused",
                                     "RemoteDisconnected", "URLError", "HTTPError",
                                     "503", "502", "504", "TimeoutError")):
            queue_async_add_fn(
                content=add_content,
                finding_type=args.type,
                project=args.project,
                sprint=args.sprint,
                tags=args.tags,
                evidence=args.evidence,
                summary=args.summary,
            )
            sys.exit(0)
        raise
    finally:
        if _saved_retries is None:
            _os.environ.pop("KB_EMBED_MAX_RETRIES", None)
        else:
            _os.environ["KB_EMBED_MAX_RETRIES"] = _saved_retries

    if result.get("duplicate"):
        print(f"Warning: Similar finding exists: {result['duplicate']['id']} (similarity: {result['duplicate']['similarity']:.2f})")
        print(f"  {result['duplicate']['content'][:100]}...")
        print(f"\nAdded anyway with ID: {result['id']}")
    else:
        print(f"Added: {result['id']}")
        if result.get("tags_suggested"):
            print(f"  Auto-tagged: {', '.join(result.get('tags', []))}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def run_search(kb, args, load_session_seen_ids_fn, fmt_one_line_fn, format_finding_fn) -> None:
    exclude_ids: set[str] = set(args.exclude or [])
    if not args.no_dedup:
        exclude_ids |= load_session_seen_ids_fn()
    results = kb.search(
        query=args.query,
        limit=args.limit + len(exclude_ids),
        project=args.project,
        finding_type=args.type,
        include_superseded=args.include_superseded,
        exclude_ids=exclude_ids or None,
        # Unified retrieval: also surface ingested document_sections (PDF/markdown
        # docs) alongside findings, so `kb search` and the surfacing hook inject
        # doc content. Phase-4 built search_sections but left it unwired (default
        # off); the CLI turns it on.
        include_sections=True,
    )
    results = results[:args.limit]
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    elif not results:
        print("No results found")
    elif args.long:
        for finding in results:
            print(format_finding_fn(finding, verbose=args.verbose))
            print()
    else:
        for finding in results:
            print(fmt_one_line_fn(finding))


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def run_list(kb, args, fmt_one_line_fn, format_finding_fn) -> None:
    results = kb.list_findings(
        limit=args.limit,
        project=args.project,
        sprint=args.sprint,
        finding_type=args.type,
        include_superseded=args.include_superseded,
    )
    if not results:
        print("No findings")
    elif args.long:
        for finding in results:
            print(format_finding_fn(finding, verbose=args.verbose))
            print()
    else:
        for finding in results:
            print(fmt_one_line_fn(finding))


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def run_get(kb, args) -> None:
    from kb.markdown import format_finding_markdown

    try:
        from rich.console import Console
        from rich.markdown import Markdown
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False

    finding = kb.get(args.id)
    if not finding:
        print(f"Finding not found: {args.id}")
        sys.exit(1)

    md = format_finding_markdown(finding)
    if args.raw or not RICH_AVAILABLE:
        print(md)
    else:
        console = Console()
        console.print(Markdown(md))


# ---------------------------------------------------------------------------
# correct
# ---------------------------------------------------------------------------

def run_correct(kb, args) -> None:
    result = kb.correct(
        supersedes_id=args.id,
        content=args.content,
        evidence=args.evidence,
        reason=args.reason,
    )
    print(f"Created correction: {result['id']}")
    print(f"  Supersedes: {args.id}")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def run_delete(kb, args) -> None:
    finding = kb.get(args.id)
    if not finding:
        print(f"Finding not found: {args.id}")
        sys.exit(1)

    if not args.force:
        print(f"About to delete: {args.id}")
        print(f"  Type: {finding['type']}")
        print(f"  Content: {finding['content'][:100]}...")
        confirm = input("Confirm delete? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled")
            sys.exit(0)

    kb.delete(args.id)
    print(f"Deleted: {args.id}")

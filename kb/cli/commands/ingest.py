"""CLI handler for the ingest command group (lean, scripts, python, tex)."""

import sys
from pathlib import Path


def _cwd_project() -> str | None:
    """Default a document's project to the cwd-scoped project (git-root basename,
    via kbt's resolver) when -p is omitted — otherwise ingested PDFs/markdown get a
    NULL project and surface as '(?)'. An explicit -p always wins."""
    try:
        from kb.issue_cli import _current_project_name
        return _current_project_name()
    except Exception:
        return None


def run_ingest(kb, args, ingest_parser) -> None:
    if args.ingest_cmd == "lean":
        from kb.ingest.lean import run
        rc = run(
            dry_run=args.dry_run,
            limit=args.limit,
            no_summarize=getattr(args, "no_summarize", False),
            summarize_only=getattr(args, "summarize_only", False),
            files=getattr(args, "files", None),
        )
        if rc:
            sys.exit(rc)

    elif args.ingest_cmd == "scripts":
        from kb.ingest.scripts import run
        rc = run(
            directory=Path(args.directory),
            project=args.project,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        if rc:
            sys.exit(rc)

    elif args.ingest_cmd == "python":
        from kb.ingest.python import run
        rc = run(
            root=Path(args.root) if args.root else None,
            files=getattr(args, "files", None),
            project=args.project,
            dry_run=args.dry_run,
            db_path=args.db,
            with_notations=getattr(args, "with_notations", False),
        )
        if rc:
            sys.exit(rc)

    elif args.ingest_cmd == "typescript":
        from kb.ingest.typescript import run
        rc = run(
            root=Path(args.root) if args.root else None,
            files=getattr(args, "files", None),
            deleted=getattr(args, "deleted", None),
            project=args.project,
            dry_run=args.dry_run,
            db_path=args.db,
        )
        if rc:
            sys.exit(rc)

    elif args.ingest_cmd == "rust":
        from kb.ingest.rust import run
        rc = run(
            root=Path(args.root) if args.root else None,
            files=getattr(args, "files", None),
            deleted=getattr(args, "deleted", None),
            project=args.project,
            dry_run=args.dry_run,
            db_path=args.db,
        )
        if rc:
            sys.exit(rc)

    elif args.ingest_cmd == "tex":
        from kb.ingest.tex import run
        rc = run(
            root=Path(args.root) if args.root else None,
            files=getattr(args, "files", None),
            project=args.project,
            dry_run=args.dry_run,
            db_path=args.db,
        )
        if rc:
            sys.exit(rc)

    elif args.ingest_cmd == "personas":
        from kb.ingest.personas import run
        rc = run(
            roots=getattr(args, "roots", None) or None,
            dry_run=getattr(args, "dry_run", False),
            check=getattr(args, "check", False),
            project_filter=getattr(args, "project_filter", None),
            db_path=args.db,
        )
        if rc:
            sys.exit(rc)

    elif args.ingest_cmd == "md":
        from kb.ingest.markdown import run
        rc = run(
            file_path=Path(args.file),
            db_path=args.db,
            doc_type=getattr(args, "doc_type", None),
            project=getattr(args, "project", None) or _cwd_project(),
            title=getattr(args, "title", None),
            summary=getattr(args, "summary", None),
            dry_run=getattr(args, "dry_run", False),
        )
        if rc:
            sys.exit(rc)

    elif args.ingest_cmd == "pdf":
        from kb.ingest.pdf import run
        rc = run(
            file_path=Path(args.file),
            db_path=args.db,
            doc_type=getattr(args, "doc_type", None),
            project=getattr(args, "project", None) or _cwd_project(),
            title=getattr(args, "title", None),
            summary=getattr(args, "summary", None),
            dry_run=getattr(args, "dry_run", False),
        )
        if rc:
            sys.exit(rc)

    else:
        ingest_parser.print_help()

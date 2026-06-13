"""CLI handler for the ingest command group (lean, scripts, python, tex)."""

import sys
from pathlib import Path


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
            no_notations=getattr(args, "no_notations", False),
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

    else:
        ingest_parser.print_help()

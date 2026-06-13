"""CLI handler for the ingest command group (lean, scripts, python, tex)."""

import sys
from pathlib import Path


def run_ingest(kb, args, ingest_parser) -> None:
    import subprocess as _sp
    # scripts/ lives at the project root: kb/cli/commands/ -> cli/ -> kb/ -> project root
    scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"

    if args.ingest_cmd == "lean":
        script_path = scripts_dir / "ingest_lean_direct.py"
        if not script_path.exists():
            print(f"Error: {script_path} not found")
            sys.exit(1)
        cmd = [sys.executable, str(script_path)]
        if args.dry_run:
            cmd += ["--dry-run"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if getattr(args, "no_summarize", False):
            cmd += ["--no-summarize"]
        if getattr(args, "summarize_only", False):
            cmd += ["--summarize-only"]
        if getattr(args, "files", None):
            cmd += ["--files"] + args.files
        result = _sp.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    elif args.ingest_cmd == "scripts":
        # auto_register_scripts.py lives at the project root
        script_path = Path(__file__).parent.parent.parent.parent / "auto_register_scripts.py"
        if not script_path.exists():
            print(f"Error: {script_path} not found")
            sys.exit(1)
        cmd = [sys.executable, str(script_path), str(args.directory),
               "-p", args.project, "-n", str(args.limit)]
        if args.dry_run:
            cmd += ["--dry-run"]
        _sp.run(cmd)

    elif args.ingest_cmd == "python":
        script_path = scripts_dir / "ingest_python.py"
        if not script_path.exists():
            print(f"Error: {script_path} not found")
            sys.exit(1)
        cmd = [sys.executable, str(script_path), "--root", args.root,
               "--project", args.project]
        if getattr(args, "files", None):
            cmd += ["--files"] + args.files
        if args.dry_run:
            cmd += ["--dry-run"]
        if getattr(args, "no_notations", False):
            cmd += ["--no-notations"]
        cmd += ["--db", str(args.db)]
        result = _sp.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    elif args.ingest_cmd == "tex":
        script_path = scripts_dir / "ingest_tex.py"
        if not script_path.exists():
            print(f"Error: {script_path} not found")
            sys.exit(1)
        cmd = [sys.executable, str(script_path), "--root", args.root,
               "--project", args.project]
        if getattr(args, "files", None):
            cmd += ["--files"] + args.files
        if args.dry_run:
            cmd += ["--dry-run"]
        cmd += ["--db", str(args.db)]
        result = _sp.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    else:
        ingest_parser.print_help()

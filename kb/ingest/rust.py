#!/usr/bin/env python3
"""Ingest Rust symbols (.rs) — thin wrapper over the generic code ingester.

Uses the tree-sitter Rust chunker (kb/code_ingest/chunker.py RUST_CONFIG, kb-asf.4):
fn / struct / enum / trait full-body, impl blocks split per-method, with
parent_impl / visibility / node_type metadata. All walk/insert/prune logic lives
in kb/ingest/code.py (ingest_code).
"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from kb import DEFAULT_DB_PATH
from kb.ingest.code import ingest_code


def run(
    root: Path | None = None,
    files: list[str] | None = None,
    deleted: list[str] | None = None,
    project: str = "kb",
    dry_run: bool = False,
    db_path: Path | None = None,
) -> int:
    return ingest_code("rust", (".rs",), root, files, deleted, project, dry_run, db_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Rust symbols (.rs) into the symbols table")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory (default: cwd)")
    parser.add_argument("--files", nargs="+", metavar="FILE", help="Incremental mode: only these files")
    parser.add_argument("--project", default="kb", help="KB project name (default: kb)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and print, no DB writes")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="KB database path")
    parser.add_argument("--deleted", nargs="+", metavar="FILE", help="Remove rows for these deleted files")
    args = parser.parse_args()
    sys.exit(run(root=args.root, files=args.files, deleted=args.deleted,
                 project=args.project, dry_run=args.dry_run, db_path=args.db))


if __name__ == "__main__":
    main()

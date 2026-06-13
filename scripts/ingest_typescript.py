#!/usr/bin/env python3
"""
Ingest TypeScript symbols from .ts/.tsx files into the KB python_symbols table.

Uses the tree-sitter chunker (kb/code_ingest/chunker.py) with TYPESCRIPT_CONFIG
and TSX_CONFIG — no grep/text-search.  Extracts exported functions, classes,
interfaces, type aliases, enums, and export-const declarations.  Persists each
symbol via KnowledgeBase.add_python_symbol() with the TS metadata columns
parent_impl / visibility / is_signature_only / node_type landed in kb-asf.4.1.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Locate KB package
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from kb import KnowledgeBase, DEFAULT_DB_PATH
from kb.code_ingest.chunker import chunk_file, ChunkResult

# Directories to skip when walking the filesystem
_SKIP_DIRS = {"node_modules", "dist", "build", ".git", ".next", "__pycache__"}


def _ts_module_path(file_path: Path, root: Path) -> str:
    """Convert a .ts/.tsx path to a dotted module name relative to root."""
    try:
        rel = file_path.relative_to(root)
        parts = list(rel.parts)
        # Strip extension (.ts or .tsx)
        if parts and (parts[-1].endswith(".ts") or parts[-1].endswith(".tsx")):
            parts[-1] = parts[-1].rsplit(".", 1)[0]
        # index files: drop trailing 'index'
        if parts and parts[-1] == "index":
            parts.pop()
        return ".".join(parts) if parts else file_path.stem
    except ValueError:
        return file_path.stem


def collect_ts_files(root: Path) -> list[Path]:
    """Walk root and collect all .ts/.tsx files, skipping skip-dirs."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(skip in path.parts for skip in _SKIP_DIRS):
            continue
        if path.suffix in (".ts", ".tsx") and path.is_file():
            files.append(path)
    return files


def chunk_ts_file(
    file_path: Path,
    root: Path,
) -> list[dict[str, Any]]:
    """Chunk a single .ts/.tsx file and return symbol dicts ready for add_python_symbol.

    Returns empty list on parse errors.
    """
    try:
        chunks: list[ChunkResult] = chunk_file(file_path, language="typescript", root=root)
    except Exception as e:
        print(f"  Warning: cannot chunk {file_path}: {e}", file=sys.stderr)
        return []

    module = _ts_module_path(file_path, root)
    symbols: list[dict[str, Any]] = []
    for chunk in chunks:
        extra = chunk.extra or {}
        symbols.append({
            "name": chunk.name,
            "kind": chunk.kind,
            "module": chunk.module if chunk.module and chunk.module != "<source>" else module,
            "signature": chunk.signature,
            "file": chunk.file,
            "line": chunk.line,
            "docstring_summary": chunk.doc_summary,
            # TS-specific metadata from ChunkResult.extra
            "parent_impl": extra.get("parent_impl"),
            "visibility": extra.get("visibility", ""),
            "is_signature_only": bool(extra.get("is_signature_only", False)),
            "node_type": extra.get("node_type"),
        })
    return symbols


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest TypeScript symbols (.ts/.tsx) into the KB python_symbols table"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to walk for .ts/.tsx files (default: cwd)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Incremental mode: process only these files",
    )
    parser.add_argument(
        "--project",
        default="kb",
        help="KB project name (default: kb)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print what would be indexed, without writing to DB",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="KB database path",
    )
    parser.add_argument(
        "--deleted",
        nargs="+",
        metavar="FILE",
        help="Remove all python_symbols rows for these deleted/renamed files",
    )
    args = parser.parse_args()

    kb = KnowledgeBase(db_path=args.db)

    # Handle --deleted: remove rows for explicitly-deleted files and exit.
    if args.deleted:
        for fpath in args.deleted:
            fpath_str = str(Path(fpath).expanduser().resolve())
            n = kb.delete_python_symbols_for_file(fpath_str)
            print(f"Deleted {n} rows for removed file: {fpath_str}", file=sys.stderr)
        return

    root = args.root.expanduser().resolve()

    # Collect files to process
    if args.files:
        files = [Path(f).expanduser().resolve() for f in args.files]
    else:
        files = collect_ts_files(root)

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    print(f"Processing {len(files)} TypeScript files (dry_run={args.dry_run})", file=sys.stderr)

    # Collect all symbols across all files
    all_symbols: list[dict[str, Any]] = []
    file_iter = _tqdm(files, desc="parse", unit="file", dynamic_ncols=True) if _tqdm else files
    try:
        for fpath in file_iter:
            syms = chunk_ts_file(fpath, root)
            for s in syms:
                s["project"] = args.project
            all_symbols.extend(syms)
    except KeyboardInterrupt:
        if _tqdm and hasattr(file_iter, "close"):
            file_iter.close()
        print(f"\nInterrupted during parse — {len(all_symbols)} symbols collected so far")
        return

    if _tqdm and hasattr(file_iter, "close"):
        file_iter.close()
    print(f"Found {len(all_symbols)} symbols", file=sys.stderr)

    if args.dry_run:
        for s in all_symbols[:20]:
            node_type = s.get("node_type") or "?"
            vis = s.get("visibility") or ""
            parent = s.get("parent_impl") or ""
            print(
                f"  [{s['kind']:9}] {s['module']}.{s['name']}"
                f"  node_type={node_type}"
                + (f"  vis={vis}" if vis else "")
                + (f"  parent={parent}" if parent else "")
                + f"  line={s['line']}"
            )
            if s.get("docstring_summary"):
                print(f"             {s['docstring_summary'][:80]}")
        if len(all_symbols) > 20:
            print(f"  ... and {len(all_symbols) - 20} more")
        return

    # Insert symbols — commit every COMMIT_EVERY rows
    COMMIT_EVERY = 50
    new_count = 0
    updated_count = 0
    skipped_count = 0
    sym_iter = _tqdm(all_symbols, desc="insert", unit="sym", dynamic_ncols=True) if _tqdm else all_symbols
    try:
        for i, s in enumerate(sym_iter, 1):
            result = kb.add_python_symbol(
                name=s["name"],
                kind=s["kind"],
                module=s["module"],
                signature=s["signature"],
                file=s["file"],
                line=s["line"],
                status="public",
                is_lru_cached=False,
                frame_hint=None,
                docstring_summary=s["docstring_summary"],
                lean_citations=[],
                kb_refs=[],
                project=s["project"],
                parent_impl=s.get("parent_impl"),
                visibility=s.get("visibility"),
                is_signature_only=bool(s.get("is_signature_only")),
                node_type=s.get("node_type"),
            )
            if result["is_new"]:
                new_count += 1
            elif result.get("skipped"):
                skipped_count += 1
            else:
                updated_count += 1
            if i % COMMIT_EVERY == 0:
                kb.conn.commit()
    except KeyboardInterrupt:
        if _tqdm and hasattr(sym_iter, "close"):
            sym_iter.close()
        kb.conn.commit()
        print(f"\nInterrupted — New: {new_count}  Updated: {updated_count}  Skipped: {skipped_count}")
        return
    if _tqdm and hasattr(sym_iter, "close"):
        sym_iter.close()

    # PRUNE: when ingesting with --files, delete stale rows (symbols that vanished).
    if args.files and not args.dry_run:
        pruned_total = 0
        # Build per-file live (name, module) sets from successfully-parsed symbols
        file_to_live: dict[str, set[tuple[str, str]]] = {}
        for s in all_symbols:
            fpath_str = s["file"]
            if fpath_str not in file_to_live:
                file_to_live[fpath_str] = set()
            file_to_live[fpath_str].add((s["name"], s["module"]))
        for fpath in [str(Path(f).expanduser().resolve()) for f in args.files]:
            live = file_to_live.get(fpath, set())
            n = kb.prune_python_symbols_for_file(fpath, live)
            if n:
                print(f"  Pruned {n} stale symbol(s) from {fpath}", file=sys.stderr)
            pruned_total += n
        if pruned_total:
            print(f"Total pruned: {pruned_total}", file=sys.stderr)

    # Populate also_in_modules: group by name, update rows where name appears in >1 module
    name_to_entries: dict[str, list[dict[str, Any]]] = {}
    for s in all_symbols:
        name_to_entries.setdefault(s["name"], []).append(s)

    multi_module_count = 0
    for name, entries in name_to_entries.items():
        if len(entries) <= 1:
            continue
        also = [{"module": e["module"], "file": e["file"], "line": e["line"]} for e in entries]
        also_json = json.dumps(also)
        kb.conn.execute(
            "UPDATE python_symbols SET also_in_modules = ? WHERE name = ?",
            (also_json, name),
        )
        multi_module_count += 1

    kb.conn.commit()
    print(
        f"New: {new_count}  Updated: {updated_count}"
        f"  Skipped(unchanged): {skipped_count}  Multi-module: {multi_module_count}"
    )


if __name__ == "__main__":
    main()

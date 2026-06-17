#!/usr/bin/env python3
"""Generic tree-sitter code ingester — language-agnostic.

Walks a root for files of the given extensions, chunks each via
kb/code_ingest/chunker.py (chunk_file(language=...)), and persists every symbol
through KnowledgeBase.add_symbol() (which derives the `language` column from the
file extension). The per-language ingesters (typescript, rust, ...) are thin
wrappers around ingest_code() — no per-language duplication of the walk/insert/
prune/also-in-modules machinery.
"""

import json
import sys
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from kb import KnowledgeBase, DEFAULT_DB_PATH
from kb.code_ingest.chunker import chunk_file, ChunkResult

# Directories never worth walking (covers JS, Rust, Python build/vendor trees).
_SKIP_DIRS = {"node_modules", "dist", "build", ".git", ".next", "__pycache__",
              "target", ".venv", "venv", ".eggs"}


def _module_path(file_path: Path, root: Path, extensions: tuple[str, ...]) -> str:
    """Dotted module name relative to root, with the language extension stripped."""
    try:
        rel = file_path.relative_to(root)
        parts = list(rel.parts)
        if parts:
            for ext in extensions:
                if parts[-1].endswith(ext):
                    parts[-1] = parts[-1][: -len(ext)]
                    break
        if parts and parts[-1] in ("index", "mod"):  # index.ts / mod.rs: drop the leaf
            parts.pop()
        return ".".join(parts) if parts else file_path.stem
    except ValueError:
        return file_path.stem


def _collect(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if any(skip in path.parts for skip in _SKIP_DIRS):
            continue
        if path.suffix in extensions and path.is_file():
            out.append(path)
    return out


def _chunk(file_path: Path, root: Path, language: str, extensions: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        chunks: list[ChunkResult] = chunk_file(file_path, language=language, root=root)
    except Exception as e:
        print(f"  Warning: cannot chunk {file_path}: {e}", file=sys.stderr)
        return []
    module = _module_path(file_path, root, extensions)
    syms: list[dict[str, Any]] = []
    for chunk in chunks:
        extra = chunk.extra or {}
        syms.append({
            "name": chunk.name,
            "kind": chunk.kind,
            "module": chunk.module if chunk.module and chunk.module != "<source>" else module,
            "signature": chunk.signature,
            "file": chunk.file,
            "line": chunk.line,
            "docstring_summary": chunk.doc_summary,
            "parent_impl": extra.get("parent_impl"),
            "visibility": extra.get("visibility", ""),
            "is_signature_only": bool(extra.get("is_signature_only", False)),
            "node_type": extra.get("node_type"),
        })
    return syms


def ingest_code(
    language: str,
    extensions: tuple[str, ...],
    root: Path | None = None,
    files: list[str] | None = None,
    deleted: list[str] | None = None,
    project: str = "kb",
    dry_run: bool = False,
    db_path: Path | None = None,
) -> int:
    """Ingest symbols for one language. Returns 0 on success, 1 on fatal error."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if root is None:
        root = Path.cwd()
    kb = KnowledgeBase(db_path=db_path)

    if deleted:
        for fpath in deleted:
            fpath_str = str(Path(fpath).expanduser().resolve())
            n = kb.delete_symbols_for_file(fpath_str)
            print(f"Deleted {n} rows for removed file: {fpath_str}", file=sys.stderr)
        return 0

    root = Path(root).expanduser().resolve()
    if files:
        file_list = [Path(f).expanduser().resolve() for f in files]
    else:
        file_list = _collect(root, extensions)

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    print(f"Processing {len(file_list)} {language} files (dry_run={dry_run})", file=sys.stderr)

    all_symbols: list[dict[str, Any]] = []
    file_iter = _tqdm(file_list, desc="parse", unit="file", dynamic_ncols=True) if _tqdm else file_list
    try:
        for fpath in file_iter:
            syms = _chunk(fpath, root, language, extensions)
            for s in syms:
                s["project"] = project
            all_symbols.extend(syms)
    except KeyboardInterrupt:
        if _tqdm and hasattr(file_iter, "close"):
            file_iter.close()
        print(f"\nInterrupted during parse — {len(all_symbols)} symbols collected so far")
        return 0
    if _tqdm and hasattr(file_iter, "close"):
        file_iter.close()
    print(f"Found {len(all_symbols)} symbols", file=sys.stderr)

    if dry_run:
        for s in all_symbols[:20]:
            print(f"  [{s['kind']:9}] {s['module']}.{s['name']}  node_type={s.get('node_type') or '?'}"
                  + (f"  vis={s['visibility']}" if s.get("visibility") else "")
                  + (f"  parent={s['parent_impl']}" if s.get("parent_impl") else "")
                  + f"  line={s['line']}")
            if s.get("docstring_summary"):
                print(f"             {s['docstring_summary'][:80]}")
        if len(all_symbols) > 20:
            print(f"  ... and {len(all_symbols) - 20} more")
        return 0

    COMMIT_EVERY = 50
    new_count = updated_count = skipped_count = 0
    sym_iter = _tqdm(all_symbols, desc="insert", unit="sym", dynamic_ncols=True) if _tqdm else all_symbols
    try:
        for i, s in enumerate(sym_iter, 1):
            result = kb.add_symbol(
                name=s["name"], kind=s["kind"], module=s["module"], signature=s["signature"],
                file=s["file"], line=s["line"], status="public",
                docstring_summary=s["docstring_summary"], lean_citations=[], kb_refs=[],
                project=s["project"], parent_impl=s.get("parent_impl"),
                visibility=s.get("visibility"), is_signature_only=bool(s.get("is_signature_only")),
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
        return 0
    if _tqdm and hasattr(sym_iter, "close"):
        sym_iter.close()

    # PRUNE stale rows when ingesting specific files.
    if files and not dry_run:
        file_to_live: dict[str, set[tuple[str, str]]] = {}
        for s in all_symbols:
            file_to_live.setdefault(s["file"], set()).add((s["name"], s["module"]))
        pruned_total = 0
        for fpath in [str(Path(f).expanduser().resolve()) for f in files]:
            n = kb.prune_symbols_for_file(fpath, file_to_live.get(fpath, set()))
            if n:
                print(f"  Pruned {n} stale symbol(s) from {fpath}", file=sys.stderr)
            pruned_total += n
        if pruned_total:
            print(f"Total pruned: {pruned_total}", file=sys.stderr)

    # also_in_modules: names appearing in >1 module.
    name_to_entries: dict[str, list[dict[str, Any]]] = {}
    for s in all_symbols:
        name_to_entries.setdefault(s["name"], []).append(s)
    multi_module_count = 0
    for name, entries in name_to_entries.items():
        if len(entries) <= 1:
            continue
        also = [{"module": e["module"], "file": e["file"], "line": e["line"]} for e in entries]
        kb.conn.execute("UPDATE symbols SET also_in_modules = ? WHERE name = ?", (json.dumps(also), name))
        multi_module_count += 1

    kb.conn.commit()
    print(f"New: {new_count}  Updated: {updated_count}  Skipped(unchanged): {skipped_count}  "
          f"Multi-module: {multi_module_count}")
    return 0

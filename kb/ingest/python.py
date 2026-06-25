#!/usr/bin/env python3
"""
Ingest Python symbols from cl44/clifford_common into the KB symbols table.

Uses Python's ast module (no grep/text-search). Extracts functions and classes
at module level, their signatures, docstrings, LRU cache decoration, Lean citations,
and KB references.
"""

import ast
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow standalone execution: ensure the package root is on sys.path.
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from kb import KnowledgeBase, DEFAULT_DB_PATH


# Retired symbol names → canonical replacement.
# These are inserted into symbols with status='retired' so check_symbols.py
# can block attempts to write them in cl44/ source.
# Sources: cl44/canonical_operators.py comments, CLAUDE.md, tip's T3 test fixture.
RETIRED_SYMBOLS: dict[str, str] = {
    # Krein / chirality operators
    "gamma9_48": "K_48",
    "gamma_9": "K_48",
    "chirality9_w": "K_w",
    "gamma_5_48": "C_48",
    "gamma5_48": "C_48",
    # Old generating-functional names
    "partition_function_old": "Z_species",
    "effective_action": "S_eff",
    # Color/SU(3) sector — archived; no direct replacement
    "sl3_bivector_48": "ARCHIVED",
    "physical_color_su3_48": "ARCHIVED",
    "color_generators_48": "sm_color_su3_48",
    # Triality intertwiner renames
    "triality_rotation_16": "Tw_w",
    "_signed_triality_intertwiner_mp": "Tw_fock",
    # Bare names retired in favour of explicit w-sector suffix
    "bare_Q_EM": "Q_EM_w",
    "bare_mhf": "mhf_w",
    "m_vb_16": "m_vb_w",
    "bare_m_vb": "m_vb_w",
    "bare_Tw": "Tw_w",
    # gamma9 on the 16-element Fock space
    "gamma9_16": "gamma9_cl44",
    # Grade / number operators
    "N_grade": "N_grade_w",
    "N_total_w": "N_grade_w",
    # tau2 mass half-field
    "tau2_m_hf": "tau2_M_w",
    "bare_tau2_M": "tau2_M_w",
    # Shift/mass matrices renamed this week
    "shift_matrix_48": "M_g5_odd_48",
    "scalar_shift_48": "M_g5_even_48",
    "shift_matrix_full_48": "M_full_48",
    # Sector charpoly scripts — archived; home is cl44.charpoly
    "sector_charpolys_bare_g9": "ARCHIVED",
    # Mixing angle modules renamed
    "pmns": "q_minus1_mixing",
    "ckm": "q_plus2_3_mixing",
}

# Modules considered canonical (exported + blessed)
CANONICAL_FILES = {
    "canonical_operators.py",
    "mass_spectrum.py",
    "charges.py",
    "generating_functional.py",
    "tree_yukawa.py",
    "spectral_zeta.py",
    "centralizer.py",
}

# Default corpus roots under secular-constraints
DEFAULT_SUBDIRS = [
    "cl44",
    "clifford_common",
    "cl11",
    "cl22",
    "scripts",
]


def _reconstruct_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a function signature string from AST node."""
    args = node.args
    parts = []

    # positional-only
    posonlyargs = getattr(args, "posonlyargs", [])
    defaults_offset = len(args.args) - len(args.defaults)

    def _default_str(idx: int, defaults: list, offset: int) -> str:
        d_idx = idx - offset
        if d_idx >= 0 and d_idx < len(defaults):
            try:
                return "=" + ast.unparse(defaults[d_idx])
            except Exception:
                return "=..."
        return ""

    for i, arg in enumerate(posonlyargs):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}{_default_str(i, args.defaults, defaults_offset)}")
    if posonlyargs:
        parts.append("/")

    for i, arg in enumerate(args.args):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        parts.append(f"{arg.arg}{ann}{_default_str(i, args.defaults, defaults_offset)}")

    if args.vararg:
        ann = f": {ast.unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")
    elif args.kwonlyargs:
        parts.append("*")

    kw_defaults = args.kw_defaults
    for i, arg in enumerate(args.kwonlyargs):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        d = ""
        if i < len(kw_defaults) and kw_defaults[i] is not None:
            try:
                d = "=" + ast.unparse(kw_defaults[i])
            except Exception:
                d = "=..."
        parts.append(f"{arg.arg}{ann}{d}")

    if args.kwarg:
        ann = f": {ast.unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    ret = ""
    if node.returns:
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            ret = ""

    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(parts)}){ret}:"


def _class_signature(node: ast.ClassDef) -> str:
    """Reconstruct a class signature."""
    bases = []
    for b in node.bases:
        try:
            bases.append(ast.unparse(b))
        except Exception:
            pass
    if bases:
        return f"class {node.name}({', '.join(bases)}):"
    return f"class {node.name}:"


def _has_lru_cache(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Check if a function is decorated with lru_cache or cache."""
    for dec in node.decorator_list:
        try:
            s = ast.unparse(dec)
            if "lru_cache" in s or "cache" in s:
                return True
        except Exception:
            pass
    return False


def _extract_lean_citations(text: str) -> list[str]:
    """Extract Lean: citations from docstrings or # Lean: inline comments."""
    # Matches both:  "# Lean: Foo.lean::bar"  (inline comment)
    #            and "Lean: Foo.lean::bar"      (inside a docstring)
    return re.findall(r'(?:#\s*)?Lean:\s*([\w./]+(?:::[\w.]+)?)', text)


def _extract_kb_refs(text: str) -> list[str]:
    """Extract kb-YYYYMMDD-HHMMSS-hash references from text."""
    return re.findall(r'kb-\d{8}-\d{6}-[a-f0-9]+', text)


def _docstring_first_sentence(doc: str | None) -> str | None:
    if not doc:
        return None
    # Take first non-empty sentence
    first = doc.strip().split("\n")[0].strip()
    if not first:
        lines = [l.strip() for l in doc.strip().split("\n") if l.strip()]
        first = lines[0] if lines else ""
    if len(first) > 300:
        first = first[:297] + "..."
    return first or None


def _determine_status(
    name: str,
    file_path: Path,
    all_exports: set[str],
) -> str:
    """Determine canonical/public/scratch/archived status."""
    rel = str(file_path)
    if "/archive/" in rel:
        return "archived"
    if "/tmp/" in rel or "/scripts/" in rel or "/notebooks/" in rel:
        return "scratch"
    if file_path.name in CANONICAL_FILES and name in all_exports:
        return "canonical"
    if name in all_exports:
        return "public"
    return "public"


def _determine_frame_hint(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    docstring: str | None,
) -> str | None:
    """Determine frame_hint from return annotation or docstring."""
    ret_str = ""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
        try:
            ret_str = ast.unparse(node.returns)
        except Exception:
            ret_str = ""

    if "WeightMatrix" in ret_str:
        return "weight"
    if "FockMatrix" in ret_str:
        return "fock"
    if "CliffordElement" in ret_str:
        return "algebraic"

    if docstring:
        doc_lower = docstring.lower()
        if any(k in doc_lower for k in ("weight basis", "cartan_weight", "weight frame")):
            return "weight"
        if "fock" in doc_lower:
            return "fock"
        if any(k in doc_lower for k in ("clifford", "algebraic", "algebra")):
            return "algebraic"
    return None


def _extract_all_exports(tree: ast.Module) -> set[str]:
    """Extract __all__ list from module AST."""
    exports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exports.add(elt.value)
    return exports


def _path_to_module(file_path: Path, root: Path) -> str:
    """Convert file path to dotted module name relative to root."""
    try:
        rel = file_path.relative_to(root)
        parts = list(rel.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)
    except ValueError:
        return file_path.stem


def parse_python_file(
    file_path: Path,
    root: Path,
) -> list[dict[str, Any]]:
    """Parse a Python file and extract top-level functions and classes."""
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  Warning: cannot read {file_path}: {e}", file=sys.stderr)
        return []

    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  Warning: syntax error in {file_path}: {e}", file=sys.stderr)
        return []

    module = _path_to_module(file_path, root)
    all_exports = _extract_all_exports(tree)

    # Module-level Lean citations (from module docstring)
    module_doc = ast.get_docstring(tree) or ""
    module_lean = _extract_lean_citations(module_doc)

    symbols = []
    for node in tree.body:
        # Module-level annotated constants: T_C: float = ..., PHI_TODAY: float = ...
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            try:
                ann = ast.unparse(node.annotation)
            except Exception:
                ann = "?"
            val_str = ""
            if node.value is not None:
                try:
                    val_str = " = " + ast.unparse(node.value)
                except Exception:
                    pass
            sig = f"{name}: {ann}{val_str}"
            status = _determine_status(name, file_path, all_exports)
            symbols.append({
                "name": name,
                "kind": "constant",
                "module": module,
                "signature": sig,
                "status": status,
                "is_lru_cached": False,
                "frame_hint": None,
                "docstring_summary": None,
                "lean_citations": module_lean,
                "kb_refs": [],
                "file": str(file_path),
                "line": node.lineno,
            })
            continue

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        name = node.name
        kind = "class" if isinstance(node, ast.ClassDef) else "function"

        if kind == "function":
            sig = _reconstruct_signature(node)
        else:
            sig = _class_signature(node)

        doc = ast.get_docstring(node)
        doc_summary = _docstring_first_sentence(doc)
        lean_cites = _extract_lean_citations(doc or "")
        if not lean_cites and module_lean:
            lean_cites = module_lean
        kb_refs = _extract_kb_refs(doc or "")
        is_lru = _has_lru_cache(node)
        status = _determine_status(name, file_path, all_exports)
        frame_hint = _determine_frame_hint(node, doc)

        symbols.append({
            "name": name,
            "kind": kind,
            "module": module,
            "signature": sig,
            "status": status,
            "is_lru_cached": is_lru,
            "frame_hint": frame_hint,
            "docstring_summary": doc_summary,
            "lean_citations": lean_cites,
            "kb_refs": kb_refs,
            "file": str(file_path),
            "line": node.lineno,
        })

    return symbols


def populate_retired_symbols(kb: KnowledgeBase, project: str, dry_run: bool = False) -> int:
    """Insert RETIRED_SYMBOLS entries into symbols so check_symbols.py can block them."""
    now = datetime.now().isoformat()
    inserted = 0
    for name, redirect_to in RETIRED_SYMBOLS.items():
        existing = kb.conn.execute(
            "SELECT id, status FROM symbols WHERE name=? AND project=?",
            (name, project),
        ).fetchone()
        if existing and existing[1] == 'retired':
            continue  # already present
        sym_id = f"pysym-retired-{name}"
        if not dry_run:
            kb.conn.execute("""
                INSERT OR REPLACE INTO symbols
                  (id, name, kind, module, signature, status, redirect_to, file, line, project, created_at, updated_at)
                VALUES (?, ?, 'function', 'cl44.__retired__', ?, 'retired', ?, '', 0, ?, ?, ?)
            """, (sym_id, name, f"def {name}(...):", redirect_to, project, now, now))
        inserted += 1
    if not dry_run:
        kb.conn.commit()
    return inserted


def populate_notations_from_constants(kb: KnowledgeBase, dry_run: bool = False) -> int:
    """Insert GREEK_MEANINGS entries into the notations table."""
    from kb.constants import GREEK_MEANINGS

    now = datetime.now().isoformat()
    inserted = 0

    for symbol, meaning in GREEK_MEANINGS.items():
        existing = kb.conn.execute(
            "SELECT id FROM notations WHERE current_symbol = ? AND project = ?",
            (symbol, "algebraic-genesis"),
        ).fetchone()
        if existing:
            continue
        nid = f"not-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        if not dry_run:
            kb.conn.execute("""
                INSERT OR IGNORE INTO notations (id, current_symbol, meaning, project, domain, created_at, updated_at)
                VALUES (?, ?, ?, 'algebraic-genesis', 'physics', ?, ?)
            """, (nid, symbol, meaning, now, now))
        inserted += 1

    # Also insert K and σ* overload entries from secular-constraints CLAUDE.md
    overloads = [
        ("K", "K_48 (Krein, (-1)^grade) | number field K = Q(α,β) = cl44.qfield.QAB | grams K_bare = MᵀM / K_dressed = MᵀM + 2σ*·I | htli saddle gram K(x*) — FOUR distinct objects; never write bare 'K' without binding it"),
        ("σ*", "sigma_saddle (gap-equation solution) | THREE distinct roots: sigma_vev = 17√3/6 ≈ 4.907 (quartic cold limit); sigma_star_charpoly ≈ 0.93952 (charpoly-product saddle, SigmaStarMinpoly.lean); sigma_star ≈ 1.02008 (mode-summed coth gap, transcendental). 'The σ* minpoly' applies ONLY to the charpoly-saddle root"),
    ]
    for symbol, meaning in overloads:
        existing = kb.conn.execute(
            "SELECT id FROM notations WHERE current_symbol = ? AND project = ?",
            (symbol, "algebraic-genesis"),
        ).fetchone()
        if existing:
            continue
        nid = f"not-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        if not dry_run:
            kb.conn.execute("""
                INSERT OR IGNORE INTO notations (id, current_symbol, meaning, project, domain, created_at, updated_at)
                VALUES (?, ?, ?, 'algebraic-genesis', 'physics', ?, ?)
            """, (nid, symbol, meaning, now, now))
        inserted += 1

    if not dry_run:
        kb.conn.commit()

    return inserted


def run(
    root: Path | None = None,
    files: list[str] | None = None,
    deleted: list[str] | None = None,
    project: str = "algebraic-genesis",
    dry_run: bool = False,
    db_path: Path | None = None,
    with_notations: bool = False,
) -> int:
    """Ingest Python symbols in-process.  Returns 0 on success, 1 on fatal error.

    Generic by default: root=cwd, no physics extras. The secular-constraints corpus
    layout (DEFAULT_SUBDIRS) enables physics_mode (retired-symbol injection); notations
    are opt-in via with_notations.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if root is None:
        root = Path.cwd()

    kb = KnowledgeBase(db_path=db_path)

    # Handle --deleted: remove all rows for explicitly-deleted files and exit.
    if deleted:
        for fpath in deleted:
            fpath_str = str(Path(fpath).expanduser().resolve())
            n = kb._symbols.delete_symbols_for_file(fpath_str)
            print(f"Deleted {n} rows for removed file: {fpath_str}", file=sys.stderr)
        return 0

    root = Path(root).expanduser().resolve()

    # Collect files to process
    physics_mode = False
    if files:
        file_list = [Path(f).expanduser().resolve() for f in files]
    else:
        file_list = []
        for subdir in DEFAULT_SUBDIRS:
            subpath = root / subdir
            if subpath.exists():
                for py in subpath.rglob("*.py"):
                    if "__pycache__" in py.parts:
                        continue
                    file_list.append(py)
        physics_mode = bool(file_list)  # secular-constraints corpus layout was found
        if not file_list:
            # Generic project: walk the root directly (no physics corpus subdirs present).
            _SKIP = {"__pycache__", ".venv", "venv", ".git", "node_modules", "build", "dist", ".eggs"}
            for py in root.rglob("*.py"):
                if _SKIP & set(py.parts):
                    continue
                file_list.append(py)

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    print(f"Processing {len(file_list)} Python files (dry_run={dry_run})", file=sys.stderr)

    # Collect all symbols across all files
    all_symbols: list[dict[str, Any]] = []
    file_iter = _tqdm(file_list, desc="parse", unit="file", dynamic_ncols=True) if _tqdm else file_list
    try:
        for fpath in file_iter:
            syms = parse_python_file(fpath, root)
            for s in syms:
                s["project"] = project
            all_symbols.extend(syms)
    except KeyboardInterrupt:
        if _tqdm and hasattr(file_iter, 'close'):
            file_iter.close()
        print(f"\nInterrupted during parse — {len(all_symbols)} symbols collected so far")
        return 0

    if _tqdm and hasattr(file_iter, 'close'):
        file_iter.close()
    print(f"Found {len(all_symbols)} symbols", file=sys.stderr)

    if dry_run:
        for s in all_symbols[:20]:
            print(f"  [{s['status']:9}] {s['module']}.{s['name']} ({s['kind']}) line={s['line']}")
            if s["docstring_summary"]:
                print(f"             {s['docstring_summary'][:80]}")
        if len(all_symbols) > 20:
            print(f"  ... and {len(all_symbols) - 20} more")
        return 0

    # Insert symbols — commit every COMMIT_EVERY rows so the write lock is
    # released frequently enough for concurrent kb add / MCP operations.
    COMMIT_EVERY = 50
    new_count = 0
    updated_count = 0
    skipped_count = 0
    sym_iter = _tqdm(all_symbols, desc="insert", unit="sym", dynamic_ncols=True) if _tqdm else all_symbols
    try:
        for i, s in enumerate(sym_iter, 1):
            result = kb._symbols.add_symbol(
                name=s["name"],
                kind=s["kind"],
                module=s["module"],
                signature=s["signature"],
                file=s["file"],
                line=s["line"],
                status=s["status"],
                is_lru_cached=s["is_lru_cached"],
                frame_hint=s["frame_hint"],
                docstring_summary=s["docstring_summary"],
                lean_citations=s["lean_citations"],
                kb_refs=s["kb_refs"],
                project=s["project"],
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
        if _tqdm and hasattr(sym_iter, 'close'):
            sym_iter.close()
        kb.conn.commit()
        print(f"\nInterrupted — New: {new_count}  Updated: {updated_count}  Skipped: {skipped_count}")
        return 0
    if _tqdm and hasattr(sym_iter, 'close'):
        sym_iter.close()

    # PRUNE: when ingesting with --files, delete stale rows (symbols that vanished from each file).
    if files and not dry_run:
        pruned_total = 0
        file_to_live: dict[str, set[tuple[str, str]]] = {}
        for s in all_symbols:
            fpath_str = s["file"]
            if fpath_str not in file_to_live:
                file_to_live[fpath_str] = set()
            file_to_live[fpath_str].add((s["name"], s["module"]))
        for fpath in [str(Path(f).expanduser().resolve()) for f in files]:
            live = file_to_live.get(fpath, set())
            n = kb._symbols.prune_symbols_for_file(fpath, live)
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
            "UPDATE symbols SET also_in_modules = ? WHERE name = ?",
            (also_json, name),
        )
        multi_module_count += 1

    kb.conn.commit()
    print(f"New: {new_count}  Updated: {updated_count}  Skipped(unchanged): {skipped_count}  Multi-module: {multi_module_count}")

    # Populate notations (physics-specific; opt-in)
    if with_notations:
        n = populate_notations_from_constants(kb, dry_run=False)
        print(f"Notations inserted/skipped: {n}")

    # Populate retired symbols (physics-specific RETIRED_SYMBOLS; only for the
    # secular-constraints corpus — never inject these into a generic project's index)
    if physics_mode:
        r = populate_retired_symbols(kb, project=project, dry_run=dry_run)
        print(f"Retired symbols inserted: {r}")

    return 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest Python symbols from cl44/clifford_common into the KB"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "Physics" / "secular-constraints",
        help="Root directory of secular-constraints (default: ~/Physics/secular-constraints)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Incremental mode: process only these files",
    )
    parser.add_argument(
        "--project",
        default="algebraic-genesis",
        help="KB project name (default: algebraic-genesis)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print without writing to DB",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="KB database path",
    )
    parser.add_argument(
        "--with-notations",
        action="store_true",
        help="Also populate the physics notations table (secular-constraints only; off by default)",
    )
    parser.add_argument(
        "--deleted",
        nargs="+",
        metavar="FILE",
        help="Remove all symbols rows for these deleted/renamed files",
    )
    args = parser.parse_args()
    rc = run(
        root=args.root,
        files=args.files,
        deleted=args.deleted,
        project=args.project,
        dry_run=args.dry_run,
        db_path=args.db,
        with_notations=args.with_notations,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

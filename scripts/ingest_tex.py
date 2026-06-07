#!/usr/bin/env python3
"""
Ingest TeX annotation comments into the KB tex_annotations table.

Scans .tex files for structured annotation comment blocks:
  % Python: cl44/foo.py::bar
  % Lean: File.lean::Name
  % Epic: project-XXXX
  % kb-YYYYMMDD-HHMMSS-hash

Does NOT parse LaTeX theorem bodies -- only the structured annotation comments
carry indexable information.
"""

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from kb import KnowledgeBase, DEFAULT_DB_PATH


# Regex: annotation line starts with % Python:, % Lean:, % Epic:, or % kb-YYYYMMDD
_ANN_LINE = re.compile(
    r'^\s*%\s*(Python|Lean|Epic|kb-\d{8}[^:]*)',
    re.IGNORECASE,
)

# Key-value annotation patterns
_PYTHON_RE = re.compile(r'^\s*%\s*[Pp]ython:\s*(.+)', re.IGNORECASE)
_LEAN_RE = re.compile(r'^\s*%\s*[Ll]ean:\s*(.+)', re.IGNORECASE)
_EPIC_RE = re.compile(r'^\s*%\s*[Ee]pic:\s*(.+)', re.IGNORECASE)
_KB_RE = re.compile(r'^\s*%\s*(kb-\d{8}-\S+)')

# A valid ref token: path/to/module.ext optionally followed by ::Name
# No spaces, no parentheses. Examples:
#   cl44/foo.py::bar_fn   File.lean::TheoremName   project-1234
_REF_RE = re.compile(r'^[\w./:-]+(?:::[\w.]+)?')

# Structural TeX patterns
_SECTION_RE = re.compile(r'\\(?:sub)*section\*?\{([^}]+)\}')
_LABEL_RE = re.compile(r'\\label\{([^}]+)\}')

# Any comment line (to stop context capture)
_COMMENT_LINE = re.compile(r'^\s*%')


def _is_annotation_line(line: str) -> bool:
    return bool(_ANN_LINE.match(line))


def _split_refs(raw: str) -> list[str]:
    """Split a comma-separated ref list, ignoring commas inside parens/braces.

    Each token is then stripped of any trailing free-text commentary (anything
    after the first space following the ref pattern). This handles annotations
    like:
        cl44/foo.py::bar (some note with {0,3,12}), cl44/baz.py::qux
    which should yield ['cl44/foo.py::bar', 'cl44/baz.py::qux'].
    """
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in raw:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current).strip())

    result: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        # Strip everything after the first space (free-text commentary)
        m = _REF_RE.match(tok)
        if m:
            result.append(m.group(0))
        # If the token doesn't start with a ref pattern at all, skip it entirely
        # (it was a fragment produced by a comma inside a parenthetical note)
    return result


def _parse_annotation_line(line: str) -> tuple[str, list[str]]:
    """Return (kind, values) for a single annotation line.

    kind is one of: 'python', 'lean', 'epic', 'kb'
    values are ref tokens — path/module.ext::name — with free-text commentary stripped.
    """
    m = _PYTHON_RE.match(line)
    if m:
        return "python", _split_refs(m.group(1))
    m = _LEAN_RE.match(line)
    if m:
        return "lean", _split_refs(m.group(1))
    m = _EPIC_RE.match(line)
    if m:
        return "epic", _split_refs(m.group(1))
    m = _KB_RE.match(line)
    if m:
        return "kb", [m.group(1).strip()]
    return "unknown", []


def scan_tex_file(path: Path) -> list[dict]:
    """Scan a single .tex file and return annotation block dicts."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  Warning: cannot read {path}: {e}", file=sys.stderr)
        return []

    lines = text.splitlines()
    annotations = []
    section_title: str | None = None
    section_label: str | None = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Track structural context
        sm = _SECTION_RE.search(line)
        if sm:
            section_title = sm.group(1).strip()

        lm = _LABEL_RE.search(line)
        if lm:
            section_label = lm.group(1).strip()

        # Detect start of annotation block
        if _is_annotation_line(line):
            block_start_line = i + 1  # 1-based
            python_refs: list[str] = []
            lean_refs: list[str] = []
            epic_refs: list[str] = []
            kb_refs: list[str] = []

            # Consume all consecutive annotation lines
            while i < len(lines) and _is_annotation_line(lines[i]):
                kind, values = _parse_annotation_line(lines[i])
                if kind == "python":
                    python_refs.extend(values)
                elif kind == "lean":
                    lean_refs.extend(values)
                elif kind == "epic":
                    epic_refs.extend(values)
                elif kind == "kb":
                    kb_refs.extend(values)
                i += 1

            # Capture next 2 non-empty, non-comment TeX body lines as context
            context_lines: list[str] = []
            j = i
            while j < len(lines) and len(context_lines) < 2:
                cl = lines[j].strip()
                if cl and not _COMMENT_LINE.match(lines[j]):
                    context_lines.append(cl)
                j += 1
            context = " ".join(context_lines) if context_lines else None

            # Only emit if we got at least one ref
            if python_refs or lean_refs or epic_refs or kb_refs:
                annotations.append({
                    "file": str(path),
                    "line": block_start_line,
                    "section_title": section_title,
                    "section_label": section_label,
                    "python_refs": python_refs,
                    "lean_refs": lean_refs,
                    "epic_refs": epic_refs,
                    "kb_refs": kb_refs,
                    "context": context,
                })
            continue  # i already advanced past annotation block

        i += 1

    return annotations


def check_python_staleness(kb: KnowledgeBase, annotations: list[dict]) -> list[str]:
    """For each python_ref, check if a python_symbols row exists.

    Returns list of warning strings for stale cross-references.
    """
    warnings = []
    for ann in annotations:
        for ref in ann.get("python_refs", []):
            # ref format: "cl44/module.py::function" or "cl44/module.py"
            if "::" in ref:
                file_part, name = ref.split("::", 1)
            else:
                file_part, name = ref, None

            # Normalize: strip leading path component to just the relative path
            # python_symbols.file stores full absolute paths; we do a suffix match
            row = None
            if name:
                row = kb.conn.execute(
                    "SELECT id FROM python_symbols WHERE name = ? AND file LIKE ?",
                    (name.strip(), f"%{file_part.strip()}"),
                ).fetchone()
            else:
                row = kb.conn.execute(
                    "SELECT id FROM python_symbols WHERE file LIKE ?",
                    (f"%{file_part.strip()}",),
                ).fetchone()

            if not row:
                warnings.append(
                    f"STALE python_ref '{ref}' in {ann['file']}:{ann['line']} "
                    f"-- no python_symbols row found (run: kb ingest python)"
                )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest TeX annotation comments into KB tex_annotations table"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "Physics" / "claude",
        help="Root directory to scan for .tex files (default: ~/Physics/claude)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Specific files to process (overrides --root glob)",
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
    args = parser.parse_args()

    kb = KnowledgeBase(db_path=args.db)
    root = args.root.expanduser().resolve()

    # Collect files
    if args.files:
        files = [Path(f).expanduser().resolve() for f in args.files]
    else:
        files = list(root.glob("*.tex")) + list((root / "sections").glob("*.tex"))

    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    print(f"Scanning {len(files)} TeX file(s) (dry_run={args.dry_run})", file=sys.stderr)

    # Scan all files
    all_annotations: list[dict] = []
    file_iter = _tqdm(files, desc="scan", unit="file", dynamic_ncols=True) if _tqdm else files
    try:
        for fpath in file_iter:
            anns = scan_tex_file(fpath)
            all_annotations.extend(anns)
    except KeyboardInterrupt:
        if _tqdm and hasattr(file_iter, 'close'):
            file_iter.close()
        print(f"\nInterrupted during scan — {len(all_annotations)} annotations collected")
        return
    if _tqdm and hasattr(file_iter, 'close'):
        file_iter.close()

    print(f"Found {len(all_annotations)} annotation block(s)", file=sys.stderr)

    if args.dry_run:
        for ann in all_annotations:
            print(f"  {ann['file']}:{ann['line']}")
            if ann.get("section_title"):
                print(f"    section: {ann['section_title']}")
            if ann.get("python_refs"):
                print(f"    python:  {ann['python_refs']}")
            if ann.get("lean_refs"):
                print(f"    lean:    {ann['lean_refs']}")
            if ann.get("epic_refs"):
                print(f"    epic:    {ann['epic_refs']}")
            if ann.get("kb_refs"):
                print(f"    kb:      {ann['kb_refs']}")
            if ann.get("context"):
                print(f"    context: {ann['context'][:80]}")
        return

    # Ingest
    COMMIT_EVERY = 25
    new_count = 0
    updated_count = 0
    ann_iter = _tqdm(all_annotations, desc="ingest", unit="ann", dynamic_ncols=True) if _tqdm else all_annotations
    try:
        for i, ann in enumerate(ann_iter, 1):
            result = kb.add_tex_annotation(
                file=ann["file"],
                line=ann["line"],
                section_label=ann.get("section_label"),
                section_title=ann.get("section_title"),
                python_refs=ann.get("python_refs"),
                lean_refs=ann.get("lean_refs"),
                epic_refs=ann.get("epic_refs"),
                kb_refs=ann.get("kb_refs"),
                context=ann.get("context"),
                project=args.project,
            )
            if result["is_new"]:
                new_count += 1
            else:
                updated_count += 1
            if i % COMMIT_EVERY == 0:
                kb.conn.commit()
    except KeyboardInterrupt:
        if _tqdm and hasattr(ann_iter, 'close'):
            ann_iter.close()
        kb.conn.commit()
        print(f"\nInterrupted — Inserted: {new_count}  Updated: {updated_count}")
        return
    if _tqdm and hasattr(ann_iter, 'close'):
        ann_iter.close()

    print(f"Inserted: {new_count}  Updated: {updated_count}")

    # Post-ingest staleness check
    warnings = check_python_staleness(kb, all_annotations)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()

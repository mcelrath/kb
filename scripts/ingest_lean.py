#!/usr/bin/env python3
"""
Lean Theorem Ingestion Script

Walks ~/Physics/claude/proofs/ and ingests top-level theorem/lemma declarations
into the KB theorem index. Only processes files with a MATHEMATICAL OVERVIEW
header comment (heuristic for meaningful theorems).

For each theorem:
1. Parses name, statement, full declaration, file, line
2. Derives module from directory structure
3. Optionally calls local LLM to generate statement_pure
4. Parses -- source: FILE.tex line N comments for tex_source cross-refs
5. Stores via kb.theorem_add()

Usage:
    python scripts/ingest_lean.py [--proof-root PATH] [--project NAME]
        [--no-llm] [--dry-run] [--module-filter PREFIX] [--limit N]
"""

import argparse
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from kb import KnowledgeBase


MATH_OVERVIEW_RE = re.compile(r"MATHEMATICAL OVERVIEW", re.IGNORECASE)

DECL_RE = re.compile(
    r"^(?P<kw>theorem|lemma|def|noncomputable def)\s+(?P<name>\S+)",
    re.MULTILINE,
)

STATEMENT_RE = re.compile(
    r"^(?:theorem|lemma)\s+\S+\s*(?:\([^)]*\)|\{[^}]*\}|\[[^\]]*\]|\s)*:\s*(?P<stmt>.*?)(?:\s*:=|\s*where\b)",
    re.DOTALL,
)

TEX_SOURCE_RE = re.compile(r"--\s*source:\s*(.+)", re.IGNORECASE)

PURE_MATH_PROMPT = """\
Restate the following Lean theorem in pure mathematical language. \
Use no domain-specific physics framing. Use standard mathematical notation. \
Keep it under 30 tokens. Return only the restatement, no preamble.

Lean statement:
{statement}

Pure math restatement:"""


def has_math_overview(text: str) -> bool:
    return bool(MATH_OVERVIEW_RE.search(text))


def extract_declarations(text: str, kind: tuple[str, ...] = ("theorem", "lemma")) -> list[dict]:
    """Extract theorem/lemma declarations from Lean source."""
    results = []
    lines = text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(theorem|lemma)\s+(\S+)", line.strip())
        if m and m.group(1) in kind:
            kw = m.group(1)
            name = m.group(2).rstrip("(:{[")
            decl_start = i

            decl_lines = []
            j = i
            depth = 0
            while j < min(i + 40, len(lines)):
                dl = lines[j]
                decl_lines.append(dl)
                depth += dl.count("(") + dl.count("{") + dl.count("[")
                depth -= dl.count(")") + dl.count("}") + dl.count("]")
                if ":=" in dl or "where" in dl.split("--")[0]:
                    break
                if j > i and depth <= 0 and dl.strip() and not dl.strip().startswith("--"):
                    break
                j += 1

            declaration = "\n".join(decl_lines)

            stmt_match = re.search(
                r":\s*(.*?)(?::=|where\b)",
                declaration.replace("\n", " "),
                re.DOTALL,
            )
            statement = stmt_match.group(1).strip() if stmt_match else name

            tex_source = None
            for tl in lines[max(0, decl_start - 5):decl_start + 3]:
                tsm = TEX_SOURCE_RE.search(tl)
                if tsm:
                    tex_source = tsm.group(1).strip()
                    break

            results.append({
                "name": name,
                "lean_name": name,
                "statement": statement,
                "declaration": declaration,
                "line": decl_start + 1,
                "tex_source": tex_source,
            })
        i += 1

    return results


def derive_module(file_path: Path, proof_root: Path) -> str:
    """Derive Lean module name from file path relative to proof root."""
    try:
        rel = file_path.relative_to(proof_root)
    except ValueError:
        return file_path.stem
    parts = list(rel.parts)
    parts[-1] = parts[-1].replace(".lean", "")
    return ".".join(parts)


def generate_statement_pure(llm_client, statement: str) -> str | None:
    """Call LLM to generate pure-math restatement."""
    prompt = PURE_MATH_PROMPT.format(statement=statement[:800])
    try:
        result = llm_client.complete(
            prompt,
            max_tokens=80,
            temperature=0.2,
            use_chat=False,
        )
        if result:
            return result.strip().strip('"').strip("'")
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
    return None


def ingest_file(
    lean_file: Path,
    proof_root: Path,
    kb: KnowledgeBase,
    project: str,
    use_llm: bool,
    dry_run: bool,
) -> dict:
    text = lean_file.read_text(encoding="utf-8", errors="replace")

    if not has_math_overview(text):
        return {"skipped": True, "reason": "no MATHEMATICAL OVERVIEW"}

    rel_path = str(lean_file.relative_to(proof_root.parent))
    module = derive_module(lean_file, proof_root)
    decls = extract_declarations(text)

    if not decls:
        return {"skipped": True, "reason": "no theorems/lemmas"}

    added = 0
    skipped = 0
    for d in decls:
        statement_pure = None
        if use_llm and not dry_run:
            statement_pure = generate_statement_pure(kb._llm, d["statement"])

        if dry_run:
            print(f"  [DRY] {d['name']}: {d['statement'][:80]}")
            skipped += 1
            continue

        result = kb.theorem_add(
            lean_name=f"{module}.{d['name']}",
            name=d["name"],
            statement=d["statement"],
            declaration=d["declaration"],
            file=rel_path,
            statement_pure=statement_pure,
            module=module,
            line=d["line"],
            tex_source=d.get("tex_source"),
            project=project,
        )
        if result["is_new"]:
            added += 1
        else:
            skipped += 1

    return {"added": added, "skipped": skipped, "total": len(decls)}


def main():
    parser = argparse.ArgumentParser(description="Ingest Lean theorems into KB")
    parser.add_argument(
        "--proof-root",
        default=str(Path.home() / "Physics/claude/proofs"),
        help="Root directory of Lean proofs",
    )
    parser.add_argument("--project", default="exterior_algebra")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM rewrite of statement_pure")
    parser.add_argument("--dry-run", action="store_true", help="Parse without storing")
    parser.add_argument("--module-filter", default=None, help="Only process files matching this prefix")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N files")
    args = parser.parse_args()

    proof_root = Path(args.proof_root).expanduser()
    if not proof_root.exists():
        print(f"Proof root not found: {proof_root}", file=sys.stderr)
        sys.exit(1)

    kb = KnowledgeBase()

    lean_files = sorted(
        f for f in proof_root.rglob("*.lean")
        if ".lake" not in f.parts and ".elan" not in f.parts
    )
    if args.module_filter:
        lean_files = [f for f in lean_files if args.module_filter in str(f)]
    if args.limit:
        lean_files = lean_files[: args.limit]

    total_added = 0
    total_skipped_files = 0
    total_theorems = 0
    total_added_theorems = 0

    for idx, lean_file in enumerate(lean_files):
        rel = lean_file.relative_to(proof_root.parent)
        result = ingest_file(
            lean_file, proof_root, kb,
            project=args.project,
            use_llm=not args.no_llm,
            dry_run=args.dry_run,
        )
        if result.get("skipped") and "total" not in result:
            total_skipped_files += 1
            continue

        added = result.get("added", 0)
        total = result.get("total", 0)
        total_theorems += total
        total_added_theorems += added

        if total > 0:
            print(f"[{idx+1}/{len(lean_files)}] {rel}: {added}/{total} added")

    print(f"\nDone. Files: {len(lean_files)} ({total_skipped_files} skipped)")
    print(f"Theorems: {total_added_theorems} added / {total_theorems} found")
    print(f"Total in DB: {kb._theorems.count(project=args.project)}")


if __name__ == "__main__":
    main()

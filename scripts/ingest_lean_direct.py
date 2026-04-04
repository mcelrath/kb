#!/usr/bin/env python3
"""
Direct Lean source parser for Mathlib theorem ingestion.

Bypasses lean-dojo by parsing .lean files directly with regex.
Extracts theorem/lemma declarations with their type signatures.
Only processes files that have a corresponding .olean build artifact.

Usage:
    python scripts/ingest_lean_direct.py [--mathlib-root DIR]
        [--project NAME] [--workers N] [--limit N] [--module-filter PREFIX]
        [--dry-run]
"""

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---- Lean declaration regex ----
# Matches: [@attr] [private|protected] theorem|lemma|def NAME
DECL_RE = re.compile(
    r'^(?:@\[[^\]]*\]\s*)*'              # optional attributes
    r'(?:private\s+|protected\s+)?'       # optional private/protected
    r'(theorem|lemma)\s+'                  # keyword (theorem or lemma only)
    r'([\w\'\.]+)',                        # name
    re.MULTILINE,
)

NAMESPACE_PUSH = re.compile(r'^namespace\s+([\w\'\.]+)', re.MULTILINE)
NAMESPACE_POP = re.compile(r'^end\s+([\w\'\.]+)', re.MULTILINE)
SECTION_PUSH = re.compile(r'^section\b', re.MULTILINE)
SECTION_POP = re.compile(r'^end\b(?!\s+\w)', re.MULTILINE)


def module_from_path(lean_file: Path, repo_root: Path) -> str:
    """Convert file path to dotted module name."""
    rel = lean_file.relative_to(repo_root)
    return str(rel.with_suffix("")).replace("/", ".")


def extract_statement(text: str, match_start: int, match_end: int) -> str:
    """Extract the type signature of a theorem declaration.

    Captures from the keyword position to := or where or next declaration.
    """
    # Find the end of the statement (before :=, by, where at top level)
    # We'll scan character by character tracking paren depth
    i = match_start
    n = len(text)
    depth = 0
    stmt_end = match_end  # at minimum, just the name

    # Find ':=' or 'where' or end of block at depth 0
    while i < n:
        c = text[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif depth == 0:
            # Check for := or where or by at depth 0
            if text[i:i+2] == ':=':
                stmt_end = i
                break
            if text[i:i+3] == 'by\n' or text[i:i+3] == 'by ':
                stmt_end = i
                break
            if text[i:i+5] == 'where':
                stmt_end = i
                break
        i += 1
    else:
        stmt_end = min(match_start + 500, n)

    raw = text[match_start:stmt_end].strip()
    # Collapse whitespace runs
    return re.sub(r'\s+', ' ', raw)


def parse_lean_file(lean_file: Path, repo_root: Path) -> list[dict]:
    """Parse a .lean file and return theorem declarations."""
    try:
        text = lean_file.read_text(errors='replace')
    except OSError:
        return []

    module = module_from_path(lean_file, repo_root)

    # Build a simple namespace stack by line scanning
    # For qualified names, track namespace openings/closings
    ns_stack: list[str] = []
    results = []

    lines = text.splitlines(keepends=True)
    line_starts = []  # byte offset of each line start
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)

    def offset_to_lineno(offset: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1  # 1-indexed

    # Process namespace/end interleaved with declarations
    # Use a single pass over lines
    ns_stack = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Namespace push
        m = re.match(r'^namespace\s+([\w\'\.]+)', stripped)
        if m:
            ns_stack.append(m.group(1))
            i += 1
            continue

        # End (pop namespace or section)
        m = re.match(r'^end\s+([\w\'\.]+)', stripped)
        if m:
            name = m.group(1)
            # Pop the matching namespace
            for j in range(len(ns_stack) - 1, -1, -1):
                if ns_stack[j] == name:
                    ns_stack = ns_stack[:j]
                    break
            i += 1
            continue

        # Theorem/lemma declaration (possibly with preceding attributes)
        # Collect attribute lines before the declaration
        decl_lines = []
        j = i
        # Skip leading @[...] attribute lines
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith('@[') or s.startswith('-- ') or s == '':
                if s.startswith('@[') or s == '':
                    decl_lines.append(lines[j])
                    j += 1
                else:
                    break
            else:
                break

        # Check if current or upcoming line has theorem/lemma
        if j < len(lines):
            s = lines[j].strip()
            dm = re.match(
                r'^(?:private\s+|protected\s+)?(theorem|lemma)\s+([\w\'\.]+)',
                s
            )
            if dm:
                # Skip private declarations
                is_private = bool(re.match(r'^private\s+', s))
                if not is_private:
                    kw = dm.group(1)
                    local_name = dm.group(2)
                    lineno = j + 1  # 1-indexed

                    # Full name
                    if ns_stack:
                        full_name = module + "." + ".".join(ns_stack) + "." + local_name
                    else:
                        full_name = module + "." + local_name

                    # Collect statement text (up to ~20 lines or until := or by)
                    stmt_lines = []
                    k = j
                    depth = 0
                    found_end = False
                    while k < len(lines) and k < j + 30:
                        ln_text = lines[k]
                        stmt_lines.append(ln_text.rstrip())
                        # Track bracket depth
                        for ch in ln_text:
                            if ch in '([{':
                                depth += 1
                            elif ch in ')]}':
                                depth -= 1
                        if depth == 0:
                            # Check for := or by or where
                            stripped_ln = ln_text.rstrip()
                            if re.search(r':=\s*$|:=\s*by\b|\bwhere\s*$', stripped_ln):
                                found_end = True
                                k += 1
                                break
                            if k > j and re.match(r'\s*(theorem|lemma|def|instance|class|structure)\b', ln_text):
                                # Hit next declaration
                                k = k  # don't include
                                break
                        k += 1

                    stmt = ' '.join(l.strip() for l in stmt_lines if l.strip())
                    # Clean up := and proof tail
                    stmt = re.sub(r'\s*:=\s*(by\s*)?.*$', '', stmt, flags=re.DOTALL)
                    stmt = re.sub(r'\s+', ' ', stmt).strip()

                    results.append({
                        'lean_name': full_name,
                        'name': local_name,
                        'statement': stmt,
                        'module': module + ("." + ".".join(ns_stack) if ns_stack else ""),
                        'file': str(lean_file.relative_to(repo_root)),
                        'line': lineno,
                    })
                i = j + 1
                continue

        i += 1

    return results


def has_olean(lean_file: Path, repo_root: Path) -> bool:
    """Check if a .lean file has been compiled (has .olean)."""
    rel = lean_file.relative_to(repo_root)
    olean = repo_root / ".lake" / "build" / "lib" / "lean" / rel.with_suffix(".olean")
    return olean.exists()


def process_file(args: tuple) -> list[dict]:
    lean_file, repo_root = args
    lean_file = Path(lean_file)
    repo_root = Path(repo_root)
    if not has_olean(lean_file, repo_root):
        return []
    return parse_lean_file(lean_file, repo_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mathlib-root", default=str(Path.home() / "Physics/mathlib4"))
    parser.add_argument("--project", default="mathlib")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--module-filter", default=None,
                        help="Only process files matching this module prefix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Files per batch for progress reporting")
    args = parser.parse_args()

    repo_root = Path(args.mathlib_root).expanduser()
    if not repo_root.exists():
        print(f"Not found: {repo_root}", file=sys.stderr)
        sys.exit(1)

    # Collect .lean files
    lean_files = sorted(repo_root.glob("Mathlib/**/*.lean"))
    if args.module_filter:
        prefix = args.module_filter.replace(".", "/")
        lean_files = [f for f in lean_files if prefix in str(f.relative_to(repo_root))]

    print(f"Found {len(lean_files)} .lean files in Mathlib/")

    # Filter to compiled files
    compiled = [f for f in lean_files if has_olean(f, repo_root)]
    print(f"  {len(compiled)} have .olean artifacts")

    if args.dry_run:
        # Show sample
        sample = compiled[:5]
        for f in sample:
            results = parse_lean_file(f, repo_root)
            for r in results[:3]:
                print(f"  {r['lean_name']}: {r['statement'][:80]}")
        print(f"\n[DRY RUN] Would ingest from {len(compiled)} files")
        return

    import uuid
    from datetime import datetime
    from kb import KnowledgeBase
    kb = KnowledgeBase()
    conn = kb._theorems.conn

    added = skipped = total = 0
    EMBED_BATCH = 64  # embed this many at once

    file_args = [(str(f), str(repo_root)) for f in compiled]

    print(f"Parsing {len(compiled)} files with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_file, a): a for a in file_args}
        pending: list[dict] = []
        done = 0

        def flush_batch(batch: list[dict]) -> tuple[int, int]:
            """Insert batch with batch embeddings. Returns (added, skipped)."""
            if not batch:
                return 0, 0
            a = s = 0
            # Deduplicate within batch
            seen_keys: set[tuple] = set()
            unique = []
            for thm in batch:
                k = (thm['lean_name'], thm['file'])
                if k not in seen_keys:
                    seen_keys.add(k)
                    unique.append(thm)

            # Check existing
            new_thms = []
            for thm in unique:
                ex = conn.execute(
                    "SELECT id FROM lean_theorems WHERE lean_name=? AND file=?",
                    (thm['lean_name'], thm['file'])
                ).fetchone()
                if ex:
                    s += 1
                else:
                    new_thms.append(thm)

            if not new_thms:
                return a, s

            # Batch embed
            embed_texts = [t['statement'] or t['lean_name'] for t in new_thms]
            embeddings = kb._theorems.embedding_service.embed_batch(embed_texts)
            now = datetime.utcnow().isoformat()

            for thm, emb in zip(new_thms, embeddings):
                tid = f"thm-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
                conn.execute(
                    """INSERT OR IGNORE INTO lean_theorems
                       (id, lean_name, name, statement, declaration, module,
                        file, line, project, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (tid, thm['lean_name'], thm['name'], thm['statement'],
                     thm['statement'], thm['module'], thm['file'],
                     thm['line'], args.project, now, now),
                )
                if emb is not None:
                    conn.execute("DELETE FROM lean_theorems_vec WHERE id=?", (tid,))
                    conn.execute(
                        "INSERT INTO lean_theorems_vec (id, embedding) VALUES (?,?)",
                        (tid, emb),
                    )
                a += 1
            conn.commit()
            return a, s + len(unique) - len(new_thms)

        for fut in as_completed(futures):
            theorems = fut.result()
            pending.extend(theorems)
            done += 1

            while len(pending) >= EMBED_BATCH:
                batch = pending[:EMBED_BATCH]
                pending = pending[EMBED_BATCH:]
                if args.limit and total >= args.limit:
                    break
                a, s = flush_batch(batch)
                added += a
                skipped += s
                total += a + s

            if done % 500 == 0 or done == len(file_args):
                print(f"  files: {done}/{len(file_args)}  theorems: {total}  added: {added}")

            if args.limit and total >= args.limit:
                break

        # Flush remainder
        if pending and not (args.limit and total >= args.limit):
            a, s = flush_batch(pending)
            added += a
            skipped += s
            total += a + s

    print(f"\nDone.")
    print(f"  Files processed: {done}")
    print(f"  Theorems added:  {added}")
    print(f"  Theorems skipped (dup): {skipped}")
    print(f"  Total in DB (project={args.project}): {kb._theorems.count(project=args.project)}")


if __name__ == "__main__":
    main()

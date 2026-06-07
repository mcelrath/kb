#!/usr/bin/env python3
"""
Lean theorem ingestion — direct regex parser (no LeanDojo).

Auto-discovers two repos:
  1. ~/Physics/claude/proofs/  — all compiled .lean files
  2. ~/Physics/mathlib4/       — only files authored by Bob McElrath on the ag branch

Skips files without a .olean artifact (not yet compiled).
Only inserts theorems not already in the DB (lean_name + file key).
After inserting, runs one LLM call per file to generate human-readable
LaTeX summaries stored in statement_pure.

Usage:
    python scripts/ingest_lean_direct.py [--dry-run] [--limit N]
        [--files FILE ...]       # incremental: process only these absolute paths
        [--no-summarize]         # skip LLM summary pass (fast, post-commit hook)
        [--summarize-only]       # only fill missing statement_pure, no new inserts
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

NAMESPACE_PUSH = re.compile(r'^namespace\s+([\w\'\.]+)', re.MULTILINE)
NAMESPACE_POP  = re.compile(r'^end\s+([\w\'\.]+)', re.MULTILINE)

PROOFS_ROOT    = Path.home() / "Physics/claude/proofs"
MATHLIB_ROOT   = Path.home() / "Physics/mathlib4"
MATHLIB_AUTHOR = "Bob McElrath"

# LLM context: cap file content to avoid blowing the context window
MAX_FILE_CHARS = 12_000
# LLM parallel workers (LLM is serial on the server side; 2-4 avoids starvation)
LLM_WORKERS = 3


# ---------------------------------------------------------------------------
# Repo helpers
# ---------------------------------------------------------------------------

def _repo_root_for(lean_file: Path) -> Path:
    for parent in lean_file.parents:
        if (parent / "lakefile.toml").exists() or (parent / "lakefile.lean").exists():
            return parent
    return lean_file.parent


def module_from_path(lean_file: Path, repo_root: Path) -> str:
    try:
        rel = lean_file.relative_to(repo_root)
    except ValueError:
        repo_root = _repo_root_for(lean_file)
        try:
            rel = lean_file.relative_to(repo_root)
        except ValueError:
            return lean_file.stem
    return str(rel.with_suffix("")).replace("/", ".")


def has_olean(lean_file: Path, repo_root: Path) -> bool:
    try:
        rel = lean_file.relative_to(repo_root)
    except ValueError:
        repo_root = _repo_root_for(lean_file)
        try:
            rel = lean_file.relative_to(repo_root)
        except ValueError:
            return False
    olean = repo_root / ".lake" / "build" / "lib" / "lean" / rel.with_suffix(".olean")
    return olean.exists()


def mathlib_contribution_files() -> list[Path]:
    """Return .lean files on the ag branch authored by MATHLIB_AUTHOR vs origin/master."""
    try:
        out = subprocess.check_output(
            ["git", "log", "origin/master..ag",
             "--oneline", f"--author={MATHLIB_AUTHOR}", "--name-only"],
            cwd=str(MATHLIB_ROOT), text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    files = []
    for line in out.splitlines():
        if line.endswith(".lean") and "/" in line:
            p = MATHLIB_ROOT / line.strip()
            if p.exists():
                files.append(p)
    return sorted(set(files))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_lean_file(lean_file: Path, repo_root: Path) -> list[dict]:
    try:
        text = lean_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    module = module_from_path(lean_file, repo_root)
    lines  = text.splitlines()
    results: list[dict] = []
    ns_stack: list[str] = []
    i = 0

    while i < len(lines):
        if m := NAMESPACE_PUSH.match(lines[i]):
            ns_stack.append(m.group(1))
            i += 1
            continue

        if m := NAMESPACE_POP.match(lines[i]):
            name = m.group(1)
            for j in range(len(ns_stack) - 1, -1, -1):
                if ns_stack[j] == name:
                    ns_stack = ns_stack[:j]
                    break
            i += 1
            continue

        j = i
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith("@[") or s == "":
                j += 1
            else:
                break

        if j < len(lines):
            s = lines[j].strip()
            dm = re.match(r'^(?:private\s+|protected\s+)?(theorem|lemma)\s+([\w\'\.]+)', s)
            if dm:
                is_private = bool(re.match(r'^private\s+', s))
                if not is_private:
                    local_name = dm.group(2)
                    lineno     = j + 1

                    full_name = (module + "." + ".".join(ns_stack) + "." + local_name
                                 if ns_stack else module + "." + local_name)

                    stmt_lines, depth, k = [], 0, j
                    while k < len(lines) and k < j + 30:
                        ln = lines[k]
                        stmt_lines.append(ln.rstrip())
                        for ch in ln:
                            if ch in "([{": depth += 1
                            elif ch in ")]}": depth -= 1
                        if depth == 0 and re.search(r':=\s*$|:=\s*by\b|\bwhere\s*$', ln.rstrip()):
                            break
                        if k > j and re.match(r'\s*(theorem|lemma|def|instance|class|structure)\b', ln):
                            break
                        k += 1

                    stmt = " ".join(l.strip() for l in stmt_lines if l.strip())
                    stmt = re.sub(r'\s*:=\s*(by\s*)?.*$', '', stmt, flags=re.DOTALL)
                    stmt = re.sub(r'\s+', ' ', stmt).strip()

                    results.append({
                        "lean_name": full_name,
                        "name":      local_name,
                        "statement": stmt,
                        "module":    module + ("." + ".".join(ns_stack) if ns_stack else ""),
                        "file":      str(lean_file.relative_to(repo_root)),
                        "line":      lineno,
                    })
                i = j + 1
                continue

        i += 1

    return results


def process_file(args: tuple) -> list[dict]:
    lean_file, repo_root = Path(args[0]), Path(args[1])
    if not has_olean(lean_file, repo_root):
        return []
    return parse_lean_file(lean_file, repo_root)


# ---------------------------------------------------------------------------
# LLM summarization
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM = """\
You are a mathematical expositor translating Lean 4 theorem statements into \
precise human-readable summaries. Output valid JSON only — no prose outside the JSON object.\
"""

SUMMARIZE_PROMPT = """\
Below is a Lean 4 source file, followed by a list of theorem/lemma names to summarise.

For each name, write exactly one sentence that:
- States what the theorem asserts, including all quantifiers and key hypotheses
- Uses standard LaTeX notation ($\\zeta(s)$, $\\mathrm{GL}_n$, $\\mathbb{{Z}}$, etc.)
- Is precise enough that a mathematician could reconstruct the Lean statement
- Is self-contained (do not say "the above" or refer to the file)

Output a single JSON object mapping each name to its summary string.
Do not include names not in the list. Do not add extra keys.

FILE ({filename}):
```lean
{file_content}
```

THEOREMS TO SUMMARISE (JSON keys must match exactly):
{names_json}
"""


def summarize_file_theorems(
    lean_file: Path,
    theorems: list[dict],
    llm_client,
) -> dict[str, str]:
    """Call LLM once per file; return {local_name: summary_text}."""
    try:
        file_content = lean_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    # Cap content to avoid blowing context window
    file_content = file_content[:MAX_FILE_CHARS]
    if len(file_content) == MAX_FILE_CHARS:
        file_content += "\n... (truncated)"

    names = [t["name"] for t in theorems]
    names_json = json.dumps(names, indent=2)

    prompt = SUMMARIZE_PROMPT.format(
        filename=lean_file.name,
        file_content=file_content,
        names_json=names_json,
    )

    raw = llm_client.complete(
        prompt,
        system_prompt=SUMMARIZE_SYSTEM,
        max_tokens=150 * len(names) + 200,  # ~150 tok per theorem
        temperature=0.2,
        timeout=120,
        thinking=False,
    )
    if not raw:
        return {}

    # Parse JSON — strip markdown fences if present
    raw = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip())
    raw = re.sub(r'\n?```\s*$', '', raw).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return {k: v for k, v in result.items() if isinstance(v, str) and v.strip()}
    except json.JSONDecodeError:
        pass
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest Lean theorems into KB.")
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--limit",          type=int, default=None)
    parser.add_argument("--workers",        type=int, default=32)
    parser.add_argument("--no-summarize",   action="store_true",
                        help="Skip LLM summary pass (fast mode for post-commit hook)")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Only fill missing statement_pure; skip new inserts")
    parser.add_argument("--files", nargs="+", metavar="FILE",
                        help="Incremental: process only these absolute .lean paths")
    args = parser.parse_args()

    # Post-commit hook: skip summarize for speed; human ingestion runs it
    if args.files and not args.summarize_only:
        # Incremental hook runs are fast; don't block the commit on LLM calls
        args.no_summarize = True

    # ---- Build file list ------------------------------------------------
    if args.files and not args.summarize_only:
        file_repo_pairs: list[tuple[Path, Path, str]] = []
        for f in args.files:
            p = Path(f).resolve()
            if not p.exists() or p.suffix != ".lean":
                print(f"Warning: skipping {f}", file=sys.stderr)
                continue
            if str(p).startswith(str(MATHLIB_ROOT)):
                repo, project = MATHLIB_ROOT, "mathlib"
            else:
                repo = _repo_root_for(p)
                project = "algebraic-genesis"
            file_repo_pairs.append((p, repo, project))
        print(f"Incremental mode: {len(file_repo_pairs)} .lean files")
    else:
        file_repo_pairs = []

        if PROOFS_ROOT.exists():
            proofs_files = sorted(PROOFS_ROOT.glob("**/*.lean"))
            compiled = [(f, PROOFS_ROOT, "algebraic-genesis")
                        for f in proofs_files if has_olean(f, PROOFS_ROOT)]
            print(f"proofs/: {len(proofs_files)} .lean files, {len(compiled)} compiled")
            file_repo_pairs.extend(compiled)

        if MATHLIB_ROOT.exists():
            contrib = mathlib_contribution_files()
            compiled_m = [(f, MATHLIB_ROOT, "mathlib")
                          for f in contrib if has_olean(f, MATHLIB_ROOT)]
            print(f"mathlib4/ag contributions: {len(contrib)} files, {len(compiled_m)} compiled")
            file_repo_pairs.extend(compiled_m)

    if args.dry_run:
        for f, repo, proj in file_repo_pairs[:5]:
            thms = parse_lean_file(f, repo)
            for t in thms[:3]:
                print(f"  [{proj}] {t['lean_name']}: {t['statement'][:80]}")
        print(f"\n[DRY RUN] Would process {len(file_repo_pairs)} files")
        return

    # ---- Setup DB + LLM -------------------------------------------------
    import uuid
    from datetime import datetime, timezone
    from kb import KnowledgeBase

    kb   = KnowledgeBase()
    conn = kb._theorems.conn

    llm = None
    if not args.no_summarize:
        from kb.llm.client import LLMClient
        import os
        llm_url = os.environ.get("KB_LLM_URL", "http://tardis:9510/completion")
        llm = LLMClient(llm_url=llm_url)

    # ---- summarize-only mode: fill missing statement_pure ---------------
    if args.summarize_only:
        if llm is None:
            from kb.llm.client import LLMClient
            import os
            llm = LLMClient(llm_url=os.environ.get("KB_LLM_URL", "http://tardis:9510/completion"))

        missing = conn.execute(
            "SELECT id, lean_name, name, file, project FROM lean_theorems "
            "WHERE (statement_pure IS NULL OR statement_pure = '') "
            "  AND statement IS NOT NULL AND statement != '' "
            "ORDER BY project, file"
        ).fetchall()
        print(f"Theorems missing statement_pure: {len(missing)}")
        if args.limit:
            missing = missing[:args.limit]

        # Group by (project, file)
        from itertools import groupby
        by_file: dict[tuple, list] = {}
        for row in missing:
            tid, lean_name, name, rel_file, project = row
            if project == "algebraic-genesis":
                abs_file = PROOFS_ROOT / rel_file
            else:
                abs_file = MATHLIB_ROOT / rel_file
            key = (str(abs_file), project)
            by_file.setdefault(key, []).append({"id": tid, "name": name, "lean_name": lean_name})

        updated = 0

        def _summarize_group(item):
            abs_path_str, project = item[0]
            theorems = item[1]
            abs_path = Path(abs_path_str)
            if not abs_path.exists():
                return []
            summaries = summarize_file_theorems(abs_path, theorems, llm)
            return [(t["id"], summaries.get(t["name"], "")) for t in theorems]

        print(f"Generating summaries for {len(by_file)} files with {LLM_WORKERS} LLM workers...")
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            futs = {pool.submit(_summarize_group, item): item for item in by_file.items()}
            done = 0
            for fut in as_completed(futs):
                for tid, summary in (fut.result() or []):
                    if summary:
                        conn.execute(
                            "UPDATE lean_theorems SET statement_pure=? WHERE id=?",
                            (summary, tid)
                        )
                        updated += 1
                done += 1
                if done % 20 == 0 or done == len(by_file):
                    conn.commit()
                    print(f"  files: {done}/{len(by_file)}  updated: {updated}")
        conn.commit()
        print(f"\nDone.  Summaries written: {updated}")
        return

    # ---- Full ingest pass -----------------------------------------------
    added = skipped = total = 0
    EMBED_BATCH = 64
    # newly inserted rows grouped by file, for post-insert summarization
    new_by_file: dict[str, list[dict]] = {}  # abs_file_str -> [{id, name, lean_name}]

    def flush_batch(batch: list[dict], project: str) -> tuple[int, int]:
        if not batch:
            return 0, 0
        seen: set[tuple] = set()
        unique = []
        for t in batch:
            k = (t["lean_name"], t["file"])
            if k not in seen:
                seen.add(k)
                unique.append(t)

        new_thms = [t for t in unique if not conn.execute(
            "SELECT id FROM lean_theorems WHERE lean_name=? AND file=?",
            (t["lean_name"], t["file"])
        ).fetchone()]

        if not new_thms:
            return 0, len(unique)

        embed_texts = [t["statement"] or t["lean_name"] for t in new_thms]
        embeddings  = kb._theorems.embedding_service.embed_batch(embed_texts)
        now = datetime.now(timezone.utc).isoformat()

        a = 0
        for thm, emb in zip(new_thms, embeddings):
            tid = f"thm-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            conn.execute(
                """INSERT OR IGNORE INTO lean_theorems
                   (id, lean_name, name, statement, declaration, module,
                    file, line, project, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (tid, thm["lean_name"], thm["name"], thm["statement"],
                 thm["statement"], thm["module"], thm["file"],
                 thm["line"], project, now, now),
            )
            if emb is not None:
                conn.execute("DELETE FROM lean_theorems_vec WHERE id=?", (tid,))
                conn.execute("INSERT INTO lean_theorems_vec (id, embedding) VALUES (?,?)",
                             (tid, emb))
            a += 1
            if not args.no_summarize:
                # Track for summarization: need abs path
                # file is stored relative to repo root; reconstruct abs path
                if project == "algebraic-genesis":
                    abs_f = str(PROOFS_ROOT / thm["file"])
                else:
                    abs_f = str(MATHLIB_ROOT / thm["file"])
                new_by_file.setdefault(abs_f, []).append(
                    {"id": tid, "name": thm["name"], "lean_name": thm["lean_name"]}
                )

        conn.commit()
        return a, len(unique) - a

    # ---- Parse + embed (parallel) ----------------------------------------
    from itertools import groupby
    sorted_pairs = sorted(file_repo_pairs, key=lambda x: x[2])
    for project, grp in groupby(sorted_pairs, key=lambda x: x[2]):
        group_files = list(grp)
        file_args   = [(str(f), str(repo)) for f, repo, _ in group_files]
        pending: list[dict] = []
        done = 0

        print(f"\nParsing {len(file_args)} files [{project}] with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, a): a for a in file_args}

            for fut in as_completed(futures):
                pending.extend(fut.result())
                done += 1

                while len(pending) >= EMBED_BATCH:
                    batch, pending = pending[:EMBED_BATCH], pending[EMBED_BATCH:]
                    if args.limit and total >= args.limit:
                        break
                    a, s = flush_batch(batch, project)
                    added += a; skipped += s; total += a + s

                if done % 200 == 0 or done == len(file_args):
                    print(f"  files: {done}/{len(file_args)}  added: {added}  skipped: {skipped}")

                if args.limit and total >= args.limit:
                    break

            if pending and not (args.limit and total >= args.limit):
                a, s = flush_batch(pending, project)
                added += a; skipped += s

    print(f"\nInserted: {added}  Skipped (dup): {skipped}")

    # ---- LLM summarization pass (new theorems only) ---------------------
    if not args.no_summarize and new_by_file and llm is not None:
        print(f"\nGenerating summaries for {len(new_by_file)} files "
              f"({sum(len(v) for v in new_by_file.values())} theorems) "
              f"with {LLM_WORKERS} LLM workers...")

        def _summarize_group(item):
            abs_path_str, theorems = item
            abs_path = Path(abs_path_str)
            if not abs_path.exists():
                return []
            summaries = summarize_file_theorems(abs_path, theorems, llm)
            return [(t["id"], summaries.get(t["name"], "")) for t in theorems]

        summarized = 0
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
            futs = {pool.submit(_summarize_group, item): item
                    for item in new_by_file.items()}
            done = 0
            for fut in as_completed(futs):
                for tid, summary in (fut.result() or []):
                    if summary:
                        conn.execute(
                            "UPDATE lean_theorems SET statement_pure=? WHERE id=?",
                            (summary, tid)
                        )
                        summarized += 1
                done += 1
                if done % 10 == 0 or done == len(new_by_file):
                    conn.commit()
                    print(f"  files: {done}/{len(new_by_file)}  summaries: {summarized}")
        conn.commit()
        print(f"Summaries written: {summarized}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lean Theorem Ingestion Script

Uses LeanDojo to trace Lean 4 projects and ingest theorem declarations
into the KB theorem index with compiler-accurate extraction and real
dependency edges.

Supports two sources:
  --source=proofs  : ~/Physics/claude/proofs/ (now its own git repo)
  --source=mathlib : ~/Physics/mathlib4 (local Mathlib fork)

For private proofs, patches LeanDojo to resolve local path dependencies
(e.g. path = "../../mathlib4") to their git remote URL + HEAD commit.

Requires lean-dojo-v2:
    pip install --no-deps lean-dojo-v2 loguru PyGithub

Usage:
    python scripts/ingest_lean.py [--source proofs|mathlib]
        [--project NAME] [--no-llm] [--dry-run] [--limit N]
        [--trace-cache DIR] [--module-filter PREFIX]
"""

import argparse
import re
import sys
import types
from pathlib import Path
from typing import Union, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Monkey-patch LeanDojo to support local path dependencies.
# When a lakefile.toml has  path = "../../mathlib4",  resolve the path to its
# git repo, read the origin remote URL and HEAD commit, and construct a
# LeanGitRepo from those.  This lets LeanDojo trace projects that depend on a
# local Mathlib checkout rather than a pinned GitHub URL.
# ---------------------------------------------------------------------------

def _patch_leandojo_local_paths():
    import lean_dojo_v2.lean_dojo.data_extraction.lean as _lean_mod
    from lean_dojo_v2.lean_dojo.data_extraction.lean import (
        LeanGitRepo,
        _LAKEFILE_TOML_REQUIREMENT_REGEX,
        RepoType,
    )
    from git import Repo as GitRepo

    # Patch url_to_repo to use the local lean4 elan toolchain instead of
    # hitting the GitHub API. For GitHub URLs, return a GitRepo pointing at
    # the local elan toolchain path.
    _orig_url_to_repo = _lean_mod.url_to_repo

    def _url_to_repo_patched(url, repo_type=None, tmp_dir=None):
        import subprocess as _sp, re as _re, os as _os
        repo_type = repo_type or _lean_mod.get_repo_type(url)
        if repo_type == RepoType.GITHUB:
            # Resolve via local elan instead of GitHub API
            try:
                prefix = _sp.check_output(["lean", "--print-prefix"], text=True).strip()
                # prefix is like ~/.elan/toolchains/leanprover--lean4---v4.29.0-rc6
                toolchain_root = _os.path.dirname(prefix)  # ~/.elan/toolchains
                # Find matching toolchain by URL fragment
                name_fragment = url.split("/")[-1]  # e.g. "lean4"
                candidates = [d for d in _os.listdir(toolchain_root) if name_fragment in d]
                if candidates:
                    local_path = _os.path.join(toolchain_root, candidates[0])
                    if _os.path.isdir(local_path):
                        # Initialize a bare-minimum git repo object without remote
                        try:
                            return GitRepo(local_path)
                        except Exception:
                            pass
            except Exception:
                pass
        return _orig_url_to_repo(url, repo_type=repo_type, tmp_dir=tmp_dir)

    _lean_mod.url_to_repo = _url_to_repo_patched

    def _parse_lakefile_toml_dependencies_patched(
        self, path: Union[str, Path, None]
    ) -> List[Tuple[str, "LeanGitRepo"]]:
        lakefile = (
            self.get_config("lakefile.toml")
            if path is None
            else (Path(path) / "lakefile.toml").open().read()
        )
        if isinstance(lakefile, dict) and "require" in lakefile:
            matches = lakefile["require"]
        else:
            if "content" in lakefile:
                lakefile = lakefile["content"]
            matches = []
            for req in _LAKEFILE_TOML_REQUIREMENT_REGEX.finditer(lakefile):
                match: dict = {}
                for line in req.group().strip().splitlines():
                    key, value = line.split("=")
                    match[key.strip()] = value.strip().strip('"')
                matches.append(match)

        resolved = []
        repo_root = Path(path or self.url)
        for match in matches:
            if "path" in match:
                dep_path = (repo_root / match["path"]).resolve()
                git = GitRepo(dep_path)
                url = git.remotes["origin"].url
                commit = git.head.commit.hexsha
                print(
                    f"  [local dep] {match.get('name', dep_path.name)}: "
                    f"{url} @ {commit[:12]}",
                    file=sys.stderr,
                )
                resolved.append((match.get("name", dep_path.name), LeanGitRepo(url, commit)))
            else:
                if "git" in match:
                    match["url"] = match["git"]
                    del match["git"]
                resolved.extend(self._parse_deps([match]))

        return resolved

    LeanGitRepo._parse_lakefile_toml_dependencies = (
        _parse_lakefile_toml_dependencies_patched
    )


_patch_leandojo_local_paths()

from lean_dojo_v2.lean_dojo import LeanGitRepo, TracedRepo, trace  # noqa: E402
from lean_dojo_v2.lean_dojo.data_extraction.trace import (  # noqa: E402
    LEAN4_DATA_EXTRACTOR_PATH,
    check_files,
    launch_progressbar,
)
from lean_dojo_v2.utils.common import execute  # noqa: E402


# ---------------------------------------------------------------------------
# In-place tracing: run ExtractData.lean on an already-built repo,
# skipping the clone + lake build steps.
# ---------------------------------------------------------------------------

def trace_inplace(repo_path: Path, dst_dir: Path) -> TracedRepo:
    """Run LeanDojo's AST extraction on an already-built repo in repo_path.

    Skips `git clone` and `lake build` — assumes the repo is already compiled.
    Copies ExtractData.lean into the repo, runs it, then builds a TracedRepo.
    """
    import shutil
    from lean_dojo_v2.utils.constants import NUM_PROCS

    lean_prefix = execute("lean --print-prefix", capture_output=True)[0].strip()
    packages_path = repo_path / ".lake" / "packages"
    build_path = repo_path / ".lake" / "build"

    # Copy Lean stdlib into packages (LeanDojo needs it there)
    lean4_pkg = packages_path / "lean4"
    if not lean4_pkg.exists():
        print("  Copying Lean stdlib into .lake/packages/lean4 ...")
        shutil.copytree(lean_prefix, str(lean4_pkg), dirs_exist_ok=True)

    # Copy ExtractData.lean and run it
    extractor_dst = repo_path / LEAN4_DATA_EXTRACTOR_PATH.name
    shutil.copyfile(LEAN4_DATA_EXTRACTOR_PATH, extractor_dst)
    try:
        orig_dir = Path.cwd()
        import os
        os.chdir(repo_path)
        print("  Running ExtractData.lean ...")
        with launch_progressbar([build_path]):
            execute(f"lake env lean --threads {NUM_PROCS} --run ExtractData.lean noDeps")
        check_files(packages_path, no_deps=True)
    finally:
        os.chdir(orig_dir)
        extractor_dst.unlink(missing_ok=True)

    traced = TracedRepo.from_traced_files(repo_path, build_deps=False)
    traced.save_to_disk()

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_path, dst_dir, dirs_exist_ok=True)
    return traced


# ---------------------------------------------------------------------------
# tex_source extraction
# ---------------------------------------------------------------------------

TEX_SOURCE_RE = re.compile(r"--\s*source:\s*(.+)", re.IGNORECASE)


def extract_tex_source(lean_file: Path, line: int, window: int = 5) -> Optional[str]:
    """Scan lines near `line` for a -- source: FILE.tex line N comment."""
    try:
        lines = lean_file.read_text(errors="replace").splitlines()
    except OSError:
        return None
    start = max(0, line - window - 1)
    end = min(len(lines), line + window)
    for ln in lines[start:end]:
        m = TEX_SOURCE_RE.search(ln)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_traced_repo(
    traced: TracedRepo,
    kb,
    project: str,
    proof_root: Path,
    use_llm: bool,
    dry_run: bool,
    limit: Optional[int],
    module_filter: Optional[str],
) -> dict:
    theorems = list(traced.get_traced_theorems())

    if module_filter:
        theorems = [
            t for t in theorems
            if module_filter in str(t.traced_file.path)
        ]
    if limit:
        theorems = theorems[:limit]

    added = skipped = 0
    theorem_id_map: dict[str, str] = {}  # lean_name -> kb id

    for thm in theorems:
        if thm.is_private:
            skipped += 1
            continue

        lean_name = thm.theorem.full_name
        file_rel = str(thm.theorem.file_path)
        stmt = thm.get_theorem_statement() or ""
        line = thm.start.line_nb if thm.start else None

        # Short name: last component of qualified name
        name = lean_name.split(".")[-1]

        # Module: everything before the last component
        parts = lean_name.split(".")
        module = ".".join(parts[:-1]) if len(parts) > 1 else None

        # tex_source from comment scan
        abs_file = proof_root / file_rel if not Path(file_rel).is_absolute() else Path(file_rel)
        tex_source = extract_tex_source(abs_file, line or 0) if abs_file.exists() else None

        # statement_pure via LLM
        statement_pure = None
        if use_llm and not dry_run and stmt:
            statement_pure = _generate_statement_pure(kb._llm, stmt)

        if dry_run:
            print(f"  [DRY] {lean_name}: {stmt[:80]}")
            added += 1
            continue

        result = kb.theorem_add(
            lean_name=lean_name,
            name=name,
            statement=stmt,
            declaration=stmt,
            file=file_rel,
            statement_pure=statement_pure,
            module=module,
            line=line,
            tex_source=tex_source,
            project=project,
        )
        kb_id = result["id"]
        theorem_id_map[lean_name] = kb_id
        if result["is_new"]:
            added += 1
        else:
            skipped += 1

    # Second pass: dependency edges
    dep_added = 0
    if not dry_run:
        for thm in theorems:
            if thm.is_private:
                continue
            lean_name = thm.theorem.full_name
            src_id = theorem_id_map.get(lean_name)
            if not src_id:
                continue
            for prem in thm.get_premise_full_names():
                dep_id = theorem_id_map.get(prem)
                if dep_id:
                    kb.theorem_add_dependency(src_id, dep_id)
                    dep_added += 1

    return {"added": added, "skipped": skipped, "total": len(theorems), "deps": dep_added}


PURE_MATH_PROMPT = (
    "Restate the following Lean theorem in pure mathematical language. "
    "Use no domain-specific framing. Keep it under 30 tokens. "
    "Return only the restatement.\n\nLean statement:\n{statement}\n\nPure math restatement:"
)


def _generate_statement_pure(llm_client, statement: str) -> Optional[str]:
    try:
        result = llm_client.complete(
            PURE_MATH_PROMPT.format(statement=statement[:800]),
            max_tokens=80,
            temperature=0.2,
            use_chat=False,
        )
        return result.strip().strip('"').strip("'") if result else None
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest Lean theorems into KB using LeanDojo")
    parser.add_argument(
        "--source",
        choices=["proofs", "mathlib"],
        default="proofs",
        help="Which repo to trace (default: proofs)",
    )
    parser.add_argument(
        "--proofs-root",
        default=str(Path.home() / "Physics/claude/proofs"),
        help="Path to private proofs repo",
    )
    parser.add_argument(
        "--mathlib-root",
        default=str(Path.home() / "Physics/mathlib4"),
        help="Path to local Mathlib fork",
    )
    parser.add_argument("--project", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--module-filter", default=None)
    parser.add_argument(
        "--trace-cache",
        default=str(Path.home() / ".cache/lean_dojo"),
        help="Directory for LeanDojo trace cache",
    )
    args = parser.parse_args()

    if args.source == "proofs":
        repo_path = Path(args.proofs_root).expanduser()
        default_project = "exterior_algebra"
    else:
        repo_path = Path(args.mathlib_root).expanduser()
        default_project = "mathlib"

    project = args.project or default_project

    if not repo_path.exists():
        print(f"Repo not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Tracing {repo_path} ...")
    repo = LeanGitRepo.from_path(repo_path)
    print(f"  URL:    {repo.url}")
    print(f"  Commit: {repo.commit}")

    trace_dst = Path(args.trace_cache) / f"{repo_path.name}-{repo.commit[:12]}"
    if trace_dst.exists():
        print(f"Loading cached trace from {trace_dst}")
        traced = TracedRepo.load_from_disk(trace_dst)
    else:
        print(f"Running ExtractData.lean in-place (skipping lake build) ...")
        traced = trace_inplace(repo_path, trace_dst)

    from kb import KnowledgeBase
    kb = KnowledgeBase()

    print(f"Ingesting into project={project} ...")
    result = ingest_traced_repo(
        traced=traced,
        kb=kb,
        project=project,
        proof_root=repo_path,
        use_llm=not args.no_llm,
        dry_run=args.dry_run,
        limit=args.limit,
        module_filter=args.module_filter,
    )

    print(f"\nDone.")
    print(f"  Theorems added:  {result['added']}")
    print(f"  Theorems skipped: {result['skipped']}")
    print(f"  Total processed: {result['total']}")
    print(f"  Dependency edges: {result['deps']}")
    print(f"  Total in DB (project={project}): {kb._theorems.count(project=project)}")


if __name__ == "__main__":
    main()

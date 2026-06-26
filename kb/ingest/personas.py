#!/usr/bin/env python3
"""
Ingest persona files (<project>/.claude/agents/personas/*.md) into the KB.

Stores searchable content in findings (reuses the existing embedding+FTS path)
and structural staleness metadata in persona_index.

Staleness signals (--check):
  (a) referenced file paths no longer exist
  (b) linked reviewers.yaml mtime > persona mtime (out of sync)
  (c) current project top-level-dir count != stored count
"""

import hashlib
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from kb import KnowledgeBase, DEFAULT_DB_PATH


def _git_root(path: Path) -> Path | None:
    """Return the git root for path, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=str(path)
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    return None


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML-style frontmatter (---...---) from markdown text."""
    fm: dict[str, str] = {}
    if not text.startswith("---"):
        return fm
    end = text.find("---", 3)
    if end == -1:
        return fm
    block = text[3:end]
    for line in block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip().strip('"').strip("'")
    return fm


def _find_reviewers_yaml(file_path: Path, git_root: Path | None) -> Path | None:
    """Find reviewers.yaml in the project root or parents of the persona file."""
    candidates: list[Path] = []
    if git_root:
        candidates.append(git_root / "reviewers.yaml")
    # Walk up from persona file
    p = file_path.parent
    while True:
        candidates.append(p / "reviewers.yaml")
        if p == p.parent:
            break
        p = p.parent
    for c in candidates:
        if c.exists():
            return c
    return None


def _extract_referenced_paths(text: str) -> list[str]:
    """Extract filesystem paths referenced in persona markdown.

    Only paths that contain '/' and have filesystem-relevant extensions.
    Excludes Python module dotted names, Lean identifiers, and prose.
    """
    FS_EXTENSIONS = {
        ".lean", ".yaml", ".yml", ".md", ".sh", ".tex", ".py", ".toml",
        ".json", ".rs", ".ts", ".js", ".txt", ".cfg", ".ini",
    }
    candidates: list[str] = []

    def _looks_like_fs_path(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        # Must have a slash (absolute, relative with /, or ~/) or be ~/...
        if "/" not in s:
            return False
        # Must end with a recognized filesystem extension
        suffix = Path(s.rstrip("/*").split("?")[0]).suffix
        if suffix in FS_EXTENSIONS:
            return True
        # glob wildcard patterns like *.tex in a path
        if re.search(r'\*\.[a-zA-Z]+$', s):
            return True
        return False

    # Only backtick-quoted strings (most reliable signal for file paths in .md)
    for m in re.finditer(r'`([^`\n]+)`', text):
        val = m.group(1).strip()
        if _looks_like_fs_path(val):
            candidates.append(val)

    # Markdown links [text](path) — explicit link syntax
    for m in re.finditer(r'\[[^\]]*\]\(([^)\n]+)\)', text):
        val = m.group(1).strip()
        if _looks_like_fs_path(val):
            candidates.append(val)

    return candidates


def _count_top_level_dirs(root: Path) -> int:
    """Count directories directly under root (excluding hidden)."""
    try:
        return sum(1 for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    except Exception:
        return 0


def discover_personas(roots: list[Path]) -> list[dict[str, Any]]:
    """Discover persona .md files under each root.

    Looks for <root>/.claude/agents/personas/*.md.
    Also checks ~/.claude/agents/personas/*.md.
    """
    found: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    # Include ~/.claude/agents/personas/ always
    home_personas = Path.home() / ".claude" / "agents" / "personas"
    check_dirs: list[tuple[Path, Path | None]] = [(home_personas, None)]
    for root in roots:
        check_dirs.append((root / ".claude" / "agents" / "personas", root))

    for persona_dir, explicit_root in check_dirs:
        if not persona_dir.exists():
            continue
        for md_file in sorted(persona_dir.glob("*.md")):
            if str(md_file) in seen_paths:
                continue
            seen_paths.add(str(md_file))

            # Determine git root
            if explicit_root is not None:
                git_root = _git_root(explicit_root) or explicit_root
            else:
                git_root = _git_root(md_file.parent) or md_file.parent

            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"  Warning: cannot read {md_file}: {e}", file=sys.stderr)
                continue

            fm = _parse_frontmatter(text)
            name = fm.get("name", md_file.stem)
            role = fm.get("role", fm.get("combines", ""))
            archetype = fm.get("archetype", "")
            augmentation = fm.get("augmentation", "")
            project = git_root.name if git_root else "unknown"
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            file_mtime = md_file.stat().st_mtime

            reviewers_yaml = _find_reviewers_yaml(md_file, git_root)
            reviewers_yaml_path = str(reviewers_yaml) if reviewers_yaml else None
            reviewers_yaml_mtime = reviewers_yaml.stat().st_mtime if reviewers_yaml else None

            dir_count = _count_top_level_dirs(git_root) if git_root else 0

            # Build searchable text: name + role + description + domain tags from content
            # Extract the body after the frontmatter block. Slice AFTER the closing '---'
            # LINE — NOT lstrip('---\n'), which strips the char-set {'-','\n'} and would eat
            # a body that starts with a markdown list dash.
            close = text.find("\n---", 3) if text.startswith("---") else -1
            if close != -1:
                nl = text.find("\n", close + 1)
                body = text[nl + 1:].strip() if nl != -1 else ""
            else:
                body = text
            # First 1000 chars of body as searchable content
            searchable_body = body[:1000].strip()

            found.append({
                "name": name,
                "project": project,
                "role": role,
                "file_path": str(md_file),
                "file_mtime": file_mtime,
                "content_hash": content_hash,
                "reviewers_yaml_path": reviewers_yaml_path,
                "reviewers_yaml_mtime": reviewers_yaml_mtime,
                "top_level_dir_count": dir_count,
                "git_root": str(git_root) if git_root else None,
                "searchable_content": (
                    f"Persona: {name}\nProject: {project}\nRole: {role}\n"
                    f"Archetype: {archetype}\nAugmentation: {augmentation}\n\n{searchable_body}"
                ),
            })

    return found


def run(
    roots: list[Path] | None = None,
    dry_run: bool = False,
    check: bool = False,
    project_filter: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Ingest persona files. Returns 0 on success."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    kb = KnowledgeBase(db_path=db_path)

    # --check: staleness report only, no ingest
    if check:
        return _run_check(kb)

    # Discover roots: if none given, walk git root of cwd + home
    if roots is None:
        cwd_root = _git_root(Path.cwd())
        roots = [cwd_root] if cwd_root else []

    personas = discover_personas(roots)
    if project_filter:
        personas = [p for p in personas if p["project"] == project_filter]

    print(f"Found {len(personas)} persona file(s)", file=sys.stderr)

    new_count = updated_count = skipped_count = 0

    for p in personas:
        persona_id = f"persona-{p['name'].lower().replace(' ', '-')}-{p['project']}"

        existing = kb.conn.execute(
            "SELECT id, content_hash, finding_id FROM persona_index WHERE id = ?",
            (persona_id,)
        ).fetchone()

        if existing and existing[1] == p["content_hash"]:
            if dry_run:
                print(f"  [skip-unchanged] {p['name']} ({p['project']})")
            skipped_count += 1
            continue

        if dry_run:
            action = "update" if existing else "new"
            print(f"  [{action}] {p['name']} ({p['project']}) — {p['file_path']}")
            print(f"    role: {p['role'][:80] if p['role'] else '(none)'}")
            continue

        now = datetime.now().isoformat()

        # Upsert into findings (the searchable entry)
        finding_id = existing[2] if existing else None
        content = p["searchable_content"]
        summary = f"Persona {p['name']} ({p['project']}): {p['role'][:100] if p['role'] else 'bridge agent'}"
        tags_json = '["persona", "bridge-agent"]'

        if finding_id:
            # Update existing finding
            embedding = kb._embed(content)
            kb.conn.execute(
                "UPDATE findings SET content=?, summary=?, tags=?, updated_at=? WHERE id=?",
                (content, summary, tags_json, now, finding_id)
            )
            kb.conn.execute("DELETE FROM findings_vec WHERE id=?", (finding_id,))
            kb.conn.execute("INSERT INTO findings_vec (id, embedding) VALUES (?,?)",
                            (finding_id, embedding))
        else:
            # Insert new finding
            finding_id = f"kb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            embedding = kb._embed(content)
            kb.conn.execute("""
                INSERT INTO findings (id, type, status, project, tags, content, summary, created_at, updated_at)
                VALUES (?, 'discovery', 'current', ?, ?, ?, ?, ?, ?)
            """, (finding_id, p["project"], tags_json, content, summary, now, now))
            kb.conn.execute("INSERT INTO findings_vec (id, embedding) VALUES (?,?)",
                            (finding_id, embedding))

        # Upsert persona_index
        if existing:
            kb.conn.execute("""
                UPDATE persona_index SET name=?, project=?, role=?, file_path=?,
                  file_mtime=?, content_hash=?, reviewers_yaml_path=?,
                  reviewers_yaml_mtime=?, top_level_dir_count=?, finding_id=?, updated_at=?
                WHERE id=?
            """, (p["name"], p["project"], p["role"], p["file_path"],
                  p["file_mtime"], p["content_hash"], p["reviewers_yaml_path"],
                  p["reviewers_yaml_mtime"], p["top_level_dir_count"], finding_id, now,
                  persona_id))
            updated_count += 1
        else:
            kb.conn.execute("""
                INSERT INTO persona_index
                  (id, name, project, role, file_path, file_mtime, content_hash,
                   reviewers_yaml_path, reviewers_yaml_mtime, top_level_dir_count,
                   finding_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (persona_id, p["name"], p["project"], p["role"], p["file_path"],
                  p["file_mtime"], p["content_hash"], p["reviewers_yaml_path"],
                  p["reviewers_yaml_mtime"], p["top_level_dir_count"], finding_id, now, now))
            new_count += 1

        kb.conn.commit()
        print(f"  {'Updated' if existing else 'Added'}: {p['name']} ({p['project']}) → {finding_id}")

    if not dry_run:
        print(f"Personas: new={new_count} updated={updated_count} skipped={skipped_count}")
    return 0


def _run_check(kb: KnowledgeBase) -> int:
    """Report stale personas. Returns 0 even if stale found (it's informational)."""
    rows = kb.conn.execute(
        "SELECT id, name, project, file_path, file_mtime, reviewers_yaml_path, "
        "reviewers_yaml_mtime, top_level_dir_count FROM persona_index"
    ).fetchall()

    if not rows:
        print("No personas indexed. Run: kb ingest personas")
        return 0

    stale_count = 0
    for row in rows:
        (pid, name, project, file_path, file_mtime,
         ry_path, ry_mtime, stored_dir_count) = row

        reasons: list[str] = []

        # (a) persona file itself gone
        fp = Path(file_path)
        if not fp.exists():
            reasons.append(f"file gone: {file_path}")
        else:
            # Check referenced paths in the persona file
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
                git_root = _git_root(fp.parent)
                for ref in _extract_referenced_paths(text):
                    # Skip glob patterns — they're references to sets of files
                    if "*" in ref:
                        continue
                    ref_path = Path(ref).expanduser()
                    # Check absolute (after expanduser) first
                    if ref_path.is_absolute():
                        if not ref_path.exists():
                            reasons.append(f"referenced path gone: {ref}")
                    else:
                        # Relative: try from persona dir, then from git root
                        abs1 = fp.parent / ref_path
                        abs2 = (git_root / ref_path) if git_root else None
                        if not abs1.exists() and (abs2 is None or not abs2.exists()):
                            reasons.append(f"referenced path gone: {ref}")
            except Exception:
                pass

        # (b) reviewers.yaml newer than persona (current mtime comparison)
        if ry_path:
            ry = Path(ry_path)
            if ry.exists():
                current_ry_mtime = ry.stat().st_mtime
                current_persona_mtime = fp.stat().st_mtime if fp.exists() else (file_mtime or 0)
                # Round to whole seconds: co-generated files (project-setup writes
                # reviewers.yaml then the personas within the same second) differ only in
                # sub-second float and would ALL false-flag stale under a raw `>`. A genuine
                # later edit still lands in a later second. (impl-review note 1.)
                if int(current_ry_mtime) > int(current_persona_mtime):
                    reasons.append(
                        f"reviewers.yaml newer than persona "
                        f"(ry={datetime.fromtimestamp(current_ry_mtime).strftime('%Y-%m-%d %H:%M')}, "
                        f"persona={datetime.fromtimestamp(current_persona_mtime).strftime('%Y-%m-%d %H:%M')})"
                    )
            else:
                reasons.append(f"reviewers.yaml gone: {ry_path}")

        # (c) project top-level-dir count changed
        if stored_dir_count is not None and fp.exists():
            git_root = _git_root(fp.parent)
            if git_root:
                current_count = _count_top_level_dirs(git_root)
                if current_count != stored_dir_count:
                    reasons.append(
                        f"project dir count changed: stored={stored_dir_count} current={current_count}"
                    )

        # deduplicate reasons
        reasons = list(dict.fromkeys(reasons))

        if reasons:
            stale_count += 1
            print(f"STALE  {name} ({project})")
            for r in reasons:
                print(f"       {r}")
        else:
            print(f"OK     {name} ({project})")

    print(f"\n{stale_count}/{len(rows)} persona(s) stale")
    return 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest persona .md files into the KB persona index"
    )
    parser.add_argument(
        "roots", nargs="*", type=Path,
        help="Project root directories to scan (default: git root of cwd)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed")
    parser.add_argument("--check", action="store_true",
                        help="Report stale personas (file gone / reviewers.yaml newer / dir count changed)")
    parser.add_argument("-p", "--project", dest="project_filter",
                        help="Filter to a specific project name")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="KB database path")
    args = parser.parse_args()
    rc = run(
        roots=args.roots if args.roots else None,
        dry_run=args.dry_run,
        check=args.check,
        project_filter=args.project_filter,
        db_path=args.db,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()

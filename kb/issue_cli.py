"""
kbt / kb issue CLI — kb-native issue tracker contract.

Implements the ~18-op workflow contract emitting bd-compatible --json shapes
(byte-identical key names so existing jq expressions parse without change).

## kb-column → bd-JSON-key projection table

Captured empirically from live `bd show/list/dep list/ready/blocked --json`
on 2026-06-11. Use these key names in all --json output.

bd show --json (epic)
  id                → id
  title             → title
  <design content>  → design   (INLINE CONTENT — NOT a file path; bd emits content)
  status            → status
  priority          → priority
  type              → issue_type
  assignee          → owner    (bd uses owner, not assignee)
  created_by        → created_by
  created_at        → created_at
  updated_at        → updated_at
  (no started_at on epic without started_at)
  <counted from deps> → dependent_count, dependency_count
  <counted from comments> → comment_count

bd show --json (task — differs from epic):
  same as epic PLUS:
  started_at        → started_at   (only when present)
  closed_at         → closed_at    (only when present)
  close_reason      → close_reason (only when present)
  parent_id         → parent       (string id, not nested object)
  <dep rows>        → dependencies (array, each has issue_id/depends_on_id/type/created_at/created_by/metadata)

bd list --json element:
  id, title, status, priority, issue_type, owner, created_at, created_by,
  updated_at, dependencies, dependency_count, dependent_count, comment_count,
  parent  (only when parent_id set)
  (NOTE: list DOES NOT include design/description/started_at inline)

bd dep list --json element (NOT a flat list — nested outgoing/incoming as bd wraps each as dep-target items):
  Empirically: `bd dep list ID --json` returns an array of dep-target items:
  { id, title, design, status, priority, issue_type, owner, created_at,
    created_by, updated_at, dependency_type }
  The dispatch.md jq reads .type/.id/.title/.status from these.
  NOTE: live bd `dep list` returns dependency_type (not type!) in element shape.
  We emit BOTH `type` and `dependency_type` for compat: dispatch uses .type,
  bd emits .dependency_type.

bd ready --json element:
  id, title, design, status, priority, issue_type, owner, created_at,
  created_by, updated_at, dependency_count, dependent_count, comment_count

bd blocked --json element:
  id, title, status, priority, issue_type, owner, created_at, created_by,
  updated_at, blocked_by_count, blocked_by (list of ids)

## design field: INLINE CONTENT (not a file path)

`bd show --json` emits `.design` as the FULL PLAN TEXT (inline content).
expert-review.md:31 does `bd show <epic> --json` → extract `.design` field
and uses it as the plan content directly — NOT opening a file.
The bd_import.py stores `design` content into `issues.design_file` column verbatim.
Therefore: our `show --json` MUST emit `.design` = the content stored in
`issues.design_file` (which IS the content, not a path).
This is consistent: create --design-file=PATH reads the file content and
stores it; show --json emits that content back as .design.
Wait — that would break create --design-file=PATH if we read the file.
Reconciliation: bd's CREATE takes `--design-file PATH` (reads the file and
stores content); bd's SHOW emits the stored content as `.design`.
Our kbt does the same: create stores content; show emits content as `.design`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def resolve_backend(cwd: Path | None = None) -> str:
    """Walk up from cwd to find .beads/config.yaml; return backend ('kb' or 'dolt').

    Priority (highest first):
    1. KBT_BACKEND env var (intended for tests; logged to stderr when it fires)
    2. .beads/config.yaml `backend:` key found by walking up from cwd
    3. Default: 'dolt' (no .beads/ found — matches current behaviour for
       not-yet-migrated projects)

    Returns 'kb' or 'dolt'.
    """
    env_override = os.environ.get("KBT_BACKEND")
    if env_override:
        backend = env_override.lower()
        print(
            f"kbt: KBT_BACKEND={env_override!r} overrides file-resolved backend → {backend}",
            file=sys.stderr,
        )
        return backend

    if cwd is None:
        cwd = Path.cwd()

    # Walk up directory tree looking for .beads/config.yaml
    candidate = cwd.resolve()
    while True:
        config_path = candidate / ".beads" / "config.yaml"
        if config_path.exists():
            backend = _read_backend_from_config(config_path)
            return backend
        parent = candidate.parent
        if parent == candidate:
            # Reached filesystem root, no .beads/ found → default dolt
            return "dolt"
        candidate = parent


def _read_backend_from_config(config_path: Path) -> str:
    """Parse .beads/config.yaml and return the `backend:` value."""
    try:
        import yaml  # type: ignore[import]
        with open(config_path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return str(data.get("backend", "dolt")).lower()
    except Exception:
        # yaml unavailable or parse error → try simple line scan
        try:
            content = config_path.read_text()
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("backend:"):
                    val = line[len("backend:"):].strip().strip("\"'")
                    return val.lower() if val else "dolt"
        except Exception:
            pass
    return "dolt"


# ---------------------------------------------------------------------------
# JSON projection helpers — kb columns → bd key names
# ---------------------------------------------------------------------------

def _dep_counts(conn: Any, issue_id: str) -> tuple[int, int]:
    """Return (dependency_count, dependent_count) for an issue."""
    dep_count = conn.execute(
        "SELECT COUNT(*) FROM issue_deps WHERE issue_id = ? AND type = 'blocks'",
        (issue_id,),
    ).fetchone()[0]
    dep_on_count = conn.execute(
        "SELECT COUNT(*) FROM issue_deps WHERE depends_on_id = ? AND type = 'blocks'",
        (issue_id,),
    ).fetchone()[0]
    return dep_count, dep_on_count


def _comment_count(conn: Any, issue_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM issue_comments WHERE issue_id = ?", (issue_id,)
    ).fetchone()[0]


def _project_show(row: dict[str, Any], conn: Any) -> dict[str, Any]:
    """Project a full issue row (from issue_get) to bd show --json shape."""
    issue_id = row["id"]
    dep_count, dep_on_count = _dep_counts(conn, issue_id)
    cmt_count = _comment_count(conn, issue_id)

    out: dict[str, Any] = {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "priority": row["priority"],
        "issue_type": row["type"],
        "owner": row["assignee"],
        "created_by": row.get("assignee"),  # kb has no created_by column; use assignee
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "dependent_count": dep_on_count,
        "dependency_count": dep_count,
        "comment_count": cmt_count,
    }

    # design: emit content (stored in design_file column)
    if row.get("design_file"):
        out["design"] = row["design_file"]

    # started_at / closed_at / close_reason only when set
    if row.get("started_at"):
        out["started_at"] = row["started_at"]
    if row.get("closed_at"):
        out["closed_at"] = row["closed_at"]
    if row.get("close_reason"):
        out["close_reason"] = row["close_reason"]

    # parent: emit as string id
    if row.get("parent_id"):
        out["parent"] = row["parent_id"]

    # dependencies: array of dep-edge dicts (issue_id, depends_on_id, type, ...)
    if row.get("deps"):
        out["dependencies"] = [
            {
                "issue_id": d["issue_id"],
                "depends_on_id": d["depends_on_id"],
                "type": d["type"],
                "created_at": d["created_at"],
                "created_by": d.get("created_by"),
                "metadata": "{}",
            }
            for d in row["deps"]
        ]

    return out


def _project_list_item(row: dict[str, Any], conn: Any) -> dict[str, Any]:
    """Project a list row to bd list --json element shape."""
    issue_id = row["id"]
    dep_count, dep_on_count = _dep_counts(conn, issue_id)
    cmt_count = _comment_count(conn, issue_id)

    out: dict[str, Any] = {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "priority": row["priority"],
        "issue_type": row["type"],
        "owner": row.get("assignee"),
        "created_at": row.get("created_at"),
        "created_by": row.get("assignee"),
        "updated_at": row.get("updated_at"),
        "dependency_count": dep_count,
        "dependent_count": dep_on_count,
        "comment_count": cmt_count,
    }
    if row.get("parent_id"):
        out["parent"] = row["parent_id"]
    return out


def _project_dep_list_item(dep_row: dict[str, Any], conn: Any, dep_type: str) -> dict[str, Any]:
    """Project a dep target issue to bd dep list --json element shape.

    dep list elements in bd show:
    { id, title, design, status, priority, issue_type, owner, created_at,
      created_by, updated_at, dependency_type }
    dispatch.md:49 reads .type/.id/.title/.status from these;
    bd actually emits `dependency_type` not `type` in dep list elements.
    We emit both for full compat.
    """
    out: dict[str, Any] = {
        "id": dep_row["id"],
        "title": dep_row["title"],
        "status": dep_row["status"],
        "priority": dep_row["priority"],
        "issue_type": dep_row["type"],
        "owner": dep_row.get("assignee"),
        "created_at": dep_row.get("created_at"),
        "created_by": dep_row.get("assignee"),
        "updated_at": dep_row.get("updated_at"),
        "dependency_type": dep_type,
        "type": dep_type,  # dispatch.md jq reads .type
    }
    if dep_row.get("design_file"):
        out["design"] = dep_row["design_file"]
    return out


def _project_ready_item(row: dict[str, Any], conn: Any) -> dict[str, Any]:
    """Project a ready row to bd ready --json element shape."""
    issue_id = row["id"]
    dep_count, dep_on_count = _dep_counts(conn, issue_id)
    cmt_count = _comment_count(conn, issue_id)
    out: dict[str, Any] = {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "priority": row["priority"],
        "issue_type": row["type"],
        "owner": row.get("assignee"),
        "created_at": row.get("created_at"),
        "created_by": row.get("assignee"),
        "updated_at": row.get("updated_at"),
        "dependency_count": dep_count,
        "dependent_count": dep_on_count,
        "comment_count": cmt_count,
    }
    # ready items include design when set (live bd does this)
    # We fetch it from the full row if available
    design = row.get("design_file")
    if design:
        out["design"] = design
    return out


def _project_blocked_item(row: dict[str, Any], conn: Any) -> dict[str, Any]:
    """Project a blocked row to bd blocked --json element shape."""
    out: dict[str, Any] = {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "priority": row["priority"],
        "issue_type": row["type"],
        "owner": row.get("assignee"),
        "created_at": row.get("created_at"),
        "created_by": row.get("assignee"),
        "updated_at": row.get("updated_at"),
        "blocked_by": row.get("blocker_ids", []),
        "blocked_by_count": len(row.get("blocker_ids", [])),
    }
    return out


# ---------------------------------------------------------------------------
# KnowledgeBase construction helper
# ---------------------------------------------------------------------------

def _build_kb(db_path: Path | None = None) -> Any:
    """Build a KnowledgeBase instance, optionally overriding db_path.

    Resolution: explicit db_path arg (--db) > KBT_DB env > DEFAULT_DB_PATH.
    The KBT_DB env enables an isolated tracker db (e.g. a pilot db distinct
    from the production ~/.cache/kb/knowledge.db) without a code change.
    """
    from kb.facade import KnowledgeBase
    from kb.constants import DEFAULT_DB_PATH
    if db_path is None:
        env_db = os.environ.get("KBT_DB")
        db_path = Path(env_db) if env_db else DEFAULT_DB_PATH
    return KnowledgeBase(db_path=db_path)


# ---------------------------------------------------------------------------
# Issue CLI commands
# ---------------------------------------------------------------------------

def cmd_create(args: Any, kb: Any) -> int:
    """kbt create — create a new issue."""
    title = args.title
    issue_type = args.type or "task"
    description = args.description
    priority = args.priority if args.priority is not None else 2
    parent_id = args.parent
    assignee = args.assignee
    project = getattr(args, "project", None)
    tags: list[str] = []

    # design_file: read content from path
    design_content: str | None = None
    if args.design_file:
        p = Path(args.design_file)
        if p.exists():
            design_content = p.read_text()
        else:
            print(f"kbt: design-file not found: {args.design_file}", file=sys.stderr)
            return 1

    # Determine prefix
    if parent_id:
        prefix = parent_id.split("-")[0] if "-" in parent_id else "kb"
    else:
        prefix = args.prefix if hasattr(args, "prefix") and args.prefix else "kb"

    result = kb.issue_create(
        title=title,
        type=issue_type,
        description=description,
        priority=priority,
        parent_id=parent_id,
        design_file=design_content,
        assignee=assignee,
        project=project,
        tags=tags,
        prefix=prefix,
    )

    issue_id = result["id"]

    # Handle --deps=discovered-from:EPIC (and other type:target forms)
    if args.deps:
        for dep_spec in args.deps:
            if ":" in dep_spec:
                dep_type, target = dep_spec.split(":", 1)
            else:
                # bare id → treat as blocks
                dep_type, target = "blocks", dep_spec
            kb.issue_add_dep(issue_id, target, dep_type)

    print(f"Created: {issue_id}")
    return 0


def cmd_show(args: Any, kb: Any) -> int:
    """kbt show ID [--json]"""
    row = kb.issue_get(args.id)
    if row is None:
        print(f"kbt: issue not found: {args.id}", file=sys.stderr)
        return 1

    if args.json:
        out = _project_show(row, kb.conn)
        print(json.dumps([out], indent=2))
    else:
        _print_issue_human(row)
    return 0


def _print_issue_human(row: dict[str, Any]) -> None:
    print(f"[{row['id']}] {row['title']}")
    print(f"  type={row['type']}  status={row['status']}  priority={row['priority']}")
    if row.get("parent_id"):
        print(f"  parent={row['parent_id']}")
    if row.get("assignee"):
        print(f"  assignee={row['assignee']}")
    if row.get("description"):
        print(f"  description: {row['description'][:200]}")
    if row.get("design_file"):
        print(f"  design: {row['design_file'][:200]}")
    if row.get("comments"):
        print(f"  comments ({len(row['comments'])}):")
        for c in row["comments"]:
            print(f"    [{c['created_at'][:10]}] {c['author'] or 'anon'}: {c['body'][:100]}")
    if row.get("deps"):
        print(f"  deps ({len(row['deps'])}):")
        for d in row["deps"]:
            print(f"    {d['type']}: {d['issue_id']} → {d['depends_on_id']}")


def cmd_update(args: Any, kb: Any) -> int:
    """kbt update ID [--claim | --status STATUS | --assignee A | --notes N]"""
    issue_id = args.id

    if args.claim:
        assignee = args.assignee or os.environ.get("USER", "unknown")
        result = kb.issue_claim(issue_id, assignee)
        if result.get("claimed"):
            print(f"Claimed: {issue_id}")
            return 0
        elif result.get("contended"):
            print(f"kbt: claim contended (db locked): {issue_id}", file=sys.stderr)
            return 2
        else:
            print(f"kbt: already claimed or closed: {issue_id}", file=sys.stderr)
            return 1

    if args.status:
        kb.issue_set_status(issue_id, args.status)
        print(f"Updated status: {issue_id} → {args.status}")

    if args.assignee and not args.claim:
        kb.conn.execute(
            "UPDATE issues SET assignee = ?, updated_at = ? WHERE id = ?",
            (args.assignee, _now(), issue_id),
        )
        kb.conn.commit()
        print(f"Updated assignee: {issue_id} → {args.assignee}")

    if args.notes:
        # Append notes to description
        row = kb.issue_get(issue_id)
        if row:
            existing = row.get("description") or ""
            new_desc = existing + "\n\n---\n" + args.notes if existing else args.notes
            kb.conn.execute(
                "UPDATE issues SET description = ?, updated_at = ? WHERE id = ?",
                (new_desc, _now(), issue_id),
            )
            kb.conn.commit()
            print(f"Updated notes: {issue_id}")

    return 0


def cmd_close(args: Any, kb: Any) -> int:
    """kbt close ID [--reason TEXT]"""
    issue_id = args.id
    close_reason = getattr(args, "reason", None)
    result = kb.issue_set_status(issue_id, "closed", close_reason=close_reason)
    print(f"Closed: {result['id']}")
    return 0


def cmd_list(args: Any, kb: Any) -> int:
    """kbt list [--status S] [--parent P] [--json]"""
    status = getattr(args, "status", None)
    parent = getattr(args, "parent", None)
    project = getattr(args, "project", None)
    itype = getattr(args, "type", None)
    assignee = getattr(args, "assignee", None)
    limit = getattr(args, "limit", None)
    as_json = getattr(args, "json", False)

    rows = kb.issue_list(project=project, status=status, parent_id=parent,
                         type=itype, assignee=assignee, limit=limit)

    if as_json:
        # For list, we need the full row data for projection — list() returns summary rows
        # Fetch full rows to get updated_at etc.
        out = []
        for r in rows:
            full = kb.issue_get(r["id"])
            if full:
                item = _project_list_item(full, kb.conn)
                out.append(item)
        print(json.dumps(out, indent=2))
    else:
        for r in rows:
            print(f"[{r['id']}] ({r['status']}) {r['title']}")
    return 0


def cmd_dep_add(args: Any, kb: Any) -> int:
    """kbt dep add ISSUE DEPENDS-ON [--type TYPE]"""
    dep_type = getattr(args, "type", "blocks") or "blocks"
    result = kb.issue_add_dep(args.issue, args.depends_on, dep_type)
    # Usage is `dep add ISSUE DEPENDS-ON`; print it in dependency direction (not a
    # misleading blocker-arrow): for 'blocks', depends_on must finish before issue.
    print(f"Dep added: {args.issue} depends-on {args.depends_on} [{dep_type}] (new={result['is_new']})")
    return 0


def cmd_dep_list(args: Any, kb: Any) -> int:
    """kbt dep list ID [--json]"""
    as_json = getattr(args, "json", False)
    deps = kb.issue_list_deps(args.id)

    if as_json:
        # Emit flat list of dep-target items in bd dep-list element shape
        out: list[dict[str, Any]] = []
        for d in deps.get("outgoing", []):
            target = kb.issue_get(d["id"])
            if target:
                item = _project_dep_list_item(target, kb.conn, d["type"])
                out.append(item)
        for d in deps.get("incoming", []):
            target = kb.issue_get(d["id"])
            if target:
                item = _project_dep_list_item(target, kb.conn, d["type"])
                out.append(item)
        print(json.dumps(out, indent=2))
    else:
        for d in deps.get("outgoing", []):
            print(f"  depends-on [{d['type']}]: {d['id']} ({d['status']}) {d['title']}")
        for d in deps.get("incoming", []):
            print(f"  depended-on-by [{d['type']}]: {d['id']} ({d['status']}) {d['title']}")
    return 0


def cmd_comments_add(args: Any, kb: Any) -> int:
    """kbt comments add ID BODY"""
    author = os.environ.get("USER")
    result = kb.issue_add_comment(args.id, args.body, author=author)
    print(f"Comment added: {result['id']}")
    return 0


def cmd_children(args: Any, kb: Any) -> int:
    """kbt children ID [--json]"""
    as_json = getattr(args, "json", False)
    rows = kb.issue_list(parent_id=args.id)

    if as_json:
        out = []
        for r in rows:
            full = kb.issue_get(r["id"])
            if full:
                item = _project_list_item(full, kb.conn)
                out.append(item)
        print(json.dumps(out, indent=2))
    else:
        for r in rows:
            print(f"[{r['id']}] ({r['status']}) {r['title']}")
    return 0


def cmd_ready(args: Any, kb: Any) -> int:
    """kbt ready [--json]"""
    project = getattr(args, "project", None)
    as_json = getattr(args, "json", False)
    rows = kb.issue_ready(project=project)

    if as_json:
        out = []
        for r in rows:
            # ready() returns summary rows; fetch full for design field
            full = kb.issue_get(r["id"])
            if full:
                item = _project_ready_item(full, kb.conn)
            else:
                item = _project_ready_item(r, kb.conn)
            out.append(item)
        print(json.dumps(out, indent=2))
    else:
        for r in rows:
            print(f"[{r['id']}] ({r['status']}) {r['title']}")
    return 0


def cmd_blocked(args: Any, kb: Any) -> int:
    """kbt blocked [--json]"""
    project = getattr(args, "project", None)
    as_json = getattr(args, "json", False)
    rows = kb.issue_blocked(project=project)

    if as_json:
        out = []
        for r in rows:
            # blocked() returns rows with blocker_ids
            full = kb.issue_get(r["id"])
            if full:
                full["blocker_ids"] = r.get("blocker_ids", [])
                item = _project_blocked_item(full, kb.conn)
            else:
                item = _project_blocked_item(r, kb.conn)
            out.append(item)
        print(json.dumps(out, indent=2))
    else:
        for r in rows:
            blockers = ", ".join(r.get("blocker_ids", []))
            print(f"[{r['id']}] ({r['status']}) {r['title']}  blocked-by: {blockers}")
    return 0


def cmd_search(args: Any, kb: Any) -> int:
    """kbt search QUERY"""
    project = getattr(args, "project", None)
    rows = kb.issue_search(args.query, project=project)
    for r in rows:
        print(f"[{r['id']}] ({r['status']}) {r['title']}  sim={r.get('similarity', 0):.3f}")
    return 0


# ---------------------------------------------------------------------------
# Argparse setup
# ---------------------------------------------------------------------------

def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="kbt",
        description="kb-native issue tracker (bd-compatible interface)",
    )
    parser.add_argument("--db", default=None, help="Override kb database path")
    parser.add_argument("--project", default=None, help="Filter by project")
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p = sub.add_parser("create", help="Create a new issue")
    p.add_argument("--title", required=True)
    p.add_argument("--type", default="task")
    p.add_argument("--description", default=None)
    p.add_argument("--priority", type=int, default=2)
    p.add_argument("--parent", default=None)
    p.add_argument("--design-file", default=None, dest="design_file")
    p.add_argument("--assignee", default=None)
    p.add_argument("--prefix", default="kb")
    p.add_argument("--deps", nargs="+", default=[], metavar="TYPE:TARGET",
                   help="Deps in form type:target, e.g. discovered-from:kb-sg0")

    # show
    p = sub.add_parser("show", help="Show an issue")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")

    # update
    p = sub.add_parser("update", help="Update an issue")
    p.add_argument("id")
    p.add_argument("--claim", action="store_true", help="Atomically claim the issue")
    p.add_argument("--status", default=None)
    p.add_argument("--assignee", default=None)
    p.add_argument("--notes", default=None)

    # close
    p = sub.add_parser("close", help="Close an issue")
    p.add_argument("id")
    p.add_argument("--reason", default=None)

    # list
    p = sub.add_parser("list", help="List issues")
    p.add_argument("--status", default=None)
    p.add_argument("--parent", default=None)
    p.add_argument("--type", default=None)
    p.add_argument("--assignee", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", action="store_true")

    # dep subcommand
    dep_sub = sub.add_parser("dep", help="Dependency management")
    dep_p = dep_sub.add_subparsers(dest="dep_command", required=True)

    p = dep_p.add_parser("add", help="Add a dependency")
    p.add_argument("issue")
    p.add_argument("depends_on")
    p.add_argument("--type", default="blocks")

    p = dep_p.add_parser("list", help="List dependencies")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")

    # comments subcommand
    comments_sub = sub.add_parser("comments", help="Comment management")
    comments_p = comments_sub.add_subparsers(dest="comments_command", required=True)

    p = comments_p.add_parser("add", help="Add a comment")
    p.add_argument("id")
    p.add_argument("body")

    # children
    p = sub.add_parser("children", help="List children of an issue")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")

    # ready
    p = sub.add_parser("ready", help="Show ready issues")
    p.add_argument("--json", action="store_true")

    # blocked
    p = sub.add_parser("blocked", help="Show blocked issues")
    p.add_argument("--json", action="store_true")

    # search
    p = sub.add_parser("search", help="Search issues")
    p.add_argument("query")

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()


def run(argv: list[str] | None = None) -> int:
    """Parse argv and dispatch to the appropriate command.

    Returns an integer exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    db_path = Path(args.db) if getattr(args, "db", None) else None
    kb = _build_kb(db_path)

    command = args.command

    if command == "create":
        return cmd_create(args, kb)
    elif command == "show":
        return cmd_show(args, kb)
    elif command == "update":
        return cmd_update(args, kb)
    elif command == "close":
        return cmd_close(args, kb)
    elif command == "list":
        return cmd_list(args, kb)
    elif command == "dep":
        if args.dep_command == "add":
            return cmd_dep_add(args, kb)
        elif args.dep_command == "list":
            return cmd_dep_list(args, kb)
    elif command == "comments":
        if args.comments_command == "add":
            return cmd_comments_add(args, kb)
    elif command == "children":
        return cmd_children(args, kb)
    elif command == "ready":
        return cmd_ready(args, kb)
    elif command == "blocked":
        return cmd_blocked(args, kb)
    elif command == "search":
        return cmd_search(args, kb)

    print(f"kbt: unknown command: {command}", file=sys.stderr)
    return 1

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
expert-review reads `kbt show <epic> --json` → extracts the `.design` field
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
import shutil
import sys
from pathlib import Path
from typing import Any

from kb.cli import output as _out


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def _default_backend() -> str:
    """Backend when nothing explicit is configured: ALWAYS the kb-native tracker.

    Cutover (kb-sg0.8) is now in effect: the presence of `bd` on PATH or a
    legacy `.beads/` directory no longer routes to dolt. kbt defaults to its
    own kb-native store; un-migrated dolt data is surfaced via a loud warning
    in `resolve_backend` (non-empty `.beads` detected → suggest the agent ask
    the user to run `kbt bead-migrate`) rather than silently keeping the
    project on dolt. An explicit `.beads/config.yaml backend:`, a
    `.kbt/config.toml` marker, the host-wide [tracker] backend, or
    KBT_BACKEND still override this default.
    """
    return "kb"


def _beads_has_data(beads_dir: Path) -> bool:
    """True if a `.beads/` directory holds real tracker data (not just config).

    Signals (any one suffices), all filesystem-only (no `bd`/dolt server call):
      - a `dolt/` subdir (dolt-backed database present), or
      - a non-empty `issues.jsonl` (no-db mode), or
      - a non-empty `backup/issues.jsonl` (JSONL backup of dolt issues).
    """
    try:
        if (beads_dir / "dolt").is_dir():
            return True
        for rel in ("issues.jsonl", "backup/issues.jsonl"):
            p = beads_dir / rel
            if p.is_file() and p.stat().st_size > 0:
                return True
    except OSError:
        pass
    return False


def _walk_up_for(cwd: Path, rel: str) -> Path | None:
    """Nearest existing `<ancestor>/<rel>` walking up from cwd, else None."""
    candidate = cwd.resolve()
    while True:
        p = candidate / rel
        if p.exists():
            return p
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent


def _read_kbt_marker(path: Path) -> str | None:
    """Read [tracker] backend from a per-project .kbt/config.toml. None if absent."""
    try:
        import tomllib  # type: ignore[import]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[import,no-redef]
        except ImportError:
            tomllib = None  # type: ignore[assignment]
    if tomllib is not None:
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            val = (data.get("tracker") or {}).get("backend")
            return str(val).lower() if val else None
        except Exception:
            pass
    # Fallback line scan (no tomllib): backend = "x" under a [tracker] table.
    try:
        in_tracker = False
        for line in path.read_text().splitlines():
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                in_tracker = s == "[tracker]"
                continue
            if in_tracker and s.startswith("backend"):
                _, _, v = s.partition("=")
                v = v.strip().strip("\"'")
                return v.lower() if v else None
    except Exception:
        pass
    return None


def _read_beads_backend(path: Path) -> str | None:
    """Read `backend:` from a legacy .beads/config.yaml. None if no backend key."""
    try:
        import yaml  # type: ignore[import]
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict):
            val = data.get("backend")
            return str(val).lower() if val else None
    except Exception:
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if line.startswith("backend:"):
                    val = line[len("backend:"):].strip().strip("\"'")
                    return val.lower() if val else None
        except Exception:
            pass
    return None


def resolve_backend(cwd: Path | None = None) -> str:
    """Resolve the kbt backend ('kb' or 'dolt').

    Precedence (highest first):
      1. KBT_BACKEND env var (test/override; logged to stderr)
      2. per-project .kbt/config.toml [tracker] backend  (walk up from cwd)
      3. legacy .beads/config.yaml with an EXPLICIT backend: (escape hatch;
         deprecation warn). A .beads WITHOUT an explicit backend no longer
         routes to dolt — it falls through to the kb default (step 5).
      4. host-wide ~/.config/kb/config.toml [tracker] backend
      5. default: kb (the kb-native tracker). If a NON-EMPTY legacy .beads is
         detected here, a loud warning suggests the agent ask the user to run
         `kbt bead-migrate` — kbt does NOT silently keep the project on dolt.

    The two walk-ups are INDEPENDENT and SEQUENTIAL (.kbt to root FIRST, then
    .beads to root) — never interleaved per-directory — so a per-project .kbt
    marker at a HIGHER ancestor always beats a legacy .beads at a LOWER ancestor
    (the B3 per-project isolation guarantee).
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
    cwd = cwd.resolve()

    # 2. per-project .kbt marker — walk to root FIRST
    kbt_marker = _walk_up_for(cwd, ".kbt/config.toml")
    if kbt_marker is not None:
        b = _read_kbt_marker(kbt_marker)
        if b:
            return b

    # 3. legacy .beads/config.yaml with an EXPLICIT backend — escape hatch only.
    #    A .beads WITHOUT an explicit backend: key does NOT route to dolt; it
    #    falls through to the kb default, with the non-empty-.beads warning below.
    beads_cfg = _walk_up_for(cwd, ".beads/config.yaml")
    if beads_cfg is not None:
        b = _read_beads_backend(beads_cfg)
        if b:
            print(
                f"kbt: honoring explicit backend={b!r} in legacy {beads_cfg} — "
                "deprecated; run `kbt bead-migrate` to move to the kb-native tracker",
                file=sys.stderr,
            )
            return b

    # 4. host-wide [tracker] backend
    try:
        from kb.config import load_config
        host = load_config(force_reload=True).tracker_backend
        if host:
            return host.lower()
    except Exception:
        pass

    # 5. default: the kb-native tracker. Warn (don't silently strand) if a
    #    non-empty legacy .beads exists — its issues are NOT visible to kbt.
    b = _default_backend()
    beads_dir = _walk_up_for(cwd, ".beads")
    if b == "kb" and beads_dir is not None and _beads_has_data(beads_dir):
        print(
            f"kbt: WARNING — defaulting to the kb-native tracker, but a non-empty "
            f"legacy beads tracker exists at {beads_dir}; its issues are NOT visible "
            "to kbt. ASK THE USER whether to run `kbt bead-migrate` (migrates the "
            "dolt issues into the kb-native tracker) before relying on tracker state.",
            file=sys.stderr,
        )
    return b


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


def _project_header(row: dict[str, Any]) -> dict[str, Any]:
    """Build the 9-field header block shared by all bd JSON projectors.

    This is the common prefix: id, title, status, priority, issue_type, owner,
    created_by, created_at, updated_at. Callers extend the returned dict with
    their view-specific fields. Does NOT call _dep_counts or _comment_count —
    callers that need those counts add them explicitly.
    """
    return {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "priority": row["priority"],
        "issue_type": row["type"],
        "owner": row.get("assignee"),
        "created_by": row.get("assignee"),  # kb has no created_by column; use assignee
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _project_show(row: dict[str, Any], conn: Any) -> dict[str, Any]:
    """Project a full issue row (from issue_get) to bd show --json shape."""
    issue_id = row["id"]
    dep_count, dep_on_count = _dep_counts(conn, issue_id)
    cmt_count = _comment_count(conn, issue_id)

    out = _project_header(row)
    out.update({
        "dependent_count": dep_on_count,
        "dependency_count": dep_count,
        "comment_count": cmt_count,
    })

    # description + comments: the issue BODY. Omitting them made `kbt show --json`
    # unable to surface the text at all (kb-b6c2b0.8) — include them always.
    if row.get("description"):
        out["description"] = row["description"]
    if row.get("comments"):
        out["comments"] = [
            {"author": c.get("author"), "body": c.get("body"),
             "created_at": c.get("created_at")}
            for c in row["comments"]
        ]

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

    out = _project_header(row)
    out.update({
        "dependency_count": dep_count,
        "dependent_count": dep_on_count,
        "comment_count": cmt_count,
    })
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
    We emit both for full compat. Does NOT call _dep_counts/_comment_count.
    """
    out = _project_header(dep_row)
    out["dependency_type"] = dep_type
    out["type"] = dep_type  # dispatch.md jq reads .type
    if dep_row.get("design_file"):
        out["design"] = dep_row["design_file"]
    return out


def _project_ready_item(row: dict[str, Any], conn: Any) -> dict[str, Any]:
    """Project a ready row to bd ready --json element shape."""
    issue_id = row["id"]
    dep_count, dep_on_count = _dep_counts(conn, issue_id)
    cmt_count = _comment_count(conn, issue_id)

    out = _project_header(row)
    out.update({
        "dependency_count": dep_count,
        "dependent_count": dep_on_count,
        "comment_count": cmt_count,
    })
    # ready items include design when set (live bd does this)
    if row.get("design_file"):
        out["design"] = row["design_file"]
    return out


def _project_blocked_item(row: dict[str, Any], conn: Any) -> dict[str, Any]:
    """Project a blocked row to bd blocked --json element shape.

    Does NOT call _dep_counts or _comment_count — blocked shape uses blocker_ids
    from the blocked() query result directly.
    """
    out = _project_header(row)
    out["blocked_by"] = row.get("blocker_ids", [])
    out["blocked_by_count"] = len(row.get("blocker_ids", []))
    return out


# ---------------------------------------------------------------------------
# Human row formatting — color + priority for users, plain for agents
# ---------------------------------------------------------------------------

_STATUS_COLOR: dict[str, str | None] = {
    "open": None,
    "in_progress": "cyan",
    "blocked": "red",
    "closed": "green",
    "deferred": "blue",
}


def _fmt_row(issue_id: str, status: str, title: str, priority: int | None = None) -> str:
    """Format a single list/row line.

    User mode: id dim, status colored, priority shown as P{n}, truncated to term width.
    Agent mode: plain text, no ANSI, no truncation.
    """
    if _out.AGENT_MODE:
        prio_part = f" P{priority}" if priority is not None else ""
        return f"[{issue_id}] ({status}){prio_part} {title}"
    else:
        color = _STATUS_COLOR.get(status)
        id_part = _out.c(f"[{issue_id}]", "dim")
        status_part = _out.c(f"({status})", color)
        prio_part = f" P{priority}" if priority is not None else ""
        line = f"{id_part} {status_part}{prio_part} {title}"
        return _out.fit_line(line)


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

    # Bulk children (kb-b6c2b0.6): --children-from FILE (one title per line) creates
    # a child task per line under this issue, so epic + N children = ONE invocation.
    child_ids: list[str] = []
    children_from = getattr(args, "children_from", None)
    if children_from:
        cf = Path(children_from)
        if not cf.exists():
            print(f"kbt: --children-from file not found: {children_from}", file=sys.stderr)
            return 1
        for line in cf.read_text().splitlines():
            ct = line.strip()
            if not ct:
                continue
            cres = kb.issue_create(title=ct, type="task", priority=priority,
                                   parent_id=issue_id, project=project, prefix=prefix)
            child_ids.append(cres["id"])

    # Machine-readable id output (kb-b6c2b0.2): --json emits {id, children} so a
    # scripted plan->epic->children flow captures ids without a dot-aware regex.
    if getattr(args, "json", False):
        payload: dict[str, Any] = {"id": issue_id}
        if child_ids:
            payload["children"] = child_ids
        print(json.dumps(payload))
    else:
        print(f"Created: {issue_id}")
        for cid in child_ids:
            print(f"  child: {cid}")
    return 0


def cmd_show(args: Any, kb: Any) -> int:
    """kbt show ID [--json]"""
    row = kb.issue_get(args.id)
    if row is None:
        print(f"kbt: issue not found: {args.id}", file=sys.stderr)
        return 1

    if args.json:
        out = _project_show(row, kb.conn)
        # Emit an OBJECT (bd `show --json` does; a single-element list forced
        # consumers to `.[0].design` instead of `.design`) — kb-b6c2b0.8.
        print(json.dumps(out, indent=2))
    else:
        _print_issue_human(row, kb)
    return 0


def _trunc(text: str, n: int) -> str:
    """Truncate ONLY in user mode — agents have no terminal width and need the
    full body (kb-b6c2b0.8; mirrors _fmt_row's AGENT_MODE gating, which the show
    body had missed)."""
    if _out.AGENT_MODE or len(text) <= n:
        return text
    return text[:n] + "…"


def _print_issue_human(row: dict[str, Any], kb: Any = None) -> None:
    issue_id = row["id"]
    status = row["status"]
    color = _STATUS_COLOR.get(status)
    if _out.AGENT_MODE:
        print(f"[{issue_id}] {row['title']}")
        print(f"  type={row['type']}  status={status}  priority={row['priority']}")
    else:
        print(_out.c(f"[{issue_id}]", "dim") + " " + row["title"])
        print(f"  type={row['type']}  status={_out.c(status, color)}  priority={row['priority']}")
    if row.get("parent_id"):
        print(f"  parent={row['parent_id']}")
    if row.get("assignee"):
        print(f"  assignee={row['assignee']}")
    if row.get("description"):
        print(f"  description: {_trunc(row['description'], 200)}")
    if row.get("design_file"):
        print(f"  design: {_trunc(row['design_file'], 200)}")
    if row.get("comments"):
        print(f"  comments ({len(row['comments'])}):")
        for c in row["comments"]:
            print(f"    [{c['created_at'][:10]}] {c['author'] or 'anon'}: {_trunc(c['body'], 100)}")
    if row.get("deps"):
        print(f"  deps ({len(row['deps'])}):")
        for d in row["deps"]:
            print(f"    {d['type']}: {d['issue_id']} → {d['depends_on_id']}")
    # children inline (kb-b6c2b0.12): show the issue's graph context in ONE read,
    # not a separate `kbt children` call.
    if kb is not None:
        kids = kb.issue_list(parent_id=issue_id)
        if kids:
            print(f"  children ({len(kids)}):")
            for k in kids:
                print(f"    {_fmt_row(k['id'], k['status'], k['title'], k.get('priority'))}")


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
        kb._issues.set_assignee(issue_id, args.assignee)
        print(f"Updated assignee: {issue_id} → {args.assignee}")

    # --priority (kb-b6c2b0.3): re-prioritize after create.
    if getattr(args, "priority", None) is not None:
        kb._issues.set_priority(issue_id, args.priority)
        print(f"Updated priority: {issue_id} → P{args.priority}")

    # --design-file (kb-b6c2b0.3): attach a plan to an epic after creation (reads
    # the file CONTENT into design_file, mirroring create --design-file).
    if getattr(args, "design_file", None):
        p = Path(args.design_file)
        if not p.exists():
            print(f"kbt: design-file not found: {args.design_file}", file=sys.stderr)
            return 1
        kb._issues.set_design_file(issue_id, p.read_text())
        print(f"Updated design-file: {issue_id}")

    # --notes -> a first-class COMMENT (kb-b6c2b0.5). It used to concatenate onto
    # the description with a '---' separator, which (a) bloated the body and (b)
    # had no distinct event stream. Route to issue_add_comment; description unchanged.
    if args.notes:
        kb.issue_add_comment(issue_id, args.notes, author=os.environ.get("USER"))
        print(f"Added note (comment): {issue_id}")

    return 0


def cmd_close(args: Any, kb: Any) -> int:
    """kbt close ID [--reason TEXT]"""
    issue_id = args.id
    close_reason = getattr(args, "reason", None)
    result = kb.issue_set_status(issue_id, "closed", close_reason=close_reason)
    print(f"Closed: {result['id']}")
    return 0


def _current_project_name() -> str | None:
    """Resolve the current project from cwd for default kbt scoping:
    git-root basename, else cwd basename. Issues group by id-prefix
    (kb-, mathlib4-, spec-…) == project, so the cwd dir name is the scope key."""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return os.path.basename(r.stdout.strip())
    except Exception:
        pass
    return os.path.basename(os.getcwd()) or None


def _belongs_to_project(row: Any, name: str | None) -> bool:
    """True if an issue row belongs to project `name`. name falsy => all.

    The project COLUMN is authoritative when set: a non-null project that differs
    from the scope EXCLUDES the row regardless of its id-prefix (kb-b6c2b0.9 — a
    'scrap'-project issue with a kb- id was leaking into the kb scope via the old
    'prefix OR project' rule). The id-prefix is only the FALLBACK grouping key for
    legacy rows whose project column is NULL."""
    if not name:
        return True
    proj = row.get("project")
    if proj:
        return proj == name
    rid = str(row.get("id", ""))
    prefix = rid.split("-", 1)[0] if "-" in rid else rid
    return prefix == name


def _resolve_scope(args: Any) -> str | None:
    """Scope key for list/ready/blocked: None if --all, else --project or cwd."""
    if getattr(args, "all", False):
        return None
    return getattr(args, "project", None) or _current_project_name()


def _hint_if_scoped_empty(scope: str | None, shown: list, all_rows: list) -> None:
    """If project-scoping hid everything, tell the user --all would show more
    (the scope key is the cwd dir name; projects whose dir != id-prefix and that
    don't set the project column won't match without an explicit --project/--all)."""
    if scope and not shown and all_rows:
        print(f"(no issues in project {scope!r}; {len(all_rows)} in other "
              f"projects — use --all or --project NAME)", file=sys.stderr)


def cmd_list(args: Any, kb: Any) -> int:
    """kbt list [--status S] [--parent P] [--json] [--all]"""
    status = getattr(args, "status", None)
    parent = getattr(args, "parent", None)
    itype = getattr(args, "type", None)
    assignee = getattr(args, "assignee", None)
    limit = getattr(args, "limit", None)
    as_json = getattr(args, "json", False)
    scope = _resolve_scope(args)

    # When scoping, fetch unfiltered (the project COLUMN is mostly unset; grouping
    # is by id-prefix) then post-filter, applying the limit after.
    rows = kb.issue_list(project=None, status=status, parent_id=parent,
                         type=itype, assignee=assignee,
                         limit=(None if scope else limit))
    if scope:
        scoped = [r for r in rows if _belongs_to_project(r, scope)]
        _hint_if_scoped_empty(scope, scoped, rows)
        rows = scoped[:limit] if limit else scoped

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
            print(_fmt_row(r["id"], r["status"], r["title"], r.get("priority")))
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
            row = _fmt_row(d["id"], d["status"], d["title"], d.get("priority"))
            print(f"  depends-on [{d['type']}]: {row}")
        for d in deps.get("incoming", []):
            row = _fmt_row(d["id"], d["status"], d["title"], d.get("priority"))
            print(f"  depended-on-by [{d['type']}]: {row}")
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
            print(_fmt_row(r["id"], r["status"], r["title"], r.get("priority")))
    return 0


def cmd_ready(args: Any, kb: Any) -> int:
    """kbt ready [--json] [--all]"""
    as_json = getattr(args, "json", False)
    scope = _resolve_scope(args)
    rows = kb.issue_ready(project=None)
    if scope:
        scoped = [r for r in rows if _belongs_to_project(r, scope)]
        _hint_if_scoped_empty(scope, scoped, rows)
        rows = scoped

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
            print(_fmt_row(r["id"], r["status"], r["title"], r.get("priority")))
    return 0


def cmd_blocked(args: Any, kb: Any) -> int:
    """kbt blocked [--json] [--all]"""
    as_json = getattr(args, "json", False)
    scope = _resolve_scope(args)
    rows = kb.issue_blocked(project=None)
    if scope:
        scoped = [r for r in rows if _belongs_to_project(r, scope)]
        _hint_if_scoped_empty(scope, scoped, rows)
        rows = scoped

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
            base = _fmt_row(r["id"], r["status"], r["title"], r.get("priority"))
            print(f"{base}  blocked-by: {blockers}")
    return 0


def cmd_search(args: Any, kb: Any) -> int:
    """kbt search QUERY"""
    project = getattr(args, "project", None)
    rows = kb.issue_search(args.query, project=project)
    for r in rows:
        sim = r.get("similarity")
        score = f"sim={sim:.3f}" if isinstance(sim, (int, float)) else "fts"
        print(f"[{r['id']}] ({r['status']}) {r['title']}  {score}")
    return 0


def cmd_import(args: Any, kb: Any) -> int:
    """kbt import <export.ndjson> [--project P] [--dry-run] — import a bd NDJSON
    export into the kb-native issues tables (kb-sg0.12). kb-backend only."""
    from kb.bd_import import import_bd_export
    stats = import_bd_export(kb, args.export_json, dry_run=args.dry_run,
                             project=getattr(args, "project", None))
    print(f"issues_imported={stats['issues_imported']} deps_imported={stats['deps_imported']} "
          f"comments_imported={stats['comments_imported']}"
          + (" [dry-run, rolled back]" if args.dry_run else ""))
    return 0


def _find_beads_dir(cwd: Path) -> Path | None:
    """Nearest ancestor `.beads/` directory walking up from cwd, else None."""
    candidate = cwd.resolve()
    while True:
        d = candidate / ".beads"
        if d.is_dir():
            return d
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent


def _write_kbt_marker(project_dir: Path) -> Path:
    """Write the per-project kb-native tracker marker `.kbt/config.toml`."""
    marker_dir = project_dir / ".kbt"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / "config.toml"
    marker.write_text('[tracker]\nbackend = "kb"\n')
    return marker


def cmd_bead_migrate(args: Any, kb: Any) -> int:
    """kbt bead-migrate — one-shot dolt→kb migration (kb-sg0.15).

    Gated, fail-safe order (each step hard-stops the next on failure):
      1. require bd on PATH; independent live count `bd list --all --json`
      2. `bd export` (exit==0); validate every line parses (truncation defense)
      3. import_bd_export into the kb db
      4. issues_imported == export_issue_count (export-internal integrity); warn if
         != live_count (expected: bd export omits closed ephemeral *-wisp-* tasks);
         verify_fidelity empty → else STOP
      5. write per-project .kbt/config.toml marker
      6. archive+commit .beads/ (COMMIT-BEFORE-CLOBBER), then rm -rf  (unless --keep-beads)
    --dry-run runs 1-4 (import rolled back) and skips 5-6.
    """
    import subprocess
    import tempfile

    if not shutil.which("bd"):
        print("kbt bead-migrate: bd not on PATH — nothing to migrate.", file=sys.stderr)
        return 0

    project = getattr(args, "project", None)
    cwd = Path.cwd()

    # 1. independent live count (separate query from the export → catches truncation)
    lc = subprocess.run(["bd", "list", "--all", "--json"], capture_output=True, text=True)
    if lc.returncode != 0:
        print(f"kbt bead-migrate: `bd list` failed (dolt server down?) — aborting, nothing changed.\n{lc.stderr}",
              file=sys.stderr)
        return 1
    try:
        live = json.loads(lc.stdout)
        live_count = len(live) if isinstance(live, list) else 0
    except Exception:
        print("kbt bead-migrate: could not parse `bd list --all --json` — aborting.", file=sys.stderr)
        return 1
    if live_count == 0:
        print("kbt bead-migrate: no issues in dolt — nothing to migrate.", file=sys.stderr)
        return 0

    # 2. export + truncation validation
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as tf:
        export_path = Path(tf.name)
    ex = subprocess.run(["bd", "export", "-o", str(export_path)], capture_output=True, text=True)
    if ex.returncode != 0:
        print(f"kbt bead-migrate: `bd export` failed — aborting, nothing changed.\n{ex.stderr}",
              file=sys.stderr)
        return 1
    export_n = 0
    for line in export_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)  # a truncated final line → JSONDecodeError → abort
        except Exception:
            print("kbt bead-migrate: export has a malformed/truncated line — aborting (no data deleted).",
                  file=sys.stderr)
            return 1
        if rec.get("_type") == "issue":
            export_n += 1

    # 3. import — verify must read PERSISTED rows, so we never use the SAVEPOINT
    #    rollback (dry_run=False). For --dry-run we import into a throwaway temp
    #    kb db and discard it; for the real run we import into the live kb db.
    from kb.bd_import import verify_fidelity, _build_test_kb
    temp_db_dir = None
    if args.dry_run:
        temp_db_dir = tempfile.mkdtemp(prefix="kbt_bead_migrate_dry_")
        target_kb = _build_test_kb(Path(temp_db_dir) / "dry.db")
    else:
        target_kb = kb
    try:
        stats = import_bd_export_safe(target_kb, export_path, dry_run=False, project=project)

        # 4. gates: export-internal completeness (primary), live count advisory cross-check
        imported = stats["issues_imported"]
        # Export-internal check: every issue record in the export was imported.
        # Both counters come from the SAME export file, so a mismatch signals
        # an INSERT collision or a mid-loop exception — genuine truncation is
        # caught earlier (the per-line parse in step 2).
        if imported != export_n:
            print(f"kbt bead-migrate: ABORT — imported {imported} issues but export "
                  f"contained {export_n} issue records. "
                  "The export may contain duplicate ids (INSERT OR REPLACE collapsed "
                  "records) or the import was interrupted. No marker written, .beads/ "
                  "untouched. Re-run with a fresh export.", file=sys.stderr)
            return 1
        # Advisory cross-check: live bd list vs export issue count.
        # bd export legitimately omits closed ephemeral molecule sub-tasks
        # (*-wisp-*) that bd list --all counts, so a delta here is expected
        # and is NOT a hard abort — just warn and continue.
        if imported != live_count:
            delta = live_count - imported
            # Sample up to 5 ids that are in the live list but not in the export
            export_ids: set[str] = set()
            for line in export_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("_type") == "issue":
                        export_ids.add(rec.get("id", ""))
                except Exception:
                    pass
            sample_missing = [item["id"] for item in live
                              if item.get("id") not in export_ids][:5]
            sample_str = ", ".join(sample_missing) if sample_missing else "(none sampled)"
            print(f"kbt bead-migrate: WARNING — live bd count {live_count} != "
                  f"export count {imported} (delta={delta:+d}). "
                  "bd export omits closed ephemeral molecule sub-tasks (*-wisp-*) "
                  "that bd list --all counts; this is expected and safe. "
                  f"Sample ids in live list but not in export: {sample_str}. "
                  "Continuing.", file=sys.stderr)
        diffs = verify_fidelity(target_kb, export_path)
        if diffs:
            print(f"kbt bead-migrate: ABORT — fidelity verify found {len(diffs)} discrepancies. "
                  "No marker written, .beads/ untouched.", file=sys.stderr)
            for d in diffs[:10]:
                print(f"  {d}", file=sys.stderr)
            return 1
    finally:
        if temp_db_dir is not None:
            shutil.rmtree(temp_db_dir, ignore_errors=True)

    if args.dry_run:
        print(f"[dry-run] would migrate {imported} issues (live={live_count}); fidelity OK. "
              "Marker + .beads/ deletion skipped.")
        return 0

    # 5. per-project marker
    beads_dir = _find_beads_dir(cwd)
    project_dir = beads_dir.parent if beads_dir else cwd
    marker = _write_kbt_marker(project_dir)
    print(f"Wrote {marker} (backend=kb).")

    # 6. archive-then-delete .beads/ (COMMIT-BEFORE-CLOBBER)
    if beads_dir is None:
        print(f"Migrated {imported} issues to the kb-native tracker. (no .beads/ to remove)")
        return 0
    if getattr(args, "keep_beads", False):
        print(f"Migrated {imported} issues. --keep-beads: left {beads_dir} in place.")
        return 0
    in_git = subprocess.run(["git", "-C", str(project_dir), "rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True).returncode == 0
    if in_git:
        subprocess.run(["git", "-C", str(project_dir), "add", "-A", ".beads"], check=False)
        # Commit only if `.beads` has staged changes; a clean+committed .beads is
        # ALREADY safe in git history, so deletion is recoverable without a new commit.
        staged = subprocess.run(
            ["git", "-C", str(project_dir), "diff", "--cached", "--quiet", "--", ".beads"]
        ).returncode
        if staged != 0:
            c = subprocess.run(["git", "-C", str(project_dir), "commit", "--no-gpg-sign",
                                "-m", "archive .beads before kbt bead-migrate"],
                               capture_output=True, text=True)
            if c.returncode != 0:
                print(f"kbt bead-migrate: could not commit .beads/ archive — leaving it in place.\n{c.stderr}",
                      file=sys.stderr)
                return 1
    else:
        archive = project_dir / ".beads.migrated.tar"
        a = subprocess.run(["tar", "-cf", str(archive), "-C", str(project_dir), ".beads"],
                           capture_output=True, text=True)
        if a.returncode != 0:
            print(f"kbt bead-migrate: could not archive .beads/ — leaving it in place.\n{a.stderr}",
                  file=sys.stderr)
            return 1
        print(f"Archived {beads_dir} → {archive}")
    shutil.rmtree(beads_dir)
    print(f"Migrated {imported} issues to the kb-native tracker; removed {beads_dir}.")
    return 0


def import_bd_export_safe(kb: Any, export_path: Path, dry_run: bool, project: str | None) -> dict[str, Any]:
    """Thin indirection over bd_import.import_bd_export (kept patchable for tests)."""
    from kb.bd_import import import_bd_export
    return import_bd_export(kb, export_path, dry_run=dry_run, project=project)


def cmd_version(args: Any, kb: Any) -> int:
    """kbt version — print the kb plugin version (+ git sha) so an install/upgrade
    is verifiable (kb-b6c2b0.11 — there was no way to confirm which kbt code runs)."""
    import subprocess
    root = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    ver = "?"
    try:
        with open(os.path.join(root, ".claude-plugin", "plugin.json")) as f:
            ver = json.load(f).get("version", "?")
    except Exception:
        pass
    sha = ""
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            sha = r.stdout.strip()
    except Exception:
        pass
    print(f"kbt (kb plugin) {ver}" + (f" ({sha})" if sha else ""))
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
    parser.add_argument("--project", default=None, help="Filter by project (list/ready/blocked)")
    parser.add_argument("--all", action="store_true",
                        help="list/ready/blocked: show ALL projects "
                             "(default: scope to the current project derived from cwd)")
    sub = parser.add_subparsers(dest="command", required=True)

    # Scope flags also accepted AFTER the subcommand (kb-b6c2b0.10): `kbt list --all`
    # is the universal convention but --all/--project were top-level-only. A shared
    # parent parser with default=SUPPRESS adds them to list/ready/blocked WITHOUT
    # clobbering a value already set by the top-level flags (`kbt --all list`).
    scope_parent = argparse.ArgumentParser(add_help=False)
    scope_parent.add_argument("--all", action="store_true", default=argparse.SUPPRESS,
                              help="show ALL projects (not just the cwd-scoped one)")
    scope_parent.add_argument("--project", default=argparse.SUPPRESS,
                              help="scope to this project")

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
    p.add_argument("--json", action="store_true",
                   help="emit {id[, children]} for scripting (no dot-aware regex)")
    p.add_argument("--children-from", dest="children_from", default=None, metavar="FILE",
                   help="create a child task per non-empty line under the new issue")

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
    p.add_argument("--priority", type=int, default=None, help="re-prioritize")
    p.add_argument("--design-file", dest="design_file", default=None,
                   help="attach a plan file (content) to an epic after creation")
    p.add_argument("--notes", default=None, help="add a note as a comment")

    # close
    p = sub.add_parser("close", help="Close an issue")
    p.add_argument("id")
    p.add_argument("--reason", default=None)

    # list
    p = sub.add_parser("list", help="List issues", parents=[scope_parent])
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
    p = sub.add_parser("ready", help="Show ready issues", parents=[scope_parent])
    p.add_argument("--json", action="store_true")

    # blocked
    p = sub.add_parser("blocked", help="Show blocked issues", parents=[scope_parent])
    p.add_argument("--json", action="store_true")

    # version
    sub.add_parser("version", help="Print the kb plugin version + git sha")

    # search
    p = sub.add_parser("search", help="Search issues")
    p.add_argument("query")

    # import (kb-backend only): bd NDJSON export -> kb issues
    p = sub.add_parser("import", help="Import a bd NDJSON export into kb issues")
    p.add_argument("export_json")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")

    # bead-migrate: one-shot dolt->kb migration
    p = sub.add_parser("bead-migrate",
                       help="One-shot dolt->kb migration (export+import+verify+marker+delete .beads)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")
    p.add_argument("--keep-beads", action="store_true", dest="keep_beads",
                   help="Write the kb-native marker but leave .beads/ in place")

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

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
    elif command == "version":
        return cmd_version(args, kb)
    elif command == "import":
        return cmd_import(args, kb)
    elif command == "bead-migrate":
        return cmd_bead_migrate(args, kb)

    print(f"kbt: unknown command: {command}", file=sys.stderr)
    return 1

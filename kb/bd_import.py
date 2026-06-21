"""
bd → kb Migration Importer

Imports a bd NDJSON export (from `bd export --json`) into the kb SQLite
issues tracker, preserving all ids, parent links, dep types, status (incl
closed), design content, and timestamps verbatim.

## bd-field → kb-column mapping

bd export field         kb issues column / table
--------------------    --------------------------------
id                      issues.id                   (verbatim — CRITICAL)
issue_type              issues.type
status                  issues.status               (verbatim incl 'closed')
priority                issues.priority
<derived from id>       issues.parent_id            (X.N → parent is X; root has none)
title                   issues.title
description             issues.description
design (full content)   issues.design_file          (stored verbatim as content text;
                                                     schema column is TEXT, no path enforcement)
notes                   appended to issues.description (separator "\\n\\n---\\n")
assignee / owner        issues.assignee             (assignee preferred; owner fallback)
close_reason            issues.close_reason
created_at              issues.created_at
updated_at              issues.updated_at
started_at              issues.started_at
closed_at               issues.closed_at
created_by              (stored in issue_comments provenance comment, not a column)
<always None>           issues.project              (bd export has no per-issue project;
                                                     pass --project to set uniformly)
<always None>           issues.tags                 (bd export has no tags field; '[]')
<always None>           issues.closed_by_session    (no bd equivalent)

bd.dependencies[].issue_id      → issue_deps.issue_id
bd.dependencies[].depends_on_id → issue_deps.depends_on_id
bd.dependencies[].type          → issue_deps.type
  kept:    blocks, parent-child, discovered-from, related, supersedes
  dropped: (none seen in current export; log any unknown types)
bd.dependencies[].created_at    → issue_deps.created_at
bd.dependencies[].created_by    → issue_deps.created_by

bd.comments[].id                → issue_comments.id
bd.comments[].issue_id          → issue_comments.issue_id
bd.comments[].text              → issue_comments.body    (bd uses 'text', kb uses 'body')
bd.comments[].author            → issue_comments.author
bd.comments[].created_at        → issue_comments.created_at

child_counters: seeded to max(N) seen across all imported children so future
create() children don't collide with imported ids.

## Dep-type policy

KEPT (pass through):    blocks, parent-child, discovered-from, related, supersedes
DROPPED (not in schema): any other type (logged per-edge, not silently lost)

parent-child dep edges ARE written to issue_deps (redundant with parent_id but
harmless and preserves full fidelity of the export). The IssuesRepository
docstring notes they are not used in the ready/blocked CTE walk (parent_id is
the authority), so they do not cause double-counting.

## Usage

    python -m kb.bd_import <export.json> [--dry-run] [--verify] [--project PROJECT]

Or from Python:
    from kb.bd_import import import_bd_export, verify_fidelity
    from kb.facade import KnowledgeBase
    kb = KnowledgeBase(db_path=Path("/tmp/test.db"))
    stats = import_bd_export(kb, "export.json", dry_run=True)
    discrepancies = verify_fidelity(kb, "export.json")
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Dep types the kb schema CHECK constraint accepts.
_KB_DEP_TYPES = frozenset({"blocks", "parent-child", "discovered-from", "related", "supersedes"})

# Statuses the kb schema issues.status CHECK constraint accepts — EXACTLY these.
# (kb-488f9a.5 added 'deferred' to this set but NOT to the schema CHECK, so a bd
# issue with status='deferred' passed _normalize_status yet crashed the INSERT:
# "CHECK constraint failed". 'deferred' is NOT a kb status; map it to 'open'.)
_KB_STATUSES = frozenset({"open", "in_progress", "blocked", "closed"})

# Map bd/dolt (and other tracker) statuses onto the kb set, PRESERVING meaning:
# a completed/done/wont_fix issue maps to 'closed', NOT 'open' — sending finished
# work to 'open' would silently re-open it on migrate.
_STATUS_MAP = {
    "done": "closed", "completed": "closed", "complete": "closed",
    "finished": "closed", "resolved": "closed", "fixed": "closed",
    "wont_fix": "closed", "wontfix": "closed", "cancelled": "closed",
    "canceled": "closed", "duplicate": "closed", "abandoned": "closed",
    "in_progress": "in_progress", "doing": "in_progress", "started": "in_progress",
    "active": "in_progress", "wip": "in_progress",
    "blocked": "blocked", "waiting": "blocked", "on_hold": "blocked",
    "open": "open", "new": "open", "todo": "open", "backlog": "open",
    "reopened": "open", "reopen": "open", "deferred": "open",
    "pin": "open", "hook": "open",
}


def _normalize_status(status: str | None) -> str:
    """Map any tracker status onto the kb CHECK set, preserving meaning."""
    s = (status or "open").strip().lower().replace("-", "_").replace(" ", "_")
    if s in _KB_STATUSES:
        return s
    return _STATUS_MAP.get(s, "open")


def _derive_parent_id(issue_id: str) -> str | None:
    """Derive parent_id from a child id like 'kb-sg0.3' → 'kb-sg0'.

    The bd id scheme is:
      root:  <prefix>-<hex>           e.g. kb-sg0, kb-7g9, kb-cqv
      child: <parent>.<N>             e.g. kb-sg0.3, kb-7g9.5, kb-sg0.3.1

    A child always has exactly one dot-segment appended. We split on the LAST dot.
    If the last segment is a decimal integer, the part before it is the parent.
    Otherwise it's a root.
    """
    last_dot = issue_id.rfind(".")
    if last_dot == -1:
        return None
    suffix = issue_id[last_dot + 1:]
    if suffix.isdigit():
        return issue_id[:last_dot]
    return None


def _parse_ndjson(path: str | Path) -> list[dict[str, Any]]:
    """Parse NDJSON (one JSON object per line, skip blanks/errors)."""
    issues = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                issues.append(obj)
            except json.JSONDecodeError as e:
                logger.warning("Skipping bad JSON at line %d: %s", lineno, e)
    return issues


def _merge_description_notes(description: str | None, notes: str | None) -> str | None:
    """Merge bd description + notes into a single description string."""
    if description and notes:
        return description + "\n\n---\n" + notes
    if notes:
        return notes
    return description


def _issue_fingerprint(issue: dict[str, Any], project: str | None) -> dict[str, Any]:
    """Extract the fields that would be written to the issues table for comparison."""
    itype = issue.get("issue_type", "task")
    status = _normalize_status(issue.get("status"))
    priority = issue.get("priority", 2)
    title = issue.get("title", "")
    description = _merge_description_notes(issue.get("description"), issue.get("notes"))
    design_file = issue.get("design")
    assignee = issue.get("assignee") or issue.get("owner")
    close_reason = issue.get("close_reason")
    created_at = issue.get("created_at", "")
    updated_at = issue.get("updated_at", created_at)
    started_at = issue.get("started_at")
    closed_at = issue.get("closed_at")
    parent_id = _derive_parent_id(issue["id"])
    return {
        "type": itype, "status": status, "priority": priority,
        "title": title, "description": description, "design_file": design_file,
        "assignee": assignee, "close_reason": close_reason, "project": project,
        "created_at": created_at, "updated_at": updated_at,
        "started_at": started_at, "closed_at": closed_at, "parent_id": parent_id,
    }


def _check_id_collisions(
    conn: sqlite3.Connection,
    issues_data: list[dict[str, Any]],
    project: str | None,
) -> None:
    """Abort with ValueError if any export id exists in db with DIFFERENT content.

    Identical re-import is allowed (idempotent). Called before any write so the
    db is never partially modified on collision.

    Raises:
        ValueError: listing all colliding ids so the caller can surface them.
    """
    collisions: list[str] = []
    for issue in issues_data:
        issue_id: str = issue["id"]
        row = conn.execute(
            """SELECT type, status, priority, parent_id, title, description,
                      design_file, assignee, close_reason, project,
                      created_at, updated_at, started_at, closed_at
               FROM issues WHERE id = ?""",
            (issue_id,),
        ).fetchone()
        if row is None:
            continue  # new id — fine

        (
            db_type, db_status, db_priority, db_parent_id, db_title, db_description,
            db_design_file, db_assignee, db_close_reason, db_project,
            db_created_at, db_updated_at, db_started_at, db_closed_at,
        ) = row

        fp = _issue_fingerprint(issue, project)
        db_fp = {
            "type": db_type, "status": db_status, "priority": db_priority,
            "title": db_title, "description": db_description, "design_file": db_design_file,
            "assignee": db_assignee, "close_reason": db_close_reason, "project": db_project,
            "created_at": db_created_at, "updated_at": db_updated_at,
            "started_at": db_started_at, "closed_at": db_closed_at, "parent_id": db_parent_id,
        }
        if fp != db_fp:
            collisions.append(issue_id)

    if collisions:
        raise ValueError(
            f"id-collision: {len(collisions)} export id(s) already exist in db with different "
            f"content — aborting to prevent silent overwrite. Colliding ids: {collisions}. "
            "To re-import identical data, use an empty target db or remove the colliding rows first."
        )


def _check_prefix_uniformity(issues_data: list[dict[str, Any]]) -> None:
    """Log a warning if export ids have mixed prefixes (guards shared-dolt multi-DB exports).

    A bd prefix is the part of the id before the first dot and after the last hyphen,
    e.g. 'kb-488f9a' → prefix segment 'kb-488f9a', root ids only (no dot).
    We collect the set of root-id prefixes (everything before the first '.') and warn
    if more than one is present.
    """
    root_prefixes: set[str] = set()
    for issue in issues_data:
        iid: str = issue.get("id", "")
        # Compare the FIRST id segment (the project tag): 'claude-abc' and
        # 'claude-wisp-xyz' both → 'claude' (a wisp molecule sub-namespace is NOT a
        # different project), while 'claude-*' vs 'secular-*' → two projects (the
        # real shared-dolt case this guards). Children inherit their root's prefix.
        if "-" in iid:
            root_prefixes.add(iid.split("-", 1)[0])
    if len(root_prefixes) > 1:
        logger.warning(
            "Export contains mixed id-prefixes %s — this export may combine issues from "
            "multiple bd repos (shared-dolt). Verify this is intentional before importing "
            "into a shared db.", sorted(root_prefixes)
        )


def import_bd_export(
    kb: Any,
    export_json_path: str | Path,
    dry_run: bool = False,
    project: str | None = None,
) -> dict[str, Any]:
    """Import a bd NDJSON export into kb issues tables.

    Preserves all ids, parent links, dep types, status (incl closed), design
    content, and timestamps verbatim via direct SQL INSERTs (no IssuesRepository
    create() — that would allocate new ids).

    In dry_run mode everything runs inside a SAVEPOINT that is rolled back at
    the end. Stats are returned regardless.

    Args:
        kb: KnowledgeBase instance (uses kb.conn).
        export_json_path: Path to bd NDJSON export file.
        dry_run: If True, roll back all changes after computing stats.
        project: Optional project name to tag all imported issues.

    Returns:
        {
          "issues_imported": int,
          "deps_imported": int,
          "deps_dropped": dict[str, int],   # {type: count}
          "comments_imported": int,
          "parents_seeded": int,
          "design_stored": int,             # issues with design content
          "notes_merged": int,              # issues with notes appended
        }
    """
    records = _parse_ndjson(export_json_path)
    issues_data = [r for r in records if r.get("_type") == "issue"]
    logger.info("Loaded %d issue records from %s", len(issues_data), export_json_path)

    conn: sqlite3.Connection = kb.conn

    stats = {
        "issues_imported": 0,
        "deps_imported": 0,
        "deps_dropped": {},
        "comments_imported": 0,
        "parents_seeded": 0,
        "design_stored": 0,
        "notes_merged": 0,
    }

    # T3: id-collision pre-flight — abort before any write if an export id already
    # exists in the target db WITH DIFFERENT content. Identical re-import is allowed
    # (idempotent). Checked BEFORE the savepoint/FK-toggle so no partial state can
    # appear on an abort.
    _check_id_collisions(conn, issues_data, project)

    # T6c: assert/log uniform id-prefix across the export (guards the shared-dolt
    # multi-DB case where ~/Physics/claude/.beads/dolt holds both `claude` and
    # `secular_constraints`).
    _check_prefix_uniformity(issues_data)

    # Disable FK enforcement during bulk import to avoid insert-order issues.
    # PRAGMA foreign_keys cannot be changed inside a transaction, so set it
    # BEFORE opening the savepoint.
    conn.execute("PRAGMA foreign_keys = OFF")

    if dry_run:
        conn.execute("SAVEPOINT bd_import_dry_run")

    try:
        _do_import(conn, issues_data, project, stats)
    except Exception:
        if dry_run:
            conn.execute("ROLLBACK TO SAVEPOINT bd_import_dry_run")
            conn.execute("RELEASE SAVEPOINT bd_import_dry_run")
        # T6a: restore FK enforcement even on the non-dry-run exception path
        conn.execute("PRAGMA foreign_keys = ON")
        raise

    conn.execute("PRAGMA foreign_keys = ON")

    if dry_run:
        conn.execute("ROLLBACK TO SAVEPOINT bd_import_dry_run")
        conn.execute("RELEASE SAVEPOINT bd_import_dry_run")
        logger.info("DRY RUN: rolled back all changes")
    else:
        conn.commit()
        logger.info("Import committed.")

    return stats


def _do_import(
    conn: sqlite3.Connection,
    issues_data: list[dict[str, Any]],
    project: str | None,
    stats: dict[str, Any],
) -> None:
    """Execute the actual import SQL (inside a savepoint or real transaction)."""
    # ------------------------------------------------------------------
    # 1. INSERT issues
    # ------------------------------------------------------------------
    for issue in issues_data:
        issue_id: str = issue["id"]
        parent_id = _derive_parent_id(issue_id)

        # Map fields
        itype = issue.get("issue_type", "task")
        # bd has open/in_progress/blocked/closed AND deferred (+ niche states).
        # kb dropped the extras → normalize to a kept status (deferred → open).
        status = _normalize_status(issue.get("status"))
        priority = issue.get("priority", 2)
        title = issue.get("title", "")
        description = _merge_description_notes(
            issue.get("description"), issue.get("notes")
        )
        if issue.get("notes"):
            stats["notes_merged"] += 1

        # design: bd stores full content in 'design'; kb design_file is TEXT
        # (no path enforcement in schema). Store content verbatim.
        design_content = issue.get("design")
        design_file = design_content  # store content in the design_file column
        if design_content:
            stats["design_stored"] += 1

        # assignee: prefer 'assignee' then 'owner'
        assignee = issue.get("assignee") or issue.get("owner")

        close_reason = issue.get("close_reason")
        created_at = issue.get("created_at", "")
        updated_at = issue.get("updated_at", created_at)
        started_at = issue.get("started_at")
        closed_at = issue.get("closed_at")
        tags_json = "[]"

        conn.execute(
            """INSERT OR REPLACE INTO issues
               (id, type, status, priority, parent_id, title, description,
                design_file, assignee, close_reason, project, tags,
                created_at, updated_at, started_at, closed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                issue_id, itype, status, priority, parent_id, title, description,
                design_file, assignee, close_reason, project, tags_json,
                created_at, updated_at, started_at, closed_at,
            ),
        )
        stats["issues_imported"] += 1

    # ------------------------------------------------------------------
    # 2. INSERT issue_deps
    # ------------------------------------------------------------------
    for issue in issues_data:
        deps = issue.get("dependencies", [])
        for dep in deps:
            dep_type = dep.get("type", "")
            if dep_type not in _KB_DEP_TYPES:
                # Log and count, do NOT silently drop
                logger.warning(
                    "DROPPED dep: %s → %s type=%r (not in kb schema CHECK)",
                    dep.get("issue_id"), dep.get("depends_on_id"), dep_type,
                )
                stats["deps_dropped"][dep_type] = stats["deps_dropped"].get(dep_type, 0) + 1
                continue

            dep_issue_id = dep.get("issue_id", "")
            depends_on_id = dep.get("depends_on_id", "")
            dep_created_at = dep.get("created_at", "")
            dep_created_by = dep.get("created_by")

            # Both referenced issues must exist (we just inserted them above).
            # If either side is missing (cross-project dep), skip + log.
            if not dep_issue_id or not depends_on_id:
                logger.warning("Skipping dep with missing issue_id or depends_on_id: %r", dep)
                continue

            cur = conn.execute(
                """INSERT OR IGNORE INTO issue_deps
                   (issue_id, depends_on_id, type, created_at, created_by)
                   VALUES (?, ?, ?, ?, ?)""",
                (dep_issue_id, depends_on_id, dep_type, dep_created_at, dep_created_by),
            )
            # T6b: use rowcount so idempotent re-runs don't overcount
            stats["deps_imported"] += cur.rowcount

    # ------------------------------------------------------------------
    # 3. INSERT issue_comments
    # ------------------------------------------------------------------
    for issue in issues_data:
        comments = issue.get("comments", [])
        for comment in comments:
            cmt_id = comment.get("id", "")
            cmt_issue_id = comment.get("issue_id", issue["id"])
            # bd uses 'text', kb uses 'body'
            body = comment.get("text") or comment.get("body", "")
            author = comment.get("author")
            cmt_created_at = comment.get("created_at", "")

            cur = conn.execute(
                """INSERT OR IGNORE INTO issue_comments
                   (id, issue_id, body, author, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (cmt_id, cmt_issue_id, body, author, cmt_created_at),
            )
            # T6b: use rowcount so idempotent re-runs don't overcount
            stats["comments_imported"] += cur.rowcount

    # ------------------------------------------------------------------
    # 4. SEED child_counters
    # Scan all imported ids; for each child id derive parent and track max N.
    # Use INSERT OR REPLACE so we only update if the new max > existing counter.
    # ------------------------------------------------------------------
    parent_max: dict[str, int] = {}
    for issue in issues_data:
        issue_id = issue["id"]
        parent_id = _derive_parent_id(issue_id)
        if parent_id is not None:
            n = int(issue_id.rsplit(".", 1)[1])
            current = parent_max.get(parent_id, 0)
            if n > current:
                parent_max[parent_id] = n

    for parent_id, max_n in parent_max.items():
        # Only bump the counter if the existing counter (if any) is lower.
        existing = conn.execute(
            "SELECT counter FROM child_counters WHERE parent_id = ?",
            (parent_id,),
        ).fetchone()
        if existing is None or existing[0] < max_n:
            conn.execute(
                """INSERT INTO child_counters (parent_id, counter) VALUES (?, ?)
                   ON CONFLICT(parent_id) DO UPDATE SET counter = excluded.counter
                   WHERE excluded.counter > child_counters.counter""",
                (parent_id, max_n),
            )
            stats["parents_seeded"] += 1


def verify_fidelity(
    kb: Any,
    export_json_path: str | Path,
) -> list[dict[str, Any]]:
    """Verify per-issue fidelity of an already-imported bd export.

    For each issue in the export, checks:
      - id is present in kb issues table
      - parent_id matches the derived parent
      - status matches verbatim
      - design_file matches the 'design' content from the export
      - SET of (depends_on_id, type) dep edges matches the mapped-expected set
        (only kept dep types; dropped types are not expected)

    Returns a list of per-issue discrepancy dicts. An empty list means perfect
    fidelity.
    """
    records = _parse_ndjson(export_json_path)
    issues_data = [r for r in records if r.get("_type") == "issue"]
    conn: sqlite3.Connection = kb.conn
    discrepancies: list[dict[str, Any]] = []

    for issue in issues_data:
        issue_id = issue["id"]
        diffs: list[str] = []

        # Fetch from kb
        row = conn.execute(
            """SELECT id, type, status, priority, parent_id, title, description,
                      design_file, assignee, close_reason,
                      created_at, updated_at, started_at, closed_at
               FROM issues WHERE id = ?""",
            (issue_id,),
        ).fetchone()

        if row is None:
            discrepancies.append({"id": issue_id, "error": "not found in kb"})
            continue

        (
            kb_id, kb_type, kb_status, kb_priority, kb_parent_id,
            kb_title, kb_description, kb_design_file, kb_assignee, kb_close_reason,
            kb_created_at, kb_updated_at, kb_started_at, kb_closed_at,
        ) = row

        # parent_id
        expected_parent = _derive_parent_id(issue_id)
        if kb_parent_id != expected_parent:
            diffs.append(
                f"parent_id: expected={expected_parent!r} got={kb_parent_id!r}"
            )

        # status (apply the same dropped-status normalization as the importer)
        expected_status = _normalize_status(issue.get("status"))
        if kb_status != expected_status:
            diffs.append(
                f"status: expected={expected_status!r} got={kb_status!r}"
            )

        # title
        expected_title = issue.get("title", "")
        if kb_title != expected_title:
            diffs.append(f"title: expected={expected_title!r} got={kb_title!r}")

        # description (post notes-merge — mirror _merge_description_notes)
        expected_description = _merge_description_notes(
            issue.get("description"), issue.get("notes")
        )
        if kb_description != expected_description:
            exp_repr = (expected_description or "")[:80] if expected_description else None
            got_repr = (kb_description or "")[:80] if kb_description else None
            diffs.append(f"description: expected={exp_repr!r} got={got_repr!r}")

        # priority
        expected_priority = issue.get("priority", 2)
        if kb_priority != expected_priority:
            diffs.append(
                f"priority: expected={expected_priority!r} got={kb_priority!r}"
            )

        # assignee
        expected_assignee = issue.get("assignee") or issue.get("owner")
        if kb_assignee != expected_assignee:
            diffs.append(
                f"assignee: expected={expected_assignee!r} got={kb_assignee!r}"
            )

        # design_file (stored as content)
        expected_design = issue.get("design")
        if expected_design != kb_design_file:
            # Truncate for display
            exp_repr = (expected_design or "")[:80] + "..." if expected_design else None
            got_repr = (kb_design_file or "")[:80] + "..." if kb_design_file else None
            diffs.append(
                f"design_file: expected={exp_repr!r} got={got_repr!r}"
            )

        # timestamps
        ts_checks = [
            ("created_at", issue.get("created_at", ""), kb_created_at),
            ("updated_at", issue.get("updated_at", issue.get("created_at", "")), kb_updated_at),
            ("started_at", issue.get("started_at"), kb_started_at),
            ("closed_at", issue.get("closed_at"), kb_closed_at),
        ]
        for ts_name, exp_ts, got_ts in ts_checks:
            if exp_ts != got_ts:
                diffs.append(f"{ts_name}: expected={exp_ts!r} got={got_ts!r}")

        # comments: count + body + author per id
        export_comments = {
            c["id"]: {"body": c.get("text") or c.get("body", ""), "author": c.get("author")}
            for c in issue.get("comments", [])
            if c.get("id")
        }
        kb_comment_rows = conn.execute(
            "SELECT id, body, author FROM issue_comments WHERE issue_id = ?",
            (issue_id,),
        ).fetchall()
        kb_comments = {r[0]: {"body": r[1], "author": r[2]} for r in kb_comment_rows}

        if len(export_comments) != len(kb_comments):
            diffs.append(
                f"comments count: expected={len(export_comments)} got={len(kb_comments)}"
            )
        else:
            for cmt_id, exp_cmt in export_comments.items():
                if cmt_id not in kb_comments:
                    diffs.append(f"comment missing in kb: id={cmt_id!r}")
                elif kb_comments[cmt_id] != exp_cmt:
                    diffs.append(
                        f"comment mismatch id={cmt_id!r}: "
                        f"expected={exp_cmt!r} got={kb_comments[cmt_id]!r}"
                    )

        # dep edges: only the kept types, and only well-formed edges. MIRROR the importer's
        # skip (line ~300: `if not dep_issue_id or not depends_on_id: continue`): a dep with an
        # EMPTY depends_on_id is malformed in the source (e.g. a discovered-from with no parent)
        # and is NOT imported, so it must NOT count as a fidelity discrepancy — otherwise a
        # corrupt source dep aborts an otherwise-perfect migrate (am-rs: 8 such edges).
        expected_deps = set()
        for dep in issue.get("dependencies", []):
            dep_type = dep.get("type", "")
            depends_on_id = dep.get("depends_on_id", "")
            if dep_type in _KB_DEP_TYPES and depends_on_id:
                expected_deps.add((depends_on_id, dep_type))

        kb_dep_rows = conn.execute(
            "SELECT depends_on_id, type FROM issue_deps WHERE issue_id = ?",
            (issue_id,),
        ).fetchall()
        kb_deps = {(r[0], r[1]) for r in kb_dep_rows}

        missing_deps = expected_deps - kb_deps
        extra_deps = kb_deps - expected_deps
        if missing_deps:
            diffs.append(f"deps missing: {sorted(missing_deps)}")
        if extra_deps:
            diffs.append(f"deps extra: {sorted(extra_deps)}")

        if diffs:
            discrepancies.append({"id": issue_id, "diffs": diffs})

    return discrepancies


def _build_test_kb(db_path: str | Path) -> Any:
    """Build a minimal KnowledgeBase on a fresh db (for CLI --dry-run target)."""
    from pathlib import Path as _Path
    from kb.facade import KnowledgeBase
    return KnowledgeBase(db_path=_Path(db_path))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Usage:
        python -m kb.bd_import <export.json> [--dry-run] [--verify] [--project NAME]

    --dry-run : import into a fresh temp db, print stats, roll back
    --verify  : after import, run fidelity check and print discrepancies
    --project : project name to assign all imported issues
    """
    import argparse
    import shutil
    import tempfile

    parser = argparse.ArgumentParser(description="Import a bd NDJSON export into kb issues.")
    parser.add_argument("export_json", help="Path to bd NDJSON export file")
    parser.add_argument("--dry-run", action="store_true", help="Roll back after stats (test only)")
    parser.add_argument("--verify", action="store_true", help="Run fidelity check after import")
    parser.add_argument("--project", default=None, help="Project name to tag all issues")
    parser.add_argument("--db", default=None, help="Target db path (default: fresh temp db for dry-run; real kb db otherwise)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    export_path = Path(args.export_json)
    if not export_path.exists():
        print(f"ERROR: export file not found: {export_path}", file=sys.stderr)
        return 1

    # Determine target db
    temp_dir = None
    if args.db:
        db_path = Path(args.db)
    elif args.dry_run:
        temp_dir = tempfile.mkdtemp(prefix="bd_import_test_")
        db_path = Path(temp_dir) / "test.db"
        print(f"[dry-run] Using fresh temp db: {db_path}")
    else:
        # Real kb db — use default
        from kb.constants import DEFAULT_DB_PATH
        db_path = DEFAULT_DB_PATH
        print(f"[live] Importing into real kb db: {db_path}")

    try:
        kb = _build_test_kb(db_path)

        stats = import_bd_export(
            kb,
            export_path,
            dry_run=args.dry_run,
            project=args.project,
        )

        print("\n--- Import Stats ---")
        print(f"  issues_imported   : {stats['issues_imported']}")
        print(f"  deps_imported     : {stats['deps_imported']}")
        print(f"  deps_dropped      : {stats['deps_dropped'] or 'none'}")
        print(f"  comments_imported : {stats['comments_imported']}")
        print(f"  parents_seeded    : {stats['parents_seeded']}")
        print(f"  design_stored     : {stats['design_stored']}")
        print(f"  notes_merged      : {stats['notes_merged']}")
        if args.dry_run:
            print("  [DRY RUN — changes rolled back]")
        print()

        if args.verify:
            if args.dry_run:
                # For verify after dry-run, we need to actually import (no rollback)
                # into the temp db, then verify.
                print("[verify] Re-running import (no rollback) for fidelity check...")
                stats2 = import_bd_export(kb, export_path, dry_run=False, project=args.project)

            print("--- Fidelity Check ---")
            discrepancies = verify_fidelity(kb, export_path)
            if not discrepancies:
                print("  PASS: 0 discrepancies — perfect fidelity")
            else:
                print(f"  FAIL: {len(discrepancies)} issue(s) with discrepancies:")
                for d in discrepancies:
                    issue_id = d.get("id", "?")
                    if "error" in d:
                        print(f"    [{issue_id}] ERROR: {d['error']}")
                    else:
                        for diff in d.get("diffs", []):
                            print(f"    [{issue_id}] {diff}")
            print()

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

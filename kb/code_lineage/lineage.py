"""lineage.py — function-version records and lineage clustering (Tier A, exact-hash).

A FunctionVersion captures one (function, provenance) content state.
cluster_lineages groups a list of FunctionVersions by structural_hash,
accumulating names_seen/paths_seen/commits_seen per logical function.

Schema for the `function_versions` table (created by migrate()):
  id             INTEGER PRIMARY KEY AUTOINCREMENT
  structural_hash TEXT NOT NULL       -- SHA-256 of normalized AST
  name           TEXT NOT NULL        -- original function name at this version
  qualname       TEXT                 -- dotted qualname if available
  file           TEXT NOT NULL        -- source file path
  line           INTEGER              -- 1-based line number in file
  provenance     TEXT NOT NULL        -- arbitrary tag: commit sha, branch, 'HEAD', etc.
  project        TEXT                 -- optional project label
  added_at       TEXT                 -- ISO8601 UTC timestamp

Lineage clustering is done entirely in Python (no additional SQL tables needed
for Tier A); the DB table is the canonical fact store.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FunctionVersion:
    """One (function-content, provenance) observation."""

    structural_hash: str
    name: str
    file: str
    provenance: str                 # commit SHA, branch name, 'HEAD', etc.
    qualname: str = ""
    line: int = 0
    project: str = ""
    added_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class FunctionLineage:
    """A logical function (identified by structural_hash) with its provenance history."""

    structural_hash: str
    names_seen: list[str] = field(default_factory=list)
    paths_seen: list[str] = field(default_factory=list)
    commits_seen: list[str] = field(default_factory=list)
    versions: list[FunctionVersion] = field(default_factory=list)

    @property
    def canonical_name(self) -> str:
        """Most-recently-seen name (last version added)."""
        return self.names_seen[-1] if self.names_seen else ""

    def _add(self, name: str, path: str, commit: str) -> None:
        if name not in self.names_seen:
            self.names_seen.append(name)
        if path not in self.paths_seen:
            self.paths_seen.append(path)
        if commit not in self.commits_seen:
            self.commits_seen.append(commit)


# ---------------------------------------------------------------------------
# Clustering (in-memory, Tier A exact hash)
# ---------------------------------------------------------------------------


def cluster_lineages(
    versions: list[FunctionVersion],
) -> dict[str, FunctionLineage]:
    """Group FunctionVersions by exact structural_hash into logical lineages.

    Returns a dict keyed by structural_hash.
    Order of versions matters only for canonical_name (last-seen wins).
    Delete-then-readd is transparent: the same hash reappears and is merged
    into the same lineage — recognized as the same logical function regardless
    of gap in the commit timeline.
    """
    lineages: dict[str, FunctionLineage] = {}
    for fv in versions:
        lin = lineages.setdefault(fv.structural_hash, FunctionLineage(fv.structural_hash))
        lin._add(fv.name, fv.file, fv.provenance)
        lin.versions.append(fv)
    return lineages


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS function_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    structural_hash TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    qualname        TEXT    NOT NULL DEFAULT '',
    file            TEXT    NOT NULL,
    line            INTEGER NOT NULL DEFAULT 0,
    provenance      TEXT    NOT NULL,
    project         TEXT    NOT NULL DEFAULT '',
    added_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fv_hash
    ON function_versions (structural_hash);

CREATE INDEX IF NOT EXISTS idx_fv_provenance
    ON function_versions (provenance);
"""


def migrate(conn: sqlite3.Connection) -> None:
    """Create the function_versions table (idempotent)."""
    conn.executescript(_CREATE_TABLE)
    conn.commit()


def insert_version(conn: sqlite3.Connection, fv: FunctionVersion) -> int:
    """Insert a FunctionVersion row and return the new row id."""
    cur = conn.execute(
        """
        INSERT INTO function_versions
            (structural_hash, name, qualname, file, line, provenance, project, added_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fv.structural_hash,
            fv.name,
            fv.qualname,
            fv.file,
            fv.line,
            fv.provenance,
            fv.project,
            fv.added_at,
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def load_versions(
    conn: sqlite3.Connection,
    project: Optional[str] = None,
    provenance: Optional[str] = None,
) -> list[FunctionVersion]:
    """Load FunctionVersion rows, optionally filtered by project / provenance."""
    clauses = []
    params: list = []
    if project:
        clauses.append("project = ?")
        params.append(project)
    if provenance:
        clauses.append("provenance = ?")
        params.append(provenance)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT structural_hash, name, qualname, file, line, provenance, project, added_at "
        f"FROM function_versions {where} ORDER BY id",
        params,
    ).fetchall()

    return [
        FunctionVersion(
            structural_hash=row[0],
            name=row[1],
            qualname=row[2],
            file=row[3],
            line=row[4],
            provenance=row[5],
            project=row[6],
            added_at=row[7],
        )
        for row in rows
    ]


def load_lineages(
    conn: sqlite3.Connection,
    project: Optional[str] = None,
) -> dict[str, FunctionLineage]:
    """Load all versions and cluster them into lineages."""
    versions = load_versions(conn, project=project)
    return cluster_lineages(versions)

"""
Database Connection

Manages SQLite connection with sqlite-vec extension.
"""

import sqlite3
from pathlib import Path
from collections.abc import Sequence

import sqlite_vec  # type: ignore[import-untyped]

from ..constants import DEFAULT_DB_PATH, DEFAULT_EMBEDDING_DIM


class DatabaseConnection:
    """Manages SQLite connection with sqlite-vec extension."""

    db_path: Path
    embedding_dim: int
    conn: sqlite3.Connection

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim

        # busy_timeout makes EVERY statement BLOCK on a locked db — waiting (with
        # SQLite's own internal backoff) up to this long for the lock to free,
        # instead of failing the caller with "database is locked". This is the
        # universal "retry-with-backoff, block until the write completes" for ALL
        # writes (no method re-run, so embed-before-commit paths never re-embed).
        # 120s absorbs heavy multi-writer contention (agents + hooks + server on
        # one db). The connect() timeout matches so the initial handshake waits too.
        _BUSY_TIMEOUT_MS = 120_000
        self.conn = sqlite3.connect(str(self.db_path), timeout=_BUSY_TIMEOUT_MS / 1000)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        _ = self.conn.execute("PRAGMA foreign_keys = ON")
        _ = self.conn.execute("PRAGMA journal_mode = WAL")
        _ = self.conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        # Bound WAL growth: an un-checkpointed WAL (observed at 442 MB) is the
        # symptom of checkpoint starvation under sustained writers. Keep the
        # autocheckpoint active so no single writer's log grows unbounded.
        _ = self.conn.execute("PRAGMA wal_autocheckpoint = 1000")

    def execute(self, sql: str, params: Sequence[object] | None = None) -> sqlite3.Cursor:
        """Execute SQL with optional parameters."""
        if params is None:
            return self.conn.execute(sql)
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq: Sequence[Sequence[object]]) -> sqlite3.Cursor:
        """Execute SQL for multiple parameter sets."""
        return self.conn.executemany(sql, params_seq)

    def executescript(self, sql: str) -> sqlite3.Cursor:
        """Execute multiple SQL statements."""
        return self.conn.executescript(sql)

    def commit(self) -> None:
        """Commit current transaction."""
        self.conn.commit()

    def rollback(self) -> None:
        """Rollback current transaction."""
        self.conn.rollback()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

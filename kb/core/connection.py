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

        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        _ = self.conn.execute("PRAGMA foreign_keys = ON")
        _ = self.conn.execute("PRAGMA journal_mode = WAL")
        _ = self.conn.execute("PRAGMA busy_timeout = 30000")

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

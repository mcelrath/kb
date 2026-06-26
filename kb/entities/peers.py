"""
Peers Repository — the decentralized federated-kb peer registry (epic kb-907fc8 P1).

Each node holds its OWN peer list (no central registry). A peer row carries the
peer's kb-server URL, its embedding identity (model_id/dim/quant/instruction_prefix
— the vector-comparability gate), a bearer token to authenticate to it, and a
last_seen reachability stamp for offline-tolerant fan-out.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from .base import EntityRepository


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PeersRepository(EntityRepository):
    """Registry of federated-kb peers."""

    def __init__(self, conn: sqlite3.Connection):
        super().__init__(conn)

    def add(
        self,
        url: str,
        label: str | None = None,
        model_id: str | None = None,
        dim: int | None = None,
        quant: str | None = None,
        instruction_prefix: str | None = None,
        token: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Add or update a peer (keyed by url). Returns {'url', 'is_new'}."""
        now = _now()
        existing = self.conn.execute("SELECT url FROM peers WHERE url = ?", (url,)).fetchone()
        if existing:
            self.conn.execute(
                """UPDATE peers SET label=?, model_id=?, dim=?, quant=?, instruction_prefix=?,
                   token=?, enabled=?, updated_at=? WHERE url=?""",
                (label, model_id, dim, quant, instruction_prefix, token, int(enabled), now, url),
            )
            self.conn.commit()
            return {"url": url, "is_new": False}
        self.conn.execute(
            """INSERT INTO peers
               (url, label, model_id, dim, quant, instruction_prefix, token, enabled, added_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (url, label, model_id, dim, quant, instruction_prefix, token, int(enabled), now, now),
        )
        self.conn.commit()
        return {"url": url, "is_new": True}

    def remove(self, url: str) -> bool:
        cur = self.conn.execute("DELETE FROM peers WHERE url = ?", (url,))
        self.conn.commit()
        return cur.rowcount > 0

    def get(self, url: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM peers WHERE url = ?", (url,)).fetchone()
        return dict(row) if row else None

    def list(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM peers"
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY added_at"
        return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def set_enabled(self, url: str, enabled: bool) -> None:
        self.conn.execute(
            "UPDATE peers SET enabled = ?, updated_at = ? WHERE url = ?",
            (int(enabled), _now(), url),
        )
        self.conn.commit()

    def set_last_seen(self, url: str, ts: float) -> None:
        self.conn.execute(
            "UPDATE peers SET last_seen = ?, updated_at = ? WHERE url = ?",
            (ts, _now(), url),
        )
        self.conn.commit()

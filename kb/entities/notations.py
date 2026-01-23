"""
Notations Repository

Manages notation tracking and history.
"""

import re
import sqlite3
import uuid
from datetime import datetime

from .base import EntityRepository
from ..constants import NOTATION_DOMAINS


# Greek letter mappings (lowercase and uppercase)
GREEK_LETTERS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ",
    "Epsilon": "Ε", "Zeta": "Ζ", "Eta": "Η", "Theta": "Θ",
    "Iota": "Ι", "Kappa": "Κ", "Lambda": "Λ", "Mu": "Μ",
    "Nu": "Ν", "Xi": "Ξ", "Omicron": "Ο", "Pi": "Π",
    "Rho": "Ρ", "Sigma": "Σ", "Tau": "Τ", "Upsilon": "Υ",
    "Phi": "Φ", "Chi": "Χ", "Psi": "Ψ", "Omega": "Ω",
}
GREEK_TO_LATIN = {v: k for k, v in GREEK_LETTERS.items()}


class NotationsRepository(EntityRepository):
    """Repository for notation management."""

    def add(
        self,
        symbol: str,
        meaning: str,
        project: str | None = None,
        domain: str = "general",
    ) -> str:
        """Add a new notation to track."""
        if domain not in NOTATION_DOMAINS:
            raise ValueError(f"Invalid domain: {domain}. Must be one of {NOTATION_DOMAINS}")

        # Check for existing notation with same symbol in project
        existing = self.conn.execute(
            "SELECT id FROM notations WHERE current_symbol = ? AND (project = ? OR (project IS NULL AND ? IS NULL))",
            (symbol, project, project)
        ).fetchone()

        if existing:
            raise ValueError(f"Notation '{symbol}' already exists for project '{project}'. Use update to change it.")

        notation_id = f"notation-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()

        _ = self.conn.execute("""
            INSERT INTO notations (id, current_symbol, meaning, project, domain, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (notation_id, symbol, meaning, project, domain, now, now))

        self.conn.commit()
        return notation_id

    def update(
        self,
        new_symbol: str,
        old_symbol: str | None = None,
        notation_id: str | None = None,
        meaning: str | None = None,
        reason: str | None = None,
        project: str | None = None,
    ) -> str:
        """Update a notation symbol, recording the change in history."""
        # Find the notation
        row: sqlite3.Row | None
        if notation_id:
            row = self.conn.execute(
                "SELECT * FROM notations WHERE id = ?",
                (notation_id,)
            ).fetchone()
        elif old_symbol:
            sql = "SELECT * FROM notations WHERE current_symbol = ?"
            params: list[object] = [old_symbol]
            if project:
                sql += " AND project = ?"
                params.append(project)
            row = self.conn.execute(sql, params).fetchone()
        else:
            raise ValueError("Must provide either old_symbol or notation_id")

        if not row:
            raise ValueError(f"Notation not found: {old_symbol or notation_id}")

        found_id: str = row["id"]
        old_symbol_actual = row["current_symbol"]
        now = datetime.now().isoformat()

        # Record history
        history_id = f"notation-hist-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        _ = self.conn.execute("""
            INSERT INTO notation_history (id, notation_id, old_symbol, new_symbol, reason, changed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (history_id, found_id, old_symbol_actual, new_symbol, reason, now))

        # Update notation
        if meaning:
            _ = self.conn.execute("""
                UPDATE notations SET current_symbol = ?, meaning = ?, updated_at = ? WHERE id = ?
            """, (new_symbol, meaning, now, found_id))
        else:
            _ = self.conn.execute("""
                UPDATE notations SET current_symbol = ?, updated_at = ? WHERE id = ?
            """, (new_symbol, now, found_id))

        self.conn.commit()
        return found_id

    def get(self, notation_id: str) -> dict[str, object] | None:
        """Get a notation by ID, including its history."""
        row = self.conn.execute(
            "SELECT * FROM notations WHERE id = ?",
            (notation_id,)
        ).fetchone()

        if not row:
            return None

        history = self.conn.execute(
            "SELECT * FROM notation_history WHERE notation_id = ? ORDER BY changed_at DESC",
            (notation_id,)
        ).fetchall()

        return {
            "id": row["id"],
            "current_symbol": row["current_symbol"],
            "meaning": row["meaning"],
            "project": row["project"],
            "domain": row["domain"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "history": [
                {
                    "old_symbol": h["old_symbol"],
                    "new_symbol": h["new_symbol"],
                    "reason": h["reason"],
                    "changed_at": h["changed_at"],
                }
                for h in history
            ],
        }

    def _expand_greek(self, query: str) -> list[str]:
        """Expand query to include Greek letter variants."""
        variants = [query]

        # Latin name -> Greek letter (word boundary match)
        for latin, greek in GREEK_LETTERS.items():
            pattern = rf'\b{latin}\b'
            if re.search(pattern, query, re.IGNORECASE):
                variants.append(re.sub(pattern, greek, query, flags=re.IGNORECASE))

        # Greek letter -> Latin name
        for greek, latin in GREEK_TO_LATIN.items():
            if greek in query:
                variants.append(query.replace(greek, latin))

        return list(set(variants))

    def search(
        self,
        query: str,
        project: str | None = None,
        domain: str | None = None,
    ) -> list[dict[str, object]]:
        """Search notations by symbol or meaning."""
        query_variants = self._expand_greek(query)

        conditions = []
        params: list[object] = []
        for variant in query_variants:
            escaped = variant.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            conditions.append("(current_symbol LIKE ? ESCAPE '\\' OR meaning LIKE ? ESCAPE '\\')")
            params.extend([pattern, pattern])

        sql = f"SELECT * FROM notations WHERE ({' OR '.join(conditions)})"

        if project:
            sql += " AND project = ?"
            params.append(project)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)

        sql += " ORDER BY updated_at DESC"

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "current_symbol": row["current_symbol"],
                "meaning": row["meaning"],
                "project": row["project"],
                "domain": row["domain"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def list(
        self,
        project: str | None = None,
        domain: str | None = None,
    ) -> list[dict[str, object]]:
        """List all notations with optional filters."""
        sql = "SELECT * FROM notations WHERE 1=1"
        params: list[object] = []

        if project:
            sql += " AND project = ?"
            params.append(project)
        if domain:
            sql += " AND domain = ?"
            params.append(domain)

        sql += " ORDER BY current_symbol"

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "current_symbol": row["current_symbol"],
                "meaning": row["meaning"],
                "project": row["project"],
                "domain": row["domain"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def history(self, notation_id: str) -> list[dict[str, object]]:
        """Get the change history for a notation."""
        rows = self.conn.execute(
            "SELECT * FROM notation_history WHERE notation_id = ? ORDER BY changed_at DESC",
            (notation_id,)
        ).fetchall()

        return [
            {
                "id": row["id"],
                "old_symbol": row["old_symbol"],
                "new_symbol": row["new_symbol"],
                "reason": row["reason"],
                "changed_at": row["changed_at"],
            }
            for row in rows
        ]

    def delete(self, notation_id: str) -> bool:
        """Delete a notation and its history."""
        _ = self.conn.execute("DELETE FROM notation_history WHERE notation_id = ?", (notation_id,))
        cursor = self.conn.execute("DELETE FROM notations WHERE id = ?", (notation_id,))
        self.conn.commit()
        return cursor.rowcount > 0

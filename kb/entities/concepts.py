"""
Concept Register Repository

Manages active conclusions for thinking-block prefill.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any

from .base import EntityRepository
from ..core.embedding import EmbeddingService


CONCEPT_STATUSES = ("open", "active", "verified", "superseded", "procedure")


class ConceptRepository(EntityRepository):
    """Repository for concept register management."""

    embedding_service: EmbeddingService

    def __init__(self, conn: sqlite3.Connection, embedding_service: EmbeddingService):
        super().__init__(conn)
        self.embedding_service = embedding_service

    def add(
        self,
        domain: str,
        claim: str,
        status: str = "open",
        correct_framing: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Add a concept to the register.

        Returns dict with 'id', 'is_new'.
        """
        if status not in CONCEPT_STATUSES:
            raise ValueError(f"status must be one of {CONCEPT_STATUSES}")

        cid = f"con-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.utcnow().isoformat()

        self.conn.execute(
            """INSERT INTO concepts
               (id, domain, status, claim, correct_framing, project, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, domain, status, claim, correct_framing, project, now, now),
        )

        embedding = self.embedding_service.embed(claim)
        self.conn.execute(
            "INSERT OR REPLACE INTO concepts_vec (id, embedding) VALUES (?, ?)",
            (cid, embedding),
        )
        self.conn.commit()
        return {"id": cid, "is_new": True}

    def get(self, concept_id: str) -> dict[str, Any] | None:
        """Get a concept with its linked theorems and findings."""
        row = self.conn.execute(
            "SELECT id, domain, status, claim, correct_framing, supersedes_id, project, created_at, updated_at FROM concepts WHERE id = ?",
            (concept_id,),
        ).fetchone()
        if not row:
            return None
        result = dict(zip(["id","domain","status","claim","correct_framing","supersedes_id","project","created_at","updated_at"], row))

        theorem_rows = self.conn.execute(
            """SELECT t.id, t.lean_name, t.name, t.statement_pure, t.statement, ct.role
               FROM concept_theorems ct
               JOIN lean_theorems t ON t.id = ct.theorem_id
               WHERE ct.concept_id = ?""",
            (concept_id,),
        ).fetchall()
        result["theorems"] = [
            dict(zip(["id","lean_name","name","statement_pure","statement","role"], r))
            for r in theorem_rows
        ]

        finding_rows = self.conn.execute(
            """SELECT f.id, f.content, f.type, cf.role
               FROM concept_findings cf
               JOIN findings f ON f.id = cf.finding_id
               WHERE cf.concept_id = ?""",
            (concept_id,),
        ).fetchall()
        result["findings"] = [
            dict(zip(["id","content","type","role"], r))
            for r in finding_rows
        ]
        return result

    def list(
        self,
        domain: str | None = None,
        status: str | None = None,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """List concepts with optional filters."""
        conditions = []
        params: list[Any] = []
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if project:
            conditions.append("project = ?")
            params.append(project)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        rows = self.conn.execute(
            f"SELECT id, domain, status, claim, correct_framing, project FROM concepts {where} ORDER BY status, domain",
            params,
        ).fetchall()
        return [dict(zip(["id","domain","status","claim","correct_framing","project"], r)) for r in rows]

    def search(self, query: str, project: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Semantic search over concepts."""
        embedding = self.embedding_service.embed(query)
        params: list[Any] = []
        where = ""
        if project:
            where = "WHERE c.project = ?"
            params.append(project)

        rows = self.conn.execute(
            f"""SELECT v.id, v.distance
                FROM concepts_vec v
                JOIN concepts c ON c.id = v.id
                {where}
                ORDER BY v.distance
                LIMIT ?""",
            (*params, limit),
        ).fetchall()

        results = []
        for cid, dist in rows:
            row = self.conn.execute(
                "SELECT id, domain, status, claim, correct_framing, project FROM concepts WHERE id = ?",
                (cid,),
            ).fetchone()
            if row:
                r = dict(zip(["id","domain","status","claim","correct_framing","project"], row))
                r["similarity"] = round(1 - (dist ** 2) / 2, 4)
                results.append(r)
        return results

    def verify(self, concept_id: str) -> None:
        """Mark a concept as verified."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE concepts SET status = 'verified', updated_at = ? WHERE id = ?",
            (now, concept_id),
        )
        self.conn.commit()

    def supersede(self, concept_id: str, new_claim: str, domain: str | None = None, project: str | None = None) -> dict[str, Any]:
        """Supersede a concept with a new one."""
        old = self.conn.execute(
            "SELECT domain, project FROM concepts WHERE id = ?", (concept_id,)
        ).fetchone()
        if not old:
            raise ValueError(f"Concept {concept_id} not found")

        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE concepts SET status = 'superseded', updated_at = ? WHERE id = ?",
            (now, concept_id),
        )

        new_domain = domain or old[0]
        new_project = project or old[1]
        new_id = f"con-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.conn.execute(
            """INSERT INTO concepts
               (id, domain, status, claim, supersedes_id, project, created_at, updated_at)
               VALUES (?, ?, 'active', ?, ?, ?, ?, ?)""",
            (new_id, new_domain, new_claim, concept_id, new_project, now, now),
        )
        embedding = self.embedding_service.embed(new_claim)
        self.conn.execute(
            "INSERT OR REPLACE INTO concepts_vec (id, embedding) VALUES (?, ?)",
            (new_id, embedding),
        )
        self.conn.commit()
        return {"id": new_id, "supersedes": concept_id}

    def link_theorem(self, concept_id: str, theorem_id: str, role: str = "evidence") -> None:
        """Link a theorem to a concept."""
        valid_roles = ("evidence", "depends_on", "motivates")
        if role not in valid_roles:
            raise ValueError(f"role must be one of {valid_roles}")
        self.conn.execute(
            "INSERT OR IGNORE INTO concept_theorems (concept_id, theorem_id, role) VALUES (?, ?, ?)",
            (concept_id, theorem_id, role),
        )
        self.conn.commit()

    def link_finding(self, concept_id: str, finding_id: str, role: str = "evidence") -> None:
        """Link a finding to a concept."""
        self.conn.execute(
            "INSERT OR IGNORE INTO concept_findings (concept_id, finding_id, role) VALUES (?, ?, ?)",
            (concept_id, finding_id, role),
        )
        self.conn.commit()

    def render_register(
        self,
        project: str | None = None,
        max_tokens: int = 600,
        framework_hints: list[str] | None = None,
        technique_hints: list[str] | None = None,
    ) -> str:
        """Render the concept register for thinking-block prefill.

        Produces a compact text block in pure math language.
        Domain labels are NOT rendered (routing only).
        """
        concepts = self.list(project=project)
        active = [c for c in concepts if c["status"] in ("active", "verified")]
        procedures = [c for c in concepts if c["status"] == "procedure"]
        open_c = [c for c in concepts if c["status"] == "open"]

        lines = []
        if framework_hints:
            lines.append(f"[FRAMEWORK: {', '.join(framework_hints)}]")
        if technique_hints:
            lines.append(f"[TECHNIQUE: {', '.join(technique_hints)}]")
        if lines:
            lines.append("")

        if active:
            lines.append("VERIFIED:" if any(c["status"] == "verified" for c in active) else "ACTIVE:")
            for c in active:
                theorem_ids = self.conn.execute(
                    "SELECT theorem_id FROM concept_theorems WHERE concept_id = ?", (c["id"],)
                ).fetchall()
                refs = " " + " ".join(f"[{r[0]}]" for r in theorem_ids) if theorem_ids else ""
                prefix = "VERIFIED" if c["status"] == "verified" else "active"
                label = c["claim"].split(":")[0] if ":" in c["claim"] else c["id"]
                claim_body = c["claim"]
                lines.append(f"- {claim_body}{refs}")

        if open_c:
            lines.append("")
            lines.append("ACTIVE:")
            for c in open_c:
                lines.append(f"- {c['claim']} (partial)")

        if procedures:
            lines.append("")
            lines.append("PROCEDURES:")
            for c in procedures:
                lines.append(f"- {c['claim']}")

        text = "\n".join(lines)

        estimated_tokens = len(text.split()) * 1.3
        if estimated_tokens > max_tokens:
            active_only = [c for c in concepts if c["status"] == "verified"]
            lines2 = []
            if framework_hints:
                lines2.append(f"[FRAMEWORK: {', '.join(framework_hints)}]")
            if active_only:
                lines2.append("VERIFIED:")
                for c in active_only:
                    lines2.append(f"- {c['claim']}")
            text = "\n".join(lines2)

        return text

    def count(self, project: str | None = None) -> int:
        if project:
            return self.conn.execute(
                "SELECT COUNT(*) FROM concepts WHERE project = ?", (project,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

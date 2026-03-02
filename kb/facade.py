"""
KnowledgeBase Facade

Main entry point that delegates to specialized modules while maintaining
backward compatibility with the original API.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

from .constants import (
    DEFAULT_DB_PATH,
    DEFAULT_EMBEDDING_URL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LLM_URL,
    FINDING_TYPES,
    NOTATION_DOMAINS,
    GREEK_MEANINGS,
)
from .validation import validate_finding_content, validate_tags, serialize_f32
from .core.connection import DatabaseConnection
from .core.schema import init_schema
from .core.embedding import EmbeddingService
from .llm.client import LLMClient
from .llm.analysis import ContentAnalyzer
from .search.hybrid import HybridSearch
from .entities.scripts import ScriptsRepository
from .entities.notations import NotationsRepository
from .entities.errors import ErrorsRepository
from .entities.documents import DocumentsRepository


class KnowledgeBase:
    """SQLite + sqlite-vec knowledge base for findings.

    This is a facade that delegates to specialized modules:
    - EmbeddingService for vector embeddings
    - LLMClient for LLM completions
    - ContentAnalyzer for content analysis
    - HybridSearch for search operations
    - Entity repositories for scripts, notations, errors, documents
    """

    TEMPLATES = {
        "computation_result": {
            "format": "Computed {claim} using {method}. Result: {result}",
            "required": ["claim", "method", "result"],
            "optional": ["script"],
            "default_type": "success",
        },
        "failed_approach": {
            "format": "Attempted {approach} for {goal}. Failed because: {reason}",
            "required": ["approach", "goal", "reason"],
            "optional": ["error"],
            "default_type": "failure",
        },
        "structural_discovery": {
            "format": "{structure} has {property}. This implies {implication}",
            "required": ["structure", "property", "implication"],
            "optional": ["proof_sketch"],
            "default_type": "discovery",
        },
        "verification": {
            "format": "Verified {claim} by {method}. {outcome}",
            "required": ["claim", "method", "outcome"],
            "optional": ["script", "tolerance"],
            "default_type": "success",
        },
        "hypothesis": {
            "format": "Hypothesis: {hypothesis}. Motivation: {motivation}. Status: {status}",
            "required": ["hypothesis", "motivation", "status"],
            "optional": ["tests_needed"],
            "default_type": "experiment",
        },
    }

    db_path: Path
    embedding_url: str
    embedding_dim: int
    conn: sqlite3.Connection

    # Subsystems
    _embedding: EmbeddingService
    _llm: LLMClient
    _analyzer: ContentAnalyzer
    _search: HybridSearch
    _scripts: ScriptsRepository
    _notations: NotationsRepository
    _errors: ErrorsRepository
    _documents: DocumentsRepository

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        embedding_url: str = DEFAULT_EMBEDDING_URL,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_url = embedding_url
        self.embedding_dim = embedding_dim

        # Initialize database connection
        db_conn = DatabaseConnection(db_path, embedding_dim)
        self.conn = db_conn.conn
        init_schema(self.conn, embedding_dim)

        # Initialize subsystems
        self._embedding = EmbeddingService(embedding_url, embedding_dim)
        self._llm = LLMClient(DEFAULT_LLM_URL)
        self._analyzer = ContentAnalyzer(self._llm)
        self._search = HybridSearch(
            self.conn,
            self._embedding,
            expand_query=lambda q, p, v: self._llm.expand_query(q, p, embedding_url, v)
        )

        # Initialize entity repositories
        self._scripts = ScriptsRepository(
            self.conn,
            self._embedding,
            finding_exists=lambda fid: self.get(fid) is not None
        )
        self._notations = NotationsRepository(self.conn)
        self._errors = ErrorsRepository(
            self.conn,
            normalize_signature=self._analyzer.normalize_error_signature
        )
        self._documents = DocumentsRepository(self.conn)

    # =========================================================================
    # Backward-compatible methods delegating to subsystems
    # =========================================================================

    def _embed(self, text: str) -> bytes:
        """Generate embedding for text."""
        return self._embedding.embed(text)

    def _llm_complete(self, *args: Any, **kwargs: Any) -> str | None:
        """Generic LLM completion."""
        return self._llm.complete(*args, **kwargs)

    def _extract_text_from_json(self, text: str, keys: list[str] | None = None) -> str:
        """Extract text from JSON-wrapped responses."""
        return self._llm.extract_text_from_json(text, keys)

    def _generate_summary(self, content: str, evidence: str | None = None) -> str | None:
        """Generate summary for finding."""
        return self._analyzer.generate_summary(content, evidence)

    def expand_query(self, query: str, project: str | None = None, verbose: bool = False) -> str:
        """Expand search query using LLM."""
        return self._llm.expand_query(query, project, self.embedding_url, verbose)

    def suggest_tags(self, content: str, project: str | None = None) -> list[str]:
        """Suggest tags for content."""
        existing_tags: set[str] = set()
        if project:
            rows = self.conn.execute(
                "SELECT DISTINCT tags FROM findings WHERE project = ? AND tags IS NOT NULL",
                (project,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT tags FROM findings WHERE tags IS NOT NULL"
            ).fetchall()
        for row in rows:
            if row[0]:
                try:
                    existing_tags.update(json.loads(row[0]))
                except json.JSONDecodeError:
                    pass
        return self._analyzer.suggest_tags(content, existing_tags)

    def classify_finding_type(self, content: str) -> str:
        """Classify finding type."""
        return self._analyzer.classify_type(content)

    def normalize_error_signature(self, error_text: str) -> str:
        """Normalize error signature."""
        return self._analyzer.normalize_error_signature(error_text)

    def detect_duplicates(self, content: str, project: str | None = None, threshold: float = 0.85) -> list[dict[str, Any]]:
        """Check for duplicate findings."""
        similar = self.search(content, limit=5, project=project)
        if not similar:
            return []
        candidates = [s for s in similar if s.get("similarity", 0) >= threshold]
        if not candidates:
            return []

        duplicates: list[dict[str, Any]] = []
        for candidate in candidates[:3]:
            prompt = f"""Are these two findings saying essentially the same thing? Return JSON: {{"answer": true}} or {{"answer": false}}

Finding 1: {content[:300]}

Finding 2: {candidate['content'][:300]}"""
            result = self._llm.complete(prompt, max_tokens=100, temperature=0.1, json_mode=True)
            if result:
                try:
                    data = json.loads(result)
                    answer = data.get("answer", False)
                    is_duplicate = answer is True or str(answer).upper() in ("YES", "TRUE")
                except json.JSONDecodeError:
                    is_duplicate = "YES" in result.upper()
                if is_duplicate:
                    duplicates.append(candidate)
        return duplicates

    def validate_finding_llm(self, content: str, tags: list[str] | None = None) -> dict[str, Any]:
        """LLM-based validation of finding."""
        return self._analyzer.validate_finding(content, tags)

    def suggest_finding_fix(self, content: str, issues: list[str]) -> str | None:
        """Suggest fix for finding."""
        return self._analyzer.suggest_fix(content, issues)

    def summarize_evidence(self, evidence: str, max_length: int = 200) -> str:
        """Summarize evidence."""
        return self._analyzer.summarize_evidence(evidence, max_length)

    def detect_notations(self, content: str, project: str | None = None) -> list[dict[str, Any]]:
        """Detect notations in content."""
        # Get existing symbols for project
        sql = "SELECT current_symbol FROM notations"
        params: list[Any] = []
        if project:
            sql += " WHERE project = ?"
            params = [project]
        existing = {row[0] for row in self.conn.execute(sql, params).fetchall()}
        return self._analyzer.detect_notations(content, existing)

    def extract_claims(self, text: str) -> list[str]:
        """Extract claims from text."""
        return self._analyzer.extract_claims(text)

    # =========================================================================
    # Search methods
    # =========================================================================

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Search findings."""
        return self._search.search(query, **kwargs)

    def related(self, finding_id: str, limit: int = 5, include_superseded: bool = False) -> list[dict[str, Any]]:
        """Find related findings."""
        return self._search.related(finding_id, limit, include_superseded)

    # =========================================================================
    # Findings CRUD (kept in facade for now due to complexity)
    # =========================================================================

    def _validate_tags(self, tags: list[str] | None) -> list[str]:
        """Validate tags."""
        return validate_tags(tags)

    def check_duplicate(
        self,
        content: str,
        evidence: str | None = None,
        threshold: float = 0.85,
    ) -> tuple[bool, dict[str, Any] | None, bytes]:
        """Check if similar finding exists."""
        text = content + " " + (evidence or "")
        embedding = self._embed(text)

        rows = self.conn.execute("""
            SELECT f.*, v.distance
            FROM findings f
            JOIN findings_vec v ON f.id = v.id
            WHERE v.embedding MATCH ?
            AND k = 3
            AND f.status = 'current'
        """, (embedding,)).fetchall()

        for row in rows:
            similarity = 1 - (row["distance"] ** 2) / 2
            if similarity >= threshold:
                return True, {
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "similarity": similarity,
                }, embedding
        return False, None, embedding

    def add(
        self,
        content: str,
        finding_type: str | None = None,
        project: str | None = None,
        sprint: str | None = None,
        tags: list[str] | None = None,
        evidence: str | None = None,
        check_duplicate: bool = True,
        duplicate_threshold: float = 0.85,
        check_contradictions: bool = True,
        auto_tag: bool = True,
        auto_classify: bool = True,
        auto_summarize_evidence: bool = True,
        max_evidence_length: int = 500,
    ) -> dict[str, Any]:
        """Add a new finding."""
        result: dict[str, Any] = {
            "id": None,
            "tags_suggested": False,
            "tags_missing_warning": False,
            "type_suggested": False,
            "type_mismatch_warning": None,
            "evidence_summarized": False,
            "cross_refs": None,
            "notations_detected": None,
            "content_warnings": [],
            "contradictions": [],
        }

        original_tags_missing = not tags

        if finding_type is None:
            if auto_classify:
                finding_type = self.classify_finding_type(content)
                result["type_suggested"] = True
            else:
                finding_type = "discovery"
        elif auto_classify:
            suggested_type = self.classify_finding_type(content)
            if suggested_type != finding_type:
                result["type_mismatch_warning"] = f"Provided type '{finding_type}' differs from suggested '{suggested_type}'"

        if finding_type not in FINDING_TYPES:
            raise ValueError(f"Invalid type: {finding_type}. Must be one of {FINDING_TYPES}")

        result["evidence_missing_warning"] = finding_type == "failure" and not evidence

        if original_tags_missing:
            result["tags_missing_warning"] = True

        if not tags and auto_tag:
            suggested = self.suggest_tags(content, project=project)
            if suggested:
                tags = suggested
                result["tags_suggested"] = True

        result["content_warnings"] = validate_finding_content(content, tags)

        original_evidence = evidence
        if evidence and auto_summarize_evidence and len(evidence) > max_evidence_length:
            evidence = self.summarize_evidence(evidence, max_length=max_evidence_length)
            result["evidence_summarized"] = True

        embedding: bytes | None = None
        if check_duplicate:
            is_dup, existing, embedding = self.check_duplicate(content, original_evidence or evidence, duplicate_threshold)
            if is_dup and existing:
                raise ValueError(
                    f"Similar finding already exists (similarity: {existing['similarity']:.2f}):\n"
                    f"  ID: {existing['id']}\n"
                    f"  Content: {existing['content'][:100]}...\n"
                    f"Use check_duplicate=False to add anyway, or kb_correct to update."
                )

        if check_contradictions:
            contradictions = self.check_contradictions(content, project=project)
            if contradictions:
                result["contradictions"] = contradictions

        finding_id = f"kb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()
        tags = self._validate_tags(tags)
        tags_json = json.dumps(tags)

        summary = self._generate_summary(content, evidence)

        if embedding is None:
            embedding = self._embed(content + " " + (evidence or ""))

        try:
            _ = self.conn.execute("""
                INSERT INTO findings (id, type, status, project, sprint, tags, content, summary, evidence, created_at, updated_at)
                VALUES (?, ?, 'current', ?, ?, ?, ?, ?, ?, ?, ?)
            """, (finding_id, finding_type, project, sprint, tags_json, content, summary, evidence, now, now))

            _ = self.conn.execute(
                "INSERT INTO findings_vec (id, embedding) VALUES (?, ?)",
                (finding_id, embedding)
            )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        result["id"] = finding_id

        result["cross_refs"] = self.suggest_cross_references(finding_id, content, project=project)
        result["notations_detected"] = self.detect_notations(content, project=project)

        return result

    def correct(
        self,
        supersedes_id: str,
        content: str,
        reason: Optional[str] = None,
        evidence: Optional[str] = None,
    ) -> dict[str, Any]:
        """Correct an existing finding by superseding it."""
        old = self.conn.execute(
            "SELECT id, project, sprint, tags FROM findings WHERE id = ?",
            (supersedes_id,)
        ).fetchone()

        if not old:
            raise ValueError(f"Finding not found: {supersedes_id}")

        impacted = self.find_citing_findings(supersedes_id)

        finding_id = f"kb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()

        full_content = content
        if reason:
            full_content = f"[CORRECTION: {reason}] {content}"

        summary = self._generate_summary(content, evidence)
        embedding = self._embed(full_content + " " + (evidence or ""))

        try:
            _ = self.conn.execute(
                "UPDATE findings SET status = 'superseded', updated_at = ? WHERE id = ?",
                (now, supersedes_id)
            )

            _ = self.conn.execute("""
                INSERT INTO findings (id, type, status, supersedes_id, project, sprint, tags, content, summary, evidence, created_at, updated_at)
                VALUES (?, 'correction', 'current', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (finding_id, supersedes_id, old["project"], old["sprint"], old["tags"], full_content, summary, evidence, now, now))

            _ = self.conn.execute(
                "INSERT INTO findings_vec (id, embedding) VALUES (?, ?)",
                (finding_id, embedding)
            )

            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

        return {"id": finding_id, "impacted_findings": impacted}

    def get(self, finding_id: str) -> Optional[dict[str, Any]]:
        """Get a finding by ID."""
        row = self.conn.execute(
            "SELECT * FROM findings WHERE id = ?", (finding_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "type": row["type"],
            "status": row["status"],
            "supersedes_id": row["supersedes_id"],
            "project": row["project"],
            "sprint": row["sprint"],
            "tags": json.loads(row["tags"] or "[]"),
            "content": row["content"],
            "summary": row["summary"],
            "evidence": row["evidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_findings(
        self,
        project: str | None = None,
        sprint: str | None = None,
        finding_type: str | None = None,
        include_superseded: bool = False,
        limit: int = 20,
        offset: int = 0,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """List findings with optional filters."""
        sql = "SELECT * FROM findings WHERE 1=1"
        params: list[Any] = []

        if not include_superseded:
            sql += " AND status = 'current'"
        if project:
            sql += " AND project = ?"
            params.append(project)
        if sprint:
            sql += " AND sprint = ?"
            params.append(sprint)
        if finding_type:
            sql += " AND type = ?"
            params.append(finding_type)
        if tag:
            sql += " AND EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)"
            params.append(tag)

        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "id": row["id"],
                "type": row["type"],
                "status": row["status"],
                "project": row["project"],
                "sprint": row["sprint"],
                "tags": json.loads(row["tags"] or "[]"),
                "summary": row["summary"],
                "content": row["content"][:200],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, finding_id: str) -> bool:
        """Delete a finding."""
        _ = self.conn.execute("DELETE FROM findings_vec WHERE id = ?", (finding_id,))
        cursor = self.conn.execute("DELETE FROM findings WHERE id = ?", (finding_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        """Get database statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
        by_type = {}
        for row in self.conn.execute("SELECT type, COUNT(*) FROM findings GROUP BY type").fetchall():
            by_type[row[0]] = row[1]
        by_project = {}
        for row in self.conn.execute("SELECT project, COUNT(*) FROM findings WHERE project IS NOT NULL GROUP BY project").fetchall():
            by_project[row[0]] = row[1]
        by_status = {}
        for row in self.conn.execute("SELECT status, COUNT(*) FROM findings GROUP BY status").fetchall():
            by_status[row[0]] = row[1]

        current = by_status.get("current", 0)
        superseded = by_status.get("superseded", 0)

        return {
            "db_path": str(self.db_path),
            "total": total,
            "current": current,
            "superseded": superseded,
            "by_type": by_type,
            "by_project": by_project,
        }

    # =========================================================================
    # Cross-references and related
    # =========================================================================

    def suggest_cross_references(self, finding_id: str, content: str, project: str | None = None) -> dict[str, Any]:
        """Suggest related findings, scripts, and docs."""
        suggestions: dict[str, Any] = {"findings": [], "scripts": [], "docs": []}

        related = self.search(content, limit=5, project=project)
        for r in related:
            if r["id"] != finding_id and r.get("similarity", 0) > 0.6:
                suggestions["findings"].append({
                    "id": r["id"],
                    "content": r["content"][:100],
                    "similarity": r.get("similarity", 0)
                })

        scripts = self.script_search(content, project=project, limit=3)
        for s in scripts:
            if s.get("similarity", 0) > 0.5:
                suggestions["scripts"].append({
                    "id": s["id"],
                    "filename": s["filename"],
                    "purpose": s.get("purpose", "")[:100],
                    "similarity": s.get("similarity", 0)
                })

        docs = self.doc_search(content, project=project)
        for d in docs[:3]:
            suggestions["docs"].append({
                "id": d["id"],
                "title": d["title"],
            })

        return suggestions

    def find_citing_findings(self, finding_id: str) -> list[dict[str, Any]]:
        """Find findings that reference another finding."""
        pattern = f"%{finding_id}%"
        rows = self.conn.execute(
            "SELECT id, content FROM findings WHERE content LIKE ? AND id != ?",
            (pattern, finding_id)
        ).fetchall()
        return [{"id": row["id"], "content": row["content"][:100]} for row in rows]

    def check_contradictions(self, content: str, project: str | None = None) -> list[dict[str, Any]]:
        """Check if content contradicts existing findings."""
        similar = self.search(content, limit=5, project=project)
        contradictions: list[dict[str, Any]] = []

        for s in similar:
            if s.get("similarity", 0) < 0.5:
                continue
            prompt = f"""Do these two findings contradict each other? Return JSON: {{"contradicts": true/false, "reason": "..."}}

Finding 1: {content[:300]}

Finding 2: {s['content'][:300]}"""

            result = self._llm.complete(prompt, max_tokens=150, temperature=0.2, json_mode=True)
            if result:
                try:
                    data = json.loads(result)
                    if data.get("contradicts") is True:
                        contradictions.append({
                            "finding_id": s["id"],
                            "content": s["content"][:100],
                            "reason": data.get("reason", ""),
                        })
                except json.JSONDecodeError:
                    pass

        return contradictions

    # =========================================================================
    # Scripts delegation
    # =========================================================================

    def script_add(self, path: str, purpose: str, **kwargs: Any) -> str:
        """Add a script."""
        result = self._scripts.add(path, purpose, **kwargs)
        return str(result["id"])

    def script_get(self, script_id: str) -> dict[str, Any] | None:
        """Get a script."""
        return self._scripts.get(script_id)

    def script_search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Search scripts."""
        return self._scripts.search(query, **kwargs)

    def script_list(self, **kwargs: Any) -> list[dict[str, Any]]:
        """List scripts."""
        return self._scripts.list(**kwargs)

    def script_link_finding(self, finding_id: str, script_id: str, relationship: str = "generated_by") -> None:
        """Link finding to script."""
        self._scripts.link_finding(finding_id, script_id, relationship)

    def script_findings(self, script_id: str) -> list[dict[str, Any]]:
        """Get findings for script."""
        return self._scripts.get_findings(script_id)

    def finding_scripts(self, finding_id: str) -> list[dict[str, Any]]:
        """Get scripts for finding."""
        return self._scripts.get_for_finding(finding_id)

    def script_delete(self, script_id: str) -> bool:
        """Delete a script."""
        return self._scripts.delete(script_id)

    # =========================================================================
    # Notations delegation
    # =========================================================================

    def notation_add(self, symbol: str, meaning: str, **kwargs: Any) -> str:
        """Add a notation."""
        return self._notations.add(symbol, meaning, **kwargs)

    def notation_update(self, new_symbol: str, **kwargs: Any) -> str:
        """Update a notation."""
        return self._notations.update(new_symbol, **kwargs)

    def notation_get(self, notation_id: str) -> dict[str, Any] | None:
        """Get a notation."""
        return self._notations.get(notation_id)

    def notation_search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Search notations."""
        return self._notations.search(query, **kwargs)

    def notation_list(self, **kwargs: Any) -> list[dict[str, Any]]:
        """List notations."""
        return self._notations.list(**kwargs)

    def notation_history(self, notation_id: str) -> list[dict[str, Any]]:
        """Get notation history."""
        return self._notations.history(notation_id)

    def notation_delete(self, notation_id: str) -> bool:
        """Delete a notation."""
        return self._notations.delete(notation_id)

    # =========================================================================
    # Errors delegation
    # =========================================================================

    def error_add(self, signature: str, **kwargs: Any) -> dict[str, Any]:
        """Add an error."""
        return self._errors.add(signature, **kwargs)

    def error_link(self, error_id: str, finding_id: str, verified: bool = False) -> bool:
        """Link error to solution."""
        return self._errors.link(error_id, finding_id, verified)

    def error_verify(self, error_id: str, finding_id: str) -> bool:
        """Verify error solution."""
        return self._errors.verify(error_id, finding_id)

    def error_get(self, error_id: str) -> dict[str, Any] | None:
        """Get an error."""
        return self._errors.get(error_id)

    def error_search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Search errors."""
        return self._errors.search(query, **kwargs)

    def error_list(self, **kwargs: Any) -> list[dict[str, Any]]:
        """List errors."""
        return self._errors.list(**kwargs)

    def error_solutions(self, error_id: str) -> list[dict[str, Any]]:
        """Get solutions for error."""
        return self._errors.get_solutions(error_id)

    def solution_errors(self, finding_id: str) -> list[dict[str, Any]]:
        """Get errors for solution."""
        return self._errors.get_errors_for_solution(finding_id)

    def error_delete(self, error_id: str) -> bool:
        """Delete an error."""
        return self._errors.delete(error_id)

    # =========================================================================
    # Documents delegation
    # =========================================================================

    def doc_add(self, title: str, doc_type: str, **kwargs: Any) -> str:
        """Add a document."""
        return self._documents.add(title, doc_type, **kwargs)

    def doc_get(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document."""
        return self._documents.get(doc_id)

    def doc_list(self, **kwargs: Any) -> list[dict[str, Any]]:
        """List documents."""
        return self._documents.list(**kwargs)

    def doc_search(self, query: str, project: str | None = None) -> list[dict[str, Any]]:
        """Search documents."""
        return self._documents.search(query, project)

    def doc_supersede(self, doc_id: str, new_doc_id: str) -> bool:
        """Supersede a document."""
        return self._documents.supersede(doc_id, new_doc_id)

    def doc_cite(self, finding_id: str, doc_id: str, **kwargs: Any) -> bool:
        """Cite a document."""
        return self._documents.cite(finding_id, doc_id, **kwargs)

    def doc_citations(self, doc_id: str) -> list[dict[str, Any]]:
        """Get citations for document."""
        return self._documents.get_citations(doc_id)

    def finding_docs(self, finding_id: str) -> list[dict[str, Any]]:
        """Get documents for finding."""
        return self._documents.get_docs_for_finding(finding_id)

    def doc_delete(self, doc_id: str) -> bool:
        """Delete a document."""
        return self._documents.delete(doc_id)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_all_tags(self) -> list[str]:
        """Get all unique tags."""
        tags: set[str] = set()
        for row in self.conn.execute("SELECT DISTINCT tags FROM findings WHERE tags IS NOT NULL").fetchall():
            if row[0]:
                try:
                    tags.update(json.loads(row[0]))
                except json.JSONDecodeError:
                    pass
        return sorted(tags)

    # =========================================================================
    # Additional methods required by MCP
    # =========================================================================

    def ask(
        self,
        question: str,
        project: str | None = None,
        limit: int = 10,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Answer a natural language question using KB findings.

        Searches for relevant findings and uses LLM to synthesize an answer.

        Args:
            question: Natural language question
            project: Filter to specific project
            limit: Max findings to consider
            verbose: Include search results in response

        Returns:
            dict with 'answer', 'sources', and optionally 'search_results'
        """
        results = self.search(
            query=question,
            project=project,
            limit=limit,
            expand=True,
            deprioritize_index=True,
            exclude_corrections=True,
        )

        if not results:
            return {
                "answer": "No relevant findings found in the knowledge base.",
                "sources": [],
                "search_results": [] if verbose else None,
            }

        context_parts = []
        sources = []
        for i, r in enumerate(results, 1):
            sim = r.get("similarity", r.get("relevance", 0))
            finding_text = f"[{i}] ({r['type']}, {r['project'] or 'no project'}, sim={sim:.2f})\n{r['content']}"
            if r.get("evidence"):
                finding_text += f"\nEvidence: {r['evidence'][:200]}"
            context_parts.append(finding_text)
            sources.append({
                "id": r["id"],
                "type": r["type"],
                "project": r["project"],
                "similarity": sim,
                "content": r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"],
            })

        context = "\n\n".join(context_parts)

        system_prompt = """You are a knowledge base assistant. Answer questions based ONLY on the provided findings.
Output JSON: {"answer": "..."}.
- Cite findings by their number [1], [2], etc.
- If findings conflict, explain the discrepancy
- If findings don't fully answer the question, say what's missing
- Be concise but thorough"""

        prompt = f"""QUESTION: {question}

RELEVANT FINDINGS:
{context}

Answer the question based on these findings. Cite sources by number."""

        answer = self._llm.complete(
            prompt,
            max_tokens=500,
            temperature=0.3,
            system_prompt=system_prompt,
            timeout=60,
        )

        if answer:
            answer = self._llm.extract_text_from_json(answer, keys=["answer", "response", "text"])

        if not answer:
            answer = "LLM unavailable. Top findings:\n\n" + "\n\n".join(
                f"- {r['content'][:200]}" for r in results[:3]
            )

        result: dict[str, Any] = {
            "answer": answer,
            "sources": sources,
        }
        if verbose:
            result["search_results"] = results

        return result

    def bulk_add_tags(self, finding_ids: list[str], tags: list[str]) -> dict[str, Any]:
        """Add tags to multiple findings.

        Args:
            finding_ids: List of finding IDs to update
            tags: Tags to add (merged with existing tags)

        Returns:
            dict with 'updated' count and 'skipped' (not found) count
        """
        tags = self._validate_tags(tags)
        if not tags:
            return {"updated": 0, "skipped": len(finding_ids), "error": "No valid tags provided"}

        updated = 0
        skipped = 0
        now = datetime.now().isoformat()

        for fid in finding_ids:
            row = self.conn.execute(
                "SELECT tags FROM findings WHERE id = ?", (fid,)
            ).fetchone()

            if not row:
                skipped += 1
                continue

            existing = json.loads(row["tags"]) if row["tags"] else []
            merged = list(set(existing + tags))

            self.conn.execute(
                "UPDATE findings SET tags = ?, updated_at = ? WHERE id = ?",
                (json.dumps(merged), now, fid)
            )
            updated += 1

        self.conn.commit()
        return {"updated": updated, "skipped": skipped}

    def consolidate_cluster(
        self,
        finding_ids: list[str],
        summary: str,
        reason: str,
        finding_type: str = "discovery",
        tags: list[str] | None = None,
        evidence: str | None = None,
    ) -> dict[str, Any]:
        """Supersede multiple findings with a single consolidated finding.

        Args:
            finding_ids: List of finding IDs to supersede
            summary: Content of the new consolidated finding
            reason: Why these findings are being merged
            finding_type: Type for the new finding (default: discovery)
            tags: Tags for new finding (if None, merges tags from all superseded findings)
            evidence: Evidence for new finding

        Returns:
            dict with 'new_id', 'superseded_count', 'skipped' (not found) count
        """
        if not finding_ids:
            raise ValueError("No finding IDs provided")

        superseded = 0
        skipped = 0
        merged_tags: set[str] = set()
        project = None
        sprint = None
        now = datetime.now().isoformat()

        for fid in finding_ids:
            row = self.conn.execute(
                "SELECT id, project, sprint, tags, status FROM findings WHERE id = ?",
                (fid,)
            ).fetchone()

            if not row:
                skipped += 1
                continue

            if row["status"] == "superseded":
                skipped += 1
                continue

            if project is None:
                project = row["project"]
                sprint = row["sprint"]

            if row["tags"]:
                merged_tags.update(json.loads(row["tags"]))

        final_tags = tags if tags is not None else list(merged_tags)

        new_id = f"kb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        for fid in finding_ids:
            result = self.conn.execute(
                "UPDATE findings SET status = 'superseded', updated_at = ? WHERE id = ? AND status = 'current'",
                (now, fid)
            )
            if result.rowcount > 0:
                superseded += 1

        if superseded == 0:
            raise ValueError("No valid findings to consolidate (all not found or already superseded)")

        full_content = f"[CONSOLIDATION: {reason}] {summary}"

        self.conn.execute("""
            INSERT INTO findings (id, type, status, project, sprint, tags, content, evidence, created_at, updated_at)
            VALUES (?, ?, 'current', ?, ?, ?, ?, ?, ?, ?)
        """, (new_id, finding_type, project, sprint, json.dumps(final_tags), full_content, evidence, now, now))

        embedding = self._embed(full_content + " " + (evidence or ""))
        self.conn.execute(
            "INSERT INTO findings_vec (id, embedding) VALUES (?, ?)",
            (new_id, embedding)
        )

        self.conn.commit()
        return {"new_id": new_id, "superseded_count": superseded, "skipped": skipped}

    def suggest_consolidation(self, project: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Find clusters of related findings that might be consolidated."""
        findings = self.list_findings(project=project, limit=limit)
        if len(findings) < 3:
            return []

        clusters: list[dict[str, Any]] = []
        used_ids: set[str] = set()

        for f in findings:
            if f["id"] in used_ids:
                continue

            similar = self.search(f["content"], limit=5, project=project)
            cluster_members = [f]

            for s in similar:
                if s["id"] != f["id"] and s["id"] not in used_ids:
                    if s.get("similarity", 0) > 0.7:
                        cluster_members.append(s)
                        used_ids.add(s["id"])

            if len(cluster_members) >= 2:
                used_ids.add(f["id"])
                contents = "\n---\n".join([m["content"][:200] for m in cluster_members[:4]])
                system_prompt = "You analyze related findings for consolidation. Return JSON with 'analysis' field."
                prompt = f"""Analyze these related findings. Return JSON: {{"analysis": "<your analysis>"}}

Should they be consolidated? If yes, suggest a combined summary. If no, explain why distinct.

Findings:
{contents}"""

                result = self._llm.complete(prompt, max_tokens=400, temperature=0.3, system_prompt=system_prompt, json_mode=True)
                analysis = None
                if result:
                    analysis = self._llm.extract_text_from_json(result, keys=[
                        "analysis", "summary", "result", "text", "response",
                        "consolidated_summary", "combined_summary", "recommendation", "reasoning"
                    ])
                clusters.append({
                    "members": [{"id": m["id"], "content": m["content"][:100]} for m in cluster_members],
                    "analysis": analysis or "Analysis unavailable",
                })

        return clusters

    def add_from_template(
        self,
        template_name: str,
        project: str | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a finding using a pre-defined template.

        Templates ensure consistent formatting for common finding types.

        Available templates:
        - computation_result: claim, method, result [script]
        - failed_approach: approach, goal, reason [error]
        - structural_discovery: structure, property, implication [proof_sketch]
        - verification: claim, method, outcome [script, tolerance]
        - hypothesis: hypothesis, motivation, status [tests_needed]

        Args:
            template_name: Name of template to use
            project: Project name
            tags: Tags (auto-suggested if not provided)
            **kwargs: Template fields (required and optional)

        Returns:
            Result from add() method
        """
        if template_name not in self.TEMPLATES:
            available = ", ".join(self.TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")

        template = self.TEMPLATES[template_name]

        required_fields = cast(list[str], template["required"])
        missing = [f for f in required_fields if f not in kwargs]
        if missing:
            raise ValueError(f"Missing required fields for {template_name}: {missing}")

        format_str = cast(str, template["format"])
        content = format_str.format(**{k: kwargs.get(k, "") for k in required_fields})

        evidence_parts: list[str] = []
        optional_fields = cast(list[str], template.get("optional", []))
        for opt in optional_fields:
            if opt in kwargs and kwargs[opt]:
                evidence_parts.append(f"{opt}: {kwargs[opt]}")
        evidence = "\n".join(evidence_parts) if evidence_parts else None

        default_type = cast(str, template["default_type"])
        return self.add(
            content=content,
            finding_type=default_type,
            project=project,
            tags=tags,
            evidence=evidence,
        )

    def review_queue(
        self,
        project: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get findings that need attention.

        Returns findings grouped by issue type:
        - untagged: Findings with no tags
        - low_quality: Findings flagged by validation
        - stale: Findings older than 30 days not recently cited
        - orphaned: Superseded findings with no replacement

        Args:
            project: Filter by project
            limit: Max findings per category

        Returns:
            dict with categories as keys, each containing list of findings
        """
        from datetime import timedelta

        queue: dict[str, list[Any]] = {
            "untagged": [],
            "low_quality": [],
            "stale": [],
            "orphaned": [],
        }

        base_where = "WHERE status = 'current'"
        params: list[Any] = []
        if project:
            base_where += " AND project = ?"
            params = [project]

        rows = self.conn.execute(
            f"""SELECT id, type, content, created_at, project
                FROM findings {base_where}
                AND (tags IS NULL OR tags = '[]')
                ORDER BY created_at DESC LIMIT ?""",
            params + [limit]
        ).fetchall()
        queue["untagged"] = [
            {"id": r["id"], "type": r["type"], "content": r["content"][:100], "created_at": r["created_at"]}
            for r in rows
        ]

        all_findings = self.conn.execute(
            f"SELECT id, type, content, tags, created_at FROM findings {base_where} LIMIT 100",
            params
        ).fetchall()
        for row in all_findings:
            warnings = validate_finding_content(row["content"], json.loads(row["tags"] or "[]"))
            if warnings:
                queue["low_quality"].append({
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"][:100],
                    "warnings": [w["message"] for w in warnings],
                })
                if len(queue["low_quality"]) >= limit:
                    break

        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        rows = self.conn.execute(
            f"""SELECT id, type, content, created_at
                FROM findings {base_where}
                AND created_at < ?
                ORDER BY created_at ASC LIMIT ?""",
            params + [cutoff, limit]
        ).fetchall()
        queue["stale"] = [
            {"id": r["id"], "type": r["type"], "content": r["content"][:100], "created_at": r["created_at"]}
            for r in rows
        ]

        rows = self.conn.execute(
            f"""SELECT f.id, f.type, f.content, f.supersedes_id
                FROM findings f
                LEFT JOIN findings f2 ON f.supersedes_id = f2.id
                WHERE f.status = 'current' AND f.supersedes_id IS NOT NULL AND f2.id IS NULL
                LIMIT ?""",
            [limit]
        ).fetchall()
        queue["orphaned"] = [
            {"id": r["id"], "type": r["type"], "content": r["content"][:100], "missing_ref": r["supersedes_id"]}
            for r in rows
        ]

        return queue

    def generate_open_questions(
        self,
        project: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Analyze findings to identify knowledge gaps and open questions.

        Uses LLM to analyze existing findings and identify:
        - Areas lacking coverage
        - Unresolved issues
        - Natural next steps

        Args:
            project: Filter by project
            limit: Number of questions to generate

        Returns:
            List of dicts with 'question', 'context', 'related_findings'
        """
        findings = self.list_findings(project=project, limit=50)
        if not findings:
            return []

        summaries = []
        for f in findings[:30]:
            summaries.append(f"[{f['type']}] {f['content'][:150]}")

        knowledge_summary = "\n".join(summaries)

        prompt = f"""Analyze these findings and identify {limit} open research questions.

Findings:
{knowledge_summary}

Return a JSON object with key "questions" containing an array of objects with "question", "importance", and "related_topics" fields."""

        response = self._llm.complete(prompt, max_tokens=800, json_mode=True)
        if not response:
            return []

        try:
            data = json.loads(response)
            if isinstance(data, dict) and "questions" in data:
                return data["questions"][:limit]
            elif isinstance(data, list):
                return data[:limit]
        except json.JSONDecodeError:
            pass

        return []

    def summarize_topic(
        self,
        topic: str,
        project: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Synthesize a summary of all findings on a topic.

        Searches for relevant findings and uses LLM to create a coherent
        summary that captures the current state of knowledge.

        Args:
            topic: Topic to summarize
            project: Filter by project
            limit: Max findings to consider

        Returns:
            dict with 'summary', 'key_findings', 'open_questions', 'sources'
        """
        findings = self.search(
            query=topic,
            project=project,
            limit=limit,
            expand=True,
            hybrid=True,
        )

        if not findings:
            return {
                "summary": f"No findings found for topic: {topic}",
                "key_findings": [],
                "open_questions": [],
                "sources": [],
            }

        by_type: dict[str, list[Any]] = {}
        for f in findings:
            by_type.setdefault(f["type"], []).append(f)

        context_parts = []
        for ftype, flist in by_type.items():
            context_parts.append(f"\n=== {ftype.upper()} ===")
            for f in flist[:10]:
                context_parts.append(f"[{f['id']}] {f['content']}")

        context = "\n".join(context_parts)

        prompt = f"""Summarize the current state of knowledge about "{topic}" based on these findings.
Output JSON: {{"summary": "..."}}.

{context}

Include: coherent summary, key facts, open questions, contradictions. Cite finding IDs."""

        response = self._llm.complete(prompt, max_tokens=1000)

        if response:
            response = self._llm.extract_text_from_json(response, keys=["summary", "text", "response"])

        return {
            "summary": response or "Failed to generate summary",
            "finding_count": len(findings),
            "types_found": list(by_type.keys()),
            "sources": [{"id": f["id"], "type": f["type"], "similarity": f.get("similarity", 0)} for f in findings[:10]],
        }

    def get_supersession_chain(self, finding_id: str) -> list[dict[str, Any]]:
        """Get the chain of findings that supersede each other."""
        chain: list[dict[str, Any]] = []
        current_id: str | None = finding_id

        while current_id:
            finding = self.get(current_id)
            if not finding:
                break
            chain.append(finding)

            row = self.conn.execute(
                "SELECT id FROM findings WHERE supersedes_id = ?",
                (current_id,)
            ).fetchone()
            current_id = row["id"] if row else None

        return chain

    def get_latest_update(self) -> tuple[int, str]:
        """Get count and latest timestamp for change detection."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt, MAX(updated_at) as latest FROM findings"
        ).fetchone()
        return (row["cnt"] or 0, row["latest"] or "")

    def reembed_all(self) -> dict[str, Any]:
        """Re-generate embeddings for all findings.

        Use this after fixing the embedding model or algorithm.
        Returns stats on what was re-embedded.
        """
        import sys

        findings = self.conn.execute(
            "SELECT id, content, evidence FROM findings"
        ).fetchall()

        updated = 0
        failed = 0

        for row in findings:
            try:
                text = row["content"] + " " + (row["evidence"] or "")
                embedding = self._embed(text)
                self.conn.execute(
                    "UPDATE findings_vec SET embedding = ? WHERE id = ?",
                    (embedding, row["id"])
                )
                updated += 1
            except Exception as e:
                print(f"Failed to re-embed {row['id']}: {e}", file=sys.stderr)
                failed += 1

        self.conn.commit()
        return {"updated": updated, "failed": failed, "total": len(findings)}

    def backfill_summaries(
        self, project: str | None = None, batch_size: int = 20
    ) -> dict[str, Any]:
        """Generate summaries for findings that don't have one.

        Args:
            project: Optional project filter
            batch_size: How many to process in one batch

        Returns:
            Dict with updated/failed/total counts
        """
        query = "SELECT id, content, evidence FROM findings WHERE summary IS NULL"
        params: list[Any] = []
        if project:
            query += " AND project = ?"
            params.append(project)
        query += f" LIMIT {batch_size}"

        findings = self.conn.execute(query, params).fetchall()

        updated = 0
        failed = 0

        for row in findings:
            try:
                summary = self._generate_summary(row["content"], row["evidence"])
                if summary:
                    self.conn.execute(
                        "UPDATE findings SET summary = ? WHERE id = ?",
                        (summary, row["id"])
                    )
                    updated += 1
                    print(f"  {row['id']}: {summary}")
                else:
                    failed += 1
            except Exception as e:
                print(f"Failed to generate summary for {row['id']}: {e}")
                failed += 1

        self.conn.commit()
        return {"updated": updated, "failed": failed, "total": len(findings)}

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

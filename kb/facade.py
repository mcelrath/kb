"""
KnowledgeBase Facade

Main entry point that delegates to specialized modules while maintaining
backward compatibility with the original API.
"""

import json
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

from .config import load_config as _load_config
from .constants import (
    FINDING_TYPES,
    NOTATION_DOMAINS,
    GREEK_MEANINGS,
)
from .validation import validate_finding_content, validate_tags, serialize_f32, normalize_project_name, detect_project_from_cwd
from .core.connection import DatabaseConnection
from .core.schema import init_schema
from .core.embedding import EmbeddingService
from .llm.client import LLMClient
from .llm.analysis import ContentAnalyzer
from .search.hybrid import HybridSearch
from .entities.scripts import ScriptsRepository
from .entities.documents import DocumentsRepository
from .entities.theorems import TheoremRepository
from .entities.issues import IssuesRepository
from .entities.bridge import BridgeMessagesRepository
from .entities.symbols import SymbolsRepository


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
    _documents: DocumentsRepository
    _theorems: TheoremRepository
    _issues: IssuesRepository
    _bridge: BridgeMessagesRepository
    _symbols: SymbolsRepository

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_url: str | None = None,
        embedding_dim: int | None = None,
    ):
        cfg = _load_config()
        resolved_db_path = Path(db_path) if db_path is not None else cfg.db_path
        resolved_embedding_url = embedding_url if embedding_url is not None else cfg.embedding_url
        resolved_embedding_dim = embedding_dim if embedding_dim is not None else cfg.embedding_dim

        self.db_path = resolved_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_url = resolved_embedding_url
        self.embedding_dim = resolved_embedding_dim

        # Initialize database connection
        db_conn = DatabaseConnection(resolved_db_path, resolved_embedding_dim)
        self.conn = db_conn.conn
        init_schema(self.conn, resolved_embedding_dim)

        # Initialize subsystems
        self._embedding = EmbeddingService(resolved_embedding_url, resolved_embedding_dim)
        self._llm = LLMClient(cfg.llm_url)
        self._analyzer = ContentAnalyzer(self._llm)
        self._search = HybridSearch(
            self.conn,
            self._embedding,
            expand_query=lambda q, p, v: self._llm.expand_query(q, p, embedding_url, v)
        )

        # Seed embedding_meta on first run (idempotent; no-op if row exists)
        try:
            self._ensure_embedding_meta()
        except Exception:
            pass  # Non-fatal: DB may be read-only or during migration

        # Initialize entity repositories
        self._scripts = ScriptsRepository(
            self.conn,
            self._embedding,
            finding_exists=lambda fid: self.get(fid) is not None
        )
        self._documents = DocumentsRepository(self.conn)
        self._theorems = TheoremRepository(self.conn, self._embedding)
        self._issues = IssuesRepository(self.conn, self._embedding)
        self._bridge = BridgeMessagesRepository(self.conn, self._embedding)
        self._symbols = SymbolsRepository(self.conn, self._embedding)

    # =========================================================================
    # Backward-compatible methods delegating to subsystems
    # =========================================================================

    def _embed(self, text: str) -> bytes:
        """Generate embedding for text."""
        return self._embedding.embed(text)

    def _generate_summary(self, content: str, evidence: str | None = None) -> str | None:
        """Generate summary for finding.

        Routing controlled by load_config().summary_mode — config.toml [llm]
        summary_mode, with KB_SUMMARY_MODE env overriding it (precedence resolved
        in kb/config.py). Default: extractive.
          extractive       -> no-LLM first-sentence blurb (DEFAULT — zero VRAM/cost,
                              no second model; the easy out-of-the-box path)
          none             -> always return None (search shows raw content)
          local-llm        -> ContentAnalyzer via local LLM server (needs a 2nd model)
          subscription-sdk -> claude_agent_sdk.query (Haiku, subscription OAuth,
                              ANTHROPIC_API_KEY scrubbed so stale key is bypassed)
          api              -> same as local-llm for now (future: direct API path)
        """
        from .config import load_config
        mode = load_config().summary_mode
        if mode == "none":
            return None
        if mode == "extractive":
            from .llm.extractive import extractive_summary
            return extractive_summary(content, evidence)
        if mode == "subscription-sdk":
            from .llm.summary_sdk import summarize_one
            text = content + (" " + evidence if evidence else "")
            return summarize_one(text)
        # local-llm and api use the LLM analyzer path
        return self._analyzer.generate_summary(content, evidence)

    def expand_query(self, query: str, project: str | None = None, verbose: bool = False) -> str:
        """Expand search query using LLM."""
        return self._llm.expand_query(query, project, self.embedding_url, verbose)

    def suggest_tags(self, content: str, project: str | None = None) -> list[str]:
        """Suggest tags for content."""
        existing_tags = self._fetch_existing_tags(project)
        return self._analyzer.suggest_tags(content, existing_tags)

    def _fetch_existing_tags(self, project: str | None = None) -> set[str]:
        """Fetch the set of existing tags from the DB (main-thread safe)."""
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
        return existing_tags

    def classify_finding_type(self, content: str) -> str:
        """Classify finding type."""
        return self._analyzer.classify_type(content)

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
        if "project" in kwargs:
            kwargs["project"] = normalize_project_name(kwargs["project"])
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
        summary: str | None = None,
        check_duplicate: bool = True,
        duplicate_threshold: float = 0.85,
        check_contradictions: bool = True,
        auto_tag: bool = True,
        auto_classify: bool = True,
        auto_summarize_evidence: bool = True,
        max_evidence_length: int = 500,
    ) -> dict[str, Any]:
        """Add a new finding."""
        project = normalize_project_name(project) or detect_project_from_cwd()
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

        # Caller-provided summary wins: the LLM that WROTE this finding hands us a
        # one-liner in the same turn (smart, free, no second model). Only fall back
        # to auto-generation (extractive by default) when no summary is supplied.
        if summary is None:
            summary = self._generate_summary(content, evidence)

        if embedding is None:
            embedding = self._embed(content + " " + (evidence or ""))

        import time as _time
        _max_retries = 5
        for _attempt in range(_max_retries):
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
                break
            except sqlite3.OperationalError as _e:
                if "locked" in str(_e) and _attempt < _max_retries - 1:
                    self.conn.rollback()
                    _time.sleep(0.5 * (_attempt + 1))
                    continue
                self.conn.rollback()
                raise
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
        project = normalize_project_name(project)
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

        no_summary = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE status='current' AND (summary IS NULL OR summary='')"
        ).fetchone()[0]
        no_embedding = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE status='current' AND id NOT IN (SELECT id FROM findings_vec)"
        ).fetchone()[0]

        no_summary_by_project: dict[str, int] = {}
        for row in self.conn.execute(
            "SELECT COALESCE(project,'(none)'), COUNT(*) FROM findings "
            "WHERE status='current' AND (summary IS NULL OR summary='') "
            "GROUP BY project ORDER BY COUNT(*) DESC"
        ).fetchall():
            no_summary_by_project[row[0]] = row[1]

        no_embedding_by_project: dict[str, int] = {}
        for row in self.conn.execute(
            "SELECT COALESCE(project,'(none)'), COUNT(*) FROM findings "
            "WHERE status='current' AND id NOT IN (SELECT id FROM findings_vec) "
            "GROUP BY project ORDER BY COUNT(*) DESC"
        ).fetchall():
            no_embedding_by_project[row[0]] = row[1]

        return {
            "db_path": str(self.db_path),
            "total": total,
            "current": current,
            "superseded": superseded,
            "no_summary": no_summary,
            "no_embedding": no_embedding,
            "no_summary_by_project": no_summary_by_project,
            "no_embedding_by_project": no_embedding_by_project,
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

        scripts = self._scripts.search(content, project=project, limit=3)
        for s in scripts:
            if s.get("similarity", 0) > 0.5:
                suggestions["scripts"].append({
                    "id": s["id"],
                    "filename": s["filename"],
                    "purpose": s.get("purpose", "")[:100],
                    "similarity": s.get("similarity", 0)
                })

        docs = self._documents.search(content, project=project)
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

    def finding_scripts(self, finding_id: str) -> list[dict[str, Any]]:
        """Get scripts for finding."""
        return self._scripts.get_for_finding(finding_id)

    def finding_docs(self, finding_id: str) -> list[dict[str, Any]]:
        """Get documents for finding."""
        return self._documents.get_docs_for_finding(finding_id)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_all_tags(self, limit: int | None = None) -> list[str]:
        """Get unique tags, optionally limited to the top N most-used."""
        if limit is not None:
            rows = self.conn.execute(
                "SELECT value, COUNT(*) as cnt FROM findings, json_each(findings.tags) "
                "WHERE findings.tags IS NOT NULL GROUP BY value ORDER BY cnt DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return sorted(row[0] for row in rows)
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
        input_limit: int = 20,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Identify open research questions from KB findings.

        Args:
            limit: Number of questions to generate.
            input_limit: Number of KB entries to feed the LLM.
            query: If given, seeds entries via semantic search; otherwise most recent.

        Returns list of dicts with 'question', 'why', 'related_ids' keys.
        """
        import sys
        if query:
            raw = self.search(query=query, limit=input_limit, project=project)
        else:
            raw = self.list_findings(project=project, limit=input_limit)

        if not raw:
            return []

        # Use summary if available, fall back to full content (no truncation)
        lines = []
        for f in raw:
            text = f.get("summary") or f.get("content", "")
            lines.append(f"[{f['id']}] [{f['type'].upper()}] {text}")

        findings_block = "\n".join(lines)

        # Warn if likely to exceed 256k context (≈4 tokens/char, reserve 4k for output)
        estimated_tokens = len(findings_block) // 4
        if estimated_tokens > 252_000:
            print(
                f"WARNING: ~{estimated_tokens:,} tokens estimated for {len(raw)} findings "
                f"(256k context limit). Consider reducing -n or using a search query to filter.",
                file=sys.stderr,
            )

        prompt = f"""You are analyzing a researcher's knowledge base to surface open questions and gaps.

FINDINGS ({len(raw)} entries):
{findings_block}

Return exactly {limit} concrete open questions that these findings do NOT yet answer.
Each question must be directly motivated by the findings above and must NOT already be answered by them.
Prefer questions that would unlock or unblock multiple other open problems.

Return a complete JSON object with exactly {limit} items:
{{"questions": [{{"question": "...", "why": "one sentence explaining what in the findings motivates this", "related_ids": ["kb-id-here"]}}]}}"""

        response = self._llm.complete(prompt, max_tokens=-1, json_mode=False, timeout=300, thinking=True)
        if not response:
            return []

        try:
            # Strip markdown code fences the model sometimes emits
            text = response.strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-z]*\n?", "", text)
                text = re.sub(r"\n?```$", "", text.rstrip())
            data = json.loads(text)
            if isinstance(data, dict) and "questions" in data:
                return data["questions"][:limit]
            if isinstance(data, list):
                return data[:limit]
        except (json.JSONDecodeError, TypeError):
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

    # =========================================================================
    # Embedding model metadata
    # =========================================================================

    def _embedding_signature(self) -> str:
        """SHA-256 of format|url|model|dim from current EmbeddingService config."""
        import hashlib
        raw = "|".join([
            self._embedding.embedding_format,
            self._embedding.embedding_url,
            self._embedding.embedding_model,
            str(self._embedding.embedding_dim),
        ])
        return hashlib.sha256(raw.encode()).hexdigest()

    def _ensure_embedding_meta(self) -> None:
        """Seed embedding_meta on first run; warn on signature drift if row exists.

        If no row exists (brand-new DB or pre-migration DB), seed silently —
        we assume the current config matches whatever was used.

        If a row already exists but the configured signature differs, emit a
        one-line stderr warning so the drift is never silent (outage prevention).
        Does NOT block init: the warning points the operator to `kb reembed`.
        """
        import sys as _sys
        row = self.conn.execute(
            "SELECT id, signature, dim FROM embedding_meta WHERE id = 1"
        ).fetchone()

        sig = self._embedding_signature()
        now = datetime.now().isoformat()

        if row is None:
            # First-time seed: no prior vectors, no drift possible.
            self.conn.execute("""
                INSERT INTO embedding_meta (id, format, url, model, dim, signature, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?)
            """, (
                self._embedding.embedding_format,
                self._embedding.embedding_url,
                self._embedding.embedding_model,
                self._embedding.embedding_dim,
                sig,
                now,
            ))
            self.conn.commit()
            return

        stored_sig = row["signature"] if isinstance(row, dict) else row[1]
        stored_dim = row["dim"] if isinstance(row, dict) else row[2]
        if stored_sig != sig:
            if stored_dim != self._embedding.embedding_dim:
                print(
                    f"kb WARNING: embedding dim changed "
                    f"stored={stored_dim} configured={self._embedding.embedding_dim}. "
                    "Run: kb reembed --force",
                    file=_sys.stderr,
                )
            else:
                print(
                    f"kb WARNING: embedding model/format changed (dim={self._embedding.embedding_dim} unchanged). "
                    "Run: kb reembed --force",
                    file=_sys.stderr,
                )

    def embedding_status(self) -> dict[str, Any]:
        """Return embedding config status: stored vs configured + verdict.

        Verdicts:
          ok                   signatures match (or meta just seeded)
          no-meta              table missing or no row (legacy DB pre-migration)
          mismatch-same-dim    model/url/format changed but dim is same
          mismatch-dim-change  dim changed (requires _vec recreate + full reembed)
        """
        configured_sig = self._embedding_signature()
        configured = {
            "format": self._embedding.embedding_format,
            "url": self._embedding.embedding_url,
            "model": self._embedding.embedding_model,
            "dim": self._embedding.embedding_dim,
            "signature": configured_sig,
        }

        try:
            row = self.conn.execute(
                "SELECT format, url, model, dim, signature, updated_at FROM embedding_meta WHERE id = 1"
            ).fetchone()
        except Exception:
            # Table doesn't exist yet (very old DB)
            return {
                "configured": configured,
                "stored": None,
                "verdict": "no-meta",
                "message": "No embedding_meta row; run `kb reembed --force` to initialize.",
            }

        if row is None:
            return {
                "configured": configured,
                "stored": None,
                "verdict": "no-meta",
                "message": "No embedding_meta row; run `kb reembed --force` to initialize.",
            }

        stored = {
            "format": row[0],
            "url": row[1],
            "model": row[2],
            "dim": row[3],
            "signature": row[4],
            "updated_at": row[5],
        }

        if stored["signature"] == configured_sig:
            verdict = "ok"
            message = "Embedding config matches stored metadata."
        elif stored["dim"] != configured["dim"]:
            verdict = "mismatch-dim-change"
            message = (
                f"Embedding dim changed: stored={stored['dim']} configured={configured['dim']}. "
                "All _vec tables must be recreated. Run: kb reembed --force"
            )
        else:
            verdict = "mismatch-same-dim"
            message = (
                f"Embedding model/format changed (dim unchanged at {configured['dim']}). "
                "Run: kb reembed --force"
            )

        return {
            "configured": configured,
            "stored": stored,
            "verdict": verdict,
            "message": message,
        }

    def reembed_all(
        self,
        *,
        resume: bool = False,
        commit_every: int = 50,
        force_dim: int | None = None,
    ) -> dict[str, Any]:
        """Re-generate embeddings for all entities across all 8 vec tables.

        Covers ALL eight _vec tables:
          findings_vec, scripts_vec, lean_theorems_vec, concepts_vec, issues_vec,
          symbols_vec, tex_annotations_vec, document_sections_vec.

        For symbols and tex_annotations, BOTH the base-table `embedding BLOB`
        column AND the _vec row are regenerated (mirroring add_symbol /
        add_tex_annotation dual-write).

        Dim-change handling: if force_dim is provided (or stored dim != configured dim),
        all 8 _vec tables are DROPPED and RECREATED at the new dim BEFORE the reembed
        loop.  The drop is gated on coverage — we only drop tables we will repopulate.

        POST-REEMBED ASSERTION: after each table, verifies that _vec rowcount matches
        its base table's embedded-row count.  Raises RuntimeError if any table is short.

        On completion, writes embedding_meta = configured signature.

        Args:
            resume: If True, skip rows already present in the vec table (safe only if
                existing vec rows are from the CURRENT embedding model).
            commit_every: COMMIT after every N successful rows (default 50).
            force_dim: If given, treat as a dim-change regardless of stored meta.
        """
        import sys
        import time
        from .core.schema import init_schema

        stats: dict[str, Any] = {}

        # Determine if a dim-change is needed.
        configured_dim = self._embedding.embedding_dim
        stored_dim: int | None = None
        try:
            row = self.conn.execute(
                "SELECT dim FROM embedding_meta WHERE id = 1"
            ).fetchone()
            if row:
                stored_dim = row[0]
        except Exception:
            pass

        dim_changed = (force_dim is not None and force_dim != configured_dim) or (
            stored_dim is not None and stored_dim != configured_dim
        )

        # The 7 vec tables we own and will repopulate.
        ALL_VEC_TABLES = [
            "findings_vec",
            "scripts_vec",
            "lean_theorems_vec",
            "issues_vec",
            "symbols_vec",
            "tex_annotations_vec",
            "document_sections_vec",
        ]

        if dim_changed:
            new_dim = force_dim if force_dim is not None else configured_dim
            print(
                f"reembed_all: DIM CHANGE detected "
                f"stored={stored_dim} -> configured={new_dim}. "
                f"Dropping and recreating all {len(ALL_VEC_TABLES)} _vec tables.",
                file=sys.stderr,
                flush=True,
            )
            # Drop all 7 (gated: we repopulate all of them in this loop)
            for vtable in ALL_VEC_TABLES:
                self.conn.execute(f"DROP TABLE IF EXISTS {vtable}")
            self.conn.commit()
            # Recreate at configured_dim
            init_schema(self.conn, configured_dim)
            # Force a full re-embed (resume would be meaningless after recreate)
            resume = False

        # Up-front totals.
        counts = {
            "findings": self.conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0],
            "scripts": self.conn.execute("SELECT COUNT(*) FROM scripts").fetchone()[0],
            "lean_theorems": self.conn.execute("SELECT COUNT(*) FROM lean_theorems").fetchone()[0],
            "issues": self.conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0],
            "symbols": self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
            "tex_annotations": self.conn.execute("SELECT COUNT(*) FROM tex_annotations").fetchone()[0],
            "document_sections": self.conn.execute("SELECT COUNT(*) FROM document_sections").fetchone()[0],
        }
        grand_total = sum(counts.values())
        print(
            f"reembed_all: " + " ".join(f"{k}={v}" for k, v in counts.items())
            + f" GRAND_TOTAL={grand_total}",
            file=sys.stderr,
            flush=True,
        )

        def _do_table(table: str, select_sql: str, vec_table: str, text_fn,
                      base_blob_update: Any = None) -> None:
            """Re-embed one table.

            base_blob_update: optional callable(conn, row_id, embedding_bytes) that
            also writes the embedding back to the base table's BLOB column.
            Used for symbols and tex_annotations.
            """
            t0 = time.monotonic()
            all_rows = self.conn.execute(select_sql).fetchall()
            total_all = len(all_rows)
            already_done: set = set()
            if resume:
                already_done = {
                    r[0] for r in self.conn.execute(f"SELECT id FROM {vec_table}").fetchall()
                }
            rows = [r for r in all_rows if r["id"] not in already_done]
            total = len(rows)
            skipped = total_all - total
            updated = failed = since_commit = 0
            interval = max(10, total // 100) if total else 1
            print(
                f"  [{table}] starting: {total} to process"
                + (f" ({skipped} already-done skipped)" if skipped else "")
                + f"; commit_every={commit_every}",
                file=sys.stderr,
                flush=True,
            )
            for i, row in enumerate(rows, 1):
                try:
                    emb = self._embed(text_fn(row))
                    self.conn.execute(f"DELETE FROM {vec_table} WHERE id = ?", (row["id"],))
                    self.conn.execute(
                        f"INSERT INTO {vec_table} (id, embedding) VALUES (?, ?)",
                        (row["id"], emb),
                    )
                    if base_blob_update is not None:
                        base_blob_update(self.conn, row["id"], emb)
                    updated += 1
                    since_commit += 1
                    if since_commit >= commit_every:
                        self.conn.commit()
                        since_commit = 0
                except Exception as e:
                    print(f"{table} {row['id']}: {e}", file=sys.stderr, flush=True)
                    failed += 1
                if i % interval == 0 or i == total:
                    elapsed = time.monotonic() - t0
                    rate = i / elapsed if elapsed > 0 else 0.0
                    remaining = total - i
                    eta_sec = remaining / rate if rate > 0 else 0.0
                    eta_min = eta_sec / 60.0
                    pct = 100.0 * i / total if total else 100.0
                    print(
                        f"  [{table}] {i}/{total} ({pct:5.1f}%) "
                        f"rate={rate:6.2f}/s "
                        f"elapsed={elapsed/60.0:6.1f}m "
                        f"eta={eta_min:6.1f}m "
                        f"updated={updated} failed={failed}",
                        file=sys.stderr,
                        flush=True,
                    )
            self.conn.commit()
            elapsed_total = time.monotonic() - t0
            stats[table] = {
                "updated": updated,
                "failed": failed,
                "total": total,
                "total_all": total_all,
                "skipped_already_done": skipped,
                "elapsed_sec": elapsed_total,
            }
            print(
                f"{table}: {updated}/{total} re-embedded in "
                f"{elapsed_total/60.0:.1f}m "
                f"({failed} failed, {skipped} skipped)",
                file=sys.stderr,
                flush=True,
            )

        _do_table(
            "findings",
            "SELECT id, content, evidence FROM findings",
            "findings_vec",
            lambda r: r["content"] + (" " + r["evidence"] if r["evidence"] else ""),
        )
        _do_table(
            "scripts",
            "SELECT id, purpose FROM scripts",
            "scripts_vec",
            lambda r: r["purpose"] or "",
        )
        _do_table(
            "lean_theorems",
            "SELECT id, statement_pure, statement FROM lean_theorems",
            "lean_theorems_vec",
            lambda r: r["statement_pure"] if r["statement_pure"] else r["statement"],
        )
        _do_table(
            "issues",
            "SELECT id, title, description FROM issues",
            "issues_vec",
            lambda r: r["title"] + (" " + r["description"] if r["description"] else ""),
        )

        # symbols: regenerate embed text as add_symbol does it,
        # and write BOTH the _vec row AND the base-table embedding BLOB.
        def _symbols_text(row: Any) -> str:
            module = row["module"] or ""
            name = row["name"] or ""
            signature = row["signature"] or ""
            docstring_summary = row["docstring_summary"] or ""
            return f"{module}.{name}: {signature} {docstring_summary}"

        def _update_python_symbol_blob(conn: Any, row_id: str, emb: bytes) -> None:
            conn.execute(
                "UPDATE symbols SET embedding = ? WHERE id = ?",
                (emb, row_id),
            )

        _do_table(
            "symbols",
            "SELECT id, module, name, signature, docstring_summary FROM symbols",
            "symbols_vec",
            _symbols_text,
            base_blob_update=_update_python_symbol_blob,
        )

        # tex_annotations: regenerate embed text as add_tex_annotation does it,
        # and write BOTH the _vec row AND the base-table embedding BLOB.
        def _tex_annotations_text(row: Any) -> str:
            parts = []
            if row["section_title"]:
                parts.append(row["section_title"])
            if row["section_label"]:
                parts.append(row["section_label"])
            if row["python_refs"]:
                parts.append(f"python:{row['python_refs']}")
            if row["lean_refs"]:
                parts.append(f"lean:{row['lean_refs']}")
            if row["context"]:
                parts.append(row["context"])
            return " ".join(filter(None, parts))

        def _update_tex_annotation_blob(conn: Any, row_id: str, emb: bytes) -> None:
            conn.execute(
                "UPDATE tex_annotations SET embedding = ? WHERE id = ?",
                (emb, row_id),
            )

        _do_table(
            "tex_annotations",
            "SELECT id, section_label, section_title, python_refs, lean_refs, context "
            "FROM tex_annotations",
            "tex_annotations_vec",
            _tex_annotations_text,
            base_blob_update=_update_tex_annotation_blob,
        )

        # document_sections: embed embed_text if present, else content.
        def _document_sections_text(row: Any) -> str:
            return row["embed_text"] or row["content"] or row["heading"] or ""

        _do_table(
            "document_sections",
            "SELECT id, embed_text, content, heading FROM document_sections",
            "document_sections_vec",
            _document_sections_text,
        )

        # POST-REEMBED ASSERTION: each _vec must have >= base embedded-row count.
        # For findings/scripts/lean_theorems/concepts/issues: all rows should have a vec entry.
        # For symbols/tex_annotations: rows with non-null embedding should match.
        assertion_checks = [
            ("findings_vec",           "SELECT COUNT(*) FROM findings"),
            ("scripts_vec",            "SELECT COUNT(*) FROM scripts"),
            ("lean_theorems_vec",      "SELECT COUNT(*) FROM lean_theorems"),
            ("concepts_vec",           "SELECT COUNT(*) FROM concepts"),
            ("issues_vec",             "SELECT COUNT(*) FROM issues"),
            ("symbols_vec",            "SELECT COUNT(*) FROM symbols WHERE embedding IS NOT NULL"),
            ("tex_annotations_vec",    "SELECT COUNT(*) FROM tex_annotations WHERE embedding IS NOT NULL"),
            ("document_sections_vec",  "SELECT COUNT(*) FROM document_sections"),
        ]
        assertion_errors = []
        for vec_table, base_sql in assertion_checks:
            vec_count = self.conn.execute(f"SELECT COUNT(*) FROM {vec_table}").fetchone()[0]
            base_count = self.conn.execute(base_sql).fetchone()[0]
            if vec_count < base_count:
                msg = (
                    f"ASSERTION FAILED: {vec_table} has {vec_count} rows "
                    f"but base has {base_count} embedded rows "
                    f"(short by {base_count - vec_count})"
                )
                assertion_errors.append(msg)
                print(f"ERROR: {msg}", file=sys.stderr, flush=True)
            else:
                print(
                    f"  ASSERT OK: {vec_table} {vec_count} == {base_count}",
                    file=sys.stderr,
                    flush=True,
                )

        if assertion_errors:
            raise RuntimeError(
                "reembed_all post-reembed assertion failed:\n"
                + "\n".join(assertion_errors)
            )

        # Write embedding_meta = configured signature.
        sig = self._embedding_signature()
        now = datetime.now().isoformat()
        self.conn.execute("""
            INSERT OR REPLACE INTO embedding_meta (id, format, url, model, dim, signature, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?)
        """, (
            self._embedding.embedding_format,
            self._embedding.embedding_url,
            self._embedding.embedding_model,
            self._embedding.embedding_dim,
            sig,
            now,
        ))
        self.conn.commit()

        print(f"reembed_all: complete; embedding_meta updated (sig={sig[:16]}...)", file=sys.stderr, flush=True)

        return stats

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


    # =========================================================================
    # TeX annotation index methods
    # =========================================================================

    def add_tex_annotation(
        self,
        file: str,
        line: int,
        section_label: str | None = None,
        section_title: str | None = None,
        python_refs: list[str] | None = None,
        lean_refs: list[str] | None = None,
        epic_refs: list[str] | None = None,
        kb_refs: list[str] | None = None,
        context: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Add or update a TeX annotation in the index.

        Returns dict with 'id', 'is_new'.
        """
        existing = self.conn.execute(
            "SELECT id FROM tex_annotations WHERE file = ? AND line = ?",
            (file, line),
        ).fetchone()

        now = datetime.now().isoformat()
        ann_id = f"texann-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        embed_text = " ".join(filter(None, [
            section_title or "",
            section_label or "",
            f"python:{json.dumps(python_refs or [])}",
            f"lean:{json.dumps(lean_refs or [])}",
            context or "",
        ]))
        embedding = self._embed(embed_text)

        python_json = json.dumps(python_refs or [])
        lean_json = json.dumps(lean_refs or [])
        epic_json = json.dumps(epic_refs or [])
        kb_json = json.dumps(kb_refs or [])

        if existing:
            ann_id = existing[0]
            self.conn.execute("""
                UPDATE tex_annotations SET section_label=?, section_title=?,
                    python_refs=?, lean_refs=?, epic_refs=?, kb_refs=?,
                    context=?, updated_at=?, embedding=?
                WHERE id=?
            """, (section_label, section_title, python_json, lean_json, epic_json,
                  kb_json, context, now, embedding, ann_id))
            self.conn.execute("DELETE FROM tex_annotations_vec WHERE id = ?", (ann_id,))
            self.conn.execute(
                "INSERT INTO tex_annotations_vec (id, embedding) VALUES (?, ?)",
                (ann_id, embedding),
            )
            self.conn.commit()
            return {"id": ann_id, "is_new": False}

        self.conn.execute("""
            INSERT INTO tex_annotations
                (id, section_label, section_title, python_refs, lean_refs, epic_refs,
                 kb_refs, context, file, line, created_at, updated_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ann_id, section_label, section_title, python_json, lean_json, epic_json,
              kb_json, context, file, line, now, now, embedding))
        self.conn.execute(
            "INSERT INTO tex_annotations_vec (id, embedding) VALUES (?, ?)",
            (ann_id, embedding),
        )
        self.conn.commit()
        return {"id": ann_id, "is_new": True}

    def search_tex_annotations(
        self,
        query: str,
        file: str | None = None,
        section_label: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search TeX annotations by semantic similarity."""
        embedding = self._embed(query)

        conditions = []
        params: list[Any] = []
        if file:
            conditions.append("t.file = ?")
            params.append(file)
        if section_label:
            conditions.append("t.section_label = ?")
            params.append(section_label)

        vec_results = self.conn.execute(
            f"""SELECT v.id, v.distance
                FROM tex_annotations_vec v
                JOIN tex_annotations t ON t.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
                {"AND " + " AND ".join(conditions) if conditions else ""}
                ORDER BY v.distance""",
            (embedding, limit * 2, *params),
        ).fetchall()

        seen: dict[str, float] = {}
        for sid, dist in vec_results:
            if dist is not None:
                seen[sid] = 1 - (dist ** 2) / 2

        top_ids = sorted(seen, key=lambda x: seen[x], reverse=True)[:limit]
        results = []
        for sid in top_ids:
            row = self.conn.execute(
                """SELECT id, section_label, section_title, python_refs, lean_refs,
                          epic_refs, kb_refs, context, file, line
                   FROM tex_annotations WHERE id = ?""",
                (sid,),
            ).fetchone()
            if row:
                r = dict(zip([
                    "id", "section_label", "section_title", "python_refs", "lean_refs",
                    "epic_refs", "kb_refs", "context", "file", "line"
                ], row))
                r["similarity"] = round(seen[sid], 4)
                for fld in ("python_refs", "lean_refs", "epic_refs", "kb_refs"):
                    if r.get(fld):
                        try:
                            r[fld] = json.loads(r[fld])
                        except json.JSONDecodeError:
                            r[fld] = []
                results.append(r)
        return results

    # =========================================================================
    # Findings refresh methods (encapsulate inline SQL from _run_refresh /
    # _fetch_refresh_rows in kb.py so schema changes localize here)
    # =========================================================================

    def fetch_refresh_rows(
        self,
        ids: list[str] | None = None,
        project: str | None = None,
        all_rows: bool = False,
        limit: int = 0,
    ) -> list[Any]:
        """Return (id, project, content, evidence) rows for the refresh loop.

        Args:
            ids: Explicit list of finding IDs to refresh (ignores other filters).
            project: Restrict to this project when ids is None.
            all_rows: If False (default), only rows with NULL/empty summary.
            limit: Max rows (0 = no limit).
        """
        if ids:
            placeholders = ",".join("?" * len(ids))
            return self.conn.execute(
                f"SELECT id, project, content, evidence FROM findings WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        sql = "SELECT id, project, content, evidence FROM findings WHERE status = 'current'"
        params: list[Any] = []
        if not all_rows:
            sql += " AND (summary IS NULL OR summary = '')"
        if project:
            sql += " AND project = ?"
            params.append(project)
        sql += " ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.conn.execute(sql, params).fetchall()

    def update_finding_refresh(
        self,
        fid: str,
        summary: str | None,
        tags: list[str] | None,
        embedding: bytes | None,
    ) -> None:
        """Write one refresh row: BEGIN IMMEDIATE / UPDATE / COMMIT.

        Lock is held for microseconds — exactly one row per transaction,
        matching the deliberate per-row commit cadence of the refresh loop.
        """
        import json as _json
        self.conn.execute("BEGIN IMMEDIATE")
        if summary:
            self.conn.execute(
                "UPDATE findings SET summary=?, updated_at=datetime('now') WHERE id=?",
                (summary, fid),
            )
        if tags:
            self.conn.execute(
                "UPDATE findings SET tags=?, updated_at=datetime('now') WHERE id=?",
                (_json.dumps(tags), fid),
            )
        if embedding is not None:
            self.conn.execute("DELETE FROM findings_vec WHERE id=?", (fid,))
            self.conn.execute(
                "INSERT INTO findings_vec (id, embedding) VALUES (?,?)",
                (fid, embedding),
            )
        self.conn.commit()

    # =========================================================================
    # lean_work_queue repository methods (encapsulate queue-defer SQL so the
    # CLI handler no longer opens its own sqlite3 connection)
    # =========================================================================

    def list_deferred_queue_rows(self, limit: int = 50) -> list[Any]:
        """Return deferred lean_work_queue rows (defer_reason IS NOT NULL)."""
        return self.conn.execute("""
            SELECT id, file, decl_name, class, readiness, defer_reason, defer_detail, updated_at
            FROM lean_work_queue
            WHERE defer_reason IS NOT NULL AND defer_reason != ''
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

    def get_queue_row(self, row_id: str) -> Any:
        """Return a single lean_work_queue row by id, or None."""
        return self.conn.execute(
            "SELECT id, class, readiness, defer_reason FROM lean_work_queue WHERE id = ?",
            (row_id,),
        ).fetchone()

    def set_defer_reason(self, row_id: str, reason: str, detail: str | None) -> None:
        """Set defer_reason (and optional detail) on a lean_work_queue row."""
        self.conn.execute(
            "UPDATE lean_work_queue SET defer_reason = ?, defer_detail = ?, updated_at = datetime('now') WHERE id = ?",
            (reason, detail or None, row_id),
        )
        self.conn.commit()

    def clear_defer_reason(self, row_id: str) -> None:
        """Clear defer_reason on a lean_work_queue row (re-activates it)."""
        self.conn.execute(
            "UPDATE lean_work_queue SET defer_reason = NULL, defer_detail = NULL, updated_at = datetime('now') WHERE id = ?",
            (row_id,),
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

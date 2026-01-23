"""
Hybrid Search

Combines vector similarity and full-text search using Reciprocal Rank Fusion.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Callable

from ..core.embedding import EmbeddingService


class HybridSearch:
    """Hybrid vector + FTS search with RRF merging."""

    conn: sqlite3.Connection
    embedding_service: EmbeddingService
    expand_query: Callable[[str, str | None, bool], str] | None

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedding_service: EmbeddingService,
        expand_query: Callable[[str, str | None, bool], str] | None = None,
    ):
        self.conn = conn
        self.embedding_service = embedding_service
        self.expand_query = expand_query

    def search(
        self,
        query: str,
        limit: int = 10,
        include_superseded: bool = False,
        project: str | None = None,
        finding_type: str | None = None,
        tags: list[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        hybrid: bool = True,
        expand: bool = False,
        verbose: bool = False,
        deprioritize_index: bool = True,
        exclude_corrections: bool = True,
        recency_weight: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Search findings using hybrid vector + keyword search.

        Args:
            query: Search query string
            limit: Maximum results to return
            include_superseded: Include superseded findings
            project: Filter by project
            finding_type: Filter by type (success/failure/discovery/experiment)
            tags: Filter by tags (all must match)
            after: Filter to findings after this date (ISO format: YYYY-MM-DD)
            before: Filter to findings before this date (ISO format: YYYY-MM-DD)
            hybrid: Combine vector and keyword search (default True)
            expand: Use LLM to expand query with synonyms/related terms
            verbose: Show expanded query and other debug info
            deprioritize_index: Demote INDEX/entry-point findings in ranking
            exclude_corrections: Exclude [CORRECTION:...] entries
            recency_weight: Weight for recency boost (0-1, higher = more recency bias)
        """
        # Optionally expand query using LLM
        search_query = query
        if expand and self.expand_query:
            search_query = self.expand_query(query, project, verbose)

        vector_results: dict[str, dict[str, Any]] = {}
        fts_results: dict[str, dict[str, Any]] = {}

        # Vector similarity search
        query_embedding = self.embedding_service.embed(search_query)
        sql = """
            SELECT f.*, v.distance
            FROM findings f
            JOIN findings_vec v ON f.id = v.id
            WHERE v.embedding MATCH ?
            AND k = ?
        """
        params: list[Any] = [query_embedding, limit * 3]
        rows = self.conn.execute(sql, params).fetchall()

        for rank, row in enumerate(rows, 1):
            distance = float(row["distance"])
            vector_results[row["id"]] = {
                "row": row,
                "rank": rank,
                "similarity": 1 - (distance ** 2) / 2,
            }

        # Full-text search (for hybrid)
        if hybrid:
            fts_query = search_query.replace('"', '""')
            try:
                sql = """
                    SELECT f.*, fts.rank
                    FROM findings f
                    JOIN findings_fts fts ON f.rowid = fts.rowid
                    WHERE findings_fts MATCH ?
                    ORDER BY fts.rank
                    LIMIT ?
                """
                fts_rows = self.conn.execute(sql, [fts_query, limit * 3]).fetchall()

                for rank, row in enumerate(fts_rows, 1):
                    fts_results[row["id"]] = {
                        "row": row,
                        "rank": rank,
                        "relevance": -float(row["rank"]),
                    }
            except sqlite3.OperationalError:
                # FTS query failed (e.g., syntax error), skip keyword results
                if verbose:
                    import sys
                    print(f"FTS query failed for: {fts_query}", file=sys.stderr)

        # Merge results using Reciprocal Rank Fusion (RRF)
        k = 60  # RRF constant
        all_ids = set(vector_results.keys()) | set(fts_results.keys())
        merged: dict[str, dict[str, Any]] = {}

        for finding_id in all_ids:
            rrf_score = 0.0
            row = None

            if finding_id in vector_results:
                rrf_score += 1 / (k + vector_results[finding_id]["rank"])
                row = vector_results[finding_id]["row"]

            if finding_id in fts_results:
                rrf_score += 1 / (k + fts_results[finding_id]["rank"])
                if row is None:
                    row = fts_results[finding_id]["row"]

            vec_data = vector_results.get(finding_id, {})
            fts_data = fts_results.get(finding_id, {})

            merged[finding_id] = {
                "row": row,
                "rrf_score": rrf_score,
                "vector_sim": vec_data.get("similarity", 0.0),
                "fts_relevance": fts_data.get("relevance", 0.0),
            }

        # Convert to result list and apply filters
        results: list[dict[str, Any]] = []
        now = datetime.now()

        for finding_id, data in merged.items():
            row = data["row"]
            if row is None:
                continue

            # Apply filters
            if not include_superseded and row["status"] == "superseded":
                continue
            if project and row["project"] != project:
                continue
            if finding_type and row["type"] != finding_type:
                continue

            # Tag filtering (all specified tags must match)
            if tags:
                finding_tags = json.loads(row["tags"] or "[]")
                if not all(t in finding_tags for t in tags):
                    continue

            # Date filtering
            if after and row["created_at"] < after:
                continue
            if before and row["created_at"] > before:
                continue

            # Calculate recency boost
            try:
                created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                age_days = (now - created.replace(tzinfo=None)).days
                recency_factor = 1 + recency_weight * (2.718 ** (-age_days / 180))
            except (ValueError, TypeError):
                recency_factor = 1.0

            # Combined score
            base_score = data["rrf_score"] if hybrid else data["vector_sim"]
            final_score = base_score * recency_factor

            results.append({
                "id": row["id"],
                "type": row["type"],
                "status": row["status"],
                "supersedes_id": row["supersedes_id"],
                "project": row["project"],
                "sprint": row["sprint"],
                "tags": json.loads(row["tags"] or "[]"),
                "summary": row["summary"],
                "content": row["content"],
                "similarity": data["vector_sim"],
                "score": final_score,
            })

        # Post-process results: filter and re-rank
        if exclude_corrections:
            results = [r for r in results if not str(r["content"]).startswith("[CORRECTION:")]

        if deprioritize_index:
            for r in results:
                content_lower = str(r["content"]).lower()
                result_tags = r.get("tags", [])
                is_list = isinstance(result_tags, list)
                is_index = (
                    content_lower.startswith("index:")
                    or (is_list and "entry-point" in result_tags)
                    or (is_list and "index" in result_tags)
                )
                if is_index:
                    r["score"] = r["score"] * 0.7

        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:limit]

    def related(
        self,
        finding_id: str,
        limit: int = 5,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Find findings related to a given finding by embedding similarity."""
        # Get the finding's embedding
        row = self.conn.execute(
            "SELECT embedding FROM findings_vec WHERE id = ?",
            (finding_id,)
        ).fetchone()

        if not row:
            return []

        embedding = row[0]

        # Search for similar findings
        sql = """
            SELECT f.*, v.distance
            FROM findings f
            JOIN findings_vec v ON f.id = v.id
            WHERE v.embedding MATCH ?
            AND k = ?
            AND f.id != ?
        """
        params: list[object] = [embedding, limit + 5, finding_id]

        if not include_superseded:
            sql = sql.replace("AND f.id != ?", "AND f.status = 'current' AND f.id != ?")

        rows = self.conn.execute(sql, params).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            distance = float(row["distance"])
            results.append({
                "id": row["id"],
                "type": row["type"],
                "status": row["status"],
                "project": row["project"],
                "content": row["content"][:200],
                "similarity": 1 - (distance ** 2) / 2,
            })

        return results[:limit]

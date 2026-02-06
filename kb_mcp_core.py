#!/usr/bin/env python3
"""
Knowledge Base MCP Server — Core Tools

Lightweight server with 6 essential tools for everyday KB operations.
Advanced tools (notation, documents, scripts, errors, LLM helpers)
available in kb_mcp_advanced.py.

Tools: kb_add, kb_search, kb_correct, kb_list, kb_get, kb_stats
"""

import asyncio
from typing import Annotated

from mcp.server.fastmcp import FastMCP

from kb import KnowledgeBase, FINDING_TYPES, validate_finding_content

mcp = FastMCP(
    name="knowledge-base",
    instructions="""Knowledge Base — core tools for recording and retrieving findings.

Use kb_add to record successes, failures, experiments, and discoveries.
Use kb_search to find relevant past findings before attempting something.
Use kb_correct to fix outdated/wrong findings (creates supersession chain).
Use kb_list to see findings filtered by project/sprint/type.

Finding types: success, failure, experiment, discovery, correction

IMPORTANT: Always search before trying something that might already be recorded!
""",
)

_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


_FINDING_FIELDS = {
    "required": {"content"},
    "optional": {"type", "project", "sprint", "tags", "evidence",
                 "assumptions", "method", "verified", "implications",
                 "constraints", "open_questions", "caveats",
                 "supersedes", "correction_reason"},
}


def _format_structured_content(data: dict) -> str:
    parts = [data["content"]]
    if data.get("assumptions"):
        parts.append("\n**Assumptions:**")
        for a in data["assumptions"]:
            parts.append(f"- {a}")
    if data.get("method"):
        parts.append(f"\n**Method:** {data['method']}")
    if data.get("verified"):
        parts.append("\n**Status:** VERIFIED")
    if data.get("constraints"):
        parts.append("\n**Constraints:**")
        for c in data["constraints"]:
            parts.append(f"- {c}")
    if data.get("implications"):
        parts.append("\n**Implications:**")
        for i in data["implications"]:
            parts.append(f"- {i}")
    if data.get("open_questions"):
        parts.append("\n**Open Questions:**")
        for q in data["open_questions"]:
            parts.append(f"- {q}")
    if data.get("caveats"):
        parts.append("\n**Caveats:**")
        for c in data["caveats"]:
            parts.append(f"- {c}")
    return "\n".join(parts)


@mcp.tool()
def kb_add(
    content: Annotated[str, "Main finding text (required)"],
    finding_type: Annotated[str | None, "success|failure|discovery|experiment|correction"] = None,
    project: Annotated[str | None, "Project name"] = None,
    sprint: Annotated[str | None, "Sprint identifier"] = None,
    tags: Annotated[str | None, "Comma-separated tags or JSON array"] = None,
    evidence: Annotated[str | None, "Supporting evidence"] = None,
    assumptions: Annotated[list[str] | None, "Array of assumptions made"] = None,
    method: Annotated[str | None, "How this was verified/discovered"] = None,
    verified: Annotated[bool | None, "True if verified"] = None,
    implications: Annotated[list[str] | None, "Array of consequences"] = None,
    constraints: Annotated[list[str] | None, "Array of conditions/requirements"] = None,
    open_questions: Annotated[list[str] | None, "Array of remaining unknowns"] = None,
    caveats: Annotated[list[str] | None, "Array of warnings/limitations"] = None,
    supersedes: Annotated[str | None, "ID of finding to correct (creates correction)"] = None,
    correction_reason: Annotated[str | None, "Why correction is needed"] = None,
) -> str:
    """Record a finding in the knowledge base with explicit structural fields.

    Use structural fields (assumptions, method, verified, implications, constraints,
    open_questions, caveats) instead of embedding this information in content text.
    """
    data: dict = {"content": content}
    if finding_type:
        data["type"] = finding_type
    if project:
        data["project"] = project
    if sprint:
        data["sprint"] = sprint
    if tags:
        data["tags"] = tags
    if evidence:
        data["evidence"] = evidence
    if assumptions:
        data["assumptions"] = assumptions
    if method:
        data["method"] = method
    if verified:
        data["verified"] = verified
    if implications:
        data["implications"] = implications
    if constraints:
        data["constraints"] = constraints
    if open_questions:
        data["open_questions"] = open_questions
    if caveats:
        data["caveats"] = caveats
    if supersedes:
        data["supersedes"] = supersedes
    if correction_reason:
        data["correction_reason"] = correction_reason

    if "content" not in data:
        return "Error: Missing required field 'content'"

    all_fields = _FINDING_FIELDS["required"] | _FINDING_FIELDS["optional"]
    unknown = set(data.keys()) - all_fields
    if unknown:
        return f"Error: Unknown field(s): {', '.join(sorted(unknown))}. Valid fields: {', '.join(sorted(all_fields))}"

    finding_type = data.get("type")
    if finding_type and finding_type not in FINDING_TYPES:
        return f"Error: Invalid type '{finding_type}'. Must be one of: {', '.join(FINDING_TYPES)}"

    if data.get("supersedes"):
        kb = get_kb()
        try:
            result = kb.correct(
                supersedes_id=data["supersedes"],
                content=_format_structured_content(data),
                reason=data.get("correction_reason"),
                evidence=data.get("evidence"),
            )
            return f"Recorded correction: {result['id']} (supersedes {data['supersedes']})"
        except ValueError as e:
            return f"Error: {e}"

    formatted_content = _format_structured_content(data)
    kb = get_kb()

    tags_list: list[str] | None = None
    if tags:
        tags_list = [t.strip() for t in tags.split(",")]

    try:
        result = kb.add(
            content=formatted_content,
            finding_type=finding_type,
            project=project,
            sprint=sprint,
            tags=tags_list,
            evidence=evidence,
        )

        output = [f"Recorded finding: {result['id']}"]

        if result.get("type_suggested"):
            f = kb.get(result["id"])
            if f:
                output.append(f"  [auto] Type: {f['type']}")

        if result.get("tags_suggested"):
            f = kb.get(result["id"])
            if f and f.get("tags"):
                output.append(f"  [auto] Tags: {', '.join(f['tags'])}")

        structural = []
        if data.get("assumptions"):
            structural.append(f"{len(data['assumptions'])} assumptions")
        if data.get("method"):
            structural.append("method")
        if data.get("verified"):
            structural.append("verified")
        if data.get("implications"):
            structural.append(f"{len(data['implications'])} implications")
        if data.get("constraints"):
            structural.append(f"{len(data['constraints'])} constraints")
        if data.get("open_questions"):
            structural.append(f"{len(data['open_questions'])} open questions")
        if data.get("caveats"):
            structural.append(f"{len(data['caveats'])} caveats")

        if structural:
            output.append(f"  [structural] {', '.join(structural)}")

        xr = result.get("cross_refs", {})
        if xr.get("findings"):
            output.append("  [related]")
            for rf in xr.get("findings", [])[:2]:
                output.append(f"    - {rf['id']} (sim: {rf['similarity']:.2f})")

        return "\n".join(output)
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def kb_search(
    query: Annotated[str, "Search query (semantic similarity search)"],
    limit: Annotated[int, "Maximum results to return"] = 10,
    project: Annotated[str | None, "Filter by project"] = None,
    finding_type: Annotated[str | None, "Filter by type"] = None,
    include_superseded: Annotated[bool, "Include superseded findings"] = False,
    include_index: Annotated[bool, "Include INDEX/entry-point findings at full rank"] = False,
) -> str:
    """Search findings using semantic similarity.

    Use this BEFORE attempting something that might already be recorded.
    Returns findings ranked by relevance to your query.
    By default, excludes superseded findings and demotes INDEX entries.
    """
    kb = get_kb()

    results = kb.search(
        query=query,
        limit=limit,
        project=project,
        finding_type=finding_type,
        include_superseded=include_superseded,
        deprioritize_index=not include_index,
    )

    if not results:
        return "No findings found matching your query."

    output = [f"Found {len(results)} finding(s):\n"]

    for r in results:
        status = " [SUPERSEDED]" if r["status"] == "superseded" else ""
        meta = []
        if r.get("project"):
            meta.append(f"project={r['project']}")
        if r.get("sprint"):
            meta.append(f"sprint={r['sprint']}")
        meta_str = f" ({', '.join(meta)})" if meta else ""

        sim_str = f" sim={r['similarity']:.3f}" if r.get("similarity") else ""

        output.append(f"[{r['type'].upper()}]{status} {r['id']}{meta_str}{sim_str}")
        display_text = r.get("summary") or "(no summary)"
        output.append(f"  {display_text}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_correct(
    supersedes_id: Annotated[str, "ID of the finding to correct/supersede"],
    content: Annotated[str, "The corrected finding content"],
    reason: Annotated[str | None, "Why the original was wrong"] = None,
    evidence: Annotated[str | None, "Supporting evidence for the correction"] = None,
) -> str:
    """Correct an existing finding by superseding it.

    Use this when you discover a previous finding was wrong or outdated.
    The old finding is marked as superseded, and a new correction is created.
    Future searches will return the correction, not the superseded finding.

    Also warns about any findings that cite the superseded finding (may need review).
    """
    kb = get_kb()

    try:
        result = kb.correct(
            supersedes_id=supersedes_id,
            content=content,
            reason=reason,
            evidence=evidence,
        )
        output = [f"Created correction: {result['id']}", f"Superseded: {supersedes_id}"]

        if result.get("impacted_findings"):
            output.append(f"\n⚠️  {len(result['impacted_findings'])} finding(s) cite the superseded finding:")
            for f in result["impacted_findings"][:5]:
                output.append(f"  - {f['id']}: {f['content'][:60]}...")
            if len(result["impacted_findings"]) > 5:
                output.append(f"  ... and {len(result['impacted_findings']) - 5} more")
            output.append("\nConsider reviewing these findings for consistency.")

        return "\n".join(output)
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def kb_list(
    project: Annotated[str | None, "Filter by project"] = None,
    sprint: Annotated[str | None, "Filter by sprint"] = None,
    finding_type: Annotated[str | None, "Filter by type"] = None,
    limit: Annotated[int, "Maximum results"] = 20,
    include_superseded: Annotated[bool, "Include superseded findings"] = False,
) -> str:
    """List findings with optional filters.

    Use to see all findings for a project or sprint.
    By default shows only current (non-superseded) findings.
    """
    kb = get_kb()

    results = kb.list_findings(
        project=project,
        sprint=sprint,
        finding_type=finding_type,
        limit=limit,
        include_superseded=include_superseded,
    )

    if not results:
        return "No findings found."

    output = [f"Found {len(results)} finding(s):\n"]

    for r in results:
        status = " [SUPERSEDED]" if r["status"] == "superseded" else ""
        meta = []
        if r.get("project"):
            meta.append(f"project={r['project']}")
        if r.get("sprint"):
            meta.append(f"sprint={r['sprint']}")
        meta_str = f" ({', '.join(meta)})" if meta else ""

        output.append(f"[{r['type'].upper()}]{status} {r['id']}{meta_str}")
        display_text = r.get("summary") or r.get("content", "")[:100]
        output.append(f"  {display_text}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_get(
    finding_id: Annotated[str, "The finding ID to retrieve"],
) -> str:
    """Get a specific finding by ID.

    Returns full details formatted as Markdown.
    """
    kb = get_kb()

    finding = kb.get(finding_id)
    if not finding:
        return f"Finding not found: {finding_id}"

    lines = [f"## [{finding['type'].upper()}] {finding['id']}"]

    meta = []
    if finding.get("project"):
        meta.append(f"**Project:** {finding['project']}")
    if finding.get("sprint"):
        meta.append(f"**Sprint:** {finding['sprint']}")
    if finding.get("status") == "superseded":
        meta.append("*SUPERSEDED*")
    if meta:
        lines.append(" | ".join(meta))

    if finding.get("summary"):
        lines.append(f"\n**Summary:** {finding['summary']}")

    lines.append(f"\n### Content\n{finding['content']}")

    if finding.get("evidence"):
        lines.append(f"\n### Evidence\n```\n{finding['evidence']}\n```")

    if finding.get("tags"):
        lines.append(f"\n**Tags:** {', '.join(finding['tags'])}")

    if finding.get("supersedes_id"):
        lines.append(f"\n**Supersedes:** {finding['supersedes_id']}")

    lines.append(f"\n*Created: {finding['created_at']}*")

    return "\n".join(lines)


@mcp.tool()
def kb_stats() -> str:
    """Get knowledge base statistics.

    Shows total findings, breakdown by type and project.
    """
    kb = get_kb()
    stats = kb.stats()

    output = [
        f"Database: {stats['db_path']}",
        f"Total findings: {stats['total']}",
        f"  Current: {stats['current']}",
        f"  Superseded: {stats['superseded']}",
    ]

    if stats["by_type"]:
        output.append("\nBy type:")
        for t, count in sorted(stats["by_type"].items()):
            output.append(f"  {t}: {count}")

    if stats["by_project"]:
        output.append("\nBy project:")
        for p, count in sorted(stats["by_project"].items()):
            output.append(f"  {p}: {count}")

    return "\n".join(output)


# Resources
@mcp.resource("kb://recent")
def get_recent_findings() -> str:
    """Recent findings from the knowledge base (last 20)."""
    kb = get_kb()
    results = kb.list_findings(limit=20)

    if not results:
        return "No findings recorded yet."

    output = ["Recent findings:\n"]
    for r in results:
        meta = []
        if r.get("project"):
            meta.append(r["project"])
        if r.get("sprint"):
            meta.append(r["sprint"])
        meta_str = f" [{', '.join(meta)}]" if meta else ""

        output.append(f"[{r['type'].upper()}] {r['id']}{meta_str}")
        output.append(f"  {r['content'][:200]}...")
        output.append("")

    return "\n".join(output)


@mcp.resource("kb://recent/{project}")
def get_project_findings(project: str) -> str:
    """Recent findings for a specific project."""
    kb = get_kb()
    results = kb.list_findings(project=project, limit=20)

    if not results:
        return f"No findings for project: {project}"

    output = [f"Recent findings for {project}:\n"]
    for r in results:
        sprint_str = f" [sprint={r['sprint']}]" if r.get("sprint") else ""

        output.append(f"[{r['type'].upper()}] {r['id']}{sprint_str}")
        output.append(f"  {r['content'][:200]}...")
        output.append("")

    return "\n".join(output)


@mcp.resource("kb://stats")
def get_stats_resource() -> str:
    """Knowledge base statistics."""
    return kb_stats()


def main():
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()

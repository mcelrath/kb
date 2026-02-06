#!/usr/bin/env python3
"""
Knowledge Base MCP Server

Exposes the knowledge base as MCP tools and resources for Claude integration.

Tools:
  - kb_add: Record a new finding
  - kb_search: Search findings with vector similarity
  - kb_correct: Correct/supersede an existing finding
  - kb_list: List findings with filters
  - kb_get: Get a specific finding by ID
  - kb_stats: Get database statistics

Resources:
  - kb://recent: Recent findings (last 20)
  - kb://recent/{project}: Recent findings for a project
"""

import asyncio
from typing import Annotated

from mcp.server.fastmcp import FastMCP

from kb import KnowledgeBase, FINDING_TYPES, NOTATION_DOMAINS, validate_finding_content

# Initialize MCP server
mcp = FastMCP(
    name="knowledge-base",
    instructions="""Knowledge Base for recording and retrieving findings and notation evolution.

Use kb_add to record successes, failures, experiments, and discoveries.
Use kb_search to find relevant past findings before attempting something.
Use kb_correct to fix outdated/wrong findings (creates supersession chain).
Use kb_list to see findings filtered by project/sprint/type.

Finding types:
- success: Verified working approach
- failure: Confirmed non-working (document WHY)
- experiment: Tried but inconclusive
- discovery: New understanding gained
- correction: Supersedes a previous finding

NOTATION TRACKING:
Use kb_notation_add to record a new notation (symbol with meaning).
Use kb_notation_update when notation changes (tracks history: old→new).
Use kb_notation_search to find notations by symbol or meaning.
Use kb_notation_list to list all notations for a project.

Domains: physics, math, cs, general

IMPORTANT: Always search before trying something that might already be recorded!
""",
)

# Lazy-loaded knowledge base instance
_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """Get or create the knowledge base instance."""
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


# ============================================================================
# TOOLS
# ============================================================================


# Valid fields for structured findings
_FINDING_FIELDS = {
    "required": {"content"},
    "optional": {"type", "project", "sprint", "tags", "evidence",
                 "assumptions", "method", "verified", "implications",
                 "constraints", "open_questions", "caveats",
                 "supersedes", "correction_reason"},
}


def _format_structured_content(data: dict) -> str:
    """Format structured JSON into readable content for storage."""
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
    # Build data dict from parameters
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

    # Validate required fields
    if "content" not in data:
        return "Error: Missing required field 'content'"

    # Validate no unknown fields (catches typos)
    all_fields = _FINDING_FIELDS["required"] | _FINDING_FIELDS["optional"]
    unknown = set(data.keys()) - all_fields
    if unknown:
        return f"Error: Unknown field(s): {', '.join(sorted(unknown))}. Valid fields: {', '.join(sorted(all_fields))}"

    # Validate type if provided
    finding_type = data.get("type")
    if finding_type and finding_type not in FINDING_TYPES:
        return f"Error: Invalid type '{finding_type}'. Must be one of: {', '.join(FINDING_TYPES)}"

    # Handle correction case
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

    # Format content with structural fields
    formatted_content = _format_structured_content(data)

    kb = get_kb()

    # Convert tags string to list
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

        # Build response
        output = [f"Recorded finding: {result['id']}"]

        if result.get("type_suggested"):
            f = kb.get(result["id"])
            if f:
                output.append(f"  [auto] Type: {f['type']}")

        if result.get("tags_suggested"):
            f = kb.get(result["id"])
            if f and f.get("tags"):
                output.append(f"  [auto] Tags: {', '.join(f['tags'])}")

        # Show structural fields recorded
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

        # Build similarity string
        sim_str = f" sim={r['similarity']:.3f}" if r.get("similarity") else ""

        output.append(f"[{r['type'].upper()}]{status} {r['id']}{meta_str}{sim_str}")
        # Search returns summary only (content stripped for efficiency)
        display_text = r.get("summary") or "(no summary)"
        output.append(f"  {display_text}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_ask(
    question: Annotated[str, "Natural language question to ask the knowledge base"],
    project: Annotated[str | None, "Filter to specific project"] = None,
    limit: Annotated[int, "Maximum findings to consider for answer"] = 10,
) -> str:
    """Ask a natural language question and get an answer from the knowledge base.

    Searches for relevant findings and synthesizes an answer using LLM.
    Cites sources by their finding IDs.
    """
    kb = get_kb()

    result = kb.ask(
        question=question,
        project=project,
        limit=limit,
        verbose=False,
    )

    output = [result["answer"], ""]

    if result["sources"]:
        output.append("Sources:")
        for src in result["sources"]:
            sim = src.get("similarity", 0)
            output.append(f"  [{src['id']}] ({sim:.2f}) {src['content']}")

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

        # Warn about impacted findings
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
def kb_bulk_tag(
    finding_ids: Annotated[list[str], "List of finding IDs to tag"],
    tags: Annotated[list[str], "Tags to add to all findings"],
) -> str:
    """Add tags to multiple findings at once.

    Merges new tags with existing tags on each finding.
    Useful for batch tagging findings that share a common attribute.
    """
    kb = get_kb()

    if not finding_ids:
        return "Error: No finding IDs provided"
    if not tags:
        return "Error: No tags provided"

    result = kb.bulk_add_tags(finding_ids, tags)
    output = [f"Updated: {result['updated']} findings"]
    if result["skipped"] > 0:
        output.append(f"Skipped: {result['skipped']} (not found)")
    return "\n".join(output)


@mcp.tool()
def kb_bulk_consolidate(
    finding_ids: Annotated[list[str], "List of finding IDs to consolidate"],
    summary: Annotated[str, "Content for the consolidated finding"],
    reason: Annotated[str, "Why these findings are being merged"],
    finding_type: Annotated[str, "Type for new finding"] = "discovery",
    tags: Annotated[list[str] | None, "Tags (default: merge from sources)"] = None,
    evidence: Annotated[str | None, "Evidence text"] = None,
) -> str:
    """Consolidate multiple findings into a single finding.

    Supersedes all source findings and creates a new consolidated finding.
    Tags are merged from source findings unless explicitly specified.
    """
    kb = get_kb()

    if not finding_ids:
        return "Error: No finding IDs provided"
    if len(finding_ids) < 2:
        return "Error: Need at least 2 findings to consolidate"

    try:
        result = kb.consolidate_cluster(
            finding_ids=finding_ids,
            summary=summary,
            reason=reason,
            finding_type=finding_type,
            tags=tags,
            evidence=evidence,
        )
        output = [
            f"Created: {result['new_id']}",
            f"Superseded: {result['superseded_count']} findings",
        ]
        if result["skipped"] > 0:
            output.append(f"Skipped: {result['skipped']} (not found or already superseded)")
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

    # Format as Markdown
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


# ============================================================================
# NOTATION TOOLS
# ============================================================================


@mcp.tool()
def kb_notation_add(
    symbol: Annotated[str, "The notation symbol (e.g., 'H_D', '\\slashed{k}')"],
    meaning: Annotated[str, "What the notation means"],
    project: Annotated[str | None, "Project this notation belongs to"] = None,
    domain: Annotated[str | None, "Domain: physics, math, cs, general"] = None,
) -> str:
    """Record a new notation in the knowledge base.

    Use this to track notation conventions in a project.
    When notation changes later, use kb_notation_update to track the evolution.
    """
    if domain and domain not in NOTATION_DOMAINS:
        return f"Error: Invalid domain '{domain}'. Must be one of: {', '.join(NOTATION_DOMAINS)}"

    kb = get_kb()

    try:
        notation_id = kb.notation_add(
            symbol=symbol,
            meaning=meaning,
            project=project,
            domain=domain or "general",
        )
        return f"Recorded notation: {notation_id}\nSymbol: {symbol}\nMeaning: {meaning}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def kb_notation_update(
    new_symbol: Annotated[str, "The new notation symbol"],
    old_symbol: Annotated[str | None, "The old symbol to find and update"] = None,
    notation_id: Annotated[str | None, "The notation ID to update (alternative to old_symbol)"] = None,
    meaning: Annotated[str | None, "Updated meaning (optional)"] = None,
    reason: Annotated[str | None, "Why the notation changed"] = None,
    project: Annotated[str | None, "Project to search in (when using old_symbol)"] = None,
) -> str:
    """Update a notation to a new symbol, tracking the change history.

    Either provide old_symbol (and optionally project) to find the notation,
    or provide notation_id directly. The change is recorded in history.
    """
    if not old_symbol and not notation_id:
        return "Error: Must provide either old_symbol or notation_id"

    kb = get_kb()

    try:
        updated_id = kb.notation_update(
            new_symbol=new_symbol,
            old_symbol=old_symbol,
            notation_id=notation_id,
            meaning=meaning,
            reason=reason,
            project=project,
        )
        return f"Updated notation: {updated_id}\nNew symbol: {new_symbol}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def kb_notation_search(
    query: Annotated[str, "Search query (matches symbol or meaning)"],
    project: Annotated[str | None, "Filter by project"] = None,
    domain: Annotated[str | None, "Filter by domain"] = None,
) -> str:
    """Search for notations by symbol or meaning.

    Use this to find what notation is used for a concept,
    or to check if a symbol is already defined.
    """
    kb = get_kb()
    results = kb.notation_search(query=query, project=project, domain=domain)

    if not results:
        return "No notations found matching your query."

    output = [f"Found {len(results)} notation(s):\n"]

    for n in results:
        meta = []
        if n.get("project"):
            meta.append(f"project={n['project']}")
        if n.get("domain"):
            meta.append(f"domain={n['domain']}")
        meta_str = f" ({', '.join(meta)})" if meta else ""

        output.append(f"{n['id']}{meta_str}")
        output.append(f"  Symbol: {n['current_symbol']}")
        output.append(f"  Meaning: {n['meaning']}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_notation_list(
    project: Annotated[str | None, "Filter by project"] = None,
    domain: Annotated[str | None, "Filter by domain"] = None,
) -> str:
    """List all notations, optionally filtered by project or domain."""
    kb = get_kb()
    results = kb.notation_list(project=project, domain=domain)

    if not results:
        return "No notations recorded."

    output = [f"Found {len(results)} notation(s):\n"]

    for n in results:
        meta = []
        if n.get("project"):
            meta.append(n["project"])
        if n.get("domain"):
            meta.append(n["domain"])
        meta_str = f" [{', '.join(meta)}]" if meta else ""

        output.append(f"{n['current_symbol']} → {n['meaning']}{meta_str}")
        output.append(f"  ID: {n['id']}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_notation_history(
    notation_id: Annotated[str, "The notation ID to get history for"],
) -> str:
    """Get the change history for a notation.

    Shows all symbol changes over time with reasons.
    """
    kb = get_kb()
    history = kb.notation_history(notation_id)

    if not history:
        return f"No history found for notation: {notation_id}"

    output = [f"History for {notation_id}:\n"]

    for h in history:
        output.append(f"{h['changed_at']}: {h['old_symbol']} → {h['new_symbol']}")
        if h.get("reason"):
            output.append(f"  Reason: {h['reason']}")
        output.append("")

    return "\n".join(output)


# ============================================================================
# ERROR TRACKING TOOLS
# ============================================================================


@mcp.tool()
def kb_error_add(
    signature: Annotated[str, "The error signature/message to record"],
    error_type: Annotated[str | None, "Error type (e.g., build, runtime, test)"] = None,
    project: Annotated[str | None, "Project name"] = None,
) -> str:
    """Record an error signature in the knowledge base with auto-normalization.

    If the error already exists (same signature + project), increments occurrence count.
    Error signatures are automatically normalized using LLM to improve matching.
    """
    kb = get_kb()

    try:
        result = kb.error_add(
            signature=signature,
            error_type=error_type,
            project=project,
        )

        output = []
        if result.get("is_new"):
            output.append(f"Recorded new error: {result['id']}")
        else:
            output.append(f"Updated existing error: {result['id']} (occurrence #{result['occurrence_count']})")

        if result.get("normalized"):
            output.append(f"  [auto] Signature normalized")
            output.append(f"  Original: {result['original_signature'][:80]}...")

        return "\n".join(output)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def kb_error_link(
    error_id: Annotated[str, "The error ID to link"],
    finding_id: Annotated[str, "The finding ID (solution) to link to"],
    verified: Annotated[bool, "Mark the solution as verified"] = False,
) -> str:
    """Link an error to a solution (finding).

    Use this to associate a recorded error with a finding that solves it.
    Mark as verified if the solution has been confirmed to work.
    """
    kb = get_kb()

    if kb.error_link(error_id, finding_id, verified=verified):
        verified_msg = " (marked as verified)" if verified else ""
        return f"Linked: {error_id} → {finding_id}{verified_msg}"
    else:
        return "Link already exists or invalid IDs"


@mcp.tool()
def kb_error_verify(
    error_id: Annotated[str, "The error ID"],
    finding_id: Annotated[str, "The finding ID to verify"],
) -> str:
    """Mark a solution as verified for an error.

    Use this when you confirm that a linked solution actually fixes the error.
    """
    kb = get_kb()

    if kb.error_verify(error_id, finding_id):
        return f"Verified: {error_id} → {finding_id}"
    else:
        return "Link not found"


@mcp.tool()
def kb_error_get(
    error_id: Annotated[str, "The error ID to retrieve"],
) -> str:
    """Get an error by ID with all linked solutions.

    Returns full details including occurrence count and linked solutions.
    """
    kb = get_kb()
    error = kb.error_get(error_id)

    if not error:
        return f"Error not found: {error_id}"

    output = [
        f"ID: {error['id']}",
        f"Signature: {error['signature']}",
    ]
    if error.get("error_type"):
        output.append(f"Type: {error['error_type']}")
    if error.get("project"):
        output.append(f"Project: {error['project']}")
    output.append(f"Occurrences: {error['occurrence_count']}")
    output.append(f"First seen: {error['first_seen']}")
    output.append(f"Last seen: {error['last_seen']}")

    if error["solutions"]:
        output.append(f"\nSolutions ({len(error['solutions'])}):")
        for s in error["solutions"]:
            verified = " [VERIFIED]" if s["verified"] else ""
            output.append(f"  [{s['type'].upper()}]{verified} {s['finding_id']}")
            output.append(f"    {s['content'][:100]}...")

    return "\n".join(output)


@mcp.tool()
def kb_error_search(
    query: Annotated[str, "Search query (matches error signature)"],
    project: Annotated[str | None, "Filter by project"] = None,
) -> str:
    """Search for errors by signature pattern.

    Use this to find previously recorded errors matching a pattern.
    """
    kb = get_kb()
    results = kb.error_search(query=query, project=project)

    if not results:
        return "No errors found matching your query."

    output = [f"Found {len(results)} error(s):\n"]

    for e in results:
        meta = f" [{e['project']}]" if e.get("project") else ""
        output.append(f"{e['id']}{meta} (×{e['occurrence_count']})")
        output.append(f"  {e['signature'][:100]}...")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_error_list(
    project: Annotated[str | None, "Filter by project"] = None,
    error_type: Annotated[str | None, "Filter by error type"] = None,
    limit: Annotated[int, "Maximum results"] = 20,
) -> str:
    """List recorded errors with optional filters.

    Use to see what errors have been tracked for a project.
    """
    kb = get_kb()
    results = kb.error_list(project=project, error_type=error_type, limit=limit)

    if not results:
        return "No errors recorded."

    output = [f"Found {len(results)} error(s):\n"]

    for e in results:
        meta = f" [{e['project']}]" if e.get("project") else ""
        output.append(f"{e['id']}{meta} (×{e['occurrence_count']})")
        output.append(f"  {e['signature'][:100]}...")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_error_solutions(
    error_id: Annotated[str, "The error ID to get solutions for"],
) -> str:
    """Get all solutions linked to an error.

    Returns solutions sorted by verification status (verified first).
    """
    kb = get_kb()
    solutions = kb.error_solutions(error_id)

    if not solutions:
        return f"No solutions for: {error_id}"

    output = [f"Solutions for {error_id}:\n"]

    for s in solutions:
        verified = " [VERIFIED]" if s["verified"] else ""
        output.append(f"[{s['type'].upper()}]{verified} {s['finding_id']}")
        output.append(f"  {s['content'][:100]}...")
        output.append("")

    return "\n".join(output)


# ============================================================================
# DOCUMENT TRACKING TOOLS
# ============================================================================


@mcp.tool()
def kb_doc_add(
    title: Annotated[str, "Document title"],
    doc_type: Annotated[str, "Type: spec, paper, standard, internal, reference"],
    url: Annotated[str | None, "URL or file path"] = None,
    project: Annotated[str | None, "Project name"] = None,
    summary: Annotated[str | None, "Brief description of the document"] = None,
) -> str:
    """Add an authoritative document to the knowledge base.

    Use this to track specs, papers, standards, and reference documents.
    Link findings to documents using kb_doc_cite.
    """
    valid_types = ["spec", "paper", "standard", "internal", "reference"]
    if doc_type not in valid_types:
        return f"Error: Invalid type '{doc_type}'. Must be one of: {', '.join(valid_types)}"

    kb = get_kb()

    try:
        doc_id = kb.doc_add(
            title=title,
            doc_type=doc_type,
            url=url,
            project=project,
            summary=summary,
        )
        return f"Added document: {doc_id}\nTitle: {title}\nType: {doc_type}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def kb_doc_get(
    doc_id: Annotated[str, "The document ID to retrieve"],
) -> str:
    """Get a document by ID with citation count.

    Returns full details including summary and supersession info.
    """
    kb = get_kb()
    doc = kb.doc_get(doc_id)

    if not doc:
        return f"Document not found: {doc_id}"

    output = [
        f"ID: {doc['id']}",
        f"Title: {doc['title']}",
        f"Type: {doc['doc_type']}",
    ]
    if doc.get("url"):
        output.append(f"URL: {doc['url']}")
    if doc.get("project"):
        output.append(f"Project: {doc['project']}")
    output.append(f"Status: {doc['status']}")
    if doc.get("summary"):
        output.append(f"Summary: {doc['summary']}")
    output.append(f"Citations: {doc['citation_count']}")
    output.append(f"Created: {doc['created_at']}")
    if doc.get("superseded_by"):
        output.append(f"Superseded by: {doc['superseded_by']}")

    return "\n".join(output)


@mcp.tool()
def kb_doc_list(
    project: Annotated[str | None, "Filter by project"] = None,
    doc_type: Annotated[str | None, "Filter by type"] = None,
    include_superseded: Annotated[bool, "Include superseded documents"] = False,
    limit: Annotated[int, "Maximum results"] = 50,
) -> str:
    """List documents with optional filters.

    Use to see what authoritative documents are tracked for a project.
    """
    kb = get_kb()
    results = kb.doc_list(
        project=project,
        doc_type=doc_type,
        include_superseded=include_superseded,
        limit=limit,
    )

    if not results:
        return "No documents found."

    output = [f"Found {len(results)} document(s):\n"]

    for d in results:
        meta = [d["doc_type"]]
        if d.get("project"):
            meta.append(d["project"])
        if d["status"] != "active":
            meta.append(d["status"].upper())
        output.append(f"{d['id']} [{', '.join(meta)}]")
        output.append(f"  {d['title']}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_doc_search(
    query: Annotated[str, "Search query (matches title or summary)"],
    project: Annotated[str | None, "Filter by project"] = None,
) -> str:
    """Search documents by title or summary.

    Use to find relevant authoritative documents.
    """
    kb = get_kb()
    results = kb.doc_search(query=query, project=project)

    if not results:
        return "No documents found."

    output = [f"Found {len(results)} document(s):\n"]

    for d in results:
        meta = [d["doc_type"]]
        if d.get("project"):
            meta.append(d["project"])
        output.append(f"{d['id']} [{', '.join(meta)}]")
        output.append(f"  {d['title']}")
        if d.get("summary"):
            output.append(f"  {d['summary'][:80]}...")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_doc_cite(
    finding_id: Annotated[str, "The finding ID that cites the document"],
    doc_id: Annotated[str, "The document ID being cited"],
    citation_type: Annotated[str, "Type: references, implements, contradicts, extends"] = "references",
    notes: Annotated[str | None, "Citation notes"] = None,
) -> str:
    """Link a finding to a document it cites.

    Use this to establish relationships between findings and authoritative documents.
    Citation types:
    - references: The finding references this document
    - implements: The finding implements something from this document
    - contradicts: The finding contradicts this document
    - extends: The finding extends/builds on this document
    """
    valid_types = ["references", "implements", "contradicts", "extends"]
    if citation_type not in valid_types:
        return f"Error: Invalid type '{citation_type}'. Must be one of: {', '.join(valid_types)}"

    kb = get_kb()

    if kb.doc_cite(finding_id, doc_id, citation_type, notes):
        return f"Linked: {finding_id} → {doc_id}\nType: {citation_type}"
    else:
        return "Citation already exists or invalid IDs"


@mcp.tool()
def kb_doc_citations(
    doc_id: Annotated[str, "The document ID to get citations for"],
) -> str:
    """Get all findings that cite a document.

    Shows which findings reference, implement, contradict, or extend this document.
    """
    kb = get_kb()
    citations = kb.doc_citations(doc_id)

    if not citations:
        return f"No citations for: {doc_id}"

    output = [f"Findings citing {doc_id}:\n"]

    for c in citations:
        output.append(f"[{c['type'].upper()}] {c['finding_id']} ({c['citation_type']})")
        output.append(f"  {c['content'][:100]}...")
        if c.get("notes"):
            output.append(f"  Notes: {c['notes']}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_doc_finding_docs(
    finding_id: Annotated[str, "The finding ID to get cited documents for"],
) -> str:
    """Get all documents cited by a finding.

    Shows which authoritative documents a finding references.
    """
    kb = get_kb()
    docs = kb.finding_docs(finding_id)

    if not docs:
        return f"No documents cited by: {finding_id}"

    output = [f"Documents cited by {finding_id}:\n"]

    for d in docs:
        output.append(f"[{d['doc_type'].upper()}] {d['document_id']} ({d['citation_type']})")
        output.append(f"  {d['title']}")
        if d.get("url"):
            output.append(f"  URL: {d['url']}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_doc_supersede(
    doc_id: Annotated[str, "The document ID to supersede"],
    new_doc_id: Annotated[str, "The document ID that supersedes it"],
) -> str:
    """Mark a document as superseded by another.

    Use this when a newer version of a document replaces an old one.
    """
    kb = get_kb()

    if kb.doc_supersede(doc_id, new_doc_id):
        return f"Superseded: {doc_id} → {new_doc_id}"
    else:
        return f"Document not found: {doc_id}"


# ============================================================================
# SCRIPT REGISTRY
# ============================================================================


@mcp.tool()
def kb_script_add(
    file_path: Annotated[str, "Path to the script file"],
    purpose: Annotated[str, "Description of what the script does/tests"],
    project: Annotated[str | None, "Project name"] = None,
    language: Annotated[str | None, "Language (python, sage, bash, other)"] = None,
) -> str:
    """Register a script in the knowledge base.

    Use this to track hypothesis-testing scripts, analysis code, and computations.
    Scripts are indexed by content hash and searchable by purpose.
    """
    from pathlib import Path

    kb = get_kb()
    path = Path(file_path)

    if not path.exists():
        return f"File not found: {file_path}"

    script_id = kb.script_add(
        path=str(path.resolve()),
        purpose=purpose,
        project=project,
        language=language,
    )

    return f"Registered: {script_id}\n  File: {path.name}\n  Purpose: {purpose}"


@mcp.tool()
def kb_script_get(
    script_id: Annotated[str, "The script ID to retrieve"],
) -> str:
    """Get details of a registered script."""
    kb = get_kb()
    result = kb.script_get(script_id)

    if not result:
        return f"Script not found: {script_id}"

    output = [
        f"ID: {result['id']}",
        f"File: {result['filename']}",
        f"Path: {result['path']}",
        f"Purpose: {result['purpose']}",
    ]
    if result.get("project"):
        output.append(f"Project: {result['project']}")
    output.append(f"Language: {result['language']}")
    output.append(f"Created: {result['created_at']}")

    return "\n".join(output)


@mcp.tool()
def kb_script_search(
    query: Annotated[str, "Search query for script purposes"],
    project: Annotated[str | None, "Filter by project"] = None,
    limit: Annotated[int, "Maximum results to return"] = 10,
) -> str:
    """Search for scripts by purpose using semantic similarity.

    Use this to find scripts related to a topic or computation.
    """
    kb = get_kb()
    results = kb.script_search(query, project=project, limit=limit)

    if not results:
        return f"No scripts found matching: {query}"

    output = [f"Found {len(results)} scripts:\n"]
    for r in results:
        proj_str = f" [{r['project']}]" if r.get("project") else ""
        output.append(f"  {r['id']} (sim: {r['similarity']:.2f}){proj_str}")
        output.append(f"    {r['filename']}: {r['purpose'][:60]}...")

    return "\n".join(output)


@mcp.tool()
def kb_script_list(
    project: Annotated[str | None, "Filter by project"] = None,
    language: Annotated[str | None, "Filter by language"] = None,
    limit: Annotated[int, "Maximum results to return"] = 50,
) -> str:
    """List registered scripts with optional filters."""
    kb = get_kb()
    results = kb.script_list(project=project, language=language, limit=limit)

    if not results:
        return "No scripts registered."

    output = [f"Found {len(results)} scripts:\n"]
    for r in results:
        meta = []
        if r.get("project"):
            meta.append(r["project"])
        meta.append(r["language"])
        meta_str = f" [{', '.join(meta)}]"

        output.append(f"  {r['id']}{meta_str}")
        output.append(f"    {r['filename']}: {r['purpose'][:60]}...")

    return "\n".join(output)


@mcp.tool()
def kb_script_link_finding(
    script_id: Annotated[str, "The script ID"],
    finding_id: Annotated[str, "The finding ID to link"],
    relationship: Annotated[str, "Relationship type: generated_by, validates, contradicts"] = "generated_by",
) -> str:
    """Link a finding to a script that generated or validates it.

    Relationship types:
    - generated_by: The script produced this finding
    - validates: The script validates/confirms this finding
    - contradicts: The script contradicts this finding
    """
    kb = get_kb()

    if kb.script_link_finding(script_id, finding_id, relationship):
        return f"Linked: {finding_id} --[{relationship}]--> {script_id}"
    else:
        return f"Failed to link: script or finding not found"


@mcp.tool()
def kb_script_findings(
    script_id: Annotated[str, "The script ID"],
) -> str:
    """Get all findings linked to a script."""
    kb = get_kb()
    results = kb.script_findings(script_id)

    if not results:
        return f"No findings linked to script: {script_id}"

    output = [f"Findings for {script_id}:\n"]
    for r in results:
        output.append(f"  [{r['relationship']}] {r['id']} ({r['type']})")
        output.append(f"    {r['content'][:80]}...")

    return "\n".join(output)


@mcp.tool()
def kb_script_delete(
    script_id: Annotated[str, "The script ID to delete"],
) -> str:
    """Delete a registered script."""
    kb = get_kb()

    if kb.script_delete(script_id):
        return f"Deleted: {script_id}"
    else:
        return f"Script not found: {script_id}"


# ============================================================================
# DELETE OPERATIONS
# ============================================================================


@mcp.tool()
def kb_delete(
    finding_id: Annotated[str, "The finding ID to delete"],
) -> str:
    """Delete a finding from the knowledge base.

    Use with caution - prefer kb_correct to preserve history.
    """
    kb = get_kb()

    if kb.delete(finding_id):
        return f"Deleted: {finding_id}"
    else:
        return f"Finding not found: {finding_id}"


@mcp.tool()
def kb_notation_delete(
    notation_id: Annotated[str, "The notation ID to delete"],
) -> str:
    """Delete a notation from the knowledge base."""
    kb = get_kb()

    if kb.notation_delete(notation_id):
        return f"Deleted: {notation_id}"
    else:
        return f"Notation not found: {notation_id}"


@mcp.tool()
def kb_error_delete(
    error_id: Annotated[str, "The error ID to delete"],
) -> str:
    """Delete an error from the knowledge base."""
    kb = get_kb()

    if kb.error_delete(error_id):
        return f"Deleted: {error_id}"
    else:
        return f"Error not found: {error_id}"


@mcp.tool()
def kb_doc_delete(
    doc_id: Annotated[str, "The document ID to delete"],
) -> str:
    """Delete a document from the knowledge base."""
    kb = get_kb()

    if kb.doc_delete(doc_id):
        return f"Deleted: {doc_id}"
    else:
        return f"Document not found: {doc_id}"


# ============================================================================
# LLM-POWERED ANALYSIS
# ============================================================================


@mcp.tool()
def kb_suggest_tags(
    content: Annotated[str, "Content to analyze for tag suggestions"],
    project: Annotated[str | None, "Project for context"] = None,
) -> str:
    """Suggest tags for content using LLM analysis.

    Analyzes content and suggests 2-5 relevant tags based on:
    - Existing tags in the project
    - Status indicators (proven, heuristic, open-problem)
    - Importance markers (core-result, technique, detail)
    """
    kb = get_kb()
    tags = kb.suggest_tags(content, project=project)

    if tags:
        return f"Suggested tags: {', '.join(tags)}"
    else:
        return "No tags suggested (LLM unavailable or analysis failed)"


@mcp.tool()
def kb_classify_type(
    content: Annotated[str, "Content to classify"],
) -> str:
    """Classify finding type from content using LLM.

    Returns one of: success, failure, discovery, experiment
    """
    kb = get_kb()
    finding_type = kb.classify_finding_type(content)
    return f"Suggested type: {finding_type}"


@mcp.tool()
def kb_detect_duplicates(
    content: Annotated[str, "Content to check for duplicates"],
    project: Annotated[str | None, "Project filter"] = None,
    threshold: Annotated[float, "Similarity threshold (0.0-1.0)"] = 0.85,
) -> str:
    """Check if similar findings already exist before adding.

    Uses vector similarity + LLM confirmation to find semantic duplicates.
    """
    kb = get_kb()
    duplicates = kb.detect_duplicates(content, project=project, threshold=threshold)

    if not duplicates:
        return "No duplicates found"

    output = [f"Found {len(duplicates)} potential duplicate(s):"]
    for d in duplicates:
        output.append(f"  {d['id']} (sim: {d.get('similarity', 0):.2f})")
        output.append(f"    {d['content'][:100]}...")

    return "\n".join(output)


@mcp.tool()
def kb_normalize_error(
    error_text: Annotated[str, "Error text to normalize"],
) -> str:
    """Normalize an error message to a canonical signature.

    Removes paths, line numbers, addresses and extracts the core error pattern.
    """
    kb = get_kb()
    signature = kb.normalize_error_signature(error_text)
    return f"Normalized signature: {signature}"


@mcp.tool()
def kb_suggest_xrefs(
    finding_id: Annotated[str, "Finding ID to find cross-references for"],
    project: Annotated[str | None, "Project filter"] = None,
) -> str:
    """Suggest cross-references for a finding.

    Finds related findings, scripts, and documents that could be linked.
    """
    kb = get_kb()
    finding = kb.get(finding_id)

    if not finding:
        return f"Finding not found: {finding_id}"

    xrefs = kb.suggest_cross_references(finding_id, finding["content"], project=project)

    output = []
    if xrefs["findings"]:
        output.append("Related findings:")
        for f in xrefs["findings"]:
            output.append(f"  {f['id']} (sim: {f['similarity']:.2f}): {f['content'][:60]}...")

    if xrefs["scripts"]:
        output.append("Related scripts:")
        for s in xrefs["scripts"]:
            output.append(f"  {s['id']}: {s['filename']}")

    if xrefs["docs"]:
        output.append("Related documents:")
        for d in xrefs["docs"]:
            output.append(f"  {d['id']}: {d['title']}")

    if not output:
        return "No cross-references found"

    return "\n".join(output)


@mcp.tool()
def kb_summarize_evidence(
    evidence: Annotated[str, "Evidence text to summarize"],
    max_length: Annotated[int, "Maximum summary length"] = 200,
) -> str:
    """Summarize long evidence text using LLM."""
    kb = get_kb()
    return kb.summarize_evidence(evidence, max_length=max_length)


@mcp.tool()
def kb_detect_notations(
    content: Annotated[str, "Content to analyze for notations"],
    project: Annotated[str | None, "Project for context"] = None,
) -> str:
    """Detect mathematical/physics notations in content.

    Identifies symbols (Greek letters, operators) and their meanings.
    """
    kb = get_kb()
    notations = kb.detect_notations(content, project=project)

    if not notations:
        return "No notations detected"

    output = [f"Found {len(notations)} notation(s):"]
    for n in notations:
        status = "[exists]" if n["exists"] else "[new]"
        output.append(f"  {status} {n['symbol']}: {n['meaning']}")

    return "\n".join(output)


@mcp.tool()
def kb_extract_claims(
    text: Annotated[str, "Text to extract claims from"],
) -> str:
    """Extract factual claims from text using LLM.

    Useful for reconciliation and identifying assertions to verify.
    """
    kb = get_kb()
    claims = kb.extract_claims(text)

    if not claims:
        return "No claims extracted"

    output = [f"Extracted {len(claims)} claim(s):"]
    for i, c in enumerate(claims, 1):
        output.append(f"  {i}. {c}")

    return "\n".join(output)


@mcp.tool()
def kb_suggest_consolidation(
    project: Annotated[str | None, "Project filter"] = None,
    limit: Annotated[int, "Max findings to analyze"] = 50,
) -> str:
    """Find clusters of related findings that might be consolidated.

    Identifies groups of similar findings and suggests whether to merge them.
    """
    kb = get_kb()
    clusters = kb.suggest_consolidation(project=project, limit=limit)

    if not clusters:
        return "No consolidation opportunities found"

    output = [f"Found {len(clusters)} cluster(s) to potentially consolidate:\n"]
    for i, c in enumerate(clusters, 1):
        output.append(f"Cluster {i}:")
        for m in c["members"]:
            output.append(f"  - {m['id']}: {m['content'][:60]}...")
        output.append(f"Analysis: {c['analysis']}\n")

    return "\n".join(output)


@mcp.tool()
def kb_validate(
    project: Annotated[str | None, "Project filter"] = None,
    limit: Annotated[int, "Max findings to check"] = 50,
    use_llm: Annotated[bool, "Use LLM for deeper semantic validation"] = False,
) -> str:
    """Validate existing findings for anti-patterns.

    Checks for:
    - Paper update logs (transient, shouldn't be in KB)
    - File path references (KB should be self-contained)
    - INDEX/GOTCHAS entries (get stale)
    - Nested KB references (findings should be standalone)
    - Specific counts (may become stale)

    With use_llm=True, also performs semantic analysis to catch
    subtle issues that regex patterns miss.

    Returns findings with issues and recommended actions.
    """
    kb = get_kb()
    results = kb.list_findings(project=project, limit=limit)

    if not results:
        return "No findings to validate."

    issues_found = []
    for r in results:
        # Regex-based validation (fast)
        warnings = validate_finding_content(r["content"], r.get("tags"))

        # LLM-based validation (slower but catches more)
        llm_result = None
        if use_llm:
            llm_result = kb.validate_finding_llm(r["content"], r.get("tags"))
            if llm_result.get("issues"):
                for issue in llm_result["issues"]:
                    warnings.append({"type": "llm_detected", "message": issue})

        if warnings or (llm_result and llm_result.get("quality_score", 5) <= 2):
            issues_found.append({
                "id": r["id"],
                "content_preview": r["content"][:80] + "..." if len(r["content"]) > 80 else r["content"],
                "warnings": warnings,
                "quality_score": llm_result.get("quality_score") if llm_result else None,
                "suggestions": llm_result.get("suggestions", []) if llm_result else [],
            })

    if not issues_found:
        return f"Validated {len(results)} findings - no issues found."

    output = [f"Found {len(issues_found)} finding(s) with issues:\n"]
    for issue in issues_found:
        output.append(f"{issue['id']}:")
        output.append(f"  Content: {issue['content_preview']}")
        if issue.get("quality_score"):
            output.append(f"  Quality: {issue['quality_score']}/5")
        for w in issue["warnings"]:
            output.append(f"  [WARNING] {w['message']}")
        for s in issue.get("suggestions", []):
            output.append(f"  [SUGGEST] {s}")
        output.append("")

    output.append("Recommended actions:")
    output.append("  - Use kb_suggest_fix() to get LLM-generated corrections")
    output.append("  - Use kb_correct() to apply fixes")
    output.append("  - Use kb_delete() to remove unfixable findings")

    return "\n".join(output)


@mcp.tool()
def kb_suggest_fix(
    finding_id: Annotated[str, "ID of finding to fix"],
) -> str:
    """Get LLM-suggested corrected content for a problematic finding.

    The LLM analyzes the finding, identifies issues, and generates
    corrected content that removes anti-patterns while preserving
    the substantive information.

    Returns the suggested fix which can then be applied with kb_correct().
    """
    kb = get_kb()
    finding = kb.get(finding_id)

    if not finding:
        return f"Finding not found: {finding_id}"

    # First validate to identify issues
    regex_warnings = validate_finding_content(finding["content"], finding.get("tags"))
    llm_validation = kb.validate_finding_llm(finding["content"], finding.get("tags"))

    all_issues = [w["message"] for w in regex_warnings]
    all_issues.extend(llm_validation.get("issues", []))

    if not all_issues:
        return f"No issues found with {finding_id} - no fix needed."

    # Generate fix
    suggested_fix = kb.suggest_finding_fix(finding["content"], all_issues)

    if not suggested_fix:
        return f"Could not generate fix for {finding_id}. Consider deleting if unfixable."

    output = [
        f"Original ({finding_id}):",
        f"  {finding['content'][:200]}{'...' if len(finding['content']) > 200 else ''}",
        "",
        "Issues identified:",
    ]
    for issue in all_issues:
        output.append(f"  - {issue}")

    output.extend([
        "",
        "Suggested fix:",
        f"  {suggested_fix}",
        "",
        "To apply this fix, run:",
        f'  kb_correct(supersedes_id="{finding_id}", content="<paste suggested fix>")',
    ])

    return "\n".join(output)


@mcp.tool()
def kb_related(
    finding_id: Annotated[str, "ID of finding to find related content for"],
    limit: Annotated[int, "Maximum related findings to return"] = 5,
) -> str:
    """Find findings related to a given finding by embedding similarity.

    Returns findings that are semantically similar to the specified finding.
    """
    kb = get_kb()
    results = kb.related(finding_id, limit=limit)

    if not results:
        return f"No related findings found for {finding_id}"

    output = [f"Findings related to {finding_id}:\n"]
    for r in results:
        sim = r.get("similarity", 0)
        output.append(f"[{r['type']}] {r['id']} (sim={sim:.2f})")
        output.append(f"  {r['content'][:150]}...")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_add_from_template(
    template_name: Annotated[str, "Template name: computation_result, failed_approach, structural_discovery, verification, hypothesis"],
    project: Annotated[str | None, "Project name"] = None,
    tags: Annotated[str | None, "Comma-separated tags"] = None,
    claim: Annotated[str | None, "For computation_result/verification"] = None,
    method: Annotated[str | None, "For computation_result/verification"] = None,
    result: Annotated[str | None, "For computation_result"] = None,
    outcome: Annotated[str | None, "For verification"] = None,
    approach: Annotated[str | None, "For failed_approach"] = None,
    goal: Annotated[str | None, "For failed_approach"] = None,
    reason: Annotated[str | None, "For failed_approach"] = None,
    structure: Annotated[str | None, "For structural_discovery"] = None,
    property: Annotated[str | None, "For structural_discovery"] = None,
    implication: Annotated[str | None, "For structural_discovery"] = None,
    hypothesis: Annotated[str | None, "For hypothesis"] = None,
    motivation: Annotated[str | None, "For hypothesis"] = None,
    status: Annotated[str | None, "For hypothesis"] = None,
    script: Annotated[str | None, "Optional script reference"] = None,
    error: Annotated[str | None, "Optional error message"] = None,
) -> str:
    """Add a finding using a pre-defined template for consistent formatting.

    Templates:
    - computation_result: claim, method, result [script]
    - failed_approach: approach, goal, reason [error]
    - structural_discovery: structure, property, implication
    - verification: claim, method, outcome [script]
    - hypothesis: hypothesis, motivation, status
    """
    kb = get_kb()

    # Build kwargs from provided parameters
    kwargs = {}
    for key, val in [
        ("claim", claim), ("method", method), ("result", result),
        ("outcome", outcome), ("approach", approach), ("goal", goal),
        ("reason", reason), ("structure", structure), ("property", property),
        ("implication", implication), ("hypothesis", hypothesis),
        ("motivation", motivation), ("status", status),
        ("script", script), ("error", error),
    ]:
        if val is not None:
            kwargs[key] = val

    tags_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        result_dict = kb.add_from_template(
            template_name=template_name,
            project=project,
            tags=tags_list,
            **kwargs,
        )
        return f"Added finding: {result_dict['id']}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def kb_review_queue(
    project: Annotated[str | None, "Filter by project"] = None,
    limit: Annotated[int, "Max findings per category"] = 10,
) -> str:
    """Get findings that need attention.

    Returns findings grouped by issue type:
    - untagged: Findings with no tags
    - low_quality: Findings flagged by validation
    - stale: Findings older than 30 days
    - orphaned: Superseded findings with broken references
    """
    kb = get_kb()
    queue = kb.review_queue(project=project, limit=limit)

    output = ["=== Review Queue ===\n"]

    for category, findings in queue.items():
        output.append(f"\n{category.upper()} ({len(findings)}):")
        if not findings:
            output.append("  (none)")
        else:
            for f in findings[:limit]:
                output.append(f"  {f['id']}: {f.get('content', '')[:60]}...")
                if f.get("warnings"):
                    output.append(f"    Warnings: {', '.join(f['warnings'][:2])}")
                if f.get("missing_ref"):
                    output.append(f"    Missing ref: {f['missing_ref']}")

    return "\n".join(output)


@mcp.tool()
def kb_open_questions(
    project: Annotated[str | None, "Filter by project"] = None,
    limit: Annotated[int, "Number of questions to generate"] = 5,
) -> str:
    """Analyze findings to identify knowledge gaps and open questions.

    Uses LLM to analyze existing findings and identify areas lacking coverage,
    unresolved issues, and natural next steps.
    """
    kb = get_kb()
    questions = kb.generate_open_questions(project=project, limit=limit)

    if not questions:
        return "Could not generate open questions. Try adding more findings first."

    output = ["=== Open Questions ===\n"]
    for i, q in enumerate(questions, 1):
        output.append(f"{i}. {q.get('question', 'Unknown')}")
        if q.get("importance"):
            output.append(f"   Why: {q['importance']}")
        if q.get("related_topics"):
            output.append(f"   Topics: {', '.join(q['related_topics'])}")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def kb_check_contradictions(
    content: Annotated[str, "Content to check for contradictions"],
    project: Annotated[str | None, "Filter to specific project"] = None,
) -> str:
    """Check if content contradicts existing findings.

    Uses semantic search + LLM analysis to find potential contradictions.
    Run this before adding a finding to catch conflicts early.
    """
    kb = get_kb()
    contradictions = kb.check_contradictions(content, project=project)

    if not contradictions:
        return "No contradictions found with existing findings."

    output = [f"Found {len(contradictions)} potential contradiction(s):\n"]
    for c in contradictions:
        output.append(f"Contradicts: {c['existing_id']}")
        output.append(f"  Existing: {c['existing_content'][:100]}...")
        output.append(f"  Explanation: {c['explanation']}")
        output.append("")

    output.append("Consider:")
    output.append("  - Using kb_correct() to update the existing finding")
    output.append("  - Modifying your content to resolve the contradiction")

    return "\n".join(output)


@mcp.tool()
def kb_summarize_topic(
    topic: Annotated[str, "Topic to summarize"],
    project: Annotated[str | None, "Filter by project"] = None,
    limit: Annotated[int, "Max findings to consider"] = 20,
) -> str:
    """Synthesize a summary of all findings on a topic.

    Searches for relevant findings and uses LLM to create a coherent
    summary that captures the current state of knowledge.
    """
    kb = get_kb()
    result = kb.summarize_topic(topic, project=project, limit=limit)

    output = [f"=== Summary: {topic} ===\n"]
    output.append(result.get("summary", "No summary available"))
    output.append(f"\n(Based on {result.get('finding_count', 0)} findings)")

    if result.get("sources"):
        output.append("\nTop sources:")
        for s in result["sources"][:5]:
            output.append(f"  - {s['id']} ({s['type']}, sim={s.get('similarity', 0):.2f})")

    return "\n".join(output)


# ============================================================================
# RESOURCES
# ============================================================================


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


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run the MCP server."""
    # Run with stdio transport for Claude Code integration
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()

"""Web page route handlers for kb serve.

Extracted verbatim from kb.py:1675-1801 (R1), then migrated to the
kb.server.renderers template layer (R2 / kb-ez9.3).

render_html_page, render_sidebar, markdown_to_html, and format_finding_markdown
now live in kb.server.renderers — no kb.py import or importlib hack required.
"""

import html as _html
from urllib.parse import quote, urlencode

from starlette.requests import Request
from starlette.responses import HTMLResponse

from ..markdown import format_finding_markdown, markdown_to_html
from .renderers import render_html_page, render_sidebar


def _finding_card(f: dict, meta_html: str) -> str:
    """Render one finding card. EVERY interpolation of finding data is escaped here
    (id/type/content/tags); `meta_html` is the caller's already-escaped meta fragment.
    Centralized so no route hand-builds a card with an unescaped field."""
    fid = _html.escape(str(f["id"]))
    ftype = _html.escape(str(f["type"]))
    raw = f["content"][:200] + "..." if len(f["content"]) > 200 else f["content"]
    summary = _html.escape(raw)
    tags_html = " ".join(
        f'<span class="tag">{_html.escape(t)}</span>'
        for t in f.get("tags", [])[:5]
    )
    return (
        f'<div class="finding">'
        f'<span class="finding-type {ftype}">[{ftype}]</span>'
        f'<a href="/finding/{quote(str(f["id"]), safe="")}">{fid}</a>'
        f'<span class="meta">{meta_html}</span>'
        f'<p>{summary}</p>'
        f'{f"<div>{tags_html}</div>" if tags_html else ""}'
        f'</div>'
    )


def make_web_handlers(kb):
    """Return (index, search_page, finding_page) route handlers bound to kb."""

    async def index(request: Request) -> HTMLResponse:
        page = int(request.query_params.get("page", 1))
        project = request.query_params.get("project", "")
        finding_type = request.query_params.get("type", "")
        tag = request.query_params.get("tag", "")
        include_superseded = request.query_params.get("superseded", "") == "1"

        per_page = 20
        offset = (page - 1) * per_page

        filters = {}
        if project:
            filters["project"] = project
        if finding_type:
            filters["type"] = finding_type
        if include_superseded:
            filters["superseded"] = "1"
        if tag:
            filters["tag"] = tag

        # Get findings with filters
        findings = kb.list_findings(
            limit=per_page + 1,  # +1 to check if more pages
            offset=offset,
            project=project or None,
            finding_type=finding_type or None,
            include_superseded=include_superseded,
            tag=tag or None,
        )

        has_more = len(findings) > per_page
        findings = findings[:per_page]

        # Build HTML
        stats = kb.stats()
        all_tags = kb.get_all_tags(limit=100)
        sidebar = render_sidebar(stats, all_tags, filters)

        items = []
        for f in findings:
            proj = f"({_html.escape(f['project'])})" if f.get("project") else ""
            items.append(_finding_card(f, proj))

        # Pagination — urlencode percent-encodes every value, so reflected query
        # params (project/type/tag) cannot break out of the href attribute.
        pagination = '<div class="pagination">'
        if page > 1:
            prev_params = dict(filters)
            prev_params["page"] = page - 1
            pagination += f'<a href="/?{urlencode(prev_params)}">← Prev</a>'
        if has_more:
            next_params = dict(filters)
            next_params["page"] = page + 1
            pagination += f'<a href="/?{urlencode(next_params)}">Next →</a>'
        pagination += "</div>"

        title = "Findings"
        if project:
            title += f" - {project}"
        if finding_type:
            title += f" [{finding_type}]"

        content = "\n".join(items) + pagination
        return HTMLResponse(render_html_page(title, content, sidebar))

    async def search_page(request: Request) -> HTMLResponse:
        query = request.query_params.get("q", "")
        stats = kb.stats()
        all_tags = kb.get_all_tags(limit=100)
        sidebar = render_sidebar(stats, all_tags, {})

        if query:
            results = kb.search(query, limit=50)
            items = []
            for f in results:
                proj = f"({_html.escape(f['project'])})" if f.get("project") else ""
                score = f.get("score", 0)
                sim = f.get("similarity", 0)
                meta = f"score={score:.4f} sim={sim:.3f} {proj}"
                items.append(_finding_card(f, meta))

            content = f"""<form class="search-form" method="get">
                        <input type="text" name="q" value="{_html.escape(query)}" placeholder="Search findings...">
                        <button type="submit">Search</button>
                    </form>
                    <p>Found {len(results)} result(s)</p>
                    {''.join(items)}"""
        else:
            content = """<form class="search-form" method="get">
                        <input type="text" name="q" placeholder="Search findings...">
                        <button type="submit">Search</button>
                    </form>"""

        return HTMLResponse(render_html_page("Search", content, sidebar))

    async def finding_page(request: Request) -> HTMLResponse:
        finding_id = request.path_params["id"]
        finding = kb.get(finding_id)

        if not finding:
            return HTMLResponse(
                render_html_page("Not Found", "<p>Finding not found.</p>"), status_code=404
            )

        stats = kb.stats()
        all_tags = kb.get_all_tags(limit=100)
        sidebar = render_sidebar(stats, all_tags, {})

        md = format_finding_markdown(finding)
        content = markdown_to_html(md)
        return HTMLResponse(render_html_page(f"Finding {finding_id}", content, sidebar))

    return index, search_page, finding_page

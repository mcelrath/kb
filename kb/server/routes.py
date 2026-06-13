"""Web page route handlers for kb serve.

Extracted verbatim from kb.py:1675-1801.

render_html_page, render_sidebar, markdown_to_html, and format_finding_markdown
are imported from kb.py (they remain there so the existing imports from tests
and other callers in kb.py keep working).  This module re-exports them for
callers that import from kb.server.routes.
"""

import html as _html

from starlette.requests import Request
from starlette.responses import HTMLResponse

# Import the HTML renderers and markdown formatter from the top-level kb.py.
# These are module-level functions defined there and are safe to import.
# kb.py is the CLI entry-point script; it is importable because it has no
# if __name__ == "__main__" guard around the function definitions.
import importlib.util, sys
from pathlib import Path as _Path


def _import_kbpy():
    """Return the kb.py module object (top-level CLI script, not the kb/ package)."""
    mod_name = "_kb_cli_script"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        _Path(__file__).parent.parent.parent / "kb.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't run main() — the module defines only functions at module scope; the
    # if __name__ == "__main__" guard at the bottom keeps main() from firing.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_kbpy = _import_kbpy()

render_html_page = _kbpy.render_html_page
render_sidebar = _kbpy.render_sidebar
markdown_to_html = _kbpy.markdown_to_html
format_finding_markdown = _kbpy.format_finding_markdown


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
            type_class = f["type"]
            raw = f["content"][:200] + "..." if len(f["content"]) > 200 else f["content"]
            summary = _html.escape(raw)
            proj = f"({_html.escape(f['project'])})" if f.get("project") else ""
            tags_html = " ".join(
                f'<span class="tag">{_html.escape(t)}</span>'
                for t in f.get("tags", [])[:5]
            )
            items.append(
                f"""<div class="finding">
                        <span class="finding-type {type_class}">[{f['type']}]</span>
                        <a href="/finding/{f['id']}">{f['id']}</a>
                        <span class="meta">{proj}</span>
                        <p>{summary}</p>
                        {f'<div>{tags_html}</div>' if tags_html else ''}
                    </div>"""
            )

        # Pagination
        pagination = '<div class="pagination">'
        if page > 1:
            prev_params = dict(filters)
            prev_params["page"] = page - 1
            pagination += f'<a href="/?{"&".join(f"{k}={v}" for k,v in prev_params.items())}">← Prev</a>'
        if has_more:
            next_params = dict(filters)
            next_params["page"] = page + 1
            pagination += f'<a href="/?{"&".join(f"{k}={v}" for k,v in next_params.items())}">Next →</a>'
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
                type_class = f["type"]
                raw = f["content"][:200] + "..." if len(f["content"]) > 200 else f["content"]
                summary = _html.escape(raw)
                proj = f"({_html.escape(f['project'])})" if f.get("project") else ""
                score = f.get("score", 0)
                sim = f.get("similarity", 0)
                tags_html = " ".join(
                    f'<span class="tag">{_html.escape(t)}</span>'
                    for t in f.get("tags", [])[:5]
                )
                items.append(
                    f"""<div class="finding">
                            <span class="finding-type {type_class}">[{f['type']}]</span>
                            <a href="/finding/{f['id']}">{f['id']}</a>
                            <span class="meta">score={score:.4f} sim={sim:.3f} {proj}</span>
                            <p>{summary}</p>
                            {f'<div>{tags_html}</div>' if tags_html else ''}
                        </div>"""
                )

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

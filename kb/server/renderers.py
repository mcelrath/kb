"""HTML page-shell renderers for kb serve (web-only).

Extracted from kb.py (lines 457-680) as part of R2 (kb-ez9.3).

This module is web-only: render_html_page uses jinja2 (LAZY-imported inside the
function, so jinja2 is an OPTIONAL serve-only dependency like starlette/uvicorn)
and render_sidebar builds filter HTML with explicit html.escape().  The pure
markdown helpers (format_finding_markdown, markdown_to_html) live in kb/markdown.py
so the core `kb get` CLI command needs neither this module nor starlette/jinja2.
"""

import html as _html
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_jinja_env = None


def _get_jinja_env():
    """Lazily build the autoescaping jinja2 Environment (serve-only dep)."""
    global _jinja_env
    if _jinja_env is None:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        _jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(["html"]),
        )
    return _jinja_env


# ---------------------------------------------------------------------------
# render_sidebar — builds sidebar HTML
# ---------------------------------------------------------------------------

def render_sidebar(stats: dict, all_tags: list, current_filters: dict) -> str:
    """Render the filter sidebar for kb serve."""
    project = current_filters.get('project', '')
    finding_type = current_filters.get('type', '')
    tag = current_filters.get('tag', '')
    include_superseded = current_filters.get('superseded', False)

    def build_url(add_params: dict = None, remove_params: list = None) -> str:
        params = dict(current_filters)
        if remove_params:
            for p in remove_params:
                params.pop(p, None)
        if add_params:
            params.update(add_params)
        params.pop('page', None)  # Reset page when filtering
        if not params:
            return "/"
        return "/?" + "&".join(f"{k}={_html.escape(str(v))}" for k, v in params.items() if v)

    lines = []

    # Projects
    lines.append('<h3>Projects</h3><div class="scroll-list"><ul>')
    lines.append(f'<li><a href="{build_url(remove_params=["project"])}" class="{"active" if not project else ""}">All</a></li>')
    for proj, count in sorted(stats.get('by_project', {}).items()):
        active = 'active' if project == proj else ''
        lines.append(f'<li><a href="{build_url({"project": proj})}" class="{active}">{_html.escape(proj)} <span class="count">({count})</span></a></li>')
    lines.append('</ul></div>')

    # Types
    lines.append('<h3>Types</h3><ul>')
    lines.append(f'<li><a href="{build_url(remove_params=["type"])}" class="{"active" if not finding_type else ""}">All</a></li>')
    for t, count in sorted(stats.get('by_type', {}).items()):
        active = 'active' if finding_type == t else ''
        lines.append(f'<li><a href="{build_url({"type": t})}" class="{active} {t}">{t} <span class="count">({count})</span></a></li>')
    lines.append('</ul>')

    # Tags (scrollable list)
    if all_tags:
        lines.append('<h3>Tags</h3><div class="scroll-list"><ul>')
        lines.append(f'<li><a href="{build_url(remove_params=["tag"])}" class="{"active" if not tag else ""}">All</a></li>')
        for t in all_tags:
            active = 'active' if tag == t else ''
            lines.append(f'<li><a href="{build_url({"tag": t})}" class="{active}">{_html.escape(t)}</a></li>')
        lines.append('</ul></div>')

    # Superseded toggle
    lines.append('<h3>Status</h3>')
    checked = 'checked' if include_superseded else ''
    lines.append(f'<label><input type="checkbox" {checked} onchange="location.href=\'{build_url({"superseded": "1"} if not include_superseded else {}, ["superseded"] if include_superseded else [])}\'"> Show superseded</label>')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# render_html_page — full page shell via Jinja2 template
# ---------------------------------------------------------------------------

def render_html_page(title: str, content: str, sidebar: str = "") -> str:
    """Render an HTML page with consistent styling for kb serve.

    `content` and `sidebar` are already-sanitised HTML fragments (built by the
    route handlers with explicit html.escape() calls on all user data).  They
    are injected with | safe so Jinja2 does not double-escape them.

    `title` is rendered by Jinja2 autoescaping ({{ title }}).
    """
    tmpl = _get_jinja_env().get_template("page.html")
    return tmpl.render(title=title, content=content, sidebar=sidebar)

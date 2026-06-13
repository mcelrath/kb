"""Dependency-free markdown helpers shared by the CLI (`kb get`) and the web UI.

These are pure-Python (stdlib html + re) with NO server/template dependency, so
the core `kb get` command can import them without pulling in starlette/jinja2.
The HTML page shell (render_html_page / render_sidebar) lives in
kb/server/renderers.py instead, since it is web-only.
"""

import html as _html
import re


def format_finding_markdown(finding: dict) -> str:
    """Format a finding as Markdown for detailed display (kb get)."""
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


def markdown_to_html(text: str) -> str:
    """Convert simple markdown to HTML for web display."""
    # Escape HTML first (security)
    text = _html.escape(text)
    # Headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    # Bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Code blocks
    text = re.sub(r'```\n?(.*?)\n?```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # Paragraphs (double newline)
    text = re.sub(r'\n\n+', '</p><p>', text)
    return f'<p>{text}</p>'

#!/usr/bin/env python3
"""
Knowledge Base CLI - Command-line interface for the KB system.

This module provides the CLI for interacting with the Knowledge Base,
including web server functionality. The core library is in the kb/ package.
"""

import argparse
import html
import re
import sys
from pathlib import Path

# Import from kb package
from kb import (
    KnowledgeBase,
    FINDING_TYPES,
    NOTATION_DOMAINS,
    DEFAULT_DB_PATH,
)

# Optional: rich for terminal markdown rendering
try:
    from rich.console import Console
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Optional: starlette/uvicorn for web server
try:
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket
    import asyncio
    import uvicorn
    SERVE_AVAILABLE = True
except ImportError:
    SERVE_AVAILABLE = False


def parse_markdown_findings(file_path: Path) -> list[dict]:
    """Parse a markdown file and extract findings.

    Looks for patterns like:
    - **[SUCCESS]** or **[FAILURE]** markers
    - Bullet points with key findings
    - Sections with results/conclusions
    """
    import re

    content = file_path.read_text()
    findings = []

    # Pattern 1: Explicit markers like **[SUCCESS]**, **[FAILURE]**, etc.
    marker_pattern = re.compile(
        r'\*\*\[(SUCCESS|FAILURE|EXPERIMENT|DISCOVERY)\]\*\*[:\s]*(.+?)(?=\n\n|\n\*\*\[|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    for match in marker_pattern.finditer(content):
        finding_type = match.group(1).lower()
        text = match.group(2).strip()
        findings.append({
            'type': finding_type,
            'content': text[:500],
            'evidence': None,
        })

    # Pattern 2: Key result sections (## Results, ## Findings, ## Conclusions)
    section_pattern = re.compile(
        r'^##\s+(Results?|Findings?|Conclusions?|Key\s+Findings?)\s*\n(.*?)(?=\n##|\Z)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    for match in section_pattern.finditer(content):
        section_content = match.group(2).strip()
        # Extract bullet points
        bullets = re.findall(r'^[-*]\s+(.+)$', section_content, re.MULTILINE)
        for bullet in bullets:
            if len(bullet) > 30:  # Skip short bullets
                findings.append({
                    'type': 'discovery',
                    'content': bullet.strip()[:500],
                    'evidence': None,
                })

    # Pattern 3: Numbered conclusions/results
    numbered_pattern = re.compile(r'^\d+\.\s+\*\*(.+?)\*\*[:\s]*(.+?)(?=\n\d+\.|\n\n|\Z)', re.MULTILINE | re.DOTALL)
    for match in numbered_pattern.finditer(content):
        title = match.group(1).strip()
        desc = match.group(2).strip()
        full = f"{title}: {desc}" if desc else title
        if len(full) > 40:
            findings.append({
                'type': 'discovery',
                'content': full[:500],
                'evidence': None,
            })

    # Deduplicate by content similarity
    seen = set()
    unique = []
    for f in findings:
        key = f['content'][:100].lower()
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def parse_script_findings(file_path: Path) -> list[dict]:
    """Parse a Python script and extract docstrings as findings.

    Extracts:
    - Module-level docstrings
    - Class docstrings with class name
    - Function/method docstrings with function name
    """
    import ast

    content = file_path.read_text()
    findings = []

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return [{"type": "failure", "content": f"Syntax error in {file_path}: {e}", "evidence": None}]

    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc and len(module_doc) > 30:
        findings.append({
            "type": "discovery",
            "content": f"[{file_path.name}] {module_doc[:500]}",
            "evidence": None,
        })

    # Class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 30:
                findings.append({
                    "type": "discovery",
                    "content": f"[class {node.name}] {doc[:500]}",
                    "evidence": None,
                })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc and len(doc) > 30:
                findings.append({
                    "type": "discovery",
                    "content": f"[{node.name}()] {doc[:500]}",
                    "evidence": None,
                })

    return findings


def format_finding(finding: dict, verbose: bool = False) -> str:
    """Format a finding for terminal display (list/search output)."""
    dim = "\033[2m"
    reset = "\033[0m"
    type_colors = {
        "success": "\033[32m",   # green
        "failure": "\033[31m",   # red
        "experiment": "\033[33m",  # yellow
        "discovery": "\033[36m",  # cyan
        "correction": "\033[35m",  # magenta
    }

    color = type_colors.get(finding["type"], "")
    lines = [f"[{color}{finding['type'].upper()}{reset}] {dim}{finding['id']}{reset}"]

    if finding.get("project"):
        lines[0] += f" {dim}({finding['project']}){reset}"

    if finding.get("similarity") is not None:
        sim = finding["similarity"]
        # Color code by similarity: green (>0.8), yellow (0.6-0.8), red (<0.6)
        if sim >= 0.8:
            sim_color = "\033[32m"  # green
        elif sim >= 0.6:
            sim_color = "\033[33m"  # yellow
        else:
            sim_color = "\033[31m"  # red
        lines[0] += f" {sim_color}({sim:.2f}){reset}"

    lines.append(f"  {finding['content']}")

    if verbose:
        if finding.get("evidence"):
            lines.append(f"  {dim}Evidence: {finding['evidence'][:200]}...{reset}" if len(finding.get("evidence", "")) > 200 else f"  {dim}Evidence: {finding['evidence']}{reset}")
        if finding.get("supersedes_id"):
            lines.append(f"  {dim}Supersedes: {finding['supersedes_id']}{reset}")
        if finding.get("tags"):
            lines.append(f"  {dim}Tags: {', '.join(finding['tags'])}{reset}")
        lines.append(f"  {dim}Created: {finding['created_at']}{reset}")
        if finding.get("similarity"):
            lines.append(f"  {dim}Similarity: {finding['similarity']:.3f}{reset}")

    return "\n".join(lines)


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
    text = html.escape(text)
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


def render_html_page(title: str, content: str, sidebar: str = "") -> str:
    """Render an HTML page with consistent styling for kb serve."""
    sidebar_html = f'<aside class="sidebar">{sidebar}</aside>' if sidebar else ''
    main_class = "main-with-sidebar" if sidebar else "main-full"
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)} - Knowledge Base</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #1a1a1a; color: #e0e0e0; }}
        .container {{ display: flex; min-height: 100vh; }}
        .sidebar {{ position: fixed; top: 0; left: 0; width: 220px; background: #151515; padding: 1rem; border-right: 1px solid #333; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; overflow-y: auto; }}
        .sidebar h3 {{ margin: 0.5rem 0; font-size: 0.85rem; color: #888; text-transform: uppercase; flex-shrink: 0; }}
        .sidebar > ul {{ list-style: none; padding: 0; margin: 0 0 1rem 0; flex-shrink: 0; }}
        .sidebar ul {{ list-style: none; padding: 0; margin: 0; }}
        .sidebar li {{ margin: 0.2rem 0; }}
        .sidebar a {{ color: #e0e0e0; text-decoration: none; display: block; padding: 0.3rem 0.5rem; border-radius: 3px; font-size: 0.9rem; }}
        .sidebar a:hover {{ background: #252525; }}
        .sidebar a.active {{ background: #6db3f2; color: #000; }}
        .sidebar .count {{ color: #666; font-size: 0.8rem; }}
        .sidebar .tags-scroll {{ flex: 1; overflow-y: auto; min-height: 0; }}
        .sidebar label {{ display: block; font-size: 0.9rem; padding: 0.3rem 0; cursor: pointer; flex-shrink: 0; }}
        .sidebar input[type="checkbox"] {{ margin-right: 0.5rem; }}
        .main-with-sidebar {{ flex: 1; padding: 1rem; max-width: 900px; margin-left: 240px; }}
        .main-full {{ flex: 1; padding: 1rem; max-width: 900px; margin: 0 auto; }}
        nav {{ margin-bottom: 1rem; }}
        nav a {{ color: #6db3f2; margin-right: 1rem; text-decoration: none; }}
        nav a:hover {{ text-decoration: underline; }}
        h1 {{ font-size: 1.5rem; margin: 0 0 1rem 0; }}
        h2 {{ font-size: 1.2rem; color: #6db3f2; margin: 1.5rem 0 0.5rem 0; }}
        h3 {{ font-size: 1rem; color: #888; }}
        p {{ line-height: 1.6; margin: 0.5rem 0; }}
        pre {{ background: #252525; padding: 1rem; border-radius: 5px; overflow-x: auto; }}
        code {{ background: #252525; padding: 0.2rem 0.4rem; border-radius: 3px; font-family: 'SF Mono', Monaco, monospace; }}
        pre code {{ background: none; padding: 0; }}
        .finding {{ background: #252525; padding: 1rem; margin: 0.5rem 0; border-radius: 5px; border-left: 3px solid #444; }}
        .finding-type {{ font-weight: bold; text-transform: uppercase; margin-right: 0.5rem; }}
        .finding-type.success {{ color: #4caf50; }}
        .finding-type.failure {{ color: #f44336; }}
        .finding-type.experiment {{ color: #ff9800; }}
        .finding-type.discovery {{ color: #2196f3; }}
        .finding-type.correction {{ color: #9c27b0; }}
        .finding a {{ color: #6db3f2; text-decoration: none; }}
        .finding a:hover {{ text-decoration: underline; }}
        .finding p {{ margin: 0.5rem 0 0 0; color: #bbb; }}
        .meta {{ color: #666; font-size: 0.9rem; margin-left: 0.5rem; }}
        .tag {{ background: #333; color: #aaa; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.8rem; margin-right: 0.3rem; }}
        .pagination {{ margin: 1rem 0; display: flex; gap: 0.5rem; }}
        .pagination a {{ padding: 0.3rem 0.8rem; background: #333; color: #e0e0e0; text-decoration: none; border-radius: 3px; }}
        .pagination a:hover {{ background: #444; }}
        .pagination a.active {{ background: #6db3f2; color: #000; }}
        .search-form {{ margin-bottom: 1rem; }}
        .search-form input[type="text"] {{ background: #252525; border: 1px solid #444; color: #e0e0e0; padding: 0.5rem; border-radius: 3px; width: 300px; }}
        .search-form button {{ background: #6db3f2; border: none; color: #000; padding: 0.5rem 1rem; border-radius: 3px; cursor: pointer; }}
        .live-indicator {{ position: fixed; bottom: 1rem; right: 1rem; padding: 0.3rem 0.8rem; border-radius: 3px; font-size: 0.8rem; }}
        .live-indicator.connected {{ background: #1b5e20; color: #a5d6a7; }}
        .live-indicator.disconnected {{ background: #b71c1c; color: #ef9a9a; }}
    </style>
</head>
<body>
    <div class="container">
        {sidebar_html}
        <main class="{main_class}">
            <nav><a href="/">Recent</a> <a href="/search">Search</a></nav>
            <h1>{html.escape(title)}</h1>
            {content}
        </main>
    </div>
    <div id="live-indicator" class="live-indicator disconnected">&#x25cf; Connecting...</div>
    <script>
    (function() {{
        var indicator = document.getElementById('live-indicator');
        var ws = null;
        var reconnectDelay = 1000;

        function connect() {{
            var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(proto + '//' + location.host + '/ws');

            ws.onopen = function() {{
                indicator.className = 'live-indicator connected';
                indicator.innerHTML = '&#x25cf; Live';
                reconnectDelay = 1000;
            }};

            ws.onmessage = function(e) {{
                var msg = JSON.parse(e.data);
                if (msg.type === 'update') {{
                    indicator.innerHTML = '&#x25cf; Updating...';
                    location.reload();
                }}
            }};

            ws.onclose = function() {{
                indicator.className = 'live-indicator disconnected';
                indicator.innerHTML = '&#x25cf; Reconnecting...';
                setTimeout(connect, reconnectDelay);
                reconnectDelay = Math.min(reconnectDelay * 2, 30000);
            }};

            ws.onerror = function() {{
                ws.close();
            }};
        }}

        connect();
    }})();
    </script>
</body>
</html>'''


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
        return "/?" + "&".join(f"{k}={html.escape(str(v))}" for k, v in params.items() if v)

    lines = []

    # Projects
    lines.append('<h3>Projects</h3><ul>')
    lines.append(f'<li><a href="{build_url(remove_params=["project"])}" class="{"active" if not project else ""}">All</a></li>')
    for proj, count in sorted(stats.get('by_project', {}).items()):
        active = 'active' if project == proj else ''
        lines.append(f'<li><a href="{build_url({"project": proj})}" class="{active}">{html.escape(proj)} <span class="count">({count})</span></a></li>')
    lines.append('</ul>')

    # Types
    lines.append('<h3>Types</h3><ul>')
    lines.append(f'<li><a href="{build_url(remove_params=["type"])}" class="{"active" if not finding_type else ""}">All</a></li>')
    for t, count in sorted(stats.get('by_type', {}).items()):
        active = 'active' if finding_type == t else ''
        lines.append(f'<li><a href="{build_url({"type": t})}" class="{active} {t}">{t} <span class="count">({count})</span></a></li>')
    lines.append('</ul>')

    # Tags (scrollable list)
    if all_tags:
        lines.append('<h3>Tags</h3><div class="tags-scroll"><ul>')
        lines.append(f'<li><a href="{build_url(remove_params=["tag"])}" class="{"active" if not tag else ""}">All</a></li>')
        for t in all_tags:
            active = 'active' if tag == t else ''
            lines.append(f'<li><a href="{build_url({"tag": t})}" class="{active}">{html.escape(t)}</a></li>')
        lines.append('</ul></div>')

    # Superseded toggle
    lines.append('<h3>Status</h3>')
    checked = 'checked' if include_superseded else ''
    lines.append(f'<label><input type="checkbox" {checked} onchange="location.href=\'{build_url({"superseded": "1"} if not include_superseded else {}, ["superseded"] if include_superseded else [])}\'"> Show superseded</label>')

    return '\n'.join(lines)


def format_finding_summary(finding: dict) -> str:
    """Format a finding as a single-line summary."""
    type_abbrev = {
        "success": "✓",
        "failure": "✗",
        "experiment": "?",
        "discovery": "→",
        "correction": "↻",
    }
    symbol = type_abbrev.get(finding["type"], "·")
    content = finding["content"][:80] + "..." if len(finding["content"]) > 80 else finding["content"]
    return f"{symbol} {finding['id']}: {content}"


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base - Record and retrieve findings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a finding
  kb add -t success -p my-project "Fixed the memory leak by using weak refs"

  # Search findings
  kb search "memory leak"

  # List recent findings
  kb list -n 10

  # Get a specific finding
  kb get kb-20241201-143022-abc123

  # Correct a finding (supersede it)
  kb correct kb-20241201-143022-abc123 "Actually the issue was thread safety"

  # Web interface
  kb serve --port 8080

  # Check for similar findings before adding
  kb check "memory management approach"

  # Add finding from file
  kb add -f notes.md -p my-project

  # Export/import
  kb export findings.json
  kb import findings.json
"""
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Database path")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new finding")
    add_parser.add_argument("content", nargs="?", help="Finding content (or use -f for file)")
    add_parser.add_argument("-t", "--type", choices=FINDING_TYPES, default="discovery", help="Finding type")
    add_parser.add_argument("-p", "--project", help="Project name")
    add_parser.add_argument("-s", "--sprint", help="Sprint name")
    add_parser.add_argument("--tags", nargs="+", help="Tags")
    add_parser.add_argument("-e", "--evidence", help="Evidence/code snippet")
    add_parser.add_argument("-f", "--file", type=Path, help="Read content from file")
    add_parser.add_argument("--no-duplicate-check", action="store_true", help="Skip duplicate checking")
    add_parser.add_argument("--no-auto-tag", action="store_true", help="Skip auto-tagging")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search findings")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    search_parser.add_argument("-p", "--project", help="Filter by project")
    search_parser.add_argument("-t", "--type", choices=FINDING_TYPES, help="Filter by type")
    search_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")
    search_parser.add_argument("-v", "--verbose", action="store_true", help="Show full details")

    # List command
    list_parser = subparsers.add_parser("list", help="List findings")
    list_parser.add_argument("-n", "--limit", type=int, default=20, help="Max results")
    list_parser.add_argument("-p", "--project", help="Filter by project")
    list_parser.add_argument("-s", "--sprint", help="Filter by sprint")
    list_parser.add_argument("-t", "--type", choices=FINDING_TYPES, help="Filter by type")
    list_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show full details")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get a specific finding")
    get_parser.add_argument("id", help="Finding ID")
    get_parser.add_argument("--raw", action="store_true", help="Output raw markdown")

    # Correct command
    correct_parser = subparsers.add_parser("correct", help="Correct a finding (supersede)")
    correct_parser.add_argument("id", help="ID of finding to correct")
    correct_parser.add_argument("content", help="New correct content")
    correct_parser.add_argument("-e", "--evidence", help="Evidence for correction")
    correct_parser.add_argument("-r", "--reason", help="Reason for correction")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a finding")
    delete_parser.add_argument("id", help="Finding ID to delete")
    delete_parser.add_argument("--force", action="store_true", help="Delete without confirmation")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check for similar findings")
    check_parser.add_argument("content", help="Content to check")
    check_parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export findings to JSON")
    export_parser.add_argument("output", type=Path, help="Output file path")
    export_parser.add_argument("-p", "--project", help="Filter by project")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import findings from JSON")
    import_parser.add_argument("input", type=Path, help="Input file path")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web interface")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch import from file")
    batch_parser.add_argument("file", type=Path, help="File to import (.md or .py)")
    batch_parser.add_argument("-p", "--project", help="Project name")
    batch_parser.add_argument("--dry-run", action="store_true", help="Show what would be imported")

    # Notation commands
    notation_parser = subparsers.add_parser("notation", help="Manage notations")
    notation_sub = notation_parser.add_subparsers(dest="notation_cmd")

    notation_add_parser = notation_sub.add_parser("add", help="Add notation")
    notation_add_parser.add_argument("symbol", help="Symbol (e.g., 'α', 'SO(3)')")
    notation_add_parser.add_argument("meaning", help="Meaning")
    notation_add_parser.add_argument("-d", "--domain", choices=NOTATION_DOMAINS, help="Domain")
    notation_add_parser.add_argument("-p", "--project", help="Project")

    notation_list_parser = notation_sub.add_parser("list", help="List notations")
    notation_list_parser.add_argument("-d", "--domain", choices=NOTATION_DOMAINS, help="Filter by domain")
    notation_list_parser.add_argument("-p", "--project", help="Filter by project")

    notation_search_parser = notation_sub.add_parser("search", help="Search notations")
    notation_search_parser.add_argument("query", help="Search query")
    notation_search_parser.add_argument("-d", "--domain", choices=NOTATION_DOMAINS, help="Filter by domain")
    notation_search_parser.add_argument("-p", "--project", help="Filter by project")

    notation_update_parser = notation_sub.add_parser("update", help="Update notation")
    notation_update_parser.add_argument("old_symbol", help="Old symbol")
    notation_update_parser.add_argument("new_symbol", help="New symbol")
    notation_update_parser.add_argument("-r", "--reason", help="Reason for change")
    notation_update_parser.add_argument("-p", "--project", help="Project")

    notation_history_parser = notation_sub.add_parser("history", help="Show notation history")
    notation_history_parser.add_argument("id", help="Notation ID")

    # Script commands
    script_parser = subparsers.add_parser("script", help="Manage scripts")
    script_sub = script_parser.add_subparsers(dest="script_cmd")

    script_add_parser = script_sub.add_parser("add", help="Register script")
    script_add_parser.add_argument("file", type=Path, help="Script file path")
    script_add_parser.add_argument("purpose", help="Script purpose")
    script_add_parser.add_argument("-p", "--project", help="Project")
    script_add_parser.add_argument("-l", "--language", help="Language (auto-detected if not specified)")

    script_list_parser = script_sub.add_parser("list", help="List scripts")
    script_list_parser.add_argument("-p", "--project", help="Filter by project")
    script_list_parser.add_argument("-l", "--language", help="Filter by language")

    script_search_parser = script_sub.add_parser("search", help="Search scripts")
    script_search_parser.add_argument("query", help="Search query")
    script_search_parser.add_argument("-p", "--project", help="Filter by project")

    script_link_parser = script_sub.add_parser("link", help="Link script to finding")
    script_link_parser.add_argument("script_id", help="Script ID")
    script_link_parser.add_argument("finding_id", help="Finding ID")
    script_link_parser.add_argument("-r", "--relationship", default="generated_by",
                                     choices=["generated_by", "validates", "contradicts"],
                                     help="Relationship type")

    # Error commands
    error_parser = subparsers.add_parser("error", help="Manage errors")
    error_sub = error_parser.add_subparsers(dest="error_cmd")

    error_add_parser = error_sub.add_parser("add", help="Record error")
    error_add_parser.add_argument("signature", help="Error signature")
    error_add_parser.add_argument("-t", "--type", dest="error_type", help="Error type")
    error_add_parser.add_argument("-p", "--project", help="Project")

    error_link_parser = error_sub.add_parser("link", help="Link error to solution")
    error_link_parser.add_argument("error_id", help="Error ID")
    error_link_parser.add_argument("finding_id", help="Finding ID (solution)")
    error_link_parser.add_argument("--verify", action="store_true", help="Mark as verified")

    error_search_parser = error_sub.add_parser("search", help="Search errors")
    error_search_parser.add_argument("query", help="Search query")
    error_search_parser.add_argument("-p", "--project", help="Filter by project")

    error_list_parser = error_sub.add_parser("list", help="List errors")
    error_list_parser.add_argument("-p", "--project", help="Filter by project")
    error_list_parser.add_argument("-t", "--type", dest="error_type", help="Filter by type")

    # Document commands
    doc_parser = subparsers.add_parser("doc", help="Manage documents")
    doc_sub = doc_parser.add_subparsers(dest="doc_cmd")

    doc_add_parser = doc_sub.add_parser("add", help="Add document")
    doc_add_parser.add_argument("title", help="Document title")
    doc_add_parser.add_argument("doc_type", help="Document type (spec, paper, standard, etc.)")
    doc_add_parser.add_argument("-u", "--url", help="Document URL")
    doc_add_parser.add_argument("-s", "--summary", help="Document summary")
    doc_add_parser.add_argument("-p", "--project", help="Project")

    doc_cite_parser = doc_sub.add_parser("cite", help="Cite document in finding")
    doc_cite_parser.add_argument("finding_id", help="Finding ID")
    doc_cite_parser.add_argument("doc_id", help="Document ID")
    doc_cite_parser.add_argument("-t", "--type", default="references",
                                  choices=["references", "implements", "contradicts", "extends"],
                                  help="Citation type")
    doc_cite_parser.add_argument("-n", "--notes", help="Citation notes")

    doc_list_parser = doc_sub.add_parser("list", help="List documents")
    doc_list_parser.add_argument("-p", "--project", help="Filter by project")
    doc_list_parser.add_argument("-t", "--type", dest="doc_type", help="Filter by type")

    doc_search_parser = doc_sub.add_parser("search", help="Search documents")
    doc_search_parser.add_argument("query", help="Search query")
    doc_search_parser.add_argument("-p", "--project", help="Filter by project")

    # Bulk commands
    bulk_tag_parser = subparsers.add_parser("bulk-tag", help="Add tags to multiple findings")
    bulk_tag_parser.add_argument("--ids", nargs="+", required=True, help="Finding IDs")
    bulk_tag_parser.add_argument("--tags", nargs="+", required=True, help="Tags to add")

    bulk_consolidate_parser = subparsers.add_parser("bulk-consolidate", help="Consolidate findings")
    bulk_consolidate_parser.add_argument("--ids", nargs="+", required=True, help="Finding IDs")
    bulk_consolidate_parser.add_argument("--summary", required=True, help="Summary of consolidated finding")
    bulk_consolidate_parser.add_argument("--reason", required=True, help="Reason for consolidation")
    bulk_consolidate_parser.add_argument("-t", "--type", choices=FINDING_TYPES, default="discovery", help="Finding type")
    bulk_consolidate_parser.add_argument("--tags", nargs="+", help="Tags (merged from source if not specified)")

    # Ask command (LLM query)
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the knowledge base")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("-p", "--project", help="Filter by project")
    ask_parser.add_argument("-n", "--limit", type=int, default=10, help="Max findings to consider")

    # Related command
    related_parser = subparsers.add_parser("related", help="Find related findings")
    related_parser.add_argument("id", help="Finding ID")
    related_parser.add_argument("-n", "--limit", type=int, default=5, help="Max results")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate findings for issues")
    validate_parser.add_argument("-p", "--project", help="Filter by project")
    validate_parser.add_argument("-n", "--limit", type=int, default=50, help="Max to check")
    validate_parser.add_argument("--llm", action="store_true", help="Use LLM for deeper analysis")

    # Review queue command
    review_parser = subparsers.add_parser("review", help="Show findings needing review")
    review_parser.add_argument("-p", "--project", help="Filter by project")
    review_parser.add_argument("-n", "--limit", type=int, default=10, help="Max per category")

    # Open questions command
    questions_parser = subparsers.add_parser("questions", help="Identify open questions")
    questions_parser.add_argument("-p", "--project", help="Filter by project")
    questions_parser.add_argument("-n", "--limit", type=int, default=5, help="Max questions")

    # Reembed command
    reembed_parser = subparsers.add_parser("reembed", help="Re-generate all embeddings")
    reembed_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    # Reconcile command
    reconcile_parser = subparsers.add_parser("reconcile", help="Reconcile KB with source document")
    reconcile_parser.add_argument("document", type=Path, help="Source document to reconcile against")
    reconcile_parser.add_argument("-p", "--project", help="Project name")
    reconcile_parser.add_argument("--export-missing", type=Path, help="Export missing claims to file")
    reconcile_parser.add_argument("--import-missing", type=Path, help="Import missing claims from file")

    # Notation audit command
    audit_parser = subparsers.add_parser("notation-audit", help="Audit notations against source document")
    audit_parser.add_argument("document", type=Path, help="Source document")
    audit_parser.add_argument("-p", "--project", help="Project name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize KB
    kb = KnowledgeBase(
        db_path=args.db,
    )

    try:
        if args.command == "add":
            if args.file:
                content = args.file.read_text().strip()
            elif args.content:
                content = args.content
            else:
                print("Error: Either content or --file required")
                sys.exit(1)

            result = kb.add(
                content=content,
                finding_type=args.type,
                project=args.project,
                sprint=args.sprint,
                tags=args.tags,
                evidence=args.evidence,
                check_duplicate=not args.no_duplicate_check,
                auto_tag=not args.no_auto_tag,
            )

            if result.get("duplicate"):
                print(f"Warning: Similar finding exists: {result['duplicate']['id']} (similarity: {result['duplicate']['similarity']:.2f})")
                print(f"  {result['duplicate']['content'][:100]}...")
                print(f"\nAdded anyway with ID: {result['id']}")
            else:
                print(f"Added: {result['id']}")
                if result.get("tags_suggested"):
                    print(f"  Auto-tagged: {', '.join(result.get('tags', []))}")

        elif args.command == "search":
            results = kb.search(
                query=args.query,
                limit=args.limit,
                project=args.project,
                finding_type=args.type,
                include_superseded=args.include_superseded,
            )
            if not results:
                print("No results found")
            else:
                for finding in results:
                    print(format_finding(finding, verbose=args.verbose))
                    print()

        elif args.command == "list":
            results = kb.list_findings(
                limit=args.limit,
                project=args.project,
                sprint=args.sprint,
                finding_type=args.type,
                include_superseded=args.include_superseded,
            )
            if not results:
                print("No findings")
            else:
                for finding in results:
                    print(format_finding(finding, verbose=args.verbose))
                    print()

        elif args.command == "get":
            finding = kb.get(args.id)
            if not finding:
                print(f"Finding not found: {args.id}")
                sys.exit(1)

            md = format_finding_markdown(finding)
            if args.raw or not RICH_AVAILABLE:
                print(md)
            else:
                console = Console()
                console.print(Markdown(md))

        elif args.command == "correct":
            result = kb.correct(
                supersedes_id=args.id,
                content=args.content,
                evidence=args.evidence,
                reason=args.reason,
            )
            print(f"Created correction: {result['id']}")
            print(f"  Supersedes: {args.id}")

        elif args.command == "delete":
            finding = kb.get(args.id)
            if not finding:
                print(f"Finding not found: {args.id}")
                sys.exit(1)

            if not args.force:
                print(f"About to delete: {args.id}")
                print(f"  Type: {finding['type']}")
                print(f"  Content: {finding['content'][:100]}...")
                confirm = input("Confirm delete? [y/N] ")
                if confirm.lower() != "y":
                    print("Cancelled")
                    sys.exit(0)

            kb.delete(args.id)
            print(f"Deleted: {args.id}")

        elif args.command == "check":
            is_dup, existing, _ = kb.check_duplicate(args.content, threshold=args.threshold)
            if is_dup and existing:
                print(f"Similar finding exists: {existing['id']} (similarity: {existing['similarity']:.2f})")
                print(f"  {existing['content']}")
            else:
                print("No similar findings found")

        elif args.command == "stats":
            stats = kb.stats()
            print(f"Database: {stats['db_path']}")
            print(f"Total findings: {stats['total']}")
            print(f"  Current: {stats['current']}")
            print(f"  Superseded: {stats['superseded']}")
            print("\nBy type:")
            for t, count in sorted(stats['by_type'].items()):
                print(f"  {t}: {count}")
            print("\nBy project:")
            for p, count in sorted(stats['by_project'].items()):
                print(f"  {p}: {count}")

        elif args.command == "export":
            result = kb.export_findings(args.output, project=args.project)
            print(f"Exported {result['count']} findings to {args.output}")

        elif args.command == "import":
            result = kb.import_findings(args.input)
            print(f"Imported {result['imported']} findings ({result['skipped']} skipped as duplicates)")

        elif args.command == "serve":
            if not SERVE_AVAILABLE:
                print("Error: starlette and uvicorn required for 'kb serve'")
                print("Install with: pip install starlette uvicorn")
                sys.exit(1)

            # Web server routes
            async def index(request):
                page = int(request.query_params.get('page', 1))
                project = request.query_params.get('project', '')
                finding_type = request.query_params.get('type', '')
                tag = request.query_params.get('tag', '')
                include_superseded = request.query_params.get('superseded', '') == '1'

                per_page = 20
                offset = (page - 1) * per_page

                filters = {}
                if project:
                    filters['project'] = project
                if finding_type:
                    filters['type'] = finding_type
                if include_superseded:
                    filters['superseded'] = '1'
                if tag:
                    filters['tag'] = tag

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
                all_tags = kb.get_all_tags()
                sidebar = render_sidebar(stats, all_tags, filters)

                items = []
                for f in findings:
                    type_class = f['type']
                    summary = f['content'][:200] + "..." if len(f['content']) > 200 else f['content']
                    proj = f"({f['project']})" if f.get('project') else ""
                    tags_html = ' '.join(f'<span class="tag">{html.escape(t)}</span>' for t in f.get('tags', [])[:5])
                    items.append(f'''<div class="finding">
                        <span class="finding-type {type_class}">[{f['type']}]</span>
                        <a href="/finding/{f['id']}">{f['id']}</a>
                        <span class="meta">{proj}</span>
                        <p>{summary}</p>
                        {f'<div>{tags_html}</div>' if tags_html else ''}
                    </div>''')

                # Pagination
                pagination = '<div class="pagination">'
                if page > 1:
                    prev_params = dict(filters)
                    prev_params['page'] = page - 1
                    pagination += f'<a href="/?{"&".join(f"{k}={v}" for k,v in prev_params.items())}">← Prev</a>'
                if has_more:
                    next_params = dict(filters)
                    next_params['page'] = page + 1
                    pagination += f'<a href="/?{"&".join(f"{k}={v}" for k,v in next_params.items())}">Next →</a>'
                pagination += '</div>'

                title = "Findings"
                if project:
                    title += f" - {project}"
                if finding_type:
                    title += f" [{finding_type}]"

                content = '\n'.join(items) + pagination
                return HTMLResponse(render_html_page(title, content, sidebar))

            async def search_page(request):
                query = request.query_params.get('q', '')
                stats = kb.stats()
                all_tags = kb.get_all_tags()
                sidebar = render_sidebar(stats, all_tags, {})

                if query:
                    results = kb.search(query, limit=50)
                    items = []
                    for f in results:
                        type_class = f['type']
                        summary = f['content'][:200] + "..." if len(f['content']) > 200 else f['content']
                        proj = f"({f['project']})" if f.get('project') else ""
                        score = f.get('score', 0)
                        sim = f.get('similarity', 0)
                        tags_html = ' '.join(f'<span class="tag">{html.escape(t)}</span>' for t in f.get('tags', [])[:5])
                        items.append(f'''<div class="finding">
                            <span class="finding-type {type_class}">[{f['type']}]</span>
                            <a href="/finding/{f['id']}">{f['id']}</a>
                            <span class="meta">score={score:.4f} sim={sim:.3f} {proj}</span>
                            <p>{summary}</p>
                            {f'<div>{tags_html}</div>' if tags_html else ''}
                        </div>''')

                    content = f'''<form class="search-form" method="get">
                        <input type="text" name="q" value="{html.escape(query)}" placeholder="Search findings...">
                        <button type="submit">Search</button>
                    </form>
                    <p>Found {len(results)} result(s)</p>
                    {''.join(items)}'''
                else:
                    content = '''<form class="search-form" method="get">
                        <input type="text" name="q" placeholder="Search findings...">
                        <button type="submit">Search</button>
                    </form>'''

                return HTMLResponse(render_html_page("Search", content, sidebar))

            async def finding_page(request):
                finding_id = request.path_params['id']
                finding = kb.get(finding_id)

                if not finding:
                    return HTMLResponse(render_html_page("Not Found", "<p>Finding not found.</p>"), status_code=404)

                stats = kb.stats()
                all_tags = kb.get_all_tags()
                sidebar = render_sidebar(stats, all_tags, {})

                md = format_finding_markdown(finding)
                content = markdown_to_html(md)
                return HTMLResponse(render_html_page(f"Finding {finding_id}", content, sidebar))

            # WebSocket for live updates
            connected_clients: set = set()
            last_state = {"count": 0, "latest": ""}

            async def ws_updates(websocket: WebSocket):
                await websocket.accept()
                connected_clients.add(websocket)
                try:
                    # Send current state on connect
                    count, latest = kb.get_latest_update()
                    await websocket.send_json({"type": "state", "count": count, "latest": latest})
                    # Keep connection alive
                    while True:
                        try:
                            await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        except asyncio.TimeoutError:
                            # Send ping to keep alive
                            await websocket.send_json({"type": "ping"})
                except Exception:
                    pass
                finally:
                    connected_clients.discard(websocket)

            async def check_for_updates():
                """Background task to check for DB changes and notify clients."""
                while True:
                    await asyncio.sleep(2)  # Check every 2 seconds
                    if connected_clients:
                        count, latest = kb.get_latest_update()
                        if count != last_state["count"] or latest != last_state["latest"]:
                            last_state["count"] = count
                            last_state["latest"] = latest
                            # Broadcast to all connected clients
                            dead = set()
                            for ws in connected_clients:
                                try:
                                    await ws.send_json({"type": "update", "count": count, "latest": latest})
                                except Exception:
                                    dead.add(ws)
                            connected_clients.difference_update(dead)

            async def on_startup():
                asyncio.create_task(check_for_updates())

            routes = [
                Route("/", index),
                Route("/search", search_page),
                Route("/finding/{id:path}", finding_page),
                WebSocketRoute("/ws", ws_updates),
            ]
            app = Starlette(routes=routes, on_startup=[on_startup])
            print(f"Starting KB server at http://{args.host}:{args.port}")
            print("WebSocket live updates enabled at /ws")
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

        elif args.command == "batch":
            file_path = args.file
            if not file_path.exists():
                print(f"File not found: {file_path}")
                sys.exit(1)

            if file_path.suffix == ".md":
                findings = parse_markdown_findings(file_path)
            elif file_path.suffix == ".py":
                findings = parse_script_findings(file_path)
            else:
                print(f"Unsupported file type: {file_path.suffix}")
                sys.exit(1)

            print(f"Found {len(findings)} potential findings in {file_path}")

            if args.dry_run:
                for f in findings:
                    print(f"  [{f['type']}] {f['content'][:80]}...")
            else:
                added = 0
                for f in findings:
                    result = kb.add(
                        content=f['content'],
                        finding_type=f['type'],
                        project=args.project,
                        evidence=f.get('evidence'),
                    )
                    if not result.get('duplicate'):
                        added += 1
                        print(f"  Added: {result['id']}")
                    else:
                        print(f"  Skipped (duplicate): {f['content'][:50]}...")
                print(f"\nAdded {added} new findings")

        elif args.command == "notation":
            if args.notation_cmd == "add":
                result = kb.notation_add(
                    symbol=args.symbol,
                    meaning=args.meaning,
                    domain=args.domain,
                    project=args.project,
                )
                print(f"Added notation: {result['id']}")

            elif args.notation_cmd == "list":
                notations = kb.notation_list(domain=args.domain, project=args.project)
                for n in notations:
                    domain = f"[{n['domain']}]" if n.get('domain') else ""
                    project = f"({n['project']})" if n.get('project') else ""
                    print(f"  {n['symbol']} = {n['meaning']} {domain} {project}")

            elif args.notation_cmd == "search":
                notations = kb.notation_search(query=args.query, domain=args.domain, project=args.project)
                for n in notations:
                    print(f"  {n['symbol']} = {n['meaning']}")

            elif args.notation_cmd == "update":
                result = kb.notation_update(
                    old_symbol=args.old_symbol,
                    new_symbol=args.new_symbol,
                    reason=args.reason,
                    project=args.project,
                )
                print(f"Updated: {args.old_symbol} -> {args.new_symbol}")

            elif args.notation_cmd == "history":
                history = kb.notation_history(args.id)
                for entry in history:
                    print(f"  {entry['old_symbol']} -> {entry['new_symbol']}")
                    if entry.get('reason'):
                        print(f"    Reason: {entry['reason']}")
                    print(f"    Changed: {entry['changed_at']}")

        elif args.command == "script":
            if args.script_cmd == "add":
                result = kb.script_add(
                    file_path=str(args.file.absolute()),
                    purpose=args.purpose,
                    project=args.project,
                    language=args.language,
                )
                print(f"Registered script: {result['id']}")

            elif args.script_cmd == "list":
                scripts = kb.script_list(project=args.project, language=args.language)
                for s in scripts:
                    lang = f"[{s['language']}]" if s.get('language') else ""
                    print(f"  {s['id']}: {s['filename']} {lang}")
                    print(f"    Purpose: {s['purpose']}")

            elif args.script_cmd == "search":
                scripts = kb.script_search(query=args.query, project=args.project)
                for s in scripts:
                    print(f"  {s['id']}: {s['filename']}")
                    print(f"    Purpose: {s['purpose']}")

            elif args.script_cmd == "link":
                result = kb.script_link_finding(
                    script_id=args.script_id,
                    finding_id=args.finding_id,
                    relationship=args.relationship,
                )
                print(f"Linked script {args.script_id} to finding {args.finding_id}")

        elif args.command == "error":
            if args.error_cmd == "add":
                result = kb.error_add(
                    signature=args.signature,
                    error_type=args.error_type,
                    project=args.project,
                )
                if result.get('is_new'):
                    print(f"Recorded new error: {result['id']}")
                else:
                    print(f"Error already exists: {result['id']} (count: {result.get('occurrences', 1)})")

            elif args.error_cmd == "link":
                result = kb.error_link(
                    error_id=args.error_id,
                    finding_id=args.finding_id,
                    verified=args.verify,
                )
                print(f"Linked error {args.error_id} to solution {args.finding_id}")

            elif args.error_cmd == "search":
                errors = kb.error_search(query=args.query, project=args.project)
                for e in errors:
                    print(f"  {e['id']}: {e['signature'][:80]}...")
                    if e.get('solutions'):
                        print(f"    Solutions: {len(e['solutions'])}")

            elif args.error_cmd == "list":
                errors = kb.error_list(project=args.project, error_type=args.error_type)
                for e in errors:
                    type_str = f"[{e['error_type']}]" if e.get('error_type') else ""
                    print(f"  {e['id']} {type_str}: {e['signature'][:60]}...")
                    print(f"    Occurrences: {e.get('occurrences', 1)}")

        elif args.command == "doc":
            if args.doc_cmd == "add":
                result = kb.doc_add(
                    title=args.title,
                    doc_type=args.doc_type,
                    url=args.url,
                    summary=args.summary,
                    project=args.project,
                )
                print(f"Added document: {result['id']}")

            elif args.doc_cmd == "cite":
                result = kb.doc_cite(
                    finding_id=args.finding_id,
                    doc_id=args.doc_id,
                    citation_type=args.type,
                    notes=args.notes,
                )
                print(f"Cited document {args.doc_id} in finding {args.finding_id}")

            elif args.doc_cmd == "list":
                docs = kb.doc_list(project=args.project, doc_type=args.doc_type)
                for d in docs:
                    print(f"  {d['id']}: {d['title']} [{d['doc_type']}]")
                    if d.get('url'):
                        print(f"    URL: {d['url']}")

            elif args.doc_cmd == "search":
                docs = kb.doc_search(query=args.query, project=args.project)
                for d in docs:
                    print(f"  {d['id']}: {d['title']} [{d['doc_type']}]")

        elif args.command == "bulk-tag":
            result = kb.bulk_add_tags(finding_ids=args.ids, tags=args.tags)
            print(f"Updated {result['updated']} findings")

        elif args.command == "bulk-consolidate":
            result = kb.consolidate_cluster(
                finding_ids=args.ids,
                summary=args.summary,
                reason=args.reason,
                finding_type=args.type,
                tags=args.tags,
            )
            print(f"Created consolidated finding: {result['id']}")
            print(f"Superseded {result['superseded_count']} findings")

        elif args.command == "ask":
            result = kb.ask(question=args.question, project=args.project, limit=args.limit)
            print(result['answer'])
            if result.get('sources'):
                print("\nSources:")
                for s in result['sources']:
                    print(f"  - {s}")

        elif args.command == "related":
            results = kb.related(finding_id=args.id, limit=args.limit)
            for f in results:
                print(format_finding(f))
                print()

        elif args.command == "validate":
            result = kb.validate(project=args.project, limit=args.limit, use_llm=args.llm)
            if result.get('issues'):
                print(f"Found {len(result['issues'])} issues:")
                for issue in result['issues']:
                    print(f"\n  {issue['id']}:")
                    print(f"    {issue['content'][:80]}...")
                    for w in issue.get('warnings', []):
                        print(f"    ⚠ {w}")
            else:
                print("No issues found")

        elif args.command == "review":
            result = kb.review_queue(project=args.project, limit=args.limit)
            for category, items in result.items():
                if items:
                    print(f"\n{category.upper()} ({len(items)}):")
                    for item in items:
                        print(f"  {item['id']}: {item['content'][:60]}...")

        elif args.command == "questions":
            result = kb.open_questions(project=args.project, limit=args.limit)
            for i, q in enumerate(result.get('questions', []), 1):
                print(f"\n{i}. {q['question']}")
                print(f"   Priority: {q.get('priority', 'unknown')}")
                if q.get('related_topics'):
                    print(f"   Topics: {', '.join(q['related_topics'])}")

        elif args.command == "reembed":
            if not args.force:
                confirm = input("This will re-generate all embeddings. Continue? [y/N] ")
                if confirm.lower() != "y":
                    print("Cancelled")
                    sys.exit(0)

            result = kb.reembed_all()
            print(f"Re-embedded {result['updated']} findings ({result.get('failed', 0)} failed, {result.get('total', 0)} total)")

        elif args.command == "reconcile":
            try:
                from kb_reconcile import DocumentReconciler
            except ImportError:
                print("Error: kb_reconcile module not found")
                sys.exit(1)

            reconciler = DocumentReconciler(kb)

            if args.import_missing:
                result = reconciler.import_missing_claims(args.import_missing)
                print(f"Imported {result['imported']} claims")
            else:
                result = reconciler.reconcile(args.document, project=args.project)
                print(f"\nReconciliation complete:")
                print(f"  Document claims: {result['doc_claims']}")
                print(f"  KB findings: {result['kb_findings']}")
                print(f"  Matched: {result['matched']}")
                print(f"  Missing from KB: {result['missing']}")
                print(f"  Extra in KB: {result['extra']}")

                if args.export_missing and result.get('missing_claims'):
                    reconciler.export_missing_claims(args.export_missing, result['missing_claims'])
                    print(f"\nExported {len(result['missing_claims'])} missing claims to {args.export_missing}")

        elif args.command == "notation-audit":
            try:
                from kb_notation_audit import NotationAuditor
            except ImportError:
                print("Error: kb_notation_audit module not found")
                sys.exit(1)

            auditor = NotationAuditor(kb)
            result = auditor.audit(args.document, project=args.project)
            print(f"\nNotation audit complete:")
            print(f"  Document notations: {result['doc_notations']}")
            print(f"  KB notations: {result['kb_notations']}")
            print(f"  Matched: {result['matched']}")
            print(f"  Missing from KB: {result['missing']}")
            print(f"  Conflicts: {result['conflicts']}")

    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

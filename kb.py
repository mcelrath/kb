#!/usr/bin/env python3
"""
Knowledge Base CLI - Command-line interface for the KB system.

This module provides the CLI for interacting with the Knowledge Base,
including web server functionality. The core library is in the kb/ package.
"""

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path


def _is_agent_context() -> bool:
    """Detect if called from an agent (non-interactive) context.

    Checks in order:
      KB_AGENT=1/0  — explicit override
      CLAUDECODE=1  — set by Claude Code in all subprocesses
      stdout isatty — fallback: pipe/subprocess = agent
    """
    override = os.environ.get("KB_AGENT", "")
    if override == "1":
        return True
    if override == "0":
        return False
    if os.environ.get("CLAUDECODE") == "1":
        return True
    return not sys.stdout.isatty()


AGENT_MODE = _is_agent_context()

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


PENDING_QUEUE_DIR = Path.home() / ".claude" / "pending-kb-adds"
KB_STATE_DIR = "/tmp/claude-kb-state"


def _load_session_seen_ids() -> set[str]:
    """Return kb IDs already in agent context this session.

    Walks the PPID chain to find the Claude Code session file written by
    history-isolation.sh, then reads ${session_id}-kb-seen (maintained by
    dedupe-kb-get.sh and kb-search-track.sh).  The file contains one kb-ID
    per line: written on kb add, kb get, and kb search result output.
    """
    try:
        pid = os.getpid()
        for _ in range(6):
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("PPid:"):
                            pid = int(line.split()[1])
                            break
                    else:
                        break
            except OSError:
                break
            session_file = f"{KB_STATE_DIR}/session-{pid}"
            if os.path.exists(session_file):
                with open(session_file) as f:
                    session_id = f.read().strip()
                seen_file = f"{KB_STATE_DIR}/{session_id}-kb-seen"
                if os.path.exists(seen_file):
                    with open(seen_file) as f:
                        return {ln.strip() for ln in f if ln.strip().startswith("kb-")}
                return set()
    except Exception:
        pass
    return set()


def _queue_async_add(
    content: str,
    finding_type: str | None,
    project: str | None,
    sprint: str | None,
    tags: list[str] | None,
    evidence: str | None,
) -> None:
    """Write a queue file readable by `kb flush-pending` and detach a flusher.

    Returns immediately; the flusher runs in its own session so the parent shell
    (and the agent) does not wait on embedding/LLM I/O.
    """
    import os
    import subprocess
    import secrets
    from datetime import datetime, timezone

    PENDING_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = secrets.token_hex(3)
    qfile = PENDING_QUEUE_DIR / f"{stamp}-{rand}.txt"

    header_lines = []
    if finding_type:
        header_lines.append(f"# type: {finding_type}")
    if project:
        header_lines.append(f"# project: {project}")
    if sprint:
        header_lines.append(f"# sprint: {sprint}")
    if tags:
        header_lines.append(f"# tags: {','.join(tags)}")
    if evidence:
        # Header is single-line; multi-line evidence belongs in content.
        header_lines.append(f"# evidence: {evidence}")
    payload = "\n".join(header_lines) + ("\n\n" if header_lines else "") + content + "\n"

    tmp = qfile.with_suffix(".txt.partial")
    tmp.write_text(payload)
    tmp.rename(qfile)  # atomic publish

    # Detach a flusher. Same interpreter, same script. The flusher probes the
    # embedding server itself and exits silently if it's down.
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "flush-pending", "--quiet"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    print(f"Queued: {qfile.name}")


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


_TYPE_COLORS = {
    "success":    "\033[32m",   # green
    "failure":    "\033[31m",   # red
    "experiment": "\033[33m",   # yellow
    "discovery":  "\033[36m",   # cyan
    "correction": "\033[35m",   # magenta
}
_TYPE_ABBREV = {
    "success":    "SUC",
    "failure":    "FAI",
    "experiment": "EXP",
    "discovery":  "DIS",
    "correction": "COR",
}


_TYPE_ABBREV_AGENT = {
    "correction": "COR", "discovery": "DIS", "success": "SUC",
    "failure": "FAI", "experiment": "EXP",
}
# Tags that add no information in the one-liner (they duplicate the type field)
_SKIP_TAGS = frozenset({"discovery", "success", "failure", "experiment", "correction",
                        "core-result", "technique", "detail", "proven", "heuristic",
                        "open-problem"})


def _fmt_age(created_at: str | None) -> str:
    """Human-readable age: 3d, 2w, 4m, 1y."""
    if not created_at:
        return ""
    try:
        from datetime import datetime
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        days = (datetime.now(created.tzinfo) - created).days
        if days < 1:   return "today"
        if days < 14:  return f"{days}d"
        if days < 60:  return f"{days // 7}w"
        if days < 365: return f"{days // 30}m"
        return f"{days // 365}y"
    except Exception:
        return ""


def _fmt_one_line(finding: dict) -> str:
    """One-line summary for search/list/related. Colored in user mode, plain in agent mode."""
    text = finding.get("summary") or finding["content"].split("\n")[0][:100]
    sim = finding.get("similarity")

    if AGENT_MODE:
        sim_str = f" ({sim:.2f})" if sim is not None else ""
        type_abbr = _TYPE_ABBREV_AGENT.get(finding.get("type", ""), "???")
        age = _fmt_age(finding.get("created_at"))
        tags = [t for t in (finding.get("tags") or []) if t not in _SKIP_TAGS]
        meta_parts = [type_abbr]
        if age:
            meta_parts.append(age)
        if tags:
            meta_parts.append(",".join(tags[:3]))
        meta = "[" + " ".join(meta_parts) + "]"
        return f"{finding['id']}{sim_str} {meta}  {text}"

    dim   = "\033[2m"
    reset = "\033[0m"
    color = _TYPE_COLORS.get(finding.get("type", ""), "")
    abbr  = _TYPE_ABBREV.get(finding.get("type", ""), "???")

    if sim is not None:
        if sim >= 0.7:   sim_color = "\033[32m"
        elif sim >= 0.5: sim_color = "\033[33m"
        else:            sim_color = "\033[31m"
        sim_str = f" {sim_color}({sim:.2f}){reset}"
    else:
        sim_str = ""

    proj = f" {dim}({finding['project']}){reset}" if finding.get("project") else ""
    return f"{color}[{abbr}]{reset} {dim}{finding['id']}{reset}{sim_str}{proj}  {text}"


def format_finding(finding: dict, verbose: bool = False) -> str:
    """Format a finding for terminal display (list/search output)."""
    if AGENT_MODE:
        dim = reset = ""
        type_colors: dict = {}
    else:
        dim = "\033[2m"
        reset = "\033[0m"
        type_colors = {
            "success": "\033[32m",
            "failure": "\033[31m",
            "experiment": "\033[33m",
            "discovery": "\033[36m",
            "correction": "\033[35m",
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
        .sidebar {{ position: fixed; top: 0; left: 0; width: 220px; background: #151515; padding: 1rem; border-right: 1px solid #333; height: 100vh; box-sizing: border-box; overflow-y: auto; }}
        .sidebar h3 {{ margin: 0.5rem 0; font-size: 0.85rem; color: #888; text-transform: uppercase; }}
        .sidebar > ul {{ list-style: none; padding: 0; margin: 0 0 1rem 0; }}
        .sidebar ul {{ list-style: none; padding: 0; margin: 0; }}
        .sidebar li {{ margin: 0.2rem 0; }}
        .sidebar a {{ color: #e0e0e0; text-decoration: none; display: block; padding: 0.3rem 0.5rem; border-radius: 3px; font-size: 0.9rem; }}
        .sidebar a:hover {{ background: #252525; }}
        .sidebar a.active {{ background: #6db3f2; color: #000; }}
        .sidebar .count {{ color: #666; font-size: 0.8rem; }}
        .sidebar .scroll-list {{ max-height: 300px; overflow-y: auto; }}
        .sidebar label {{ display: block; font-size: 0.9rem; padding: 0.3rem 0; cursor: pointer; }}
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
    lines.append('<h3>Projects</h3><div class="scroll-list"><ul>')
    lines.append(f'<li><a href="{build_url(remove_params=["project"])}" class="{"active" if not project else ""}">All</a></li>')
    for proj, count in sorted(stats.get('by_project', {}).items()):
        active = 'active' if project == proj else ''
        lines.append(f'<li><a href="{build_url({"project": proj})}" class="{active}">{html.escape(proj)} <span class="count">({count})</span></a></li>')
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


def _run_refresh(kb, rows, dry_run: bool, commit_every: int, label: str = "refresh"):
    """Core loop: summarize + retag + reembed each finding row.

    Embedding (ash:8081) and LLM (tardis:9510) run in parallel per row — they
    are independent servers, so there is no reason to queue them sequentially.

    rows: list of (id, project, content, evidence)
    Returns (ok, fail) counts.

    Stop/start safe: rows with NULL summary are re-fetched on restart (default mode).
    Committed every commit_every rows, so interrupting loses at most commit_every rows.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    ok = fail = 0
    total = len(rows)
    t0 = _time.time()
    interval = max(5, total // 100) if total else 1

    # Single pool shared across all rows; max_workers=2 so embed and LLM run
    # side-by-side without spawning unbounded threads.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (fid, fproject, content, evidence) in enumerate(rows, 1):
            embed_text = content + (" " + evidence if evidence else "")

            if dry_run:
                # Skip embedding in dry-run; only preview LLM output.
                summary = kb._analyzer.generate_summary(content, evidence)
                tags: list = kb.suggest_tags(content, fproject)
                embedding = None
            else:
                # Pre-fetch existing tags from DB here in the main thread —
                # sqlite3 connections cannot be used across threads.
                existing_tags = kb._fetch_existing_tags(fproject)

                # Fire both concurrently; collect when both finish.
                embed_fut = pool.submit(kb._embedding.embed, embed_text)

                def _llm(c=content, e=evidence, et=existing_tags):
                    s = kb._analyzer.generate_summary(c, e)
                    t = kb._analyzer.suggest_tags(c, et)
                    return s, t

                llm_fut = pool.submit(_llm)

                try:
                    embedding = embed_fut.result()
                except Exception as e:
                    embedding = None
                    print(f"  EMBED FAIL {fid}: {e}")

                try:
                    summary, tags = llm_fut.result()
                except Exception as e:
                    summary, tags = None, []
                    print(f"  LLM FAIL {fid}: {e}")

            if summary and len(summary) >= 10:
                if dry_run:
                    print(f"[DRY] {fid} ({fproject}): {summary}")
                else:
                    kb.conn.execute(
                        "UPDATE findings SET summary=?, updated_at=datetime('now') WHERE id=?",
                        (summary, fid),
                    )
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {fid} ({fproject}): {(content or '')[:60]!r}")

            if not dry_run:
                if tags:
                    kb.conn.execute(
                        "UPDATE findings SET tags=?, updated_at=datetime('now') WHERE id=?",
                        (json.dumps(tags), fid),
                    )
                if embedding is not None:
                    kb.conn.execute("DELETE FROM findings_vec WHERE id=?", (fid,))
                    kb.conn.execute(
                        "INSERT INTO findings_vec (id, embedding) VALUES (?,?)",
                        (fid, embedding),
                    )
                if i % commit_every == 0:
                    kb.conn.commit()

            if i % interval == 0 or i == total:
                elapsed = _time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0.0
                eta_sec = (total - i) / rate if rate > 0 else 0.0
                print(
                    f"  {i}/{total} ({100*i//total}%)  ok={ok} fail={fail}"
                    f"  rate={rate:.2f}/s  elapsed={elapsed/60:.1f}m  eta={eta_sec/60:.1f}m",
                    flush=True,
                )

    if not dry_run:
        kb.conn.commit()
    elapsed = _time.time() - t0
    print(f"{label}: ok={ok} fail={fail} total={total} elapsed={elapsed/60:.1f}m")
    return ok, fail


def _fetch_refresh_rows(kb, ids=None, project=None, all_rows=False, limit=0):
    """Build the findings row list for refresh/retag/resummarize."""
    if ids:
        placeholders = ",".join("?" * len(ids))
        return kb.conn.execute(
            f"SELECT id, project, content, evidence FROM findings WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    sql = "SELECT id, project, content, evidence FROM findings WHERE status = 'current'"
    params: list = []
    if not all_rows:
        sql += " AND (summary IS NULL OR summary = '')"
    if project:
        sql += " AND project = ?"
        params.append(project)
    sql += " ORDER BY created_at DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return kb.conn.execute(sql, params).fetchall()


def _backfill_statement_pure(kb, project=None, limit=None, workers=8, dry_run=False):
    """Backfill statement_pure for lean theorems using the KB's LLM client."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    conn = kb._theorems.conn
    where = "WHERE statement_pure IS NULL OR statement_pure = ''"
    params: list = []
    if project:
        where += " AND project = ?"
        params.append(project)
    if limit:
        where += f" LIMIT {limit}"

    rows = conn.execute(
        f"SELECT id, lean_name, statement FROM lean_theorems {where}", params
    ).fetchall()
    print(f"  theorem backfill: {len(rows)} without statement_pure")
    if not rows:
        return {"updated": 0, "failed": 0}

    if dry_run:
        for tid, lean_name, stmt in rows[:3]:
            print(f"  [DRY] {lean_name}: {stmt[:80]}")
        return {"updated": 0, "failed": 0}

    PROMPT = (
        "Restate this Lean 4 theorem in pure mathematical language. "
        "No Lean syntax, no type annotations. Standard math notation. "
        "One sentence, under 30 words.\n\nLean:\n{statement}\n\nMath:"
    )

    def restate_one(row):
        tid, lean_name, statement = row
        result = kb._analyzer.llm_client.complete(
            PROMPT.format(statement=statement[:600]),
            max_tokens=80, temperature=0.1, timeout=30,
        )
        if result:
            result = result.strip().strip('"').strip("'")
        return tid, lean_name, result or None

    updated = failed = 0
    t0 = _time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(restate_one, row): row for row in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            tid, lean_name, pure = fut.result()
            if pure:
                conn.execute("UPDATE lean_theorems SET statement_pure=? WHERE id=?", (pure, tid))
                updated += 1
            else:
                failed += 1
            if i % 100 == 0:
                conn.commit()
                elapsed = _time.time() - t0
                rate = i / max(elapsed, 0.001)
                print(f"  {i}/{len(rows)} done  {rate:.1f}/s", flush=True)

    conn.commit()

    # Re-embed updated theorems
    if updated > 0:
        print(f"  re-embedding {updated} theorems...")
        updated_rows = conn.execute(
            "SELECT id, statement_pure FROM lean_theorems "
            "WHERE statement_pure IS NOT NULL AND statement_pure != ''"
        ).fetchall()
        for j, (tid, pure) in enumerate(updated_rows):
            emb = kb._theorems.embedding_service.embed(pure)
            conn.execute("DELETE FROM lean_theorems_vec WHERE id=?", (tid,))
            conn.execute("INSERT INTO lean_theorems_vec (id, embedding) VALUES (?,?)", (tid, emb))
            if j % 100 == 0:
                conn.commit()
        conn.commit()

    elapsed = _time.time() - t0
    print(f"  theorem backfill done: updated={updated} failed={failed} elapsed={elapsed:.0f}s")
    return {"updated": updated, "failed": failed}


_AGENT_CMDS = [
    ("add",           '-t TYPE -p PROJECT "content"   record a finding (sync, prints kb-id)'),
    ("search",        '"query" [-n N] [-p PROJECT] [-t TYPE] [-l]   semantic search'),
    ("list",          '[-n N] [-p PROJECT] [-s SPRINT] [-t TYPE] [-l]   list findings'),
    ("get",           "<kb-id>   full entry"),
    ("correct",       '<kb-id> "new content" [-r reason]   supersede a finding'),
    ("related",   "<kb-id> [-n N]   find semantically similar findings"),
]

_MAINT_CMDS = [
    ("refresh",       "retag + resummarize + reembed  [-p PROJECT] [--all] [--theorems]"),
    ("review",        "findings needing attention  [-p PROJECT]"),
    ("questions",     "LLM: identify research gaps  [-p PROJECT] [-n N] [-i N] [query]"),
    ("ask",           'LLM: answer a question from KB  "question" [-p PROJECT]'),
    ("stats",         "counts by type and project"),
    ("flush-pending", "drain the offline-add queue"),
    ("ingest",        "lean [--source proofs|mathlib] | scripts <dir>"),
    ("delete",        "<kb-id> [--force]"),
    ("export",        "<file.json> [-p PROJECT]"),
    ("import",        "<file.json>"),
    ("serve",         "[--port 8000]"),
]

_LEGACY_CMDS = None  # no legacy commands remain


def _print_main_help():
    W = 14  # column width for command names
    if AGENT_MODE:
        print("kb add|search|list|get|correct|related\n")
        for cmd, desc in _AGENT_CMDS:
            print(f"  {cmd:<{W}}{desc}")
        print("\nRun any command with --help for full flag list.")
        print("Set KB_AGENT=0 for user mode.")
    else:
        bold   = "\033[1m"
        cyan   = "\033[36m"
        yellow = "\033[33m"
        dim    = "\033[2m"
        reset  = "\033[0m"

        print(f"{bold}Knowledge Base{reset}  {dim}(set CLAUDECODE=1 or KB_AGENT=1 for agent mode){reset}\n")

        print(f"{bold}Agent commands:{reset}")
        for cmd, desc in _AGENT_CMDS:
            parts = desc.split("   ")
            summary = parts[-1] if len(parts) > 1 else desc
            print(f"  {cyan}{cmd:<{W}}{reset}{summary}")
        print()

        print(f"{bold}Maintenance:{reset}")
        for cmd, desc in _MAINT_CMDS:
            # desc format: "plain description  [-flags]" or just "[-flags]"
            # Split on first double-space to separate description from flags
            if "  " in desc:
                plain, flags = desc.split("  ", 1)
                flags = "  " + flags
            else:
                plain, flags = desc, ""
            print(f"  {yellow}{cmd:<{W}}{reset}{plain}{dim}{flags}{reset}")
        print()

        if _LEGACY_CMDS:
            print(f"{bold}Legacy{reset} {dim}(still work; run with --help for flags):{reset}")
            print(f"  {dim}{_LEGACY_CMDS}{reset}")
            print()

        print(f"{bold}Options:{reset}")
        print(f"  {dim}{'--db PATH':<{W}}database path (default: ~/.cache/kb/knowledge.db){reset}")
        print(f"  {dim}{'-h, --help':<{W}}show this help{reset}")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", default=False)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Database path")
    subparsers = parser.add_subparsers(dest="command")

    def _add_parser(name: str, help_text: str,
                    agent_visible: bool = False,
                    user_visible: bool = True,
                    **kwargs):
        """Register a subcommand, hiding it from --help based on context.

        agent_visible=True  : shown in agent mode help (implies user_visible too)
        user_visible=False  : hidden in user mode (legacy/niche; still callable by name)
        """
        hide = (not agent_visible and AGENT_MODE) or (not user_visible and not AGENT_MODE)
        p = subparsers.add_parser(name, help=argparse.SUPPRESS if hide else help_text, **kwargs)
        if hide:
            subparsers._name_parser_map.pop(name, None)
            subparsers._choices_actions = [
                a for a in subparsers._choices_actions if a.dest != name
            ]
            subparsers._name_parser_map[name] = p  # re-add so it's still callable
        return p

    # Add command
    add_parser = _add_parser("add", "Add a new finding", agent_visible=True)
    add_parser.add_argument("content", nargs="?", help="Finding content (or use -f for file)")
    add_parser.add_argument("-t", "--type", choices=FINDING_TYPES, default="discovery", help="Finding type")
    add_parser.add_argument("-p", "--project", help="Project name")
    add_parser.add_argument("-s", "--sprint", help="Sprint name")
    add_parser.add_argument("--tags", nargs="+", help="Tags")
    add_parser.add_argument("-e", "--evidence", help="Evidence/code snippet")
    add_parser.add_argument("-f", "--file", type=Path, help="Read content from file")
    add_parser.add_argument("--no-duplicate-check", action="store_true", help="Skip duplicate checking")
    add_parser.add_argument("--no-auto-tag", action="store_true", help="Skip auto-tagging")
    add_parser.add_argument("--async", dest="async_add", action="store_true",
        help="Fire-and-forget: write to queue file and return immediately without waiting "
             "for embedding. Use when embedding server may be slow or unavailable.")

    # Search command
    search_parser = _add_parser("search", "Search findings", agent_visible=True)
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    search_parser.add_argument("-p", "--project", help="Filter by project")
    search_parser.add_argument("-t", "--type", choices=FINDING_TYPES, help="Filter by type")
    search_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")
    search_parser.add_argument("-v", "--verbose", action="store_true", help="Show full details")
    search_parser.add_argument("-l", "--long", action="store_true", help="Show full content (default: one line per result)")
    search_parser.add_argument("--exclude", nargs="*", default=[], metavar="ID",
        help="KB IDs to exclude from results (auto-loaded from session seen file)")
    search_parser.add_argument("--no-dedup", action="store_true",
        help="Disable automatic exclusion of session-seen IDs")
    search_parser.add_argument("--json", action="store_true",
        help="Output results as JSON array (full metadata, for hook/script use)")

    # List command
    list_parser = _add_parser("list", "List findings", agent_visible=True)
    list_parser.add_argument("-n", "--limit", type=int, default=20, help="Max results")
    list_parser.add_argument("-p", "--project", help="Filter by project")
    list_parser.add_argument("-s", "--sprint", help="Filter by sprint")
    list_parser.add_argument("-t", "--type", choices=FINDING_TYPES, help="Filter by type")
    list_parser.add_argument("--include-superseded", action="store_true", help="Include superseded")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Show full details")
    list_parser.add_argument("-l", "--long", action="store_true", help="Show full content (default: one line per result)")

    # Get command
    get_parser = _add_parser("get", "Get a specific finding", agent_visible=True)
    get_parser.add_argument("id", help="Finding ID")
    get_parser.add_argument("--raw", action="store_true", help="Output raw markdown")

    # Correct command
    correct_parser = _add_parser("correct", "Correct a finding (supersede)", agent_visible=True)
    correct_parser.add_argument("id", help="ID of finding to correct")
    correct_parser.add_argument("content", help="New correct content")
    correct_parser.add_argument("-e", "--evidence", help="Evidence for correction")
    correct_parser.add_argument("-r", "--reason", help="Reason for correction")

    # Delete command
    delete_parser = _add_parser("delete", "Delete a finding")
    delete_parser.add_argument("id", help="Finding ID to delete")
    delete_parser.add_argument("--force", action="store_true", help="Delete without confirmation")

    # Stats command
    _add_parser("stats", "Show database statistics")

    # Export command
    export_parser = _add_parser("export", "Export findings to JSON")
    export_parser.add_argument("output", type=Path, help="Output file path")
    export_parser.add_argument("-p", "--project", help="Filter by project")

    # Import command
    import_parser = _add_parser("import", "Import findings from JSON")
    import_parser.add_argument("input", type=Path, help="Input file path")

    # Serve command
    serve_parser = _add_parser("serve", "Start web interface")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")


    # Ask command: LLM Q&A over KB findings
    ask_parser = _add_parser("ask", "Ask a natural language question about the KB")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("-p", "--project", help="Filter by project")
    ask_parser.add_argument("-n", "--limit", type=int, default=10, help="Max findings to consider")

    # Related command
    related_parser = _add_parser("related", "Find semantically related findings", agent_visible=True)
    related_parser.add_argument("id", help="Finding ID")
    related_parser.add_argument("-n", "--limit", type=int, default=5, help="Max results")
    related_parser.add_argument("-l", "--long", action="store_true", help="Show full content")

    # Open questions: LLM identifies research gaps from existing findings
    questions_parser = _add_parser("questions", "Identify research gaps using LLM")
    questions_parser.add_argument("query", nargs="?", help="Search query to seed findings (default: most recent)")
    questions_parser.add_argument("-p", "--project", help="Filter by project")
    questions_parser.add_argument("-n", "--limit", type=int, default=5, help="Number of questions to generate (default: 5)")
    questions_parser.add_argument("-i", "--input", type=int, default=20, help="Number of KB entries to feed the LLM (default: 20)")

    # Review queue: surfaces untagged, stale, orphaned findings
    review_parser = _add_parser("review", "Show findings needing attention")
    review_parser.add_argument("-p", "--project", help="Filter by project")
    review_parser.add_argument("-n", "--limit", type=int, default=10, help="Max per category")


    # Refresh command: retag + resummarize + optional reembed + optional theorem backfill
    refresh_parser = _add_parser("refresh", "Retag + resummarize findings")
    refresh_parser.add_argument("targets", nargs="*", metavar="ID_OR_PROJECT",
        help="kb-ids to refresh, OR a project name (auto-detected by prefix)")
    refresh_parser.add_argument("-p", "--project", help="Restrict to one project (alias for passing project as positional arg)")
    refresh_parser.add_argument("--all", action="store_true",
        help="Regenerate ALL summaries/tags (default: only rows with NULL summary)")
    refresh_parser.add_argument("-n", "--limit", type=int, default=0,
        help="Max rows to process (0 = no limit)")
    refresh_parser.add_argument("--dry-run", action="store_true")
    refresh_parser.add_argument("--commit-every", type=int, default=20)
    refresh_parser.add_argument("--theorems", action="store_true",
        help="Also backfill statement_pure for lean theorems")
    refresh_parser.add_argument("--theorem-workers", type=int, default=8,
        help="Parallel workers for theorem backfill (default 8)")

    # Ingest command group
    ingest_parser = _add_parser("ingest", "Ingest external content into KB")
    ingest_sub = ingest_parser.add_subparsers(dest="ingest_cmd")

    ingest_lean_parser = ingest_sub.add_parser("lean", help="Ingest Lean theorems + backfill statement_pure")
    ingest_lean_parser.add_argument("--source", choices=["proofs", "mathlib"], default="proofs",
        help="Repo to ingest (default: proofs)")
    ingest_lean_parser.add_argument("--direct", action="store_true",
        help="Use direct regex parser (no LeanDojo required)")
    ingest_lean_parser.add_argument("--project", default=None)
    ingest_lean_parser.add_argument("--dry-run", action="store_true")
    ingest_lean_parser.add_argument("--limit", type=int, default=None)
    ingest_lean_parser.add_argument("--module-filter", default=None)
    ingest_lean_parser.add_argument("--workers", type=int, default=8,
        help="Parallel workers for statement_pure backfill (default 8)")
    ingest_lean_parser.add_argument("--no-backfill", action="store_true",
        help="Skip statement_pure generation after ingestion")

    ingest_scripts_parser = ingest_sub.add_parser("scripts", help="Register scripts with LLM-generated purposes")
    ingest_scripts_parser.add_argument("directory", type=Path, help="Directory to scan")
    ingest_scripts_parser.add_argument("-p", "--project", default="hypercomplex")
    ingest_scripts_parser.add_argument("--dry-run", action="store_true")
    ingest_scripts_parser.add_argument("-n", "--limit", type=int, default=50)

    # Reconcile command
    reconcile_parser = _add_parser("reconcile", "Reconcile KB with source document", user_visible=False)
    reconcile_parser.add_argument("document", type=Path, help="Source document to reconcile against")
    reconcile_parser.add_argument("-p", "--project", help="Project name")
    reconcile_parser.add_argument("--export-missing", type=Path, help="Export missing claims to file")
    reconcile_parser.add_argument("--import-missing", type=Path, help="Import missing claims from file")

    # Notation audit command
    audit_parser = _add_parser("notation-audit", "Audit notations against source document", user_visible=False)
    audit_parser.add_argument("document", type=Path, help="Source document")
    audit_parser.add_argument("-p", "--project", help="Project name")

    # Flush-pending (user/debugging only — automatic via hooks and --async spawner)
    flush_parser = _add_parser(
        "flush-pending",
        "Drain ~/.claude/pending-kb-adds/*.txt (kb-down fallback queue)",
    )
    flush_parser.add_argument(
        "--queue-dir",
        type=Path,
        default=Path.home() / ".claude" / "pending-kb-adds",
        help="Queue directory (default ~/.claude/pending-kb-adds)",
    )
    flush_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; only print summary line",
    )

    args = parser.parse_args()

    if args.help or not args.command:
        _print_main_help()
        sys.exit(0 if args.help else 1)
        sys.exit(1)

    # Async `kb add`: write a queue file and detach a flusher. No DB / no network.
    # Resolve content early (needed for both sync and async paths)
    if args.command == "add":
        if args.file:
            _add_content = args.file.read_text().strip()
        elif args.content:
            _add_content = args.content
        else:
            print("Error: Either content or --file required")
            sys.exit(1)
        if not _add_content:
            print("Error: empty content")
            sys.exit(1)

        if args.async_add:
            _queue_async_add(
                content=_add_content,
                finding_type=args.type,
                project=args.project,
                sprint=args.sprint,
                tags=args.tags,
                evidence=args.evidence,
            )
            sys.exit(0)

    # Initialize KB
    kb = KnowledgeBase(
        db_path=args.db,
    )

    try:
        if args.command == "add":
            import os as _os
            # Use a short retry budget for interactive add so it returns quickly.
            # If the server is busy/loaded, fall back to the queue silently —
            # flush-pending will drain it at the next SessionStart/UserPromptSubmit.
            _saved_retries = _os.environ.get("KB_EMBED_MAX_RETRIES")
            _os.environ["KB_EMBED_MAX_RETRIES"] = "1"
            try:
                result = kb.add(
                    content=_add_content,
                    finding_type=args.type,
                    project=args.project,
                    sprint=args.sprint,
                    tags=args.tags,
                    evidence=args.evidence,
                    check_duplicate=not args.no_duplicate_check,
                    auto_tag=not args.no_auto_tag,
                )
            except Exception as e:
                err = str(e)
                if any(kw in err for kw in ("Remote embedding", "Connection refused",
                                             "RemoteDisconnected", "URLError", "HTTPError",
                                             "503", "502", "504", "TimeoutError")):
                    _queue_async_add(
                        content=_add_content,
                        finding_type=args.type,
                        project=args.project,
                        sprint=args.sprint,
                        tags=args.tags,
                        evidence=args.evidence,
                    )
                    sys.exit(0)
                raise
            finally:
                if _saved_retries is None:
                    _os.environ.pop("KB_EMBED_MAX_RETRIES", None)
                else:
                    _os.environ["KB_EMBED_MAX_RETRIES"] = _saved_retries

            if result.get("duplicate"):
                print(f"Warning: Similar finding exists: {result['duplicate']['id']} (similarity: {result['duplicate']['similarity']:.2f})")
                print(f"  {result['duplicate']['content'][:100]}...")
                print(f"\nAdded anyway with ID: {result['id']}")
            else:
                print(f"Added: {result['id']}")
                if result.get("tags_suggested"):
                    print(f"  Auto-tagged: {', '.join(result.get('tags', []))}")

        elif args.command == "search":
            exclude_ids: set[str] = set(args.exclude or [])
            if not args.no_dedup:
                exclude_ids |= _load_session_seen_ids()
            results = kb.search(
                query=args.query,
                limit=args.limit + len(exclude_ids),  # fetch extra to compensate for filtered rows
                project=args.project,
                finding_type=args.type,
                include_superseded=args.include_superseded,
                exclude_ids=exclude_ids or None,
            )
            results = results[:args.limit]
            if args.json:
                print(json.dumps(results, indent=2, default=str))
            elif not results:
                print("No results found")
            elif args.long:
                for finding in results:
                    print(format_finding(finding, verbose=args.verbose))
                    print()
            else:
                for finding in results:
                    print(_fmt_one_line(finding))

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
            elif args.long:
                for finding in results:
                    print(format_finding(finding, verbose=args.verbose))
                    print()
            else:
                for finding in results:
                    print(_fmt_one_line(finding))

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

        elif args.command == "stats":
            stats = kb.stats()
            print(f"Database: {stats['db_path']}")
            print(f"Total findings: {stats['total']}")
            print(f"  Current:    {stats['current']}")
            print(f"  Superseded: {stats['superseded']}")
            no_sum = stats.get('no_summary', 0)
            no_emb = stats.get('no_embedding', 0)
            print(f"  No summary: {no_sum}"
                  + (f"  (run: kb refresh -p PROJECT)" if no_sum else ""))
            if stats.get('no_summary_by_project'):
                for proj, cnt in stats['no_summary_by_project'].items():
                    print(f"    {proj}: {cnt}")
            print(f"  No embed:   {no_emb}"
                  + (f"  (run: kb refresh --all -p PROJECT)" if no_emb else ""))
            if stats.get('no_embedding_by_project'):
                for proj, cnt in stats['no_embedding_by_project'].items():
                    print(f"    {proj}: {cnt}")
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
                all_tags = kb.get_all_tags(limit=100)
                sidebar = render_sidebar(stats, all_tags, filters)

                items = []
                for f in findings:
                    type_class = f['type']
                    raw = f['content'][:200] + "..." if len(f['content']) > 200 else f['content']
                    summary = html.escape(raw)
                    proj = f"({html.escape(f['project'])})" if f.get('project') else ""
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
                all_tags = kb.get_all_tags(limit=100)
                sidebar = render_sidebar(stats, all_tags, {})

                if query:
                    results = kb.search(query, limit=50)
                    items = []
                    for f in results:
                        type_class = f['type']
                        raw = f['content'][:200] + "..." if len(f['content']) > 200 else f['content']
                        summary = html.escape(raw)
                        proj = f"({html.escape(f['project'])})" if f.get('project') else ""
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
                all_tags = kb.get_all_tags(limit=100)
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


        elif args.command == "ask":
            result = kb.ask(question=args.question, project=args.project, limit=args.limit)
            print(result['answer'])
            if result.get('sources'):
                print("\nSources:")
                for s in result['sources']:
                    print(f"  - {s}")

        elif args.command == "questions":
            questions = kb.generate_open_questions(
                project=args.project,
                limit=args.limit,
                input_limit=args.input,
                query=args.query,
            )
            if not questions:
                print("No questions generated (try adding more findings or a search query).")
            else:
                seed = f'"{args.query}"' if args.query else "recent findings"
                print(f"Open questions from {seed}:\n")
                for i, q in enumerate(questions, 1):
                    print(f"{i}. {q.get('question', '')}")
                    if q.get('why'):
                        print(f"   Why: {q['why']}")
                    if q.get('related_ids'):
                        print(f"   See: {', '.join(q['related_ids'][:3])}")
                    print()

        elif args.command == "related":
            results = kb.related(finding_id=args.id, limit=args.limit)
            if not results:
                print("No related findings.")
            elif args.long:
                for f in results:
                    print(format_finding(f))
                    print()
            else:
                for f in results:
                    print(_fmt_one_line(f))

        elif args.command == "review":
            result = kb.review_queue(project=args.project, limit=args.limit)
            any_issues = False
            for category, items in result.items():
                if items:
                    any_issues = True
                    print(f"\n{category.upper()} ({len(items)}):")
                    for item in items:
                        proj = f" ({item['project']})" if item.get('project') else ""
                        print(f"  {item['id']}{proj}: {item.get('content', '')[:60]}...")
            if not any_issues:
                print("No findings need attention.")

        elif args.command == "refresh":
            # Partition positional targets into kb-ids vs project names.
            targets = args.targets or []
            explicit_ids = [t for t in targets if t.startswith("kb-")]
            project_args = [t for t in targets if not t.startswith("kb-")]
            if len(project_args) > 1:
                print(f"Error: only one project name allowed, got: {project_args}")
                sys.exit(1)
            project = args.project or (project_args[0] if project_args else None)
            rows = _fetch_refresh_rows(
                kb,
                ids=explicit_ids or None,
                project=project,
                all_rows=args.all,
                limit=args.limit,
            )
            print(f"refresh: {len(rows)} findings "
                  f"(project={project or 'ALL'}, all={args.all}, dry={args.dry_run})"
                  f"\n  (Ctrl+C safe: work is committed every {args.commit_every} rows;"
                  f" restart without --all to resume from unprocessed rows)")
            _run_refresh(kb, rows, dry_run=args.dry_run, commit_every=args.commit_every)
            if args.theorems:
                _backfill_statement_pure(
                    kb, project=project,
                    workers=args.theorem_workers, dry_run=args.dry_run,
                )

        elif args.command == "ingest":
            import subprocess as _sp
            scripts_dir = Path(__file__).parent / "scripts"

            if args.ingest_cmd == "lean":
                script = "ingest_lean_direct.py" if args.direct else "ingest_lean.py"
                script_path = scripts_dir / script
                if not script_path.exists():
                    print(f"Error: {script_path} not found")
                    sys.exit(1)
                cmd = [sys.executable, str(script_path)]
                if args.direct:
                    # ingest_lean_direct.py only supports mathlib-style repos
                    root = (Path.home() / "Physics/mathlib4") if args.source == "mathlib" \
                           else (Path.home() / "Physics/claude/proofs")
                    cmd += ["--mathlib-root", str(root)]
                else:
                    cmd += ["--source", args.source]
                if args.project:
                    cmd += ["--project", args.project]
                if args.dry_run:
                    cmd += ["--dry-run"]
                if args.limit:
                    cmd += ["--limit", str(args.limit)]
                if args.module_filter:
                    cmd += ["--module-filter", args.module_filter]
                result = _sp.run(cmd)
                if result.returncode != 0:
                    sys.exit(result.returncode)
                if not args.no_backfill and not args.dry_run:
                    print("\nBackfilling statement_pure...")
                    _backfill_statement_pure(kb, project=args.project,
                                              workers=args.workers, dry_run=False)

            elif args.ingest_cmd == "scripts":
                script_path = Path(__file__).parent / "auto_register_scripts.py"
                if not script_path.exists():
                    print(f"Error: {script_path} not found")
                    sys.exit(1)
                cmd = [sys.executable, str(script_path), str(args.directory),
                       "-p", args.project, "-n", str(args.limit)]
                if args.dry_run:
                    cmd += ["--dry-run"]
                _sp.run(cmd)

            else:
                ingest_parser.print_help()

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

        elif args.command == "flush-pending":
            import os
            import fcntl
            from urllib.parse import urlsplit, urlunsplit
            from urllib.request import urlopen
            qdir = args.queue_dir
            if not qdir.is_dir():
                if not args.quiet:
                    print(f"queue dir empty/missing: {qdir}")
                sys.exit(0)

            # Only one flusher at a time. Detached spawns from rapid `kb add`
            # calls all try to grab this lock; losers exit silently so they
            # don't pile up behind the slow embedding server.
            lock_path = qdir / ".flush.lock"
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not args.quiet:
                    print("another flush-pending is running; exiting")
                os.close(lock_fd)
                sys.exit(0)

            # Give the embedding server time to finish slow CPU work.
            # The flusher is detached; nothing waits on it.
            os.environ.setdefault("KB_EMBED_TIMEOUT", "900")

            files = sorted(qdir.glob("*.txt"))
            if not files:
                if not args.quiet:
                    print("no pending entries")
                sys.exit(0)

            # Probe the embedding server's /health endpoint. If it's not up,
            # leave files in the queue and exit silently — a later flush retries.
            parts = urlsplit(kb.embedding_url)
            health_url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
            try:
                with urlopen(health_url, timeout=5) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"health HTTP {resp.status}")
            except Exception as e:
                if not args.quiet:
                    print(f"embedding server not healthy ({health_url}): {e}; leaving {len(files)} file(s) queued")
                sys.exit(0)

            ok = fail = 0
            for f in files:
                # Atomic-claim via rename so a concurrent flush doesn't double-process.
                claimed = f.with_suffix(f.suffix + ".flushing")
                try:
                    f.rename(claimed)
                except OSError:
                    continue  # someone else got it
                try:
                    raw = claimed.read_text()
                    headers = {}
                    body_lines = []
                    in_body = False
                    for line in raw.splitlines():
                        if in_body:
                            body_lines.append(line)
                            continue
                        if line.startswith("# ") and ":" in line:
                            k, v = line[2:].split(":", 1)
                            headers[k.strip().lower()] = v.strip()
                        elif line.strip() == "":
                            in_body = True
                        else:
                            body_lines.append(line)
                            in_body = True
                    content = "\n".join(body_lines).strip()
                    if not content:
                        raise ValueError("empty content")
                    tags_str = headers.get("tags", "")
                    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
                    result = kb.add(
                        content=content,
                        finding_type=headers.get("type", "discovery"),
                        project=headers.get("project") or None,
                        sprint=headers.get("sprint") or None,
                        tags=tags,
                        evidence=headers.get("evidence") or None,
                        check_duplicate=False,  # caller already decided to queue
                    )
                    finding_id = result.get("id") if isinstance(result, dict) else result
                    if not finding_id:
                        raise RuntimeError(f"kb.add returned no id: {result}")
                    claimed.unlink()
                    ok += 1
                    if not args.quiet:
                        print(f"flushed {f.name} -> {finding_id}")
                except Exception as e:
                    # Restore so a later flush can retry
                    try:
                        claimed.rename(f)
                    except OSError:
                        pass
                    fail += 1
                    if not args.quiet:
                        print(f"FAILED {f.name}: {e}")

            print(f"flush-pending: {ok} ok, {fail} failed, {len(files)} total")
            sys.exit(0 if fail == 0 else 1)

    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

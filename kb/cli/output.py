"""Shared CLI output helpers: agent/user mode, color, terminal width, and
ANSI-aware truncation. Importable by kb.py, kb/issue_cli.py (kbt), and every
kb/cli/commands/* module so the output discipline is consistent:

  - colorized for users, PLAIN for agents (gated by KB_AGENT / CLAUDECODE / TTY);
  - one-line ROW entries truncated to the terminal width for users — NEVER for
    agents/pipes, and never for multi-line / full-content views.

Mode is resolved once at import (`AGENT_MODE`). Tests monkeypatch that attribute;
the helpers read it by name at call time so the patch takes effect.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from datetime import datetime


def is_agent() -> bool:
    """True when output is consumed by an agent (or a pipe), not a human terminal.

    KB_AGENT=1/0 is an explicit override; CLAUDECODE=1 is set by Claude Code in
    all subprocesses; otherwise a non-TTY stdout (pipe/subprocess) = agent.
    Mirrors kb.py::_is_agent_context so the two never diverge.
    """
    override = os.environ.get("KB_AGENT", "")
    if override == "1":
        return True
    if override == "0":
        return False
    if os.environ.get("CLAUDECODE") == "1":
        return True
    return not sys.stdout.isatty()


AGENT_MODE = is_agent()

# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------
RESET = "\033[0m"
_NAMED = {
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    "dim": "\033[2m", "bold": "\033[1m",
}


def c(text: str, color: str | None) -> str:
    """Wrap `text` in `color` (a name in _NAMED or a raw SGR code). No-op in
    agent mode or when `color` is falsy."""
    if AGENT_MODE or not color:
        return text
    code = _NAMED.get(color, color if color.startswith("\033") else "")
    return f"{code}{text}{RESET}" if code else text


def sim_color(sim: float) -> str:
    """Threshold color for a cosine similarity (green/yellow/red)."""
    return "green" if sim >= 0.7 else "yellow" if sim >= 0.5 else "red"


# ---------------------------------------------------------------------------
# Terminal width + ANSI-aware truncation
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def visible_len(s: str) -> int:
    """Length of `s` ignoring ANSI SGR escapes (what the terminal renders)."""
    return len(_ANSI_RE.sub("", s))


def term_width(default: int = 100) -> int | None:
    """Terminal column count for users; None for agents/pipes (= do not truncate)."""
    if AGENT_MODE:
        return None
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
        return cols if cols and cols > 0 else default
    except OSError:
        return default


def truncate(text: str, width: int | None) -> str:
    """Truncate `text` to <= `width` VISIBLE chars (ANSI escapes don't count),
    appending an ellipsis when cut and a RESET so color never bleeds. A falsy or
    non-positive `width`, or text already within width, is returned unchanged."""
    if not width or width <= 0 or visible_len(text) <= width:
        return text
    out: list[str] = []
    vis = 0
    i = 0
    n = len(text)
    limit = max(1, width - 1)  # reserve one column for the ellipsis
    while i < n and vis < limit:
        m = _ANSI_RE.match(text, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(text[i])
        vis += 1
        i += 1
    return "".join(out) + "…" + ("" if AGENT_MODE else RESET)


def fit_line(line: str) -> str:
    """Truncate a single ROW to the terminal width in user mode; passthrough for
    agents/pipes. Use ONLY for one-line list/row entries, never full views."""
    return truncate(line, term_width())


# ---------------------------------------------------------------------------
# Finding type display constants
# ---------------------------------------------------------------------------
_TYPE_COLORS = {
    "success":    "green",
    "failure":    "red",
    "experiment": "yellow",
    "discovery":  "cyan",
    "correction": "magenta",
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


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_age(created_at: str | None) -> str:
    """Human-readable age: 3d, 2w, 4m, 1y."""
    if not created_at:
        return ""
    try:
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
    # Document-section hits (from ingested PDFs/markdown) have no finding `type`
    # (they carry result_type='section' + kind); render a DOC tag + path instead
    # of falling through to '???'.
    if finding.get("result_type") == "section":
        kind = finding.get("kind", "prose")
        tag = {"table": "DOC·tbl", "figure": "DOC·fig"}.get(kind, "DOC")
        sim = finding.get("similarity")
        path = finding.get("path", "")
        text = (finding.get("heading")
                or (finding.get("content") or "").split("\n")[0][:100])
        proj = finding.get("project") or "?"
        if AGENT_MODE:
            sim_str = f" ({sim:.2f})" if sim is not None else ""
            return f"{finding['id']}{sim_str} [{tag} {path}] ({proj})  {text}"
        sim_str = (f" {c(f'({sim:.2f})', sim_color(sim))}" if sim is not None else "")
        return (f"{c(f'[{tag} {path}]', 'cyan')} {c(finding['id'], 'dim')}"
                f"{sim_str} {c(f'({proj})', 'dim')}  {text}")

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

    color = _TYPE_COLORS.get(finding.get("type", ""), "")
    abbr  = _TYPE_ABBREV.get(finding.get("type", ""), "???")
    sim_str = (f" {c(f'({sim:.2f})', sim_color(sim))}" if sim is not None else "")
    proj_str = f"({finding['project']})" if finding.get("project") else ""
    proj = f" {c(proj_str, 'dim')}" if proj_str else ""
    return f"{c(f'[{abbr}]', color)} {c(finding['id'], 'dim')}{sim_str}{proj}  {text}"


def format_results(findings: list[dict]) -> str:
    """Render the one-line search/list view for a whole result set.

    Agent mode: one `_fmt_one_line` per row, joined — byte-identical to the prior
    per-row output, so agents/hooks that parse it are unaffected.
    User mode: column-aligned, colored table (tag / id / similarity / project /
    text) so a human sees tidy columns instead of ragged per-row strings. Handles
    both finding rows and ingested document-section rows (which carry a DOC tag).
    """
    if not findings:
        return ""
    if AGENT_MODE:
        return "\n".join(_fmt_one_line(f) for f in findings)

    rows = []  # (tag_raw, tag_color, fid, sim, sim_raw, proj_raw, text)
    for f in findings:
        sim = f.get("similarity")
        sim_raw = f"({sim:.2f})" if isinstance(sim, (int, float)) else ""
        if f.get("result_type") == "section":
            kind = f.get("kind", "prose")
            tg = {"table": "DOC·tbl", "figure": "DOC·fig"}.get(kind, "DOC")
            tag_raw = f"[{tg} {f.get('path', '')}]"
            tag_color = "cyan"
            proj_raw = f"({f.get('project') or '?'})"
            text = f.get("heading") or (f.get("content") or "").split("\n")[0][:100]
        else:
            tag_raw = f"[{_TYPE_ABBREV.get(f.get('type', ''), '???')}]"
            tag_color = _TYPE_COLORS.get(f.get("type", ""), "")
            proj_raw = f"({f['project']})" if f.get("project") else ""
            text = f.get("summary") or f["content"].split("\n")[0][:100]
        rows.append((tag_raw, tag_color, f["id"], sim, sim_raw, proj_raw, text))

    tagw = max(len(r[0]) for r in rows)
    idw = max(len(r[2]) for r in rows)
    simw = max(len(r[4]) for r in rows)
    projw = max(len(r[5]) for r in rows)

    out = []
    for tag_raw, tag_color, fid, sim, sim_raw, proj_raw, text in rows:
        cells = [c(f"{tag_raw:<{tagw}}", tag_color),
                 c(f"{fid:<{idw}}", "dim")]
        if simw:
            cells.append(c(f"{sim_raw:<{simw}}", sim_color(sim))
                         if sim is not None else " " * simw)
        if projw:
            cells.append(c(f"{proj_raw:<{projw}}", "dim") if proj_raw else " " * projw)
        row = " ".join(cells) + f"  {text}"
        out.append(fit_line(row))
    return "\n".join(out)


def format_finding(finding: dict, verbose: bool = False) -> str:
    """Format a finding for terminal display (list/search output)."""
    type_color = _TYPE_COLORS.get(finding["type"], "")
    header = f"[{c(finding['type'].upper(), type_color)}] {c(finding['id'], 'dim')}"

    if finding.get("project"):
        proj_label = f"({finding['project']})"
        header += f" {c(proj_label, 'dim')}"

    if finding.get("similarity") is not None:
        sim = finding["similarity"]
        # Thresholds differ slightly here (0.8/0.6) vs sim_color (0.7/0.5) — preserve original
        sc = "green" if sim >= 0.8 else "yellow" if sim >= 0.6 else "red"
        header += f" {c(f'({sim:.2f})', sc)}"

    lines = [header, f"  {finding['content']}"]

    if verbose:
        if finding.get("evidence"):
            ev = finding["evidence"]
            ev_text = f"{ev[:200]}..." if len(ev) > 200 else ev
            lines.append("  " + c(f"Evidence: {ev_text}", "dim"))
        if finding.get("supersedes_id"):
            lines.append("  " + c(f"Supersedes: {finding['supersedes_id']}", "dim"))
        if finding.get("tags"):
            lines.append("  " + c(f"Tags: {', '.join(finding['tags'])}", "dim"))
        lines.append("  " + c(f"Created: {finding['created_at']}", "dim"))
        if finding.get("similarity"):
            lines.append("  " + c(f"Similarity: {finding['similarity']:.3f}", "dim"))

    return "\n".join(lines)


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

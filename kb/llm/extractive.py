"""
Extractive (no-LLM) one-line summary for kb findings.

Zero model, zero VRAM, zero network — the easy default so kb needs no second LLM.
Produces a short display blurb from the finding's own text: the first meaningful
sentence (or a clean prefix), whitespace-collapsed and length-clamped. Search
itself uses the full content via vector+FTS; this is only the result-list blurb.
"""

from __future__ import annotations

import re

_MAX_CHARS = 120

# Strip leading markdown noise (headings, list bullets, blockquotes, code fences).
_LEAD_NOISE = re.compile(r"^[\s>#*\-+`]+")
_CODE_FENCE = re.compile(r"^```.*?$", re.MULTILINE)
_WS = re.compile(r"\s+")
# Sentence end: . ! ? followed by space/end, but not a decimal or abbreviation dot.
_SENT_END = re.compile(r"(?<=[a-z0-9\)\]])[.!?](?:\s|$)", re.IGNORECASE)


def extractive_summary(content: str, evidence: str | None = None,
                       max_chars: int = _MAX_CHARS) -> str | None:
    """Return a one-line extractive summary of `content`, or None if empty.

    `evidence` is ignored (the blurb summarizes the claim, not its logs).
    """
    if not content or not content.strip():
        return None

    text = _CODE_FENCE.sub(" ", content)
    # First non-empty line is usually the claim; fall back to whole text.
    line = ""
    for raw in text.splitlines():
        stripped = _LEAD_NOISE.sub("", raw).strip()
        if stripped:
            line = stripped
            break
    if not line:
        line = _LEAD_NOISE.sub("", text).strip()

    line = _WS.sub(" ", line).strip()
    if not line:
        return None

    # Prefer the first sentence if it fits reasonably; else clamp at a word boundary.
    m = _SENT_END.search(line)
    if m and m.end() <= max_chars + 40:
        candidate = line[: m.start() + 1].strip()
        if candidate:
            line = candidate

    if len(line) <= max_chars:
        return line

    # Clamp at the last word boundary within max_chars; add an ellipsis.
    clamped = line[:max_chars].rsplit(" ", 1)[0].strip()
    if not clamped:
        clamped = line[:max_chars].strip()
    return clamped + "…"

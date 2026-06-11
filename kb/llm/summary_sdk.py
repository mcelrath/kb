"""
Subscription-SDK summarizer for kb findings.

Calls claude_agent_sdk.query with:
- setting_sources=[]  (no CLAUDE.md ingestion)
- allowed_tools=[]
- model pinned to claude-haiku-4-5
- ANTHROPIC_API_KEY scrubbed from the subprocess env so the claude
  binary falls back to subscription OAuth (a stale/dead API key in the
  global env would win in non-interactive mode and FAIL headless calls).

Enable via:  KB_SUMMARY_MODE=subscription-sdk

Gate result (2026-06-11, this host): subscription OAuth path is reached
correctly (env scrub works); calls return "Credit balance is too low"
because the subscription account currently has insufficient credits.
This mode is safe to wire behind the KB_SUMMARY_MODE env var but is NOT
the default.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Model to pin.  Haiku is cheapest and fastest for one-line summaries.
_HAIKU_MODEL = "claude-haiku-4-5"

# Per-call timeout in seconds.
_TIMEOUT_S = 30

# Maximum characters of text to send per summary item.
_MAX_TEXT_CHARS = 800


def _sdk_available() -> bool:
    """Return True if claude_agent_sdk can be imported."""
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except ImportError:
        return False


def _scrubbed_env() -> dict[str, str]:
    """Return os.environ with ANTHROPIC_API_KEY removed.

    The stale/dead key wins in non-interactive mode and would cause headless
    calls to fail with an auth error.  Removing it forces the claude binary to
    fall back to subscription OAuth.
    """
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    return env


async def _query_one(text: str) -> Optional[str]:
    """Async: summarize *text* in one line via the subscription SDK.

    Returns the summary string, or None on any failure.
    """
    try:
        import claude_agent_sdk

        prompt = (
            "Summarize in ONE technical line (max 120 chars). "
            "No preamble, no markdown. "
            f"Text: {text[:_MAX_TEXT_CHARS]}"
        )

        options = claude_agent_sdk.ClaudeAgentOptions(
            setting_sources=[],     # do NOT ingest any CLAUDE.md
            allowed_tools=[],       # no tool use
            model=_HAIKU_MODEL,
            env=_scrubbed_env(),    # scrub stale ANTHROPIC_API_KEY
            cwd="/tmp",             # neutral cwd with no CLAUDE.md
        )

        collected: list[str] = []
        async for msg in claude_agent_sdk.query(prompt=prompt, options=options):
            # AssistantMessage has .content = list[TextBlock | ...]
            content = getattr(msg, "content", None)
            if content and isinstance(content, list):
                for block in content:
                    t = getattr(block, "text", None)
                    if t:
                        collected.append(t)
            # ResultMessage: check is_error to surface failures
            is_error = getattr(msg, "is_error", False)
            result_val = getattr(msg, "result", None)
            if is_error and result_val:
                logger.warning("summary_sdk: SDK result error: %s", result_val)
                return None

        summary = " ".join(collected).strip()
        if not summary:
            return None
        # Clamp to 120 chars
        return summary[:120]

    except Exception as exc:
        logger.debug("summary_sdk._query_one failed: %s", exc)
        return None


def summarize_one(text: str) -> Optional[str]:
    """Summarize *text* in one line using the subscription SDK.

    Returns a short string (<=120 chars) or None on failure/unavailability.
    The caller degrades gracefully on None.
    """
    if not _sdk_available():
        logger.debug("summary_sdk: claude_agent_sdk not installed; returning None")
        return None
    if not text or not text.strip():
        return None
    try:
        return asyncio.run(
            asyncio.wait_for(_query_one(text), timeout=_TIMEOUT_S)
        )
    except asyncio.TimeoutError:
        logger.debug("summary_sdk.summarize_one: timed out after %ds", _TIMEOUT_S)
        return None
    except Exception as exc:
        logger.debug("summary_sdk.summarize_one: %s", exc)
        return None


def summarize_batch(texts: list[str]) -> list[Optional[str]]:
    """Summarize a list of texts, one result per input.

    Attempts a single batched SDK call returning N lines; falls back to
    per-item calls on parse mismatch.  Returns None for any item that fails.
    """
    if not _sdk_available():
        return [None] * len(texts)
    if not texts:
        return []

    # Try batch: send all items in one prompt, parse N lines back.
    batch_result = _summarize_batch_single_call(texts)
    if batch_result is not None and len(batch_result) == len(texts):
        return batch_result

    # Fallback: per-item
    logger.debug("summary_sdk.summarize_batch: batch mismatch, falling back to per-item")
    return [summarize_one(t) for t in texts]


def _summarize_batch_single_call(texts: list[str]) -> Optional[list[Optional[str]]]:
    """One SDK call for N texts; expect N lines back."""
    n = len(texts)
    numbered = "\n".join(
        f"{i + 1}. {t[:_MAX_TEXT_CHARS]}" for i, t in enumerate(texts)
    )
    prompt = (
        f"Summarize each of the following {n} items in ONE technical line each "
        "(max 120 chars per line). No preamble, no markdown, no extra text. "
        f"Return exactly {n} numbered lines matching the input order.\n\n"
        f"{numbered}"
    )

    async def _run() -> Optional[list[Optional[str]]]:
        try:
            import claude_agent_sdk

            options = claude_agent_sdk.ClaudeAgentOptions(
                setting_sources=[],
                allowed_tools=[],
                model=_HAIKU_MODEL,
                env=_scrubbed_env(),
                cwd="/tmp",
            )

            collected: list[str] = []
            async for msg in claude_agent_sdk.query(prompt=prompt, options=options):
                content = getattr(msg, "content", None)
                if content and isinstance(content, list):
                    for block in content:
                        t = getattr(block, "text", None)
                        if t:
                            collected.append(t)
                is_error = getattr(msg, "is_error", False)
                result_val = getattr(msg, "result", None)
                if is_error and result_val:
                    logger.warning("summary_sdk batch: SDK error: %s", result_val)
                    return None

            raw = " ".join(collected).strip()
            if not raw:
                return None

            # Parse: expect lines like "1. ...", "2. ...", ...
            import re
            lines = raw.splitlines()
            parsed: list[Optional[str]] = []
            for line in lines:
                m = re.match(r"^\d+\.\s*(.+)", line.strip())
                if m:
                    parsed.append(m.group(1).strip()[:120])
            if len(parsed) == n:
                return parsed
            return None

        except Exception as exc:
            logger.debug("summary_sdk batch call failed: %s", exc)
            return None

    try:
        return asyncio.run(asyncio.wait_for(_run(), timeout=_TIMEOUT_S * n))
    except (asyncio.TimeoutError, Exception) as exc:
        logger.debug("summary_sdk._summarize_batch_single_call: %s", exc)
        return None
